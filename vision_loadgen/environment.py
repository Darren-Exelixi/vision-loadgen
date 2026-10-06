"""Builds the AppConfig from the worker's own settings, the environment and optional files.

Precedence, lowest first:
  1. built-in presets (presets.py)
  2. shared values: the worker's Settings object (when run beside a worker), then environment
     variables, then vision_shared/.env when vision_shared is installed beside it
  3. the local worker's own values (DATABASE_URL, CAMERAS_PER_WORKER, VRAM_PER_WORKER_MB,
     PUBLIC_BASE_IP), applied to that worker only
  4. a --config file (YAML or JSON)
  5. command-line flags
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from vision_loadgen.config import AppConfig, ConfigError, deep_merge, expand_env, load_file
from vision_loadgen.presets import WORKER_PRESETS

log = logging.getLogger(__name__)

DEFAULT_WORKER_SETTINGS = "app.core.config:settings"
Lookup = Callable[[str], Optional[str]]


def load_worker_settings(spec: str = DEFAULT_WORKER_SETTINGS) -> Optional[Any]:
    """The worker's Settings object when this process runs inside a worker image, else None."""
    if not spec:
        return None
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            return None
        log.warning("Worker settings %s could not be imported: %s", spec, exc)
        return None
    except Exception as exc:
        log.warning("Worker settings %s could not be loaded: %s", spec, exc)
        return None
    settings = getattr(module, attribute or "settings", None)
    if not getattr(settings, "FUNCTION_NAME", None):
        return None
    return settings


def load_env_file(path: Optional[str] = None, environ: Optional[dict[str, str]] = None) -> Optional[Path]:
    """Load KEY=value lines into the environment; variables already set win.

    `path` (--env-file), else LOADGEN_ENV_FILE, else ./.env when it exists. '' disables it.
    Returns the file loaded, if any.
    """
    environ = os.environ if environ is None else environ
    if path is None:
        path = environ.get("LOADGEN_ENV_FILE")
        explicit = path is not None
        path = ".env" if path is None else path
    else:
        explicit = True
    if not path:
        return None
    env_file = Path(path)
    if not env_file.is_file():
        if explicit:
            raise ConfigError(f"env file not found: {env_file}")
        return None
    from dotenv import dotenv_values

    for key, value in dotenv_values(env_file).items():
        if value is not None and key not in environ:
            environ[key] = value
    return env_file


def _package_env() -> dict[str, str]:
    """Shared defaults shipped in vision_shared/.env, when vision_shared is installed (worker images).

    Located with find_spec so vision_shared is never imported: importing it validates worker
    settings that a standalone run does not have.
    """
    spec = importlib.util.find_spec("vision_shared")
    if spec is None or not spec.submodule_search_locations:
        return {}
    env_file = Path(list(spec.submodule_search_locations)[0]) / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def make_lookup(worker_settings: Optional[Any], environ: Optional[dict[str, str]] = None,
                package_env: Optional[dict[str, str]] = None) -> Lookup:
    environ = dict(os.environ) if environ is None else environ
    package_env = _package_env() if package_env is None else package_env

    def lookup(name: str) -> Optional[str]:
        if worker_settings is not None:
            value = getattr(worker_settings, name, None)
            if value not in (None, ""):
                return str(value)
        return environ.get(name) or package_env.get(name) or None

    return lookup


def _shared_only(environ: Optional[dict[str, str]], package_env: Optional[dict[str, str]]) -> Lookup:
    """Values every module shares; excludes the local worker's own overrides."""
    return make_lookup(None, environ, package_env)


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value).split("#", 1)[0].strip()) if value not in (None, "") else None
    except ValueError:
        return None


def base_config(lookup: Lookup, shared: Lookup) -> dict[str, Any]:
    frames_mount = lookup("FRAMES_MOUNT_PATH") or "/data/frames"
    logs_dir = lookup("LOGS_DIR")
    results_dir = lookup("LOADGEN_RESULTS_DIR") or (os.path.join(logs_dir, "loadgen") if logs_dir else "./loadgen_results")
    workers = copy.deepcopy(WORKER_PRESETS)
    for worker in workers.values():
        worker.update({
            "cameras_per_worker": _int(shared("CAMERAS_PER_WORKER")),
            "vram_per_worker_mb": _int(shared("VRAM_PER_WORKER_MB")),
            "gpu_settings_source": "shared default (a module may override it)",
        })
    return {
        "environment": lookup("LOADGEN_ENVIRONMENT") or "production",
        "kafka": {
            "bootstrap_servers": lookup("KAFKA_BOOTSTRAP_SERVERS") or "kafka:7010",
            "topic": lookup("KAFKA_TOPIC") or "exelixi.frames.raw",
        },
        "database": {
            "main_url": lookup("VISION_MAIN_DATABASE_URL") or "",
            "postgres_base_url": lookup("VISION_BASE_URL") or lookup("POSTGRES_URL") or "",
        },
        "auth": {
            "jwt_secret_key": lookup("JWT_SECRET_KEY") or "",
            "jwt_algorithm": lookup("JWT_ALGORITHM") or "HS256",
        },
        "frames": {
            "shared_mount_path": frames_mount,
            "corpus_dir": lookup("LOADGEN_CORPUS_DIR") or os.path.join(frames_mount, "loadgen_corpus"),
            "corpus_image_root": lookup("LOADGEN_CORPUS_IMAGE_ROOT") or "",
            "timezone": lookup("TIMEZONE") or "UTC",
        },
        "registration": {
            "template_camera_id": lookup("LOADGEN_TEMPLATE_CAMERA_ID") or lookup("TEMPLATE_CAMERA_ID") or "",
        },
        "output": {
            "results_dir": results_dir,
            "metrics_port": _int(lookup("LOADGEN_METRICS_PORT")),
            "metrics_addr": lookup("LOADGEN_METRICS_ADDR") or "127.0.0.1",
        },
        "events": {"dir": lookup("EVENTS_DIR") or "",
                   "container_dir": lookup("LOADGEN_CONTAINER_EVENTS_DIR") or "/app/events"},
        "worker_logs": {
            "ssh_target": lookup("LOADGEN_WORKER_LOGS_SSH") or "",
            "password": lookup("LOADGEN_WORKER_LOGS_PASSWORD") or "",
            "docker_command": lookup("LOADGEN_WORKER_LOGS_DOCKER") or "docker",
        },
        "host_stats": {"enabled": (lookup("LOADGEN_HOST_STATS") or "1").strip().lower() not in ("0", "false", "off", "no")},
        "workers": expand_env(workers, lookup),
    }


def apply_local_worker(raw: dict[str, Any], worker_settings: Any) -> None:
    """Point the preset matching the worker's FUNCTION_NAME at that worker's own settings."""
    function_key = str(worker_settings.FUNCTION_NAME)
    name = next((key for key, worker in raw["workers"].items() if worker["function_key"] == function_key), None)
    if name is None:
        log.warning("Running beside '%s', which has no preset; add it with --config to test it", function_key)
        return
    worker = raw["workers"][name]
    database_url = getattr(worker_settings, "DATABASE_URL", None)
    if database_url:
        worker["db_url"] = str(database_url)
    for field, attribute in (("cameras_per_worker", "CAMERAS_PER_WORKER"), ("vram_per_worker_mb", "VRAM_PER_WORKER_MB")):
        value = _int(getattr(worker_settings, attribute, None))
        if value is not None:
            worker[field] = value
    worker["gpu_settings_source"] = "this worker's settings"
    public_ip = getattr(worker_settings, "PUBLIC_BASE_IP", "")
    if public_ip:
        worker["server_ip"] = str(public_ip)
    raw["local_worker"] = name


def build_app_config(config_path: Optional[str] = None, worker_settings_spec: str = DEFAULT_WORKER_SETTINGS,
                     overrides: Optional[dict[str, Any]] = None, environ: Optional[dict[str, str]] = None,
                     package_env: Optional[dict[str, str]] = None, worker_settings: Any = None) -> AppConfig:
    if worker_settings is None:
        worker_settings = load_worker_settings(worker_settings_spec)
    lookup = make_lookup(worker_settings, environ, package_env)
    raw = base_config(lookup, _shared_only(environ, package_env))
    if worker_settings is not None:
        apply_local_worker(raw, worker_settings)
    config_path = config_path or lookup("LOADGEN_CONFIG")
    if config_path:
        raw = deep_merge(raw, load_file(config_path, lookup))
    if overrides:
        raw = deep_merge(raw, overrides)
    try:
        return AppConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
