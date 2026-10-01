from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(Exception):
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any, lookup: Callable[[str], Optional[str]] = os.environ.get) -> Any:
    """Expand ${VAR} and ${VAR:-default}. Unset variables without a default become ""."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            current = lookup(match.group(1))
            if current:
                return str(current)
            return match.group(2) or ""

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: expand_env(item, lookup) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item, lookup) for item in value]
    return value


def load_file(path: str | Path, lookup: Callable[[str], Optional[str]] = os.environ.get) -> dict:
    """Read a YAML (needs PyYAML) or JSON mapping and expand ${VAR} references."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(f"{path} is YAML but PyYAML is not installed; use JSON or `pip install PyYAML`") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return expand_env(data, lookup)


def deep_merge(base: dict, override: dict) -> dict:
    """Mappings merge recursively; anything else in `override` replaces the base value."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(data: dict, dotted_key: str, raw_value: str) -> None:
    """Apply a `--set a.b=value` override; the value is parsed as JSON when it can be."""
    try:
        value: Any = json.loads(raw_value)
    except ValueError:
        value = raw_value
    parts = dotted_key.split(".")
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ConfigError(f"Cannot set {dotted_key}: {part} is not a mapping")
    target[parts[-1]] = value


class KafkaConfig(BaseModel):
    bootstrap_servers: str = "kafka:7010"
    topic: str = "exelixi.frames.raw"


class DatabaseConfig(BaseModel):
    main_url: str = ""
    # Server URL without a database name (the workers' POSTGRES_URL / VISION_BASE_URL);
    # module databases are derived from it the way each worker's Settings does.
    postgres_base_url: str = ""


class AuthConfig(BaseModel):
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    service_name: str = "load-generator"


class FramesConfig(BaseModel):
    shared_mount_path: str = "/data/frames"
    corpus_dir: str = "/data/frames/loadgen_corpus"
    # Where the workers read corpus images whose manifest paths are relative; empty = the
    # corpus's own corpus.json, else the corpus folder itself.
    corpus_image_root: str = ""
    timezone: str = "UTC"
    frame_date_format: str = "%Y-%m-%d"
    frame_hour_format: str = "%H"


class RegistrationConfig(BaseModel):
    name_prefix: str = "loadtest"
    template_camera_id: str = ""
    camera_ip: str = "127.0.0.1"
    base_port: int = 20000
    sync_timeout_s: float = 90.0
    snapshot_dir_name: str = "snapshots"


class OutputConfig(BaseModel):
    results_dir: str = "./loadgen_results"
    sample_interval_s: float = 5.0
    # Live metrics endpoint for `run` (what the web UI reads); None = off. Linger keeps it up after
    # the run so the final values are read.
    metrics_port: Optional[int] = Field(default=None, ge=0, le=65535)
    metrics_addr: str = "127.0.0.1"
    metrics_linger_s: float = Field(default=15.0, ge=0)

    @field_validator("metrics_port", mode="before")
    @classmethod
    def _blank_port_is_off(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


class EventsConfig(BaseModel):
    # The workers' EVENTS_DIR, mounted at the same path here; empty = event files are not cleaned.
    dir: str = ""


class WorkerLogsConfig(BaseModel):
    """Where the web UI reads worker container logs (`docker logs`), for its Worker logs tab."""

    # user@host of the Docker host the workers run on, reached with key login (ssh BatchMode);
    # "" = run docker on this machine.
    ssh_target: str = ""
    # How to call docker there, e.g. "sudo -n docker".
    docker_command: str = "docker"
    # Lines of history shown when following starts.
    tail: int = Field(default=300, ge=0, le=100_000)


class SettingsTarget(BaseModel):
    table: str
    camera_columns: list[str]
    enabled_where: str = "is_enabled = TRUE AND deleted_at IS NULL"
    order_by: str = "id"
    key_column: str = "id"


class CameraRowsTarget(BaseModel):
    """Per-camera config rows cloned from the template camera for each synthetic camera."""

    table: str
    camera_column: str = "camera_id"
    key_column: str = "id"


class PurgeTarget(BaseModel):
    table: str
    camera_column: str = "camera_id"
    db: Literal["module", "main"] = "module"
    # Columns holding event image/video paths (absolute, or relative to EVENTS_DIR).
    file_columns: list[str] = Field(default_factory=list)


class RestoreTarget(BaseModel):
    """Rows without a camera column that synthetic cameras may modify (e.g. frs_attendance)."""

    table: str
    key_column: str = "id"
    snapshot_where: str
    events_table: str
    events_link_column: str
    events_camera_column: str = "camera_id"
    events_time_column: str = "created_at"


class WorkerMetricsConfig(BaseModel):
    """The worker's own Prometheus endpoint, read once per sample (optional)."""

    # Relative to the worker's api_prefix; "" = the worker has none.
    path: str = ""
    # Histogram of per-frame seconds by `stage` label, and counter of frames by `camera_id`.
    stage_histogram: str = ""
    frames_counter: str = ""
    # Stages written to timeseries.csv (all stages are still exported live).
    stages: list[str] = Field(default_factory=list)
    # stage -> p50 seconds above which the stage looks CPU-bound; reported as a hint, never a failure.
    gpu_stage_limits: dict[str, float] = Field(default_factory=dict)


class WorkerConfig(BaseModel):
    function_key: str
    database_name: str = ""
    db_url: str = ""
    base_url: str = ""
    server_ip: str = ""
    api_prefix: str = "/api/v1"
    consumer_group_prefix: str = ""
    settings: SettingsTarget
    camera_rows: list[CameraRowsTarget] = Field(default_factory=list)
    purge: list[PurgeTarget] = Field(default_factory=list)
    restore: Optional[RestoreTarget] = None
    # GPU sizing, mirroring GpuInferenceEngine: one model copy per cameras_per_worker cameras.
    cameras_per_worker: Optional[int] = Field(default=None, ge=1)
    vram_per_worker_mb: Optional[int] = Field(default=None, ge=1)
    api_worker: bool = False
    gpu_free_vram_mb: Optional[int] = Field(default=None, ge=0)
    gpu_settings_source: str = ""
    # Event files live under EVENTS_DIR/<events_subdir>; defaults to the function key.
    events_subdir: str = ""
    metrics: WorkerMetricsConfig = Field(default_factory=WorkerMetricsConfig)
    # Docker container name, for the web UI's Worker logs tab.
    container: str = ""
    note: str = ""

    @field_validator("cameras_per_worker", "vram_per_worker_mb", "gpu_free_vram_mb", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        """An unset ${VAR} in a config file expands to ""."""
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def group_prefix(self) -> str:
        return self.consumer_group_prefix or f"{self.function_key}_consumer_group_"

    @property
    def events_folder(self) -> str:
        return self.events_subdir or self.function_key


class AppConfig(BaseModel):
    environment: Literal["staging", "production"] = "production"
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    frames: FramesConfig = Field(default_factory=FramesConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    worker_logs: WorkerLogsConfig = Field(default_factory=WorkerLogsConfig)
    workers: dict[str, WorkerConfig] = Field(default_factory=dict)
    # Set when running inside (a one-off container of) a worker: the worker it belongs to.
    local_worker: str = ""

    def worker_db_url(self, name: str) -> str:
        worker = self.workers[name]
        if worker.db_url:
            return worker.db_url
        if self.database.postgres_base_url and worker.database_name:
            return f"{self.database.postgres_base_url.rstrip('/')}/{worker.database_name}"
        raise ConfigError(f"No database URL for worker '{name}' (set POSTGRES_URL or its db_url)")

    def configured_workers(self) -> list[str]:
        """Workers with a database URL; the only ones cleanup can act on."""
        configured = []
        for name in self.workers:
            try:
                self.worker_db_url(name)
            except ConfigError:
                continue
            configured.append(name)
        return configured

    def main_db_url(self) -> str:
        if self.database.main_url:
            return self.database.main_url
        if self.database.postgres_base_url:
            return f"{self.database.postgres_base_url.rstrip('/')}/vision_main_db"
        raise ConfigError("No vision-main database URL (set VISION_MAIN_DATABASE_URL or POSTGRES_URL)")

    def worker_name(self, name_or_key: str) -> str:
        """Accept a worker name ("crowd") or its function key ("crowd-monitoring")."""
        if name_or_key in self.workers:
            return name_or_key
        for name, worker in self.workers.items():
            if worker.function_key == name_or_key:
                return name
        raise ConfigError(f"Unknown worker '{name_or_key}'; known: {', '.join(sorted(self.workers))}")


class SourceConfig(BaseModel):
    mode: Literal["live", "corpus"] = "live"
    source_camera_ids: list[str] = Field(default_factory=list)
    corpus_name: str = ""
    # Overrides frames.corpus_image_root for this scenario.
    image_root: str = ""
    first_frame_timeout_s: float = 30.0

    @model_validator(mode="after")
    def _corpus_needs_name(self) -> "SourceConfig":
        if self.mode == "corpus" and not self.corpus_name:
            raise ValueError("source.corpus_name is required when source.mode is 'corpus'")
        return self


class GuardConfig(BaseModel):
    enabled: bool = True
    baseline_samples: int = 3
    max_real_staleness_increase_s: float = 10.0
    grace_s: float = 15.0
    action: Literal["abort", "backoff"] = "abort"
    backoff_step: int = 5
    # Worker health, independent of real cameras (so it also protects corpus runs, and stays on
    # with --no-guard). Either check aborts the run; 0 turns a check off.
    worker_health: bool = True
    # The worker's status endpoint has failed for this long.
    unreachable_s: float = 45.0
    # The consumer backlog is this many seconds of publishing behind, and still growing.
    max_backlog_s: float = 120.0


class SaturationConfig(BaseModel):
    window_s: float = 30.0
    max_staleness_s: float = 5.0
    max_lag_growth_per_s: float = 1.0


class ScenarioConfig(BaseModel):
    name: str
    type: Literal["throughput", "latency", "soak"]
    workers: list[str]
    fps_per_camera: float = 5.0
    source: SourceConfig = Field(default_factory=SourceConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    saturation: SaturationConfig = Field(default_factory=SaturationConfig)
    # per_stage: each stage enables exactly its camera count and re-syncs the workers, so the
    # GPU engine runs as many model copies as it would in production. fixed: enable the
    # largest count once (one sync, but model copies are sized for the largest stage).
    sizing: Literal["per_stage", "fixed"] = "per_stage"
    # Seconds after a re-sync before samples count (pipeline restart and model load).
    settle_s: float = 20.0

    start_cameras: int = 5
    step_cameras: int = 5
    max_cameras: int = 50
    step_duration_s: float = 60.0

    levels: list[int] = Field(default_factory=list)
    level_duration_s: float = 120.0

    cameras: int = 0
    duration_s: float = 3600.0

    @model_validator(mode="after")
    def _check_type_fields(self) -> "ScenarioConfig":
        if not self.workers:
            raise ValueError("scenario.workers must list at least one worker")
        if self.fps_per_camera <= 0:
            raise ValueError("fps_per_camera must be positive")
        if self.settle_s < 0:
            raise ValueError("settle_s must not be negative")
        measured = self.settle_s + self.saturation.window_s
        if self.type == "throughput":
            if self.start_cameras < 1 or self.step_cameras < 1 or self.max_cameras < self.start_cameras:
                raise ValueError("throughput needs 1 <= start_cameras <= max_cameras and step_cameras >= 1")
            if self.step_duration_s <= measured:
                raise ValueError("step_duration_s must be longer than settle_s + saturation.window_s")
        elif self.type == "latency":
            if not self.levels or min(self.levels) < 1:
                raise ValueError("latency needs a non-empty list of positive levels")
            if self.level_duration_s <= measured:
                raise ValueError("level_duration_s must be longer than settle_s + saturation.window_s")
        elif self.cameras < 1:
            raise ValueError("soak needs cameras >= 1")
        return self

    @property
    def registered_cameras(self) -> int:
        if self.type == "throughput":
            return self.max_cameras
        if self.type == "latency":
            return max(self.levels)
        return self.cameras


def load_scenario(ref: str, app: AppConfig, overrides: Optional[list[str]] = None,
                  workers: Optional[list[str]] = None) -> ScenarioConfig:
    """A preset name (throughput | latency | soak) or a scenario file, plus `--set` overrides.

    Workers default to the scenario's list, then to the worker this process runs beside.
    """
    from vision_loadgen.presets import SCENARIO_PRESETS

    data = copy.deepcopy(SCENARIO_PRESETS[ref]) if ref in SCENARIO_PRESETS else load_file(ref)
    for item in overrides or []:
        key, separator, value = item.partition("=")
        if not separator:
            raise ConfigError(f"--set expects key=value, got '{item}'")
        set_dotted(data, key.strip(), value.strip())
    if workers:
        data["workers"] = list(workers)
    if not data.get("workers"):
        if not app.local_worker:
            raise ConfigError("No workers to test: pass --worker (e.g. --worker crowd) or run beside a worker")
        data["workers"] = [app.local_worker]
    data["workers"] = [app.worker_name(name) for name in data["workers"]]
    return ScenarioConfig.model_validate(data)
