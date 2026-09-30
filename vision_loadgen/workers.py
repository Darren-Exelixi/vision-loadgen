from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from jose import jwt

from vision_loadgen.config import AuthConfig, ConfigError, WorkerConfig


def service_token(auth: AuthConfig) -> str:
    """Same claims as vision_shared.db.vision_main.create_service_token (workers check `service`).

    Not imported from there: vision_loadgen does not depend on vision_shared.
    """
    if not auth.jwt_secret_key:
        raise ConfigError("auth.jwt_secret_key is empty (set JWT_SECRET_KEY)")
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    return jwt.encode({"service": auth.service_name, "exp": expires}, auth.jwt_secret_key, algorithm=auth.jwt_algorithm)


@dataclass
class Deployment:
    server_function_id: str
    server_id: str
    server_ip: str
    base_url: str


def resolve_deployment(cur, name: str, worker: WorkerConfig) -> Deployment:
    """The single active server_function running this worker's function (vision-main)."""
    cur.execute(
        """
        SELECT sf.id::text AS server_function_id,
               s.id::text AS server_id,
               s.server_ip,
               s.server_protocol,
               COALESCE(sf.container_port, f.container_port) AS container_port
        FROM server_functions sf
        JOIN servers s ON s.id = sf.server_id
        JOIN functions f ON f.id = sf.function_id
        WHERE f.key = %s
          AND sf.deleted_at IS NULL AND s.deleted_at IS NULL AND f.deleted_at IS NULL
          AND COALESCE(sf.is_active, FALSE) = TRUE
          AND COALESCE(s.is_active, FALSE) = TRUE
          AND COALESCE(f.is_exelixi_activated, FALSE) = TRUE
          AND (%s = '' OR s.server_ip = %s)
        ORDER BY s.server_ip
        """,
        (worker.function_key, worker.server_ip, worker.server_ip),
    )
    rows = cur.fetchall()
    if not rows:
        raise ConfigError(f"No active deployment of '{worker.function_key}' for worker '{name}'")
    if len(rows) > 1:
        servers = ", ".join(row["server_ip"] for row in rows)
        raise ConfigError(
            f"Worker '{name}' runs on several servers ({servers}); pass --server-ip, or run the load "
            "generator beside the worker you want to test (its PUBLIC_BASE_IP picks the server)"
        )
    row = rows[0]
    base_url = worker.base_url
    if not base_url:
        if not row["container_port"]:
            raise ConfigError(f"No container_port for '{worker.function_key}'; set its base_url with --config")
        base_url = f"{row['server_protocol'] or 'http'}://{row['server_ip']}:{row['container_port']}"
    return Deployment(
        server_function_id=row["server_function_id"],
        server_id=row["server_id"],
        server_ip=row["server_ip"],
        base_url=base_url.rstrip("/"),
    )


@dataclass
class WorkerStatus:
    running: bool
    active_cameras: set[str]
    processed_timestamps: dict[str, float] = field(default_factory=dict)


class WorkerClient:
    def __init__(self, name: str, base_url: str, api_prefix: str, auth: AuthConfig, timeout_s: float = 10.0) -> None:
        self.name = name
        self._root = f"{base_url.rstrip('/')}/{api_prefix.strip('/')}".rstrip("/")
        self._auth = auth
        self._timeout_s = timeout_s

    def sync(self) -> WorkerStatus:
        return parse_status(self._call("POST", "/worker/sync", timeout_s=60))

    def status(self) -> WorkerStatus:
        return parse_status(self._call("GET", "/worker/status", timeout_s=self._timeout_s))

    def _call(self, method: str, path: str, timeout_s: float) -> dict:
        url = f"{self._root}{path}"
        req = urllib.request.Request(
            url,
            data=b"{}" if method == "POST" else None,
            headers={"Authorization": f"Bearer {service_token(self._auth)}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
        return _data(body)


def _data(body) -> dict:
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        return body["data"]
    return body if isinstance(body, dict) else {}


def parse_status(data: dict) -> WorkerStatus:
    processed: dict[str, float] = {}
    for source_id, info in (data.get("processed_buffer_info") or {}).items():
        if isinstance(info, dict) and info.get("has_frame") and info.get("timestamp") is not None:
            try:
                processed[str(source_id)] = float(info["timestamp"])
            except (TypeError, ValueError):
                continue
    return WorkerStatus(
        running=bool(data.get("running")),
        active_cameras={str(camera_id) for camera_id in data.get("active_cameras") or []},
        processed_timestamps=processed,
    )
