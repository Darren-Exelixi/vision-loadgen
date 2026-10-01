from __future__ import annotations

import email.utils
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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


class ClockOffset:
    """How far the worker's clock is ahead of this machine's, from the HTTP Date header.

    Date has 1 s resolution, so each response only bounds the offset to an interval; intersecting
    the intervals of successive responses narrows it to about the round-trip time.
    """

    def __init__(self) -> None:
        self._low: Optional[float] = None
        self._high: Optional[float] = None

    def add(self, sent: float, received: float, server_second: float) -> None:
        # The server stamped Date at some local time in [sent, received], within the second
        # [server_second, server_second + 1).
        low, high = server_second - received, server_second + 1.0 - sent
        if self._low is None or self._high is None or low > self._high or high < self._low:
            # First response, or a clock was stepped since: start over.
            self._low, self._high = low, high
        else:
            self._low, self._high = max(self._low, low), min(self._high, high)

    @property
    def known(self) -> bool:
        return self._low is not None

    @property
    def seconds(self) -> float:
        """Best estimate; 0.0 until a response with a Date header arrives."""
        if self._low is None or self._high is None:
            return 0.0
        return (self._low + self._high) / 2

    @property
    def error_s(self) -> Optional[float]:
        if self._low is None or self._high is None:
            return None
        return (self._high - self._low) / 2

    def as_dict(self) -> dict[str, Any]:
        if not self.known:
            return {"offset_s": None, "error_s": None}
        return {"offset_s": round(self.seconds, 3), "error_s": round(self.error_s or 0.0, 3)}


class WorkerClient:
    def __init__(self, name: str, base_url: str, api_prefix: str, auth: AuthConfig, timeout_s: float = 10.0) -> None:
        self.name = name
        self._root = f"{base_url.rstrip('/')}/{api_prefix.strip('/')}".rstrip("/")
        self._auth = auth
        self._timeout_s = timeout_s
        self.clock = ClockOffset()

    def sync(self) -> WorkerStatus:
        return parse_status(self._call("POST", "/worker/sync", timeout_s=60))

    def status(self) -> WorkerStatus:
        return parse_status(self._call("GET", "/worker/status", timeout_s=self._timeout_s))

    def metrics_text(self, path: str) -> str:
        """The worker's own Prometheus exposition at `path` (relative to the API prefix)."""
        return self._request("GET", "/" + path.lstrip("/"), timeout_s=self._timeout_s).decode("utf-8", "replace")

    def _call(self, method: str, path: str, timeout_s: float) -> dict:
        raw = self._request(method, path, timeout_s)
        return _data(json.loads(raw.decode("utf-8") or "{}"))

    def _request(self, method: str, path: str, timeout_s: float) -> bytes:
        url = f"{self._root}{path}"
        req = urllib.request.Request(
            url,
            data=b"{}" if method == "POST" else None,
            headers={"Authorization": f"Bearer {service_token(self._auth)}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            sent = time.time()
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                raw = response.read()
                self._record_clock(sent, time.time(), response.headers.get("Date"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
        return raw

    def _record_clock(self, sent: float, received: float, date_header: Optional[str]) -> None:
        if not date_header:
            return
        try:
            server_second = email.utils.parsedate_to_datetime(date_header).timestamp()
        except (TypeError, ValueError):
            return
        self.clock.add(sent, received, server_second)


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
