"""A stand-in GPU worker with the real contract: service-JWT /worker/sync and /worker/status, a fresh
timestamped consumer group per (re)start, camera filtering by region assignment + settings row,
GpuInferenceEngine's refusal to start when the cameras need more model copies than fit in VRAM, and
its own Prometheus metrics (per-stage time histogram, frames per camera) like the emotion worker's.

api="attendance" mimics face-attendance-backend instead: POST /worker/sync needs {module_name,
server_function_id} and runs as a job (GET /worker/sync/status), and cameras are only listed by
GET /streaming/processed/list once their frames were processed."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

import psycopg2
from jose import JWTError, jwt
from kafka import KafkaConsumer
from psycopg2.extras import RealDictCursor


class FakeWorker:
    def __init__(self, function_key: str, main_url: str, module_url: str, settings_table: str, camera_column: str,
                 bootstrap: str, topic: str, secret: str,
                 on_frame: Optional[Callable[[str, dict], None]] = None,
                 max_cameras: Optional[int] = None, api: str = "standard") -> None:
        self.function_key = function_key
        self.api = api
        self.sync_jobs: list[dict] = []
        self.sync_bodies: list[dict] = []
        self.main_url = main_url
        self.module_url = module_url
        self.settings_table = settings_table
        self.camera_column = camera_column
        self.bootstrap = bootstrap
        self.topic = topic
        self.secret = secret
        self.on_frame = on_frame
        self.max_cameras = max_cameras
        self.active: set[str] = set()
        self.processed: dict[str, float] = {}
        self.syncs = 0
        self.peak_cameras = 0
        self.running = False
        # Set to make GET /worker/status and /metrics fail, like a worker too busy to answer.
        self.unresponsive = False
        self.frames_processed: dict[str, int] = {}
        self.stage_seconds = 0.004
        self._lock = threading.Lock()
        self._consumer_stop: Optional[threading.Event] = None
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self._server.server_address[1]

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.refresh()

    def stop(self) -> None:
        self._server.shutdown()
        if self._consumer_stop:
            self._consumer_stop.set()

    def refresh(self) -> None:
        with psycopg2.connect(self.main_url, cursor_factory=RealDictCursor) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id::text AS id FROM cameras c
                JOIN function_camera_regions fcr ON fcr.camera_region_id = c.region_id AND fcr.deleted_at IS NULL
                  AND COALESCE(fcr.status, 'assigned') NOT IN ('stopping', 'stopped', 'disabled', 'deleted')
                JOIN server_functions sf ON sf.id = fcr.server_function_id AND sf.is_active
                JOIN functions f ON f.id = sf.function_id AND f.key = %s
                WHERE c.deleted_at IS NULL AND c.is_active
                """,
                (self.function_key,),
            )
            assigned = {row["id"] for row in cur.fetchall()}
        with psycopg2.connect(self.module_url, cursor_factory=RealDictCursor) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {self.camera_column} AS cams FROM {self.settings_table} "
                        "WHERE is_enabled AND deleted_at IS NULL ORDER BY id LIMIT 1")
            row = cur.fetchone()
            selected = {str(camera) for camera in (row["cams"] if row else [])}
        with self._lock:
            self.active = assigned & selected
            self.syncs += 1
            self.peak_cameras = max(self.peak_cameras, len(self.active))
            if self.max_cameras is not None and len(self.active) > self.max_cameras:
                self.running = False
                if self._consumer_stop:
                    self._consumer_stop.set()
                raise RuntimeError(f"Insufficient GPU VRAM for {len(self.active)} cameras")
            self.running = True
        self._restart_consumer()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "active_cameras": sorted(self.active),
                "processed_buffer_info": {
                    camera: {"has_frame": True, "timestamp": ts} for camera, ts in self.processed.items()
                },
            }

    def processed_list(self) -> dict:
        with self._lock:
            cameras = sorted(camera for camera in self.processed if camera in self.active)
            return {"status": "success", "cameras": [{"camera_id": camera} for camera in cameras],
                    "count": len(cameras),
                    "buffer_info": {camera: {"has_frame": True, "timestamp": self.processed[camera]} for camera in cameras},
                    "diagnostics": {"kafka_consumer_running": self.running}}

    def sync_status(self) -> dict:
        with self._lock:
            current = next((job for job in self.sync_jobs if job["status"] == "running"), None)
            done = [job for job in self.sync_jobs if job["status"] != "running"]
            return {"current": current, "last": done[-1] if done else None, "pending": None}

    def _sync_job(self, job: dict) -> None:
        time.sleep(0.3)  # the real job rebuilds face enrollments for a few seconds
        try:
            self.refresh()
            status = "completed"
        except RuntimeError:
            status = "failed"
        with self._lock:
            job.update(status=status, phase=status,
                       source_diagnostics={"assigned_camera_count": len(self.active), "mapped_camera_count": len(self.active),
                                           "unmapped_camera_count": 0})

    def metrics(self) -> str:
        with self._lock:
            frames = dict(self.frames_processed)
        total = sum(frames.values())
        lines = ["# TYPE fake_stage_seconds histogram"]
        for bound in ("0.002", "0.005", "0.05", "0.5", "+Inf"):
            count = total if float(bound) >= self.stage_seconds else 0
            lines.append(f'fake_stage_seconds_bucket{{le="{bound}",stage="detect"}} {float(count)}')
        lines.append(f'fake_stage_seconds_count{{stage="detect"}} {float(total)}')
        lines.append("# TYPE fake_frames_processed_total counter")
        lines += [f'fake_frames_processed_total{{camera_id="{camera}"}} {float(count)}' for camera, count in frames.items()]
        return "\n".join(lines) + "\n"

    def _restart_consumer(self) -> None:
        if self._consumer_stop:
            self._consumer_stop.set()
        stop = threading.Event()
        self._consumer_stop = stop
        group = f"{self.function_key}_consumer_group_{int(time.time() * 1000)}"
        threading.Thread(target=self._consume, args=(group, stop), daemon=True).start()

    def _consume(self, group: str, stop: threading.Event) -> None:
        consumer = KafkaConsumer(self.topic, bootstrap_servers=self.bootstrap, group_id=group,
                                 auto_offset_reset="latest", enable_auto_commit=True,
                                 auto_commit_interval_ms=1000,
                                 value_deserializer=lambda raw: json.loads(raw.decode("utf-8")))
        try:
            while not stop.is_set():
                for records in consumer.poll(timeout_ms=300).values():
                    for record in records:
                        camera = str(record.value.get("camera_id"))
                        with self._lock:
                            if camera not in self.active:
                                continue
                            self.processed[camera] = float(record.value["timestamp"])
                            self.frames_processed[camera] = self.frames_processed.get(camera, 0) + 1
                        if self.on_frame:
                            self.on_frame(camera, record.value)
        finally:
            consumer.close()

    def _handler(self):
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _authorised(self) -> bool:
                header = self.headers.get("Authorization", "")
                try:
                    claims = jwt.decode(header.removeprefix("Bearer "), worker.secret, algorithms=["HS256"])
                    return bool(claims.get("service"))
                except JWTError:
                    return False

            def _reply(self, code: int, body: dict) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if not self._authorised():
                    return self._reply(401, {"detail": "Invalid service token"})
                if worker.unresponsive:
                    return self._reply(503, {"detail": "busy"})
                if worker.api == "attendance":
                    if self.path == "/api/v1/streaming/processed/list":
                        return self._reply(200, worker.processed_list())
                    if self.path == "/api/v1/worker/sync/status":
                        return self._reply(200, worker.sync_status())
                    return self._reply(404, {"detail": "Not Found"})
                if self.path == "/api/v1/worker/status":
                    return self._reply(200, {"success": True, "data": worker.status()})
                if self.path == "/api/v1/metrics":
                    payload = worker.metrics().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self._reply(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if not self._authorised():
                    return self._reply(401, {"detail": "Invalid service token"})
                if worker.api == "attendance" and self.path == "/api/v1/worker/sync":
                    body = json.loads(raw or b"{}")
                    missing = [key for key in ("module_name", "server_function_id") if not body.get(key)]
                    if missing:
                        return self._reply(422, {"detail": [{"type": "missing", "loc": ["body", key]} for key in missing]})
                    job = {"job_id": f"job-{len(worker.sync_jobs) + 1}", "status": "running", "phase": "fetching", **body}
                    with worker._lock:
                        worker.sync_bodies.append(body)
                        worker.sync_jobs.append(job)
                    threading.Thread(target=worker._sync_job, args=(job,), daemon=True).start()
                    return self._reply(202, {"status": "accepted", "job_id": job["job_id"]})
                if self.path == "/api/v1/worker/sync":
                    try:
                        worker.refresh()
                    except RuntimeError as exc:
                        return self._reply(500, {"detail": str(exc)})
                    return self._reply(200, {"success": True, "data": worker.status()})
                self._reply(404, {})

        return Handler
