"""`vision-loadgen ui`: a small local web page that runs the CLI and shows its metrics.

Stdlib only. Every action runs `python -m vision_loadgen ...` as a child process with validated
arguments (no shell); its output goes to a log file the page polls. Runs always serve Prometheus
metrics on a free port, which this server polls and keeps per job for the live charts. History
comes from the results folder (summary.json, timeseries.csv).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from vision_loadgen.config import AppConfig, ConfigError
from vision_loadgen.exporter import METRICS
from vision_loadgen.sources import HEADER_NAME, MANIFEST_NAME, read_corpus_header

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
UPLOADS = "_uploads"
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
METRICS_PORT = 9464
POLL_S = 2.0
MAX_SNAPSHOTS = 25_000  # ~14 h at 2 s

UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,252}$")
CORPUS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
CAMERA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,59}$")
SET_KEY = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./\-]{1,250}$")
COPY_TARGET = re.compile(r"^[A-Za-z0-9_.\-]+@[A-Za-z0-9.\-]+:[A-Za-z0-9_./~\-]+$")
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
RUN_STARTED = re.compile(r"Run (\d{8}-\d{6}-[0-9a-f]{4}):")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,120}\.(mp4|avi|mov|mkv|webm|m4v)$", re.IGNORECASE)
SCENARIOS = ("latency", "throughput", "soak")
EXCLUSIVE = {"run", "cleanup"}  # change what is registered; one at a time


class BadRequest(Exception):
    pass


# ---------------------------------------------------------------------------- validation

def _str(params: dict, key: str, pattern: re.Pattern, required: bool = True) -> str:
    value = str(params.get(key) or "").strip()
    if not value:
        if required:
            raise BadRequest(f"{key} is required")
        return ""
    if not pattern.match(value):
        raise BadRequest(f"Invalid {key}: {value!r}")
    return value


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    raw = params.get(key)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"Invalid {key}: {raw!r}")
    if not low <= value <= high:
        raise BadRequest(f"{key} must be between {low} and {high}")
    return value


def scenario_args(app: AppConfig, params: dict) -> list[str]:
    """--scenario/--worker/--server-ip/--template-camera/--set for check and run."""
    scenario = str(params.get("scenario") or "latency")
    if scenario not in SCENARIOS:
        raise BadRequest(f"Unknown scenario {scenario!r}")
    workers = params.get("workers") or []
    if not isinstance(workers, list) or not workers:
        raise BadRequest("Pick at least one worker")
    argv = ["--scenario", scenario]
    for worker in workers:
        try:
            argv += ["--worker", app.worker_name(str(worker))]
        except ConfigError as exc:
            raise BadRequest(str(exc))
    server_ip = _str(params, "server_ip", HOST, required=False)
    if server_ip:
        argv += ["--server-ip", server_ip]
    template = _str(params, "template_camera", UUID, required=False)
    if template:
        argv += ["--template-camera", template]
    overrides = list(params.get("set") or [])
    if params.get("source") == "corpus":
        overrides += [f"source.mode=corpus", f"source.corpus_name={_str(params, 'corpus', CORPUS_NAME)}"]
    elif params.get("source") not in (None, "", "live"):
        raise BadRequest("source must be live or corpus")
    for item in overrides:
        key, sep, value = str(item).partition("=")
        if not sep or not SET_KEY.match(key.strip()) or len(value) > 200 or "\n" in value:
            raise BadRequest(f"Invalid override {item!r} (expected key=value)")
        argv += ["--set", f"{key.strip()}={value.strip()}"]
    return argv


def build_command(app: AppConfig, action: str, params: dict, allow_production: bool,
                  upload_dir: Path) -> tuple[str, list[str]]:
    """(job kind, CLI arguments after `python -m vision_loadgen`) for an allowed action."""
    if action == "check":
        return "check", ["check", *scenario_args(app, params)]
    if action == "run":
        argv = ["run", *scenario_args(app, params)]
        if params.get("no_guard"):
            argv.append("--no-guard")
        if params.get("keep_events"):
            argv.append("--keep-events")
        if allow_production:
            argv.append("--allow-production")
        return "run", argv
    if action == "corpus":
        video = _str(params, "video", FILE_NAME)
        if not (upload_dir / video).is_file():
            raise BadRequest(f"Upload {video} first")
        argv = ["corpus", "from-video", "--video", str(upload_dir / video), "--name", _str(params, "name", CORPUS_NAME),
                "--fps", str(_number(params, "fps", 5.0, 0.1, 60.0)),
                "--max-seconds", str(_number(params, "max_seconds", 0.0, 0.0, 86_400.0))]
        image_root = _str(params, "image_root", REMOTE_PATH, required=False)
        if image_root:
            argv += ["--image-root", image_root]
        copy_target = _str(params, "copy_target", COPY_TARGET, required=False)
        if copy_target:
            argv += ["--copy-target", copy_target]
        return "corpus", argv
    if action == "video-camera-add":
        argv = ["video-camera", "add", "--name", _str(params, "name", CAMERA_NAME),
                "--like", _str(params, "like", UUID)]
        zones = _str(params, "zones_from", UUID, required=False)
        if zones:
            argv += ["--zones-from", zones]
        corpus = _str(params, "corpus", CORPUS_NAME, required=False)
        if corpus:
            argv += ["--corpus", corpus]
        return "video-camera", argv
    if action == "video-camera-remove":
        return "video-camera", ["video-camera", "remove", _str(params, "camera_id", UUID)]
    if action == "cleanup-orphans":
        return "cleanup", ["cleanup", "--orphans", *(["--yes"] if params.get("yes") else [])]
    if action == "cleanup-run":
        return "cleanup", ["cleanup", "--run-id", _str(params, "run_id", RUN_ID)]
    raise BadRequest(f"Unknown action {action!r}")


# ---------------------------------------------------------------------------- metrics

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, list[dict[str, Any]]]:
    """{metric: [{"labels": {...}, "value": float}]} for the loadgen metrics in an exposition."""
    metrics: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match or match.group(1) not in METRICS:
            continue
        labels = {key: value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
                  for key, value in _LABEL.findall(match.group(2) or "")}
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        metrics.setdefault(match.group(1), []).append({"labels": labels, "value": value})
    return metrics


def free_port(start: int = METRICS_PORT, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}-{start + attempts - 1} for run metrics")


# ---------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    kind: str
    argv: list[str]
    log_path: Path
    process: subprocess.Popen
    started_at: float
    metrics_port: Optional[int] = None
    run_id: str = ""
    ended_at: Optional[float] = None
    returncode: Optional[int] = None
    stop_requested: bool = False
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "argv": self.argv, "started_at": self.started_at,
                "ended_at": self.ended_at, "returncode": self.returncode, "running": self.ended_at is None,
                "run_id": self.run_id, "metrics_port": self.metrics_port, "stop_requested": self.stop_requested}


class JobManager:
    def __init__(self, log_dir: Path, base_argv: list[str], env: Optional[dict[str, str]] = None) -> None:
        self.log_dir = log_dir
        self.base_argv = base_argv
        self.env = env
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, argv: list[str]) -> Job:
        with self._lock:
            if kind in EXCLUSIVE:
                busy = [job for job in self.jobs.values() if job.kind in EXCLUSIVE and job.ended_at is None]
                if busy:
                    raise BadRequest(f"A {busy[0].kind} is already in progress (job {busy[0].id}); stop it first")
            metrics_port = None
            if kind == "run":
                metrics_port = free_port()
                argv = [*argv, "--metrics-port", str(metrics_port)]
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{job_id}-{kind}.log"
            env = dict(self.env if self.env is not None else os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["LOADGEN_STOP_FILE"] = str(log_path.with_suffix(".stop"))
            env["PYTHONIOENCODING"] = "utf-8"
            # Own process group: Ctrl+C in the UI's terminal does not hit the children directly;
            # serve() stops them through their stop files so runs still tear down.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            handle = log_path.open("wb")
            process = subprocess.Popen([*self.base_argv, *argv], stdout=handle, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env, creationflags=flags,
                                       start_new_session=os.name != "nt")
            handle.close()
            job = Job(job_id, kind, argv, log_path, process, time.time(), metrics_port)
            self.jobs[job_id] = job
        threading.Thread(target=self._watch, args=(job,), name=f"job-{job_id}", daemon=True).start()
        if metrics_port:
            threading.Thread(target=self._poll_metrics, args=(job,), name=f"metrics-{job_id}", daemon=True).start()
        log.info("Job %s: %s", job_id, " ".join(argv))
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def output(self, job: Job, since: int) -> tuple[str, int]:
        with job.log_path.open("rb") as handle:
            handle.seek(max(0, since))
            data = handle.read(1_000_000)
        if not job.run_id:
            match = RUN_STARTED.search(job.log_path.read_text(encoding="utf-8", errors="replace")[:20_000])
            if match:
                job.run_id = match.group(1)
        return data.decode("utf-8", errors="replace"), since + len(data)

    def stop(self, job: Job) -> None:
        """A run stops gracefully through its stop file (teardown runs, as on Ctrl+C); other jobs
        have nothing to undo and are ended."""
        if job.ended_at is not None:
            return
        job.stop_requested = True
        if job.kind == "run":
            job.log_path.with_suffix(".stop").touch()
        elif job.kind != "cleanup":  # never cut a cleanup short
            job.process.kill()

    def kill(self, job: Job) -> None:
        if job.ended_at is None:
            job.process.kill()

    def _watch(self, job: Job) -> None:
        job.returncode = job.process.wait()
        job.ended_at = time.time()
        self.output(job, 0)

    def _poll_metrics(self, job: Job) -> None:
        url = f"http://127.0.0.1:{job.metrics_port}/metrics"
        while job.ended_at is None or time.time() - job.ended_at < 5:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    text = response.read().decode("utf-8", errors="replace")
                metrics = parse_prometheus(text)
                if metrics:
                    with job.lock:
                        job.snapshots.append({"t": time.time(), "metrics": metrics})
                        del job.snapshots[:-MAX_SNAPSHOTS]
                    info = metrics.get("loadgen_run_info")
                    if info and not job.run_id:
                        job.run_id = info[0]["labels"].get("run_id", "")
            except Exception:
                pass  # not serving yet, or already gone
            time.sleep(POLL_S)


# ---------------------------------------------------------------------------- history

def list_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    if not results_dir.is_dir():
        return runs
    for path in sorted(results_dir.glob("*/summary.json"), reverse=True):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        scenario = summary.get("scenario") or {}
        runs.append({
            "run_id": summary.get("run_id", path.parent.name),
            "scenario": scenario.get("name"), "type": scenario.get("type"),
            "workers": scenario.get("workers", []), "source": (scenario.get("source") or {}).get("mode"),
            "started_at": summary.get("started_at"), "ended_at": summary.get("ended_at"),
            "stages": len(summary.get("stages") or []), "error": summary.get("error"),
            "aborted": summary.get("aborted"), "throughput": summary.get("throughput"),
        })
    return runs


def load_run(results_dir: Path, run_id: str) -> dict[str, Any]:
    directory = results_dir / run_id
    summary_path = directory / "summary.json"
    if not RUN_ID.match(run_id) or not summary_path.is_file():
        raise KeyError(run_id)
    rows: list[dict[str, str]] = []
    timeseries = directory / "timeseries.csv"
    if timeseries.is_file():
        with timeseries.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return {"summary": json.loads(summary_path.read_text(encoding="utf-8")), "timeseries": rows}


def list_corpora(corpus_dir: Path) -> list[dict[str, Any]]:
    corpora = []
    if not corpus_dir.is_dir():
        return corpora
    for directory in sorted(corpus_dir.iterdir()):
        if directory.name == UPLOADS or not (directory / MANIFEST_NAME).is_file():
            continue
        header = read_corpus_header(directory)
        corpora.append({"name": directory.name, "frames": header.get("frames"), "fps": header.get("fps"),
                        "image_root": header.get("image_root"), "source_video": header.get("source_video"),
                        "has_header": (directory / HEADER_NAME).is_file()})
    return corpora


# ---------------------------------------------------------------------------- HTTP

class UiServer:
    def __init__(self, app: AppConfig, allow_production: bool = False, env_file: Optional[str] = None,
                 base_argv: Optional[list[str]] = None, env: Optional[dict[str, str]] = None) -> None:
        self.app = app
        self.allow_production = allow_production
        self.results_dir = Path(app.output.results_dir)
        self.corpus_dir = Path(app.frames.corpus_dir)
        self.upload_dir = self.corpus_dir / UPLOADS
        argv = base_argv or [sys.executable, "-m", "vision_loadgen"]
        if env_file is not None and base_argv is None:
            argv += ["--env-file", env_file]
        self.jobs = JobManager(self.results_dir / "ui-jobs", argv, env)

    def config(self) -> dict[str, Any]:
        return {
            "environment": self.app.environment,
            "allow_production": self.allow_production,
            "workers": [{"name": name, "function_key": worker.function_key}
                        for name, worker in self.app.workers.items()],
            "corpus_dir": str(self.corpus_dir.resolve()),
            "results_dir": str(self.results_dir.resolve()),
            "corpus_image_root": self.app.frames.corpus_image_root,
            "copy_target": os.environ.get("LOADGEN_CORPUS_COPY_TARGET", ""),
            "template_camera": self.app.registration.template_camera_id,
        }

    def video_cameras(self) -> list[dict[str, Any]]:
        from vision_loadgen.video_camera import list_video_cameras
        return list_video_cameras(self.app)

    def save_upload(self, name: str, length: int, stream) -> dict[str, Any]:
        if not FILE_NAME.match(name):
            raise BadRequest("Video file names may use letters, digits, space, _ . - and end in .mp4/.avi/.mov/.mkv/.webm")
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise BadRequest("Missing or too large upload")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.upload_dir / name
        partial = target.with_suffix(target.suffix + ".part")
        remaining = length
        with partial.open("wb") as handle:
            while remaining:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        if remaining:
            partial.unlink(missing_ok=True)
            raise BadRequest("Upload interrupted")
        partial.replace(target)
        return {"video": name, "bytes": length}

    def make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def _send(self, status: int, body: Any, content_type: str = "application/json") -> None:
                data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _json_body(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1_000_000:
                    raise BadRequest("Request too large")
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    raise BadRequest("Body must be JSON")
                if not isinstance(body, dict):
                    raise BadRequest("Body must be a JSON object")
                return body

            def _local_origin(self) -> bool:
                # Refuse cross-site form posts from other pages open in the browser.
                origin = self.headers.get("Origin")
                return origin is None or urlparse(origin).netloc == self.headers.get("Host")

            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                query = parse_qs(url.query)
                parts = [unquote(part) for part in url.path.strip("/").split("/") if part]
                try:
                    if not parts or parts == ["index.html"]:
                        return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                    if parts[0] != "api":
                        return self._send(404, {"error": "not found"})
                    route = parts[1:]
                    if route == ["config"]:
                        return self._send(200, server.config())
                    if route == ["corpora"]:
                        return self._send(200, list_corpora(server.corpus_dir))
                    if route == ["uploads"]:
                        files = sorted(p.name for p in server.upload_dir.glob("*") if FILE_NAME.match(p.name)) \
                            if server.upload_dir.is_dir() else []
                        return self._send(200, files)
                    if route == ["video-cameras"]:
                        return self._send(200, server.video_cameras())
                    if route == ["runs"]:
                        return self._send(200, list_runs(server.results_dir))
                    if len(route) == 2 and route[0] == "runs":
                        return self._send(200, load_run(server.results_dir, route[1]))
                    if route == ["jobs"]:
                        jobs = sorted(server.jobs.jobs.values(), key=lambda job: job.started_at, reverse=True)
                        return self._send(200, [job.as_dict() for job in jobs])
                    if len(route) == 2 and route[0] == "jobs":
                        job = server.jobs.get(route[1])
                        text, offset = server.jobs.output(job, int(query.get("since", ["0"])[0]))
                        return self._send(200, {**job.as_dict(), "output": text, "offset": offset})
                    if len(route) == 3 and route[0] == "jobs" and route[2] == "metrics":
                        job = server.jobs.get(route[1])
                        since = int(query.get("since", ["0"])[0])
                        with job.lock:
                            snapshots = job.snapshots[since:]
                            total = len(job.snapshots)
                        return self._send(200, {"snapshots": snapshots, "next": total, "run_id": job.run_id})
                    return self._send(404, {"error": "not found"})
                except KeyError as exc:
                    return self._send(404, {"error": f"not found: {exc}"})
                except BadRequest as exc:
                    return self._send(400, {"error": str(exc)})
                except Exception as exc:
                    log.exception("GET %s failed", self.path)
                    return self._send(500, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                parts = [unquote(part) for part in url.path.strip("/").split("/") if part]
                try:
                    if not self._local_origin():
                        return self._send(403, {"error": "cross-origin request refused"})
                    route = parts[1:] if parts and parts[0] == "api" else []
                    if route == ["videos"]:
                        name = parse_qs(url.query).get("name", [""])[0]
                        result = server.save_upload(name, int(self.headers.get("Content-Length") or 0), self.rfile)
                        return self._send(200, result)
                    if route == ["jobs"]:
                        body = self._json_body()
                        kind, argv = build_command(server.app, str(body.get("action")), body.get("params") or {},
                                                   server.allow_production, server.upload_dir)
                        return self._send(200, server.jobs.start(kind, argv).as_dict())
                    if len(route) == 3 and route[0] == "jobs" and route[2] in ("stop", "kill"):
                        job = server.jobs.get(route[1])
                        (server.jobs.stop if route[2] == "stop" else server.jobs.kill)(job)
                        return self._send(200, job.as_dict())
                    return self._send(404, {"error": "not found"})
                except KeyError as exc:
                    return self._send(404, {"error": f"not found: {exc}"})
                except BadRequest as exc:
                    return self._send(400, {"error": str(exc)})
                except Exception as exc:
                    log.exception("POST %s failed", self.path)
                    return self._send(500, {"error": str(exc)})

        return Handler

    def http_server(self, host: str, port: int) -> ThreadingHTTPServer:
        httpd = ThreadingHTTPServer((host, port), self.make_handler())
        httpd.daemon_threads = True
        return httpd


def serve(app: AppConfig, host: str = "127.0.0.1", port: int = 8765, allow_production: bool = False,
          env_file: Optional[str] = None) -> int:
    if app.environment == "production" and not allow_production:
        log.warning("LOADGEN_ENVIRONMENT is production: runs from the page will be refused "
                    "(start the UI with --allow-production to allow them)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Serving on %s: anyone who can reach it can start load tests", host)
    ui = UiServer(app, allow_production, env_file)
    httpd = ui.http_server(host, port)
    print(f"vision-loadgen UI on http://{host}:{httpd.server_address[1]}/ (Ctrl+C to quit)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        running = [job for job in ui.jobs.jobs.values() if job.ended_at is None]
        for job in running:
            print(f"Stopping job {job.id} ({job.kind}); waiting for its teardown...", flush=True)
            ui.jobs.stop(job)
        for job in running:
            try:
                job.process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print(f"Job {job.id} did not stop; run `cleanup --orphans` later", flush=True)
        httpd.server_close()
    return 0
