"""Live metrics endpoint for a run (Prometheus text format), written with the stdlib only.

The web UI reads a run through it; anything that speaks the format can scrape it too. Worker
images get this package copied in without pip, so prometheus_client is not an option; the text
exposition format for gauges and counters is simple enough to write and parse directly.
Off unless a port is configured (--metrics-port; the UI always sets one).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# name -> (type, help, label names). Every series also carries `run_id`.
METRICS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "loadgen_run_info": ("gauge", "Run being executed (always 1)",
                         ("scenario", "scenario_type", "environment", "template_camera")),
    "loadgen_run_start_time_seconds": ("gauge", "Unix time the run started",
                                       ("scenario", "scenario_type", "environment")),
    "loadgen_run_end_time_seconds": ("gauge", "Unix time the run finished", ()),
    "loadgen_phase": ("gauge", "Current phase of the run (1 for the active phase)", ("phase",)),
    "loadgen_stage_info": ("gauge", "Stage being executed (always 1)", ("stage", "stage_index")),
    "loadgen_stage_index": ("gauge", "1-based index of the stage being executed", ()),
    "loadgen_active_cameras": ("gauge", "Synthetic cameras being published to", ()),
    "loadgen_planned_cameras": ("gauge", "Synthetic cameras planned for the current stage", ()),
    "loadgen_stage_planned_seconds": ("gauge", "Planned duration of the current stage", ()),
    "loadgen_stage_count": ("gauge", "Stages planned for the run", ()),
    "loadgen_guard_real_cameras": ("gauge", "Real cameras the guard watches (0: it cannot protect this worker)", ("worker",)),
    "loadgen_frames_sent_total": ("counter", "Frame pointers sent to Kafka", ()),
    "loadgen_frames_acked_total": ("counter", "Frame pointers acknowledged by Kafka", ()),
    "loadgen_publish_errors_total": ("counter", "Frame pointers Kafka rejected", ()),
    "loadgen_publish_rate": ("gauge", "Frame pointers acknowledged per second since the previous sample", ()),
    "loadgen_abort_info": ("gauge", "Why the run was aborted (always 1; absent unless aborted)", ("reason",)),
    "loadgen_worker_up": ("gauge", "1 when the worker's /worker/status answered", ("worker",)),
    "loadgen_worker_running": ("gauge", "1 when the worker reports its pipeline running", ("worker",)),
    "loadgen_consumer_lag": ("gauge", "Kafka consumer lag of the worker's consumer groups", ("worker",)),
    "loadgen_synthetic_staleness_seconds": ("gauge", "Synthetic-camera staleness across cameras",
                                            ("worker", "quantile")),
    "loadgen_real_staleness_excess_seconds": ("gauge", "Real-camera staleness above the pre-run baseline",
                                              ("worker",)),
    "loadgen_gpu_copy_staleness_p95_seconds": ("gauge", "Synthetic-camera staleness p95 per GPU model copy",
                                               ("worker", "gpu_copy")),
    "loadgen_gpu_workers_expected": ("gauge", "GPU model copies the worker should run at this stage",
                                     ("worker",)),
    "loadgen_worker_stage_seconds": ("gauge", "Worker's own per-frame time in each pipeline stage, "
                                              "since the previous sample", ("worker", "stage", "quantile")),
    "loadgen_worker_processed_fps": ("gauge", "Synthetic-camera frames the worker processed per second "
                                              "(from its own metrics)", ("worker",)),
    "loadgen_host_cpu_percent": ("gauge", "CPU use of the workers' Docker host, all cores (0-100)", ()),
    "loadgen_host_memory_used_bytes": ("gauge", "RAM in use on the workers' Docker host", ()),
    "loadgen_host_memory_total_bytes": ("gauge", "RAM installed on the workers' Docker host", ()),
    "loadgen_gpu_utilization_percent": ("gauge", "GPU utilization on the workers' Docker host", ("gpu",)),
    "loadgen_gpu_memory_used_bytes": ("gauge", "VRAM in use", ("gpu",)),
    "loadgen_gpu_memory_total_bytes": ("gauge", "VRAM installed", ("gpu",)),
    "loadgen_container_cpu_percent": ("gauge", "CPU use of the worker's container (100 = one core)", ("worker",)),
    "loadgen_container_memory_bytes": ("gauge", "RAM in use by the worker's container", ("worker",)),
    "loadgen_clock_offset_seconds": ("gauge", "Worker clock minus this machine's clock", ("worker",)),
    "loadgen_threshold_seconds": ("gauge", "Scenario thresholds (max_staleness: keep-up limit; "
                                           "guard: allowed real-camera staleness increase)", ("kind",)),
    "loadgen_stage_kept_up": ("gauge", "Stage verdict: 1 kept up, 0 fell behind",
                              ("worker", "stage", "stage_index")),
    "loadgen_stage_staleness_p95_seconds": ("gauge", "Synthetic-camera staleness p95 over the verdict window",
                                            ("worker", "stage", "stage_index")),
    "loadgen_max_sustained_cameras": ("gauge", "Throughput result: most synthetic cameras kept up with",
                                      ("worker",)),
}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_exposition(text: str, names: Optional[Iterable[str]] = None) -> dict[str, list[dict[str, Any]]]:
    """{metric: [{"labels": {...}, "value": float}]} from Prometheus text, optionally only `names`.

    Histogram series keep their suffixed names (`x_bucket` with an `le` label, `x_count`, `x_sum`).
    """
    wanted = set(names) if names is not None else None
    metrics: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match or (wanted is not None and match.group(1) not in wanted):
            continue
        labels = {key: value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
                  for key, value in _LABEL.findall(match.group(2) or "")}
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        metrics.setdefault(match.group(1), []).append({"labels": labels, "value": value})
    return metrics


class MetricsExporter:
    def __init__(self, run_id: str, addr: str = "127.0.0.1", port: int = 9464) -> None:
        self.run_id = run_id
        self.addr = addr
        self.port = port
        self._lock = threading.Lock()
        self._series: dict[str, dict[tuple[str, ...], float]] = {name: {} for name in METRICS}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _key(self, name: str, labels: dict[str, object]) -> tuple[str, ...]:
        if name not in METRICS:
            raise KeyError(f"Undeclared metric {name}")
        expected = METRICS[name][2]
        if set(labels) != set(expected):
            raise ValueError(f"{name} takes labels {expected}, got {tuple(labels)}")
        return tuple(str(labels[label]) for label in expected)

    def set(self, name: str, value: Optional[float], **labels: object) -> None:
        """Set a series; None removes it."""
        key = self._key(name, labels)
        with self._lock:
            if value is None:
                self._series[name].pop(key, None)
            else:
                self._series[name][key] = float(value)

    def inc(self, name: str, amount: float = 1.0, **labels: object) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._series[name][key] = self._series[name].get(key, 0.0) + amount

    def clear(self, name: str, **match: object) -> None:
        """Drop the series of `name` whose labels equal `match` (all of them without `match`)."""
        names = METRICS[name][2]
        positions = {names.index(label): str(value) for label, value in match.items()}
        with self._lock:
            series = self._series[name]
            for key in [key for key in series if all(key[i] == value for i, value in positions.items())]:
                del series[key]

    def value(self, name: str, **labels: object) -> Optional[float]:
        with self._lock:
            return self._series[name].get(self._key(name, labels))

    def render(self) -> str:
        lines = []
        run_label = f'run_id="{_escape(self.run_id)}"'
        with self._lock:
            for name, (kind, help_text, label_names) in METRICS.items():
                series = self._series[name]
                if not series:
                    continue
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                for key, value in sorted(series.items()):
                    labels = [run_label] + [f'{label}="{_escape(item)}"' for label, item in zip(label_names, key)]
                    lines.append(f"{name}{{{','.join(labels)}}} {_number(value)}")
        return "\n".join(lines) + "\n" if lines else ""

    def start(self) -> bool:
        """Serve /metrics in the background. Returns False (and logs) when the port is unavailable."""
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                if self.path.split("?", 1)[0] != "/metrics":
                    body, status, content_type = b"vision-loadgen: see /metrics\n", 404, "text/plain"
                else:
                    body, status, content_type = exporter.render().encode("utf-8"), 200, CONTENT_TYPE
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                return

        try:
            self._server = ThreadingHTTPServer((self.addr, self.port), Handler)
        except OSError as exc:
            log.warning("Metrics endpoint not started on %s:%s: %s (the run continues)", self.addr, self.port, exc)
            return False
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="loadgen-metrics", daemon=True)
        self._thread.start()
        log.info("Prometheus metrics on http://%s:%d/metrics", self.addr, self.port)
        return True

    def stop(self, linger_s: float = 0.0) -> None:
        """Keep serving for `linger_s` so the final values get scraped, then stop."""
        if self._server is None:
            return
        if linger_s > 0:
            log.info("Serving final metrics for %.0fs before exiting", linger_s)
            time.sleep(linger_s)
        self._server.shutdown()
        self._server.server_close()
        self._server = None
