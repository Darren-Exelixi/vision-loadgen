from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from vision_loadgen.analysis import (
    WorkerSample,
    group_by_gpu_worker,
    histogram_quantile,
    percentile,
    real_staleness,
    synthetic_staleness_by_camera,
)
from vision_loadgen.capacity import gpu_worker_map
from vision_loadgen.config import WorkerMetricsConfig
from vision_loadgen.exporter import MetricsExporter, parse_exposition
from vision_loadgen.hoststats import HostSample, HostStats
from vision_loadgen.kafka_io import LagReader, ProducerStats
from vision_loadgen.workers import WorkerClient

log = logging.getLogger(__name__)


class ActiveCameras:
    """Which synthetic cameras are being published to, and since when. Shared with the sampler."""

    def __init__(self, camera_ids: list[str]) -> None:
        self._ids = list(camera_ids)
        self._lock = threading.Lock()
        self._count = 0
        self._activated_at: dict[str, float] = {}

    def set_count(self, count: int, now: float, restart: bool = False) -> None:
        """`restart`: the workers were re-synced, so every active camera starts measuring now."""
        with self._lock:
            start = 0 if restart else self._count
            for camera_id in self._ids[start:count]:
                self._activated_at[camera_id] = now
            self._count = count

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def snapshot(self) -> tuple[list[str], dict[str, float]]:
        with self._lock:
            active = self._ids[: self._count]
            return active, {camera_id: self._activated_at[camera_id] for camera_id in active}


QUANTILES = ("0.5", "0.95")


def stage_buckets(text: str, histogram: str) -> dict[str, dict[float, float]]:
    """stage -> {upper bound: cumulative count} from a worker's exposition."""
    buckets: dict[str, dict[float, float]] = {}
    for item in parse_exposition(text, [f"{histogram}_bucket"]).get(f"{histogram}_bucket", []):
        stage, bound = item["labels"].get("stage"), item["labels"].get("le")
        if stage is None or bound is None:
            continue
        try:
            buckets.setdefault(stage, {})[float(bound)] = item["value"]
        except ValueError:
            continue
    return buckets


def stage_quantiles(buckets: dict[str, dict[float, float]]) -> dict[str, dict[str, float]]:
    """stage -> {"0.5": s, "0.95": s}; stages without observations are left out."""
    result = {}
    for stage, counts in buckets.items():
        ordered = sorted(counts.items())
        values = {q: histogram_quantile(float(q), ordered) for q in QUANTILES}
        if all(value is not None for value in values.values()):
            result[stage] = values
    return result


class WorkerInternals:
    """A worker's own metrics turned into per-interval values: stage time quantiles and the rate
    at which it processed synthetic cameras' frames. Its counters are cumulative since the worker
    started, so each sample uses the difference from the previous one; a worker restart (counts
    going down) skips one interval."""

    def __init__(self, cfg: WorkerMetricsConfig, synthetic_ids: set[str]) -> None:
        self.cfg = cfg
        self._synthetic_ids = synthetic_ids
        self._buckets: Optional[dict[str, dict[float, float]]] = None
        self._frames: Optional[tuple[float, float]] = None

    def update(self, text: str, t: float) -> tuple[dict[str, dict[str, float]], Optional[float]]:
        stages: dict[str, dict[str, float]] = {}
        if self.cfg.stage_histogram:
            current = stage_buckets(text, self.cfg.stage_histogram)
            if self._buckets is not None:
                deltas = {}
                for stage, counts in current.items():
                    previous = self._buckets.get(stage, {})
                    delta = {bound: count - previous.get(bound, 0.0) for bound, count in counts.items()}
                    if all(value >= 0 for value in delta.values()):
                        deltas[stage] = delta
                stages = stage_quantiles(deltas)
            self._buckets = current
        fps = None
        if self.cfg.frames_counter:
            series = parse_exposition(text, [self.cfg.frames_counter]).get(self.cfg.frames_counter, [])
            frames = sum(item["value"] for item in series if item["labels"].get("camera_id") in self._synthetic_ids)
            if self._frames is not None:
                previous_t, previous = self._frames
                if t > previous_t and frames >= previous:
                    fps = (frames - previous) / (t - previous_t)
            self._frames = (t, frames)
        return stages, fps


class Sampler:
    def __init__(
        self,
        clients: dict[str, WorkerClient],
        group_prefixes: dict[str, str],
        lag_reader: Optional[LagReader],
        active: ActiveCameras,
        synthetic_ids: set[str],
        interval_s: float,
        cameras_per_worker: Optional[dict[str, Optional[int]]] = None,
        worker_metrics: Optional[dict[str, WorkerMetricsConfig]] = None,
    ) -> None:
        self._clients = clients
        self._group_prefixes = group_prefixes
        self._lag_reader = lag_reader
        self._active = active
        self._synthetic_ids = synthetic_ids
        self._interval_s = interval_s
        self._cameras_per_worker = cameras_per_worker or {}
        self._internals = {
            name: WorkerInternals(cfg, synthetic_ids)
            for name, cfg in (worker_metrics or {}).items() if cfg.path and name in clients
        }
        self._internals_failed: set[str] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.batches: "queue.Queue[list[WorkerSample]]" = queue.Queue()

    def sample_once(self) -> list[WorkerSample]:
        active_ids, activated_at = self._active.snapshot()
        batch = []
        for name, client in self._clients.items():
            lag = None
            if self._lag_reader is not None:
                try:
                    lag = self._lag_reader.lag(self._group_prefixes[name])
                except Exception as exc:
                    log.debug("Lag read failed for %s: %s", name, exc)
            try:
                status = client.status()
            except Exception as exc:
                batch.append(WorkerSample(worker=name, t=time.time(), ok=False, lag=lag, error=str(exc)))
                continue
            stage_seconds, processed_fps = self._read_internals(name, client)
            now = time.time()
            by_camera = synthetic_staleness_by_camera(now, active_ids, activated_at, status.processed_timestamps)
            cameras_per_worker = self._cameras_per_worker.get(name)
            batch.append(WorkerSample(
                worker=name,
                t=now,
                ok=True,
                running=status.running,
                lag=lag,
                synthetic_staleness=list(by_camera.values()),
                # Real cameras' timestamps come from the ingestion side's clock, so compare them in
                # the worker's time. Synthetic ones are stamped by this process: no correction.
                real_staleness=real_staleness(now + client.clock.seconds, status.active_cameras,
                                              self._synthetic_ids, status.processed_timestamps),
                gpu_staleness=group_by_gpu_worker(by_camera, gpu_worker_map(status.active_cameras, cameras_per_worker))
                if cameras_per_worker else {},
                stage_seconds=stage_seconds,
                processed_fps=processed_fps,
            ))
        return batch

    def _read_internals(self, name: str, client: WorkerClient) -> tuple[dict[str, dict[str, float]], Optional[float]]:
        """Optional: a worker without a metrics endpoint (or an unreachable one) never fails a sample."""
        internals = self._internals.get(name)
        if internals is None:
            return {}, None
        try:
            text = client.metrics_text(internals.cfg.path)
        except Exception as exc:
            if name not in self._internals_failed:
                self._internals_failed.add(name)
                log.warning("%s: worker metrics unavailable (%s); stage times will be missing", name, exc)
            return {}, None
        return internals.update(text, time.time())

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="loadgen-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self.batches.put(self.sample_once())
            self._stop.wait(max(0.0, self._interval_s - (time.monotonic() - started)))


def _fmt(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(value, digits)


def _summary(values: list[Optional[float]], *, average: bool = False) -> dict[str, float]:
    present = [value for value in values if value is not None]
    if not present:
        return {}
    out = {"max": round(max(present), 1)}
    if average:
        out["avg"] = round(sum(present) / len(present), 1)
    return out


class Recorder:
    """Console line, timeseries.csv row and (optionally) live metrics per sample batch."""

    def __init__(self, directory: Path, workers: list[str], exporter: Optional[MetricsExporter] = None,
                 stage_columns: Optional[dict[str, list[str]]] = None, host: Optional[HostStats] = None) -> None:
        self._workers = workers
        # Host CPU/RAM/GPU readings; columns exist only when the host answered before the run began.
        self._host = host
        self._host_by_stage: dict[str, list[HostSample]] = {}
        # Worker stages written to the CSV (its columns are fixed when the file is opened).
        self._stage_columns = stage_columns or {}
        self._exporter = exporter
        self._started = time.time()
        self._last_acked = 0
        self._last_time = self._started
        self._file = (directory / "timeseries.csv").open("w", newline="", encoding="utf-8")
        columns = ["time", "elapsed_s", "stage", "active_cameras", "sent", "acked", "errors", "publish_rate"]
        for worker in workers:
            columns += [f"{worker}_ok", f"{worker}_lag", f"{worker}_staleness_p50", f"{worker}_staleness_p95",
                        f"{worker}_staleness_max", f"{worker}_real_excess", f"{worker}_processed_fps"]
            columns += [f"{worker}_stage_{stage}_p95" for stage in self._stage_columns.get(worker, [])]
        if host is not None:
            columns += ["host_cpu_pct", "host_ram_used_mb", "host_ram_total_mb"]
            for gpu in range(host.gpu_count):
                columns += [f"gpu{gpu}_util_pct", f"gpu{gpu}_vram_used_mb", f"gpu{gpu}_vram_total_mb"]
            for worker in self._host_workers():
                columns += [f"{worker}_container_cpu_pct", f"{worker}_container_ram_mb"]
        self._writer = csv.writer(self._file)
        self._writer.writerow(columns)

    def record(self, stage: str, active: int, stats: ProducerStats, batch: list[WorkerSample],
               excess: dict[str, Optional[float]]) -> None:
        now = time.time()
        rate = (stats.acked - self._last_acked) / max(1e-6, now - self._last_time)
        self._last_acked, self._last_time = stats.acked, now
        by_worker = {sample.worker: sample for sample in batch}
        row: list = [
            datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            round(now - self._started, 1), stage, active, stats.sent, stats.acked, stats.errors, round(rate, 1),
        ]
        parts = []
        for worker in self._workers:
            sample = by_worker.get(worker)
            stages = self._stage_columns.get(worker, [])
            if sample is None or not sample.ok:
                row += [False, sample.lag if sample else None, None, None, None, None, None] + [None] * len(stages)
                parts.append(f"{worker}: DOWN")
                continue
            p50 = percentile(sample.synthetic_staleness, 50)
            p95 = percentile(sample.synthetic_staleness, 95)
            worst = max(sample.synthetic_staleness) if sample.synthetic_staleness else None
            row += [True, sample.lag, _fmt(p50, 2), _fmt(p95, 2), _fmt(worst, 2), _fmt(excess.get(worker), 2),
                    _fmt(sample.processed_fps, 1)]
            row += [_fmt(sample.stage_seconds.get(stage, {}).get("0.95"), 4) for stage in stages]
            part = (f"{worker}: lag={sample.lag if sample.lag is not None else '-'} "
                    f"p95={_fmt(p95)}s real+={_fmt(excess.get(worker))}s")
            if sample.processed_fps is not None:
                part += f" done={sample.processed_fps:.1f}/s"
            parts.append(part)
        host_sample = self._host.latest() if self._host is not None else None
        if self._host is not None:
            row += self._host_row(host_sample)
            if host_sample is not None:
                self._host_by_stage.setdefault(stage, []).append(host_sample)
        self._writer.writerow(row)
        self._file.flush()
        if self._exporter is not None:
            self._export(active, stats, rate, by_worker, excess)
            if self._host is not None:
                self._export_host(host_sample)
        print(f"[{now - self._started:7.0f}s] {stage:<14} cams={active:<4} pub={rate:7.1f}/s "
              f"err={stats.errors} | " + " | ".join(parts), flush=True)

    def _host_workers(self) -> list[str]:
        return [worker for worker in self._workers if worker in self._host.containers]

    def _host_row(self, sample: Optional[HostSample]) -> list:
        """CSV cells for the host columns, in the order the header lists them; blanks without a reading."""
        row: list = [_round(sample.cpu_pct) if sample else None, _round(sample.ram_used_mb, 0) if sample else None,
                     _round(sample.ram_total_mb, 0) if sample else None]
        gpus = {gpu.index: gpu for gpu in sample.gpus} if sample else {}
        for index in range(self._host.gpu_count):
            gpu = gpus.get(index)
            row += [_round(gpu.util_pct) if gpu else None, _round(gpu.vram_used_mb, 0) if gpu else None,
                    _round(gpu.vram_total_mb, 0) if gpu else None]
        for worker in self._host_workers():
            container = sample.containers.get(self._host.containers[worker]) if sample else None
            row += [_round(container.cpu_pct) if container else None, _round(container.ram_mb, 0) if container else None]
        return row

    def _export_host(self, sample: Optional[HostSample]) -> None:
        exporter = self._exporter
        mb = 1024 * 1024

        def bytes_of(value: Optional[float]) -> Optional[float]:
            return None if value is None else value * mb

        exporter.set("loadgen_host_cpu_percent", sample.cpu_pct if sample else None)
        exporter.set("loadgen_host_memory_used_bytes", bytes_of(sample.ram_used_mb) if sample else None)
        exporter.set("loadgen_host_memory_total_bytes", bytes_of(sample.ram_total_mb) if sample else None)
        for name in ("loadgen_gpu_utilization_percent", "loadgen_gpu_memory_used_bytes", "loadgen_gpu_memory_total_bytes"):
            exporter.clear(name)
        for gpu in sample.gpus if sample else []:
            exporter.set("loadgen_gpu_utilization_percent", gpu.util_pct, gpu=gpu.index)
            exporter.set("loadgen_gpu_memory_used_bytes", bytes_of(gpu.vram_used_mb), gpu=gpu.index)
            exporter.set("loadgen_gpu_memory_total_bytes", bytes_of(gpu.vram_total_mb), gpu=gpu.index)
        for worker in self._host_workers():
            container = sample.containers.get(self._host.containers[worker]) if sample else None
            exporter.set("loadgen_container_cpu_percent", container.cpu_pct if container else None, worker=worker)
            exporter.set("loadgen_container_memory_bytes", bytes_of(container.ram_mb) if container else None, worker=worker)

    def stage_resources(self, stage: str) -> dict:
        """Peaks and averages of the host readings taken during `stage`, for summary.json."""
        samples = self._host_by_stage.get(stage)
        if not samples:
            return {}
        out: dict = {"readings": len(samples), "host": {
            "cpu_pct": _summary([s.cpu_pct for s in samples], average=True),
            "ram_used_mb": _summary([s.ram_used_mb for s in samples])}}
        gpus: dict[str, dict] = {}
        for index in range(self._host.gpu_count):
            readings = [gpu for s in samples for gpu in s.gpus if gpu.index == index]
            gpus[str(index)] = {"util_pct": _summary([g.util_pct for g in readings], average=True),
                                "vram_used_mb": _summary([g.vram_used_mb for g in readings]),
                                "vram_total_mb": readings[-1].vram_total_mb if readings else None}
        if gpus:
            out["gpus"] = gpus
        containers = {}
        for worker in self._host_workers():
            readings = [s.containers[self._host.containers[worker]] for s in samples
                        if self._host.containers[worker] in s.containers]
            if readings:
                containers[worker] = {"cpu_pct": _summary([c.cpu_pct for c in readings], average=True),
                                      "ram_mb": _summary([c.ram_mb for c in readings])}
        if containers:
            out["containers"] = containers
        return out

    def _export(self, active: int, stats: ProducerStats, rate: float, by_worker: dict[str, WorkerSample],
                excess: dict[str, Optional[float]]) -> None:
        exporter = self._exporter
        exporter.set("loadgen_active_cameras", active)
        exporter.set("loadgen_publish_rate", rate)
        exporter.set("loadgen_frames_sent_total", stats.sent)
        exporter.set("loadgen_frames_acked_total", stats.acked)
        exporter.set("loadgen_publish_errors_total", stats.errors)
        for worker in self._workers:
            sample = by_worker.get(worker)
            ok = sample is not None and sample.ok
            exporter.set("loadgen_worker_up", 1 if ok else 0, worker=worker)
            exporter.set("loadgen_consumer_lag", sample.lag if sample else None, worker=worker)
            exporter.clear("loadgen_gpu_copy_staleness_p95_seconds", worker=worker)
            if not ok:
                exporter.set("loadgen_worker_running", None, worker=worker)
                exporter.clear("loadgen_synthetic_staleness_seconds", worker=worker)
                exporter.set("loadgen_real_staleness_excess_seconds", None, worker=worker)
                exporter.clear("loadgen_worker_stage_seconds", worker=worker)
                exporter.set("loadgen_worker_processed_fps", None, worker=worker)
                continue
            values = sample.synthetic_staleness
            exporter.set("loadgen_worker_running", 1 if sample.running else 0, worker=worker)
            exporter.set("loadgen_synthetic_staleness_seconds", percentile(values, 50), worker=worker, quantile="0.5")
            exporter.set("loadgen_synthetic_staleness_seconds", percentile(values, 95), worker=worker, quantile="0.95")
            exporter.set("loadgen_synthetic_staleness_seconds", max(values) if values else None,
                         worker=worker, quantile="max")
            exporter.set("loadgen_real_staleness_excess_seconds", excess.get(worker), worker=worker)
            for index, copy_values in sample.gpu_staleness.items():
                exporter.set("loadgen_gpu_copy_staleness_p95_seconds", percentile(copy_values, 95),
                             worker=worker, gpu_copy=index)
            # A stage with no frames in this interval keeps its last value rather than flickering out.
            for stage, quantiles in sample.stage_seconds.items():
                for quantile, value in quantiles.items():
                    exporter.set("loadgen_worker_stage_seconds", value, worker=worker, stage=stage, quantile=quantile)
            if sample.processed_fps is not None:
                exporter.set("loadgen_worker_processed_fps", sample.processed_fps, worker=worker)

    def close(self) -> None:
        self._file.close()
