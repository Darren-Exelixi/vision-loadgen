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
    percentile,
    real_staleness,
    synthetic_staleness_by_camera,
)
from vision_loadgen.capacity import gpu_worker_map
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
    ) -> None:
        self._clients = clients
        self._group_prefixes = group_prefixes
        self._lag_reader = lag_reader
        self._active = active
        self._synthetic_ids = synthetic_ids
        self._interval_s = interval_s
        self._cameras_per_worker = cameras_per_worker or {}
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
            ))
        return batch

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


class Recorder:
    """Console line, timeseries.csv row per sample batch."""

    def __init__(self, directory: Path, workers: list[str]) -> None:
        self._workers = workers
        self._started = time.time()
        self._last_acked = 0
        self._last_time = self._started
        self._file = (directory / "timeseries.csv").open("w", newline="", encoding="utf-8")
        columns = ["time", "elapsed_s", "stage", "active_cameras", "sent", "acked", "errors", "publish_rate"]
        for worker in workers:
            columns += [f"{worker}_ok", f"{worker}_lag", f"{worker}_staleness_p50", f"{worker}_staleness_p95",
                        f"{worker}_staleness_max", f"{worker}_real_excess"]
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
            if sample is None or not sample.ok:
                row += [False, sample.lag if sample else None, None, None, None, None]
                parts.append(f"{worker}: DOWN")
                continue
            p50 = percentile(sample.synthetic_staleness, 50)
            p95 = percentile(sample.synthetic_staleness, 95)
            worst = max(sample.synthetic_staleness) if sample.synthetic_staleness else None
            row += [True, sample.lag, _fmt(p50, 2), _fmt(p95, 2), _fmt(worst, 2), _fmt(excess.get(worker), 2)]
            parts.append(f"{worker}: lag={sample.lag if sample.lag is not None else '-'} "
                         f"p95={_fmt(p95)}s real+={_fmt(excess.get(worker))}s")
        self._writer.writerow(row)
        self._file.flush()
        print(f"[{now - self._started:7.0f}s] {stage:<14} cams={active:<4} pub={rate:7.1f}/s "
              f"err={stats.errors} | " + " | ".join(parts), flush=True)

    def close(self) -> None:
        self._file.close()
