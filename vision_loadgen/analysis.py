"""Pure measurement logic: staleness, verdicts and the real-camera guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vision_loadgen.config import GuardConfig, SaturationConfig


@dataclass
class WorkerSample:
    worker: str
    t: float
    ok: bool
    running: bool = False
    lag: Optional[int] = None
    synthetic_staleness: list[float] = field(default_factory=list)
    real_staleness: dict[str, float] = field(default_factory=dict)
    # Synthetic staleness per GPU model copy (index -> values), when CAMERAS_PER_WORKER is known.
    gpu_staleness: dict[int, list[float]] = field(default_factory=dict)
    error: str = ""


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y over x; 0 with fewer than two distinct x values."""
    if len(points) < 2:
        return 0.0
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def synthetic_staleness_by_camera(
    now: float,
    active_ids: list[str],
    activated_at: dict[str, float],
    processed: dict[str, float],
) -> dict[str, float]:
    """Seconds since each active synthetic camera's newest processed frame.

    A camera not processed since it was activated counts from its activation time, so a worker
    that silently drops a camera's frames shows up as growing staleness.
    """
    values = {}
    for camera_id in active_ids:
        activated = activated_at[camera_id]
        processed_at = processed.get(camera_id)
        reference = processed_at if processed_at is not None and processed_at >= activated else activated
        values[camera_id] = max(0.0, now - reference)
    return values


def synthetic_staleness(now: float, active_ids: list[str], activated_at: dict[str, float],
                        processed: dict[str, float]) -> list[float]:
    return list(synthetic_staleness_by_camera(now, active_ids, activated_at, processed).values())


def group_by_gpu_worker(staleness: dict[str, float], gpu_map: dict[str, int]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for camera_id, value in staleness.items():
        if camera_id in gpu_map:
            grouped.setdefault(gpu_map[camera_id], []).append(value)
    return grouped


def real_staleness(now: float, active: set[str], synthetic: set[str], processed: dict[str, float]) -> dict[str, float]:
    return {
        camera_id: max(0.0, now - processed_at)
        for camera_id, processed_at in processed.items()
        if camera_id in active and camera_id not in synthetic
    }


@dataclass
class StageVerdict:
    kept_up: bool
    staleness_p95: Optional[float]
    lag_growth_per_s: float
    samples: int
    reason: str


def stage_verdict(samples: list[WorkerSample], window_start: float, cfg: SaturationConfig) -> StageVerdict:
    window = [sample for sample in samples if sample.t >= window_start and sample.ok]
    if not window:
        return StageVerdict(False, None, 0.0, 0, "no successful status samples in window")
    staleness = [value for sample in window for value in sample.synthetic_staleness]
    p95 = percentile(staleness, 95)
    growth = slope([(sample.t, float(sample.lag)) for sample in window if sample.lag is not None])
    reasons = []
    if p95 is None:
        reasons.append("no synthetic cameras measured")
    elif p95 > cfg.max_staleness_s:
        reasons.append(f"staleness p95 {p95:.1f}s > {cfg.max_staleness_s:.1f}s")
    if growth > cfg.max_lag_growth_per_s:
        reasons.append(f"lag growing {growth:.1f} msg/s > {cfg.max_lag_growth_per_s:.1f}")
    if any(not sample.running for sample in window):
        reasons.append("worker reported not running")
    return StageVerdict(not reasons, p95, growth, len(window), "; ".join(reasons) or "kept up")


def max_sustained(results: list[tuple[int, StageVerdict]]) -> Optional[int]:
    """Highest camera count kept up with before the first failed stage."""
    best = None
    for cameras, verdict in results:
        if not verdict.kept_up:
            break
        best = cameras
    return best


def stage_stats(samples: list[WorkerSample]) -> dict:
    staleness = [value for sample in samples if sample.ok for value in sample.synthetic_staleness]
    lags = [sample.lag for sample in samples if sample.lag is not None]
    per_gpu: dict[int, list[float]] = {}
    for sample in samples:
        if sample.ok:
            for index, values in sample.gpu_staleness.items():
                per_gpu.setdefault(index, []).extend(values)
    stats: dict = {
        "staleness_p50": percentile(staleness, 50),
        "staleness_p95": percentile(staleness, 95),
        "staleness_max": max(staleness) if staleness else None,
        "lag_max": max(lags) if lags else None,
        "status_failures": sum(1 for sample in samples if not sample.ok),
    }
    if per_gpu:
        stats["gpu_workers"] = {
            str(index): {"staleness_p95": percentile(values, 95), "staleness_max": max(values)}
            for index, values in sorted(per_gpu.items())
        }
    return stats


def soak_summary(samples: list[WorkerSample], start: float, bucket_s: float = 3600.0) -> dict:
    buckets: dict[int, list[float]] = {}
    for sample in samples:
        if sample.ok:
            buckets.setdefault(int((sample.t - start) // bucket_s), []).extend(sample.synthetic_staleness)
    hourly = [
        {"bucket": index, "staleness_p95": percentile(values, 95)}
        for index, values in sorted(buckets.items())
    ]
    first = hourly[0]["staleness_p95"] if hourly else None
    last = hourly[-1]["staleness_p95"] if hourly else None
    return {
        "buckets": hourly,
        "staleness_p95_drift_s": (last - first) if first is not None and last is not None else None,
        "lag_slope_per_s": slope([(sample.t, float(sample.lag)) for sample in samples if sample.lag is not None]),
        "status_failures": sum(1 for sample in samples if not sample.ok),
        "not_running_samples": sum(1 for sample in samples if sample.ok and not sample.running),
    }


class RealCameraGuard:
    """Trips when a worker's real cameras fall behind their pre-load baseline for too long."""

    def __init__(self, cfg: GuardConfig, offline_after_s: float = 60.0) -> None:
        self.cfg = cfg
        self.offline_after_s = offline_after_s
        self.baseline: dict[str, dict[str, float]] = {}
        self._breach_since: dict[str, float] = {}
        self._hold_until = 0.0

    def hold(self, until: float) -> None:
        """Ignore samples taken before `until`: a re-sync restarts the pipeline for every camera."""
        self._hold_until = max(self._hold_until, until)
        self._breach_since.clear()

    def calibrate(self, samples: list[WorkerSample]) -> None:
        worst: dict[str, dict[str, float]] = {}
        for sample in samples:
            per_camera = worst.setdefault(sample.worker, {})
            for camera_id, value in sample.real_staleness.items():
                per_camera[camera_id] = max(per_camera.get(camera_id, 0.0), value)
        self.baseline = {
            worker: {camera_id: value for camera_id, value in cameras.items() if value <= self.offline_after_s}
            for worker, cameras in worst.items()
        }

    def excess(self, sample: WorkerSample) -> Optional[float]:
        baseline = self.baseline.get(sample.worker, {})
        deltas = [
            sample.real_staleness[camera_id] - value
            for camera_id, value in baseline.items()
            if camera_id in sample.real_staleness
        ]
        return max(deltas) if deltas else None

    def check(self, sample: WorkerSample) -> Optional[str]:
        if not self.cfg.enabled or not sample.ok:
            return None
        if sample.t < self._hold_until:
            self._breach_since.pop(sample.worker, None)
            return None
        excess = self.excess(sample)
        if excess is None or excess <= self.cfg.max_real_staleness_increase_s:
            self._breach_since.pop(sample.worker, None)
            return None
        since = self._breach_since.setdefault(sample.worker, sample.t)
        if sample.t - since < self.cfg.grace_s:
            return None
        return (
            f"{sample.worker}: real cameras {excess:.1f}s behind baseline for {sample.t - since:.0f}s "
            f"(limit {self.cfg.max_real_staleness_increase_s:.0f}s)"
        )

    def reset(self, worker: str) -> None:
        self._breach_since.pop(worker, None)
