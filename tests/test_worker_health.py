"""Worker-health guard and the worker's own metrics (stage times, processed fps)."""

import logging

import pytest

from vision_loadgen.analysis import (
    WorkerHealthGuard,
    WorkerSample,
    histogram_quantile,
    stage_hint,
    stage_stats,
    stage_verdict,
)
from vision_loadgen.config import GuardConfig, SaturationConfig, WorkerMetricsConfig
from vision_loadgen.metrics import ActiveCameras, Sampler, WorkerInternals, stage_buckets, stage_quantiles
from vision_loadgen.presets import WORKER_PRESETS
from vision_loadgen.workers import WorkerStatus

EMOTION = WorkerMetricsConfig(**WORKER_PRESETS["emotion"]["metrics"])


def _down(t, worker="emotion"):
    return WorkerSample(worker=worker, t=t, ok=False, error="timed out")


def _up(t, lag=0, worker="emotion"):
    return WorkerSample(worker=worker, t=t, ok=True, running=True, lag=lag)


# ------------------------------------------------------------------ guard

def test_unreachable_worker_trips_after_the_limit_and_resets_when_it_answers():
    guard = WorkerHealthGuard(GuardConfig(unreachable_s=30))
    assert guard.check(_down(100), 50) is None
    assert guard.check(_down(125), 50) is None
    assert guard.check(_up(126), 50) is None  # answered: the clock starts over
    assert guard.check(_down(130), 50) is None
    assert guard.check(_down(159), 50) is None
    reason = guard.check(_down(160), 50)
    assert reason and "unreachable for 30s" in reason and "timed out" in reason


def test_unreachable_is_ignored_while_a_resync_settles():
    guard = WorkerHealthGuard(GuardConfig(unreachable_s=10))
    guard.check(_down(100), 50)
    guard.hold(200)  # re-sync: the worker restarts its pipeline
    assert guard.check(_down(150), 50) is None
    assert guard.check(_down(205), 50) is None
    assert guard.check(_down(214), 50) is None
    assert guard.check(_down(215), 50) is not None


def test_runaway_backlog_trips_only_while_growing_past_the_limit():
    guard = WorkerHealthGuard(GuardConfig(max_backlog_s=60, grace_s=15))
    # 50 frames/s published: 60 s of backlog is 3000 messages.
    assert guard.check(_up(0, lag=2000), 50) is None
    assert guard.check(_up(5, lag=3500), 50) is None
    assert guard.check(_up(10, lag=4500), 50) is None
    reason = guard.check(_up(20, lag=6000), 50)
    assert reason and "backlog 6000 messages (120s of publishing)" in reason

    # Grew early, then flat for longer than grace_s: judged on the recent window only.
    guard = WorkerHealthGuard(GuardConfig(max_backlog_s=60, grace_s=15))
    for t, lag in [(0, 8000), (5, 9000), (10, 9000), (20, 9000), (30, 9000), (40, 9000)]:
        assert guard.check(_up(t, lag=lag), 50) is None

    # Over the limit but draining: the worker is catching up, leave it.
    guard = WorkerHealthGuard(GuardConfig(max_backlog_s=60, grace_s=15))
    for t, lag in [(0, 9000), (10, 8000), (20, 7000), (30, 6000)]:
        assert guard.check(_up(t, lag=lag), 50) is None


def test_health_checks_can_be_turned_off():
    for cfg in (GuardConfig(enabled=False), GuardConfig(worker_health=False)):
        guard = WorkerHealthGuard(cfg)
        assert guard.check(_down(0), 50) is None and guard.check(_down(1000), 50) is None
    guard = WorkerHealthGuard(GuardConfig(unreachable_s=0, max_backlog_s=0))
    assert guard.check(_down(0), 50) is None and guard.check(_down(1000), 50) is None
    assert guard.check(_up(0, lag=10**9), 50) is None and guard.check(_up(100, lag=10**10), 50) is None


# ------------------------------------------------------------------ stage times

def test_histogram_quantile_interpolates_like_prometheus():
    buckets = [(0.01, 0.0), (0.02, 50.0), (0.05, 90.0), (0.1, 100.0), (float("inf"), 100.0)]
    assert histogram_quantile(0.5, buckets) == pytest.approx(0.02)
    assert histogram_quantile(0.25, buckets) == pytest.approx(0.015)
    assert histogram_quantile(0.95, buckets) == pytest.approx(0.075)
    assert histogram_quantile(0.5, [(0.1, 0.0), (float("inf"), 0.0)]) is None
    # Everything above the highest bound: report that bound.
    assert histogram_quantile(0.5, [(0.1, 0.0), (2.0, 0.0), (float("inf"), 10.0)]) == 2.0


def _exposition(detect: dict, frames: dict) -> str:
    lines = ["# HELP emotion_stage_seconds Per-frame time", "# TYPE emotion_stage_seconds histogram"]
    for bound, count in detect.items():
        lines.append(f'emotion_stage_seconds_bucket{{le="{bound}",stage="detect"}} {count}')
    lines.append(f'emotion_stage_seconds_count{{stage="detect"}} {max(detect.values())}')
    for camera, count in frames.items():
        lines.append(f'emotion_frames_processed_total{{camera_id="{camera}"}} {count}')
    return "\n".join(lines) + "\n"


def test_worker_internals_use_the_interval_since_the_previous_sample():
    internals = WorkerInternals(EMOTION, synthetic_ids={"syn-1", "syn-2"})
    # Since the worker started: slow (CPU) frames. Only the next interval should count.
    first = _exposition({"0.05": 0, "0.5": 0, "1.0": 100, "+Inf": 100}, {"syn-1": 10, "real": 500})
    assert internals.update(first, 100.0) == ({}, None)
    second = _exposition({"0.05": 100, "0.5": 100, "1.0": 200, "+Inf": 200}, {"syn-1": 40, "syn-2": 20, "real": 900})
    stages, fps = internals.update(second, 110.0)
    assert stages["detect"]["0.5"] == pytest.approx(0.025)  # all 100 new frames under 50 ms
    assert fps == pytest.approx(5.0)  # (60 - 10) synthetic frames in 10 s; the real camera is ignored

    # The worker restarted (counters went down): skip one interval rather than report nonsense.
    restarted = _exposition({"0.05": 5, "0.5": 5, "1.0": 5, "+Inf": 5}, {"syn-1": 3})
    assert internals.update(restarted, 120.0) == ({}, None)


def test_stage_hint_and_stats():
    slow = [WorkerSample(worker="emotion", t=t, ok=True, running=True,
                         stage_seconds={"detect": {"0.5": 0.4, "0.95": 0.6}}, processed_fps=fps)
            for t, fps in [(10, 4.0), (15, 6.0)]]
    assert "detect p50 400 ms > 150 ms" in stage_hint(slow, EMOTION.gpu_stage_limits)
    assert stage_hint(slow, {"detect": 0.5}) == ""
    verdict = stage_verdict([*slow], 0, SaturationConfig(), EMOTION.gpu_stage_limits)
    assert "CPU-bound" in verdict.hint
    stats = stage_stats(slow)
    assert stats["worker_stage_p95_s"] == {"detect": pytest.approx(0.6)}
    assert stats["processed_fps"] == pytest.approx(5.0)


class _Client:
    def __init__(self, text=None):
        self.text = text
        self.clock = type("Clock", (), {"seconds": 0.0})()

    def status(self):
        return WorkerStatus(running=True, active_cameras={"syn-1"}, processed_timestamps={})

    def metrics_text(self, path):
        if self.text is None:
            raise RuntimeError("HTTP 404")
        return self.text


def test_sampler_reads_worker_metrics_and_tolerates_a_missing_endpoint(caplog):
    active = ActiveCameras(["syn-1"])
    active.set_count(1, 0.0)
    text = _exposition({"0.05": 10, "+Inf": 10}, {"syn-1": 10})
    sampler = Sampler({"emotion": _Client(text), "crowd": _Client(None)}, {"emotion": "e_", "crowd": "c_"}, None,
                      active, {"syn-1"}, 1.0,
                      worker_metrics={"emotion": EMOTION, "crowd": EMOTION})
    with caplog.at_level(logging.WARNING):
        sampler.sample_once()
        batch = {sample.worker: sample for sample in sampler.sample_once()}
    assert batch["crowd"].ok and batch["crowd"].stage_seconds == {}
    assert sum("worker metrics unavailable" in record.message for record in caplog.records) == 1
    assert batch["emotion"].ok  # identical counters: no stage times for an empty interval
    assert batch["emotion"].stage_seconds == {}


def test_check_reads_cumulative_stage_times():
    text = _exposition({"0.05": 0, "0.5": 90, "1.0": 100, "+Inf": 100}, {})
    stages = stage_quantiles(stage_buckets(text, "emotion_stage_seconds"))
    assert stages["detect"]["0.5"] == pytest.approx(0.3)
