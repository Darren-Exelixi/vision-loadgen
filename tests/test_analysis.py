import pytest

from vision_loadgen.analysis import (
    RealCameraGuard,
    StageVerdict,
    WorkerSample,
    group_by_gpu_worker,
    max_sustained,
    percentile,
    real_staleness,
    slope,
    soak_summary,
    stage_stats,
    stage_verdict,
    synthetic_staleness,
)
from vision_loadgen.capacity import gpu_worker_map
from vision_loadgen.config import GuardConfig, SaturationConfig


def test_percentile_and_slope():
    assert percentile([], 95) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert slope([(0, 0), (1, 2), (2, 4)]) == pytest.approx(2)
    assert slope([(0, 5)]) == 0.0


def test_synthetic_staleness_counts_unprocessed_from_activation():
    values = synthetic_staleness(
        now=100.0,
        active_ids=["a", "b", "c"],
        activated_at={"a": 90.0, "b": 90.0, "c": 90.0},
        processed={"a": 99.0, "b": 80.0},
    )
    assert values == [1.0, 10.0, 10.0]


def test_real_staleness_excludes_synthetic_and_inactive():
    assert real_staleness(10.0, {"r", "s"}, {"s"}, {"r": 8.0, "s": 9.0, "gone": 1.0}) == {"r": 2.0}


def _sample(t, staleness, lag=0, ok=True, running=True, worker="crowd", real=None):
    return WorkerSample(worker=worker, t=t, ok=ok, running=running, lag=lag,
                        synthetic_staleness=staleness, real_staleness=real or {})


def test_stage_verdict_kept_up_and_failures():
    cfg = SaturationConfig(window_s=30, max_staleness_s=5, max_lag_growth_per_s=1)
    healthy = [_sample(t, [0.5, 1.0], lag=10) for t in range(0, 60, 5)]
    assert stage_verdict(healthy, 30, cfg).kept_up

    slow = [_sample(t, [8.0], lag=10) for t in range(0, 60, 5)]
    assert "staleness" in stage_verdict(slow, 30, cfg).reason

    lagging = [_sample(t, [0.5], lag=t * 50) for t in range(0, 60, 5)]
    verdict = stage_verdict(lagging, 30, cfg)
    assert not verdict.kept_up and "lag growing" in verdict.reason

    assert not stage_verdict([_sample(40, [0.1], ok=False)], 30, cfg).kept_up
    assert not stage_verdict([_sample(40, [0.1], running=False)], 30, cfg).kept_up


def test_max_sustained_stops_at_first_failure():
    ok = StageVerdict(True, 1.0, 0.0, 5, "kept up")
    bad = StageVerdict(False, 9.0, 0.0, 5, "slow")
    assert max_sustained([(5, ok), (10, ok), (15, bad), (20, ok)]) == 10
    assert max_sustained([(5, bad)]) is None


def test_guard_trips_only_after_grace_and_ignores_offline_cameras():
    guard = RealCameraGuard(GuardConfig(max_real_staleness_increase_s=10, grace_s=15), offline_after_s=60)
    guard.calibrate([_sample(0, [], real={"cam": 1.0, "offline": 500.0})])
    assert guard.baseline == {"crowd": {"cam": 1.0}}

    assert guard.check(_sample(10, [], real={"cam": 5.0, "offline": 900.0})) is None
    assert guard.check(_sample(20, [], real={"cam": 20.0})) is None
    assert guard.check(_sample(30, [], real={"cam": 25.0})) is None
    assert guard.check(_sample(36, [], real={"cam": 25.0})) is not None


def test_guard_recovers_and_can_be_disabled():
    guard = RealCameraGuard(GuardConfig(max_real_staleness_increase_s=10, grace_s=5))
    guard.calibrate([_sample(0, [], real={"cam": 1.0})])
    assert guard.check(_sample(10, [], real={"cam": 30.0})) is None
    assert guard.check(_sample(12, [], real={"cam": 2.0})) is None
    assert guard.check(_sample(16, [], real={"cam": 30.0})) is None

    disabled = RealCameraGuard(GuardConfig(enabled=False))
    disabled.calibrate([_sample(0, [], real={"cam": 1.0})])
    assert disabled.check(_sample(100, [], real={"cam": 999.0})) is None


def test_guard_hold_ignores_resync_window():
    guard = RealCameraGuard(GuardConfig(max_real_staleness_increase_s=10, grace_s=5))
    guard.calibrate([_sample(0, [], real={"cam": 1.0})])
    guard.hold(until=50)
    assert guard.check(_sample(10, [], real={"cam": 40.0})) is None
    assert guard.check(_sample(49, [], real={"cam": 40.0})) is None
    assert guard.check(_sample(51, [], real={"cam": 40.0})) is None
    assert guard.check(_sample(57, [], real={"cam": 40.0})) is not None


def test_staleness_grouped_by_gpu_model_copy():
    # Sorted ids: a r1 s1 s2 -> copies 0 0 1 1 with two cameras per copy.
    gpu_map = gpu_worker_map({"s2", "a", "s1", "r1"}, cameras_per_worker=2)
    assert gpu_map == {"a": 0, "r1": 0, "s1": 1, "s2": 1}
    grouped = group_by_gpu_worker({"a": 1.0, "s1": 9.0, "s2": 7.0}, gpu_map)
    assert grouped == {0: [1.0], 1: [9.0, 7.0]}
    stats = stage_stats([WorkerSample(worker="crowd", t=0, ok=True, synthetic_staleness=[1.0, 9.0, 7.0],
                                      gpu_staleness=grouped)])
    assert stats["gpu_workers"]["1"]["staleness_max"] == 9.0


def test_soak_summary_drift():
    samples = [_sample(0, [1.0], lag=0), _sample(3600, [3.0], lag=0), _sample(7200, [4.0], lag=0, ok=False)]
    summary = soak_summary(samples, start=0)
    assert summary["staleness_p95_drift_s"] == pytest.approx(2.0)
    assert summary["status_failures"] == 1
