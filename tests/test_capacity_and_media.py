import pytest

from vision_loadgen.capacity import gpu_workers_for, plan_capacity
from vision_loadgen.config import ConfigError, ScenarioConfig, SettingsTarget, WorkerConfig
from vision_loadgen.media import delete_files, resolve_event_path, sweep_camera_folders
from vision_loadgen.scenarios import build_stages, cap_stages


def _worker(**overrides):
    return WorkerConfig(function_key="crowd-monitoring",
                        settings=SettingsTarget(table="t", camera_columns=["c"]), **overrides)


def test_gpu_workers_mirror_engine_rounding():
    assert [gpu_workers_for(n, 10) for n in (0, 1, 10, 11, 25)] == [1, 1, 1, 2, 3]


def test_capacity_from_configured_budget():
    capacity = plan_capacity("crowd", _worker(cameras_per_worker=10, vram_per_worker_mb=1000, gpu_free_vram_mb=4500),
                             real_cameras=12)
    assert capacity.max_total_cameras == 40 and capacity.max_synthetic_cameras == 28
    assert capacity.gpu_workers(8) == 2 and capacity.gpu_workers(9) == 3


def test_capacity_estimated_from_probe_and_api_model():
    worker = _worker(cameras_per_worker=10, vram_per_worker_mb=1000, api_worker=True)
    # 1500 MB free now + (2 copies for 12 cameras + 1 API model) x 1000 MB = 4500 MB -> 4 copies - API = 3.
    capacity = plan_capacity("crowd", worker, real_cameras=12, probe_free_mb=1500)
    assert capacity.budget_mb == 4500 and capacity.max_total_cameras == 30
    assert capacity.budget_source.startswith("estimated")


def test_capacity_unknown_without_settings_or_budget():
    assert plan_capacity("crowd", _worker(), 3).max_synthetic_cameras is None
    assert plan_capacity("crowd", _worker(cameras_per_worker=10, vram_per_worker_mb=1000), 3).max_synthetic_cameras is None


def _throughput(**overrides):
    return ScenarioConfig.model_validate({"name": "t", "type": "throughput", "workers": ["crowd"],
                                          "start_cameras": 10, "step_cameras": 10, "max_cameras": 50, **overrides})


def test_cap_stages_ends_ramp_at_gpu_cap():
    capacity = plan_capacity("crowd", _worker(cameras_per_worker=10, vram_per_worker_mb=1000, gpu_free_vram_mb=4000),
                             real_cameras=5)
    scenario = _throughput()
    stages, note = cap_stages(build_stages(scenario), {"crowd": capacity}, scenario)
    assert [stage.cameras for stage in stages] == [10, 20, 30, 35]
    assert "35" in note and "40, 50" in note


def test_cap_stages_without_room_raises_and_unknown_does_not_cap():
    full = plan_capacity("crowd", _worker(cameras_per_worker=10, vram_per_worker_mb=1000, gpu_free_vram_mb=1000), 10)
    soak = ScenarioConfig.model_validate({"name": "s", "type": "soak", "workers": ["crowd"], "cameras": 5})
    with pytest.raises(ConfigError):
        cap_stages(build_stages(soak), {"crowd": full}, soak)
    scenario = _throughput()
    stages, note = cap_stages(build_stages(scenario), {"crowd": plan_capacity("crowd", _worker(), 1)}, scenario)
    assert len(stages) == 5 and note is None


def test_event_paths_stay_inside_events_dir(tmp_path):
    root = tmp_path.resolve()
    assert resolve_event_path(root, "crowd/img/a.jpg") == root / "crowd" / "img" / "a.jpg"
    assert resolve_event_path(root, str(root / "x.jpg")) == root / "x.jpg"
    assert resolve_event_path(root, "../outside.jpg") is None
    assert resolve_event_path(root, "") is None


def test_delete_files_and_sweep_camera_folders(tmp_path):
    events = tmp_path / "events"
    synthetic = events / "crowd-monitoring" / "crowd_monitoring_images" / "syn-1" / "2026-09-29"
    real = events / "crowd-monitoring" / "crowd_monitoring_images" / "real-1" / "2026-09-29"
    for folder in (synthetic, real):
        folder.mkdir(parents=True)
        (folder / "a.jpg").write_bytes(b"x")
    outside = tmp_path / "keep.jpg"
    outside.write_bytes(b"x")

    result = delete_files(str(events), ["crowd-monitoring/crowd_monitoring_images/syn-1/2026-09-29/a.jpg",
                                        str(outside), "missing.jpg"])
    assert result == {"files_deleted": 1, "paths_outside_events_dir": 1}
    assert outside.exists()

    assert sweep_camera_folders(str(events / "crowd-monitoring"), ["syn-1"]) == 1
    assert not synthetic.parent.exists() and (real / "a.jpg").exists()
    assert sweep_camera_folders(str(events / "nope"), ["syn-1"]) == 0
