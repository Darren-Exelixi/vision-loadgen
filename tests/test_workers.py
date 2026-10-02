import os

import pytest

from vision_loadgen.config import ConfigError
from vision_loadgen.environment import load_env_file
from vision_loadgen.workers import ClockOffset, WorkerClient


def test_clock_offset_narrows_with_more_responses():
    clock = ClockOffset()
    assert not clock.known and clock.seconds == 0.0 and clock.as_dict()["offset_s"] is None
    offset = 3.4  # worker is 3.4 s ahead
    for local in (1000.05, 1000.35, 1000.65, 1000.95, 1001.25):
        sent, received = local, local + 0.02
        server = local + 0.01 + offset
        clock.add(sent, received, float(int(server)))  # Date header: whole seconds
    assert clock.known
    assert abs(clock.seconds - offset) <= clock.error_s + 1e-9
    assert clock.error_s < 0.2


def test_clock_offset_restarts_when_a_clock_steps():
    clock = ClockOffset()
    clock.add(1000.0, 1000.1, 1000.0)
    clock.add(2000.0, 2000.1, 2060.0)  # 60 s jump: incompatible with the first interval
    assert abs(clock.seconds - 60.0) <= 1.0


def test_record_clock_parses_date_header():
    client = WorkerClient("crowd", "http://w:1", "/api/v1", auth=None)
    client._record_clock(1_700_000_000.2, 1_700_000_000.3, "Tue, 14 Nov 2023 22:13:30 GMT")  # 1_700_000_010
    assert 9.0 <= client.clock.seconds <= 10.8
    before = client.clock.as_dict()
    client._record_clock(0.0, 0.1, "not a date")
    client._record_clock(0.0, 0.1, None)
    assert client.clock.as_dict() == before


def test_load_env_file_keeps_set_variables(tmp_path, monkeypatch):
    env_file = tmp_path / "staging.env"
    env_file.write_text("# comment\nKAFKA_TOPIC=from-file\nPOSTGRES_URL=postgresql://file\nEMPTY=\n", encoding="utf-8")
    environ = {"POSTGRES_URL": "postgresql://shell"}
    assert load_env_file(str(env_file), environ) == env_file
    assert environ == {"POSTGRES_URL": "postgresql://shell", "KAFKA_TOPIC": "from-file", "EMPTY": ""}

    with pytest.raises(ConfigError):
        load_env_file(str(tmp_path / "missing.env"), {})
    assert load_env_file("", {}) is None
    with pytest.raises(ConfigError):
        load_env_file(None, {"LOADGEN_ENV_FILE": str(tmp_path / "missing.env")})


def test_load_env_file_defaults_to_dot_env_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_env_file(None, {}) is None  # no ./.env: nothing, no error
    (tmp_path / ".env").write_text("TIMEZONE=Asia/Dubai\n", encoding="utf-8")
    environ: dict[str, str] = {}
    assert load_env_file(None, environ) is not None
    assert environ["TIMEZONE"] == "Asia/Dubai"


def _attendance_client(responses, calls):
    """An attendance WorkerClient whose HTTP calls replay `responses[(method, path)]` in order."""
    client = WorkerClient("attendance", "http://w:7021", "/api/v1", auth=None, api="attendance",
                          function_key="ai-attendance", server_function_id="sf-1", sync_timeout_s=5, poll_s=0)

    def fake_call(method, path, timeout_s, body=None):
        calls.append((method, path, body))
        queue = responses[(method, path)]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    client._call = fake_call
    return client


PROCESSED = {"status": "success", "cameras": [{"camera_id": "cam-a"}, "cam-b"],
             "buffer_info": {"cam-a": {"has_frame": True, "timestamp": 100.5}, "cam-b": {"last_timestamp": 99.0}},
             "diagnostics": {"kafka_consumer_running": True}}


def test_attendance_sync_posts_its_body_and_waits_for_the_new_job():
    old = {"job_id": "old", "status": "completed"}
    new = {"job_id": "job-2", "status": "completed", "phase": "completed", "duration_seconds": 5.8,
           "source_diagnostics": {"assigned_camera_count": 4, "mapped_camera_count": 4, "unmapped_camera_count": 0}}
    calls = []
    client = _attendance_client({
        ("GET", "/worker/sync/status"): [
            {"current": None, "last": old, "pending": None},                          # before the POST
            {"current": {"job_id": "job-2", "status": "running"}, "last": old, "pending": None},
            {"current": None, "last": new, "pending": None},
        ],
        ("POST", "/worker/sync"): [{"status": "accepted", "job_id": "job-2"}],
        ("GET", "/streaming/processed/list"): [PROCESSED],
    }, calls)
    status = client.sync()
    assert ("POST", "/worker/sync", {"module_name": "ai-attendance", "server_function_id": "sf-1"}) in calls
    assert [c[1] for c in calls].count("/worker/sync/status") == 3
    assert status.running and status.active_cameras == {"cam-a", "cam-b"}
    assert status.processed_timestamps == {"cam-a": 100.5, "cam-b": 99.0}
    assert not client.lists_enabled_cameras


def test_attendance_sync_without_a_job_id_waits_for_a_different_last_job():
    calls = []
    client = _attendance_client({
        ("GET", "/worker/sync/status"): [
            {"current": None, "last": {"job_id": "old", "status": "completed"}, "pending": None},
            {"current": None, "last": {"job_id": "old", "status": "completed"}, "pending": None},
            {"current": None, "last": {"job_id": "new", "status": "completed"}, "pending": None},
        ],
        ("POST", "/worker/sync"): [{"status": "accepted"}],
        ("GET", "/streaming/processed/list"): [PROCESSED],
    }, calls)
    client.sync()
    assert [c[1] for c in calls].count("/worker/sync/status") == 3


def test_attendance_sync_failures_raise():
    failed = {"job_id": "job-2", "status": "failed", "phase": "milvus_rebuild", "error": "milvus down"}
    client = _attendance_client({
        ("GET", "/worker/sync/status"): [{"current": None, "last": None, "pending": None},
                                         {"current": None, "last": failed, "pending": None}],
        ("POST", "/worker/sync"): [{"job_id": "job-2"}],
    }, [])
    with pytest.raises(RuntimeError, match="failed.*milvus down"):
        client.sync()

    stuck = _attendance_client({
        ("GET", "/worker/sync/status"): [{"current": {"job_id": "job-2"}, "last": None, "pending": None}],
        ("POST", "/worker/sync"): [{"job_id": "job-2"}],
    }, [])
    stuck._sync_timeout_s = 0
    with pytest.raises(TimeoutError):
        stuck.sync()

    with pytest.raises(ConfigError):
        WorkerClient("attendance", "http://w:1", "/api/v1", auth=None, api="attendance").sync()


def test_standard_status_still_requires_has_frame():
    from vision_loadgen.workers import parse_status
    status = parse_status({"running": True, "active_cameras": ["a", "b"],
                           "processed_buffer_info": {"a": {"has_frame": True, "timestamp": 1.0},
                                                     "b": {"timestamp": 2.0}}})
    assert status.processed_timestamps == {"a": 1.0}
    assert WorkerClient("crowd", "http://w:1", "/api/v1", auth=None).lists_enabled_cameras
