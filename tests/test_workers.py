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
