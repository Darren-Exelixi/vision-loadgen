from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vision_loadgen.config import ConfigError, ScenarioConfig, deep_merge, expand_env, load_scenario, set_dotted
from vision_loadgen.environment import build_app_config, load_worker_settings

SHARED = {"POSTGRES_URL": "postgresql://u:p@pg:5432", "JWT_SECRET_KEY": "k",
          "CAMERAS_PER_WORKER": "10", "VRAM_PER_WORKER_MB": "1000"}


def _app(environ=None, **kwargs):
    return build_app_config(environ={**SHARED, **(environ or {})}, package_env={}, worker_settings_spec="", **kwargs)


def test_expand_env_uses_value_default_and_empty(monkeypatch):
    monkeypatch.setenv("LG_SET", "value")
    monkeypatch.delenv("LG_UNSET", raising=False)
    assert expand_env("a-${LG_SET}") == "a-value"
    assert expand_env("${LG_UNSET:-fallback}") == "fallback"
    assert expand_env("${LG_UNSET}") == ""
    assert expand_env({"k": ["${LG_SET}", 3]}) == {"k": ["value", 3]}


def test_deep_merge_and_dotted_set():
    assert deep_merge({"a": {"b": 1, "c": 2}, "l": [1]}, {"a": {"b": 3}, "l": [2]}) == {"a": {"b": 3, "c": 2}, "l": [2]}
    data: dict = {}
    set_dotted(data, "guard.action", "backoff")
    set_dotted(data, "levels", "[5, 10]")
    assert data == {"guard": {"action": "backoff"}, "levels": [5, 10]}


def _scenario(**overrides):
    base = {"name": "t", "type": "throughput", "workers": ["crowd"]}
    return ScenarioConfig.model_validate({**base, **overrides})


def test_registered_cameras_per_type():
    assert _scenario(max_cameras=40).registered_cameras == 40
    assert _scenario(type="latency", levels=[5, 20, 10]).registered_cameras == 20
    assert _scenario(type="soak", cameras=7).registered_cameras == 7


@pytest.mark.parametrize("overrides", [
    {"workers": []},
    {"start_cameras": 10, "max_cameras": 5},
    {"step_duration_s": 40, "settle_s": 20, "saturation": {"window_s": 30}},
    {"type": "latency", "levels": []},
    {"type": "latency", "levels": [5], "level_duration_s": 30},
    {"type": "soak", "cameras": 0},
    {"source": {"mode": "corpus"}},
])
def test_invalid_scenarios_rejected(overrides):
    with pytest.raises(ValidationError):
        _scenario(**overrides)


def test_presets_derive_databases_from_postgres_url():
    app = _app()
    assert app.environment == "production"
    assert set(app.workers) == {"crowd", "emotion", "attendance", "ppe", "intrusion", "fire_smoke",
                                "obstacle", "productivity", "fall"}
    assert app.worker_db_url("crowd") == "postgresql://u:p@pg:5432/crowd_gathering_db"
    assert app.worker_db_url("attendance") == "postgresql://u:p@pg:5432/frs_db"
    assert app.worker_db_url("productivity") == "postgresql://u:p@pg:5432/productivity_monitoring_db"
    assert app.worker_name("fire-smoke-detection") == "fire_smoke"
    assert app.main_db_url() == "postgresql://u:p@pg:5432/vision_main_db"
    assert app.workers["emotion"].group_prefix == "sentiment-analysis_consumer_group_"
    assert app.workers["crowd"].cameras_per_worker == 10 and app.workers["crowd"].vram_per_worker_mb == 1000
    assert app.local_worker == ""


def test_explicit_urls_and_config_file_override(tmp_path):
    path = tmp_path / "over.json"
    path.write_text('{"workers": {"crowd": {"gpu_free_vram_mb": "${GPU_FREE:-}", "cameras_per_worker": 4}}}',
                    encoding="utf-8")
    app = _app({"CROWD_DATABASE_URL": "postgresql://crowd", "LOADGEN_ENVIRONMENT": "staging"},
               config_path=str(path))
    assert app.environment == "staging"
    assert app.worker_db_url("crowd") == "postgresql://crowd"
    assert app.workers["crowd"].cameras_per_worker == 4
    assert app.workers["crowd"].gpu_free_vram_mb is None


def test_local_worker_settings_apply_to_that_worker_only():
    settings = SimpleNamespace(
        FUNCTION_NAME="crowd-monitoring", DATABASE_URL="postgresql://w/crowd_gathering_db",
        CAMERAS_PER_WORKER=4, VRAM_PER_WORKER_MB=2500, PUBLIC_BASE_IP="10.0.0.5",
        EVENTS_DIR="/app/events", KAFKA_BOOTSTRAP_SERVERS="kafka:9", POSTGRES_URL="postgresql://w",
    )
    app = build_app_config(environ=dict(SHARED), package_env={}, worker_settings=settings)
    assert app.local_worker == "crowd"
    crowd, emotion = app.workers["crowd"], app.workers["emotion"]
    assert (crowd.cameras_per_worker, crowd.vram_per_worker_mb, crowd.server_ip) == (4, 2500, "10.0.0.5")
    assert app.worker_db_url("crowd") == "postgresql://w/crowd_gathering_db"
    assert (emotion.cameras_per_worker, emotion.server_ip) == (10, "")
    assert app.worker_db_url("emotion") == "postgresql://w/emotion_detection_db"
    assert app.kafka.bootstrap_servers == "kafka:9" and app.events.dir == "/app/events"


def test_worker_settings_absent_outside_worker_images():
    assert load_worker_settings("no_such_package.core.config:settings") is None
    assert load_worker_settings("") is None


def test_load_scenario_presets_workers_and_overrides():
    app = _app()
    scenario = load_scenario("throughput", app, ["max_cameras=20", "guard.action=backoff"], ["crowd-monitoring"])
    assert scenario.workers == ["crowd"] and scenario.max_cameras == 20 and scenario.guard.action == "backoff"
    with pytest.raises(ConfigError):
        load_scenario("latency", app)
    with pytest.raises(ConfigError):
        load_scenario("latency", app, workers=["nope"])
    app.local_worker = "emotion"
    assert load_scenario("soak", app).workers == ["emotion"]


def test_worker_settings_override_is_parsed_and_validated():
    app = _app()
    scenario = load_scenario("soak", app, ["cameras=2", "worker_settings.emotion.max_faces=12"], ["emotion"])
    assert scenario.worker_settings == {"emotion": {"max_faces": 12}}
    # The function key is accepted and normalised to the worker name.
    scenario = load_scenario("soak", app, ["cameras=2", "worker_settings.sentiment-analysis.max_faces=4"],
                             ["emotion"])
    assert scenario.worker_settings == {"emotion": {"max_faces": 4}}


@pytest.mark.parametrize("override, message", [
    ("worker_settings.emotion.min_faces=3", "not tunable"),
    ("worker_settings.emotion.max_faces=0", "below the minimum"),
    ("worker_settings.emotion.max_faces=101", "above the maximum"),
    ("worker_settings.emotion.max_faces=2.5", "integer"),
    ("worker_settings.emotion.max_faces=lots", "number"),
    ("worker_settings.crowd.max_faces=4", "not one of this run's workers"),
    ("worker_settings.nope.max_faces=4", "Unknown worker"),
])
def test_worker_settings_override_rejects_bad_values(override, message):
    with pytest.raises(ConfigError, match=message):
        load_scenario("soak", _app(), ["cameras=2", override], ["emotion"])


def test_scenario_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"name": "x", "type": "soak", "cameras": 1, "workers": ["attendance"]}', encoding="utf-8")
    assert load_scenario(str(path), _app()).workers == ["attendance"]


def test_package_env_reads_vision_shared_env_without_importing_it(tmp_path, monkeypatch):
    import sys

    from vision_loadgen.environment import _package_env

    package = tmp_path / "vision_shared"
    package.mkdir()
    # Importing this package would fail the test: _package_env must only locate it.
    (package / "__init__.py").write_text("raise RuntimeError('vision_shared was imported')\n", encoding="utf-8")
    (package / ".env").write_text("KAFKA_TOPIC=from-shared\nEMPTY=\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vision_shared", raising=False)
    assert _package_env()["KAFKA_TOPIC"] == "from-shared"
    assert "vision_shared" not in sys.modules
    (package / ".env").unlink()
    assert _package_env() == {}
