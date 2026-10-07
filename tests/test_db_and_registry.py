import json

import pytest

from vision_loadgen.config import ScenarioConfig
from vision_loadgen.db import (
    Column,
    audit_overrides,
    build_clone_sql,
    build_json_array_edit,
    build_restore_sql,
    new_key_override,
    scrub_stream_overrides,
    to_text,
)
from vision_loadgen.registrar import plan_restore
from vision_loadgen.registry import Assignment, Registry, SettingsEdit, SettingsOverride
from vision_loadgen.scenarios import build_stages

CAMERA_COLUMNS = [
    Column("id", "uuid"),
    Column("name", "character varying(255)", not_null=True),
    Column("region_id", "uuid"),
    Column("ip", "character varying"),
    Column("rtsp_url", "text"),
    Column("user", "text"),
    Column("created_at", "timestamp with time zone"),
    Column("deleted_at", "timestamp with time zone"),
    Column("search", "tsvector", generated="s"),
]


def test_clone_sql_overrides_copies_and_skips_generated():
    overrides = {**new_key_override(CAMERA_COLUMNS, "id"), **audit_overrides(CAMERA_COLUMNS),
                 **scrub_stream_overrides(CAMERA_COLUMNS), "name": "loadtest-x-000"}
    sql, params = build_clone_sql("cameras", CAMERA_COLUMNS, "id", "id", "tmpl", overrides)
    assert sql.startswith('INSERT INTO "cameras" ("id", "name", "region_id", "ip", "rtsp_url", "user", "created_at", "deleted_at")')
    assert "md5(random()" in sql
    assert '%s::character varying(255)' in sql
    assert 'src."region_id"' in sql and 'src."ip"' in sql
    assert "NULL::text" in sql and "NULL::timestamp with time zone" in sql and "now()" in sql
    assert '"search"' not in sql
    assert 'WHERE src."id"::text = %s AND src.deleted_at IS NULL' in sql
    assert params == ["loadtest-x-000", "tmpl"]


def test_clone_sql_lets_database_generate_serial_key():
    columns = [Column("id", "integer", has_default=True), Column("camera_id", "character varying")]
    assert new_key_override(columns, "id") == {}
    sql, params = build_clone_sql("lines", columns, "id", "camera_id", "tmpl", {"camera_id": "syn"})
    assert sql.startswith('INSERT INTO "lines" ("camera_id") SELECT %s::character varying')
    assert params == ["syn", "tmpl"]


def test_clone_sql_rejects_unknown_override():
    with pytest.raises(ValueError):
        build_clone_sql("cameras", CAMERA_COLUMNS, "id", "id", "t", {"nope": 1})


def test_new_key_override_errors_for_ungeneratable_key():
    with pytest.raises(ValueError):
        new_key_override([Column("id", "integer")], "id")


def test_json_array_edit_add_and_remove():
    sql, params = build_json_array_edit("emotion_settings", "selected_cameras", "json", "id", "7", ["a"], ["a"])
    assert sql.startswith('UPDATE "emotion_settings" SET "selected_cameras" = ((SELECT')
    assert sql.endswith('|| %s::jsonb)::json WHERE "id"::text = %s')
    assert params == [["a"], '["a"]', "7"]

    sql, params = build_json_array_edit("frs_settings", "check_in_cameras", "jsonb", "id", "7", ["a"], [])
    assert "|| %s::jsonb" not in sql and sql.count("%s") == 2
    assert params == [["a"], "7"]

    with pytest.raises(ValueError):
        build_json_array_edit("t", "c", "text", "id", "1", [], [])


def test_restore_sql_and_text_round_trip():
    columns = [Column("id", "uuid"), Column("is_late", "boolean"), Column("shift_working_days", "jsonb")]
    row = {"id": "k", "is_late": to_text(False), "shift_working_days": to_text(["Mon"])}
    sql, params = build_restore_sql("frs_attendance", columns, "id", row)
    assert sql == ('UPDATE "frs_attendance" SET "is_late" = %s::boolean, "shift_working_days" = %s::jsonb '
                   'WHERE "id"::text = %s')
    assert params == ["false", '["Mon"]', "k"]
    assert to_text(None) is None and to_text(b"\x01") == "\\x01"


def test_plan_restore_splits_rows():
    restore, delete, manual = plan_restore(touched={"a", "b", "c"}, shared={"c"}, snapshot_keys={"a", "c"})
    assert (restore, delete, manual) == ({"a"}, {"b"}, {"c"})


def test_registry_round_trip(tmp_path):
    registry = Registry.create(tmp_path, "run-1", "staging")
    registry.camera_ids = ["c1"]
    registry.active_count = 1
    registry.assignments.append(Assignment(worker="crowd", table="function_camera_regions", key="a1"))
    registry.settings.append(SettingsEdit(worker="crowd", table="t", key_column="id", key="1", columns=["c"]))
    registry.overrides.append(SettingsOverride(worker="emotion", table="emotion_settings", key_column="id",
                                               key="1", column="max_faces", original=8, value=12))
    registry.set_status("running")
    loaded = Registry.load(registry.path)
    assert loaded.status == "running" and loaded.camera_ids == ["c1"] and loaded.active_count == 1
    assert loaded.assignments[0].key == "a1" and loaded.settings[0].columns == ["c"]
    assert loaded.overrides == registry.overrides
    assert json.loads((tmp_path / "run-1" / "registry.json").read_text())["run_id"] == "run-1"
    with pytest.raises(FileExistsError):
        Registry.create(tmp_path, "run-1", "staging")


def test_throughput_stages_include_max():
    scenario = ScenarioConfig.model_validate({"name": "t", "type": "throughput", "workers": ["crowd"],
                                              "start_cameras": 5, "step_cameras": 10, "max_cameras": 30})
    assert [stage.cameras for stage in build_stages(scenario)] == [5, 15, 25, 30]


def test_event_folders_swept_inside_the_worker_container(tmp_path, monkeypatch):
    import subprocess

    from vision_loadgen import dockerhost
    from vision_loadgen.environment import build_app_config
    from vision_loadgen.registrar import Registrar

    environ = {"LOADGEN_ENVIRONMENT": "staging", "POSTGRES_URL": "postgresql://x@127.0.0.1:1",
               "LOADGEN_WORKER_LOGS_SSH": "admin1@10.10.10.22", "LOADGEN_WORKER_LOGS_PASSWORD": "pw"}
    app = build_app_config(worker_settings_spec="", environ=environ, package_env={})
    registrar = Registrar(app, ["emotion"], Registry(run_id="r", environment="staging", path=str(tmp_path / "r.json")))
    ids = ["04c9047e-ba79-4f89-26de-aa5611842ee2", "ed5dc71f-2e03-090e-cf92-92b6520562c7"]
    calls = []

    def fake_run(cfg, remote, timeout_s=120):
        calls.append(remote)
        out = "/app/events/sentiment-analysis/images/04c9047e-ba79-4f89-26de-aa5611842ee2\n"
        return subprocess.CompletedProcess(remote, 0, out.encode(), b"")

    monkeypatch.setattr(dockerhost, "run", fake_run)
    assert registrar._sweep_event_folders("emotion", ids) == \
        "1 camera folders removed from sentiment_analysis_backend:/app/events/sentiment-analysis"
    assert calls == [["docker", "exec", "sentiment_analysis_backend", "find", "/app/events/sentiment-analysis",
                      "-mindepth", "1", "-maxdepth", "3", "-type", "d", "(", "-name", ids[0], "-o", "-name", ids[1], ")",
                      "-prune", "-print", "-exec", "rm", "-rf", "{}", "+"]]
    # The password reaches the helper through its environment, never its arguments.
    assert dockerhost.env(app.worker_logs)["LOADGEN_SSH_FOLLOW_PASSWORD"] == "pw"
    assert "pw" not in dockerhost.argv(app.worker_logs, calls[0])

    with pytest.raises(ValueError):
        registrar._sweep_event_folders("emotion", ["../../etc"])
    app.events.container_dir = "/"
    with pytest.raises(ValueError):
        registrar._sweep_event_folders("emotion", ids[:1])
    app.worker_logs.ssh_target = ""
    assert registrar._sweep_event_folders("emotion", ids).startswith("skipped")


def test_event_files_saved_then_deleted_inside_the_worker_container(tmp_path, monkeypatch):
    import subprocess

    from vision_loadgen import db as db_module
    from vision_loadgen import dockerhost
    from vision_loadgen.environment import build_app_config
    from vision_loadgen.registrar import Registrar

    environ = {"LOADGEN_ENVIRONMENT": "staging", "POSTGRES_URL": "postgresql://x@127.0.0.1:1",
               "LOADGEN_WORKER_LOGS_SSH": "admin1@10.10.10.22"}
    app = build_app_config(worker_settings_spec="", environ=environ, package_env={})
    registry = Registry(run_id="r", environment="staging", path=str(tmp_path / "r.json"))
    registrar = Registrar(app, ["attendance"], registry)
    paths = ["ai-attendance/attendance_images/2026-08-21/10-57-08-036_002_time_in.jpg",
             "/app/events/ai-attendance/attendance_videos/2026-08-21/10-57-08-036_002_time_in.mp4",
             "../../etc/passwd"]
    order = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(db_module, "connect", lambda url: Conn())
    monkeypatch.setattr(db_module, "select_values", lambda cur, table, column, ids, columns: list(paths))

    def delete_rows(cur, table, column, ids):
        # The paths must already be on disk in the registry when the rows go.
        order.append(json.loads(open(registry.path, encoding="utf-8").read())["event_files"])
        return 3

    monkeypatch.setattr(db_module, "delete_by_values", delete_rows)
    calls = []
    monkeypatch.setattr(dockerhost, "run", lambda cfg, remote, timeout_s=120: calls.append(remote)
                        or subprocess.CompletedProcess(remote, 0, b"", b""))

    target = app.workers["attendance"].purge[0]
    result = registrar._purge_events("attendance", "postgresql://frs", target, ["cam-1"])
    assert order == [{"attendance:frs_recognition_events": paths}]
    assert result == {"rows_deleted": 3, "files_removed_in_container": 2, "paths_outside_events_dir": 1}
    assert calls == [["docker", "exec", "frs_backend", "rm", "-f", "--",
                      "/app/events/ai-attendance/attendance_images/2026-08-21/10-57-08-036_002_time_in.jpg",
                      "/app/events/ai-attendance/attendance_videos/2026-08-21/10-57-08-036_002_time_in.mp4"]]

    # A retry after the rows are gone still deletes the saved files.
    monkeypatch.setattr(db_module, "select_values", lambda *args: [])
    calls.clear()
    retry = registrar._purge_events("attendance", "postgresql://frs", target, ["cam-1"])
    assert retry["files_removed_in_container"] == 2 and len(calls) == 1


def test_department_links_copied_from_template_region_and_removed(tmp_path, monkeypatch):
    from vision_loadgen import db as db_module
    from vision_loadgen.db import Column
    from vision_loadgen.environment import build_app_config
    from vision_loadgen.registrar import Registrar

    app = build_app_config(worker_settings_spec="", environ={"LOADGEN_ENVIRONMENT": "staging",
                                                             "POSTGRES_URL": "postgresql://x@127.0.0.1:1"},
                           package_env={})
    registry = Registry(run_id="r", environment="staging", path=str(tmp_path / "r.json"))
    registry.region_id = "syn-region"
    executed = []

    class Cursor:
        rowcount = 2

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            executed.append((" ".join(sql.split()), params))

        def fetchone(self):
            return {"present": True}

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(db_module, "table_columns", lambda cur, table: [
        Column("department_id", "uuid", False, "", "", True), Column("camera_region_id", "uuid", False, "", "", True),
        Column("created_by", "uuid", False, "", "", False)])
    Registrar(app, ["attendance"], registry)._copy_department_links(Conn(), "template-region")
    assert executed[-1] == (
        "INSERT INTO department_camera_regions (department_id, camera_region_id, created_by) "
        "SELECT department_id, %s::uuid, created_by FROM department_camera_regions "
        "WHERE camera_region_id = %s::uuid ON CONFLICT DO NOTHING", ("syn-region", "template-region"))

    monkeypatch.setattr(db_module, "connect", lambda url: Conn())
    assert Registrar._remove_department_links("postgresql://main", "syn-region") == "2 deleted"
    assert executed[-1] == ("DELETE FROM department_camera_regions WHERE camera_region_id = %s::uuid", ("syn-region",))
    assert app.workers["attendance"].copy_department_links and not app.workers["crowd"].copy_department_links


def _override_db(monkeypatch, row_value):
    """Fake DB holding one settings row; records executed statements."""
    from vision_loadgen import db as db_module

    state = {"value": row_value, "statements": []}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params):
            state["statements"].append((sql, params))
            if sql.startswith("UPDATE"):
                state["value"] = params[0]

        def fetchone(self):
            return {"value": state["value"]}

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(db_module, "connect", lambda url: Conn())
    return state


def _emotion_registrar(tmp_path, worker_settings=None):
    from vision_loadgen.environment import build_app_config
    from vision_loadgen.registrar import Registrar

    environ = {"LOADGEN_ENVIRONMENT": "staging", "POSTGRES_URL": "postgresql://x@127.0.0.1:1"}
    app = build_app_config(worker_settings_spec="", environ=environ, package_env={})
    registry = Registry.create(tmp_path, "run-o", "staging")
    return Registrar(app, ["emotion"], registry, worker_settings), registry


def test_worker_settings_override_recorded_before_update_then_restored(tmp_path, monkeypatch):
    state = _override_db(monkeypatch, 8)
    registrar, registry = _emotion_registrar(tmp_path, {"emotion": {"max_faces": 12}})
    seen_on_disk = []
    original_write = registrar._write_setting

    def write(cur, table, key_column, key, column, value):
        seen_on_disk.append(json.loads(open(registry.path, encoding="utf-8").read())["overrides"])
        original_write(cur, table, key_column, key, column, value)

    monkeypatch.setattr(registrar, "_write_setting", write)
    registrar._apply_overrides({"emotion": "5"})

    assert state["value"] == 12
    assert seen_on_disk[0] == [{"worker": "emotion", "table": "emotion_settings", "key_column": "id", "key": "5",
                                "column": "max_faces", "original": 8, "value": 12}]
    update_sql, update_params = state["statements"][-1]
    assert update_sql == 'UPDATE "emotion_settings" SET "max_faces" = %s WHERE "id"::text = %s'
    assert update_params == (12, "5")

    # A later cleanup works from the registry file alone.
    loaded = Registry.load(registry.path)
    assert registrar._restore_override(loaded.overrides[0]) == "restored 8"
    assert state["value"] == 8


def test_teardown_restores_overrides_before_syncing_workers(tmp_path, monkeypatch):
    state = _override_db(monkeypatch, 12)
    registrar, registry = _emotion_registrar(tmp_path)
    registry.workers = {"emotion": {}}
    registry.overrides.append(SettingsOverride(worker="emotion", table="emotion_settings", key_column="id",
                                               key="5", column="max_faces", original=8, value=12))
    synced_with = []
    monkeypatch.setattr(registrar, "_client_for", lambda name: None)
    monkeypatch.setattr(registrar, "_sync", lambda name, client: synced_with.append(state["value"]) or True)
    monkeypatch.setattr(registrar, "_remove_main", lambda url, table, keys: "0 deleted")

    report = registrar.teardown(keep_events=True)
    assert report["steps"]["override:emotion.max_faces"] == "restored 8"
    assert synced_with == [8]
    assert not report["errors"]
