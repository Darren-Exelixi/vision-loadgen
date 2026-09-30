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
from vision_loadgen.registry import Assignment, Registry, SettingsEdit
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
    registry.set_status("running")
    loaded = Registry.load(registry.path)
    assert loaded.status == "running" and loaded.camera_ids == ["c1"] and loaded.active_count == 1
    assert loaded.assignments[0].key == "a1" and loaded.settings[0].columns == ["c"]
    assert json.loads((tmp_path / "run-1" / "registry.json").read_text())["run_id"] == "run-1"
    with pytest.raises(FileExistsError):
        Registry.create(tmp_path, "run-1", "staging")


def test_throughput_stages_include_max():
    scenario = ScenarioConfig.model_validate({"name": "t", "type": "throughput", "workers": ["crowd"],
                                              "start_cameras": 5, "step_cameras": 10, "max_cameras": 30})
    assert [stage.cameras for stage in build_stages(scenario)] == [5, 15, 25, 30]
