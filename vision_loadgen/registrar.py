"""Creates synthetic cameras in both registration layers and removes everything afterwards."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from vision_loadgen import db, dockerhost, media
from vision_loadgen.config import AppConfig, ConfigError, PurgeTarget
from vision_loadgen.frames import build_message
from vision_loadgen.kafka_io import TopicWaker
from vision_loadgen.registry import Assignment, CameraRows, Registry, SettingsEdit, SettingsOverride, Snapshot
from vision_loadgen.workers import Deployment, WorkerClient, WorkerStatus, resolve_deployment

log = logging.getLogger(__name__)

CAMERAS = "cameras"
REGIONS = "camera_regions"
ASSIGNMENTS = "function_camera_regions"
DEPARTMENT_LINKS = "department_camera_regions"
# Teardown removes folders named after these inside worker containers; keep both strict.
CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]{7,63}$")
CONTAINER_PATH = re.compile(r"^(/[A-Za-z0-9_][A-Za-z0-9_.\-]*){2,}$")  # no "..", at least two levels


def _returned_key(cur, what: str) -> str:
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Could not clone the template {what}: source row missing or soft-deleted")
    return row["key"]


def plan_restore(touched: set[str], shared: set[str], snapshot_keys: set[str]) -> tuple[set[str], set[str], set[str]]:
    """Split rows touched by synthetic cameras into (restore, delete, manual_review)."""
    manual = touched & shared
    safe = touched - shared
    return safe & snapshot_keys, safe - snapshot_keys, manual


class Registrar:
    def __init__(self, app: AppConfig, worker_names: list[str], registry: Registry,
                 worker_settings: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self.app = app
        self.worker_names = worker_names
        self.registry = registry
        # Validated scenario.worker_settings: {worker: {column: value}}, applied in create().
        self.worker_settings = worker_settings or {}
        self.clients: dict[str, WorkerClient] = {}
        self._template: dict[str, Any] = {}
        self._settings_rows: dict[str, str] = {}
        self._waker: Optional[TopicWaker] = None

    # ------------------------------------------------------------- wake messages

    def set_wake_frame(self, frame: Optional[dict]) -> None:
        """Remember a source frame so syncs can keep the topic busy (see kafka_io.TopicWaker)."""
        path = (frame or {}).get("image_path")
        if path and path != self.registry.wake_image_path:
            self.registry.wake_image_path = path
            self.registry.save()

    def _wake_message(self, now: float) -> Optional[dict]:
        if not self.registry.wake_image_path:
            return None
        run_id = self.registry.run_id
        message = build_message({"image_path": self.registry.wake_image_path}, f"loadgen-wake-{run_id}", now,
                                run_id, self.app.frames, passthrough_dates=False)
        message["wake"] = True
        return message

    def _sync(self, name: str, client: WorkerClient) -> WorkerStatus:
        if self._waker is None:
            self._waker = TopicWaker(self.app.kafka, self._wake_message)
        with self._waker.during(f"{name} sync"):
            return client.sync()

    def close_waker(self) -> None:
        if self._waker is not None:
            self._waker.close()
            self._waker = None

    # ------------------------------------------------------------------ setup

    def resolve(self) -> dict[str, Deployment]:
        deployments: dict[str, Deployment] = {}
        conn = db.connect(self.app.main_db_url())
        try:
            with conn.cursor() as cur:
                for name in self.worker_names:
                    deployment = resolve_deployment(cur, name, self.app.workers[name])
                    deployments[name] = deployment
                    self.registry.workers[name] = asdict(deployment)
                    self.clients[name] = self._client(name, deployment.base_url)
        finally:
            conn.close()
        self.registry.save()
        return deployments

    def check_settings_rows(self) -> dict[str, str]:
        rows: dict[str, str] = {}
        for name in self.worker_names:
            target = self.app.workers[name].settings
            conn = db.connect(self.app.worker_db_url(name))
            try:
                with conn.cursor() as cur:
                    key = self._enabled_settings_key(cur, target)
            finally:
                conn.close()
            if key is None:
                raise ConfigError(
                    f"Worker '{name}' has no enabled row in {target.table}. Enable the module first; "
                    "the load generator never enables modules itself."
                )
            rows[name] = key
        return rows

    def pick_template(self, max_age_s: float = 60.0) -> str:
        """The real camera the first target worker processed most recently."""
        name = self.worker_names[0]
        client = self.clients[name]
        status = client.status()
        now = time.time() + client.clock.seconds  # real cameras are stamped in the worker's time
        live = {camera_id: at for camera_id, at in status.processed_timestamps.items()
                if camera_id in status.active_cameras and now - at <= max_age_s}
        if not live:
            raise ConfigError(f"{name} has no live camera to copy; pass --template-camera <camera id>")
        camera_id = max(live, key=live.get)
        log.info("Template camera: %s (freshest live camera on %s)", camera_id, name)
        return camera_id

    def template_camera(self, camera_id: str) -> dict[str, Any]:
        conn = db.connect(self.app.main_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id, name, region_id::text AS region_id FROM cameras "
                    "WHERE id::text = %s AND deleted_at IS NULL",
                    (camera_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            raise ConfigError(f"Template camera {camera_id} not found in vision-main")
        if not row["region_id"]:
            raise ConfigError(f"Template camera {camera_id} has no region")
        return dict(row)

    def setup(self, camera_count: int, template_camera_id: str) -> list[str]:
        """prepare + create + activate(all): every synthetic camera enabled, workers synced."""
        self.prepare(template_camera_id)
        self.create(camera_count)
        self.activate(camera_count)
        return list(self.registry.camera_ids)

    def prepare(self, template_camera_id: str = "") -> dict[str, Deployment]:
        """Read-only checks: worker deployments, template camera, enabled settings rows."""
        deployments = self.resolve()
        template = self.template_camera(template_camera_id or self.pick_template())
        self._settings_rows = self.check_settings_rows()
        self.registry.template_camera_id = template["id"]
        self._template = template
        self.registry.save()
        return deployments

    def create(self, camera_count: int) -> list[str]:
        """Region, assignments, cameras, per-camera rows and snapshots. Nothing is enabled yet."""
        deployments = {name: Deployment(**self.registry.workers[name]) for name in self.worker_names}
        main = db.connect(self.app.main_db_url())
        try:
            self._create_region(main, self._template["region_id"])
            if any(self._worker(name).copy_department_links for name in self.worker_names):
                self._copy_department_links(main, self._template["region_id"])
            self._assign_region(main, deployments)
            self._create_cameras(main, self._template["id"], camera_count)
        finally:
            main.close()

        self._clone_camera_rows(self._template["id"])
        self._take_snapshots()
        self._record_settings_edits(self._settings_rows)
        self._apply_overrides(self._settings_rows)
        return list(self.registry.camera_ids)

    def activate(self, count: int) -> dict[str, WorkerStatus]:
        """Enable exactly the first `count` synthetic cameras in every module, then sync.

        A sync restarts the worker's pipeline and GPU engine, which then sizes its model copies
        for its real cameras plus these `count` cameras, as it would in production.
        """
        ids = list(self.registry.camera_ids)
        if not 0 <= count <= len(ids):
            raise ValueError(f"Cannot enable {count} of {len(ids)} synthetic cameras")
        for edit in self.registry.settings:
            self._edit_settings(edit.worker, edit.table, edit.key_column, edit.key, edit.columns,
                                add=True, ids=ids, add_ids=ids[:count])
        self.registry.active_count = count
        self.registry.save()
        return self._sync_and_verify(set(ids[:count]))

    def _create_region(self, conn, template_region_id: str) -> None:
        with conn.cursor() as cur:
            columns = db.table_columns(cur, REGIONS)
            names = db.column_names(columns)
            overrides: dict[str, Any] = {
                **db.audit_overrides(columns),
                **db.new_key_override(columns, "id"),
                "name": f"{self.app.registration.name_prefix}-{self.registry.run_id}",
            }
            if "description" in names:
                overrides["description"] = f"Load generator run {self.registry.run_id}"
            sql, params = db.build_clone_sql(REGIONS, columns, "id", "id", template_region_id, overrides)
            cur.execute(sql, params)
            region_id = _returned_key(cur, "region")
        conn.commit()
        self.registry.region_id = region_id
        self.registry.save()
        log.info("Created region %s", region_id)

    def _copy_department_links(self, conn, template_region_id: str) -> None:
        """Link the synthetic region to the template region's departments (AI attendance scopes
        recognition to employees of those departments). Teardown removes them by region id."""
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS present", (DEPARTMENT_LINKS,))
            if not cur.fetchone()["present"]:
                log.warning("No %s table in vision-main; department links not copied", DEPARTMENT_LINKS)
                return
            extra = ", created_by" if "created_by" in db.column_names(db.table_columns(cur, DEPARTMENT_LINKS)) else ""
            cur.execute(
                f"INSERT INTO {DEPARTMENT_LINKS} (department_id, camera_region_id{extra}) "
                f"SELECT department_id, %s::uuid{extra} FROM {DEPARTMENT_LINKS} "
                "WHERE camera_region_id = %s::uuid ON CONFLICT DO NOTHING",
                (self.registry.region_id, template_region_id),
            )
            copied = cur.rowcount
        conn.commit()
        if copied:
            log.info("Linked the synthetic region to %d departments of the template region", copied)
        else:
            log.warning("Template camera's region has no departments; attendance will not recognise anyone "
                        "on the synthetic cameras")

    @staticmethod
    def _remove_department_links(url: str, region_id: str) -> str:
        conn = db.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL AS present", (DEPARTMENT_LINKS,))
                if not cur.fetchone()["present"]:
                    return "skipped: no table"
                cur.execute(f"DELETE FROM {DEPARTMENT_LINKS} WHERE camera_region_id = %s::uuid", (region_id,))
                deleted = cur.rowcount
            conn.commit()
            return f"{deleted} deleted"
        finally:
            conn.close()

    def _assign_region(self, conn, deployments: dict[str, Deployment]) -> None:
        for name, deployment in deployments.items():
            with conn.cursor() as cur:
                columns = db.table_columns(cur, ASSIGNMENTS)
                names = db.column_names(columns)
                cur.execute(
                    f"SELECT id::text AS id FROM {ASSIGNMENTS} WHERE server_function_id::text = %s "
                    "AND deleted_at IS NULL LIMIT 1",
                    (deployment.server_function_id,),
                )
                existing = cur.fetchone()
                status = {"status": "assigned"} if "status" in names else {}
                if existing:
                    overrides = {
                        **db.audit_overrides(columns),
                        **db.new_key_override(columns, "id"),
                        "camera_region_id": self.registry.region_id,
                        **status,
                    }
                    sql, params = db.build_clone_sql(ASSIGNMENTS, columns, "id", "id", existing["id"], overrides)
                    cur.execute(sql, params)
                else:
                    extra_columns = ", status" if status else ""
                    extra_values = ", 'assigned'" if status else ""
                    cur.execute(
                        f"INSERT INTO {ASSIGNMENTS} (server_function_id, camera_region_id{extra_columns}) "
                        f"VALUES (%s::uuid, %s::uuid{extra_values}) RETURNING id::text AS key",
                        (deployment.server_function_id, self.registry.region_id),
                    )
                key = _returned_key(cur, "region assignment")
            conn.commit()
            self.registry.assignments.append(Assignment(worker=name, table=ASSIGNMENTS, key=key))
            self.registry.save()
            log.info("Assigned region to %s (server_function %s)", name, deployment.server_function_id)

    def _create_cameras(self, conn, template_id: str, count: int) -> None:
        reg = self.app.registration
        with conn.cursor() as cur:
            columns = db.table_columns(cur, CAMERAS)
            names = db.column_names(columns)
            base: dict[str, Any] = {
                **db.scrub_stream_overrides(columns),
                **db.audit_overrides(columns),
                **db.new_key_override(columns, "id"),
                "region_id": self.registry.region_id,
            }
            optional = {
                "ip": reg.camera_ip,
                "is_active": True,
                "description": f"Load generator run {self.registry.run_id}",
                "user": "",
                "password": "",
            }
            base.update({key: value for key, value in optional.items() if key in names})
            keys: list[str] = []
            for index in range(count):
                overrides = {**base, "name": f"{reg.name_prefix}-{self.registry.run_id}-{index:03d}"}
                if "port" in names:
                    overrides["port"] = reg.base_port + index
                sql, params = db.build_clone_sql(CAMERAS, columns, "id", "id", template_id, overrides)
                cur.execute(sql, params)
                keys.append(_returned_key(cur, "camera"))
        conn.commit()
        self.registry.camera_ids = keys
        self.registry.save()
        log.info("Created %d synthetic cameras", len(keys))

    def _clone_camera_rows(self, template_id: str) -> None:
        for name in self.worker_names:
            for target in self.app.workers[name].camera_rows:
                self.registry.camera_rows.append(CameraRows(worker=name, table=target.table, camera_column=target.camera_column))
                self.registry.save()
                conn = db.connect(self.app.worker_db_url(name))
                try:
                    with conn.cursor() as cur:
                        columns = db.table_columns(cur, target.table)
                        cloned = 0
                        for camera_id in self.registry.camera_ids:
                            overrides = {
                                **db.audit_overrides(columns),
                                **db.new_key_override(columns, target.key_column),
                                target.camera_column: camera_id,
                            }
                            sql, params = db.build_clone_sql(
                                target.table, columns, target.key_column, target.camera_column, template_id, overrides
                            )
                            cur.execute(sql, params)
                            cloned += cur.rowcount
                    conn.commit()
                finally:
                    conn.close()
                if cloned == 0:
                    log.warning("Template camera has no rows in %s; %s cameras run without them", target.table, name)

    def _take_snapshots(self) -> None:
        for name in self.worker_names:
            target = self.app.workers[name].restore
            if target is None:
                continue
            conn = db.connect(self.app.worker_db_url(name))
            try:
                with conn.cursor() as cur:
                    taken_at = time.time()
                    cur.execute(f"SELECT * FROM {db.quote_table(target.table)} WHERE {target.snapshot_where}")
                    rows = [{key: db.to_text(value) for key, value in row.items()} for row in cur.fetchall()]
            finally:
                conn.close()
            directory = self.registry.directory / self.app.registration.snapshot_dir_name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{name}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"table": target.table, "taken_at": taken_at, "rows": rows}, handle)
            self.registry.snapshots.append(Snapshot(worker=name, path=str(path)))
            self.registry.save()
            log.info("Snapshot of %d %s rows for %s", len(rows), target.table, name)

    def _record_settings_edits(self, settings_rows: dict[str, str]) -> None:
        """Written before any edit so teardown removes the ids even if a run dies mid-edit."""
        for name, key in settings_rows.items():
            target = self.app.workers[name].settings
            self.registry.settings.append(
                SettingsEdit(worker=name, table=target.table, key_column=target.key_column, key=key, columns=target.camera_columns)
            )
        self.registry.save()

    def _apply_overrides(self, settings_rows: dict[str, str]) -> None:
        """Set scenario worker_settings on each worker's enabled settings row. The original value is
        recorded (and saved) before the update so teardown restores it even if the run dies here.
        The worker picks the new value up on the sync that activate() runs next."""
        for name, columns in self.worker_settings.items():
            if name not in settings_rows:
                continue
            target = self.app.workers[name].settings
            key = settings_rows[name]
            conn = db.connect(self.app.worker_db_url(name))
            try:
                with conn.cursor() as cur:
                    for column, value in columns.items():
                        original = self._read_setting(cur, target.table, target.key_column, key, column)
                        self.registry.overrides.append(SettingsOverride(
                            worker=name, table=target.table, key_column=target.key_column, key=key,
                            column=column, original=original, value=value,
                        ))
                        self.registry.save()
                        self._write_setting(cur, target.table, target.key_column, key, column, value)
                        conn.commit()
                        log.info("%s %s.%s: %s -> %s for this run", name, target.table, column, original, value)
            finally:
                conn.close()

    def _restore_override(self, override: SettingsOverride) -> str:
        conn = db.connect(self.app.worker_db_url(override.worker))
        try:
            with conn.cursor() as cur:
                self._write_setting(cur, override.table, override.key_column, override.key,
                                    override.column, override.original)
            conn.commit()
        finally:
            conn.close()
        return f"restored {override.original}"

    @staticmethod
    def _read_setting(cur, table: str, key_column: str, key: str, column: str) -> Any:
        cur.execute(
            f"SELECT {db.quote_ident(column)} AS value FROM {db.quote_table(table)} "
            f"WHERE {db.quote_ident(key_column)}::text = %s",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"{table} row {key_column}={key} disappeared before {column} could be set")
        value = row["value"]
        # Tunable columns are int/float/bool; keep those typed so the registry JSON restores them as-is.
        return value if value is None or isinstance(value, (bool, int, float, str)) else db.to_text(value)

    @staticmethod
    def _write_setting(cur, table: str, key_column: str, key: str, column: str, value: Any) -> None:
        cur.execute(
            f"UPDATE {db.quote_table(table)} SET {db.quote_ident(column)} = %s "
            f"WHERE {db.quote_ident(key_column)}::text = %s",
            (value, key),
        )

    def _sync_and_verify(self, expected: set[str]) -> dict[str, WorkerStatus]:
        statuses: dict[str, WorkerStatus] = {}
        for name, client in self.clients.items():
            try:
                self._sync(name, client)
            except Exception as exc:
                raise RuntimeError(
                    f"{name} failed to sync with {len(expected)} synthetic cameras ({exc}). If its log says "
                    "'Insufficient GPU VRAM', lower the camera count or set gpu_free_vram_mb so the run is capped."
                ) from exc
            if not client.lists_enabled_cameras:
                # This worker lists a camera only after processing its frames; the run's first-frame
                # and staleness checks catch synthetic cameras it never picks up.
                statuses[name] = client.status()
                log.info("%s synced; its %d synthetic cameras are confirmed once frames flow", name, len(expected))
                continue
            deadline = time.monotonic() + self.app.registration.sync_timeout_s
            while True:
                status = client.status()
                missing = expected - status.active_cameras
                if (not missing and (status.running or not expected)) or time.monotonic() >= deadline:
                    break
                time.sleep(2)
            if missing:
                raise RuntimeError(
                    f"{name} did not activate {len(missing)} of {len(expected)} synthetic cameras. Check that the "
                    "region assignment belongs to the worker's server (PUBLIC_BASE_IP) and the function is activated."
                )
            statuses[name] = status
            log.info("%s is processing %d synthetic cameras (%d cameras in total)",
                     name, len(expected), len(status.active_cameras))
        return statuses

    def real_camera_counts(self) -> dict[str, int]:
        """Cameras each worker runs besides this run's synthetic ones."""
        synthetic = set(self.registry.camera_ids)
        return {name: len(client.status().active_cameras - synthetic) for name, client in self.clients.items()}

    # --------------------------------------------------------------- teardown

    def teardown(self, keep_events: bool) -> dict[str, Any]:
        report: dict[str, Any] = {"errors": [], "steps": {}, "manual_review": {}, "restored": {}}
        ids = list(self.registry.camera_ids)

        def step(label: str, action: Callable[[], Any]) -> Any:
            try:
                result = action()
                report["steps"][label] = result if result is not None else "ok"
                return result
            except Exception as exc:
                log.exception("Teardown step failed: %s", label)
                report["errors"].append(f"{label}: {exc}")
                return None

        self.registry.set_status("tearing_down")
        for edit in self.registry.settings:
            step(f"settings:{edit.worker}", lambda edit=edit: self._edit_settings(
                edit.worker, edit.table, edit.key_column, edit.key, edit.columns, add=False, ids=ids))
        # Before the syncs below, so workers reload the original values.
        for override in self.registry.overrides:
            step(f"override:{override.worker}.{override.column}",
                 lambda override=override: self._restore_override(override))
        for name in self.registry.workers:
            step(f"sync:{name}", lambda name=name: self._sync(name, self._client_for(name)) and "synced")
        self.close_waker()
        for rows in self.registry.camera_rows:
            step(f"camera_rows:{rows.table}", lambda rows=rows: self._delete_rows(
                self.app.worker_db_url(rows.worker), rows.table, rows.camera_column, ids))

        restore_plans = {
            snapshot.worker: step(f"restore_plan:{snapshot.worker}", lambda snapshot=snapshot: self._plan_restore(snapshot, ids))
            for snapshot in self.registry.snapshots
        }

        if keep_events:
            report["steps"]["purge"] = "skipped (--keep-events)"
        else:
            for name in self.worker_names:
                for target in self._worker(name).purge:
                    url = self.app.main_db_url() if target.db == "main" else self.app.worker_db_url(name)
                    step(f"purge:{target.table}",
                         lambda name=name, url=url, target=target: self._purge_events(name, url, target, ids))
                step(f"event_folders:{name}", lambda name=name: self._sweep_event_folders(name, ids))

        for worker, plan in restore_plans.items():
            if plan is None:
                continue
            report["manual_review"][worker] = sorted(plan["manual"])
            report["restored"][worker] = step(f"restore:{worker}", lambda worker=worker, plan=plan: self._apply_restore(worker, plan))

        main_url = self.app.main_db_url()
        for assignment in self.registry.assignments:
            step(f"unassign:{assignment.worker}", lambda assignment=assignment: self._remove_main(
                main_url, assignment.table, [assignment.key]))
        step("cameras", lambda: self._remove_main(main_url, CAMERAS, ids))
        if self.registry.region_id:
            step("department_links", lambda: self._remove_department_links(main_url, self.registry.region_id))
            step("region", lambda: self._remove_main(main_url, REGIONS, [self.registry.region_id]))

        self.registry.teardown = report
        self.registry.set_status("done" if not report["errors"] else "teardown_incomplete")
        return report

    def _plan_restore(self, snapshot: Snapshot, ids: list[str]) -> dict[str, Any]:
        target = self._worker(snapshot.worker).restore
        with open(snapshot.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = {str(row[target.key_column]): row for row in data["rows"]}
        conn = db.connect(self.app.worker_db_url(snapshot.worker))
        try:
            with conn.cursor() as cur:
                link = db.quote_ident(target.events_link_column)
                camera = db.quote_ident(target.events_camera_column)
                event_time = db.quote_ident(target.events_time_column)
                events = db.quote_table(target.events_table)
                cur.execute(
                    f"SELECT DISTINCT {link}::text AS key FROM {events} "
                    f"WHERE {camera}::text = ANY(%s) AND {link} IS NOT NULL",
                    (ids,),
                )
                touched = {row["key"] for row in cur.fetchall()}
                cur.execute(
                    f"SELECT DISTINCT {link}::text AS key FROM {events} "
                    f"WHERE {link}::text = ANY(%s) AND NOT ({camera}::text = ANY(%s)) "
                    f"AND {event_time} >= to_timestamp(%s)",
                    (sorted(touched), ids, data["taken_at"]),
                )
                shared = {row["key"] for row in cur.fetchall()}
        finally:
            conn.close()
        restore, delete, manual = plan_restore(touched, shared, set(rows))
        return {"restore": [rows[key] for key in sorted(restore)], "delete": sorted(delete), "manual": manual}

    def _apply_restore(self, worker: str, plan: dict[str, Any]) -> dict[str, int]:
        target = self._worker(worker).restore
        conn = db.connect(self.app.worker_db_url(worker))
        try:
            with conn.cursor() as cur:
                columns = db.table_columns(cur, target.table)
                for row in plan["restore"]:
                    sql, params = db.build_restore_sql(target.table, columns, target.key_column, row)
                    cur.execute(sql, params)
                deleted = db.delete_by_values(cur, target.table, target.key_column, plan["delete"])
            conn.commit()
        finally:
            conn.close()
        return {"restored": len(plan["restore"]), "deleted": deleted}

    # ---------------------------------------------------------------- helpers

    def _edit_settings(self, worker: str, table: str, key_column: str, key: str, columns: list[str],
                       add: bool, ids: list[str] | None = None, add_ids: list[str] | None = None) -> str:
        """Remove every synthetic id from the camera columns, then append `add_ids` (all ids by default)."""
        ids = list(ids if ids is not None else self.registry.camera_ids)
        to_add = (list(add_ids) if add_ids is not None else ids) if add else []
        conn = db.connect(self.app.worker_db_url(worker))
        try:
            with conn.cursor() as cur:
                types = {column.name: column.type for column in db.table_columns(cur, table)}
                for column in columns:
                    sql, params = db.build_json_array_edit(
                        table, column, types[column], key_column, key, remove_ids=ids, add_ids=to_add
                    )
                    cur.execute(sql, params)
            conn.commit()
        finally:
            conn.close()
        return "added" if add else "removed"

    @staticmethod
    def _enabled_settings_key(cur, target) -> str | None:
        cur.execute(
            f"SELECT {db.quote_ident(target.key_column)}::text AS key FROM {db.quote_table(target.table)} "
            f"WHERE {target.enabled_where} ORDER BY {target.order_by} LIMIT 1"
        )
        row = cur.fetchone()
        return row["key"] if row else None

    def _purge_events(self, worker: str, url: str, target: PurgeTarget, ids: list[str]) -> dict[str, Any]:
        """Delete event rows, then the image/video files they point at (only inside EVENTS_DIR).

        The paths are saved in the registry before the rows go, so a retried cleanup still has them.
        """
        key = f"{worker}:{target.table}"
        conn = db.connect(url)
        try:
            with conn.cursor() as cur:
                paths = db.select_values(cur, target.table, target.camera_column, ids, target.file_columns)
                if paths:
                    saved = self.registry.event_files.get(key, [])
                    known = set(saved)
                    self.registry.event_files[key] = saved + [path for path in paths if path not in known]
                    self.registry.save()
                deleted = db.delete_by_values(cur, target.table, target.camera_column, ids)
            conn.commit()
        finally:
            conn.close()
        result: dict[str, Any] = {"rows_deleted": deleted}
        if target.file_columns:
            result.update(self._delete_event_files(worker, self.registry.event_files.get(key, [])))
        return result

    def _delete_event_files(self, worker: str, paths: list[str]) -> dict[str, Any]:
        if self._events_dir_available():
            return media.delete_files(self.app.events.dir, paths)
        container = self._worker(worker).container
        if not (self.app.worker_logs.ssh_target and container):
            return {"files": f"skipped: EVENTS_DIR not available here ({len(paths)} paths)"}
        root = self.app.events.container_dir
        resolved = [media.container_event_path(root, path) for path in paths]
        targets = sorted({path for path in resolved if path})
        cfg = self.app.worker_logs
        for start in range(0, len(targets), 200):
            remote = [*dockerhost.docker_command(cfg), "exec", dockerhost.check_container(container),
                      "rm", "-f", "--", *targets[start:start + 200]]
            result = dockerhost.run(cfg, remote)
            if result.returncode != 0:
                output = ((result.stdout or b"") + (result.stderr or b"")).decode("utf-8", errors="replace")
                raise RuntimeError(f"docker exec {container} rm failed ({result.returncode}): {output.strip()[-300:]}")
        # rm -f does not say which files existed, so this counts the paths removed or already gone.
        return {"files_removed_in_container": len(targets), "paths_outside_events_dir": len(paths) - len(
            [path for path in resolved if path])}

    def _sweep_event_folders(self, worker: str, ids: list[str]) -> str:
        if self._events_dir_available():
            folder = Path(self.app.events.dir) / self._worker(worker).events_folder
            return f"{media.sweep_camera_folders(str(folder), ids)} camera folders removed from {folder}"
        container = self._worker(worker).container
        if self.app.worker_logs.ssh_target and container:
            return self._sweep_event_folders_in_container(container, self._worker(worker).events_folder, ids)
        return "skipped: EVENTS_DIR not available here (set LOADGEN_WORKER_LOGS_SSH to clean them on the Docker host)"

    def _sweep_event_folders_in_container(self, container: str, events_folder: str, ids: list[str]) -> str:
        """Remove the synthetic cameras' folders (workers keep event media under .../<camera_id>/...)."""
        if not ids:
            return "0 camera folders removed (no cameras)"
        root = f"{self.app.events.container_dir.rstrip('/')}/{events_folder}"
        if not CONTAINER_PATH.match(root) or not all(CAMERA_ID.match(camera_id) for camera_id in ids):
            raise ValueError(f"Refusing to sweep {root!r}: unexpected folder or camera id")
        cfg = self.app.worker_logs
        removed = 0
        for start in range(0, len(ids), 200):
            names: list[str] = []
            for camera_id in ids[start:start + 200]:
                names += ["-o", "-name", camera_id] if names else ["-name", camera_id]
            remote = [*dockerhost.docker_command(cfg), "exec", dockerhost.check_container(container),
                      "find", root, "-mindepth", "1", "-maxdepth", str(media.SWEEP_DEPTH), "-type", "d",
                      "(", *names, ")", "-prune", "-print", "-exec", "rm", "-rf", "{}", "+"]
            result = dockerhost.run(cfg, remote)
            output = (result.stdout or b"").decode("utf-8", errors="replace")
            errors = (result.stderr or b"").decode("utf-8", errors="replace")
            if result.returncode != 0:
                if "No such file or directory" in output + errors and root in output + errors:
                    return f"0 camera folders removed ({root} does not exist in {container})"
                raise RuntimeError(f"docker exec {container} find failed ({result.returncode}): {(output + errors).strip()[-300:]}")
            removed += sum(1 for line in output.splitlines() if line.startswith(root + "/"))
        return f"{removed} camera folders removed from {container}:{root}"

    def _events_dir_available(self) -> bool:
        return bool(self.app.events.dir) and Path(self.app.events.dir).is_dir()

    @staticmethod
    def _delete_rows(url: str, table: str, column: str, ids: list[str]) -> int:
        conn = db.connect(url)
        try:
            with conn.cursor() as cur:
                deleted = db.delete_by_values(cur, table, column, ids)
            conn.commit()
            return deleted
        finally:
            conn.close()

    @staticmethod
    def _remove_main(url: str, table: str, keys: list[str]) -> str:
        conn = db.connect(url)
        try:
            return db.delete_or_soft_delete(conn, table, "id", keys)
        finally:
            conn.close()

    def _worker(self, name: str):
        if name not in self.app.workers:
            raise ConfigError(f"Worker '{name}' from the registry is not configured (presets or --config)")
        return self.app.workers[name]

    def _client(self, name: str, base_url: str) -> WorkerClient:
        server_function_id = (self.registry.workers.get(name) or {}).get("server_function_id", "")
        return WorkerClient.for_worker(name, self._worker(name), base_url, server_function_id, self.app.auth,
                                       self.app.registration.sync_timeout_s)

    def _client_for(self, name: str) -> WorkerClient:
        if name not in self.clients:
            self.clients[name] = self._client(name, self.registry.workers[name]["base_url"])
        return self.clients[name]


def find_orphans(app: AppConfig) -> dict[str, dict[str, Any]]:
    """Synthetic cameras/regions in vision-main grouped by run id (by the name prefix)."""
    prefix = f"{app.registration.name_prefix}-"
    runs: dict[str, dict[str, Any]] = {}
    conn = db.connect(app.main_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id::text AS id, name FROM {REGIONS} WHERE name LIKE %s AND deleted_at IS NULL",
                (prefix + "%",),
            )
            for row in cur.fetchall():
                run_id = row["name"][len(prefix):]
                runs.setdefault(run_id, {"region_id": "", "camera_ids": [], "assignments": []})["region_id"] = row["id"]
            cur.execute(
                f"SELECT id::text AS id, name FROM {CAMERAS} WHERE name LIKE %s AND deleted_at IS NULL",
                (prefix + "%",),
            )
            for row in cur.fetchall():
                run_id = row["name"][len(prefix):].rsplit("-", 1)[0]
                runs.setdefault(run_id, {"region_id": "", "camera_ids": [], "assignments": []})["camera_ids"].append(row["id"])
            for run in runs.values():
                if run["region_id"]:
                    cur.execute(
                        f"SELECT id::text AS id FROM {ASSIGNMENTS} WHERE camera_region_id::text = %s AND deleted_at IS NULL",
                        (run["region_id"],),
                    )
                    run["assignments"] = [row["id"] for row in cur.fetchall()]
    finally:
        conn.close()
    return runs


def registry_for_orphan(app: AppConfig, results_dir: str, run_id: str, found: dict[str, Any]) -> Registry:
    """Rebuild a registry for a run whose registry file is missing, covering every configured worker."""
    directory = Path(results_dir) / f"orphan-{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    registry = Registry(run_id=run_id, environment=app.environment, path=str(directory / "registry.json"))
    registry.region_id = found["region_id"]
    registry.camera_ids = found["camera_ids"]
    registry.assignments = [Assignment(worker="unknown", table=ASSIGNMENTS, key=key) for key in found["assignments"]]
    skipped = sorted(set(app.workers) - set(app.configured_workers()))
    if skipped:
        log.warning("No database configured for %s; their settings and events are not cleaned", ", ".join(skipped))
    for name in app.configured_workers():
        worker = app.workers[name]
        registry.camera_rows.extend(
            CameraRows(worker=name, table=target.table, camera_column=target.camera_column) for target in worker.camera_rows
        )
        conn = db.connect(app.worker_db_url(name))
        try:
            with conn.cursor() as cur:
                for column in worker.settings.camera_columns:
                    cur.execute(
                        f"SELECT {db.quote_ident(worker.settings.key_column)}::text AS key "
                        f"FROM {db.quote_table(worker.settings.table)} WHERE EXISTS ("
                        f"SELECT 1 FROM jsonb_array_elements_text(COALESCE({db.quote_ident(column)}::jsonb, '[]'::jsonb)) AS e "
                        "WHERE e = ANY(%s))",
                        (found["camera_ids"],),
                    )
                    for row in cur.fetchall():
                        registry.settings.append(SettingsEdit(
                            worker=name, table=worker.settings.table, key_column=worker.settings.key_column,
                            key=row["key"], columns=[column]))
        finally:
            conn.close()
        try:
            main = db.connect(app.main_db_url())
            try:
                with main.cursor() as cur:
                    registry.workers[name] = asdict(resolve_deployment(cur, name, worker))
            finally:
                main.close()
        except ConfigError as exc:
            log.warning("Skipping sync for %s during orphan cleanup: %s", name, exc)
    registry.save()
    return registry
