"""A persistent camera row standing for a video's perspective, used as a run's template camera.

It is cloned from an existing camera (region, timezone, ...) with its stream scrubbed and
`is_active` off, so nothing tries to open it. It is not enabled in any module: each run clones it
into synthetic cameras and enables those. Its name prefix keeps `cleanup --orphans` away from it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from vision_loadgen import db
from vision_loadgen.config import AppConfig, ConfigError

log = logging.getLogger(__name__)

CAMERAS = "cameras"
PREFIX = "loadgen-video-"
PORT = 19999  # outside the synthetic cameras' base_port range


def camera_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ConfigError(f"Invalid video camera name '{name}'")
    return slug if slug.startswith(PREFIX) else PREFIX + slug


def clone_overrides(columns: list[db.Column], name: str, description: str, ip: str) -> dict[str, Any]:
    names = db.column_names(columns)
    overrides: dict[str, Any] = {
        **db.scrub_stream_overrides(columns),
        **db.audit_overrides(columns),
        **db.new_key_override(columns, "id"),
        "name": name,
    }
    optional = {"ip": ip, "is_active": False, "description": description, "user": "", "password": "", "port": PORT}
    overrides.update({key: value for key, value in optional.items() if key in names})
    return overrides


def _live(columns: list[db.Column]) -> str:
    return " AND deleted_at IS NULL" if "deleted_at" in db.column_names(columns) else ""


def add_video_camera(app: AppConfig, name: str, like_id: str, workers: list[str],
                     zones_from: Optional[str] = None, corpus: str = "") -> dict[str, Any]:
    full_name = camera_name(name)
    description = "vision-loadgen video camera" + (f" (corpus {corpus})" if corpus else "")
    conn = db.connect(app.main_db_url())
    try:
        with conn.cursor() as cur:
            columns = db.table_columns(cur, CAMERAS)
            cur.execute(f"SELECT id FROM cameras WHERE name = %s{_live(columns)}", (full_name,))
            if cur.fetchone():
                raise ConfigError(f"A camera named {full_name} already exists")
            sql, params = db.build_clone_sql(CAMERAS, columns, "id", "id", like_id,
                                             clone_overrides(columns, full_name, description, app.registration.camera_ip))
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                raise ConfigError(f"Camera {like_id} not found (or deleted); pass an existing camera to --like")
            camera_id = row["key"]
            cur.execute("SELECT region_id::text AS region_id FROM cameras WHERE id::text = %s", (camera_id,))
            region_id = cur.fetchone()["region_id"]
        conn.commit()
    finally:
        conn.close()

    cloned: dict[str, int] = {}
    if zones_from:
        for worker in workers:
            for target in app.workers[worker].camera_rows:
                cloned[f"{worker}:{target.table}"] = _clone_rows(app, worker, target, zones_from, camera_id)
    log.info("Created video camera %s (%s) in region %s", full_name, camera_id, region_id)
    return {"id": camera_id, "name": full_name, "region_id": region_id, "cloned_rows": cloned}


def _clone_rows(app: AppConfig, worker: str, target, source_id: str, camera_id: str) -> int:
    conn = db.connect(app.worker_db_url(worker))
    try:
        with conn.cursor() as cur:
            columns = db.table_columns(cur, target.table)
            overrides = {**db.audit_overrides(columns), **db.new_key_override(columns, target.key_column),
                         target.camera_column: camera_id}
            sql, params = db.build_clone_sql(target.table, columns, target.key_column, target.camera_column,
                                             source_id, overrides)
            cur.execute(sql, params)
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def list_video_cameras(app: AppConfig) -> list[dict[str, Any]]:
    conn = db.connect(app.main_db_url())
    try:
        with conn.cursor() as cur:
            columns = db.table_columns(cur, CAMERAS)
            description = "description" if "description" in db.column_names(columns) else "NULL"
            cur.execute(
                f"SELECT id::text AS id, name, region_id::text AS region_id, {description} AS description "
                f"FROM cameras WHERE name LIKE %s{_live(columns)} ORDER BY name",
                (PREFIX + "%",),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def remove_video_camera(app: AppConfig, camera_id: str) -> dict[str, Any]:
    conn = db.connect(app.main_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM cameras WHERE id::text = %s", (camera_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfigError(f"Camera {camera_id} not found")
    if not str(row["name"]).startswith(PREFIX):
        raise ConfigError(f"Camera {camera_id} ('{row['name']}') is not a video camera; refusing to remove it")

    report: dict[str, Any] = {"id": camera_id, "name": row["name"], "rows": {}, "errors": []}
    for worker in app.configured_workers():
        for target in app.workers[worker].camera_rows:
            try:
                conn = db.connect(app.worker_db_url(worker))
                try:
                    with conn.cursor() as cur:
                        report["rows"][f"{worker}:{target.table}"] = db.delete_by_values(
                            cur, target.table, target.camera_column, [camera_id])
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:
                report["errors"].append(f"{worker}:{target.table}: {exc}")
    conn = db.connect(app.main_db_url())
    try:
        report["camera"] = db.delete_or_soft_delete(conn, CAMERAS, "id", [camera_id])
    finally:
        conn.close()
    return report
