"""Removes the image/video files that synthetic cameras' events wrote under EVENTS_DIR."""

from __future__ import annotations

import logging
import os
import posixpath
import shutil
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# Workers write EVENTS_DIR/<function>/<kind>/<camera_id>/<date>/<file>; camera folders sit at depth 2.
SWEEP_DEPTH = 3


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError:
        return False


def resolve_event_path(events_dir: Path, value: str) -> Path | None:
    """An event row's file path (absolute, or relative to EVENTS_DIR) if it stays inside EVENTS_DIR."""
    if not value:
        return None
    path = Path(value)
    candidate = (path if path.is_absolute() else events_dir / path).resolve()
    return candidate if _inside(events_dir, candidate) and candidate != events_dir else None


def container_event_path(events_dir: str, value: str) -> str | None:
    """resolve_event_path for a POSIX EVENTS_DIR inside a worker container (not this machine)."""
    if not value or "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    root = posixpath.normpath(events_dir)
    if any(part == ".." for part in value.split("/")):
        return None
    candidate = posixpath.normpath(value if value.startswith("/") else posixpath.join(root, value))
    return candidate if candidate.startswith(root.rstrip("/") + "/") else None


def delete_files(events_dir: str, paths: Iterable[str]) -> dict[str, int]:
    root = Path(events_dir).resolve()
    deleted = outside = 0
    for value in paths:
        target = resolve_event_path(root, value)
        if target is None:
            outside += 1
            continue
        try:
            if target.is_file():
                target.unlink()
                deleted += 1
        except OSError as exc:
            log.warning("Could not delete %s: %s", target, exc)
    return {"files_deleted": deleted, "paths_outside_events_dir": outside}


def sweep_camera_folders(folder: str, camera_ids: Iterable[str], depth: int = SWEEP_DEPTH) -> int:
    """Delete directories named exactly like a synthetic camera id, up to `depth` levels down.

    Camera ids are UUIDs, so a folder with that name can only hold that camera's files. Only
    directory entries are listed, and never below `depth`, so large event trees stay cheap.
    """
    ids = set(camera_ids)
    root = Path(folder)
    if not ids or not root.is_dir():
        return 0
    removed = 0
    pending = [(root, 0)]
    while pending:
        directory, level = pending.pop()
        if level >= depth:
            continue
        try:
            entries = [entry for entry in os.scandir(directory) if entry.is_dir(follow_symlinks=False)]
        except OSError as exc:
            log.warning("Could not list %s: %s", directory, exc)
            continue
        for entry in entries:
            if entry.name in ids:
                shutil.rmtree(entry.path, ignore_errors=True)
                removed += 1
            else:
                pending.append((Path(entry.path), level + 1))
    return removed
