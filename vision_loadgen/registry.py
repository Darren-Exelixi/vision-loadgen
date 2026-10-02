"""Write-ahead record of everything a run creates, so teardown and `cleanup` can undo it."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_NAME = "registry.json"


@dataclass
class Assignment:
    worker: str
    table: str
    key: str


@dataclass
class SettingsEdit:
    worker: str
    table: str
    key_column: str
    key: str
    columns: list[str]


@dataclass
class CameraRows:
    worker: str
    table: str
    camera_column: str


@dataclass
class Snapshot:
    worker: str
    path: str


@dataclass
class Registry:
    run_id: str
    environment: str
    path: str
    status: str = "setting_up"
    created_at: float = field(default_factory=time.time)
    run_started_at: float = 0.0
    template_camera_id: str = ""
    region_id: str = ""
    camera_ids: list[str] = field(default_factory=list)
    active_count: int = 0
    assignments: list[Assignment] = field(default_factory=list)
    settings: list[SettingsEdit] = field(default_factory=list)
    camera_rows: list[CameraRows] = field(default_factory=list)
    snapshots: list[Snapshot] = field(default_factory=list)
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Event image/video paths read before their rows were purged ("<worker>:<table>" -> paths), so a
    # retried cleanup can still delete files the rows no longer point at.
    event_files: dict[str, list[str]] = field(default_factory=dict)
    # A source frame's image_path for the wake messages sent while workers sync; kept so a later
    # `cleanup --run-id` can send them too.
    wake_image_path: str = ""
    teardown: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, results_dir: str | Path, run_id: str, environment: str) -> "Registry":
        directory = Path(results_dir) / run_id
        directory.mkdir(parents=True, exist_ok=False)
        registry = cls(run_id=run_id, environment=environment, path=str(directory / REGISTRY_NAME))
        registry.save()
        return registry

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["path"] = str(path)
        data["assignments"] = [Assignment(**item) for item in data.get("assignments", [])]
        data["settings"] = [SettingsEdit(**item) for item in data.get("settings", [])]
        data["camera_rows"] = [CameraRows(**item) for item in data.get("camera_rows", [])]
        data["snapshots"] = [Snapshot(**item) for item in data.get("snapshots", [])]
        return cls(**data)

    @property
    def directory(self) -> Path:
        return Path(self.path).parent

    def save(self) -> None:
        target = Path(self.path)
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, default=str)
        os.replace(temporary, target)

    def set_status(self, status: str) -> None:
        self.status = status
        self.save()
