from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from vision_loadgen.config import FramesConfig, KafkaConfig
from vision_loadgen.frames import is_synthetic
from vision_loadgen.kafka_io import TopicTap

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"


class FrameSource(Protocol):
    passthrough_dates: bool

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def wait_for_first_frame(self, timeout_s: float) -> bool: ...

    def frame_for(self, camera_index: int) -> Optional[dict]: ...


def _accept(payload: dict, synthetic_ids: set[str], camera_filter: set[str]) -> bool:
    if is_synthetic(payload, synthetic_ids) or not payload.get("image_path"):
        return False
    return not camera_filter or str(payload.get("camera_id")) in camera_filter


class LiveTapSource:
    """Latest real frame per source camera; synthetic camera i uses source camera i mod n."""

    passthrough_dates = True

    def __init__(self, kafka: KafkaConfig, source_camera_ids: list[str], synthetic_ids: set[str]) -> None:
        self._filter = {str(camera_id) for camera_id in source_camera_ids}
        self._synthetic_ids = synthetic_ids
        self._latest: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._tap = TopicTap(kafka, self._on_message)

    def start(self) -> None:
        self._tap.start()

    def stop(self) -> None:
        self._tap.stop()

    def wait_for_first_frame(self, timeout_s: float) -> bool:
        return self._first_frame.wait(timeout_s)

    def frame_for(self, camera_index: int) -> Optional[dict]:
        with self._lock:
            if not self._order:
                return None
            return self._latest[self._order[camera_index % len(self._order)]]

    def _on_message(self, payload: dict) -> None:
        if not _accept(payload, self._synthetic_ids, self._filter):
            return
        camera_id = str(payload["camera_id"])
        with self._lock:
            if camera_id not in self._latest:
                self._order = sorted([*self._order, camera_id])
            self._latest[camera_id] = payload
        self._first_frame.set()


class CorpusSource:
    """Loops a captured corpus; each synthetic camera starts at a different offset."""

    passthrough_dates = False

    def __init__(self, frames: FramesConfig, corpus_name: str) -> None:
        manifest = Path(frames.corpus_dir) / corpus_name / MANIFEST_NAME
        if not manifest.is_file():
            raise FileNotFoundError(f"Corpus manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            self._entries = [json.loads(line) for line in handle if line.strip()]
        if not self._entries:
            raise ValueError(f"Corpus '{corpus_name}' is empty")
        missing = [entry["image_path"] for entry in self._entries[:20] if not Path(entry["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Corpus images missing, e.g. {missing[0]}")
        self._cursors: dict[int, int] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait_for_first_frame(self, timeout_s: float) -> bool:
        return True

    def frame_for(self, camera_index: int) -> Optional[dict]:
        count = len(self._entries)
        cursor = self._cursors.get(camera_index, (camera_index * 7919) % count)
        self._cursors[camera_index] = cursor + 1
        return self._entries[cursor % count]


@dataclass
class CaptureResult:
    directory: Path
    frames: int
    missing_images: int


def capture_corpus(
    kafka: KafkaConfig,
    frames: FramesConfig,
    name: str,
    duration_s: float,
    source_camera_ids: list[str],
    max_frames: int = 0,
) -> CaptureResult:
    destination = Path(frames.corpus_dir) / name
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Corpus '{name}' already exists at {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if not str(destination.resolve()).startswith(str(Path(frames.shared_mount_path).resolve())):
        log.warning("Corpus dir %s is outside the shared frames mount %s; workers may not read it",
                    destination, frames.shared_mount_path)

    camera_filter = {str(camera_id) for camera_id in source_camera_ids}
    inbox: "queue.Queue[dict]" = queue.Queue()
    tap = TopicTap(kafka, lambda payload: inbox.put(payload) if _accept(payload, set(), camera_filter) else None)
    tap.start()
    saved = 0
    missing = 0
    deadline = time.monotonic() + duration_s
    try:
        with (destination / MANIFEST_NAME).open("w", encoding="utf-8") as manifest:
            while time.monotonic() < deadline and (max_frames <= 0 or saved < max_frames):
                try:
                    payload = inbox.get(timeout=0.5)
                except queue.Empty:
                    if tap.error:
                        raise RuntimeError(f"Kafka tap failed: {tap.error}")
                    continue
                source_path = Path(payload["image_path"])
                if not source_path.is_file():
                    missing += 1
                    continue
                target = destination / f"{saved:06d}{source_path.suffix or '.jpg'}"
                shutil.copy2(source_path, target)
                manifest.write(json.dumps({
                    "image_path": str(target),
                    "source_camera_id": str(payload.get("camera_id")),
                    "source_timestamp": payload.get("timestamp"),
                }) + "\n")
                saved += 1
    finally:
        tap.stop()
    return CaptureResult(directory=destination, frames=saved, missing_images=missing)
