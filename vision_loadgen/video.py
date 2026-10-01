"""Turn a video file into a replayable corpus (JPEGs + manifest) for `source.mode=corpus`.

The workers read each frame from the `image_path` in the Kafka message, so the corpus must be
on the workers' disk. Frames are written locally with relative paths; `corpus.json` records the
folder the workers will see them in (`image_root`) once the folder is copied there.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from vision_loadgen.config import ConfigError
from vision_loadgen.sources import HEADER_NAME, MANIFEST_NAME

log = logging.getLogger(__name__)

# The workers' EVENTS_DIR: a bind mount shared by the modules and kept across restarts (the frames
# mount is a tmpfs whose old files are deleted).
DEFAULT_REMOTE_CORPUS_ROOT = "/app/events/loadgen_corpus"


@dataclass
class VideoCorpus:
    name: str
    directory: str
    image_root: str
    frames: int
    fps: float
    source_fps: float
    width: int
    height: int
    source_video: str
    created_at: float


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ConfigError("Reading videos needs OpenCV: pip install \"vision-loadgen[video]\" "
                          "(or opencv-python-headless)") from exc
    return cv2


def corpus_from_video(
    video: str | Path,
    name: str,
    out_dir: str | Path,
    fps: float = 5.0,
    max_seconds: float = 0.0,
    image_root: str = "",
    quality: int = 90,
    progress: Optional[Callable[[int, int], None]] = None,
) -> VideoCorpus:
    """Sample `video` at about `fps` into `<out_dir>/<name>/NNNNNN.jpg` with a relative manifest."""
    if not name or any(char in name for char in "/\\:") or name.startswith("."):
        raise ConfigError(f"Invalid corpus name '{name}'")
    if fps <= 0:
        raise ConfigError("--fps must be positive")
    video = Path(video)
    if not video.is_file():
        raise ConfigError(f"Video not found: {video}")
    destination = Path(out_dir) / name
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Corpus '{name}' already exists at {destination}")

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ConfigError(f"OpenCV cannot open {video}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps <= 0 or source_fps > 1000:
            log.warning("%s reports no frame rate; assuming %.1f fps", video, fps)
            source_fps = fps
        step = max(1, round(source_fps / fps))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = int(max_seconds * source_fps) if max_seconds > 0 else 0
        expected = (min(total, limit) if limit and total else (limit or total)) // step
        destination.mkdir(parents=True, exist_ok=True)
        saved = 0
        width = height = 0
        index = -1
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        with (destination / MANIFEST_NAME).open("w", encoding="utf-8") as manifest:
            while True:
                grabbed = capture.grab()
                if not grabbed:
                    break
                index += 1
                if limit and index >= limit:
                    break
                if index % step:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    continue
                height, width = frame.shape[:2]
                filename = f"{saved:06d}.jpg"
                if not cv2.imwrite(str(destination / filename), frame, params):
                    raise RuntimeError(f"Could not write {destination / filename}")
                manifest.write(json.dumps({"image_path": filename, "source_video": video.name,
                                           "frame_index": index}) + "\n")
                saved += 1
                if progress and saved % 50 == 0:
                    progress(saved, expected)
    finally:
        capture.release()
    if not saved:
        raise ConfigError(f"No frames could be decoded from {video}")

    corpus = VideoCorpus(
        name=name,
        directory=str(destination),
        image_root=image_root or f"{DEFAULT_REMOTE_CORPUS_ROOT}/{name}",
        frames=saved,
        fps=round(source_fps / step, 3),
        source_fps=round(source_fps, 3),
        width=width,
        height=height,
        source_video=video.name,
        created_at=time.time(),
    )
    (destination / HEADER_NAME).write_text(json.dumps(asdict(corpus), indent=2), encoding="utf-8")
    if progress:
        progress(saved, saved)
    return corpus


def copy_hint(corpus: VideoCorpus, target: str = "") -> str:
    """The command that puts the corpus where the workers read it."""
    parent = corpus.image_root.rsplit("/", 1)[0]
    target = target or f"<user>@<modules host>:<compose folder>/events/{parent.rsplit('/', 1)[-1]}/"
    return f'scp -r "{corpus.directory}" {target}'
