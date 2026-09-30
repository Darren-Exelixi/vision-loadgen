from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from vision_loadgen.config import FramesConfig

LOADGEN_TAG = "loadgen"


def is_synthetic(payload: dict, synthetic_ids: set[str]) -> bool:
    return bool(payload.get(LOADGEN_TAG)) or str(payload.get("camera_id")) in synthetic_ids


def build_message(
    source: dict,
    camera_id: str,
    now: float,
    run_id: str,
    frames: FramesConfig,
    passthrough_dates: bool,
) -> dict:
    """Synthetic frame pointer: same image as the source frame, new camera and timestamp."""
    if passthrough_dates and source.get("frame_date") and source.get("frame_hour"):
        frame_date = source["frame_date"]
        frame_hour = source["frame_hour"]
    else:
        local = datetime.fromtimestamp(now, tz=timezone.utc).astimezone(ZoneInfo(frames.timezone))
        frame_date = local.strftime(frames.frame_date_format)
        frame_hour = local.strftime(frames.frame_hour_format)
    return {
        "camera_id": camera_id,
        "timestamp": now,
        "image_path": source["image_path"],
        "frame_date": frame_date,
        "frame_hour": frame_hour,
        LOADGEN_TAG: run_id,
    }
