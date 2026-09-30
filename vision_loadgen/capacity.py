"""GPU sizing, mirroring GpuInferenceEngine.

The engine runs one model copy per CAMERAS_PER_WORKER cameras (plus one API model when it has
one), refuses to start when copies x VRAM_PER_WORKER_MB exceeds the VRAM that was free when the
worker process started, and gives camera i (of the sorted camera ids) to copy i // CAMERAS_PER_WORKER.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from vision_loadgen.config import WorkerConfig


def gpu_workers_for(camera_count: int, cameras_per_worker: int) -> int:
    """Model copies the engine starts for this many cameras (GpuInferenceEngine._calculate_worker_count)."""
    return max(1, math.ceil(camera_count / cameras_per_worker))


def gpu_worker_map(camera_ids: Iterable[str], cameras_per_worker: int) -> dict[str, int]:
    """Which model copy serves each camera (GpuInferenceEngine._build_camera_worker_map)."""
    return {camera_id: index // cameras_per_worker for index, camera_id in enumerate(sorted(set(camera_ids)))}


def nvidia_smi_free_mb() -> Optional[int]:
    """Free VRAM on the first visible GPU, the way the engine measures it; None without nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return None


@dataclass
class Capacity:
    worker: str
    cameras_per_worker: Optional[int]
    vram_per_worker_mb: Optional[int]
    api_worker: bool
    real_cameras: int
    budget_mb: Optional[int]
    budget_source: str
    settings_source: str
    max_total_cameras: Optional[int] = None
    max_synthetic_cameras: Optional[int] = None

    def gpu_workers(self, synthetic: int) -> Optional[int]:
        if not self.cameras_per_worker:
            return None
        return gpu_workers_for(self.real_cameras + synthetic, self.cameras_per_worker)

    def as_dict(self) -> dict:
        return asdict(self)


def plan_capacity(name: str, worker: WorkerConfig, real_cameras: int,
                  probe_free_mb: Optional[int] = None) -> Capacity:
    """How many synthetic cameras fit next to the worker's real ones.

    The budget is gpu_free_vram_mb when configured. Otherwise, when nvidia-smi is visible here
    (a one-off container of the worker's own service), it is estimated as free VRAM now plus
    what the worker's current model copies hold.
    """
    capacity = Capacity(
        worker=name,
        cameras_per_worker=worker.cameras_per_worker,
        vram_per_worker_mb=worker.vram_per_worker_mb,
        api_worker=worker.api_worker,
        real_cameras=real_cameras,
        budget_mb=None,
        budget_source="unknown",
        settings_source=worker.gpu_settings_source or "config",
    )
    if not worker.cameras_per_worker or not worker.vram_per_worker_mb:
        capacity.budget_source = "unknown (CAMERAS_PER_WORKER / VRAM_PER_WORKER_MB not set)"
        return capacity
    extra = 1 if worker.api_worker else 0
    if worker.gpu_free_vram_mb is not None:
        capacity.budget_mb = worker.gpu_free_vram_mb
        capacity.budget_source = "gpu_free_vram_mb"
    elif probe_free_mb is not None:
        held = (gpu_workers_for(real_cameras, worker.cameras_per_worker) if real_cameras else 0) + extra
        capacity.budget_mb = probe_free_mb + held * worker.vram_per_worker_mb
        capacity.budget_source = "estimated: nvidia-smi free + current model copies"
    if capacity.budget_mb is None:
        return capacity
    copies = capacity.budget_mb // worker.vram_per_worker_mb - extra
    capacity.max_total_cameras = max(0, copies) * worker.cameras_per_worker
    capacity.max_synthetic_cameras = max(0, capacity.max_total_cameras - real_cameras)
    return capacity


def probe_free_mb() -> Optional[int]:
    if os.environ.get("LOADGEN_NO_GPU_PROBE"):
        return None
    return nvidia_smi_free_mb()
