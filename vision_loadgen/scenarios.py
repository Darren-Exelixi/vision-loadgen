from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vision_loadgen.capacity import Capacity
from vision_loadgen.config import ConfigError, ScenarioConfig


@dataclass(frozen=True)
class Stage:
    name: str
    cameras: int
    duration_s: float
    evaluate: bool


def build_stages(scenario: ScenarioConfig) -> list[Stage]:
    if scenario.type == "throughput":
        counts = list(range(scenario.start_cameras, scenario.max_cameras + 1, scenario.step_cameras))
        if counts[-1] != scenario.max_cameras:
            counts.append(scenario.max_cameras)
        return [Stage(f"{count} cameras", count, scenario.step_duration_s, True) for count in counts]
    if scenario.type == "latency":
        return [Stage(f"level {count}", count, scenario.level_duration_s, True) for count in scenario.levels]
    return [Stage(f"soak {scenario.cameras}", scenario.cameras, scenario.duration_s, False)]


def cap_stages(stages: list[Stage], capacities: dict[str, Capacity],
               scenario: ScenarioConfig) -> tuple[list[Stage], Optional[str]]:
    """Drop stages that need more GPU memory than a worker has; a throughput ramp ends at the cap.

    A worker asked for more model copies than fit refuses to start its pipeline, which stops its
    real cameras too, so those stages are never run. Workers with unknown capacity do not cap.
    """
    limits = {name: capacity.max_synthetic_cameras for name, capacity in capacities.items()
              if capacity.max_synthetic_cameras is not None}
    if not limits:
        return stages, None
    limiting = min(limits, key=limits.get)
    cap = limits[limiting]
    kept = [stage for stage in stages if stage.cameras <= cap]
    if len(kept) == len(stages):
        return stages, None
    if scenario.type == "throughput" and cap >= 1 and (not kept or kept[-1].cameras < cap):
        kept.append(Stage(f"{cap} cameras (GPU cap)", cap, scenario.step_duration_s, True))
    if not kept:
        raise ConfigError(
            f"{limiting} has GPU memory for at most {cap} synthetic cameras next to its "
            f"{capacities[limiting].real_cameras} real ones; lower the camera count"
        )
    dropped = sorted({stage.cameras for stage in stages} - {stage.cameras for stage in kept})
    return kept, (
        f"GPU capacity caps the run at {cap} synthetic cameras ({limiting}); "
        f"skipping stages with {', '.join(map(str, dropped))} cameras"
    )
