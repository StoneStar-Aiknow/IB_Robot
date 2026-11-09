"""Shared HRI motion helpers (design D2 of the package-layout change).

``AnimationPlan`` and the motion-config/waypoint validation are used by both
the imitation executor (its fallback animations) and the retargeting stage
(its capture plans); they live here so retargeting does not import the
executor. The bodies moved unchanged from ``hri/imitate/executor.py``; the two
private helpers became public names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MAX_IMITATION_DURATION_SEC = 20.0


@dataclass(frozen=True)
class AnimationPlan:
    animation_id: str
    waypoints: tuple[tuple[float, ...], ...]
    duration_sec: float = MAX_IMITATION_DURATION_SEC
    # > 0: waypoints form a uniform grid with this period, played as one batch.
    grid_period_sec: float = 0.0


def normalized_motion_config(
    joint_names: Sequence[str],
    reset_positions: Mapping[str, float],
    joint_limits: Mapping[str, Mapping[str, float] | Sequence[float]],
) -> tuple[tuple[str, ...], dict[str, float], dict[str, tuple[float, float]]]:
    names = tuple(str(name).strip() for name in joint_names)
    if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("imitation motion requires at least two unique arm joint names")

    reset: dict[str, float] = {}
    limits: dict[str, tuple[float, float]] = {}
    for name in names:
        if name not in reset_positions or name not in joint_limits:
            raise ValueError(f"imitation motion configuration is missing joint {name}")
        position = float(reset_positions[name])
        raw_limits = joint_limits[name]
        if isinstance(raw_limits, Mapping):
            lower = float(raw_limits["min"])
            upper = float(raw_limits["max"])
        else:
            lower, upper = (float(value) for value in raw_limits)
        if not all(math.isfinite(value) for value in (position, lower, upper)) or lower >= upper:
            raise ValueError(f"imitation motion configuration for joint {name} is invalid")
        if not lower <= position <= upper:
            raise ValueError(f"reset position is outside joint {name} limits")
        reset[name] = position
        limits[name] = (lower, upper)
    return names, reset, limits


def waypoints(
    joint_names: tuple[str, ...],
    joint_limits: Mapping[str, tuple[float, float]],
    *points: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    for point in points:
        if len(point) != len(joint_names) or any(not math.isfinite(value) for value in point):
            raise ValueError("animation waypoint must contain all arm joint values")
        for joint_name, value in zip(joint_names, point, strict=True):
            lower, upper = joint_limits[joint_name]
            if not lower <= value <= upper:
                raise ValueError(f"animation waypoint is outside joint {joint_name} limits")
    return tuple(points)
