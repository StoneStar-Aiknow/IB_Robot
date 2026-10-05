"""Shape grid trajectories into safe, playable animation plans.

Every trajectory -- retargeted or idle -- goes through the same steps: low-pass,
clamp, rate limit, then an entry blend from the prepare pose and a final clamp
and rate limit that starts at the prepare pose. The result is checked against
the invariants the robot relies on before an :class:`AnimationPlan` is built.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..imitate_human_motion_executor import AnimationPlan, _waypoints
from . import cadence, smoothing

V_MAX_RAD_S = math.radians(120.0)
# Kept between the plan and the configured joint limits: the batch goes out as
# float32, and a value rounded past a limit would be rejected downstream.
LIMIT_MARGIN_RAD = 1e-3
_STEP_TOLERANCE_RAD = 1e-9


@dataclass(frozen=True)
class Grid:
    """Joint order, bounds and timing shared by every trajectory of one capture."""

    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    prepare: np.ndarray
    count: int
    period: float
    duration_sec: float

    @property
    def max_step(self) -> float:
        return V_MAX_RAD_S * self.period


def make_grid(
    joint_names: tuple[str, ...],
    joint_limits: Mapping[str, tuple[float, float]],
    prepare: Mapping[str, float],
    working_range: Mapping[str, tuple[float, float]],
    duration_sec: float,
) -> Grid:
    """Bounds are the working range inside the joint limits, never excluding the prepare pose."""
    if not (math.isfinite(duration_sec) and duration_sec > 0.0):
        raise ValueError(f"imitation duration must be positive, got {duration_sec!r}")
    lower, upper, start = [], [], []
    for name in joint_names:
        limit_lower, limit_upper = joint_limits[name]
        work_lower, work_upper = working_range.get(name, (limit_lower, limit_upper))
        low = max(work_lower, limit_lower + LIMIT_MARGIN_RAD)
        high = min(work_upper, limit_upper - LIMIT_MARGIN_RAD)
        position = float(prepare[name])
        lower.append(min(low, position))
        upper.append(max(high, position))
        start.append(position)
    count, period = cadence.grid(duration_sec)
    return Grid(joint_names, np.array(lower), np.array(upper), np.array(start), count, period, duration_sec)


def expand(grid: Grid, channels: Sequence[str], values: np.ndarray, fill: np.ndarray | None = None) -> np.ndarray:
    """Place per-channel columns into full joint rows; other joints take ``fill`` (the prepare pose)."""
    full = np.tile(grid.prepare if fill is None else fill, (values.shape[0], 1))
    for column, channel in enumerate(channels):
        full[:, grid.joint_names.index(channel)] = values[:, column]
    return full


def shape(grid: Grid, values: np.ndarray) -> tuple[np.ndarray, int]:
    """Low-pass, clamp and rate limit a ``(count, joints)`` trajectory.

    Returns the trajectory and how many samples the clamp and rate limit had to
    change, which tells how much of the motion was cut by the safety bounds.
    """
    smoothed = smoothing.low_pass(values)
    clamped, clamped_count = smoothing.clamp(smoothed, grid.lower, grid.upper)
    limited, limited_count = smoothing.rate_limit(clamped, grid.max_step)
    return limited, clamped_count + limited_count


def finish(grid: Grid, values: np.ndarray) -> np.ndarray:
    """Blend in from the prepare pose and bound the result once more from there."""
    blended = cadence.entry_blend(values, grid.prepare, grid.period, V_MAX_RAD_S)
    clamped, _ = smoothing.clamp(blended, grid.lower, grid.upper)
    limited, _ = smoothing.rate_limit(clamped, grid.max_step, start=grid.prepare)
    check(grid, limited)
    return limited


def check(grid: Grid, values: np.ndarray) -> None:
    """Assert the invariants playback relies on; a violation is a bug, not bad input."""
    if values.shape != (grid.count, len(grid.joint_names)):
        raise ValueError(f"trajectory shape {values.shape} does not match the grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory is not finite")
    if np.any(values < grid.lower) or np.any(values > grid.upper):
        raise ValueError("trajectory leaves the joint working range")
    steps = np.abs(np.diff(np.vstack((grid.prepare, values)), axis=0))
    if np.any(steps > grid.max_step + _STEP_TOLERANCE_RAD):
        raise ValueError(f"trajectory step {float(np.max(steps)):.4f} rad exceeds {grid.max_step:.4f} rad")


def build_plan(
    grid: Grid, animation_id: str, values: np.ndarray, joint_limits: Mapping[str, tuple[float, float]]
) -> AnimationPlan:
    points = tuple(tuple(float(value) for value in row) for row in values)
    return AnimationPlan(
        animation_id,
        _waypoints(grid.joint_names, joint_limits, *points),
        duration_sec=grid.duration_sec,
        grid_period_sec=grid.period,
    )
