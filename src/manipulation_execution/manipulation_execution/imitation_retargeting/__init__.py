"""Turn a recorded human capture into an arm animation plan.

``retarget_capture`` is the only entry point. It never raises for bad input or
a retargeting bug: anything that keeps the capture from being played back
falls back to an idle sway around the prepare pose, and the reason is carried
in the returned message so the fallback does not hide the problem.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..imitate_human_motion_executor import AnimationPlan, _normalized_motion_config
from . import cadence, plan_builder, scoring, smoothing
from .base import HUMAN_SIDES, Retargeter, RetargetRequest
from .frames import parse_frames
from .idle import idle_sway
from .smplx2so101 import Smplx2So101

IDLE_ANIMATION_ID = "idle_v1"
DEFAULT_RETARGETER = "so101_arm_v1"
_RETARGETERS: dict[str, type] = {Smplx2So101.name: Smplx2So101}


@dataclass(frozen=True)
class RetargetResult:
    plan: AnimationPlan
    message: str
    fell_back: bool


class _Fallback(Exception):
    """Capture cannot be played back; the message says why."""


def retarget_capture(
    frames: Sequence[Mapping[str, object]],
    *,
    arm_side: str,
    duration_sec: float,
    joint_names: Sequence[str],
    joint_limits: Mapping[str, Mapping[str, float] | Sequence[float]],
    prepare_positions: Mapping[str, float],
    retargeter_name: str = DEFAULT_RETARGETER,
) -> RetargetResult:
    """Retarget ``frames`` onto the arm, or fall back to the idle sway.

    Raises only when the robot configuration itself is unusable, in which
    case not even the idle sway can be built.
    """
    started = time.perf_counter()
    names, prepare, limits = _normalized_motion_config(joint_names, prepare_positions, joint_limits)
    retargeter: Retargeter = _RETARGETERS[retargeter_name]()
    missing = [channel for channel in retargeter.channels if channel not in names]
    if missing:
        raise ValueError(f"retargeter {retargeter_name} drives unconfigured joints {missing}")
    working = dict(zip(retargeter.channels, retargeter.working_range, strict=True))
    grid = plan_builder.make_grid(names, limits, prepare, working, float(duration_sec))
    amplitude = plan_builder.expand(
        grid, retargeter.channels, np.array([retargeter.idle_amplitude]), fill=np.zeros_like(grid.prepare)
    )[0]
    idle, _ = plan_builder.shape(
        grid, idle_sway(grid.count, grid.period, grid.prepare, amplitude, grid.lower, grid.upper)
    )
    idle_travel = scoring.travel(idle, grid.lower, grid.upper)

    notes: list[str] = []
    try:
        animation_id, values = _retarget(retargeter, grid, frames, arm_side, idle_travel, notes)
        plan = plan_builder.build_plan(grid, animation_id, plan_builder.finish(grid, values), limits)
        fell_back = False
    except Exception as error:  # noqa: BLE001 - any failure here must end in the idle sway
        reason = str(error) if isinstance(error, _Fallback) else f"{type(error).__name__}: {error}"
        notes.insert(0, f"idle fallback: {reason}")
        plan = plan_builder.build_plan(grid, IDLE_ANIMATION_ID, plan_builder.finish(grid, idle), limits)
        fell_back = True
    notes.append(f"{grid.count} waypoints x {grid.period:.3f}s")
    notes.append(f"retarget {1000.0 * (time.perf_counter() - started):.0f}ms")
    return RetargetResult(plan, f"{plan.animation_id}: " + ", ".join(notes), fell_back)


def _retarget(
    retargeter: Retargeter,
    grid: plan_builder.Grid,
    frames: Sequence[Mapping[str, object]],
    arm_side: str,
    idle_travel: float,
    notes: list[str],
) -> tuple[str, np.ndarray]:
    track = parse_frames(frames)
    notes.append(f"{len(track)} frames ({track.dropped} dropped) over {track.span_sec:.2f}s")
    reason = cadence.gate(track.t, grid.duration_sec)
    if reason is not None:
        raise _Fallback(reason)

    shaped: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    changed: dict[str, int] = {}
    for side in HUMAN_SIDES:
        output = retargeter.retarget(RetargetRequest(track, side))
        values, _ = smoothing.reject_outliers(np.asarray(output.values, dtype=np.float64))
        resampled = plan_builder.expand(grid, output.channels, cadence.resample(output.timestamps, values, grid.count))
        shaped[side], changed[side] = plan_builder.shape(grid, resampled)
        scores[side] = scoring.travel(shaped[side], grid.lower, grid.upper)
    notes.append(f"travel left={scores['left']:.2f} right={scores['right']:.2f} idle={idle_travel:.2f}")
    side = scoring.pick_side(scores, arm_side)
    notes.append(f"{side} arm ({arm_side}), {100.0 * changed[side] / shaped[side].size:.1f}% bounded")
    if scores[side] < idle_travel:
        raise _Fallback(f"{side} arm travel {scores[side]:.2f} is below the idle sway's {idle_travel:.2f}")
    return f"imitation_{side}_v1", shaped[side]


__all__ = ["DEFAULT_RETARGETER", "IDLE_ANIMATION_ID", "RetargetResult", "retarget_capture"]
