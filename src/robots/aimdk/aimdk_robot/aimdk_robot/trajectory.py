"""Recorded arm trajectories that replay named motions on the upper-body stream.

The platform plays its preset motions only from stable stand. In the
upper-body stream mode the same gestures are replayed from joint trajectories
recorded off the platform's own presets (``aimdk_record_named_motion``), so a
caller does not have to leave the stream mode to wave.

A trajectory file is JSON::

    {"name": "wave", "target": "right",
     "joints": [<every arm joint, any order>],
     "time_from_start_s": [0.0, 0.02, ...],
     "positions": [[...one value per joint...], ...]}

Pure functions only: loading, validation and sampling are testable without ROS.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aimdk_robot.projection import VENDOR_ARM_ORDER

# The sides a recording can carry: the preset targets of named_motion's
# CONTROL_AREAS, without its empty "use the profile's" default.
TARGETS: tuple[str, ...] = ("left", "right", "both", "body")


class TrajectoryError(ValueError):
    """A trajectory file that must not be replayed."""


@dataclass(frozen=True)
class ArmTrajectory:
    """Arm positions in vendor order, sampled at strictly increasing times."""

    name: str
    target: str
    times: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]

    @property
    def duration(self) -> float:
        return self.times[-1]

    def sample(self, t: float) -> list[float]:
        """Linearly interpolated positions at ``t`` seconds, held at both ends."""
        if t <= self.times[0]:
            return list(self.positions[0])
        if t >= self.times[-1]:
            return list(self.positions[-1])
        upper = bisect.bisect_right(self.times, t)
        t0, t1 = self.times[upper - 1], self.times[upper]
        ratio = (t - t0) / (t1 - t0)
        return [a + (b - a) * ratio for a, b in zip(self.positions[upper - 1], self.positions[upper], strict=True)]


def load_trajectory(path: Path, *, max_speed_radps: float) -> ArmTrajectory:
    """Load and validate one trajectory file; reorder its joints to vendor order.

    A file is refused rather than repaired: an unknown side, a missing joint, a
    non-finite value, time running backwards or a step faster than
    ``max_speed_radps`` would each execute a motion nobody recorded.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrajectoryError(f"{path}: {exc}") from exc
    target = str(data.get("target", ""))
    if target not in TARGETS:
        # An unknown side cannot be checked against a request, so it would
        # replay for whichever side the caller asked for.
        raise TrajectoryError(f"{path}: target must be the recorded side, one of {list(TARGETS)}; got {target!r}")
    joints = [str(joint) for joint in data.get("joints", [])]
    if sorted(joints) != sorted(VENDOR_ARM_ORDER):
        raise TrajectoryError(f"{path}: joints must be exactly the arm joints {list(VENDOR_ARM_ORDER)}")
    times = [float(t) for t in data.get("time_from_start_s", [])]
    rows = data.get("positions", [])
    if len(times) < 2 or len(times) != len(rows):
        raise TrajectoryError(f"{path}: needs at least two samples and one time per sample")
    if times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise TrajectoryError(f"{path}: time_from_start_s must start at 0 and strictly increase")
    index = {joint: i for i, joint in enumerate(joints)}
    positions: list[tuple[float, ...]] = []
    for row in rows:
        if len(row) != len(joints) or not all(math.isfinite(float(value)) for value in row):
            raise TrajectoryError(f"{path}: every sample needs {len(joints)} finite values")
        positions.append(tuple(float(row[index[joint]]) for joint in VENDOR_ARM_ORDER))
    for k in range(1, len(positions)):
        dt = times[k] - times[k - 1]
        step = max(abs(b - a) for a, b in zip(positions[k - 1], positions[k], strict=True))
        if step / dt > max_speed_radps:
            raise TrajectoryError(
                f"{path}: {step / dt:.2f} rad/s at t={times[k]:.2f}s exceeds the {max_speed_radps} rad/s limit"
            )
    return ArmTrajectory(
        name=str(data.get("name", Path(path).stem)),
        target=target,
        times=tuple(times),
        positions=tuple(positions),
    )


def blend_duration(start: Sequence[float], goal: Sequence[float], *, speed_radps: float, minimum_s: float) -> float:
    """Time to move from ``start`` to ``goal`` at about ``speed_radps`` on the farthest joint."""
    distance = max((abs(b - a) for a, b in zip(start, goal, strict=True)), default=0.0)
    return max(minimum_s, distance / speed_radps)


def blend(start: Sequence[float], goal: Sequence[float], ratio: float) -> list[float]:
    """Cosine ease from ``start`` (ratio 0) to ``goal`` (ratio 1): zero velocity at both ends."""
    ratio = min(max(ratio, 0.0), 1.0)
    weight = 0.5 - 0.5 * math.cos(math.pi * ratio)
    return [a + (b - a) * weight for a, b in zip(start, goal, strict=True)]


# A caller-planned trajectory longer than this is refused before it is
# sampled: the samples are built up front, so an unbounded time_from_start
# would exhaust memory before any other check ran.
PLANNED_TRAJECTORY_MAX_S = 120.0
# A point at time_from_start 0 is the caller's view of where the arm starts
# (MoveIt and most planners send one). It is accepted, and dropped, only within
# this distance of the start this trajectory eases from or of the measured arm.
START_POINT_TOLERANCE_RAD = 0.05


def planned_trajectory(
    name: str,
    start: Sequence[float],
    joint_names: Sequence[str],
    times: Sequence[float],
    points: Sequence[Sequence[float]],
    *,
    limits: dict[str, tuple[float, float]],
    max_speed_radps: float,
    rate_hz: float,
    measured: Sequence[float] | None = None,
) -> ArmTrajectory:
    """A caller-computed arm trajectory, from ``start``, ready to replay.

    ``start`` is where the command starts (the platform's arm target or the
    measured pose, see ``arm_takeover_start``); ``measured`` is the measured
    pose when it differs. ``joint_names`` may be any subset of the arm joints;
    the others hold ``start``. A first point at time 0 must be the start
    (within ``START_POINT_TOLERANCE_RAD`` of ``start`` or ``measured``) and is
    dropped. Waypoints are joined linearly in joint space and sampled at
    ``rate_hz``, as a joint trajectory controller does with positions only:
    a planner's time parameterization is reproduced, and a dense plan is not
    stopped at every waypoint. A single waypoint is reached by a cosine ease
    instead, so a bare "go there in t seconds" starts and ends at rest.
    Velocities, if sent, are not used. A goal is refused rather than repaired:
    an unknown or repeated joint, a non-finite value, time not moving forward,
    a start point elsewhere, a duration over ``PLANNED_TRAJECTORY_MAX_S``, a
    target outside ``limits`` or a step faster than ``max_speed_radps`` would
    each execute a motion the caller did not plan.
    """
    names = [str(joint) for joint in joint_names]
    unknown = sorted(set(names) - set(VENDOR_ARM_ORDER))
    if not names or unknown or len(set(names)) != len(names):
        raise TrajectoryError(f"joint_names must be distinct arm joints; unknown {unknown}" if unknown else
                              "joint_names must be distinct and non-empty")  # fmt: skip
    if not points or len(points) != len(times):
        raise TrajectoryError("needs at least one point and one time per point")
    times = [float(t) for t in times]
    if not all(math.isfinite(t) for t in times):
        raise TrajectoryError("time_from_start must be finite")
    if times[0] < 0.0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise TrajectoryError("time_from_start must not be negative and must be strictly increasing")
    if times[-1] > PLANNED_TRAJECTORY_MAX_S:
        raise TrajectoryError(f"lasts {times[-1]:.1f}s, over the {PLANNED_TRAJECTORY_MAX_S:g}s limit")
    for row in points:
        if len(row) != len(names) or not all(math.isfinite(float(v)) for v in row):
            raise TrajectoryError(f"every point needs {len(names)} finite positions")
    index = {joint: i for i, joint in enumerate(names)}
    if times[0] == 0.0:
        _check_start_point(names, points[0], start, measured)
        times, points = times[1:], points[1:]
        if not points:
            raise TrajectoryError("needs a point after time 0")
    knots = [tuple(float(v) for v in start)]
    for row in points:
        for joint, value in zip(names, row, strict=True):
            low, high = limits[joint]
            if not low <= float(value) <= high:
                raise TrajectoryError(f"{joint} target {float(value):.3f} outside [{low:.3f}, {high:.3f}]")
        knots.append(tuple(float(row[index[j]]) if j in index else knots[0][k] for k, j in enumerate(VENDOR_ARM_ORDER)))
    knot_times = [0.0, *times]
    join = blend if len(knots) == 2 else _lerp
    sample_times, samples = [0.0], [knots[0]]
    for k in range(1, len(knots)):
        t0, t1 = knot_times[k - 1], knot_times[k]
        steps = max(1, math.ceil((t1 - t0) * rate_hz))
        for i in range(1, steps + 1):
            sample_times.append(t0 + (t1 - t0) * i / steps)
            samples.append(tuple(join(knots[k - 1], knots[k], i / steps)))
    for k in range(1, len(samples)):
        dt = sample_times[k] - sample_times[k - 1]
        step = max(abs(b - a) for a, b in zip(samples[k - 1], samples[k], strict=True))
        if step / dt > max_speed_radps:
            raise TrajectoryError(
                f"{step / dt:.2f} rad/s at t={sample_times[k]:.2f}s exceeds the {max_speed_radps} rad/s limit"
            )
    return ArmTrajectory(name=name, target="both", times=tuple(sample_times), positions=tuple(samples))


def _lerp(start: Sequence[float], goal: Sequence[float], ratio: float) -> list[float]:
    return [a + (b - a) * ratio for a, b in zip(start, goal, strict=True)]


def _check_start_point(
    names: Sequence[str], row: Sequence[float], start: Sequence[float], measured: Sequence[float] | None
) -> None:
    for joint, value in zip(names, row, strict=True):
        k = VENDOR_ARM_ORDER.index(joint)
        nearest = min(abs(float(value) - float(ref[k])) for ref in (start, measured) if ref is not None)
        if nearest > START_POINT_TOLERANCE_RAD:
            raise TrajectoryError(
                f"the point at time 0 puts {joint} {nearest:.3f} rad from where the arm starts "
                f"(over {START_POINT_TOLERANCE_RAD} rad)"
            )


def goal_tolerances(
    joints: Sequence[str],
    requested: dict[str, float],
    start: Sequence[float],
    final: Sequence[float],
    *,
    default_rad: float,
    default_ratio: float,
) -> dict[str, float]:
    """The final error each goal joint may keep, by name; joints the caller exempted are left out.

    ``requested`` is the goal's position tolerance per joint, as
    FollowJointTrajectory defines it: positive is the tolerance, negative is
    "not checked", zero (or absent) is the default. The default is a coarse
    arrival check, not a precision gate: the X2 holds a pose with an error
    against gravity and a caller may raise its commands by the sag it
    predicts, so it only catches a joint that stopped well short of its final
    target: off by more than ``default_rad`` and more than ``default_ratio``
    of its commanded travel from ``start``.
    """
    unknown = sorted(set(requested) - set(VENDOR_ARM_ORDER))
    if unknown:
        raise TrajectoryError(f"goal_tolerance names joints that are not arm joints: {unknown}")
    tolerances = {}
    for joint in joints:
        value = float(requested.get(joint, 0.0))
        if not math.isfinite(value):
            raise TrajectoryError(f"goal_tolerance of {joint} must be finite")
        if value < 0.0:
            continue
        k = VENDOR_ARM_ORDER.index(joint)
        travel = abs(float(final[k]) - float(start[k]))
        tolerances[joint] = value if value > 0.0 else max(default_rad, default_ratio * travel)
    return tolerances
