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
    hands: tuple[tuple[float, ...], ...] = ()

    def sample_hands(self, t: float) -> list[float]:
        if not self.hands:
            return []
        return ArmTrajectory(self.name, self.target, self.times, self.hands).sample(t)

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
    hand_rows = ()
    if data.get("format") == "ibrobot.fixed_motion.delivery":
        arms, hands = data.get("arms", {}), data.get("hands", {})
        if data.get("schema_version") != 1 or arms.get("units") != "rad" or hands.get("units") != "rad":
            raise TrajectoryError(f"{path}: delivery requires schema 1 and rad arms/hands")
        hand_rows = hands.get("positions", [])
        if len(hand_rows) != len(data.get("time_from_start_s", [])) or not hand_rows:
            raise TrajectoryError(f"{path}: hands require one sample per time")
        if any(len(row) != 20 or any(not math.isfinite(float(v)) for v in row) for row in hand_rows):
            raise TrajectoryError(f"{path}: hands require 20 finite rad values")
        from aimdk_robot.projection import DEXTEROUS_JOINT_ORDER

        expected = [f"{side}_{joint}" for side in ("left", "right") for joint in DEXTEROUS_JOINT_ORDER]
        if hands.get("joints") != expected:
            raise TrajectoryError(f"{path}: hands must be left then right in vendor joint order")
        hand_rows = tuple(tuple(float(v) for v in row) for row in hand_rows)
        data = {**data, "target": "both", "joints": arms.get("joints", []), "positions": arms.get("positions_rad", [])}
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
        hands=hand_rows,
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
) -> ArmTrajectory:
    """A caller-computed arm trajectory, from the measured ``start``, ready to replay.

    ``joint_names`` may be any subset of the arm joints; the others hold
    ``start``. Waypoints are joined by a cosine ease (zero velocity at each
    waypoint, the same profile the replay uses to ease in), sampled at
    ``rate_hz``. A goal is refused rather than repaired: an unknown or repeated
    joint, a non-finite value, time not moving forward, a target outside
    ``limits`` or a step faster than ``max_speed_radps`` would each execute a
    motion the caller did not plan.
    """
    names = [str(joint) for joint in joint_names]
    unknown = sorted(set(names) - set(VENDOR_ARM_ORDER))
    if not names or unknown or len(set(names)) != len(names):
        raise TrajectoryError(f"joint_names must be distinct arm joints; unknown {unknown}" if unknown else
                              "joint_names must be distinct and non-empty")  # fmt: skip
    if not points or len(points) != len(times):
        raise TrajectoryError("needs at least one point and one time per point")
    if times[0] <= 0.0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise TrajectoryError("time_from_start must be positive and strictly increasing")
    index = {joint: i for i, joint in enumerate(names)}
    knots = [tuple(float(v) for v in start)]
    for row in points:
        if len(row) != len(names) or not all(math.isfinite(float(v)) for v in row):
            raise TrajectoryError(f"every point needs {len(names)} finite positions")
        for joint, value in zip(names, row, strict=True):
            low, high = limits[joint]
            if not low <= float(value) <= high:
                raise TrajectoryError(f"{joint} target {float(value):.3f} outside [{low:.3f}, {high:.3f}]")
        knots.append(tuple(float(row[index[j]]) if j in index else knots[0][k] for k, j in enumerate(VENDOR_ARM_ORDER)))
    knot_times = [0.0, *(float(t) for t in times)]
    sample_times, samples = [0.0], [knots[0]]
    for k in range(1, len(knots)):
        t0, t1 = knot_times[k - 1], knot_times[k]
        steps = max(1, math.ceil((t1 - t0) * rate_hz))
        for i in range(1, steps + 1):
            sample_times.append(t0 + (t1 - t0) * i / steps)
            samples.append(tuple(blend(knots[k - 1], knots[k], i / steps)))
    for k in range(1, len(samples)):
        dt = sample_times[k] - sample_times[k - 1]
        step = max(abs(b - a) for a, b in zip(samples[k - 1], samples[k], strict=True))
        if step / dt > max_speed_radps:
            raise TrajectoryError(
                f"{step / dt:.2f} rad/s at t={sample_times[k]:.2f}s exceeds the {max_speed_radps} rad/s limit"
            )
    return ArmTrajectory(name=name, target="both", times=tuple(sample_times), positions=tuple(samples))
