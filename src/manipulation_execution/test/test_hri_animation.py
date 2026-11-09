"""Characterization tests for the shared HRI motion helpers (design D2/D10).

These pin the behaviour of ``AnimationPlan``, the motion-config normalization
and the waypoint validation that ``hri/animation.py`` will carry, plus one
normal ``retarget_capture`` path, BEFORE the D2 split. The split may change
only these tests' imports (openspec reorganize-manipulation-execution-layout,
task 2.1).
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from manipulation_execution.hri.animation import (
    MAX_IMITATION_DURATION_SEC,
    AnimationPlan,
    normalized_motion_config,
    waypoints,
)
from manipulation_execution.hri.retargeting import retarget_capture

# The SO-101 follower arm as the generic configuration under test.
ARM_NAMES = ("1", "2", "3", "4", "5")
ARM_LIMITS = {name: (-2.9, 2.9) for name in ARM_NAMES}
ARM_HOME = {name: 0.0 for name in ARM_NAMES}


class TestAnimationPlanDefaults:
    def test_default_duration_is_the_maximum(self):
        plan = AnimationPlan("a", ((0.0,) * 5,))
        assert plan.duration_sec == MAX_IMITATION_DURATION_SEC == 20.0

    def test_default_grid_period_is_zero(self):
        plan = AnimationPlan("a", ((0.0,) * 5,))
        assert plan.grid_period_sec == 0.0

    def test_is_frozen(self):
        plan = AnimationPlan("a", ((0.0,) * 5,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.animation_id = "b"


class TestNormalizedMotionConfig:
    def test_valid_configuration_is_normalized(self):
        names, reset, limits = normalized_motion_config(ARM_NAMES, ARM_HOME, ARM_LIMITS)
        assert names == ARM_NAMES
        assert reset == dict(ARM_HOME)
        assert limits == {name: (-2.9, 2.9) for name in ARM_NAMES}

    def test_limits_accept_mapping_and_sequence_forms(self):
        mixed = dict(ARM_LIMITS)
        mixed["1"] = {"min": -1.5, "max": 1.5}
        mixed["2"] = [-0.5, 0.75]
        _, _, limits = normalized_motion_config(ARM_NAMES, ARM_HOME, mixed)
        assert limits["1"] == (-1.5, 1.5)
        assert limits["2"] == (-0.5, 0.75)

    def test_requires_at_least_two_unique_names(self):
        with pytest.raises(ValueError, match="at least two unique arm joint names"):
            normalized_motion_config(("only",), {"only": 0.0}, {"only": (-1.0, 1.0)})
        with pytest.raises(ValueError, match="at least two unique arm joint names"):
            normalized_motion_config(("1", "1"), {"1": 0.0}, {"1": (-1.0, 1.0)})

    def test_rejects_blank_names(self):
        with pytest.raises(ValueError, match="at least two unique arm joint names"):
            normalized_motion_config(("1", " "), {"1": 0.0, " ": 0.0}, {"1": (-1.0, 1.0), " ": (-1.0, 1.0)})

    @pytest.mark.parametrize("missing", ["reset_positions", "joint_limits"])
    def test_missing_joint_in_either_mapping_is_rejected(self, missing):
        reset = dict(ARM_HOME)
        limits = dict(ARM_LIMITS)
        if missing == "reset_positions":
            del reset["3"]
        else:
            del limits["3"]
        with pytest.raises(ValueError, match="missing joint 3"):
            normalized_motion_config(ARM_NAMES, reset, limits)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_values_are_rejected(self, value):
        with pytest.raises(ValueError, match="configuration for joint 1 is invalid"):
            normalized_motion_config(ARM_NAMES, {"1": value, **{n: 0.0 for n in ARM_NAMES[1:]}}, ARM_LIMITS)

    def test_degenerate_limits_are_rejected(self):
        limits = dict(ARM_LIMITS)
        limits["2"] = (1.0, 1.0)
        with pytest.raises(ValueError, match="configuration for joint 2 is invalid"):
            normalized_motion_config(ARM_NAMES, ARM_HOME, limits)

    def test_reset_position_outside_limits_is_rejected(self):
        reset = dict(ARM_HOME)
        reset["4"] = 5.0
        with pytest.raises(ValueError, match="reset position is outside joint 4 limits"):
            normalized_motion_config(ARM_NAMES, reset, ARM_LIMITS)


class TestWaypoints:
    def test_points_pass_through_as_a_tuple(self):
        points = ((0.0,) * 5, (0.1, -0.1, 0.2, -0.2, 0.0))
        assert waypoints(ARM_NAMES, ARM_LIMITS, *points) == points

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="must contain all arm joint values"):
            waypoints(ARM_NAMES, ARM_LIMITS, (0.0, 0.0))

    def test_non_finite_value_is_rejected(self):
        with pytest.raises(ValueError, match="must contain all arm joint values"):
            waypoints(ARM_NAMES, ARM_LIMITS, (0.0, 0.0, float("nan"), 0.0, 0.0))

    def test_value_outside_limits_is_rejected(self):
        with pytest.raises(ValueError, match="outside joint 5 limits"):
            waypoints(ARM_NAMES, ARM_LIMITS, (0.0, 0.0, 0.0, 0.0, 9.0))


def _identity_pose() -> list[float]:
    """One SMPL-X frame whose 52 rotations all decode to the identity."""
    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return [value for _ in range(52) for value in identity_6d]


def _shoulder_swing_pose(angle_rad: float) -> list[float]:
    """One frame whose left-shoulder rotation (SMPL-X joint 16) is a swing.

    The 6D encoding is the first two columns of the rotation matrix; a swing
    about the X axis by ``angle`` therefore encodes as
    ``[1, 0, 0, 0, cos(angle), sin(angle)]``. Everything else stays identity,
    so the capture moves exactly one human arm joint.
    """
    pose = _identity_pose()
    base = 16 * 6
    pose[base : base + 6] = [1.0, 0.0, 0.0, 0.0, math.cos(angle_rad), math.sin(angle_rad)]
    return pose


def _capture_frames(count: int, rate_hz: float = 15.0) -> list[dict[str, object]]:
    """A capture whose left shoulder swings: real motion for the retargeter."""
    period_ns = int(1e9 / rate_hz)
    frames = []
    for index in range(count):
        angle = 0.6 * math.sin(2.0 * math.pi * index / 8.0)
        frames.append(
            {
                "stamp_sec": index * period_ns // 1_000_000_000,
                "stamp_nanosec": index * period_ns % 1_000_000_000,
                "smplx_pose_raw": _shoulder_swing_pose(angle),
            }
        )
    return frames


class TestRetargetCaptureNormalPath:
    def test_a_dense_capture_returns_a_playable_plan(self):
        result = retarget_capture(
            _capture_frames(count=16),
            arm_side="left",
            duration_sec=1.0,
            joint_names=ARM_NAMES,
            joint_limits=ARM_LIMITS,
            prepare_positions=ARM_HOME,
        )
        assert result.fell_back is False
        assert result.plan.animation_id != "idle_v1"
        assert len(result.plan.waypoints) == 20  # 1.0 s on the 0.05 s grid
        assert all(len(point) == 5 for point in result.plan.waypoints)
        assert all(math.isfinite(value) and -2.9 <= value <= 2.9 for point in result.plan.waypoints for value in point)
        assert result.plan.duration_sec == 1.0
        assert "retarget" in result.message
