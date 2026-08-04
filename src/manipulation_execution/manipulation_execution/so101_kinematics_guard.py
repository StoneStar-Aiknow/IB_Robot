"""SO101 joint-5 branch and orientation guards for grasp execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import JointState

from manipulation_execution.grasp_geometry import Quaternion, quaternion_matrix

__all__ = [
    "Joint5RetryResult",
    "apply_joint5_retry",
    "canonicalize_joint5",
    "joint5_branch_continuity_check",
    "joint5_branch_delta",
    "joint5_branch_filter_check",
    "joint5_closing_axis_correction",
    "joint5_within_abs_limit",
]


def _normalized_axis(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(value), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("vector must be non-zero")
    return vector / norm


def canonicalize_joint5(joint5: float) -> float:
    """Map SO101 joint5 onto the equivalent half-turn branch nearest zero."""

    value = float(joint5)
    while value > math.pi / 2.0:
        value -= math.pi
    while value < -math.pi / 2.0:
        value += math.pi
    return value


def joint5_closing_axis_correction(
    target_quaternion: Quaternion,
    actual_quaternion: Quaternion,
    approach_axis_ee: Iterable[float],
    closing_axis_ee: Iterable[float],
    *,
    closing_axis_180_symmetric: bool = False,
) -> float:
    """Return the signed SO101 joint5 correction that aligns the closing axis."""

    target_rotation = quaternion_matrix(target_quaternion)
    actual_rotation = quaternion_matrix(actual_quaternion)
    approach_axis = actual_rotation @ _normalized_axis(approach_axis_ee)
    actual_closing = np.asarray(actual_rotation @ _normalized_axis(closing_axis_ee), dtype=np.float64)
    target_closing = np.asarray(target_rotation @ _normalized_axis(closing_axis_ee), dtype=np.float64)
    if closing_axis_180_symmetric and float(np.dot(actual_closing, target_closing)) < 0.0:
        target_closing *= -1.0

    actual_closing -= approach_axis * float(np.dot(actual_closing, approach_axis))
    target_closing -= approach_axis * float(np.dot(target_closing, approach_axis))
    actual_norm = float(np.linalg.norm(actual_closing))
    target_norm = float(np.linalg.norm(target_closing))
    if actual_norm <= 1e-9 or target_norm <= 1e-9:
        raise ValueError("could not project closing axes onto the joint5 rotation plane")
    actual_closing /= actual_norm
    target_closing /= target_norm
    return math.atan2(
        float(np.dot(approach_axis, np.cross(actual_closing, target_closing))),
        float(np.dot(actual_closing, target_closing)),
    )


def joint5_branch_delta(seed_joint5: float | None, solution_joint5: float | None) -> float | None:
    """Return the absolute difference between two joint-5 values."""

    if seed_joint5 is None or solution_joint5 is None:
        return None
    return abs(float(solution_joint5) - float(seed_joint5))


def joint5_branch_filter_check(
    seed_joint5: float | None,
    solution_joint5: float | None,
    threshold: float,
) -> bool:
    """Return whether the SO101 IK solution crossed the configured joint5 branch boundary."""

    delta = joint5_branch_delta(seed_joint5, solution_joint5)
    if delta is None:
        return False
    return delta > float(threshold)


def joint5_branch_continuity_check(
    seed_joint5: float | None,
    solution_joint5: float | None,
    threshold: float = math.pi / 2.0,
) -> bool:
    """Return whether consecutive SO101 IK solutions remain on the same joint5 branch."""

    return not joint5_branch_filter_check(seed_joint5, solution_joint5, threshold)


def joint5_within_abs_limit(joint5: float | None, limit: float | None) -> bool:
    """Return whether joint5 is within the grasp-specific absolute execution limit."""

    if limit is None or joint5 is None:
        return True
    return abs(float(joint5)) <= float(limit)


@dataclass(frozen=True)
class Joint5RetryResult:
    """Outcome of retrying IK with a canonicalized SO101 joint5 seed."""

    joint_state: JointState
    original_joint5: float | None
    retried: bool
    retry_solution: JointState | None
    retry_joint5: float | None
    passed: bool


def apply_joint5_retry(
    *,
    joint_state: JointState,
    safety_limit: float | None,
    solve_ik: Callable[[JointState], JointState | None],
    joint_position: Callable[[JointState, str], float | None],
    joint_state_with_joint5: Callable[[JointState, float], JointState],
    flip_threshold: float = math.pi / 2.0,
) -> Joint5RetryResult:
    """Retry IK after canonicalizing a branch-flipped SO101 joint5 seed."""

    if safety_limit is None:
        return Joint5RetryResult(joint_state, None, False, None, None, True)

    original_joint5 = joint_position(joint_state, "5")
    if original_joint5 is None or abs(original_joint5) <= float(flip_threshold):
        return Joint5RetryResult(joint_state, None, False, None, None, True)

    retry_seed = joint_state_with_joint5(joint_state, canonicalize_joint5(original_joint5))
    retry_solution = solve_ik(retry_seed)
    if retry_solution is None:
        return Joint5RetryResult(joint_state, original_joint5, True, None, None, False)

    retry_joint5 = joint_position(retry_solution, "5")
    passed = retry_joint5 is not None and joint5_within_abs_limit(retry_joint5, safety_limit)
    return Joint5RetryResult(
        retry_solution if passed else joint_state,
        original_joint5,
        True,
        retry_solution,
        retry_joint5,
        passed,
    )
