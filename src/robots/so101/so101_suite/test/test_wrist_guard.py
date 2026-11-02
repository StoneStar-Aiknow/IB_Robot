import math

import numpy as np
import pytest
from sensor_msgs.msg import JointState

from so101_suite.wrist_guard import (
    apply_joint5_retry,
    canonicalize_joint5,
    joint5_branch_continuity_check,
    joint5_branch_delta,
    joint5_branch_filter_check,
    joint5_closing_axis_correction,
    joint5_within_abs_limit,
)


# Local test helpers (D4: so101_suite is a dependency leaf and must not import
# manipulation_execution even from its tests). Same semantics the suite's
# modules implement: Rz @ Ry @ Rx for intrinsic XYZ angles, quaternion [x, y,
# z, w] from a rotation matrix.
def euler_xyz_matrix(rpy):
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(matrix):
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(float(value / norm) for value in quaternion)


def _joint_state(joint5: float) -> JointState:
    state = JointState()
    state.name = ["1", "2", "3", "4", "5"]
    state.position = [0.0, 0.0, 0.0, 0.0, joint5]
    return state


def _joint_position(joint_state: JointState, joint_name: str) -> float | None:
    positions = dict(zip(joint_state.name, joint_state.position, strict=False))
    value = positions.get(joint_name)
    return None if value is None else float(value)


def _joint_state_with_joint5(joint_state: JointState, joint5: float) -> JointState:
    seed = JointState()
    seed.name = list(joint_state.name)
    seed.position = [
        float(joint5) if str(name) == "5" else float(position)
        for name, position in zip(joint_state.name, joint_state.position, strict=False)
    ]
    return seed


def test_joint5_helpers_select_equivalent_closing_axis_branch():
    target = quaternion_from_matrix(euler_xyz_matrix((0.0, 0.0, math.pi / 3.0)))
    actual = quaternion_from_matrix(np.eye(3, dtype=np.float64))

    correction = joint5_closing_axis_correction(
        target,
        actual,
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        closing_axis_180_symmetric=True,
    )

    assert math.isclose(correction, math.pi / 3.0, abs_tol=1e-8)
    assert math.isclose(canonicalize_joint5(2.0 * math.pi / 3.0), -math.pi / 3.0, abs_tol=1e-8)
    assert math.isclose(canonicalize_joint5(1.2, center=0.5), 1.2, abs_tol=1e-8)
    assert math.isclose(canonicalize_joint5(2.2, center=0.5), 2.2 - math.pi, abs_tol=1e-8)


def test_closing_axis_correction_normalizes_non_unit_quaternions():
    # Legacy semantics (the deleted manipulation_execution.so101_kinematics_guard
    # via grasp_geometry.quaternion_matrix): a scaled quaternion describes the
    # same rotation. Identity actual orientation, Z approach axis, X closing
    # axis and the (0, 0, sqrt(2), sqrt(2)) target must correct joint5 by
    # exactly pi/2, as the normalised (0, 0, sqrt(0.5), sqrt(0.5)) does.
    scaled = (0.0, 0.0, math.sqrt(2.0), math.sqrt(2.0))
    normalized = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    identity = quaternion_from_matrix(np.eye(3, dtype=np.float64))

    for target in (scaled, normalized):
        correction = joint5_closing_axis_correction(
            target,
            identity,
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        assert math.isclose(correction, math.pi / 2.0, abs_tol=1e-9)

    # The actual orientation is normalised with the same rule: a scaled actual
    # quaternion gives the same correction as its normalised form.
    assert math.isclose(
        joint5_closing_axis_correction(
            normalized,
            scaled,
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        joint5_closing_axis_correction(
            normalized,
            normalized,
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        abs_tol=1e-9,
    )


@pytest.mark.parametrize("zero_is_target", [True, False])
def test_closing_axis_correction_rejects_zero_quaternions(zero_is_target):
    zero = (0.0, 0.0, 0.0, 0.0)
    identity = quaternion_from_matrix(np.eye(3, dtype=np.float64))
    target, actual = (zero, identity) if zero_is_target else (identity, zero)

    with pytest.raises(ValueError, match="quaternion must be non-zero"):
        joint5_closing_axis_correction(
            target,
            actual,
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("seed", "solution", "expected"),
    [
        (0.0, math.pi, math.pi),
        (0.5, 0.5, 0.0),
        (None, 1.0, None),
        (1.0, None, None),
    ],
)
def test_joint5_branch_delta(seed, solution, expected):
    result = joint5_branch_delta(seed, solution)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("seed", "solution", "threshold", "expected"),
    [
        (0.0, math.pi, math.pi / 2.0, True),
        (0.0, 0.5, math.pi / 2.0, False),
        (None, 1.0, math.pi / 2.0, False),
        (1.0, None, math.pi / 2.0, False),
        (0.0, 0.1, 0.05, True),
        (0.0, 0.04, 0.05, False),
    ],
)
def test_joint5_branch_filter_check(seed, solution, threshold, expected):
    assert joint5_branch_filter_check(seed, solution, threshold) is expected


def test_joint5_branch_continuity_uses_same_branch_policy():
    assert joint5_branch_continuity_check(0.0, 0.5)
    assert not joint5_branch_continuity_check(0.0, math.pi)
    assert joint5_branch_continuity_check(None, 1.0)
    assert not joint5_branch_continuity_check(0.0, math.pi / 2.0 + 0.01)


@pytest.mark.parametrize(
    ("joint5", "limit", "expected"),
    [(0.5, 1.0, True), (1.5, 1.0, False), (100.0, None, True), (None, 1.0, True)],
)
def test_joint5_within_abs_limit(joint5, limit, expected):
    assert joint5_within_abs_limit(joint5, limit) is expected


def test_joint5_home_limit_allows_epsilon_at_configured_2_rad():
    limit = 2.0

    assert joint5_within_abs_limit(limit + 0.0005, limit, epsilon=0.001)
    assert joint5_within_abs_limit(-limit - 0.0005, limit, epsilon=0.001)
    assert not joint5_within_abs_limit(limit + 0.002, limit, epsilon=0.001)


def test_joint5_home_limit_uses_configured_center():
    assert joint5_within_abs_limit(0.6, 0.2, center=0.5)
    assert not joint5_within_abs_limit(0.8, 0.2, center=0.5)


def test_apply_joint5_retry_skips_when_guard_is_inactive_or_branch_is_already_bounded():
    solution = _joint_state(0.5)

    disabled = apply_joint5_retry(
        joint_state=solution,
        safety_limit=None,
        solve_ik=lambda _seed: pytest.fail("should not retry"),
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
    )
    bounded = apply_joint5_retry(
        joint_state=solution,
        safety_limit=2.0,
        solve_ik=lambda _seed: pytest.fail("should not retry"),
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
    )

    assert not disabled.retried and disabled.passed
    assert not bounded.retried and bounded.passed
    assert disabled.joint_state is solution
    assert bounded.joint_state is solution


def test_apply_joint5_retry_skips_when_joint5_is_missing():
    solution = JointState()
    solution.name = ["1", "2", "3", "4"]
    solution.position = [0.0, 0.0, 0.0, 0.0]

    result = apply_joint5_retry(
        joint_state=solution,
        safety_limit=2.0,
        solve_ik=lambda _seed: pytest.fail("should not retry"),
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
    )

    assert not result.retried
    assert result.passed


def test_apply_joint5_retry_uses_canonicalized_seed():
    original_joint5 = 2.0
    expected_seed = canonicalize_joint5(original_joint5)
    retry_solution = _joint_state(expected_seed)
    calls = []

    def solve_ik(seed):
        calls.append(seed)
        return retry_solution

    result = apply_joint5_retry(
        joint_state=_joint_state(original_joint5),
        safety_limit=2.0,
        solve_ik=solve_ik,
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
    )

    assert result.retried and result.passed
    assert result.original_joint5 == pytest.approx(original_joint5)
    assert result.retry_joint5 == pytest.approx(expected_seed)
    assert result.joint_state is retry_solution
    assert _joint_position(calls[0], "5") == pytest.approx(expected_seed)


@pytest.mark.parametrize(
    ("retry_solution", "expected_joint5"),
    [(None, None), (_joint_state(1.5), 1.5)],
)
def test_apply_joint5_retry_reports_failed_retry(retry_solution, expected_joint5):
    result = apply_joint5_retry(
        joint_state=_joint_state(2.0),
        safety_limit=1.0,
        solve_ik=lambda _seed: retry_solution,
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
    )

    assert result.retried
    assert not result.passed
    assert result.retry_joint5 == expected_joint5


def test_apply_joint5_retry_respects_custom_flip_threshold():
    skipped = apply_joint5_retry(
        joint_state=_joint_state(0.1),
        safety_limit=2.0,
        solve_ik=lambda _seed: pytest.fail("should not retry"),
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
        flip_threshold=0.2,
    )
    retried = apply_joint5_retry(
        joint_state=_joint_state(0.3),
        safety_limit=2.0,
        solve_ik=lambda seed: seed,
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
        flip_threshold=0.2,
    )

    assert not skipped.retried
    assert retried.retried and retried.passed


def test_apply_joint5_retry_does_not_retry_exact_limit_plus_epsilon_boundary():
    center = 0.4
    limit = 2.0
    epsilon = 0.001
    solution = _joint_state(center + limit + epsilon)

    result = apply_joint5_retry(
        joint_state=solution,
        safety_limit=limit,
        solve_ik=lambda _seed: pytest.fail("accepted boundary must not retry"),
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
        flip_threshold=limit + epsilon,
        safety_center=center,
        safety_epsilon=epsilon,
    )

    assert not result.retried
    assert result.passed


def test_apply_joint5_retry_canonicalizes_around_home_center():
    result = apply_joint5_retry(
        joint_state=_joint_state(2.2),
        safety_limit=math.pi / 2.0,
        solve_ik=lambda seed: seed,
        joint_position=_joint_position,
        joint_state_with_joint5=_joint_state_with_joint5,
        safety_center=0.5,
    )

    assert result.retried and result.passed
    assert result.retry_joint5 == pytest.approx(2.2 - math.pi)
