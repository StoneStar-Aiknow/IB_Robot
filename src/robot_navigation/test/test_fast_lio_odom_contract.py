"""Pure contracts for adapting official FAST-LIO odometry to robot frames."""

import math

import pytest

from robot_navigation.fast_lio_odom_contract import (
    compose_pose,
    project_planar_pose,
    transform_twist,
    validate_odometry_sample,
    validate_odometry_timestamp,
)


def test_official_fast_lio_frames_are_accepted():
    assert validate_odometry_sample("camera_init", "body", [0.0, 1.0, -2.0, 1.0]) is None


def test_unexpected_fast_lio_frames_are_rejected():
    assert (
        validate_odometry_sample("odom", "base_link", [0.0, 0.0, 0.0, 1.0])
        == "expected camera_init -> body, got odom -> base_link"
    )


def test_non_finite_fast_lio_sample_is_rejected():
    assert (
        validate_odometry_sample("camera_init", "body", [0.0, float("nan"), 0.0, 1.0])
        == "odometry contains non-finite values"
    )


def test_zero_fast_lio_timestamp_is_rejected():
    assert validate_odometry_timestamp(0, 0, now_sec=10.0) == "odometry timestamp is zero"


def test_fast_lio_timestamp_far_in_the_future_is_rejected():
    assert (
        validate_odometry_timestamp(12, 0, now_sec=10.0, max_future_skew_sec=0.1)
        == "odometry timestamp is too far in the future"
    )


def test_body_to_base_extrinsic_is_composed_with_lio_pose():
    position, orientation = compose_pose(
        (1.0, 2.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, -0.3),
        (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)),
    )

    assert position == pytest.approx((1.0, 2.0, -0.3))
    assert orientation == pytest.approx((0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)))


def test_planar_projection_removes_height_roll_and_pitch():
    position, orientation = project_planar_pose((1.2, -0.4, 0.35), (0.2, -0.1, 0.3, 0.9))

    assert position == (1.2, -0.4, 0.0)
    assert orientation[0:2] == (0.0, 0.0)
    assert math.sqrt(sum(value * value for value in orientation)) == pytest.approx(1.0)


def test_twist_includes_body_to_base_lever_arm_and_rotation():
    linear, angular = transform_twist(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
    )

    assert linear == pytest.approx((0.0, 1.0, 0.0))
    assert angular == pytest.approx((0.0, 0.0, 2.0))


@pytest.mark.parametrize(
    "rotation",
    [(0.0, 0.0, 0.0, 0.0), (0.0, float("inf"), 0.0, 1.0)],
)
def test_invalid_body_to_base_rotation_is_rejected(rotation):
    with pytest.raises(ValueError, match="quaternion must be finite and non-zero"):
        compose_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), rotation)
