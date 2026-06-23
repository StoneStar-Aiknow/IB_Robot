"""Tests for the URDF-based LeRobot joint conversion helper."""

import pytest

from robot_config.utils import build_joint_conversion_table_from_urdf


def test_urdf_limits_build_range_norm_table():
    urdf = """
    <robot name="test">
      <joint name="1" type="revolute"><limit lower="-2.0" upper="2.0"/></joint>
      <joint name="6" type="revolute"><limit lower="0.0" upper="1.0"/></joint>
    </robot>
    """

    table = build_joint_conversion_table_from_urdf(
        urdf,
        joint_names=["1", "6"],
        gripper_joints=["6"],
        norm_mode="range_m100_100",
    )

    assert table == [(-2.0, 2.0, 200.0, -100.0), (0.0, 1.0, 100.0, 0.0)]


def test_urdf_limits_build_degrees_norm_table_aligned_with_calibration():
    """In degrees mode the span must match the tick-based calibration formula.

    Real-hardware calibration uses ``span = (tick_max - tick_min) * 360 / 4095``.
    Because URDF radians are produced via ``rad = (tick - 2048) * 2π / 4096``,
    the URDF helper must scale ``math.degrees(rad_max - rad_min)`` by
    ``4096/4095`` so sim and real tables stay numerically identical.
    """
    urdf = """
    <robot name="test">
      <joint name="1" type="revolute"><limit lower="-1.57079632679" upper="1.57079632679"/></joint>
    </robot>
    """

    (rad_min, rad_max, span, offset) = build_joint_conversion_table_from_urdf(
        urdf,
        joint_names=["1"],
        norm_mode="degrees",
    )[0]

    expected_span = 180.0 * 4096.0 / 4095.0
    assert rad_min == pytest.approx(-1.57079632679)
    assert rad_max == pytest.approx(1.57079632679)
    assert span == pytest.approx(expected_span)
    assert offset == pytest.approx(-expected_span / 2.0)


def test_urdf_limits_fall_back_to_ros2_control_command_limits():
    urdf = """
    <robot name="test">
      <ros2_control name="RobotSystem" type="system">
        <joint name="1">
          <command_interface name="position">
            <param name="min">-0.5</param>
            <param name="max">0.5</param>
          </command_interface>
        </joint>
      </ros2_control>
    </robot>
    """

    table = build_joint_conversion_table_from_urdf(
        urdf,
        joint_names=["1"],
        norm_mode="range_m100_100",
    )

    assert table == [(-0.5, 0.5, 200.0, -100.0)]
