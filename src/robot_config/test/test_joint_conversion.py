"""Tests for LeRobot joint conversion helpers."""

import json
import os

import pytest

from robot_config.utils import (
    build_joint_conversion_table,
    build_joint_conversion_table_from_calibration,
    build_lerobot_conversion_metadata,
    load_calibration_data,
    resolve_calibration_paths_from_config,
    resolve_gripper_joints_from_config,
)


def test_degrees_norm_mode_keeps_gripper_range_0_100_semantics():
    """Test degrees mode preserves RANGE_0_100 for configured gripper joints."""
    calibration = {
        "1": {"range_min": 1024, "range_max": 3072},
        "6": {"range_min": 1200, "range_max": 2200},
    }

    table = build_joint_conversion_table_from_calibration(
        calibration,
        joint_names=["1", "6"],
        gripper_joints=["6"],
        norm_mode="degrees",
    )

    arm_entry, gripper_entry = table

    assert arm_entry[2] == pytest.approx(2048 * 360.0 / 4095.0)
    assert arm_entry[3] == pytest.approx(-1024 * 360.0 / 4095.0)
    assert gripper_entry[2:] == (100.0, 0.0)


def test_degrees_norm_mode_maps_gripper_actions_to_calibration_ends():
    """Test gripper action 0/100 map to the calibrated closed/open endpoints."""
    calibration = {"6": {"range_min": 1200, "range_max": 2200}}
    (rad_min, rad_max, span, offset) = build_joint_conversion_table_from_calibration(
        calibration,
        joint_names=["6"],
        gripper_joints=["6"],
        norm_mode="degrees",
    )[0]

    action_zero_rad = (0.0 - offset) / span * (rad_max - rad_min) + rad_min
    action_full_rad = (100.0 - offset) / span * (rad_max - rad_min) + rad_min

    assert action_zero_rad == pytest.approx(rad_min)
    assert action_full_rad == pytest.approx(rad_max)


def test_dual_arm_xacro_calibrations_resolve_in_numeric_order(tmp_path):
    """Dual-arm configs reuse the same calibration files passed to xacro."""
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    robot_config = {
        "joints": {
            "gripper": ["joint6_left"],
            "left_gripper": ["joint6_left"],
            "right_gripper": ["joint6_right"],
        },
        "ros2_control": {
            "xacro_args": {
                "calib_file_2": str(right),
                "calib_file_1": str(left),
            }
        },
    }

    assert resolve_calibration_paths_from_config(robot_config) == [str(left), str(right)]
    assert resolve_gripper_joints_from_config(robot_config) == ["joint6_left", "joint6_right"]


def test_multi_source_calibration_maps_numeric_keys_to_dual_arm_joints(tmp_path):
    """Per-arm SO-101 calibration files are namespaced for dual-arm joints."""
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(
        json.dumps(
            {
                "1": {"range_min": 1000, "range_max": 3000},
                "6": {"range_min": 1100, "range_max": 2100},
            }
        )
    )
    right.write_text(
        json.dumps(
            {
                "1": {"range_min": 1200, "range_max": 3200},
                "6": {"range_min": 1300, "range_max": 2300},
            }
        )
    )

    sources = os.pathsep.join([str(left), str(right)])
    calibration = load_calibration_data(sources)
    assert calibration["joint1_left"]["range_min"] == 1000
    assert calibration["joint6_left"]["range_min"] == 1100
    assert calibration["joint1_right"]["range_min"] == 1200
    assert calibration["joint6_right"]["range_max"] == 2300

    table = build_joint_conversion_table(
        sources,
        joint_names=["joint1_left", "joint6_left", "joint1_right", "joint6_right"],
        gripper_joints=["joint6_left", "joint6_right"],
    )
    assert len(table) == 4
    assert table[1][2:] == (100.0, 0.0)
    assert table[3][2:] == (100.0, 0.0)

    metadata = build_lerobot_conversion_metadata(
        sources,
        joint_names=["joint1_left", "joint6_left", "joint1_right", "joint6_right"],
        gripper_joints=["joint6_left", "joint6_right"],
    )
    assert metadata["calibration_source"] == str(left.resolve())
    assert metadata["calibration_sources"] == [str(left.resolve()), str(right.resolve())]
    assert set(metadata["calibration"]) == {
        "joint1_left",
        "joint6_left",
        "joint1_right",
        "joint6_right",
    }
