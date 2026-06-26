"""Tests for LeRobot joint conversion helpers."""

import json
import os

import pytest

from robot_config.config import RobotConfig, Ros2ControlConfig
from robot_config.loader import validate_config
from robot_config.utils import (
    build_joint_conversion_table,
    build_joint_conversion_table_from_calibration,
    build_lerobot_conversion_metadata,
    load_calibration_data,
    resolve_calibration_paths_from_config,
    resolve_calibration_source_specs_from_config,
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


def test_dual_arm_xacro_calibrations_resolve_in_namespace_order(tmp_path):
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
                "calib_file_right": str(right),
                "calib_file_left": str(left),
            }
        },
    }

    assert resolve_calibration_paths_from_config(robot_config) == [str(left), str(right)]
    assert resolve_gripper_joints_from_config(robot_config) == ["joint6_left", "joint6_right"]


def test_named_xacro_calibrations_load_namespaced_numeric_keys(tmp_path):
    front = tmp_path / "front.json"
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    for index, path in enumerate((front, left, right), start=1):
        path.write_text(json.dumps({"1": {"range_min": 1000 + index, "range_max": 3000 + index}}))

    robot_config = {
        "ros2_control": {
            "xacro_args": {
                "calib_file_front": str(front),
                "calib_file_left": str(left),
                "calib_file_right": str(right),
            }
        }
    }

    specs = resolve_calibration_source_specs_from_config(robot_config)
    assert [(spec.resolved_path, spec.namespace) for spec in specs] == [
        (str(front), "front"),
        (str(left), "left"),
        (str(right), "right"),
    ]
    assert resolve_calibration_paths_from_config(robot_config) == [str(front), str(left), str(right)]

    calibration = load_calibration_data(specs)
    assert calibration["joint1_front"]["range_min"] == 1001
    assert calibration["joint1_left"]["range_min"] == 1002
    assert calibration["joint1_right"]["range_min"] == 1003
    assert "1" not in calibration

    table = build_joint_conversion_table(
        specs,
        joint_names=["joint1_front", "joint1_left", "joint1_right"],
    )
    assert len(table) == 3

    metadata = build_lerobot_conversion_metadata(
        specs,
        joint_names=["joint1_front", "joint1_left", "joint1_right"],
    )
    assert metadata["calibration_sources"] == [str(front.resolve()), str(left.resolve()), str(right.resolve())]
    assert set(metadata["calibration"]) == {"joint1_front", "joint1_left", "joint1_right"}


def test_numeric_xacro_calibration_namespace_is_supported(tmp_path):
    calib = tmp_path / "numeric.json"
    calib.write_text(json.dumps({"1": {"range_min": 1001, "range_max": 3001}}))
    robot_config = {
        "ros2_control": {
            "xacro_args": {
                "calib_file_1": str(calib),
            }
        }
    }

    specs = resolve_calibration_source_specs_from_config(robot_config)

    assert [(spec.resolved_path, spec.namespace) for spec in specs] == [(str(calib), "1")]
    assert load_calibration_data(specs)["joint1_1"]["range_min"] == 1001


def test_explicit_calibration_merge_key_collision_raises(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"serial": {"range_min": 1000, "range_max": 3000}}))
    second.write_text(json.dumps({"serial": {"range_min": 1200, "range_max": 3200}}))

    specs = [
        {"file": str(first), "namespace": "left"},
        {"file": str(second), "namespace": "right"},
    ]

    with pytest.raises(ValueError, match="Calibration key collision for 'serial'"):
        _ = load_calibration_data(specs)


def test_calibration_namespaces_all_numeric_keys(tmp_path):
    calib = tmp_path / "seven_axis.json"
    calib.write_text(
        json.dumps({"1": {"range_min": 1000, "range_max": 3000}, "7": {"range_min": 1100, "range_max": 3100}})
    )

    calibration = load_calibration_data({"file": str(calib), "namespace": "arm"})

    assert calibration["joint1_arm"]["range_min"] == 1000
    assert calibration["joint7_arm"]["range_min"] == 1100
    assert "1" not in calibration
    assert "7" not in calibration


def test_mixed_explicit_specs_and_legacy_paths_raise(tmp_path):
    calib = tmp_path / "single.json"
    specs = [{"file": str(calib), "namespace": "arm"}, str(calib)]

    with pytest.raises(TypeError, match="must not mix explicit specs"):
        _ = load_calibration_data(specs)


def test_legacy_single_calib_file_resolves_arm_source(tmp_path):
    calib = tmp_path / "single.json"
    robot_config = {"ros2_control": {"calib_file": str(calib)}}

    specs = resolve_calibration_source_specs_from_config(robot_config)

    assert [(spec.resolved_path, spec.namespace) for spec in specs] == [(str(calib), "arm")]
    assert resolve_calibration_paths_from_config(robot_config) == [str(calib)]


def test_calib_file_cannot_mix_with_named_xacro_calibrations(tmp_path):
    legacy = tmp_path / "single.json"
    left = tmp_path / "left.json"
    robot_config = {
        "ros2_control": {
            "calib_file": str(legacy),
            "xacro_args": {
                "calib_file_left": str(left),
            },
        }
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        _ = resolve_calibration_source_specs_from_config(robot_config)


def test_named_xacro_calibrations_support_more_than_two_sources(tmp_path):
    paths = [tmp_path / f"source_{index}.json" for index in range(3)]
    robot_config = {
        "ros2_control": {
            "xacro_args": {
                "calib_file_front": str(paths[0]),
                "calib_file_left": str(paths[1]),
                "calib_file_right": str(paths[2]),
            }
        }
    }

    specs = resolve_calibration_source_specs_from_config(robot_config)

    assert [(spec.resolved_path, spec.namespace) for spec in specs] == [
        (str(paths[0]), "front"),
        (str(paths[1]), "left"),
        (str(paths[2]), "right"),
    ]


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
    assert "1" not in calibration
    assert "6" not in calibration

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


def test_single_source_calibration_maps_numeric_keys_to_arm_joints(tmp_path):
    calib = tmp_path / "single.json"
    calib.write_text(
        json.dumps(
            {
                "1": {"range_min": 1000, "range_max": 3000},
                "6": {"range_min": 1100, "range_max": 2100},
            }
        )
    )

    calibration = load_calibration_data(str(calib))

    assert calibration == {
        "joint1_arm": {"range_min": 1000, "range_max": 3000},
        "joint6_arm": {"range_min": 1100, "range_max": 2100},
    }
    assert "1" not in calibration
    assert "6" not in calibration

    table = build_joint_conversion_table(
        str(calib),
        joint_names=["1", "6"],
        gripper_joints=["6"],
    )
    assert len(table) == 2
    assert table[1][2:] == (100.0, 0.0)

    metadata = build_lerobot_conversion_metadata(
        str(calib),
        joint_names=["1", "6"],
        gripper_joints=["6"],
    )
    assert set(metadata["calibration"]) == {"1", "6"}
    assert metadata["calibration"]["1"]["range_min"] == 1000
    assert metadata["calibration"]["6"]["range_max"] == 2100


def test_dual_source_numeric_joint_request_remains_ambiguous(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"1": {"range_min": 1000, "range_max": 3000}}))
    right.write_text(json.dumps({"1": {"range_min": 1200, "range_max": 3200}}))
    sources = os.pathsep.join([str(left), str(right)])

    with pytest.raises(KeyError, match="Joint '1' missing"):
        _ = build_joint_conversion_table(sources, joint_names=["1"])

    with pytest.raises(KeyError, match="Joint '1' missing"):
        _ = build_lerobot_conversion_metadata(sources, joint_names=["1"])


def test_single_source_leading_zero_joint_request_does_not_match_arm_namespace(tmp_path):
    calib = tmp_path / "single.json"
    calib.write_text(json.dumps({"1": {"range_min": 1000, "range_max": 3000}}))

    with pytest.raises(KeyError, match="Joint '01' missing"):
        _ = build_joint_conversion_table(str(calib), joint_names=["01"])

    with pytest.raises(KeyError, match="Joint '01' missing"):
        _ = build_lerobot_conversion_metadata(str(calib), joint_names=["01"])


def test_multi_source_calibration_rejects_more_than_two_sources(tmp_path):
    paths = []
    for index in range(3):
        calib = tmp_path / f"source_{index}.json"
        calib.write_text(json.dumps({"1": {"range_min": 1000 + index, "range_max": 3000 + index}}))
        paths.append(str(calib))

    with pytest.raises(ValueError, match="use explicit calibration source specs"):
        _ = load_calibration_data(os.pathsep.join(paths))


def test_validate_config_checks_named_calibration_files(tmp_path):
    missing = tmp_path / "missing.json"
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={
                "xacro_args": {
                    "calib_file_left": str(missing),
                }
            },
            urdf_path="",
        ),
    )

    errors = validate_config(config)

    assert errors == [f"Calibration file not found: {missing}"]


def test_validate_config_reports_invalid_mixed_calibration_sources(tmp_path):
    legacy = tmp_path / "single.json"
    left = tmp_path / "left.json"
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={
                "calib_file": str(legacy),
                "xacro_args": {
                    "calib_file_left": str(left),
                },
            },
            urdf_path="",
        ),
    )

    errors = validate_config(config)

    assert errors == [
        "Invalid ros2_control calibration configuration: "
        "ros2_control.calib_file cannot be combined with ros2_control.xacro_args.calib_file_<namespace>"
    ]
