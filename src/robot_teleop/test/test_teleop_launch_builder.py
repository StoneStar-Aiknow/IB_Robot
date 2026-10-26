"""Launch-builder tests for the teleop entry (robot_teleop's contribution).

Migrated from ``robot_config/test/test_launch_readiness.py`` (the teleop node
generation and config-validation cases) and ``robot_config/test/
test_config_overlay.py`` (the legacy mhandpro rejection) — openspec change
``migrate-business-launch-builders``, task B.3/B.9: the tests follow the
teleop module to its owning package. Assertions are verbatim; robot_config
assets load through the installed share directory and the so101_motion
source-tree path keeps the same depth (``src/<pkg>/test`` -> ``src/``).
"""

import json
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_teleop.launch_builders.teleop import generate_teleop_nodes, validate_teleop_config

_HAND_JOINTS = [f"hand_{index}" for index in range(7)]


def _joint_limits(names):
    return {name: {"min": -1.0, "max": 1.0} for name in names}


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _node_parameters(node):
    def decode_parameter(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode_parameter(tuple(item)) for item in value]
        if all(isinstance(item, bool | int | float) for item in value):
            return list(value)
        text = _text(value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text.strip()

    raw = node._Node__parameters[0]
    parsed = {}
    for key, value in raw.items():
        name = _text(key)
        parsed[name] = decode_parameter(value)
    return parsed


def test_generate_teleop_nodes_injects_target_joint_names_into_device_config():
    nodes = generate_teleop_nodes(
        {
            "joints": {
                "arm": ["1", "2", "3", "4", "5"],
                "gripper": ["6"],
            },
            "teleoperation": {
                "enabled": True,
                "active_device": "left_leader",
                "devices": [
                    {
                        "name": "left_leader",
                        "type": "leader_arm",
                        "port": "/dev/ttyACM0",
                        "target": {
                            "arm_joint_names": ["joint1_left", "joint2_left"],
                            "gripper_joint_names": ["joint6_left"],
                        },
                    }
                ],
                "safety": {
                    "joint_limits": {
                        "joint1_left": {"min": -1.0, "max": 1.0},
                        "joint2_left": {"min": -1.0, "max": 1.0},
                        "joint6_left": {"min": 0.0, "max": 1.0},
                    }
                },
            },
        }
    )

    params = _node_parameters(nodes[0])
    device_config = json.loads(params["device_config"].strip("'"))

    assert device_config["arm_joint_names"] == ["joint1_left", "joint2_left"]
    assert device_config["gripper_joint_names"] == ["joint6_left"]

    publish_groups = json.loads(params["publish_groups"].strip("'"))
    assert publish_groups == [
        {
            "name": "arm",
            "joint_names": ["joint1_left", "joint2_left"],
            "topic": "/arm_position_controller/commands",
        },
        {
            "name": "gripper",
            "joint_names": ["joint6_left"],
            "topic": "/gripper_position_controller/commands",
        },
    ]


def test_dual_arm_legacy_targets_keep_topics_and_joint_order(tmp_path):
    config_path = Path(get_package_share_directory("robot_config")) / "config" / "robots" / "so101_dual_arm.yaml"
    robot_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    calibration = tmp_path / "leader.json"
    calibration.write_text("{}\n", encoding="utf-8")
    for device in robot_config["teleoperation"]["devices"]:
        device["calib_file"] = str(calibration)

    nodes = generate_teleop_nodes(robot_config)

    assert len(nodes) == 2
    expected = [
        (
            ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left"],
            ["joint6_left"],
            "/arm_position_controller_left/commands",
            "/gripper_position_controller_left/commands",
        ),
        (
            ["joint1_right", "joint2_right", "joint3_right", "joint4_right", "joint5_right"],
            ["joint6_right"],
            "/arm_position_controller_right/commands",
            "/gripper_position_controller_right/commands",
        ),
    ]
    for node, (arm_joints, gripper_joints, arm_topic, gripper_topic) in zip(nodes, expected, strict=True):
        params = _node_parameters(node)
        assert params["arm_joint_names"] == arm_joints
        assert params["gripper_joint_names"] == gripper_joints
        assert params["arm_command_topic"] == arm_topic
        assert params["gripper_command_topic"] == gripper_topic
        assert json.loads(params["publish_groups"].strip("'")) == [
            {"name": "arm", "joint_names": arm_joints, "topic": arm_topic},
            {"name": "gripper", "joint_names": gripper_joints, "topic": gripper_topic},
        ]


def test_required_teleop_process_exits_shutdown_launch():
    nodes = generate_teleop_nodes(
        {
            "teleoperation": {
                "enabled": True,
                "active_device": "leader",
                "devices": [
                    {
                        "name": "leader",
                        "type": "leader_arm",
                        "port": "/dev/leader",
                        "target": {"arm_joint_names": ["1"], "gripper_joint_names": ["6"]},
                    }
                ],
            },
            "joints": {"arm": ["1"], "gripper": ["6"]},
            "safety": {"joint_limits": _joint_limits(["1", "6"])},
        },
        {},
    )

    assert nodes[0]._ExecuteLocal__on_exit is not None


def test_hand_retarget_resolves_output_contract_from_auxiliary_actuator():
    joints = ["thumb", "index", "middle"]
    nodes = generate_teleop_nodes(
        {
            "joints": {"arm": ["1"], "gripper": ["6"]},
            "auxiliary_actuators": {
                "amazing_hand_right": {
                    "type": "amazing_hand",
                    "mock": True,
                    "driver": {"package": "amazing_hand_hardware", "executable": "amazing_hand_node"},
                    "joint_names": joints,
                    "command_topic": "/amazing_hand_right/commands",
                    "joint_state_topic": "/amazing_hand_right/joint_states",
                }
            },
            "teleoperation": {
                "enabled": True,
                "active_device": "amazing_retarget",
                "safety": {"joint_limits": _joint_limits(joints)},
                "devices": [
                    {
                        "name": "amazing_retarget",
                        "type": "hand_retarget",
                        "side": "right",
                        "source_topic": "/hand_sources/mhandpro/right/state",
                        "retargeter": {
                            "type": "synergy_matrix",
                            "input_features": ["index_mcp_flex"],
                            "matrix": [[1.0], [1.0], [1.0]],
                        },
                        "target": {"actuator": "amazing_hand_right"},
                    }
                ],
            },
        }
    )

    params = _node_parameters(nodes[0])
    assert json.loads(params["publish_groups"].strip("'")) == [
        {"name": "hand", "joint_names": joints, "topic": "/amazing_hand_right/commands"}
    ]
    device_config = json.loads(params["device_config"].strip("'"))
    assert device_config["joint_names"] == joints


def test_phone_placo_uses_explicit_arm_group_topic():
    nodes = generate_teleop_nodes(
        {
            "joints": {"arm": ["1", "2"], "gripper": ["6"]},
            "moveit": {
                "base_link": "base",
                "ee_link": "gripper",
                # Source-tree path (same precedent as test_phone_teleop_config.py
                # and test_teleop_home_action.py): $(find so101_motion) would
                # make this test require so101_motion's install for a YAML that
                # is part of the same source checkout.
                "so101_placo_servo_config_path": str(
                    Path(__file__).resolve().parents[2]
                    / "robots"
                    / "so101"
                    / "so101_motion"
                    / "config"
                    / "so101_placo_servo.yaml"
                ),
            },
            "ros2_control": {"reset_positions": {"1": 0.0, "2": 0.0}},
            "teleoperation": {
                "enabled": True,
                "active_device": "phone",
                "cartesian": {"solver": "placo_servo"},
                "safety": {"joint_limits": _joint_limits(["1", "2", "6"])},
                "devices": [
                    {
                        "name": "phone",
                        "type": "phone",
                        "phone_config": {
                            "backend": "webphone",
                            "end_effector_bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
                        },
                        "target": {
                            "publish_groups": [
                                {"name": "arm", "joint_names": ["1", "2"], "topic": "/custom/arm"},
                                {"name": "gripper", "joint_names": ["6"], "topic": "/custom/gripper"},
                            ]
                        },
                    }
                ],
            },
        }
    )

    teleop_params = _node_parameters(nodes[0])
    device_config = json.loads(teleop_params["device_config"].strip("'"))
    placo = device_config["cartesian_backend_config"]
    assert placo["command_lease_topic"] == "/so101_placo_servo_node/command_lease"
    placo_node = next(node for node in nodes if _text(node.node_executable) == "so101_placo_servo_node.py")
    assert _node_parameters(placo_node)["command_out_topic"] == "/custom/arm"


def test_synergy_hand_retarget_does_not_require_aero_calibration():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "amazing",
            "safety": {"joint_limits": _joint_limits(["thumb", "index", "middle"])},
            "devices": [
                {
                    "name": "amazing",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0], [1.0], [1.0]],
                    },
                    "target": {
                        "publish_groups": [
                            {
                                "name": "hand",
                                "joint_names": ["thumb", "index", "middle"],
                                "topic": "/amazing/commands",
                            }
                        ]
                    },
                }
            ],
        }
    )

    assert errors == []


def test_active_devices_allow_disjoint_arm_and_hand_topics():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["leader", "glove"],
            "safety": {"joint_limits": _joint_limits(["1", *_HAND_JOINTS])},
            "devices": [
                {"name": "leader", "type": "leader_arm", "port": "/dev/leader"},
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                    "target": {
                        "publish_groups": [
                            {"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}
                        ]
                    },
                },
            ],
        }
    )

    assert errors == []


def test_vr_and_glove_allow_disjoint_command_topics():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["vr", "glove"],
            "safety": {"joint_limits": _joint_limits(_HAND_JOINTS)},
            "devices": [
                {"name": "vr", "type": "vr_teleop", "vr_config": {"output_profile": "so101"}},
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                    "target": {
                        "publish_groups": [
                            {"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}
                        ]
                    },
                },
            ],
        }
    )

    assert errors == []


def test_two_gloves_reject_shared_aero_hand_topic():
    devices = []
    for name in ("first", "second"):
        devices.append(
            {
                "name": name,
                "type": "hand_retarget",
                "side": "right",
                "source_topic": f"/hands/{name}/state",
                "retargeter": {
                    "type": "synergy_matrix",
                    "input_features": ["index_mcp_flex"],
                    "matrix": [[1.0]] * 7,
                },
                "target": {
                    "publish_groups": [{"name": "hand", "joint_names": _HAND_JOINTS, "topic": "/aero_hand/commands"}]
                },
            }
        )

    errors = validate_teleop_config({"enabled": True, "active_devices": ["first", "second"], "devices": devices})

    assert any("share command topic" in error for error in errors)


def test_hand_retarget_requires_explicit_publish_group():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "glove",
            "devices": [
                {
                    "name": "glove",
                    "type": "hand_retarget",
                    "side": "right",
                    "source_topic": "/hands/right/state",
                    "retargeter": {
                        "type": "synergy_matrix",
                        "input_features": ["index_mcp_flex"],
                        "matrix": [[1.0]] * 7,
                    },
                }
            ],
        }
    )

    assert any("requires explicit target.publish_groups" in error for error in errors)


def test_vr_rejects_target_override_that_does_not_match_its_builder():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "vr",
            "devices": [
                {
                    "name": "vr",
                    "type": "vr_teleop",
                    "vr_config": {"output_profile": "so101"},
                    "target": {
                        "publish_groups": [{"name": "other", "joint_names": ["1"], "topic": "/not_the_vr_output"}]
                    },
                }
            ],
        }
    )

    assert any("vr_teleop uses vr_config outputs" in error for error in errors)


@pytest.mark.parametrize("second_type", ["leader_arm", "phone", "xbox_controller"])
def test_active_arm_devices_reject_shared_command_topics(second_type):
    second = {"name": "second", "type": second_type}
    if second_type == "leader_arm":
        second["port"] = "/dev/leader2"
    if second_type == "phone":
        second["phone_config"] = {"phone_os": "android"}

    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_devices": ["leader", "second"],
            "safety": {"joint_limits": _joint_limits(["1"])},
            "devices": [
                {"name": "leader", "type": "leader_arm", "port": "/dev/leader"},
                second,
            ],
        }
    )

    assert any("share arm command topic" in error for error in errors)


def test_generate_joy_teleop_nodes_for_mobile_base():
    nodes = generate_teleop_nodes(
        {
            "teleoperation": {
                "enabled": True,
                "active_device": "lekiwi_gamepad",
                "devices": [
                    {
                        "name": "lekiwi_gamepad",
                        "type": "joy_teleop",
                        "config_path": "$(find robot_config)/config/lekiwi/lekiwi_teleop.yaml",
                        "input_device": "/dev/input/js0",
                    }
                ],
            }
        }
    )

    assert len(nodes) == 2
    assert all(isinstance(node, Node) for node in nodes)


def test_legacy_mhandpro_glove_is_rejected_as_active_runtime_path() -> None:
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "legacy_glove",
            "devices": [{"name": "legacy_glove", "type": "mhandpro_glove"}],
        }
    )

    assert any("has been removed" in error for error in errors)
