import json
import sys
import types
from pathlib import Path

import pytest

_MODULE_NAMES = (
    "robot_config",
    "robot_config.launch_builders",
    "robot_config.launch_builders.teleop",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _MODULE_NAMES}
package = types.ModuleType("robot_config")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "robot_config")]
sys.modules["robot_config"] = package

from robot_config.launch_builders.teleop import (  # noqa: E402
    _generate_device_nodes,
    generate_teleop_nodes,
    validate_teleop_config,
)

_TELEOP_MODULE = sys.modules["robot_config.launch_builders.teleop"]
for module_name, original_module in _ORIGINAL_MODULES.items():
    if original_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = original_module


def _robot_config():
    placo_config = Path(__file__).resolve().parents[2] / "robot_moveit" / "config" / "so101_placo_servo.yaml"
    return {
        "name": "phone_test",
        "joints": {"arm": ["1", "2", "3", "4", "5"], "gripper": ["6"]},
        "ros2_control": {"reset_positions": {str(index): 0.0 for index in range(1, 7)}},
        "moveit": {
            "base_link": "base",
            "ee_link": "gripper",
            "so101_placo_servo_config_path": str(placo_config),
        },
        "teleoperation": {
            "enabled": True,
            "active_device": "phone",
            "cartesian": {"solver": "placo_servo"},
            "devices": [
                {
                    "name": "phone",
                    "type": "phone",
                    "control_frequency": 37.0,
                    "phone_config": {
                        "backend": "webphone",
                        "web": {"command_stale_s": 0.18, "tls": {"enabled": False}},
                    },
                }
            ],
            "safety": {"joint_limits": {"1": {"min": -1.0, "max": 1.0}}},
        },
    }


def test_launch_builder_preserves_phone_control_frequency(monkeypatch):
    config = _robot_config()
    placo_kwargs = {}

    class FakeNode:
        def __init__(self, **kwargs):
            self.parameters = kwargs.get("parameters", [])

    monkeypatch.setattr(_TELEOP_MODULE, "Node", FakeNode)

    def fake_placo(*_args, **kwargs):
        placo_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(_TELEOP_MODULE, "_create_so101_placo_servo_node", fake_placo)
    nodes = _generate_device_nodes(config, config["teleoperation"]["devices"][0], {})
    teleop_node = nodes[0]
    parameters = teleop_node.parameters[0]
    device_config = json.loads(parameters["device_config"])
    assert parameters["control_frequency"] == 37.0
    assert parameters["estop_topic"] == "/emergency_stop"
    assert device_config["control_frequency"] == 37.0
    assert "input_mode" not in device_config["phone_config"]
    assert device_config["cartesian_backend_config"]["pose_topic"] == ("/so101_placo_servo_node/pose_cmd_base")
    assert device_config["cartesian_backend_config"]["home_action"] == ("/so101_placo_servo_node/return_home")
    assert device_config["cartesian_backend_config"]["command_lease_topic"] == ("/so101_placo_servo_node/command_lease")
    assert placo_kwargs["input_mode"] == "pose"
    assert placo_kwargs["resolved_params"]["input_mode"] == "pose"
    assert placo_kwargs["resolved_params"]["command_lease_timeout_s"] == 0.18


def test_invalid_webphone_ports_are_reported():
    config = _robot_config()
    config["teleoperation"]["devices"][0]["phone_config"]["web"].update({"http_port": 8765, "websocket_port": 8765})
    errors = validate_teleop_config(config["teleoperation"])
    assert any("ports must differ" in error for error in errors)


def test_stop_request_latency_above_safety_bound_is_reported():
    config = _robot_config()
    device = config["teleoperation"]["devices"][0]
    device["control_frequency"] = 30.0
    device["phone_config"]["web"]["command_stale_s"] = 0.2

    errors = validate_teleop_config(config["teleoperation"])

    assert any("stop-request latency bound" in error for error in errors)


@pytest.mark.parametrize("control_frequency", [True, float("nan"), float("inf"), float("-inf")])
def test_non_finite_or_boolean_control_frequency_is_reported(control_frequency):
    config = _robot_config()
    config["teleoperation"]["devices"][0]["control_frequency"] = control_frequency

    errors = validate_teleop_config(config["teleoperation"])

    assert any("control_frequency must be finite and positive" in error for error in errors)


@pytest.mark.parametrize("stale_s", [True, float("nan"), float("inf"), float("-inf")])
def test_non_finite_or_boolean_command_stale_is_reported(stale_s):
    config = _robot_config()
    config["teleoperation"]["devices"][0]["phone_config"]["web"]["command_stale_s"] = stale_s

    errors = validate_teleop_config(config["teleoperation"])

    assert any("command_stale_s must be finite and positive" in error for error in errors)


def test_phone_rejects_one_sided_relative_bounds():
    config = _robot_config()
    phone_config = config["teleoperation"]["devices"][0]["phone_config"]
    phone_config["end_effector_bounds"] = {
        "min": [-0.5, -0.5, 0.0],
        "max": [0.5, 0.5, 0.5],
    }

    errors = validate_teleop_config(config["teleoperation"])

    assert any("min < 0 < max" in error for error in errors)


def test_removed_hebi_backend_is_rejected_before_launch():
    config = _robot_config()
    device = config["teleoperation"]["devices"][0]
    device["phone_config"]["backend"] = "hebi"

    errors = validate_teleop_config(config["teleoperation"])
    assert any("phone backend must be 'webphone'" in error for error in errors)
    with pytest.raises(ValueError, match="phone_config.backend must be 'webphone'"):
        _generate_device_nodes(config, device, {})


def test_legacy_phone_os_does_not_change_webphone_default():
    config = _robot_config()
    phone_config = config["teleoperation"]["devices"][0]["phone_config"]
    phone_config.pop("backend")
    phone_config["phone_os"] = "ios"

    assert not validate_teleop_config(config["teleoperation"])


def test_webphone_velocity_mode_is_rejected():
    config = _robot_config()
    phone_config = config["teleoperation"]["devices"][0]["phone_config"]
    phone_config["input_mode"] = "velocity"

    errors = validate_teleop_config(config["teleoperation"])

    assert any("input_mode='velocity' is not supported" in error for error in errors)


def test_launch_generation_fails_before_creating_nodes_for_invalid_webphone_config():
    config = _robot_config()
    config["teleoperation"]["devices"][0]["phone_config"]["web"].update({"http_port": 8765, "websocket_port": 8765})

    with pytest.raises(ValueError, match="WebPhone ports must differ"):
        generate_teleop_nodes(config, {})


def test_phone_rejects_moveit_servo_before_launch():
    config = _robot_config()
    config["teleoperation"]["cartesian"]["solver"] = "moveit_servo"
    device = config["teleoperation"]["devices"][0]
    with pytest.raises(ValueError, match="Phone teleoperation requires teleoperation.cartesian.solver=placo_servo"):
        _generate_device_nodes(config, device, {})


def test_placo_phone_launch_rejects_incomplete_joint_home_targets():
    config = _robot_config()
    device = config["teleoperation"]["devices"][0]
    config["ros2_control"]["reset_positions"].pop("5")

    with pytest.raises(ValueError, match="missing: 5"):
        _generate_device_nodes(config, device, {})


def test_placo_phone_launch_rejects_non_finite_joint_home_targets():
    config = _robot_config()
    device = config["teleoperation"]["devices"][0]
    config["ros2_control"]["reset_positions"]["3"] = float("nan")

    with pytest.raises(ValueError, match="invalid: 3"):
        _generate_device_nodes(config, device, {})


def test_validation_accepts_legacy_pose_mode():
    config = _robot_config()
    phone_config = config["teleoperation"]["devices"][0]["phone_config"]
    phone_config["input_mode"] = "pose"
    errors = validate_teleop_config(config["teleoperation"])
    assert not any("input_mode" in error for error in errors)


def test_validation_rejects_webphone_without_tracking_source():
    config = _robot_config()
    phone_config = config["teleoperation"]["devices"][0]["phone_config"]
    phone_config["optical_flow_fallback_enabled"] = False
    phone_config["web"]["ar_enabled"] = False

    errors = validate_teleop_config(config["teleoperation"])

    assert any("requires WebXR AR or optical-flow fallback" in error for error in errors)


def test_launch_builder_injects_target_and_solver_endpoints_into_both_sides(monkeypatch):
    config = _robot_config()
    device = config["teleoperation"]["devices"][0]
    device["target"] = {
        "arm_joint_names": ["joint1_left", "joint2_left"],
        "arm_command_topic": "/left_arm/commands",
    }
    config["ros2_control"]["reset_positions"].update({"joint1_left": 0.1, "joint2_left": -0.2})
    config["teleoperation"]["cartesian"]["placo_servo"] = {"position_only": True}

    class FakeNode:
        def __init__(self, **kwargs):
            self.parameters = kwargs.get("parameters", [])

    monkeypatch.setattr(_TELEOP_MODULE, "Node", FakeNode)
    nodes = _generate_device_nodes(config, device, {})
    teleop_params = nodes[0].parameters[0]
    device_config = json.loads(teleop_params["device_config"])
    placo_params = nodes[1].parameters[0]

    assert device_config["cartesian_backend_config"] == {
        "linear_topic": placo_params["linear_cmd_topic"],
        "angular_topic": placo_params["angular_cmd_topic"],
        "pose_topic": placo_params["pose_cmd_topic"],
        "start_srv": placo_params["start_service"],
        "stop_srv": placo_params["stop_service"],
        "home_action": placo_params["home_action"],
        "command_lease_topic": placo_params["command_lease_topic"],
    }
    assert placo_params["arm_joint_names"] == ["joint1_left", "joint2_left"]
    assert placo_params["home_joint_positions"] == [0.1, -0.2]
    assert placo_params["command_out_topic"] == "/left_arm/commands"
    assert placo_params["position_only"] is True
    assert placo_params["command_lease_timeout_s"] == 0.18


def test_vr_launch_wires_same_return_home_action_to_client_and_placo(monkeypatch):
    config = _robot_config()
    config["teleoperation"]["safety"]["estop_topic"] = "/safety/estop"
    device = {
        "name": "vr",
        "type": "vr_teleop",
        "control_frequency": 50.0,
        "vr_config": {
            "output_profile": "so101",
            "so101_input_mode": "pose",
            "so101_home_action": "/left_arm/return_home",
        },
    }

    class FakeNode:
        def __init__(self, **kwargs):
            self.parameters = kwargs.get("parameters", [])

    monkeypatch.setattr(_TELEOP_MODULE, "Node", FakeNode)
    nodes = _generate_device_nodes(config, device, {})

    assert nodes[0].parameters[0]["so101_home_action"] == "/left_arm/return_home"
    assert nodes[0].parameters[0]["estop_topic"] == "/safety/estop"
    assert nodes[1].parameters[0]["home_action"] == "/left_arm/return_home"
    assert nodes[1].parameters[0]["estop_topic"] == "/safety/estop"


def test_validation_rejects_multiple_active_cartesian_devices():
    config = _robot_config()["teleoperation"]
    config["devices"].append(
        {
            "name": "phone_backup",
            "type": "phone",
            "phone_config": {"backend": "webphone", "web": {"tls": {"enabled": False}}},
        }
    )
    config.pop("active_device")
    config["active_devices"] = ["phone", "phone_backup"]

    errors = validate_teleop_config(config)

    assert any("only one active SO-101 Cartesian device" in error for error in errors)


def test_validation_rejects_leader_and_phone_sharing_arm_command_topic():
    config = _robot_config()["teleoperation"]
    config["devices"].append(
        {
            "name": "leader",
            "type": "leader_arm",
            "port": "/dev/ttyACM0",
        }
    )
    config.pop("active_device")
    config["active_devices"] = ["leader", "phone"]

    errors = validate_teleop_config(config)

    assert any("share arm command topic" in error for error in errors)


def test_validation_allows_leader_and_phone_with_distinct_arm_command_topics():
    config = _robot_config()["teleoperation"]
    config["devices"][0]["target"] = {
        "arm_command_topic": "/right_arm/commands",
        "gripper_command_topic": "/right_gripper/commands",
    }
    config["devices"].append(
        {
            "name": "leader",
            "type": "leader_arm",
            "port": "/dev/ttyACM0",
            "target": {
                "arm_command_topic": "/left_arm/commands",
                "gripper_command_topic": "/left_gripper/commands",
            },
        }
    )
    config.pop("active_device")
    config["active_devices"] = ["leader", "phone"]

    errors = validate_teleop_config(config)

    assert not any("share arm command topic" in error for error in errors)
    assert not any("share gripper command topic" in error for error in errors)


def test_validation_rejects_distinct_arms_sharing_gripper_command_topic():
    config = _robot_config()["teleoperation"]
    config["devices"][0]["target"] = {
        "arm_command_topic": "/right_arm/commands",
        "gripper_command_topic": "/shared_gripper/commands",
    }
    config["devices"].append(
        {
            "name": "leader",
            "type": "leader_arm",
            "port": "/dev/ttyACM0",
            "target": {
                "arm_command_topic": "/left_arm/commands",
                "gripper_command_topic": "/shared_gripper/commands",
            },
        }
    )
    config.pop("active_device")
    config["active_devices"] = ["leader", "phone"]

    errors = validate_teleop_config(config)

    assert any("share gripper command topic" in error for error in errors)


def test_validation_rejects_shared_arm_with_distinct_gripper_command_topics():
    config = _robot_config()["teleoperation"]
    config["devices"][0]["target"] = {
        "arm_command_topic": "/shared_arm/commands",
        "gripper_command_topic": "/right_gripper/commands",
    }
    config["devices"].append(
        {
            "name": "leader",
            "type": "leader_arm",
            "port": "/dev/ttyACM0",
            "target": {
                "arm_command_topic": "/shared_arm/commands",
                "gripper_command_topic": "/left_gripper/commands",
            },
        }
    )
    config.pop("active_device")
    config["active_devices"] = ["leader", "phone"]

    errors = validate_teleop_config(config)

    assert any("share arm command topic" in error for error in errors)


def test_validation_rejects_invalid_estop_and_position_only_contract():
    config = _robot_config()["teleoperation"]
    config["safety"]["estop_topic"] = ""
    config["cartesian"]["placo_servo"] = {"position_only": "false"}

    errors = validate_teleop_config(config)

    assert "safety.estop_topic must be a non-empty string" in errors
    assert any("position_only must be a boolean" in error for error in errors)


def test_validation_rejects_missing_joint_limits():
    config = _robot_config()
    config["teleoperation"]["safety"].pop("joint_limits")

    errors = validate_teleop_config(config["teleoperation"])

    assert "safety.joint_limits must be specified for teleoperation" in errors


def test_validation_does_not_require_joint_limits_for_mobile_base_joy_teleop():
    errors = validate_teleop_config(
        {
            "enabled": True,
            "active_device": "base_gamepad",
            "devices": [
                {
                    "name": "base_gamepad",
                    "type": "joy_teleop",
                    "config_path": "$(find robot_config)/config/lekiwi/lekiwi_teleop.yaml",
                }
            ],
        }
    )

    assert not any("joint_limits" in error for error in errors)
