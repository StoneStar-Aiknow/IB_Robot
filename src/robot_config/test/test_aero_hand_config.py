from pathlib import Path

import yaml

from robot_config.launch_builders.control import generate_auxiliary_actuator_nodes
from robot_config.launch_builders.hand_sources import apply_hand_profile
from robot_config.launch_builders.recording import get_recording_topics
from robot_config.loader import _load_robot_section

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_arm_aero_hand.yaml"
TELEOP_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "robots" / "aero_hand_teleop.yaml"


def _config():
    return _load_robot_section(CONFIG_PATH)[1]


def _node_parameters(node):
    def text(value):
        return "".join(item.text if hasattr(item, "text") else str(item) for item in value)

    def decode(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode(tuple(item)) for item in value]
        if all(isinstance(item, bool | int | float) for item in value):
            return list(value)
        return yaml.safe_load(text(value))

    return {text(key): decode(value) for key, value in node._Node__parameters[0].items()}


def test_aero_hand_config_keeps_hand_out_of_ros2_control_joint_set():
    config = _config()

    assert config["joints"]["all"] == ["1", "2", "3", "4", "5", "6"]
    assert len(config["joints"]["hand"]) == 7
    assert set(config["joints"]["hand"]).isdisjoint(config["joints"]["all"])
    state_selector = next(
        observation["selector"]["names"]
        for observation in config["contract"]["observations"]
        if observation["key"] == "observation.state"
    )
    assert state_selector == [f"position.{name}" for name in config["joints"]["all"]]


def test_aero_hand_config_uses_disjoint_arm_and_hand_publish_topics():
    config = _config()
    devices = {device["name"]: device for device in config["teleoperation"]["devices"]}
    target = devices["aero_glove_right"]["target"]

    assert config["teleoperation"]["active_devices"] == ["so101_leader", "aero_glove_right"]
    assert target == {"actuator": "aero_hand_right"}
    assert config["auxiliary_actuators"][target["actuator"]]["joint_names"] == config["joints"]["hand"]
    assert config["auxiliary_actuators"][target["actuator"]]["command_topic"] == "/aero_hand_right/commands"


def test_aero_hand_config_defaults_to_real_hardware_with_external_sdk():
    config = _config()
    actuator = config["auxiliary_actuators"]["aero_hand_right"]
    source = config["hand_sources"]["mhandpro"]
    devices = {device["name"]: device for device in config["teleoperation"]["devices"]}
    glove = devices["aero_glove_right"]

    assert actuator["mock"] is False
    assert actuator["active_control_modes"] == ["teleop"]
    assert actuator["port"] == "$(env AERO_HAND_RIGHT_PORT)"
    assert actuator["estop_topic"] == "/emergency_stop"
    assert actuator["estop_behavior"] == "hold"
    assert source["mock"] is False
    assert source["active_control_modes"] == ["teleop"]
    assert source["lib_path"] == "$(env MHANDPRO_SDK_LIB)"
    assert source["publish_raw_frame"] is False
    assert source["sides"] == ["right"]
    assert source["require_p_pose"] is True
    assert source["failure_policy"] == "require_all"
    assert source["auto_reconnect"] is True
    assert source["reconnect_initial_delay"] == 1.0
    assert source["reconnect_max_delay"] == 10.0
    assert source["reconnect_max_attempts"] == 0
    assert source["exclusive_resources"] == ["mhandpro_sdk"]
    assert source["calibration_service"] == "/hand_sources/mhandpro/calibrate_p_pose"
    assert glove["type"] == "hand_retarget"
    assert glove["source_name"] == "mhandpro"
    assert glove["source_topic"] == "/hand_sources/mhandpro/right/state"
    assert glove["calib_file"] == "$(env HOME)/.calibrate/aero_hand_right_calibrate.json"
    assert glove["retargeter"]["type"] == "aero_compact"
    thumb_model = glove["retargeter"]["aero_thumb_model"]
    assert thumb_model["mcp_ip_weights"] == [9.4372, 12.5]
    assert thumb_model["root_neutral_trims"] == [0.0, 0.0]
    assert thumb_model["root_output_scales"] == [0.95, 0.94]
    assert thumb_model["tendon_output_scale"] == 0.60
    assert thumb_model["max_thumb_step_rad"] == [0.04, 0.03, 0.04]
    finger_model = glove["retargeter"]["aero_finger_model"]
    assert finger_model == {
        "pip_weight": 0.55,
        "dip_weight": 0.45,
        "open_threshold_rad": 0.2617993877991494,
        "closed_threshold_rad": 0.8726646259971648,
        "active_trim_fraction": 0.08,
    }
    assert "normalized_deadbands" not in glove
    assert "task_space" not in glove


def test_aero_hand_topics_are_in_continuous_recording_set():
    topics = get_recording_topics(_config())

    assert "/aero_hand_right/commands" in topics
    assert "/aero_hand_right/joint_states" in topics
    assert "/hand_sources/mhandpro/right/frame" not in topics


def test_standalone_profiles_pass_safety_limits_to_each_aero_driver(monkeypatch):
    monkeypatch.setenv("AERO_HAND_LEFT_PORT", "/dev/aero-left")
    monkeypatch.setenv("AERO_HAND_RIGHT_PORT", "/dev/aero-right")
    source = yaml.safe_load(TELEOP_CONFIG_PATH.read_text(encoding="utf-8"))["robot"]
    for profile in ("left", "right", "dual"):
        config = yaml.safe_load(yaml.safe_dump(source))
        apply_hand_profile(config, profile)

        nodes = generate_auxiliary_actuator_nodes(config, control_mode="teleop")

        assert len(nodes) == (2 if profile == "dual" else 1)
        for node in nodes:
            parameters = _node_parameters(node)
            joint_names = parameters["joint_names"]
            limits = config["teleoperation"]["safety"]["joint_limits"]
            assert parameters["command_lower_limits"] == [limits[name]["min"] for name in joint_names]
            assert parameters["command_upper_limits"] == [limits[name]["max"] for name in joint_names]
