from copy import deepcopy
from pathlib import Path

import yaml

from robot_config.launch_builders.perception import generate_camera_nodes
from robot_config.launch_builders.task_execution import generate_task_executor_node
from robot_config.loader import load_robot_config_dict, validate_motion_mode_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/robot_config/config/robots/lekiwi_nav_grasp.yaml"


def _launch_parameter_value(node, name: str):
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if key != name:
            continue
        if not isinstance(raw_value, tuple):
            return raw_value
        text = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
        if text.endswith("\n..."):
            text = text[:-4].rstrip()
        if text in {"True", "False"}:
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    raise KeyError(name)


def test_unified_profile_preserves_arm_base_hardware_and_motion_ownership():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["robot"]

    assert config["joints"]["all"] == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    assert config["ros2_control"]["controllers_config"].endswith("lekiwi_controllers.yaml")
    assert "base_only" not in config["ros2_control"].get("xacro_mappings", {})
    assert config["motion_mode"]["manipulation_controllers"] == [
        "arm_trajectory_controller",
        "gripper_trajectory_controller",
    ]
    assert config["motion_mode"]["navigation_controllers"] == ["base_velocity_controller"]

    peripherals = {item["name"]: item for item in config["peripherals"]}
    assert set(peripherals) == {"respeaker", "wrist", "front", "mid360"}
    assert peripherals["respeaker"] == {
        "type": "microphone",
        "name": "respeaker",
        "driver": "alsa",
        "params": {
            "device_name_contains": "ReSpeaker",
            "arecord_device": "hw:0,0",
            "sample_rate": 16000,
            "channel_indices": [1, 2, 3, 4],
        },
    }
    assert peripherals["wrist"]["transform"]["parent_frame"] == "gripper"
    assert peripherals["front"]["transform"]["parent_frame"] == "base_link"
    assert peripherals["front"]["driver_camera_name"] == "front"
    assert peripherals["front"]["align_depth"] is True
    assert peripherals["mid360"]["driver"] == "livox_mid360"
    assert config["semantic_mapping"]["camera"] == {
        "peripheral": "front",
        "mounting": "fixed",
        "parent_frame": "base_link",
        "rgb_topic": "/camera/front/image_raw",
        "depth_topic": "/camera/front/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/camera/front/camera_info",
    }
    assert config["speech_direction"]["enabled"] is True
    assert config["speech_direction"]["profile"] == "ascend_310p"
    assert config["speech_direction"]["microphone"] == "respeaker"


def test_unified_profile_resolves_grasp_mapping_and_navigation_stages():
    hybrid = load_robot_config_dict(CONFIG_PATH)
    grasp = load_robot_config_dict(CONFIG_PATH, nav_stage="grasp")
    mapping = load_robot_config_dict(CONFIG_PATH, nav_stage="mapping")
    navigation = load_robot_config_dict(CONFIG_PATH, nav_stage="navigation")

    assert hybrid["nav_stage"] == "hybrid"
    assert hybrid["default_control_mode"] == "base_navigation"
    assert hybrid["skill_required_control_mode"] == "moveit_planning"
    assert hybrid["embodied"]["skill_catalog_profile"] == "lekiwi_handeye_realsense_grasp_lidar"
    assert hybrid["navigation"]["enabled"] is True
    assert hybrid["navigation"]["command_server"]["enabled"] is True
    assert hybrid["mid360_mount_file"].endswith("lekiwi_mid360_mount.yaml")
    assert [item["name"] for item in hybrid["peripherals"]] == ["respeaker", "wrist", "front", "mid360"]
    assert "peripheral_names" not in hybrid
    assert hybrid["grasp_execution"]["ik"]["worker_count"] == 1
    assert hybrid["grasp_execution"]["ik"]["verification_position_tolerance_m"] == 0.001
    assert hybrid["grasp_execution"]["ik"]["verification_orientation_tolerance_deg"] == 1.0
    assert hybrid["control_modes"]["base_navigation"]["controllers"] == [
        "joint_state_broadcaster",
        "arm_joint_state_broadcaster",
        "base_velocity_controller",
    ]
    assert set(hybrid["control_modes"]["moveit_planning"]["inactive_controllers"]) == {"base_velocity_controller"}
    assert hybrid["motion_mode"]["manipulation_control_mode"] == "moveit_planning"
    assert hybrid["motion_mode"]["navigation_control_mode"] == "base_navigation"
    assert hybrid["motion_mode"]["navigation_enabled_on_startup"] is True

    assert grasp["nav_stage"] == "grasp"
    assert grasp["navigation"]["enabled"] is False
    assert [item["name"] for item in grasp["peripherals"]] == ["respeaker", "wrist"]

    assert mapping["nav_stage"] == "mapping"
    assert mapping["default_control_mode"] == "base_navigation"
    assert mapping["navigation"]["fast_lio"]["enabled"] is True
    assert mapping["navigation"]["slam_toolbox"]["enabled"] is True
    assert mapping["navigation"]["nav2_bringup"]["enabled"] is False
    assert mapping["navigation"]["cmd_vel_bridge"]["cmd_vel_topic"] == "/cmd_vel"
    assert mapping["motion_mode"]["navigation_enabled_on_startup"] is True
    assert mapping["embodied"]["skill_catalog_profile"] == ""
    assert mapping["perception_services"]["services"] == []
    assert mapping["grasp_execution"]["enabled"] is False
    assert mapping["placement_execution"]["enabled"] is False
    assert mapping["contract"]["observations"] == []
    assert mapping["contract"]["actions"] == []
    assert [item["name"] for item in mapping["peripherals"]] == ["respeaker", "front", "mid360"]
    assert "peripheral_names" not in mapping
    assert mapping["control_modes"]["base_navigation"]["controllers"] == [
        "joint_state_broadcaster",
        "base_velocity_controller",
    ]

    assert navigation["nav_stage"] == "navigation"
    assert navigation["default_control_mode"] == "base_navigation"
    assert navigation["skill_required_control_mode"] == "base_navigation"
    assert navigation["embodied"]["skill_catalog_profile"] == ""
    assert navigation["perception_services"]["services"] == []
    assert navigation["grasp_execution"]["enabled"] is False
    assert navigation["placement_execution"]["enabled"] is False
    assert navigation["contract"]["observations"] == []
    assert navigation["contract"]["actions"] == []
    assert [item["name"] for item in navigation["peripherals"]] == ["respeaker", "front", "mid360"]
    assert "peripheral_names" not in navigation
    assert navigation["navigation"]["fast_lio"]["enabled"] is True
    assert navigation["navigation"]["slam_toolbox"]["enabled"] is False
    assert navigation["navigation"]["nav2_bringup"]["use_amcl"] is True
    assert navigation["navigation"]["cmd_vel_bridge"]["cmd_vel_topic"] == "/cmd_vel_safe"
    assert navigation["navigation"]["command_server"]["enabled"] is True
    assert navigation["motion_mode"]["navigation_enabled_on_startup"] is True


def test_hybrid_navigation_startup_keeps_manipulation_task_executor_online():
    config = load_robot_config_dict(CONFIG_PATH)

    node = generate_task_executor_node(config, "base_navigation")

    assert node is not None
    assert node.node_package == "task_dispatch"
    assert node.node_executable == "task_executor_node"
    assert _launch_parameter_value(node, "skip_redundant_gripper_open") is False
    assert _launch_parameter_value(node, "gripper_open_position") == 1.0
    assert _launch_parameter_value(node, "gripper_position_tolerance") == 0.05
    assert _launch_parameter_value(node, "joint_state_max_age_s") == 0.25


def test_unified_profile_launches_wrist_and_front_realsense_drivers():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["robot"]

    drivers = [
        node
        for node in generate_camera_nodes(config, use_sim=False)
        if vars(node).get("_Node__package") == "realsense2_camera"
    ]
    assert {vars(node).get("_Node__node_name") for node in drivers} == {"wrist_camera", "front"}


def test_hybrid_startup_mode_and_navigation_gate_must_match():
    hybrid = load_robot_config_dict(CONFIG_PATH)

    navigation_startup = deepcopy(hybrid)
    navigation_startup["default_control_mode"] = "base_navigation"
    navigation_startup["motion_mode"]["navigation_enabled_on_startup"] = True
    assert validate_motion_mode_config(navigation_startup) == []

    navigation_startup["motion_mode"]["navigation_enabled_on_startup"] = False
    assert any("must match default_control_mode" in error for error in validate_motion_mode_config(navigation_startup))


def test_unified_profile_applies_only_the_front_camera_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros/ibrobot/calib/current/base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "043322073551"},
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.12, -0.03, 0.42],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    hybrid = load_robot_config_dict(CONFIG_PATH)
    front = next(item for item in hybrid["peripherals"] if item["name"] == "front")
    wrist = next(item for item in hybrid["peripherals"] if item["name"] == "wrist")
    assert [front["transform"][key] for key in ("x", "y", "z")] == [0.12, -0.03, 0.42]
    assert wrist["transform"]["parent_frame"] == "gripper"

    grasp = load_robot_config_dict(CONFIG_PATH, nav_stage="grasp")
    assert [item["name"] for item in grasp["peripherals"]] == ["respeaker", "wrist"]
