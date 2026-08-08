from pathlib import Path

from robot_config.launch_builders import moveit as moveit_builder
from robot_config.launch_builders.task_execution import generate_task_executor_node
from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parents[1] / "config" / "robots" / "so101_handeye_realsense_grasp.yaml"


def _launch_parameter_value(node, name: str):
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if key != name:
            continue
        if not isinstance(raw_value, tuple):
            return raw_value
        text = "".join(getattr(item, "text", str(item)) for item in raw_value)
        if text in {"True", "False"}:
            return text == "True"
        return float(text)
    raise KeyError(name)


def test_310p_profile_uses_feedback_and_tf_completion_barrier(monkeypatch) -> None:
    config = load_robot_config_dict(CONFIG)
    monkeypatch.setattr(
        moveit_builder,
        "get_package_share_directory",
        lambda _package: str(Path(__file__).parents[2] / "robot_moveit"),
    )

    include = moveit_builder.generate_moveit_nodes(config, "moveit_planning", display=False)[0]
    arguments = dict(include._IncludeLaunchDescription__launch_arguments)

    assert config["moveit"]["motion_status_hold_s"] == 0.0
    assert arguments["motion_status_hold_s"] == "0.0"
    assert config["moveit"]["motion_feedback_timeout_s"] == 0.3
    assert config["moveit"]["motion_feedback_tolerance_rad"] == 0.12
    assert arguments["motion_feedback_timeout_s"] == "0.3"
    assert arguments["motion_feedback_tolerance_rad"] == "0.12"
    assert config["moveit"]["motion_require_tf_sync"] is True
    assert arguments["motion_require_tf_sync"] == "True"
    assert config["moveit"]["motion_hardware_feedback_topic"] == "/so101_follower/joint_currents"
    assert arguments["motion_hardware_feedback_topic"] == "/so101_follower/joint_currents"
    assert config["grasp_execution"]["contact_realign"]["settle_sec"] == 0.0
    assert config["grasp_execution"]["pose_diagnostics"]["settle_sec"] == 0.0


def test_sim_profile_does_not_require_a_real_hardware_heartbeat(monkeypatch) -> None:
    config = load_robot_config_dict(CONFIG)
    monkeypatch.setattr(
        moveit_builder,
        "get_package_share_directory",
        lambda _package: str(Path(__file__).parents[2] / "robot_moveit"),
    )

    include = moveit_builder.generate_moveit_nodes(config, "moveit_planning", use_sim=True, display=False)[0]
    arguments = dict(include._IncludeLaunchDescription__launch_arguments)

    assert arguments["motion_require_tf_sync"] == "True"
    assert arguments["motion_hardware_feedback_topic"] == ""


def test_310p_profile_does_not_skip_the_initial_gripper_open() -> None:
    config = load_robot_config_dict(CONFIG)

    node = generate_task_executor_node(config, "moveit_planning")

    assert _launch_parameter_value(node, "skip_redundant_gripper_open") is False
    assert _launch_parameter_value(node, "gripper_open_position") == 1.0
    assert _launch_parameter_value(node, "gripper_position_tolerance") == 0.05
    assert _launch_parameter_value(node, "joint_state_max_age_s") == 0.25
