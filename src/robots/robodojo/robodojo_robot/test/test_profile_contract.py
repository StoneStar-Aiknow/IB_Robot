"""The shipped profile serves the runtime contract with contract-v2 motion groups."""

from __future__ import annotations

from pathlib import Path

from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "arx_x5_dual.yaml"


def test_profile_declares_both_arms_as_motion_groups():
    profile = load_profile(PROFILE)
    groups = profile["capabilities"]["motion.move_to_pose"]["groups"]
    assert groups == ["left_arm", "right_arm"] == sorted(profile["motion_groups"])
    assert profile["capabilities"]["motion.move_to_pose"]["default_group"] == profile["default_group"]
    for spec in profile["motion_groups"].values():
        assert set(spec["joints"]) <= set(profile["arm_joints"])
        assert spec["gripper"] in profile["gripper_joints"]


def test_description_is_simulated_on_the_robodojo_platform_with_per_arm_interfaces():
    profile = load_profile(PROFILE)
    assert profile["use_sim"] is True and profile["sim_platform"] == "robodojo"
    description = build_description(profile)
    assert description["execution"] == "simulated" and description["sim_platform"] == "robodojo"
    assert "model" not in description
    interfaces = description["interfaces"]
    for group in ("left_arm", "right_arm"):
        assert interfaces[f"motion.ee_pose.{group}"]["endpoint"] == f"/robot_status/{group}/ee_pose"
    assert interfaces["motion.ee_pose"]["endpoint"] == "/robot_status/ee_pose"
    for side in ("left", "right"):
        action = interfaces[f"joint.trajectory_{side}_gripper"]
        assert action["endpoint"] == f"/{side}_gripper_trajectory_controller/follow_joint_trajectory"
    assert interfaces["motion.move_to_pose"]["endpoint"] == "/motion/move_to_pose"


def test_the_adapter_descriptor_comes_from_the_same_profile():
    from robodojo_robot.adapter import RoboDojoAdapter

    profile = load_profile(PROFILE)
    descriptor = RoboDojoAdapter(profile, {"robodojo_root": "/unused"}).descriptor
    assert descriptor.group_names == tuple(profile["motion_groups"])
    for group in descriptor.groups:
        assert group.joints == tuple(profile["motion_groups"][group.name]["joints"])
        assert group.tcp_offset == (0.13, 0.0, 0.0)
    assert descriptor.dt_s == 0.04 and descriptor.ik == "platform"
    # No simulated-time hold on this robot: no capability, no command, no robot-config bound.
    assert "simulation.hold" not in profile["capabilities"] and not descriptor.supports("hold")


def test_the_policy_server_listens_on_loopback_by_default():
    profile = load_profile(PROFILE)
    assert profile["simulation"]["robodojo"]["adapter_config"]["host"] == "127.0.0.1"


def test_body_facts_have_one_source():
    """The robot config repeats only consumer bindings, and they agree with this profile."""
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict

    profile = load_profile(PROFILE)
    robot = load_robot_config_dict(resolve_robot_config_path(config_name="robodojo_x5"), defer_interface_binding=True)
    assert robot["runtime"]["provider"] == "robodojo_robot" and robot["runtime"]["profile"] == PROFILE.stem
    assert robot["simulation"]["platform"] == profile["sim_platform"] and robot["runtime"]["target"] == "simulation"
    groups = profile["motion_groups"]
    execution = robot["embodied"]["execution"]
    assert execution["arm_groups"] == {spec["side"]: name for name, spec in groups.items()}
    dispatch = robot["task_dispatch"]
    assert dispatch["default_group"] == profile["default_group"]
    assert {g: spec["gripper_joint"] for g, spec in dispatch["groups"].items()} == {
        name: spec["gripper"] for name, spec in groups.items()
    }
    assert "simulation_hold" not in execution
    prefix = profile["camera_topic_prefix"]
    planner_cameras = {t.split("/")[2] for t in robot["embodied"]["agent"]["planner"]["vision_topics"]}
    assert planner_cameras == set(profile["cameras"])
    assert all(t.startswith(prefix + "/") for t in robot["embodied"]["agent"]["planner"]["vision_topics"])
