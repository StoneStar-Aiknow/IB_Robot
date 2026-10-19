"""so101_robot runtime launch tests (openspec ``so101-gazebo-runtime-transport``).

Renders the shipped ``runtime.launch.py`` against the real
``so101_single_arm`` profile on the Gazebo transport, pinning the launch
argument mapping (transport, world name, full spawn pose, sim-time motion
arguments) and the runtime-owned Gazebo camera poses (task 2.4).

This is also the package's first test directory: ``colcon test`` previously
collected 0 tests for so101_robot (it shipped no tests at all).
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import launch
import yaml
from launch_ros.actions import Node

_PACKAGE = Path(__file__).resolve().parents[1]
_LAUNCH = _PACKAGE / "launch" / "runtime.launch.py"
_PROFILE = _PACKAGE / "profiles" / "so101_single_arm.yaml"


def _launch_module():
    spec = importlib.util.spec_from_file_location("so101_robot_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(extra):
    context = launch.LaunchContext()
    configurations = {
        "profile": str(_PROFILE),
        "simulated": "",
        "transport": "",
        "gazebo_world": "",
        "spawn_x": "0.0",
        "spawn_y": "0.0",
        "spawn_z": "0.0",
        "spawn_yaw": "0.0",
        "display": "false",
        "instance_id": "",
        "initial_mode": "",
        "peripherals": "",
    }
    configurations.update(extra)
    context.launch_configurations.update(configurations)
    return _launch_module()._launch_setup(context)


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _param_value(value):
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)
        if any(isinstance(item, tuple | list) for item in value):
            return [_param_value(item) for item in value]
        return yaml.safe_load(_text(value)) if value else []
    return value


def _params(node):
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update({_text(key): _param_value(value) for key, value in entry.items()})
    return params


def test_shipped_profile_declares_the_gazebo_transport():
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    assert profile["transports"] == ["hardware", "sdk", "gazebo"]
    assert profile["transport"] == "hardware"


def test_gazebo_transport_threads_world_spawn_pose_and_sim_time():
    actions = _render(
        {
            "transport": "gazebo",
            "gazebo_world": "demo",
            "spawn_x": "0.5",
            "spawn_y": "-0.25",
            "spawn_z": "0.1",
            "spawn_yaw": "1.25",
        }
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _params(facade)["transport"] == "gazebo"
    assert _params(facade)["use_sim_time"] is True

    # the motion include runs on sim time
    motion = next(a for a in actions if type(a).__name__ == "IncludeLaunchDescription")
    motion_args = dict(motion.launch_arguments)
    assert _text(motion_args["use_sim_time"]) == "true"
    assert _text(motion_args["motion_hardware_feedback_topic"]) == ""

    # the create node waits behind the clock handler and carries the full pose
    from launch.actions import RegisterEventHandler

    deferred = []
    for action in actions:
        if isinstance(action, RegisterEventHandler):
            handler = getattr(action.event_handler, "_OnActionEventBase__on_event", None)
            if handler is not None:
                deferred.extend(handler(type("E", (), {"returncode": 0})(), None) or [])
    create = next(node for node in deferred if node._Node__node_executable == "create")
    args = create._Node__arguments
    for flag, value in (("-x", "0.5"), ("-y", "-0.25"), ("-z", "0.1"), ("-Y", "1.25"), ("-world", "demo")):
        assert value == args[args.index(flag) + 1]
    # the gazebo embodiment never starts a ros2_control_node
    assert not any(node._Node__node_executable == "ros2_control_node" for node in nodes)


def test_unknown_transport_fails_fast_before_any_action():
    import pytest

    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="unknown transport 'mujoco'"):
        _render({"transport": "mujoco"})


def test_provider_gazebo_urdf_mounts_cameras_at_the_runtime_poses(tmp_path):
    from robot_runtime.launch_support import render_gazebo_description
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    urdf = render_gazebo_description(profile, _PROFILE, spawn_pose=(0.0, 0.0, 0.0, 0.0))
    # wrist mounts on the gripper at the calibrated pose (previous preset)
    wrist = re.search(r'name="wrist_camera_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"', urdf, re.S)
    assert '<parent link="gripper"/>' in urdf
    assert wrist.group(1).split() == [str(value) for value in (0.002265, 0.073023, -0.016289)]
    assert [float(v) for v in wrist.group(2).split()] == [-1.512475, 1.079239, -1.541112]
    # top is the calibrated workspace camera at the world pose
    top = re.search(r'name="top_camera_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"', urdf, re.S)
    assert [float(v) for v in top.group(1).split()] == [0.0419, 0.1123, 0.6136]
    assert [float(v) for v in top.group(2).split()] == [0.057596, 1.518436, -1.518436]
    # advertised frames: opencv cameras stamp their frame_id, the realsense its
    # optical frame (D6)
    assert "<gz_frame_id>camera_top_frame</gz_frame_id>" in urdf
    assert "<gz_frame_id>camera_wrist_frame</gz_frame_id>" in urdf
    assert "<gz_frame_id>camera_front_optical_frame</gz_frame_id>" in urdf
    # fovy 47 at 640x480 drives the horizontal field of view
    expected = 2.0 * math.atan(math.tan(math.radians(47) / 2) * 640 / 480)
    assert f"<horizontal_fov>{expected:.4f}</horizontal_fov>" in urdf


def test_saved_camera_override_takes_precedence_on_the_provider_path(tmp_path, monkeypatch):
    from robot_runtime.gazebo import camera_sensor_urdf
    from robot_runtime.peripherals import merge_peripherals
    from robot_runtime.profile import load_profile

    home = tmp_path / "home"
    override_dir = home / ".ros" / "ibrobot" / "sim_camera_overrides"
    override_dir.mkdir(parents=True)
    (override_dir / "top.yaml").write_text(
        yaml.safe_dump(
            {
                "pose": {"x": 1.5, "y": -0.2, "z": 0.8, "roll": 0.0, "pitch": 1.4, "yaw": -1.4},
                "parent_frame": "world",
                "fovy_deg": 50,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    profile = load_profile(_PROFILE)
    peripherals = merge_peripherals(profile.get("peripherals"), [])
    urdf = camera_sensor_urdf(peripherals, platform="gazebo", frame_contract=True)
    assert 'xyz="1.5 -0.2 0.8"' in urdf
    # the untouched cameras keep the runtime-owned poses
    assert 'xyz="0.002265 0.073023 -0.016289"' in urdf
