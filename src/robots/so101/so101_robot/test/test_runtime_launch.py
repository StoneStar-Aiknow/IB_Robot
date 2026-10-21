"""so101_robot runtime launch tests (openspec ``so101-gazebo-runtime-transport``).

Renders the shipped ``runtime.launch.py`` against the real
``so101_single_arm`` profile on the Gazebo platform, pinning the launch
argument mapping (simulation selection, world name, full spawn pose, sim-time motion
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
        "use_sim": "",
        "sim_platform": "",
        "gazebo_world": "",
        "mujoco_scene": "",
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


def test_shipped_profile_declares_the_gazebo_platform():
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    assert profile["sim_platforms"] == ["sdk", "gazebo", "mujoco"]
    assert profile["sim_platform"] == "sdk"
    assert profile["use_sim"] is False


def test_gazebo_platform_threads_world_spawn_pose_and_sim_time():
    actions = _render(
        {
            "use_sim": "true",
            "sim_platform": "gazebo",
            "gazebo_world": "demo",
            "spawn_x": "0.5",
            "spawn_y": "-0.25",
            "spawn_z": "0.1",
            "spawn_yaw": "1.25",
        }
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _params(facade)["sim_platform"] == "gazebo"
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


def test_unknown_platform_fails_fast_before_any_action():
    import pytest

    from robot_runtime.launch_support import RuntimeProfileError

    # a platform outside the runtime vocabulary fails on the spot
    with pytest.raises(RuntimeProfileError, match="unknown sim_platform 'webots'"):
        _render({"use_sim": "true", "sim_platform": "webots"})


def test_gazebo_platform_applies_a_peripheral_toggle_exactly_once(tmp_path):
    """A deployment disable toggle survives the whole launch construction.

    The description is rendered from the merged peripheral set while the stack
    helpers and the facade each apply the fragment themselves; folding the
    merge into the profile they receive applied the toggle twice, and the
    second pass rejected it as a toggle for an undeclared peripheral (the
    disabled entry was already filtered out) - the so101_rtp_distributed
    front-camera disable hit exactly that.
    """
    fragment = tmp_path / "peripherals.yaml"
    fragment.write_text("peripherals:\n- type: camera\n  name: front\n  disabled: true\n", encoding="utf-8")

    actions = _render({"use_sim": "true", "sim_platform": "gazebo", "peripherals": str(fragment)})

    nodes = [action for action in actions if isinstance(action, Node)]
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    # The facade owns the live perception surface: it keeps the fragment path
    # and applies it against the raw profile itself.
    assert _params(facade)["peripherals"] == str(fragment)
    bridge = next(node for node in nodes if node._Node__node_executable == "bridge_node")
    config = Path(_params(bridge)["config_file"]).read_text(encoding="utf-8")
    assert "camera/front" not in config
    assert "camera/top" in config
    assert "camera/wrist" in config


def test_teleop_executor_follows_the_platform_clock():
    """The runtime-owned placo servo runs on sim time only on Gazebo (PR !515 P2).

    Managed mode compares the node clock against JointState stamps; a wall
    clock under Gazebo drops every sim-stamped feedback sample.
    """
    gazebo = _render({"use_sim": "true", "sim_platform": "gazebo"})
    servo = next(
        node for node in gazebo if isinstance(node, Node) and node._Node__node_executable == "so101_placo_servo_node.py"
    )
    assert _params(servo)["use_sim_time"] is True

    physical = _render({"use_sim": "false"})
    servo = next(
        node
        for node in physical
        if isinstance(node, Node) and node._Node__node_executable == "so101_placo_servo_node.py"
    )
    assert _params(servo)["use_sim_time"] is False


def test_physical_selection_passes_no_platform_to_the_facade():
    """Physical execution must reach the facade with no platform (PR !515 P1).

    The launch used to default the platform to "sdk" on the physical path,
    which the facade's selection resolver rejects; the regression goes through
    the real entry: rendered facade parameters must resolve without error.
    """
    from robot_runtime.facade_node import resolve_facade_selection
    from robot_runtime.profile import load_profile

    actions = _render({"use_sim": "false"})
    nodes = [action for action in actions if isinstance(action, Node)]
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    params = _params(facade)
    assert params["use_sim"] is False
    assert "sim_platform" not in params, "physical execution must carry no platform"

    profile = load_profile(_PROFILE)
    assert resolve_facade_selection(profile, bool(params["use_sim"]), str(params.get("sim_platform", ""))) == (
        False,
        None,
    )


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


# ---------------------------------------------------------------------------
# so101-mujoco-runtime-platform task 3.4: the MuJoCo platform on this profile
# ---------------------------------------------------------------------------


def test_shipped_profile_declares_mujoco_and_keeps_the_golden_camera_poses():
    """The MuJoCo defaults are runtime-owned and hold the provider-less
    path's preset values verbatim (top/wrist) plus the old generic base
    preset for front."""
    from robot_runtime.mujoco import resolve_mujoco_camera_pose
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    assert "mujoco" in profile["sim_platforms"]
    mujoco_cfg = profile["simulation"]["mujoco"]
    assert mujoco_cfg["sdf_meshes"] == ["wrist_roll_follower", "moving_jaw"]
    peripherals = {entry["name"]: entry for entry in profile["peripherals"]}

    top, top_fovy = resolve_mujoco_camera_pose(peripherals["top"])
    assert (top["parent_frame"], top_fovy) == ("world", 47)
    assert (top["x"], top["y"], top["z"]) == (0.0419, 0.1123, 0.6136)
    assert (top["roll"], top["pitch"], top["yaw"]) == (0.052274, 0.003013, 3.136436)

    wrist, wrist_fovy = resolve_mujoco_camera_pose(peripherals["wrist"])
    assert (wrist["parent_frame"], wrist_fovy) == ("gripper", 46)
    assert (wrist["x"], wrist["y"], wrist["z"]) == (0.002265, 0.073023, -0.016289)
    assert (wrist["roll"], wrist["pitch"], wrist["yaw"]) == (0.031197, -0.490648, -1.607245)

    front, front_fovy = resolve_mujoco_camera_pose(peripherals["front"])
    assert (front["parent_frame"], front_fovy) == ("base", 70)
    assert (front["x"], front["y"], front["z"]) == (0.0, -0.28, 0.55)
    assert (front["roll"], front["pitch"], front["yaw"]) == (0.0, 0.0, 3.1415)


def test_mujoco_scene_argument_reaches_the_composition(monkeypatch):
    import sim_models.scene_compiler as scene_compiler

    captured = {}

    def fake_compose(robot_mjcf_path, scene, sdf_mesh_names):
        captured["scene"] = scene
        captured["sdf_meshes"] = list(sdf_mesh_names)
        return Path("/tmp/ibrobot_fake_composed.mjb")

    monkeypatch.setattr(scene_compiler, "compose_mujoco_model", fake_compose)
    actions = _render({"use_sim": "true", "sim_platform": "mujoco", "mujoco_scene": "pick_banana"})
    assert captured["scene"] == "pick_banana"
    assert captured["sdf_meshes"] == ["wrist_roll_follower", "moving_jaw"]
    mujoco_nodes = [
        node for node in actions if isinstance(node, Node) and node._Node__node_executable == "ros2_control_node"
    ]
    assert [node._Node__package for node in mujoco_nodes] == ["mujoco_ros2_control"]


def test_mujoco_without_a_scene_composes_the_empty_scene(monkeypatch):
    """5b.2 revision: no scene composes the `empty` scene (lit ground), never robot-only."""
    import sim_models.scene_compiler as scene_compiler

    captured = {}

    def fake_compose(robot_mjcf_path, scene, sdf_mesh_names):
        captured["scene"] = scene
        return Path("/tmp/ibrobot_fake_composed.mjb")

    monkeypatch.setattr(scene_compiler, "compose_mujoco_model", fake_compose)
    _render({"use_sim": "true", "sim_platform": "mujoco"})
    assert captured["scene"] == "empty"


# ---------------------------------------------------------------------------
# 5b.2 / 5b.3: the empty-scene composition and the shared camera frame contract
# ---------------------------------------------------------------------------


def test_provider_alone_empty_scene_composes_ground_and_light(tmp_path):
    """5b.2: provider-alone with an empty mujoco_scene carries the lit ground."""
    import mujoco

    from robot_runtime.launch_support import render_mujoco_description
    from robot_runtime.profile import load_profile
    from sim_models.scene_compiler import composed_model_path

    profile = load_profile(_PROFILE)
    description = render_mujoco_description(
        profile, _PROFILE, profile.get("peripherals"), spawn_pose=(0.0, 0.0, 0.0, 0.0), scene=""
    )
    composed = composed_model_path("empty")
    assert str(composed) in description
    model = mujoco.MjModel.from_binary_path(str(composed))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert model.nlight >= 1


def test_advertised_camera_frames_are_identical_on_every_platform():
    """5b.3: every SO-101 camera advertises the same frame on all four platforms."""
    import xml.etree.ElementTree as ET

    from robot_runtime.interface_description import build_description
    from robot_runtime.launch_support import render_mujoco_description
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    cameras = [str(p["name"]) for p in profile.get("peripherals") or [] if p.get("type") == "camera"]
    assert set(cameras) == {"top", "wrist", "front"}

    frames = {name: {} for name in cameras}
    for platform, use_sim in (("hardware", False), ("sdk", True), ("gazebo", True), ("mujoco", True)):
        doc = build_description(profile, use_sim=use_sim, sim_platform=None if platform == "hardware" else platform)
        for name in cameras:
            frames[name][platform] = doc["interfaces"][f"camera.{name}.color"]["frame_id"]

    for name, per_platform in frames.items():
        assert len(set(per_platform.values())) == 1, f"{name} advertises different frames: {per_platform}"
    assert frames["top"]["mujoco"] == "camera_top_frame"
    assert frames["wrist"]["mujoco"] == "camera_wrist_frame"
    assert frames["front"]["mujoco"] == "camera_front_optical_frame"

    # the MuJoCo sensor blocks stamp exactly the advertised frame
    description = render_mujoco_description(profile, _PROFILE, profile.get("peripherals"), scene="")
    root = ET.fromstring(description)
    sensor_frames = {
        sensor.get("name").removesuffix("_camera"): param.text
        for sensor in root.findall(".//ros2_control/sensor")
        for param in sensor.findall("param")
        if param.get("name") == "frame_name"
    }
    assert sensor_frames == {name: frames[name]["mujoco"] for name in cameras}
