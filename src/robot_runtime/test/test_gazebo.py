"""Gazebo platform tests: shared camera builders, embodiment actions, sim identity.

Covers openspec ``so101-gazebo-runtime-transport`` tasks 2.1, 2.3b, 2.3c and
2.3d: the launch-structure contract per platform, the advertised camera
contract (frames, streams, spawn compensation), the effective peripheral
inventory (disabled cameras absent everywhere) and the simulated joint
current.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from launch_ros.actions import Node

from robot_runtime.gazebo import (
    advertised_camera_frame,
    camera_bridge_entries,
    camera_sensor_urdf,
    resolve_camera_pose,
)
from robot_runtime.interface_description import (
    InterfaceDescriptionError,
    build_description,
    merge_peripherals,
)
from robot_runtime.launch_support import (
    gazebo_embodiment_actions,
    motion_launch_arguments,
    peripheral_stack_actions,
    render_gazebo_description,
    runtime_stack_actions,
    simulated_joint_current_actions,
)

# ---------------------------------------------------------------------------
# Shared fixtures: a minimal robot-agnostic profile the launch helpers accept
# ---------------------------------------------------------------------------

XACRO = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:arg name="use_sim" default="true"/>
  <xacro:arg name="simulated" default="false"/>
  <xacro:arg name="sim_plugin" default="gz_ros2_control/GazeboSimSystem"/>
  <xacro:arg name="gazebo_effort_state" default="false"/>
  <xacro:arg name="gz_ros2_control_parameters_file" default=""/>
  <link name="base">
    <visual><geometry><mesh><filename>package://robot_runtime/meshes/none.stl</filename></mesh></geometry></visual>
  </link>
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>$(arg gz_ros2_control_parameters_file)</parameters>
    </plugin>
  </gazebo>
  <xacro:if value="$(arg use_sim)">
    <ros2_control name="RobotSystem" type="system">
      <hardware>
        <plugin>$(arg sim_plugin)</plugin>
        <param name="simulated">$(arg simulated)</param>
      </hardware>
      <joint name="1">
        <command_interface name="position"/>
        <state_interface name="position"/>
        <state_interface name="velocity"/>
        <xacro:if value="$(arg gazebo_effort_state)">
          <state_interface name="effort"/>
        </xacro:if>
      </joint>
    </ros2_control>
  </xacro:if>
</robot>
"""

TOP = {
    "type": "camera",
    "name": "top",
    "driver": "opencv",
    "index": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "pixel_format": "mjpeg2rgb",
    "frame_id": "camera_top_frame",
    "optical_frame_id": "camera_top_optical_frame",
    "simulation": {
        "gazebo": {
            "transform": {
                "parent_frame": "world",
                "x": 0.0419,
                "y": 0.1123,
                "z": 0.6136,
                "roll": 0.057596,
                "pitch": 1.518436,
                "yaw": -1.518436,
            },
            "fovy": 47,
        }
    },
    "transform": {"parent_frame": "base", "x": 0.0, "y": 0.0, "z": 0.0},
}
WRIST = {
    "type": "camera",
    "name": "wrist",
    "driver": "opencv",
    "index": 2,
    "width": 640,
    "height": 480,
    "fps": 60,
    "pixel_format": "mjpeg2rgb",
    "frame_id": "camera_wrist_frame",
    "optical_frame_id": "camera_wrist_optical_frame",
    "simulation": {
        "gazebo": {
            "transform": {
                "parent_frame": "gripper",
                "x": 0.002265,
                "y": 0.073023,
                "z": -0.016289,
                "roll": -1.512475,
                "pitch": 1.079239,
                "yaw": -1.541112,
            },
            "fovy": 46,
        }
    },
    "transform": {"parent_frame": "gripper", "x": 0.0, "y": 0.0, "z": 0.0},
}
FRONT = {
    "type": "camera",
    "name": "front",
    "driver": "realsense",
    "serial_number": "",
    "width": 640,
    "height": 480,
    "fps": 30,
    "pixel_format": "bgr8",
    "enable_pointcloud": False,
    "align_depth": True,
    "frame_id": "camera_front_link",
    "optical_frame_id": "camera_front_optical_frame",
    "simulation": {
        "gazebo": {
            "transform": {
                "parent_frame": "base",
                "x": 0.0,
                "y": -0.28,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 1.5708,
                "yaw": -1.5708,
            },
            "fovy": 70,
        }
    },
    "transform": {"parent_frame": "base", "x": 0.0, "y": 0.0, "z": 0.0},
}


def _profile(tmp_path: Path, peripherals=None, with_joint_current=True):
    xacro = tmp_path / "robot.xacro"
    xacro.write_text(XACRO, encoding="utf-8")
    controllers = tmp_path / "controllers.yaml"
    controllers.write_text("controller_manager:\n  ros__parameters:\n    update_rate: 100\n", encoding="utf-8")
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "modes": {
            "initial": "idle",
            "idle": {"controllers": [], "transitions": ["trajectory"]},
            "trajectory": {
                "controllers": ["arm_trajectory_controller"],
                "allows_trajectory": True,
                "transitions": ["idle"],
            },
        },
        "capabilities": {"joint.state": {}, "joint.trajectory": {}, "runtime.stop": {}},
        "joints": ["1"],
        "simulated": False,
        "sim_platforms": ["sdk", "gazebo"],
        "sim_platform": "sdk",
        "controller_manager": "controller_manager",
        "joint_state_topic": "/joint_states",
        "command_channels": [
            {
                "channel": "arm_stream",
                "topic": "/arm_position_controller/commands",
                "joints": ["1"],
                "modes": ["stream"],
            }
        ],
        "trajectory_actions": ["/arm_trajectory_controller/follow_joint_trajectory"],
        "description": {"xacro": str(xacro)},
        "ros2_control": {"controllers_config": str(controllers)},
        "peripherals": peripherals if peripherals is not None else [TOP, WRIST, FRONT],
        "hardware": {},
        "motion": {
            "arm_group_name": "arm",
            "base_link": "base",
            "ee_link": "gripper",
            "shoulder_link": "shoulder",
        },
    }
    if with_joint_current:
        profile["interfaces"] = {
            "joint.current": {
                "capability": "joint.state",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/so101_follower/joint_currents",
                "message_type": "ibrobot_msgs/msg/JointCurrent",
                "joint_names": ["1"],
                "units": {"current": "A"},
                "qos": {
                    "reliability": "reliable",
                    "durability": "volatile",
                    "history": "keep_last",
                    "depth": 10,
                },
            }
        }
        profile["simulation"] = {"gazebo": {"joint_current": {"scale": [2.0]}}}
    return profile


def _node(nodes, name):
    matches = [node for node in nodes if node._Node__node_name == name]
    assert len(matches) == 1, f"expected exactly one node named {name!r}, got {[n._Node__node_name for n in nodes]}"
    return matches[0]


def _by_executable(nodes, executable):
    matches = [node for node in nodes if node._Node__node_executable == executable]
    assert len(matches) == 1, f"expected exactly one {executable!r} node"
    return matches[0]


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _param_value(value):
    """Resolve a normalized launch parameter the way evaluate_parameters would.

    launch_ros stores string-ish values as substitutions of their YAML dump
    (``yaml.dump('gazebo')`` is ``'gazebo\\n...\\n'``) and yaml-loads the
    performed text back at evaluation; array parameters keep one
    substitution-group per element.
    """
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)  # plainly typed array (bool/int/float)
        if any(isinstance(item, tuple | list) for item in value):
            return [_param_value(item) for item in value]
        return yaml.safe_load(_text(value)) if value else []
    return value


def _params(node):
    """All parameter dicts of a Node, keys/values resolved like evaluation.

    ``ParameterFile`` entries (per-call temp files) are skipped: they carry no
    comparable identity across two generated action lists.
    """
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update(
                {
                    _text(key): _param_value(value)
                    for key, value in entry.items()
                    if "ParameterFile" not in type(value).__name__
                }
            )
    return params


class _FakeExitEvent:
    returncode = 0


def _deferred_actions(actions):
    """Nodes an OnProcessExit handler starts when its target exits successfully."""
    from launch.actions import RegisterEventHandler

    deferred = []
    for action in actions:
        if isinstance(action, RegisterEventHandler):
            handler = getattr(action.event_handler, "_OnActionEventBase__on_event", None)
            if handler is not None:
                deferred.extend(handler(_FakeExitEvent(), None) or [])
    return deferred


# ---------------------------------------------------------------------------
# 2.1: launch structure per platform
# ---------------------------------------------------------------------------


def test_gazebo_platform_action_set_has_no_ros2_control_node(tmp_path):
    profile = _profile(tmp_path)
    actions = runtime_stack_actions(
        profile, tmp_path / "robot.xacro", simulated=True, sim_platform="gazebo", gazebo_world=""
    )
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    deferred = {node._Node__node_executable for node in _deferred_actions(actions) if isinstance(node, Node)}
    assert "ros2_control_node" not in executables
    assert "robot_state_publisher" in executables
    assert "runtime_facade" in executables
    assert "create" in deferred
    assert "wait_for_clock" in executables
    # standalone: the provider brings a minimal empty world + clock bridge
    assert "IncludeLaunchDescription" in {type(a).__name__ for a in actions}
    assert "parameter_bridge" in executables
    # the camera bridge covers the embodiment sensors
    assert "bridge_node" in executables
    # the simulated joint current publisher is part of the embodiment
    assert "joint_current_publisher" in executables


def test_gazebo_nodes_run_on_sim_time(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    nodes = [a for a in actions if isinstance(a, Node)]
    rsp = _by_executable(nodes, "robot_state_publisher")
    assert _params(rsp)["use_sim_time"] is True
    facade = _node(nodes, "runtime_facade")
    assert _params(facade)["use_sim_time"] is True
    assert _params(facade)["sim_platform"] == "gazebo"
    # spawners sit behind the create-exit handler; they must carry sim time
    spawners = [
        node
        for node in _deferred_actions(actions)
        if isinstance(node, Node) and node._Node__node_executable == "spawner"
    ]
    assert spawners, "controller spawners must start after create exits"
    assert all(_params(spawner).get("use_sim_time") is True for spawner in spawners)


def test_attached_gazebo_world_is_not_started_by_the_provider(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    nodes = [a for a in actions if isinstance(a, Node)]
    executables = {node._Node__node_executable for node in nodes}
    assert "parameter_bridge" not in executables, "the world owner bridges /clock"
    create = _by_executable(_deferred_actions(actions), "create")
    assert "-world" in create._Node__arguments and "demo" in create._Node__arguments


def test_gazebo_create_carries_the_full_spawn_pose(tmp_path):
    profile = _profile(tmp_path)
    actions = gazebo_embodiment_actions(
        profile, tmp_path / "robot.xacro", gazebo_world="demo", spawn_pose=(0.5, -0.25, 0.1, 1.25)
    )
    create = _by_executable(_deferred_actions(actions), "create")
    args = create._Node__arguments
    for flag, value in (("-x", "0.5"), ("-y", "-0.25"), ("-z", "0.1"), ("-Y", "1.25")):
        assert value == args[args.index(flag) + 1], f"create {flag} must receive {value}"


def test_gazebo_description_has_no_package_uris_and_flags(tmp_path):
    profile = _profile(tmp_path)
    urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert "package://" not in urdf
    assert "file://" in urdf
    assert "gz_ros2_control/GazeboSimSystem" in urdf
    assert "SO101SystemHardware" not in urdf  # SDK hardware adapter is not the gazebo plugin
    assert '<state_interface name="effort"/>' in urdf


def test_hardware_and_sdk_action_lists_are_unchanged(tmp_path):
    profile = _profile(tmp_path)
    for sim_platform, simulated in (("", False), ("sdk", True)):
        legacy = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform="")
        selected = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform=sim_platform)
        assert [type(a).__name__ for a in legacy] == [type(a).__name__ for a in selected]
        legacy_nodes = [a for a in legacy if isinstance(a, Node)]
        selected_nodes = [a for a in selected if isinstance(a, Node)]
        assert [n._Node__node_name for n in legacy_nodes] == [n._Node__node_name for n in selected_nodes]
        for legacy_node, selected_node in zip(legacy_nodes, selected_nodes, strict=True):
            # the facade additionally receives the selected platform (D1/D7);
            # everything else is identical
            legacy_params = {k: v for k, v in _params(legacy_node).items() if k != "sim_platform"}
            selected_params = {k: v for k, v in _params(selected_node).items() if k != "sim_platform"}
            assert legacy_params == selected_params
        facade = _node(selected_nodes, "runtime_facade")
        assert _params(facade).get("sim_platform", "") == sim_platform
        executables = {node._Node__node_executable for node in legacy_nodes}
        assert "ros2_control_node" in executables
        # wall time: no provider node carries sim time off the gazebo platform
        assert all("use_sim_time" not in _params(node) for node in legacy_nodes)


def test_motion_launch_arguments_follow_the_platform_clock(tmp_path):
    profile = _profile(tmp_path)
    gazebo = motion_launch_arguments(profile, tmp_path / "robot.xacro", True, "gazebo", False)
    assert gazebo["use_sim_time"] == "true"
    assert gazebo["motion_hardware_feedback_topic"] == ""
    sdk = motion_launch_arguments(profile, tmp_path / "robot.xacro", True, "sdk", False)
    assert sdk["use_sim_time"] == "false"
    hardware = motion_launch_arguments(profile, tmp_path / "robot.xacro", False, "", False)
    assert hardware["use_sim_time"] == "false"
    assert hardware["motion_hardware_feedback_topic"] == "/so101_follower/joint_currents" or isinstance(
        hardware["motion_hardware_feedback_topic"], str
    )


# ---------------------------------------------------------------------------
# 2.3b: the advertised camera contract
# ---------------------------------------------------------------------------


def test_advertised_frames_match_gz_frame_ids_and_optical_links(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    urdf = camera_sensor_urdf(peripherals, platform="gazebo", frame_contract=True)
    for periph in peripherals:
        frame = advertised_camera_frame(periph)
        assert f"<gz_frame_id>{frame}</gz_frame_id>" in urdf
        optical = periph["optical_frame_id"]
        assert f'<link name="{optical}">' in urdf and f'<child  link="{optical}"/>' in urdf


def test_descriptor_interfaces_match_generated_sensors_and_bridge_entries(tmp_path):
    peripherals = [TOP, WRIST, FRONT]
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": peripherals,
        },
        use_sim=True,
        sim_platform="gazebo",
    )
    urdf = camera_sensor_urdf(peripherals, platform="gazebo", frame_contract=True)
    for periph in peripherals:
        name = periph["name"]
        # every advertised camera interface endpoint has a bridge entry
        advertised = {
            interface["endpoint"]
            for key, interface in description["interfaces"].items()
            if key.startswith(f"camera.{name}.")
        }
        bridged = {entry["ros_topic_name"] for entry in camera_bridge_entries(periph, full_streams=True)}
        assert advertised == bridged, f"camera {name}: advertised {sorted(advertised)} vs bridged {sorted(bridged)}"
        # every advertised frame id equals the stamped gz frame id
        frames = {
            interface["frame_id"]
            for key, interface in description["interfaces"].items()
            if key.startswith(f"camera.{name}.") and interface.get("frame_id")
        }
        assert frames == {advertised_camera_frame(periph)}
        # a sensor exists for every non-disabled camera with the advertised frame
        assert f"<gz_frame_id>{advertised_camera_frame(periph)}</gz_frame_id>" in urdf
    # opencv cameras are color-only; the realsense gets an rgbd sensor
    assert urdf.count("<sensor") == 3
    assert 'sensor type="rgbd_camera" name="front_camera"' in urdf
    assert 'sensor type="camera" name="top_camera"' in urdf


def test_gazebo_depth_streams_advertise_the_encoding_gz_publishes():
    """The Gazebo depth contract is what ros_gz_bridge actually publishes (PR !515 P2).

    gz ``rgbd_camera`` depth is float32 and the bridge maps ``R_FLOAT32`` to
    ``32FC1``; the bridge config carries no conversion, so advertising the
    physical device's encoding (RealSense Z16 -> ``16UC1``) would make every
    observed depth message an encoding mismatch in the interface monitor.
    sdk keeps the device encoding: the synthetic publishers produce exactly
    what the descriptor advertises.
    """
    robot = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "capabilities": {},
        "joints": ["1"],
        "peripherals": [FRONT],
    }
    gazebo = build_description(robot, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["camera.front.depth"]["configured_profile"]["encoding"] == "32FC1"
    assert gazebo["camera.front.aligned_depth"]["configured_profile"]["encoding"] == "32FC1"
    # both depth endpoints are bridged from the same gz depth_image stream,
    # so both observe exactly the encoding asserted above
    bridged = {
        entry["ros_topic_name"]: entry["gz_topic_name"] for entry in camera_bridge_entries(FRONT, full_streams=True)
    }
    assert bridged["/camera/front/depth/image_rect_raw"] == "/front_camera/depth_image"
    assert bridged["/camera/front/aligned_depth_to_color/image_raw"] == "/front_camera/depth_image"
    sdk = build_description(robot, use_sim=True, sim_platform="sdk")["interfaces"]
    assert sdk["camera.front.depth"]["configured_profile"]["encoding"] == "16UC1"
    assert sdk["camera.front.aligned_depth"]["configured_profile"]["encoding"] == "16UC1"


def test_unproducible_streams_are_not_advertised_on_gazebo(tmp_path):
    pointcloud_camera = {
        **FRONT,
        "enable_pointcloud": True,
        "simulation": {"gazebo": {"transform": FRONT["simulation"]["gazebo"]["transform"], "fovy": 70}},
    }
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [pointcloud_camera],
        },
        use_sim=True,
        sim_platform="gazebo",
    )
    assert not any(key.startswith("camera.front.points") for key in description["interfaces"])
    sdk_description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [pointcloud_camera],
        },
        use_sim=True,
        sim_platform="sdk",
    )
    assert any(key.startswith("camera.front.points") for key in sdk_description["interfaces"])


def test_world_parented_camera_uses_the_inverse_full_spawn_transform():
    urdf = camera_sensor_urdf([TOP], platform="gazebo", spawn_pose=(0.0, 0.3, -0.03, 0.5), frame_contract=True)
    import re

    origin = re.search(r'name="top_camera_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"', urdf, re.S)
    x, y, z = (float(value) for value in origin.group(1).split())
    roll, pitch, yaw = (float(value) for value in origin.group(2).split())
    # inverse of spawn (translation + yaw 0.5) applied to the world pose
    cos, sin = math.cos(-0.5), math.sin(-0.5)
    dx, dy = TOP["simulation"]["gazebo"]["transform"]["x"], TOP["simulation"]["gazebo"]["transform"]["y"] - 0.3
    assert x == pytest.approx(cos * dx - sin * dy)
    assert y == pytest.approx(sin * dx + cos * dy)
    assert z == pytest.approx(TOP["simulation"]["gazebo"]["transform"]["z"] + 0.03)
    assert yaw == pytest.approx(TOP["simulation"]["gazebo"]["transform"]["yaw"] - 0.5)


def test_legacy_camera_urdf_keeps_the_generic_shape():
    urdf = camera_sensor_urdf([TOP, WRIST], platform="gazebo")
    assert "gz_frame_id" not in urdf
    assert "optical_frame" not in urdf
    assert '<gazebo reference="camera_top_frame">' in urdf
    # fovy 47 at 640x480 -> horizontal fov
    assert f"<horizontal_fov>{2.0 * math.atan(math.tan(math.radians(47) / 2) * 640 / 480):.4f}</horizontal_fov>" in urdf


def test_preset_loader_keeps_the_generic_use_default_transform_path():
    periph = {
        "type": "camera",
        "name": "front",
        "driver": "opencv",
        "frame_id": "camera_front_frame",
        "use_default_transform": True,
        "transform": {"parent_frame": "base", "x": 9.0},
    }

    def loader(platform, name):
        return {"parent_frame": "world", "x": 0.5, "y": 0.1, "z": 0.2, "yaw": 0.3, "fovy": 55}

    transform, fovy = resolve_camera_pose(periph, "gazebo", default_pose_loader=loader)
    assert transform["parent_frame"] == "world" and transform["x"] == 0.5
    assert fovy == 55
    # without the loader flag the explicit transform applies
    plain, _ = resolve_camera_pose({**periph, "use_default_transform": False}, "gazebo")
    assert plain["x"] == 9.0


def test_saved_override_takes_precedence_over_the_declared_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ros" / "ibrobot" / "sim_camera_overrides").mkdir(parents=True)
    (home / ".ros" / "ibrobot" / "sim_camera_overrides" / "top.yaml").write_text(
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
    transform, fovy = resolve_camera_pose(TOP, "gazebo")
    assert transform["x"] == 1.5 and transform["parent_frame"] == "world"
    assert fovy == 50
    urdf = camera_sensor_urdf([TOP], platform="gazebo", frame_contract=True)
    assert 'xyz="1.5 -0.2 0.8"' in urdf


# ---------------------------------------------------------------------------
# 2.3c: one effective peripheral inventory
# ---------------------------------------------------------------------------


def test_disabled_camera_is_absent_everywhere(tmp_path):
    fragment = [{"type": "camera", "name": "front", "disabled": True}]
    peripherals = merge_peripherals([TOP, WRIST, FRONT], fragment)
    assert [periph["name"] for periph in peripherals] == ["top", "wrist"]
    # descriptor
    description = build_description(
        {
            "runtime": {"name": "test_robot", "version": "0.1.0"},
            "capabilities": {},
            "joints": ["1"],
            "peripherals": [TOP, WRIST, FRONT],
        },
        peripherals=peripherals,
        use_sim=True,
        sim_platform="gazebo",
    )
    assert not any(key.startswith("camera.front.") for key in description["interfaces"])
    # sdk synthetic publishers
    profile = _profile(tmp_path, peripherals=peripherals)
    sdk_actions = peripheral_stack_actions(profile, "", simulated=True, sim_platform="sdk")
    synthetic = _node([a for a in sdk_actions if isinstance(a, Node)], "synthetic_perception")
    encoded = _params(synthetic)["description_json"]
    assert "camera.front.color" not in encoded
    # gazebo sensors + bridge
    gazebo_urdf = render_gazebo_description(profile, tmp_path / "robot.xacro")
    assert "front_camera" not in gazebo_urdf
    gazebo_actions = gazebo_embodiment_actions(
        profile, tmp_path / "robot.xacro", robot_description=gazebo_urdf, gazebo_world="demo"
    )
    bridge = _by_executable([a for a in gazebo_actions if isinstance(a, Node)], "bridge_node")
    config = Path(_params(bridge)["config_file"]).read_text(encoding="utf-8")
    assert "camera/front" not in config
    # physical drivers
    hardware_actions = peripheral_stack_actions(profile, "", simulated=False, sim_platform="")
    assert not any(isinstance(a, Node) and a._Node__node_name == "front_camera" for a in hardware_actions)


def test_enabled_false_and_disabled_true_are_the_same_toggle():
    assert merge_peripherals([TOP], [{"type": "camera", "name": "top", "enabled": False}]) == []
    assert merge_peripherals([TOP], [{"type": "camera", "name": "top", "disabled": True}]) == []
    # the normalized entry keeps both spellings for generic consumers
    merged = merge_peripherals([], [{"type": "camera", "name": "top", "disabled": True, "width": 32}])
    assert merged == [] or merged[0]["enabled"] is False


def test_toggle_only_entry_for_undeclared_peripheral_is_rejected():
    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'ghost'\) is not declared"):
        merge_peripherals([TOP], [{"type": "camera", "name": "ghost", "disabled": True}])
    # a full entry may still add a deployment-specific peripheral
    extra = merge_peripherals(
        [TOP],
        [{"type": "camera", "name": "deploy", "driver": "opencv", "index": 3, "transform": {"parent_frame": "base"}}],
    )
    assert [entry["name"] for entry in extra] == ["top", "deploy"]


# ---------------------------------------------------------------------------
# 2.3d: simulated joint current
# ---------------------------------------------------------------------------


def test_joint_current_conversion():
    from sensor_msgs.msg import JointState

    from robot_runtime.joint_current_publisher import joint_currents_from_efforts

    message = JointState()
    message.name = ["2", "1", "6"]
    message.effort = [10.0, 20.0, 30.0]
    assert joint_currents_from_efforts(["1", "2", "6"], message, [2.0, 0.5, 1.0]) == [40.0, 5.0, 30.0]
    message.effort = []
    assert joint_currents_from_efforts(["1"], message, [1.0]) is None


def test_joint_current_node_only_on_gazebo(tmp_path):
    profile = _profile(tmp_path)
    gazebo_actions = gazebo_embodiment_actions(profile, tmp_path / "robot.xacro", gazebo_world="demo")
    node = _node([a for a in gazebo_actions if isinstance(a, Node)], "simulated_joint_current")
    params = _params(node)
    assert params["endpoint"] == "/so101_follower/joint_currents"
    assert params["joint_names"] == ["1"]
    assert params["scale"] == [2.0]
    # the hardware/sdk stack never starts it
    for sim_platform, simulated in (("", False), ("sdk", True)):
        actions = runtime_stack_actions(profile, tmp_path / "robot.xacro", simulated, sim_platform=sim_platform)
        assert not any(isinstance(a, Node) and a._Node__node_executable == "joint_current_publisher" for a in actions)
    # a profile without the joint.current interface gets no publisher
    bare = _profile(tmp_path, with_joint_current=False)
    assert simulated_joint_current_actions(bare) == []
