"""Unit tests for sim_backend adapter registry and sim_peripheral_bridge.

Tests cover the Gazebo and mock provider-less backends without requiring
a ROS or Gazebo runtime environment. MuJoCo has no adapter here: it is
served through a runtime provider (sim-platform-registry D8); its
world-side hand-off is covered by test_runtime_provider_platform.py and
test_sim_platform_generic_launch.py.
"""

from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.sim_backend import (
    SimBackendAdapter,
    get_backend_caps,
    get_sim_backend,
)
from robot_config.launch_builders.sim_backend.gazebo_adapter import GazeboAdapter
from robot_config.launch_builders.sim_backend.mock_adapter import MockAdapter

# ──────────────────────────────────────────────────────────────────────────────
# Registry tests
# ──────────────────────────────────────────────────────────────────────────────


def test_registry_gazebo():
    """get_sim_backend('gazebo') returns a GazeboAdapter instance."""
    adapter = get_sim_backend("gazebo")
    assert isinstance(adapter, GazeboAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_mujoco_is_not_a_backend():
    """MuJoCo is served through a runtime provider, not this registry (D8)."""
    with pytest.raises(ValueError, match="Unknown sim platform"):
        get_sim_backend("mujoco")


def test_registry_mock():
    """get_sim_backend('mock') returns a MockAdapter instance."""
    adapter = get_sim_backend("mock")
    assert isinstance(adapter, MockAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_unknown_raises():
    """get_sim_backend with an unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sim platform"):
        get_sim_backend("webots")


def test_backend_caps_mock():
    """Mock declares that it has no /clock and no ros2_control manager."""
    assert get_backend_caps("mock") == {"provides_clock": False, "needs_ros2_control": False}
    assert MockAdapter.provides_clock is False
    assert MockAdapter.needs_ros2_control is False


# ──────────────────────────────────────────────────────────────────────────────
# sim_peripheral_bridge tests
# ──────────────────────────────────────────────────────────────────────────────


def test_sim_peripheral_bridge_camera_count():
    """generate_peripheral_sim_bridges returns 1 bridge_node for any camera config."""
    from launch_ros.actions import Node

    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1  # single bridge_node covers all cameras via YAML config
    assert isinstance(nodes[0], Node)


def test_sim_peripheral_bridge_multi_camera():
    """generate_peripheral_sim_bridges returns 1 bridge_node for multiple cameras."""
    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
        {"type": "camera", "name": "wrist", "frame_id": "camera_wrist_frame"},
        {"type": "camera", "name": "front", "frame_id": "camera_front_link"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1  # single bridge_node handles all 3 cameras (6 topics in YAML)


def test_sim_peripheral_bridge_depth_camera_entries():
    """Depth-enabled cameras add an aligned-depth bridge entry."""
    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {
            "type": "camera",
            "name": "front",
            "frame_id": "camera_front_link",
            "align_depth": True,
        },
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1

    bridge_config = yaml.safe_load(Path(_config_file_of(nodes[0])).read_text())
    ros_topics = {entry["ros_topic_name"] for entry in bridge_config}
    gz_topics = {entry["gz_topic_name"] for entry in bridge_config}

    assert "/camera/front/image_raw" in ros_topics
    assert "/camera/front/camera_info" in ros_topics
    assert "/camera/front/aligned_depth_to_color/image_raw" in ros_topics
    assert "/front_camera/depth_image" in gz_topics


def test_sim_peripheral_bridge_writes_a_unique_config_per_construction(tmp_path):
    """Two constructions get two distinct config files with their own content (D9)."""
    import os

    from robot_config.launch_builders.sim_peripheral_bridge import generate_peripheral_sim_bridges

    peripherals = [{"type": "camera", "name": "top", "frame_id": "camera_top_frame"}]
    first = generate_peripheral_sim_bridges(peripherals, model_name="robot_a")
    second = generate_peripheral_sim_bridges(
        [{"type": "camera", "name": "wrist", "frame_id": "camera_wrist_frame"}], model_name="robot_b"
    )
    path_a = _config_file_of(first[0])
    path_b = _config_file_of(second[0])
    assert path_a != path_b, "each construction must own its bridge config file"
    assert os.path.basename(path_a).startswith("ros_gz_camera_bridge_")
    assert os.path.basename(path_b).startswith("ros_gz_camera_bridge_")
    topics_a = {entry["ros_topic_name"] for entry in yaml.safe_load(Path(path_a).read_text())}
    topics_b = {entry["ros_topic_name"] for entry in yaml.safe_load(Path(path_b).read_text())}
    assert topics_a == {"/camera/top/image_raw", "/camera/top/camera_info"}
    assert topics_b == {"/camera/wrist/image_raw", "/camera/wrist/camera_info"}


def test_sim_peripheral_bridge_ignores_lidar_and_imu_peripherals():
    """Only cameras bridge on the generic path (D9); lidar/IMU are dead here."""
    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "lidar", "name": "main_lidar", "params": {"laser_scan_topic_name": "/scan"}},
        {"type": "imu", "name": "imu_sensor", "topic": "/imu_sensor_broadcaster/imu"},
    ]
    assert generate_peripheral_sim_bridges(peripherals, model_name="test_robot") == []


def _config_file_of(node) -> str:
    """The ``config_file`` parameter of a generated bridge_node.

    launch_ros abbreviates parameter values as ``<value>\\n...\\n`` in the
    unevaluated Node; strip that suffix to recover the written path.
    """

    def text(value):
        if isinstance(value, list | tuple):
            return "".join(text(item) for item in value)
        return str(getattr(value, "text", value))

    for entry in node._Node__parameters or []:
        if isinstance(entry, dict) and "config_file" in {text(k) for k in entry}:
            (value,) = entry.values()
            path = text(value)
            return path.removesuffix("\n...\n")
    raise AssertionError("the bridge_node must carry a config_file parameter")


def test_mock_start_backend_reuses_contract_mock():
    """Mock backend starts hardware_mock directly and has no spawn node."""
    from launch_ros.actions import Node

    adapter = get_sim_backend("mock")
    actions, create_node = adapter.start_backend({"_config_path": "/tmp/robot.yaml"})

    assert create_node is None
    assert len(actions) == 1
    assert isinstance(actions[0], Node)
    assert actions[0].node_package == "hardware_mock"


def test_mock_start_backend_warns_about_a_configured_scene(caplog):
    """A configured scene stays a warning: contract mock has no scene objects."""
    adapter = get_sim_backend("mock")
    with caplog.at_level("WARNING", logger="robot_config.sim_backend.mock"):
        actions, _create_node = adapter.start_backend(
            {"_config_path": "/tmp/robot.yaml", "simulation": {"scene": "pick_banana"}}
        )
    assert actions
    assert "pick_banana" in caplog.text and "ignored by mock backend" in caplog.text


def test_gazebo_start_backend_returns_create_node():
    """GazeboAdapter.start_backend returns (actions, ros_gz_sim create Node)."""
    from launch_ros.actions import Node

    adapter = get_sim_backend("gazebo")
    actions, create_node = adapter.start_backend({"name": "test_robot", "gazebo_world_name": "demo"})
    assert isinstance(actions, list)
    assert len(actions) >= 1
    assert isinstance(create_node, Node)


def test_gazebo_adapter_bridge_delegates():
    """GazeboAdapter.spawn_peripheral_bridges returns 1 bridge_node for camera peripherals."""
    adapter = get_sim_backend("gazebo")
    adapter._model_name = "test_robot"
    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
    ]
    nodes = adapter.spawn_peripheral_bridges(peripherals)
    assert len(nodes) == 1


# ──────────────────────────────────────────────────────────────────────────────
# The provider world hand-off through build_simulation (registry-driven, D5)
# ──────────────────────────────────────────────────────────────────────────────


def test_provider_gazebo_simulation_startup_carries_world_and_pose():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext
    from robot_config.runtime_target import RuntimeTarget

    context = LaunchBuildContext(
        robot_config={
            "name": "test_robot",
            "runtime": {"provider": "test_robot", "profile": "p"},
            "simulation": {
                "platform": "gazebo",
                "scene": "pick_banana",
                "robot_spawn": {"x": 0.1, "y": 0.2, "z": 0.3, "yaw": 0.5},
            },
        },
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="gazebo",
        use_sim=True,
    )
    result = build_simulation(context)
    assert result.simulation_startup is not None
    assert result.simulation_startup.entity_action is None
    assert result.simulation_startup.world == "demo"
    assert result.simulation_startup.spawn_pose == (0.1, 0.2, 0.3, 0.5)
    # world-only: no create, no joint-state bridge, no camera bridges
    from launch_ros.actions import Node

    executables = {node._Node__node_executable for node in result.actions if isinstance(node, Node)}
    assert "parameter_bridge" in executables
    assert "create" not in executables and "bridge_node" not in executables


def test_preseeded_simulation_startup_is_reused_without_relaunching():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext, SimulationStartup
    from robot_config.runtime_target import RuntimeTarget

    preseed = SimulationStartup(entity_action=None, world="demo", spawn_pose=(0.0, 0.3, -0.03, 0.0))
    context = LaunchBuildContext(
        robot_config={"name": "test_robot", "runtime": {"provider": "test_robot", "profile": "p"}},
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="gazebo",
        use_sim=True,
        simulation_startup=preseed,
    )
    result = build_simulation(context)
    assert result.actions == [], "the second plan pass must not relaunch the world"
    assert result.simulation_startup is preseed


def test_provider_sdk_simulation_still_yields_the_empty_startup():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext
    from robot_config.runtime_target import RuntimeTarget

    context = LaunchBuildContext(
        robot_config={"name": "test_robot", "runtime": {"provider": "test_robot", "profile": "p"}},
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="sdk",
        use_sim=True,
    )
    result = build_simulation(context)
    assert result.actions == []
    assert result.simulation_startup.entity_action is None
    assert result.simulation_startup.world == ""
