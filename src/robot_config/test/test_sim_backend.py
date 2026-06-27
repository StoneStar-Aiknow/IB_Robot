"""Unit tests for sim_backend adapter registry and sim_peripheral_bridge.

Tests cover both Gazebo and MuJoCo backends without requiring
a ROS or Gazebo runtime environment.
"""

import pytest

from robot_config.launch_builders.sim_backend import (
    SimBackendAdapter,
    get_backend_caps,
    get_sim_backend,
)
from robot_config.launch_builders.sim_backend.gazebo_adapter import GazeboAdapter
from robot_config.launch_builders.sim_backend.mock_adapter import MockAdapter
from robot_config.launch_builders.sim_backend.mujoco_adapter import MujocoAdapter

# ──────────────────────────────────────────────────────────────────────────────
# Registry tests (unchanged from T2)
# ──────────────────────────────────────────────────────────────────────────────


def test_registry_gazebo():
    """get_sim_backend('gazebo') returns a GazeboAdapter instance."""
    adapter = get_sim_backend("gazebo")
    assert isinstance(adapter, GazeboAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_mujoco():
    """get_sim_backend('mujoco') returns a MujocoAdapter instance."""
    adapter = get_sim_backend("mujoco")
    assert isinstance(adapter, MujocoAdapter)
    assert isinstance(adapter, SimBackendAdapter)


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
# MuJoCo no-op tests
# ──────────────────────────────────────────────────────────────────────────────


def test_mujoco_noop_methods_return_empty():
    """MuJoCo no-op adapter methods return empty launch action lists."""
    adapter = get_sim_backend("mujoco")

    assert adapter.load_scene("fake.xml") == []
    assert adapter.ensure_controller_manager({}) == []
    assert adapter.update_object_pose("box", None) is None


def test_mujoco_spawn_peripheral_bridges_returns_empty():
    """MujocoAdapter.spawn_peripheral_bridges returns [] (no bridge nodes needed).

    MuJoCo publishes ROS2 topics directly. T6 will configure topic names
    to match the contract /camera/{name}/image_raw without extra bridge nodes.
    """
    adapter = get_sim_backend("mujoco")
    result = adapter.spawn_peripheral_bridges([])
    assert result == []
    # Also works with non-empty peripherals list
    result = adapter.spawn_peripheral_bridges(
        [
            {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
        ]
    )
    assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# sim_peripheral_bridge tests (T3)
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


def test_sim_peripheral_bridge_lidar_and_imu_supported():
    """LiDAR + IMU peripherals share one bridge_node in Gazebo."""
    from launch_ros.actions import Node

    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "lidar", "name": "main_lidar", "params": {"laser_scan_topic_name": "/scan"}},
        {"type": "imu", "name": "imu_sensor", "topic": "/imu_sensor_broadcaster/imu"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_mock_start_backend_reuses_contract_mock():
    """Mock backend starts hardware_mock directly and has no spawn node."""
    from launch_ros.actions import Node

    adapter = get_sim_backend("mock")
    actions, create_node = adapter.start_backend({"_config_path": "/tmp/robot.yaml"})

    assert create_node is None
    assert len(actions) == 1
    assert isinstance(actions[0], Node)
    assert actions[0].node_package == "hardware_mock"


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
