"""Peripheral bridge node generators for Gazebo simulation.

Uses ros_gz_bridge bridge_node with a YAML config file to map Gazebo sensor
topics directly to the ROS topic naming contract — no remappings= needed.

Why bridge_node instead of parameter_bridge + remappings=:
  parameter_bridge with Node(remappings=...) causes the process to start but
  fail to register with the ROS 2 daemon (processes appear in ps aux but not
  in ros2 node list), so no topics are published. bridge_node with a YAML
  config avoids this by specifying the ROS topic name directly in the config.

Topic naming contract (output ROS topics):
  /camera/{name}/image_raw
  /camera/{name}/camera_info
  /camera/{name}/aligned_depth_to_color/image_raw  (when depth is enabled)
where {name} is YAML peripherals[].name (e.g. "top", "wrist").

This contract matches real hardware: perception.py remaps both
usb_cam and realsense2_camera outputs to the same topic names.

Only cameras bridge here (sim-platform-registry D9): the generic
provider-less path never generates simulated lidar or IMU sensors, so their
bridge builders were dead code. The runtime providers' lidar/IMU bridging
lives in ``robot_runtime_sim.gazebo``.
"""

import os
import tempfile

import yaml
from launch_ros.actions import Node

# Prefix of the per-construction bridge config (unique file per call:
# concurrent launches on different domains or worktrees must not overwrite
# each other's config, sim-platform-registry D9).
_BRIDGE_CONFIG_PREFIX = "ros_gz_camera_bridge_"


def _build_camera_bridge_entries(periph: dict, model_name: str, world_name: str) -> list:
    """Return bridge config dicts for one camera peripheral.

    Thin re-import wrapper (openspec ``so101-gazebo-runtime-transport`` D6.1):
    the robot-agnostic builder lives in ``robot_runtime_sim.gazebo``; this call
    site keeps the generic provider-less behaviour (color, color info and the
    legacy aligned-depth entry; the runtime providers pass ``full_streams`` to
    bridge every advertised stream).
    """
    from robot_runtime_sim.gazebo import camera_bridge_entries

    return camera_bridge_entries(periph)


# Registry: peripheral type → bridge entry builder function.
_PERIPHERAL_BRIDGE_BUILDERS = {
    "camera": _build_camera_bridge_entries,
}


def _bridge_config_path() -> str:
    """A unique bridge config path (one file per construction, D9)."""
    fd, path = tempfile.mkstemp(prefix=_BRIDGE_CONFIG_PREFIX, suffix=".yaml")
    os.close(fd)
    return path


def generate_peripheral_sim_bridges(
    peripherals: list,
    model_name: str,
    world_name: str = "demo",
) -> list:
    """Generate a single bridge_node covering all peripheral cameras.

    Writes the YAML config to a unique temp file and returns one bridge_node
    Node that reads it. bridge_node maps each Gazebo sensor topic directly to
    its ROS contract name — no remappings= required.

    Topic naming contract (output ROS topics):
      /camera/{name}/image_raw
      /camera/{name}/camera_info
      /camera/{name}/aligned_depth_to_color/image_raw when the camera requests
      depth via streams, align_depth, enable_depth, or enable_pointcloud.
    where {name} is YAML peripherals[].name.

    Args:
        peripherals: List of peripheral dicts from robot_config["peripherals"].
        model_name:  Gazebo model name (robot_config["name"]).
        world_name:  Gazebo world name, default "demo" (matches the sim_models scene worlds).

    Returns:
        List with one bridge_node Node action, or [] if no known peripherals.
    """
    bridge_entries = []
    for periph in peripherals:
        ptype = periph.get("type")
        builder = _PERIPHERAL_BRIDGE_BUILDERS.get(ptype)
        if builder:
            bridge_entries.extend(builder(periph, model_name, world_name))

    if not bridge_entries:
        return []

    config_path = _bridge_config_path()
    with open(config_path, "w") as f:
        yaml.dump(bridge_entries, f, default_flow_style=False)

    return [
        Node(
            package="ros_gz_bridge",
            executable="bridge_node",
            name="peripheral_bridge",
            parameters=[{"config_file": config_path}],
            output="screen",
        )
    ]
