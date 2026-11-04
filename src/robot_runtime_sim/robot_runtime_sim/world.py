"""The shared Gazebo world launcher (openspec sim-platform-registry D6).

One implementation of the Gazebo world start: the ``gz_sim.launch.py``
include for a world file, the ``/clock`` parameter bridge and the
``IBROBOT_GAZEBO_HEADLESS`` server-only switch. Every world owner uses it —

- the Gazebo embodiment's provider-alone empty world
  (``gazebo_embodiment.py``);
- the Gazebo world hook of generic launch (``world_hooks.py``);
- the provider-less ``GazeboAdapter.start_backend`` in robot_config.
"""

from __future__ import annotations

import logging
import os

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

logger = logging.getLogger(__name__)


def _gz_server_include(gz_args: str):
    """The ros_gz_sim server launch with the given gz_args."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                ]
            )
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )


def _headless_prefix() -> str:
    """IBROBOT_GAZEBO_HEADLESS=1 switches to server-only headless mode.

    Used by sim_camera_adjuster when the primary simulator is MuJoCo and a
    parallel Gazebo stack is spawned purely for camera pose calibration.
    Default (env unset) preserves the normal Gazebo UX exactly.
    """
    if os.environ.get("IBROBOT_GAZEBO_HEADLESS", "0") == "1":
        logger.info("IBROBOT_GAZEBO_HEADLESS=1 -> Gazebo server-only headless mode")
        return "-s --headless-rendering "
    return ""


def gazebo_world_actions(world_file: str, world_name: str) -> list:
    """Start one Gazebo world: the server include plus the ``/clock`` bridge.

    Args:
        world_file: the ``.world``/``.sdf`` file the server runs (``-r``).
        world_name: the gz world instance name, used only in the log line
            that records the world every ``create`` must target.

    Returns:
        The launch actions: the ``gz_sim.launch.py`` include with the
        headless-prefixed ``-r <world_file>`` gz_args, and the
        ``/clock@rosgraph_msgs/msg/Clock`` parameter bridge.
    """
    gz_args = f"{_headless_prefix()}-r {world_file}"
    logger.info(f"Gazebo world actions: world={world_name!r}, file={world_file}")
    return [
        _gz_server_include(gz_args),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        ),
    ]
