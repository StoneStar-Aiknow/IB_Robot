"""Nav2 bringup builder for navigation.

Generates Nav2 stack IncludeLaunchDescription from navigation config.
"""

import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.nav2")

BASE_NAV_ARGS = {
    "controller_plugin": "dwb_core::DWBLocalPlanner",
    "local_costmap_plugin": "nav2_costmap_2d::ObstacleLayer",
    "observation_sources": "scan",
    "use_collision_monitor": "false",
    "dwb_critics_key": "critics",
    "mppi_critics_key": "mppi_critics",
}

DYN_AVOID_ARGS = {
    "controller_plugin": "nav2_mppi_controller::MPPIController",
    "local_costmap_plugin": "nav2_costmap_2d::VoxelLayer",
    "observation_sources": "pointcloud",
    "use_collision_monitor": "true",
    "dwb_critics_key": "dwb_critics",
    "mppi_critics_key": "critics",
}


def get_avoidance_launch_arguments(dynamic_enabled: bool) -> dict[str, str]:
    """Map the short user switch to the validated baseline or dynamic chain."""
    return (DYN_AVOID_ARGS if dynamic_enabled else BASE_NAV_ARGS).copy()


def generate_nav2_nodes(
    nav_config: dict[str, Any],
    use_sim: bool = False,
) -> list:
    """Generate Nav2 bringup nodes.

    Args:
        nav_config: navigation section from robot_config YAML
        use_sim: simulation mode flag

    Returns:
        List of launch actions
    """
    nodes = []

    nav2_config = nav_config.get("nav2_bringup", {})
    if not nav2_config.get("enabled", False):
        return nodes

    try:
        robot_navigation_share = get_package_share_directory("robot_navigation")
        nav2_launch_file = os.path.join(robot_navigation_share, "launch", "nav2_bringup.launch.py")

        # Resolve map file
        map_file = nav2_config.get("map_file", "")
        if not map_file:
            map_file = os.path.expanduser("~/.ros/ibrobot/maps/rtabmap.yaml")
        else:
            map_file = resolve_ros_path(map_file)

        # Resolve params file
        params_file = nav2_config.get("params_file", "")
        if not params_file:
            try:
                params_file = os.path.join(robot_navigation_share, "config", "nav2_params.yaml")
            except Exception:
                params_file = ""
        else:
            params_file = resolve_ros_path(params_file)

        launch_args = {
            "use_sim_time": str(use_sim).lower(),
            "map": map_file,
            "namespace": "",
            "use_namespace": "false",
            "use_composition": "False",
            "use_amcl": str(nav2_config.get("use_amcl", False)).lower(),
            "auto_global_localization": str(nav2_config.get("auto_global_localization", False)).lower(),
        }
        launch_args.update(get_avoidance_launch_arguments(nav2_config.get("dyn_avoid_enabled", False)))
        if params_file:
            launch_args["params_file"] = params_file

        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch_file),
                launch_arguments=launch_args.items(),
            )
        )
        logger.info(f"Added Nav2 bringup (map: {map_file}, sim: {use_sim})")

    except Exception as e:
        logger.error(f"Failed to add required Nav2 bringup: {e}")
        raise

    return nodes
