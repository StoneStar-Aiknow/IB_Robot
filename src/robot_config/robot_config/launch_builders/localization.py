"""Localization builder for navigation.

Generates EKF + RTAB-Map stack nodes:
- RTAB-Map SLAM
- EKF sensor fusion
"""

import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import prepare_writable_file_path, resolve_ros_path

logger = get_colored_logger("robot_config.localization")


def generate_localization_nodes(
    nav_config: dict[str, Any],
    use_sim: bool = False,
) -> list:
    """Generate localization nodes (EKF + RTAB-Map stack).

    Args:
        nav_config: navigation section from robot_config YAML
        use_sim: simulation mode flag

    Returns:
        List of Node actions
    """
    nodes = []

    slam_toolbox_config = nav_config.get("slam_toolbox", {})
    if slam_toolbox_config.get("enabled", False) and not use_sim:
        params_file = slam_toolbox_config.get("params_file", "")
        parameters = [resolve_ros_path(params_file)] if params_file else []
        parameters.append(
            {
                "use_sim_time": use_sim,
                "base_frame": slam_toolbox_config.get("base_frame", "base_link"),
                "odom_frame": slam_toolbox_config.get("odom_frame", "odom"),
                "map_frame": slam_toolbox_config.get("map_frame", "map"),
                "minimum_travel_distance": slam_toolbox_config.get("minimum_travel_distance", 0.5),
                "minimum_travel_heading": slam_toolbox_config.get("minimum_travel_heading", 0.5),
                "map_update_interval": slam_toolbox_config.get("map_update_interval", 10.0),
                "min_laser_range": slam_toolbox_config.get("min_laser_range", 0.0),
                "max_laser_range": slam_toolbox_config.get("max_laser_range", 25.0),
            }
        )
        nodes.append(
            Node(
                package="slam_toolbox",
                executable=slam_toolbox_config.get("executable", "async_slam_toolbox_node"),
                name=slam_toolbox_config.get("node_name", "slam_toolbox"),
                output="screen",
                parameters=parameters,
                remappings=[("scan", slam_toolbox_config.get("scan_topic", "/scan"))],
            )
        )
        logger.info("Added slam_toolbox node")

    ekf_rtabmap_config = nav_config.get("ekf_rtabmap", {})
    ekf_rtabmap_enabled = ekf_rtabmap_config.get("enabled", False)

    if not (ekf_rtabmap_enabled and not use_sim):
        return nodes

    try:
        # RTAB-Map
        rtabmap_config = ekf_rtabmap_config.get("rtabmap", {})
        rtabmap_dir = get_package_share_directory("rtabmap_launch")
        rtabmap_args = {
            "use_sim_time": "false",
            "localization": str(rtabmap_config.get("localization", True)).lower(),
            "rtabmap_viz": "false",
        }
        for key in [
            "rgb_topic",
            "depth_topic",
            "camera_info_topic",
            "database_path",
            "frame_id",
            "odom_frame_id",
            "odom_topic",
            "visual_odometry",
            "odom_args",
            "rtabmap_args",
            "approx_sync",
            "queue_size",
            "qos_image",
            "qos_camera_info",
            "qos_odom",
            "log_level",
        ]:
            val = rtabmap_config.get(key)
            if val is not None:
                if key == "database_path":
                    val = prepare_writable_file_path(val)
                rtabmap_args[key] = str(val)

        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(rtabmap_dir, "launch", "rtabmap.launch.py")),
                launch_arguments=rtabmap_args.items(),
            )
        )
        logger.info(f"Added RTAB-Map launch (localization: {rtabmap_args.get('localization')})")

        # Depth-to-laserscan
        dtl_config = ekf_rtabmap_config.get("depth_to_laserscan", {})
        if dtl_config.get("enabled", False):
            dtl_params = {
                "scan_height": dtl_config.get("scan_height", 10),
                "scan_time": dtl_config.get("scan_time", 0.033),
                "range_min": dtl_config.get("range_min", 0.0),
                "range_max": dtl_config.get("range_max", 10.0),
                "output_frame": dtl_config.get("output_frame", "camera_link"),
            }
            depth_topic = dtl_config.get("depth_topic", "/camera/realsense/depth/image_rect_raw")
            depth_camera_info_topic = dtl_config.get(
                "depth_camera_info_topic",
                "/camera/realsense/camera_info",
            )
            nodes.append(
                Node(
                    package="depthimage_to_laserscan",
                    executable="depthimage_to_laserscan_node",
                    name="depth_to_laserscan",
                    output="screen",
                    parameters=[dtl_params],
                    remappings=[
                        ("depth", depth_topic),
                        ("depth_camera_info", depth_camera_info_topic),
                        ("scan", "/scan"),
                    ],
                )
            )
            logger.info("Added depthimage_to_laserscan node")

        # EKF node
        ekf_node_config = ekf_rtabmap_config.get("ekf", {})
        if not ekf_node_config.get("enabled", True):
            logger.info("EKF disabled by config")
            logger.info("RTAB-Map stack enabled")
            return nodes

        ekf_config_file = ekf_node_config.get("config_file", "")
        if ekf_config_file:
            ekf_config_file = resolve_ros_path(ekf_config_file)
        if not ekf_config_file:
            try:
                robot_navigation_share = get_package_share_directory("robot_navigation")
                ekf_config_file = os.path.join(robot_navigation_share, "config", "ekf.yaml")
            except Exception:
                ekf_config_file = ""

        if ekf_config_file:
            nodes.append(
                Node(
                    package="robot_localization",
                    executable="ekf_node",
                    name="ekf_filter_node",
                    output="screen",
                    parameters=[ekf_config_file],
                )
            )
            logger.info(f"Added EKF node (config: {ekf_config_file})")

        logger.info("EKF + RTAB-Map stack enabled")

    except Exception as e:
        logger.error(f"Failed to add required EKF+RTAB-Map nodes: {e}")
        raise

    return nodes
