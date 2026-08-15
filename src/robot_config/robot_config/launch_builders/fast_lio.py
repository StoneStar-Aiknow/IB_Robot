"""FAST-LIO ROS 2 launch integration."""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.fast_lio")


def generate_fast_lio_nodes(nav_config: dict[str, Any], use_sim: bool = False) -> list:
    """Generate the official FAST-LIO node for enabled navigation stages."""
    config = nav_config.get("fast_lio", {})
    if not config.get("enabled", False) or use_sim:
        return []

    params_file = config.get("params_file", "")
    if not params_file:
        raise ValueError("navigation.fast_lio.params_file is required")

    raw_odom_topic = config.get("raw_odom_topic", "/fast_lio/odometry_raw")
    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fast_lio",
        output="screen",
        parameters=[
            resolve_ros_path(params_file),
            {"use_sim_time": False},
        ],
        remappings=[
            ("/Odometry", raw_odom_topic),
            ("/livox/lidar", config.get("lidar_topic", "/livox/lidar")),
            ("/livox/imu", config.get("imu_topic", "/livox/imu")),
            ("/tf", config.get("isolated_tf_topic", "/fast_lio/tf_raw")),
            ("/tf_static", config.get("isolated_tf_static_topic", "/fast_lio/tf_static_raw")),
        ],
        respawn=bool(config.get("respawn", True)),
        respawn_delay=float(config.get("respawn_delay", 2.0)),
    )
    bridge = Node(
        package="robot_navigation",
        executable="fast_lio_odom_bridge",
        name="fast_lio_odom_bridge",
        output="screen",
        parameters=[
            {
                "source_topic": raw_odom_topic,
                "output_topic": config.get("output_topic", "/odometry/filtered"),
                "source_odom_frame": config.get("source_odom_frame", "camera_init"),
                "source_body_frame": config.get("source_body_frame", "body"),
                "output_odom_frame": config.get("odom_frame", "odom"),
                "output_base_frame": config.get("base_frame", "base_link"),
                "body_to_base_translation": config.get("body_to_base_translation", [0.0, 0.0, 0.0]),
                "body_to_base_rotation": config.get("body_to_base_rotation", [0.0, 0.0, 0.0, 1.0]),
                "publish_tf": bool(config.get("publish_tf", True)),
                "planar_output": bool(config.get("planar_output", False)),
                "max_future_skew_sec": float(config.get("max_future_skew_sec", 0.1)),
            }
        ],
    )
    logger.info("Added official FAST-LIO with isolated native TF and normalized odometry bridge")
    return [fast_lio, bridge]
