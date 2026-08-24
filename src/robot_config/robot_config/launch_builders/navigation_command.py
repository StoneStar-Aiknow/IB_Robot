"""Typed navigation command server launch builder."""

from typing import Any

from launch_ros.actions import Node


def generate_navigation_command_nodes(nav_config: dict[str, Any], use_sim: bool = False) -> list:
    command_config = nav_config.get("command_server", {})
    if not command_config.get("enabled", False) or use_sim:
        return []
    action_name = command_config.get("action_name")
    if not isinstance(action_name, str) or not action_name.strip():
        raise ValueError("navigation.command_server.action_name must be a non-empty string")

    parameters = {
        "action_name": action_name,
        "cancel_service_name": command_config.get("cancel_service_name", "/navigation/cancel_current"),
        "nav2_action_name": command_config.get("nav2_action_name", "/navigate_to_pose"),
        "stop_velocity_topic": command_config.get("stop_velocity_topic", "/cmd_vel_safe"),
        "global_frame": command_config.get("global_frame", "map"),
        "base_frame": command_config.get("base_frame", "base_link"),
        "nav2_server_timeout": command_config.get("nav2_server_timeout", 5.0),
        "nav2_result_timeout": command_config.get("nav2_result_timeout", 300.0),
        "tf_timeout": command_config.get("tf_timeout", 1.0),
        "cancel_timeout": command_config.get("cancel_timeout", 10.0),
        "cancel_response_timeout": command_config.get("cancel_response_timeout", 2.0),
        "linear_stop_threshold": command_config.get("linear_stop_threshold", 0.01),
        "angular_stop_threshold": command_config.get("angular_stop_threshold", 0.05),
        "stop_stable_duration": command_config.get("stop_stable_duration", 0.5),
        "stop_confirmation_timeout": command_config.get("stop_confirmation_timeout", 3.0),
    }
    return [
        Node(
            package="robot_navigation",
            executable="navigation_command_server",
            name="navigation_command_server",
            output="screen",
            parameters=[parameters],
        )
    ]
