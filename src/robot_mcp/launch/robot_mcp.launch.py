"""Launch the robot_mcp server as a long-lived node (streamable-http).

stdio transport does not need a launch file (opencode spawns it as a child
process). This file is for the production topology where the MCP server lives
on the robot host independently of the agent client.

Usage:
    ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm
    ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm port:=8080
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="so101_single_arm"),
            DeclareLaunchArgument("host", default_value="0.0.0.0"),
            DeclareLaunchArgument("port", default_value="8080"),
            Node(
                package="robot_mcp",
                executable="robot_mcp_server",
                name="robot_mcp_server",
                output="screen",
                parameters=[],
                arguments=[
                    "--config-name",
                    LaunchConfiguration("robot_config"),
                    "--transport",
                    "streamable-http",
                    "--host",
                    LaunchConfiguration("host"),
                    "--port",
                    LaunchConfiguration("port"),
                ],
            ),
        ]
    )
