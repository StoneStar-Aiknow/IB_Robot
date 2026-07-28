"""Launch the robot_mcp server as a long-lived node (streamable-http).

stdio transport does not need a launch file (opencode spawns it as a child
process). This file is for the production topology where the MCP server lives
on the robot host independently of the agent client.

Usage:
    ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm
    ros2 launch robot_mcp robot_mcp.launch.py \
      robot_config:=so101_handeye_realsense_grasp port:=8080
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackagePrefix


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="so101_single_arm"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("port", default_value="8080"),
            ExecuteProcess(
                cmd=[
                    PathJoinSubstitution([FindPackagePrefix("robot_mcp"), "lib", "robot_mcp", "robot_mcp_server"]),
                    "--config-name",
                    LaunchConfiguration("robot_config"),
                    "--config-path",
                    LaunchConfiguration("config_path"),
                    "--transport",
                    "streamable-http",
                    "--host",
                    LaunchConfiguration("host"),
                    "--port",
                    LaunchConfiguration("port"),
                ],
                output="screen",
            ),
        ]
    )
