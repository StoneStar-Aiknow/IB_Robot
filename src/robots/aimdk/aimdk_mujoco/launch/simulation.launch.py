from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    arguments = ["--assets", LaunchConfiguration("assets").perform(context)]
    config_dir = LaunchConfiguration("config_dir").perform(context)
    if config_dir:
        arguments += ["--config-dir", config_dir]
    if LaunchConfiguration("headless").perform(context) == "true":
        arguments.append("--headless")
    return [Node(package="aimdk_mujoco", executable="simulation", arguments=arguments, output="screen")]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "assets", description="Directory containing prepared X2 simulation assets (required)"
            ),
            DeclareLaunchArgument("headless", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("config_dir", default_value="", description="Optional controller catalog directory"),
            OpaqueFunction(function=_launch_node),
        ]
    )
