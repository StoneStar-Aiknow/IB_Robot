"""Launch live semantic mapping directly from robot_config SSOT."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

from semantic_mapping.configuration import (
    load_semantic_mapping_robot_config,
    semantic_mapping_parameters,
    semantic_perception_nodes,
)


def launch_setup(context, *_args, **_kwargs):
    config = load_semantic_mapping_robot_config(
        context.launch_configurations.get("robot_config", "lekiwi_realsense_mapping"),
        context.launch_configurations.get("config_path", ""),
    )
    return semantic_perception_nodes(config) + [
        Node(
            package="semantic_mapping",
            executable="semantic_mapping_node",
            name="semantic_mapping",
            parameters=[semantic_mapping_parameters(config)],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="lekiwi_realsense_mapping"),
            DeclareLaunchArgument("config_path", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
