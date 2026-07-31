"""Launch offline rosbag semantic mapping directly from robot_config SSOT."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

from semantic_mapping.configuration import (
    load_semantic_mapping_robot_config,
    semantic_mapping_parameters,
    semantic_perception_nodes,
)


def launch_setup(context, *_args, **_kwargs):
    bag_path = context.launch_configurations.get("bag_path", "").strip()
    if not bag_path:
        raise ValueError("bag_path is required for offline semantic mapping")
    config = load_semantic_mapping_robot_config(
        context.launch_configurations.get("robot_config", "lekiwi_mapping"),
        context.launch_configurations.get("config_path", ""),
    )
    parameters = semantic_mapping_parameters(config, offline=True)
    parameters.update(
        {
            "bag_path": bag_path,
            "storage_id": context.launch_configurations.get("storage_id", ""),
        }
    )
    return semantic_perception_nodes(config, offline=True) + [
        Node(
            package="semantic_mapping",
            executable="offline_mapping_node",
            name="offline_mapping",
            parameters=[parameters],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="lekiwi_mapping"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("bag_path", default_value=""),
            DeclareLaunchArgument("storage_id", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
