"""Launch live semantic mapping or static-map query-only mode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

from semantic_mapping.configuration import (
    load_semantic_mapping_robot_config,
    semantic_mapping_parameters,
    semantic_perception_nodes,
)


def _query_only_parameters(config, overrides):
    """Build a minimal parameter set for loading a static semantic map without perception services."""
    params = semantic_mapping_parameters(config)
    overrides_dict = {k: v for k, v in overrides.items() if v}
    params.update(overrides_dict)
    return params


def launch_setup(context, *_args, **_kwargs):
    mode = context.launch_configurations.get("mode", "online")
    config = load_semantic_mapping_robot_config(
        context.launch_configurations.get("robot_config", "lekiwi_realsense_mapping"),
        context.launch_configurations.get("config_path", ""),
    )
    if mode == "query_only":
        overrides = {
            "database_path": context.launch_configurations.get("database_path", ""),
            "artifact_output_dir": context.launch_configurations.get("artifact_output_dir", ""),
        }
        return [
            Node(
                package="semantic_mapping",
                executable="semantic_mapping_node",
                name="semantic_mapping",
                parameters=[_query_only_parameters(config, overrides)],
                output="screen",
            )
        ]
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
            DeclareLaunchArgument("mode", default_value="online"),
            DeclareLaunchArgument("robot_config", default_value="lekiwi_realsense_mapping"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("database_path", default_value=""),
            DeclareLaunchArgument("artifact_output_dir", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
