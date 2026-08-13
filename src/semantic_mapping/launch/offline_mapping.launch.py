"""Launch offline rosbag semantic mapping directly from robot_config SSOT."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
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
            "database_path": context.launch_configurations.get("database_path", ""),
            "artifact_output_dir": context.launch_configurations.get("artifact_output_dir", ""),
            "max_frames": int(context.launch_configurations.get("max_frames", "0")),
            "start_frame": int(context.launch_configurations.get("start_frame", "0")),
            "frame_sampling": context.launch_configurations.get("frame_sampling", "sequential"),
            "diagnostics_output_dir": context.launch_configurations.get("diagnostics_output_dir", ""),
        }
    )
    mapping_node = Node(
        package="semantic_mapping",
        executable="offline_mapping_node",
        name="offline_mapping",
        parameters=[parameters],
        output="screen",
    )
    shutdown_on_completion = RegisterEventHandler(
        OnProcessExit(
            target_action=mapping_node,
            on_exit=[EmitEvent(event=Shutdown(reason="offline semantic mapping completed"))],
        )
    )
    return semantic_perception_nodes(config, offline=True) + [mapping_node, shutdown_on_completion]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="lekiwi_mapping"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("bag_path", default_value=""),
            DeclareLaunchArgument("storage_id", default_value=""),
            DeclareLaunchArgument("database_path", default_value=""),
            DeclareLaunchArgument("artifact_output_dir", default_value=""),
            DeclareLaunchArgument("max_frames", default_value="0"),
            DeclareLaunchArgument("start_frame", default_value="0"),
            DeclareLaunchArgument("frame_sampling", default_value="sequential"),
            DeclareLaunchArgument("diagnostics_output_dir", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
