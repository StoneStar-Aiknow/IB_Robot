"""Standalone launcher for the parallel benchmark shared cloud owner.

Companion to a benchmark launch that runs with
``benchmark.evaluation.parallel.cloud_deployment: external``: that launch
skips the co-located shared cloud owner and exports the per-lane cloud
configs (plus admission limits) to ``cloud_configs.json`` next to the
materialized lane configurations. Point ``cloud_configs_file`` at that
export to own the single shared runtime from this process, typically on
the cloud machine of a two-host deployment.

    ros2 launch inference_service benchmark_shared_cloud.launch.py \
        cloud_configs_file:=/path/to/benchmark_run_configs/<run_id>/cloud_configs.json

The export file must be made available on the cloud host (shared filesystem
or scp); both hosts must also share one ROS domain so the distributed
requests, heartbeats, and video control traffic reach this owner.
"""

import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_export(path: str) -> dict:
    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"cloud_configs_file does not exist: {export_path}")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(export, dict) or not isinstance(export.get("cloud_configs"), list):
        raise ValueError(f"cloud_configs_file must contain a 'cloud_configs' list: {export_path}")
    return export


def _launch_setup(context: LaunchContext):
    export = _load_export(LaunchConfiguration("cloud_configs_file").perform(context))
    max_in_flight = export.get("max_in_flight", 1)
    queue_size = export.get("queue_size", 64)
    if type(max_in_flight) is not int or max_in_flight < 1:
        raise ValueError("exported max_in_flight must be a positive int")
    if type(queue_size) is not int or queue_size < 1:
        raise ValueError("exported queue_size must be a positive int")
    shared = Node(
        package="inference_service",
        executable="benchmark_shared_cloud_node",
        name="benchmark_shared_cloud",
        output="screen",
        parameters=[
            {
                "cloud_configs_json": json.dumps(export["cloud_configs"]),
                "max_in_flight": max_in_flight,
                "queue_size": queue_size,
            }
        ],
    )
    return [shared]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cloud_configs_file",
                description="Path to the cloud_configs.json exported by the external-cloud benchmark launch",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
