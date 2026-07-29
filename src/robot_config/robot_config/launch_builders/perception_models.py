"""Launch arbitrary typed model-service plugins independently."""

import json
from typing import Any

from launch_ros.actions import Node

from robot_config.perception_runtime_config import parse_perception_runtime_config


def generate_perception_model_nodes(robot_config: dict[str, Any]) -> list[Node]:
    runtime = parse_perception_runtime_config(robot_config)
    return [
        Node(
            package="perception_service",
            executable="model_service_node",
            name=service.node_name,
            output="screen",
            parameters=[
                {
                    "instance_id": service.instance_id,
                    "bundle_path": str(service.bundle_path),
                    "deployment": service.deployment,
                    "adapter_class": service.adapter_class,
                    "service_type": service.service_type,
                    "service_endpoint": service.endpoint,
                    "required": service.required,
                    "runtime_options_json": json.dumps(dict(service.runtime_options), sort_keys=True),
                }
            ],
        )
        for service in runtime.enabled_services
    ]


__all__ = ["generate_perception_model_nodes"]
