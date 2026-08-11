"""Launch arbitrary typed model-service plugins independently."""

import json
from typing import Any

from launch_ros.actions import Node

from robot_config.perception_runtime_config import parse_perception_runtime_config


def generate_perception_model_nodes(
    robot_config: dict[str, Any],
    *,
    instance_ids: set[str] | None = None,
    configuration_generation: int = 0,
    require_semantic_identity: bool = False,
) -> list[Node]:
    runtime = parse_perception_runtime_config(robot_config)
    return [
        Node(
            package="inference_service",
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
                    "configuration_generation": configuration_generation,
                    "require_semantic_identity": require_semantic_identity,
                    "runtime_options_json": json.dumps(dict(service.runtime_options), sort_keys=True),
                }
            ],
        )
        for service in runtime.enabled_services
        if instance_ids is None or service.instance_id in instance_ids
    ]


__all__ = ["generate_perception_model_nodes"]
