"""Launch wiring for the parameterized interaction demo."""

import math
from collections.abc import Mapping

from launch_ros.actions import Node

_SPEECH_TYPE = "ibrobot_msgs/srv/SpeakText"
_MOTION_TYPE = "ibrobot_msgs/action/ExecuteNamedMotion"


def _bound_interface(config: Mapping, role: str, kind: str, type_name: str) -> str:
    interfaces = config.get("interfaces")
    if not isinstance(interfaces, Mapping):
        raise ValueError("interaction_demo.interfaces must be a mapping")
    binding = interfaces.get(role)
    if not isinstance(binding, Mapping):
        raise ValueError(f"interaction_demo.interfaces.{role} must be a mapping")
    if binding.get("kind") != kind or binding.get("type") != type_name:
        raise ValueError(f"interaction_demo.interfaces.{role} must bind {kind} {type_name}")
    endpoint = binding.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError(f"interaction_demo.interfaces.{role} was not resolved to an endpoint")
    return endpoint


def generate_interaction_demo_nodes(robot_config: dict) -> list[Node]:
    config = robot_config.get("interaction_demo", {})
    if not isinstance(config, dict):
        raise ValueError("interaction_demo must be a mapping")
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("interaction_demo.enabled must be a boolean")
    if not enabled:
        return []
    timeout = config.get("timeout_sec", 40.0)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("interaction_demo.timeout_sec must be positive and finite")
    service = config.get("service_name", "/interaction_demo/execute")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("interaction_demo.service_name must be non-empty")
    speech_endpoint = _bound_interface(config, "speech", "service", _SPEECH_TYPE)
    motion_endpoint = _bound_interface(config, "named_motion", "action", _MOTION_TYPE)
    return [
        Node(
            package="robot_interaction_demo",
            executable="interaction_demo_node",
            name="interaction_demo",
            output="screen",
            parameters=[
                {
                    "speech_service": speech_endpoint,
                    "named_motion_action": motion_endpoint,
                    "timeout_sec": float(timeout),
                    "service_name": service,
                }
            ],
        )
    ]
