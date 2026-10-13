"""Resolve the session's interfaces by ID from the bound runtime description.

Same rule as ``action_dispatch.policy_admission``: endpoints are never
configured, they come from ``runtime.interface_description.interfaces`` and
each one must have the expected ``(kind, direction, message_type)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ARM_JOINT_COUNT = 14
HAND_JOINT_COUNT = 20
POLICY_STREAM = "policy_stream"

_SHAPES = {
    "runtime.status": ("topic", "publish", "ibrobot_msgs/msg/RuntimeStatus"),
    "runtime.get_status": ("service", "serve", "ibrobot_msgs/srv/GetRuntimeStatus"),
    "runtime.set_mode": ("service", "serve", "ibrobot_msgs/srv/SetRuntimeMode"),
    "joint.state": ("topic", "publish", "sensor_msgs/msg/JointState"),
    "joint.arm_stream": ("topic", "subscribe", "std_msgs/msg/Float64MultiArray"),
    "joint.hand_stream": ("topic", "subscribe", "std_msgs/msg/Float64MultiArray"),
}
# What the one-shot release needs: it only ever calls set_mode.
RELEASE_INTERFACES = ("runtime.get_status", "runtime.set_mode")
_CHANNELS = {"joint.arm_stream": ARM_JOINT_COUNT, "joint.hand_stream": HAND_JOINT_COUNT}


class SessionBindingError(ValueError):
    """A named binding failure; codes match ``robot_config.InterfaceBindingError``."""

    def __init__(self, code: str, path: str, detail: str):
        self.code = code
        self.path = path
        super().__init__(f"{code}: {path}: {detail}")


@dataclass(frozen=True)
class Endpoint:
    endpoint: str
    qos: Mapping
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionEndpoints:
    identity: Mapping
    interfaces: Mapping[str, Endpoint]

    def __getitem__(self, interface_id: str) -> Endpoint:
        return self.interfaces[interface_id]


def runtime_identity(description: Mapping) -> dict:
    identity = description.get("robot") if isinstance(description, Mapping) else None
    if not isinstance(identity, Mapping) or any(
        not isinstance(identity.get(key), str) or not identity[key]
        for key in ("id", "type", "runtime_name", "runtime_version")
    ):
        raise SessionBindingError(
            "identity_required", "runtime.interface_description.robot", "complete runtime identity is required"
        )
    return dict(identity)


def resolve_endpoints(description: Mapping, ids=tuple(_SHAPES), policy_mode: str = POLICY_STREAM) -> SessionEndpoints:
    """Return the named interfaces, or raise ``SessionBindingError`` naming the first bad one."""
    if not isinstance(description, Mapping) or not isinstance(description.get("interfaces"), Mapping):
        raise SessionBindingError(
            "description_required", "runtime.interface_description", "bind the live description first"
        )
    identity = runtime_identity(description)
    interfaces = description["interfaces"]
    resolved = {}
    for interface_id in ids:
        item = interfaces.get(interface_id)
        if not isinstance(item, Mapping):
            raise SessionBindingError("unknown_interface", interface_id, "ID is not in the runtime description")
        kind, direction, message_type = _SHAPES[interface_id]
        if (item.get("kind"), item.get("direction"), item.get("message_type")) != _SHAPES[interface_id]:
            raise SessionBindingError("invalid_interface", interface_id, f"requires {kind} {message_type} {direction}")
        endpoint = item.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith("/"):
            raise SessionBindingError("invalid_interface", interface_id, "endpoint must be an absolute ROS name")
        joint_names = ()
        if interface_id in _CHANNELS:
            if policy_mode not in (item.get("modes") or []):
                raise SessionBindingError("invalid_interface", interface_id, f"must accept runtime mode {policy_mode}")
            joint_names = tuple(str(name) for name in item.get("joint_names") or [])
            count = _CHANNELS[interface_id]
            if len(joint_names) != count or len(set(joint_names)) != count:
                raise SessionBindingError("invalid_interface", interface_id, f"must describe {count} distinct joint_names")
        elif interface_id == "joint.state":
            joint_names = tuple(str(name) for name in item.get("joint_names") or [])
        resolved[interface_id] = Endpoint(endpoint, dict(item.get("qos") or {}), joint_names)
    if "joint.state" in resolved and "joint.arm_stream" in resolved:
        missing = set(resolved["joint.arm_stream"].joint_names) - set(resolved["joint.state"].joint_names)
        if missing:
            raise SessionBindingError("invalid_interface", "joint.state", f"does not report {sorted(missing)}")
    return SessionEndpoints(identity, resolved)
