"""Shared teleoperation command-group contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

LEGACY_TARGET_KEYS = frozenset(
    {
        "arm_joint_names",
        "gripper_joint_names",
        "arm_command_topic",
        "gripper_command_topic",
    }
)


@dataclass(frozen=True, slots=True)
class PublishGroup:
    """One ordered Float64MultiArray command output."""

    name: str
    joint_names: tuple[str, ...]
    topic: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "joint_names": list(self.joint_names), "topic": self.topic}


def parse_publish_groups(value: Any) -> list[PublishGroup]:
    """Parse and validate a JSON string or YAML-native publish-group list."""
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"publish_groups is not valid JSON: {exc}") from exc

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("publish_groups must be a list")

    groups: list[PublishGroup] = []
    seen_names: set[str] = set()
    seen_topics: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"publish_groups[{index}] must be an object")

        name = str(item.get("name", "")).strip()
        topic = str(item.get("topic", "")).strip()
        raw_joint_names = item.get("joint_names")
        if not name:
            raise ValueError(f"publish_groups[{index}].name must be a non-empty string")
        if not topic:
            raise ValueError(f"publish_groups[{index}].topic must be a non-empty string")
        if not isinstance(raw_joint_names, list) or not raw_joint_names:
            raise ValueError(f"publish_groups[{index}].joint_names must be a non-empty list")

        joint_names = tuple(str(joint).strip() for joint in raw_joint_names)
        if any(not joint for joint in joint_names):
            raise ValueError(f"publish_groups[{index}].joint_names must contain non-empty strings")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"publish_groups[{index}].joint_names must not contain duplicates")
        if name in seen_names:
            raise ValueError(f"publish_groups contains duplicate name {name!r}")
        if topic in seen_topics:
            raise ValueError(f"publish_groups contains duplicate topic {topic!r}")

        seen_names.add(name)
        seen_topics.add(topic)
        groups.append(PublishGroup(name=name, joint_names=joint_names, topic=topic))
    return groups


def legacy_publish_groups(
    arm_joint_names: list[str] | tuple[str, ...],
    gripper_joint_names: list[str] | tuple[str, ...],
    arm_command_topic: str,
    gripper_command_topic: str,
) -> list[PublishGroup]:
    """Translate the original arm/gripper parameters into publish groups."""
    groups = []
    if arm_joint_names:
        groups.append(
            PublishGroup(
                name="arm",
                joint_names=tuple(str(name) for name in arm_joint_names),
                topic=str(arm_command_topic),
            )
        )
    if gripper_joint_names:
        groups.append(
            PublishGroup(
                name="gripper",
                joint_names=tuple(str(name) for name in gripper_joint_names),
                topic=str(gripper_command_topic),
            )
        )
    return parse_publish_groups([group.to_dict() for group in groups])


def resolve_node_publish_groups(
    publish_groups: Any,
    *,
    arm_joint_names: list[str] | tuple[str, ...],
    gripper_joint_names: list[str] | tuple[str, ...],
    arm_command_topic: str,
    gripper_command_topic: str,
) -> list[PublishGroup]:
    """Resolve an explicit node parameter, falling back to legacy parameters."""
    explicit = parse_publish_groups(publish_groups)
    if explicit:
        return explicit
    return legacy_publish_groups(
        arm_joint_names,
        gripper_joint_names,
        arm_command_topic,
        gripper_command_topic,
    )


def resolve_target_publish_groups(target: Any, joints: Any, auxiliary_actuators: Any = None) -> list[PublishGroup]:
    """Resolve groups from a robot_config device target block.

    ``target.actuator`` is a reference into the robot-level
    ``auxiliary_actuators`` SSOT. It is intentionally mutually exclusive with
    both explicit groups and the legacy arm/gripper target keys.
    """
    target = target or {}
    joints = joints or {}
    if not isinstance(target, dict):
        raise ValueError("device target must be an object")
    if not isinstance(joints, dict):
        raise ValueError("robot joints must be an object")

    if "publish_groups" in target and "actuator" in target:
        raise ValueError("target.publish_groups cannot be combined with target.actuator")

    if "actuator" in target:
        extra_keys = sorted(set(target) - {"actuator", "group_name"})
        if extra_keys:
            raise ValueError("target.actuator cannot be combined with: " + ", ".join(extra_keys))
        mixed_keys = sorted(LEGACY_TARGET_KEYS.intersection(target))
        if mixed_keys:
            raise ValueError("target.actuator cannot be combined with legacy target keys: " + ", ".join(mixed_keys))
        raw_actuator = target["actuator"]
        if isinstance(raw_actuator, str):
            actuator_name = raw_actuator.strip()
        elif isinstance(raw_actuator, dict):
            actuator_name = str(raw_actuator.get("name", "")).strip()
        else:
            actuator_name = ""
        if not actuator_name:
            raise ValueError("target.actuator must name an auxiliary actuator")
        if isinstance(auxiliary_actuators, dict):
            actuator = auxiliary_actuators.get(actuator_name)
        elif isinstance(auxiliary_actuators, list):
            actuator = next(
                (
                    value
                    for value in auxiliary_actuators
                    if isinstance(value, dict) and value.get("name") == actuator_name
                ),
                None,
            )
        else:
            raise ValueError(f"target.actuator {actuator_name!r} requires robot auxiliary_actuators")
        if not isinstance(actuator, dict):
            raise ValueError(f"auxiliary actuator {actuator_name!r} was not found")
        joint_names = actuator.get("joint_names")
        topic = actuator.get("command_topic")
        if not isinstance(joint_names, list) or not joint_names or not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"auxiliary actuator {actuator_name!r} must define joint_names and command_topic")
        group_name = str(target.get("group_name", "hand")).strip()
        return parse_publish_groups([{"name": group_name, "joint_names": joint_names, "topic": topic}])

    if "publish_groups" in target:
        mixed_keys = sorted(LEGACY_TARGET_KEYS.intersection(target))
        if mixed_keys:
            raise ValueError(
                "target.publish_groups cannot be combined with legacy target keys: " + ", ".join(mixed_keys)
            )
        return parse_publish_groups(target["publish_groups"])

    return legacy_publish_groups(
        target.get("arm_joint_names", joints.get("arm", [])),
        target.get("gripper_joint_names", joints.get("gripper", [])),
        target.get("arm_command_topic", "/arm_position_controller/commands"),
        target.get("gripper_command_topic", "/gripper_position_controller/commands"),
    )
