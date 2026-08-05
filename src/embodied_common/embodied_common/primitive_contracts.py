"""Canonical static contracts for internal skill primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class PrimitiveDescriptor:
    schema_version: int
    name: str
    parameter_contract: MappingProxyType
    required_runtime_capabilities: tuple[str, ...]
    dispatch_kind: str


def _property(type_name: str, **constraints: Any) -> dict[str, Any]:
    return {"type": type_name, **constraints}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_value(item) for item in value]
    return value


def _object(
    name: str,
    properties: dict[str, dict[str, Any]] | None = None,
    required: tuple[str, ...] = (),
    *,
    one_of: tuple[dict[str, Any], ...] = (),
    all_of: tuple[dict[str, Any], ...] = (),
) -> MappingProxyType:
    schema_properties = {
        "primitive_name": {"type": "string", "const": name},
        **(properties or {}),
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": schema_properties,
        "required": ["primitive_name", *required],
        "additionalProperties": False,
    }
    if one_of:
        schema["oneOf"] = one_of
    if all_of:
        schema["allOf"] = all_of
    return _deep_freeze(schema)


def _descriptor(
    name: str,
    *,
    dispatch_kind: str,
    capabilities: tuple[str, ...],
    properties: dict[str, dict[str, Any]] | None = None,
    required: tuple[str, ...] = (),
    one_of: tuple[dict[str, Any], ...] = (),
    all_of: tuple[dict[str, Any], ...] = (),
) -> PrimitiveDescriptor:
    return PrimitiveDescriptor(
        schema_version=1,
        name=name,
        parameter_contract=_object(name, properties, required, one_of=one_of, all_of=all_of),
        required_runtime_capabilities=tuple(sorted(set(capabilities))),
        dispatch_kind=dispatch_kind,
    )


_TASK_EXECUTOR_CAPABILITIES = ("task_executor", "validate_skill")
_ARM_TRAJECTORY_CAPABILITIES = ("arm_trajectory", "validate_skill")

_JOINT_MAPPING = {"type": "object", "additionalProperties": {"type": "number"}}
_POSITIVE_DURATION = _property("number", exclusiveMinimum=0.0)


def _closed_branch(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"primitive_name": _property("string"), **properties},
        "required": ["primitive_name", *required],
        "additionalProperties": False,
    }


_JOINT_SELECTION = (
    _closed_branch({"joint_positions": _JOINT_MAPPING, "duration_sec": _POSITIVE_DURATION}, ("joint_positions",)),
    _closed_branch(
        {"joint_position_offsets": _JOINT_MAPPING, "duration_sec": _POSITIVE_DURATION},
        ("joint_position_offsets",),
    ),
)

_TRAJECTORY_TERM = {
    "type": "object",
    "properties": {
        "amplitude": _property("number"),
        "harmonic": _property("number"),
        "phase": _property("number"),
    },
    "required": ["amplitude"],
    "additionalProperties": False,
}
_TRAJECTORY_JOINT = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": _TRAJECTORY_TERM}},
    "required": ["terms"],
    "additionalProperties": False,
}
_WORKSPACE_AXIS = {
    "type": "array",
    "minItems": 2,
    "items": {"type": "number"},
}
_WORKSPACE_POINT = {
    "type": "object",
    "properties": {axis: _WORKSPACE_AXIS for axis in ("x", "y", "z")},
    "additionalProperties": False,
}
_WORKSPACE_LIMITS = {
    "type": "object",
    "properties": {
        "model": _property("string"),
        "points": {"type": "object", "additionalProperties": _WORKSPACE_POINT},
    },
    "required": ["model", "points"],
    "additionalProperties": False,
}
_TRAJECTORY_TEMPLATE = {
    "oneOf": (
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "wave_dance_v1"},
                "active_waypoint_count": _property("number", exclusiveMinimum=0.0),
                "repeat_count": _property("number", exclusiveMinimum=0.0),
                "zero_hold_count": _property("number", minimum=0.0),
                "base_pose": _JOINT_MAPPING,
                "joints": {"type": "object", "additionalProperties": _TRAJECTORY_JOINT},
                "workspace_limits": _WORKSPACE_LIMITS,
                "waypoint_duration_sec": _POSITIVE_DURATION,
            },
            "required": ["type", "active_waypoint_count", "base_pose", "joints"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "single_joint_wave_v1"},
                "active_waypoint_count": _property("number", exclusiveMinimum=0.0),
                "repeat_count": _property("number", exclusiveMinimum=0.0),
                "base_pose": _JOINT_MAPPING,
                "joint": _property("string"),
                "amplitude": _property("number"),
                "phase": _property("number"),
                "workspace_limits": _WORKSPACE_LIMITS,
                "waypoint_duration_sec": _POSITIVE_DURATION,
            },
            "required": ["type", "base_pose", "joint", "amplitude"],
            "additionalProperties": False,
        },
    )
}
_WAYPOINT = {
    "type": "object",
    "properties": {
        "primitive_name": {"type": "string", "const": "move_to_joint_positions"},
        "joint_positions": _JOINT_MAPPING,
        "duration_sec": _property("number", minimum=0.0),
    },
    "required": ["primitive_name", "joint_positions", "duration_sec"],
    "additionalProperties": False,
}

PRIMITIVE_DESCRIPTORS = MappingProxyType(
    {
        descriptor.name: descriptor
        for descriptor in (
            _descriptor(
                "move_to_named_pose",
                dispatch_kind="task_executor_action",
                capabilities=_TASK_EXECUTOR_CAPABILITIES,
                properties={
                    "pose_name": _property("string"),
                    "target_pose_key": _property("string"),
                    "place_name_from_request": _property("boolean"),
                },
                one_of=(
                    _closed_branch({"pose_name": _property("string")}, ("pose_name",)),
                    _closed_branch({"target_pose_key": _property("string")}, ("target_pose_key",)),
                    _closed_branch(
                        {"place_name_from_request": {"type": "boolean", "const": True}},
                        ("place_name_from_request",),
                    ),
                ),
            ),
            _descriptor(
                "move_to_pose",
                dispatch_kind="task_executor_action",
                capabilities=("fresh_ee_pose", *_TASK_EXECUTOR_CAPABILITIES),
            ),
            _descriptor(
                "move_to_configuration",
                dispatch_kind="move_configuration_service",
                capabilities=("move_configuration", "validate_skill"),
                properties={
                    "joint_positions": _JOINT_MAPPING,
                    "joint_position_offsets": _JOINT_MAPPING,
                    "duration_sec": _POSITIVE_DURATION,
                },
                one_of=_JOINT_SELECTION,
            ),
            _descriptor(
                "move_relative_ee",
                dispatch_kind="task_executor_action",
                capabilities=("fresh_ee_pose", *_TASK_EXECUTOR_CAPABILITIES),
                properties={
                    "motion_direction": _property("string"),
                    "motion_direction_from_request": _property("boolean"),
                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                    "motion_distance_from_request": _property("boolean"),
                },
                all_of=(
                    {
                        "oneOf": (
                            _closed_branch(
                                {
                                    "motion_direction": {
                                        "type": "string",
                                        "enum": ("forward", "backward", "left", "right", "up", "down"),
                                    },
                                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                                    "motion_distance_from_request": _property("boolean"),
                                },
                                ("motion_direction",),
                            ),
                            _closed_branch(
                                {
                                    "motion_direction_from_request": {"type": "boolean", "const": True},
                                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                                    "motion_distance_from_request": _property("boolean"),
                                },
                                ("motion_direction_from_request",),
                            ),
                        )
                    },
                    {
                        "oneOf": (
                            _closed_branch(
                                {
                                    "motion_direction": _property("string"),
                                    "motion_direction_from_request": _property("boolean"),
                                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                                },
                                ("motion_distance",),
                            ),
                            _closed_branch(
                                {
                                    "motion_direction": _property("string"),
                                    "motion_direction_from_request": _property("boolean"),
                                    "motion_distance_from_request": {"type": "boolean", "const": True},
                                },
                                ("motion_distance_from_request",),
                            ),
                        )
                    },
                ),
            ),
            _descriptor(
                "move_to_joint_positions",
                dispatch_kind="arm_trajectory_action",
                capabilities=_ARM_TRAJECTORY_CAPABILITIES,
                properties={
                    "joint_positions": _JOINT_MAPPING,
                    "joint_position_offsets": _JOINT_MAPPING,
                    "duration_sec": _POSITIVE_DURATION,
                },
                one_of=_JOINT_SELECTION,
            ),
            _descriptor(
                "move_through_joint_positions",
                dispatch_kind="arm_trajectory_action",
                capabilities=_ARM_TRAJECTORY_CAPABILITIES,
                properties={
                    "trajectory_template": _TRAJECTORY_TEMPLATE,
                    "joint_waypoints": {"type": "array", "minItems": 1, "items": _WAYPOINT},
                    "waypoint_duration_sec": _property("number", exclusiveMinimum=0.0),
                },
                one_of=(
                    _closed_branch(
                        {"trajectory_template": _TRAJECTORY_TEMPLATE},
                        ("trajectory_template",),
                    ),
                    _closed_branch(
                        {
                            "joint_waypoints": {"type": "array", "minItems": 1, "items": _WAYPOINT},
                            "waypoint_duration_sec": _POSITIVE_DURATION,
                        },
                        ("joint_waypoints", "waypoint_duration_sec"),
                    ),
                ),
            ),
            _descriptor(
                "open_gripper",
                dispatch_kind="task_executor_action",
                capabilities=_TASK_EXECUTOR_CAPABILITIES,
            ),
            _descriptor(
                "close_gripper",
                dispatch_kind="task_executor_action",
                capabilities=_TASK_EXECUTOR_CAPABILITIES,
            ),
            _descriptor(
                "rotate_gripper_cw",
                dispatch_kind="task_executor_action",
                capabilities=_TASK_EXECUTOR_CAPABILITIES,
                properties={
                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                    "motion_distance_from_request": _property("boolean"),
                },
            ),
            _descriptor(
                "rotate_gripper_ccw",
                dispatch_kind="task_executor_action",
                capabilities=_TASK_EXECUTOR_CAPABILITIES,
                properties={
                    "motion_distance": _property("number", exclusiveMinimum=0.0),
                    "motion_distance_from_request": _property("boolean"),
                },
            ),
        )
    }
)

SUPPORTED_PRIMITIVES = frozenset(PRIMITIVE_DESCRIPTORS)


def primitive_contract_preimage() -> dict[str, Any]:
    primitives = []
    for name in sorted(PRIMITIVE_DESCRIPTORS):
        descriptor = PRIMITIVE_DESCRIPTORS[name]
        primitives.append(
            {
                "schema_version": descriptor.schema_version,
                "name": descriptor.name,
                "parameter_contract": _to_json_value(descriptor.parameter_contract),
                "required_runtime_capabilities": list(descriptor.required_runtime_capabilities),
                "dispatch_kind": descriptor.dispatch_kind,
            }
        )
    return {"schema_version": 1, "primitives": primitives}


def canonical_json(value: Any) -> str:
    return json.dumps(
        _to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


PRIMITIVE_CONTRACT_DIGEST = hashlib.sha256(canonical_json(primitive_contract_preimage()).encode()).hexdigest()


def get_primitive_descriptor(name: str) -> PrimitiveDescriptor:
    try:
        return PRIMITIVE_DESCRIPTORS[name]
    except KeyError as exc:
        raise KeyError(f"unsupported primitive: {name}") from exc
