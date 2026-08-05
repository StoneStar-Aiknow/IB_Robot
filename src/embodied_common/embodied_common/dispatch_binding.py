"""Small constructors for the typed ROS dispatch envelope."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ibrobot_msgs.msg import DelegatedExecutorIdentity, DispatchBinding, WorkflowStep


def new_binding(*, task_id: str = "", root_task_id: str = "") -> DispatchBinding:
    binding = DispatchBinding()
    binding.schema_version = 1
    binding.task_id = task_id
    binding.root_task_id = root_task_id or task_id
    binding.workflow_step_index = 0
    return binding


def copy_binding(source: DispatchBinding) -> DispatchBinding:
    target = DispatchBinding()
    target.schema_version = source.schema_version
    target.task_id = source.task_id
    target.root_task_id = source.root_task_id
    target.task_budget = source.task_budget
    target.expected_registry_epoch = source.expected_registry_epoch
    target.expected_registry_generation = source.expected_registry_generation
    target.expected_registry_digest = source.expected_registry_digest
    target.workflow_digest = source.workflow_digest
    target.workflow_step_index = source.workflow_step_index
    target.root_lease_nonce = source.root_lease_nonce
    target.dispatch_nonce = source.dispatch_nonce
    return target


def delegated_executor_identity(
    *,
    name: str,
    endpoint_name: str,
    contract_version: str = "1",
    endpoint_kind: str = "ros_action",
    configuration: Any = None,
) -> dict[str, str]:
    configuration_digest = hashlib.sha256(
        json.dumps(
            {
                "endpoint_kind": endpoint_kind,
                "endpoint_name": endpoint_name,
                "configuration": configuration if configuration is not None else {},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "name": name,
        "contract_version": contract_version,
        "endpoint_kind": endpoint_kind,
        "endpoint_name": endpoint_name,
        "configuration_digest": configuration_digest,
        "model_deployment_name": "",
        "model_fingerprint": "",
        "model_bundle_digest": "",
    }


def fill_delegated_executor_identity(message: DelegatedExecutorIdentity, identity: dict[str, str]) -> None:
    message.schema_version = 1
    for field_name, value in identity.items():
        setattr(message, field_name, value)


def delegated_executor_identity_matches(message: DelegatedExecutorIdentity, identity: dict[str, str]) -> bool:
    return message.schema_version == 1 and all(
        getattr(message, field_name) == value for field_name, value in identity.items()
    )


def workflow_step(
    *,
    skill_name: str,
    target_name: str = "",
    place_name: str = "",
    motion_direction: str = "",
    motion_distance: float = 0.0,
    timeout_sec: float = 0.0,
) -> WorkflowStep:
    step = WorkflowStep()
    step.schema_version = 1
    step.skill_name = skill_name
    step.target_name = target_name
    step.place_name = place_name
    step.motion_direction = motion_direction
    step.motion_distance = float(motion_distance)
    step.timeout_sec = float(timeout_sec)
    return step


def binding_task_id(value: Any) -> str:
    return str(value.dispatch_binding.task_id).strip()
