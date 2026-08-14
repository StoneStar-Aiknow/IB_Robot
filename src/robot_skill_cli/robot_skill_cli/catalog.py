"""ROS-independent loading and querying of robot skill capabilities."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_common.dispatch_binding import delegated_executor_identity, load_delegated_model_identity
from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST, PRIMITIVE_DESCRIPTORS
from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import load_robot_config_dict, robot_config_digest
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.digest import (
    derive_capability_digest,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext, SkillRobotContext
from skill_catalog.source import AmentShareSkillSource, DevelopmentStagingSkillSource, DirectoryReleaseSkillSource

_LIST_CAPABILITY_FIELDS = (
    "summary",
    "domain",
    "moves_robot",
    "required_control_mode",
)


class UnknownSkillError(ValueError):
    """Raised when a catalog query names a skill that is not enabled."""

    code = "UNKNOWN_SKILL"

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"unknown skill: {skill_name}")


@dataclass(frozen=True)
class CatalogContext:
    view: dict[str, Any]


@dataclass(frozen=True)
class GatewayTransport:
    status_service: str
    validate_skill_service: str
    skill_action_name: str
    snapshot_service: str = "/embodied/get_skill_snapshot"
    reload_service: str = "/embodied/reload_skill_catalog"
    plan_service: str = "/embodied/plan_agent_command"
    validate_plan_service: str = "/embodied/validate_agent_plan"
    confirm_plan_service: str = "/embodied/confirm_agent_plan"
    execute_plan_action: str = "/embodied/execute_agent_plan"


def load_catalog_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load one normalized config for public catalog use (ROS-independent)."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    return CatalogContext(view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path)))


def load_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load both catalog view and ROS transport for runtime Gateway commands."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    return (
        CatalogContext(view=_snapshot_capability_view(compile_local_snapshot(robot_config, resolved_path))),
        GatewayTransport(
            status_service=embodied.get("skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
            snapshot_service=embodied.get("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot"),
            reload_service=embodied.get("skill_catalog_reload_service", "/embodied/reload_skill_catalog"),
            validate_skill_service=embodied.get("validate_skill_service", "/embodied/validate_skill"),
            skill_action_name=embodied.get("skill_action_name", "/embodied/execute_skill"),
            plan_service=embodied.get("plan_service", "/embodied/plan_agent_command"),
            validate_plan_service=embodied.get("validate_plan_service", "/embodied/validate_agent_plan"),
            confirm_plan_service=embodied.get("confirm_plan_service", "/embodied/confirm_agent_plan"),
            execute_plan_action=embodied.get("execute_plan_action", "/embodied/execute_agent_plan"),
        ),
    )


def compile_local_snapshot(robot_config: dict[str, Any], config_path: Path):
    """Compile the profile selected by one normalized robot config."""
    embodied = robot_config["embodied"]
    execution = embodied.get("execution", {})
    source_mode = embodied.get("skill_catalog_source_mode", "installed")
    source_root = Path(embodied.get("skill_catalog_source_root", ""))
    if source_mode == "development":
        if not source_root.is_absolute():
            repository_root = next((parent for parent in config_path.parents if parent.name == "src"), None)
            if repository_root is None:
                source_mode = "installed"
            else:
                source_root = repository_root.parent / source_root
        if source_mode == "installed":
            source = AmentShareSkillSource()
        else:
            source = DevelopmentStagingSkillSource(source_root)
    elif source_mode == "production":
        source = DirectoryReleaseSkillSource(source_root)
    else:
        source = AmentShareSkillSource()
    # Keep the ROS-independent CLI catalog context aligned with the runtime
    # SkillExecutorNode.  Delegated executors are selected by the enabled
    # execution configuration, not by one particular profile name; otherwise
    # the PC grasp profile cannot validate its own ``grasp_pipeline`` entry.
    delegated = {}
    grasp_execution = robot_config.get("grasp_execution", {})
    if grasp_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="grasp_pipeline",
                endpoint_name=grasp_execution.get("action_name", "/manipulation/execute_pick"),
                configuration=grasp_execution,
                **load_delegated_model_identity(grasp_execution),
            )
        )
        delegated[descriptor.name] = descriptor

    placement_execution = robot_config.get("placement_execution", {})
    if placement_execution.get("enabled", False):
        descriptor = DelegatedExecutorDescriptor(
            **delegated_executor_identity(
                name="placement_pipeline",
                endpoint_name=placement_execution.get("action_name", "/manipulation/execute_place"),
                configuration=placement_execution,
            )
        )
        delegated[descriptor.name] = descriptor
    robot_context = SkillRobotContext(
        robot_name=robot_config["name"],
        context_schema_version=1,
        robot_config_digest=robot_config_digest(robot_config),
        named_poses=embodied.get("named_poses", {}),
        named_targets=embodied.get("named_targets", {}),
        arm_joint_names=tuple(robot_config.get("joints", {}).get("arm", [])),
        joint_limits=robot_config.get("teleoperation", {}).get("safety", {}).get("joint_limits", {}),
        workspace_limits=embodied.get("safety", {}).get("workspace", {}),
        required_control_mode=robot_config["skill_required_control_mode"],
        timeout_policy=resolve_embodied_timeout_policy(embodied),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_direction_mapping=execution.get("relative_motion_direction_mapping", {}),
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        execution_endpoints={
            "skill_action": embodied.get("skill_action_name", "/embodied/execute_skill"),
            "primitive_action": embodied.get("primitive_action_name", "/embodied/execute_primitive"),
            "validate_skill_service": embodied.get("validate_skill_service", "/embodied/validate_skill"),
            "validate_primitive_service": embodied.get("validate_primitive_service", "/embodied/validate_primitive"),
            "gateway_status_service": embodied.get(
                "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
            ),
            "begin_workflow_service": embodied.get("begin_workflow_service", "/embodied/begin_workflow_execution"),
            "finalize_workflow_service": embodied.get(
                "finalize_workflow_service", "/embodied/finalize_workflow_execution"
            ),
            "task_executor_action": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
            "arm_trajectory_action": execution.get(
                "arm_trajectory_action_name", "/arm_trajectory_controller/follow_joint_trajectory"
            ),
            "move_configuration_service": execution.get(
                "move_configuration_service", "/moveit_gateway/move_to_configuration"
            ),
        },
    )
    return compile_skill_catalog(
        source,
        profile_name=embodied["skill_catalog_profile"],
        context=SkillCompileContext(
            robot=robot_context,
            primitive_contracts=PRIMITIVE_DESCRIPTORS,
            primitive_contract_digest=PRIMITIVE_CONTRACT_DIGEST,
            delegated_executors=delegated,
        ),
    )


def _snapshot_capability_view(snapshot):
    def thaw(value):
        if isinstance(value, Mapping):
            return {str(key): thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return {
        "robot_name": snapshot.robot_name,
        "skills": [thaw(snapshot.capability_view[name]) for name in sorted(snapshot.capability_view)],
        "pose_names": sorted(snapshot.robot_context.named_poses),
        "timeout_policy": thaw(snapshot.robot_context.timeout_policy),
        "capability_digest": snapshot.capability_digest,
        "profile_name": snapshot.profile_name,
    }


def load_capability_catalog(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the normalized config and return its public capability view."""
    return load_catalog_context(config_name=config_name, config_path=config_path).view


def capability_view_from_snapshot(snapshot: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Verify one exact Gateway snapshot and return its public catalog view."""
    expected_identity = (
        status["registry_epoch"],
        int(status["registry_generation"]),
        status["registry_digest"],
    )
    actual_identity = (
        snapshot["registry_epoch"],
        int(snapshot["generation"]),
        snapshot["registry_digest"],
    )
    if not snapshot["success"] or actual_identity != expected_identity:
        raise ValueError("SKILL_REGISTRY_VERSION_MISMATCH: snapshot identity does not match Gateway status")
    try:
        payload = json.loads(snapshot["snapshot_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot schema is invalid")
    if to_canonical_json(payload) != snapshot["snapshot_json"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not canonical")
    registry_preimage = payload.get("registry_preimage")
    capability_preimage = payload.get("capability_preimage")
    provenance_preimage = payload.get("provenance_preimage")
    if not all(isinstance(value, dict) for value in (registry_preimage, capability_preimage, provenance_preimage)):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot preimages are invalid")
    if derive_registry_digest(registry_preimage) != snapshot["registry_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: registry digest does not match")
    if derive_capability_digest(capability_preimage) != snapshot["capability_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability digest does not match")
    if derive_provenance_digest(provenance_preimage) != snapshot["provenance_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: provenance digest does not match")
    capability_mapping = capability_preimage.get("capability_view")
    if not isinstance(capability_mapping, dict):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability view is invalid")
    return {
        "robot_name": capability_preimage["robot_name"],
        "skills": [copy.deepcopy(capability_mapping[name]) for name in sorted(capability_mapping)],
        "pose_names": list(capability_preimage["named_pose_names"]),
        "timeout_policy": copy.deepcopy(capability_preimage["timeout_policy"]),
        "capability_digest": snapshot["capability_digest"],
        "profile_name": capability_preimage["profile_name"],
    }


def _skill_entry(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    normalized_name = skill_name.strip()
    for skill in view["skills"]:
        if skill["name"] == normalized_name:
            return skill
    raise UnknownSkillError(normalized_name)


def list_skills(view: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for skill in view["skills"]:
        entry = {
            "name": skill["name"],
            **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        }
        skills.append(entry)
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "skills": skills,
    }


def describe_skill(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    skill = _skill_entry(view, skill_name)
    result = {
        "robot_name": view["robot_name"],
        "name": skill["name"],
        **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        "parameters": copy.deepcopy(skill["parameters"]),
        "recovery_policy": copy.deepcopy(skill["recovery_policy"]),
        "timeout_policy": copy.deepcopy(view["timeout_policy"]),
        "config_digest": view["capability_digest"],
    }
    return result


def list_poses(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "poses": list(view["pose_names"]),
    }
