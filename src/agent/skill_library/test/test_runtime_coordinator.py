"""Tests for the ROS-independent skill registry owner."""

from __future__ import annotations

from threading import RLock

import pytest

from skill_catalog.models import (
    SkillRegistryError,
    SkillRobotContext,
    SkillSnapshot,
)
from skill_library.runtime_coordinator import (
    SKILL_REQUEST_ID_CONFLICT,
    SkillRegistryOwner,
)


def _snapshot(name: str = "wave_hello") -> SkillSnapshot:
    robot = SkillRobotContext(
        robot_name="test_robot",
        context_schema_version=1,
        robot_config_digest="robot-digest",
        named_poses={},
        named_targets={},
        arm_joint_names=("1",),
        joint_limits={},
        workspace_limits={},
        required_control_mode="moveit_planning",
        timeout_policy={"default_skill_timeout_sec": 30.0, "task_budget_sec": 180.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.03,
        relative_motion_direction_mapping={},
        gripper_open_position=1.0,
        gripper_closed_position=0.0,
        execution_endpoints={
            "skill_action": "/skill",
            "primitive_action": "/primitive",
            "validate_skill_service": "/validate",
            "validate_primitive_service": "/validate-primitive",
            "gateway_status_service": "/status",
            "begin_workflow_service": "/begin",
            "finalize_workflow_service": "/finalize",
            "task_executor_action": "/task",
            "arm_trajectory_action": "/trajectory",
            "move_configuration_service": "/move",
        },
    )
    return SkillSnapshot(
        robot_name="test_robot",
        profile_name="test",
        primitive_contract_digest="primitive-digest",
        robot_context=robot,
        delegated_executors={},
        templates={name: {"primitive_sequence": []}},
        semantic_levels={name: "skill"},
        aliases={name: ()},
        parameter_schemas={name: {"type": "object", "properties": {}, "required": []}},
        requirements={name: frozenset()},
        provenance={"schema_version": 1, "source_release_digest": "source-digest"},
        enabled_skill_names=(name,),
        planner_visible_skill_names=(name,),
        capability_view={name: {"name": name}},
    )


def test_initial_activation_and_reload_increment_generation() -> None:
    snapshots = [_snapshot(), _snapshot("wave_hello_v2")]
    coordinator = SkillRegistryOwner(lambda: snapshots.pop(0), registry_epoch="epoch-1")

    initial = coordinator.reload("initial")
    assert initial.success is True
    assert initial.generation == 1
    assert coordinator.state == "READY"

    reloaded = coordinator.reload("reload-1")
    assert reloaded.success is True
    assert reloaded.old_generation == 1
    assert reloaded.generation == 2
    assert coordinator.current.snapshot.enabled_skill_names == ("wave_hello_v2",)


def test_reload_request_id_is_idempotent_and_conflicts_on_field_change() -> None:
    coordinator = SkillRegistryOwner(lambda: _snapshot(), registry_epoch="epoch-1")

    first = coordinator.reload("request-1", force=False)
    assert coordinator.reload("request-1", force=False) == first

    conflict = coordinator.reload("request-1", force=True)
    assert conflict.success is False
    assert conflict.error_code == SKILL_REQUEST_ID_CONFLICT


def test_reload_failure_is_fail_closed_before_initial_activation() -> None:
    def compile_snapshot():
        raise ValueError("invalid catalog")

    coordinator = SkillRegistryOwner(compile_snapshot)
    result = coordinator.reload("request-1")

    assert result.success is False
    assert coordinator.current is None
    assert coordinator.state == "FAILED"
    with pytest.raises(SkillRegistryError):
        coordinator.get_snapshot()


def test_retained_generation_survives_reload_until_execution_scope_releases_it() -> None:
    snapshots = [_snapshot("v1"), _snapshot("v2"), _snapshot("v3")]
    coordinator = SkillRegistryOwner(
        lambda: snapshots.pop(0),
        registry_epoch="epoch-1",
        max_unretained_history=1,
    )
    assert coordinator.reload("initial").success

    retained = coordinator.retain(1)
    assert coordinator.reload("reload").success
    assert coordinator.get_snapshot(registry_epoch="epoch-1", generation=1) is retained

    coordinator.release(1)
    assert coordinator.reload("reload-again").success
    with pytest.raises(SkillRegistryError, match="not retained"):
        coordinator.get_snapshot(registry_epoch="epoch-1", generation=1)


@pytest.mark.parametrize("history", [0, -1])
def test_history_retention_must_keep_at_least_one_generation(history: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SkillRegistryOwner(lambda: _snapshot(), max_unretained_history=history)


def test_coordinator_uses_injected_executor_state_lock() -> None:
    state_lock = RLock()
    coordinator = SkillRegistryOwner(
        lambda: _snapshot(),
        registry_epoch="epoch-1",
        state_lock=state_lock,
    )

    assert coordinator._lock is state_lock  # noqa: SLF001
    assert coordinator.reload("initial").success is True
