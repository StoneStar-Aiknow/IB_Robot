from __future__ import annotations

import pytest

from embodied_common.primitive_contracts import (
    PRIMITIVE_CONTRACT_V1,
    PRIMITIVE_CONTRACT_V2,
    PRIMITIVE_CONTRACT_V3,
)
from ibrobot_msgs.srv import ValidateSkill
from safety_guard.safety_guard_node import SafetyGuardNode
from safety_guard.snapshot_cache import SafetySnapshotCache, SnapshotCacheError, SnapshotIdentity
from skill_catalog.models import SkillRobotContext, SkillSnapshot


def _snapshot(name: str = "open_gripper_skill", *, context_schema_version: int = 1) -> SkillSnapshot:
    execution_endpoints = {
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
    }
    if context_schema_version >= 2:
        execution_endpoints["navigation_action"] = "/navigation"
    robot = SkillRobotContext(
        robot_name="test_robot",
        context_schema_version=context_schema_version,
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
        execution_endpoints=execution_endpoints,
        supported_control_modes=("moveit_planning", "base_navigation") if context_schema_version == 3 else (),
    )
    template = {
        "capability": {
            "name": name,
            "summary": "Open the gripper.",
            "domain": "manipulation",
            "semantic_level": "skill",
            "planner_visible": True,
            "moves_robot": True,
            "required_control_mode": "moveit_planning",
            "parameters": {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
            "recovery_policy": "never_retry",
        },
        "primitive_sequence": [{"primitive_name": "open_gripper"}],
    }
    return SkillSnapshot(
        robot_name="test_robot",
        profile_name="test",
        primitive_contract_digest={
            1: PRIMITIVE_CONTRACT_V1.digest,
            2: PRIMITIVE_CONTRACT_V2.digest,
            3: PRIMITIVE_CONTRACT_V3.digest,
        }[context_schema_version],
        robot_context=robot,
        delegated_executors={},
        templates={name: template},
        semantic_levels={name: "skill"},
        aliases={name: ()},
        parameter_schemas={name: template["capability"]["parameters"]},
        requirements={name: frozenset()},
        provenance={"schema_version": 1, "source_release_digest": "source"},
        enabled_skill_names=(name,),
        planner_visible_skill_names=(name,),
        capability_view={name: template["capability"]},
    )


def _activate(cache: SafetySnapshotCache, snapshot: SkillSnapshot, generation: int = 1, *, current: bool = True):
    return cache.activate(
        registry_epoch="epoch-1",
        generation=generation,
        registry_digest=snapshot.registry_digest,
        capability_digest=snapshot.capability_digest,
        provenance_digest=snapshot.provenance_digest,
        snapshot_json=snapshot.snapshot_json,
        make_current=current,
    )


def test_cache_verifies_and_returns_exact_generation() -> None:
    cache = SafetySnapshotCache()
    snapshot = _snapshot()

    cached = _activate(cache, snapshot)

    assert cache.current_identity == cached.identity
    assert cache.get(cached.identity).templates["open_gripper_skill"]["primitive_sequence"][0]["primitive_name"] == (
        "open_gripper"
    )


def test_cache_accepts_v2_primitive_contract_digest_selected_by_context() -> None:
    cache = SafetySnapshotCache()
    snapshot = _snapshot(context_schema_version=2)

    cached = _activate(cache, snapshot)

    assert cached.payload["registry_preimage"]["primitive_contract_digest"] == PRIMITIVE_CONTRACT_V2.digest
    assert cached.robot_context["context_schema_version"] == 2


def test_cache_accepts_v3_hybrid_context_and_supported_control_modes() -> None:
    cache = SafetySnapshotCache()
    snapshot = _snapshot(context_schema_version=3)

    cached = _activate(cache, snapshot)

    assert cached.payload["registry_preimage"]["primitive_contract_digest"] == PRIMITIVE_CONTRACT_V3.digest
    assert cached.robot_context["context_schema_version"] == 3
    assert cached.robot_context["supported_control_modes"] == ("moveit_planning", "base_navigation")


def test_cache_rejects_digest_and_canonical_payload_mismatches() -> None:
    cache = SafetySnapshotCache()
    snapshot = _snapshot()

    with pytest.raises(SnapshotCacheError) as digest_error:
        cache.activate(
            registry_epoch="epoch-1",
            generation=1,
            registry_digest="wrong",
            capability_digest=snapshot.capability_digest,
            provenance_digest=snapshot.provenance_digest,
            snapshot_json=snapshot.snapshot_json,
            make_current=True,
        )
    assert digest_error.value.code == "SKILL_SNAPSHOT_DIGEST_MISMATCH"

    with pytest.raises(SnapshotCacheError) as canonical_error:
        cache.activate(
            registry_epoch="epoch-1",
            generation=1,
            registry_digest=snapshot.registry_digest,
            capability_digest=snapshot.capability_digest,
            provenance_digest=snapshot.provenance_digest,
            snapshot_json=snapshot.snapshot_json + " ",
            make_current=True,
        )
    assert canonical_error.value.code == "SKILL_SNAPSHOT_DIGEST_MISMATCH"


def test_missing_or_wrong_exact_identity_fails_closed() -> None:
    cache = SafetySnapshotCache()
    snapshot = _snapshot()
    _activate(cache, snapshot)

    with pytest.raises(SnapshotCacheError) as missing:
        cache.get(SnapshotIdentity("epoch-1", 2, snapshot.registry_digest))
    assert missing.value.code == "SKILL_SNAPSHOT_NOT_RETAINED"

    with pytest.raises(SnapshotCacheError) as stale:
        cache.get(SnapshotIdentity("epoch-1", 1, "wrong"))
    assert stale.value.code == "SKILL_REGISTRY_VERSION_MISMATCH"


def test_reconcile_keeps_gateway_retained_and_two_recent_generations() -> None:
    cache = SafetySnapshotCache()
    snapshots = [_snapshot(f"skill-{generation}") for generation in range(1, 5)]
    for generation, snapshot in enumerate(snapshots, start=1):
        _activate(cache, snapshot, generation, current=generation == 4)

    cache.reconcile("epoch-1", {1, 4})

    assert cache.get(SnapshotIdentity("epoch-1", 1, snapshots[0].registry_digest))
    assert cache.get(SnapshotIdentity("epoch-1", 4, snapshots[3].registry_digest))
    with pytest.raises(SnapshotCacheError):
        cache.get(SnapshotIdentity("epoch-1", 2, snapshots[1].registry_digest))
    assert cache.get(SnapshotIdentity("epoch-1", 3, snapshots[2].registry_digest))


def test_reconcile_keeps_recent_current_after_gateway_releases_it() -> None:
    cache = SafetySnapshotCache()
    old = _snapshot("old")
    new = _snapshot("new")
    _activate(cache, old, 1, current=True)
    _activate(cache, new, 2, current=False)

    cache.reconcile("epoch-1", {2})

    assert cache.current_identity == SnapshotIdentity("epoch-1", 1, old.registry_digest)
    assert cache.get(SnapshotIdentity("epoch-1", 1, old.registry_digest))


def test_validate_skill_uses_exact_cached_snapshot_and_reports_current_on_miss() -> None:
    node = object.__new__(SafetyGuardNode)
    node._snapshot_cache = SafetySnapshotCache()
    node._debug = False
    snapshot = _snapshot()
    cached = _activate(node._snapshot_cache, snapshot)
    request = ValidateSkill.Request()
    request.schema_version = 1
    request.dispatch_binding.schema_version = 1
    request.dispatch_binding.expected_registry_epoch = cached.identity.registry_epoch
    request.dispatch_binding.expected_registry_generation = cached.identity.generation
    request.dispatch_binding.expected_registry_digest = cached.identity.registry_digest
    request.skill_name = "open_gripper_skill"

    allowed = node._handle_validate_skill(request, ValidateSkill.Response())

    assert allowed.allowed is True
    assert allowed.actual_registry_digest == cached.identity.registry_digest

    request.dispatch_binding.expected_registry_generation = 2
    missing = node._handle_validate_skill(request, ValidateSkill.Response())
    assert missing.allowed is False
    assert missing.error_code == "SKILL_SNAPSHOT_NOT_RETAINED"
    assert missing.actual_registry_generation == 1


def test_validate_skill_rejects_dispatch_nonce() -> None:
    node = object.__new__(SafetyGuardNode)
    node._snapshot_cache = SafetySnapshotCache()
    node._debug = False
    cached = _activate(node._snapshot_cache, _snapshot())
    request = ValidateSkill.Request()
    request.schema_version = 1
    request.dispatch_binding.schema_version = 1
    request.dispatch_binding.expected_registry_epoch = cached.identity.registry_epoch
    request.dispatch_binding.expected_registry_generation = cached.identity.generation
    request.dispatch_binding.expected_registry_digest = cached.identity.registry_digest
    request.dispatch_binding.dispatch_nonce = "execution-nonce"
    request.skill_name = "open_gripper_skill"

    response = node._handle_validate_skill(request, ValidateSkill.Response())

    assert response.allowed is False
    assert response.error_code == "SKILL_SCHEMA_INVALID"
