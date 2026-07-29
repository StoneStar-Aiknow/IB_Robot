import threading
from dataclasses import replace

import pytest

from skill_library import gateway_policy
from skill_library.gateway_policy import (
    BoundedRequestLedger,
    ExecutionOwner,
    GatewayPolicy,
    GatewayRequest,
    RootExecutionLease,
    RuntimeSnapshot,
    SkillRequirements,
    build_skill_parameter_schemas,
    build_skill_requirements,
)

TIMEOUT_POLICY = {
    "default_skill_timeout_sec": 5.0,
    "task_budget_sec": 10.0,
}


def _parameters(properties, required):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


EXPANDED_SKILL_TEMPLATES = {
    "named": {
        "capability": {"parameters": _parameters({}, [])},
        "initial_gripper_state": "open",
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose"},
            {"primitive_name": "open_gripper"},
        ],
    },
    "relative": {
        "capability": {
            "parameters": _parameters(
                {
                    "motion_direction": {"type": "string", "enum": ["left"]},
                    "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
                },
                ["motion_direction", "motion_distance"],
            )
        },
        "primitive_sequence": [
            {"primitive_name": "move_relative_ee"},
            {"primitive_name": "rotate_gripper_cw"},
        ],
    },
    "joint": {
        "capability": {"parameters": _parameters({}, [])},
        "primitive_sequence": [
            {"primitive_name": "move_to_joint_positions"},
            {"primitive_name": "move_through_joint_positions"},
        ],
    },
}


def _policy() -> GatewayPolicy:
    return GatewayPolicy(
        TIMEOUT_POLICY,
        build_skill_requirements(EXPANDED_SKILL_TEMPLATES),
        parameter_schemas=build_skill_parameter_schemas(EXPANDED_SKILL_TEMPLATES),
    )


def _atomic_policy() -> tuple[GatewayPolicy, BoundedRequestLedger, RootExecutionLease]:
    ledger = BoundedRequestLedger(2)
    lease = RootExecutionLease()
    return (
        GatewayPolicy(
            TIMEOUT_POLICY,
            build_skill_requirements(EXPANDED_SKILL_TEMPLATES),
            parameter_schemas=build_skill_parameter_schemas(EXPANDED_SKILL_TEMPLATES),
            ledger=ledger,
            lease=lease,
        ),
        ledger,
        lease,
    )


def _request(**overrides) -> GatewayRequest:
    values = {
        "task_id": "task-1",
        "skill_name": "relative",
        "motion_direction": "left",
        "motion_distance": 0.1,
    }
    values.update(overrides)
    return GatewayRequest(**values)


def _snapshot(**overrides) -> RuntimeSnapshot:
    values = {
        "motion_authorized": True,
        "active_control_mode": "cartesian",
        "required_control_mode": "cartesian",
        "busy": False,
        "active_task_id": "",
        "validate_ready": True,
        "task_executor_ready": True,
        "arm_trajectory_ready": True,
        "ee_pose_fresh": True,
    }
    values.update(overrides)
    return RuntimeSnapshot(**values)


def test_policy_resolves_identity_through_shared_payload_and_hash_api(monkeypatch):
    policy = _policy()
    payload_calls = []
    hash_calls = []
    original_payload = gateway_policy.skill_request.canonical_skill_payload
    original_hash = gateway_policy.skill_request.skill_payload_hash

    def canonical_payload(*args, **kwargs):
        payload_calls.append((args, kwargs))
        return original_payload(*args, **kwargs)

    def payload_hash(payload):
        hash_calls.append(payload)
        return original_hash(payload)

    monkeypatch.setattr(gateway_policy.skill_request, "canonical_skill_payload", canonical_payload)
    monkeypatch.setattr(gateway_policy.skill_request, "skill_payload_hash", payload_hash)

    prepared = policy.prepare(_request(timeout_sec=None))

    assert prepared.effective_timeout_sec == 5.0
    assert payload_calls[0][1]["default_timeout_sec"] == 5.0
    assert hash_calls == [prepared.payload]


@pytest.mark.parametrize("timeout_sec", [0.0, -0.1, float("inf")])
def test_policy_keeps_invalid_request_timeout_as_local_value_error(timeout_sec):
    with pytest.raises(ValueError, match="timeout_sec"):
        _policy().prepare(_request(timeout_sec=timeout_sec))


def test_policy_rejects_timeout_over_budget_without_truncating_effective_timeout():
    decision = _policy().evaluate(_request(timeout_sec=10.1), _snapshot())

    assert not decision.admitted
    assert decision.error_code == "TIMEOUT_EXCEEDS_POLICY"
    assert decision.effective_timeout_sec == 10.1


@pytest.mark.parametrize(
    ("snapshot", "timeout_sec", "expected_reason"),
    [
        (
            _snapshot(motion_authorized=False, active_control_mode="wrong", busy=True, validate_ready=False),
            20.0,
            "MOTION_NOT_AUTHORIZED: operator authorization is disabled",
        ),
        (
            _snapshot(active_control_mode="yaml_default", busy=True, validate_ready=False),
            20.0,
            "CONTROL_MODE_MISMATCH: requires cartesian, active mode is yaml_default",
        ),
        (
            _snapshot(busy=True, active_task_id="other-task", validate_ready=False),
            20.0,
            "SKILL_BUSY: another root execution is active",
        ),
        (_snapshot(validate_ready=False), 20.0, "TIMEOUT_EXCEEDS_POLICY: "),
        (_snapshot(validate_ready=False), None, "CAPABILITY_NOT_READY: validate skill service unavailable"),
    ],
)
def test_policy_applies_stable_gate_priority(snapshot, timeout_sec, expected_reason):
    decision = _policy().evaluate(_request(timeout_sec=timeout_sec), snapshot)

    assert not decision.admitted
    assert f"{decision.error_code}: {decision.message}" == expected_reason


def test_policy_uses_snapshot_active_mode_and_allows_active_owner_child():
    root_owner = ExecutionOwner.skill_command("active-task")
    child_owner = ExecutionOwner.internal_child(root_owner, "open-gripper")
    snapshot = _snapshot(
        active_control_mode="runtime_mode",
        required_control_mode="runtime_mode",
        busy=True,
        active_task_id="active-task",
    )

    decision = _policy().evaluate(_request(task_id="active-task"), snapshot, owner=child_owner)

    assert decision.admitted
    assert decision.error_code == ""


def test_policy_returns_stable_readiness_reason():
    decision = _policy().evaluate(
        _request(skill_name="relative"),
        _snapshot(task_executor_ready=False, ee_pose_fresh=False),
    )

    assert not decision.admitted
    assert decision.error_code == "CAPABILITY_NOT_READY"
    assert decision.message == "task executor action unavailable"
    assert decision.readiness.required == ("validate_skill", "task_executor", "fresh_ee_pose")
    assert decision.readiness.unavailable == ("task_executor", "fresh_ee_pose")


@pytest.mark.parametrize(
    "gateway_request",
    [
        _request(motion_direction="right"),
        _request(motion_direction=""),
        _request(motion_distance=0.0),
        _request(target_name="object"),
        _request(skill_name="joint", motion_direction="left", motion_distance=None),
    ],
    ids=["enum", "missing-required", "non-positive-distance", "undeclared-string", "undeclared-motion"],
)
def test_policy_rejects_capability_parameter_contract_without_state_mutation(gateway_request):
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command(str(gateway_request.task_id))

    decision = policy.admit(gateway_request, _snapshot(), owner)

    assert not decision.admitted
    assert decision.error_code == gateway_policy.SKILL_REJECTED
    assert ledger.query(str(gateway_request.task_id)) == gateway_policy.LedgerQuery()
    assert lease.owner is None


def test_policy_readiness_evaluation_skips_request_parameter_validation():
    decision = _policy().evaluate(
        GatewayRequest(task_id="status-relative", skill_name="relative"),
        _snapshot(),
        validate_parameters=False,
    )

    assert decision.admitted


@pytest.mark.parametrize(
    ("skill_name", "snapshot", "expected_message"),
    [
        ("relative", _snapshot(validate_ready=False), "validate skill service unavailable"),
        ("relative", _snapshot(task_executor_ready=False), "task executor action unavailable"),
        ("joint", _snapshot(arm_trajectory_ready=False), "arm trajectory action unavailable"),
        ("relative", _snapshot(ee_pose_fresh=False), "ee pose unavailable or stale"),
    ],
)
def test_policy_reports_first_unavailable_required_capability(skill_name, snapshot, expected_message):
    decision = _policy().evaluate(_request(skill_name=skill_name), snapshot, validate_parameters=False)

    assert not decision.admitted
    assert decision.error_code == "CAPABILITY_NOT_READY"
    assert decision.message == expected_message


def test_build_skill_requirements_maps_primitives_and_initial_gripper_state():
    requirements = build_skill_requirements(EXPANDED_SKILL_TEMPLATES)

    assert requirements["named"] == SkillRequirements(validate_skill=True, task_executor=True)
    assert requirements["relative"] == SkillRequirements(
        validate_skill=True,
        task_executor=True,
        fresh_ee_pose=True,
    )
    assert requirements["joint"] == SkillRequirements(validate_skill=True, arm_trajectory=True)

    with pytest.raises(ValueError, match="unknown primitive"):
        build_skill_requirements({"unknown": {"primitive_sequence": [{"primitive_name": "not_real"}]}})


def test_root_execution_lease_uses_opaque_token_for_reuse_and_release():
    lease = RootExecutionLease()
    root_owner = ExecutionOwner.skill_command("task-1")
    child_owner = ExecutionOwner.internal_child(root_owner, "child")
    external_owner = ExecutionOwner.external_primitive("manual-1")
    rebuilt_owner = ExecutionOwner.skill_command("task-1")

    token = lease.acquire(root_owner)

    assert token is not None
    assert lease.acquire(root_owner) is token
    assert lease.acquire(rebuilt_owner) is None
    assert lease.acquire(child_owner) is None
    child_borrow = lease.reuse(child_owner, token)
    assert child_borrow is not None
    assert child_borrow is not token
    assert not lease.release(child_borrow)
    assert lease.acquire(external_owner) is None
    assert not lease.release(rebuilt_owner)
    assert lease.owner is root_owner
    assert lease.release(token)
    assert lease.acquire(external_owner) is not None


def test_finalize_is_fail_closed_until_terminalization_succeeds(monkeypatch):
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(_request(), _snapshot(), owner)
    original_terminal = ledger.terminal

    def fail_terminal(*args, **kwargs):
        raise RuntimeError("injected terminal failure")

    monkeypatch.setattr(ledger, "terminal", fail_terminal)
    with pytest.raises(RuntimeError, match="terminalization failed"):
        policy.finalize(admission)

    assert lease.owner is owner
    assert ledger.query("task-1", admission.prepared_request.identity.payload_hash).state == "active"
    competing = policy.admit(_request(task_id="task-2"), _snapshot(), ExecutionOwner.skill_command("task-2"))
    assert competing.error_code == "SKILL_BUSY"

    monkeypatch.setattr(ledger, "terminal", original_terminal)
    assert policy.finalize(admission).state == "terminal"
    assert lease.owner is None


def test_child_borrow_cannot_release_or_finalize_root_admission():
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command("task-1")
    child = ExecutionOwner.internal_child(owner, "primitive")
    admission = policy.admit(_request(), _snapshot(), owner)
    borrow = lease.reuse(child, admission.lease_token)

    assert borrow is not None
    assert not lease.release(borrow)
    with pytest.raises(ValueError, match="original active admission"):
        policy.finalize(borrow)
    assert lease.owner is owner
    assert ledger.query("task-1").state == "active"
    assert policy.finalize(admission).state == "terminal"


def test_policy_borrow_internal_requires_exact_active_admission_and_keeps_root_lease():
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(_request(), _snapshot(), owner)

    borrow = policy.borrow_internal(admission, "task-1", "child")

    assert borrow is not None
    assert not lease.release(borrow)
    assert policy.borrow_internal(replace(admission), "task-1", "child") is None
    assert policy.borrow_internal(admission, "other-task", "child") is None
    assert lease.owner is owner
    assert ledger.query("task-1").state == "active"
    assert policy.finalize(admission).state == "terminal"


def test_policy_externally_admits_and_releases_primitive_lease_atomically():
    policy, _ledger, lease = _atomic_policy()

    error_code, token = policy.admit_external_primitive("manual-task", _snapshot())
    second_error_code, second_token = policy.admit_external_primitive("second-task", _snapshot())

    assert error_code == ""
    assert token is not None
    assert lease.owner == ExecutionOwner.external_primitive("manual-task")
    assert second_error_code == "SKILL_BUSY"
    assert second_token is None
    assert policy.release_external_primitive(token)
    assert lease.owner is None


def test_finalize_accepts_only_the_original_admission_object():
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(_request(), _snapshot(), owner)

    with pytest.raises(ValueError, match="original active admission"):
        policy.finalize(replace(admission))

    assert lease.owner is owner
    assert ledger.query("task-1").state == "active"
    assert policy.finalize(admission).state == "terminal"


def test_bounded_request_ledger_keeps_active_records_and_evicts_terminal_records_fifo():
    with pytest.raises(ValueError, match="capacity"):
        BoundedRequestLedger(0)

    ledger = BoundedRequestLedger(2)
    ledger.begin("active", "active-hash")
    ledger.begin("one", "one-hash")
    first_terminal = ledger.terminal("one", "one-hash", error_code="FIRST", terminal_metadata={"source": "test"})
    ledger.begin("two", "two-hash")
    ledger.terminal("two", "two-hash")
    assert ledger.query("one", "one-hash").state == "terminal"
    ledger.begin("three", "three-hash")
    ledger.terminal("three", "three-hash")

    assert first_terminal.error_code == "FIRST"
    assert first_terminal.terminal_metadata == {"source": "test"}
    assert ledger.query("active").state == "active"
    assert ledger.query("one").state == ""
    assert ledger.query("two").state == "terminal"
    assert ledger.query("three").state == "terminal"


def test_ledger_query_matrix():
    ledger = BoundedRequestLedger(2)

    assert ledger.query() == gateway_policy.LedgerQuery()
    assert ledger.query("", "hash") == gateway_policy.LedgerQuery(error_code="INVALID_ARGUMENT")
    assert ledger.query("unknown") == gateway_policy.LedgerQuery()
    assert ledger.query("unknown", "hash") == gateway_policy.LedgerQuery()

    ledger.begin("active", "active-hash")
    assert ledger.query("active") == gateway_policy.LedgerQuery(state="active")
    assert ledger.query("active", "active-hash") == gateway_policy.LedgerQuery(
        state="active",
        error_code="DUPLICATE_TASK_ID",
    )
    assert ledger.query("active", "different-hash") == gateway_policy.LedgerQuery(
        state="active",
        error_code="TASK_ID_CONFLICT",
    )

    ledger.terminal("active", "active-hash", error_code="MOTION_NOT_AUTHORIZED")
    assert ledger.query("active") == gateway_policy.LedgerQuery(state="terminal")
    assert ledger.query("active", "active-hash") == gateway_policy.LedgerQuery(
        state="terminal",
        error_code="DUPLICATE_TASK_ID",
    )


def test_ledger_begin_request_uses_shared_identity_api(monkeypatch):
    calls = []
    original_payload = gateway_policy.skill_request.canonical_skill_payload

    def canonical_payload(*args, **kwargs):
        calls.append((args, kwargs))
        return original_payload(*args, **kwargs)

    monkeypatch.setattr(gateway_policy.skill_request, "canonical_skill_payload", canonical_payload)

    record = BoundedRequestLedger(1).begin_request(_request(), default_timeout_sec=5.0)

    assert record.task_id == "task-1"
    assert calls[0][1]["default_timeout_sec"] == 5.0


def test_executor_lifecycle_has_explicit_begin_terminal_and_release_without_early_rejection_residue():
    policy, ledger, lease = _atomic_policy()
    denied_request = _request(timeout_sec=20.0)
    denied_identity = policy.prepare(denied_request).identity

    denied = policy.admit(denied_request, _snapshot(), ExecutionOwner.skill_command("task-1"))

    assert not denied.admitted
    assert lease.owner is None
    assert ledger.query(denied_identity.task_id, denied_identity.payload_hash) == gateway_policy.LedgerQuery()

    accepted_request = _request(task_id="task-2")
    admitted = policy.admit(accepted_request, _snapshot(), ExecutionOwner.skill_command("task-2"))
    terminal = policy.finalize(admitted)

    assert admitted.admitted
    assert terminal.state == "terminal"
    assert ledger.query("task-2").state == "terminal"
    assert lease.owner is None


def test_atomic_admission_allows_one_concurrent_root_and_finalization_reopens_lease():
    policy, ledger, lease = _atomic_policy()
    barrier = threading.Barrier(3)
    results = {}

    def admit(task_id):
        owner = ExecutionOwner.skill_command(task_id)
        barrier.wait()
        results[task_id] = (owner, policy.admit(_request(task_id=task_id), _snapshot(), owner))

    first = threading.Thread(target=admit, args=("first",))
    second = threading.Thread(target=admit, args=("second",))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    admitted = [(task_id, owner, decision) for task_id, (owner, decision) in results.items() if decision.admitted]
    rejected = [(task_id, decision) for task_id, (_, decision) in results.items() if not decision.admitted]

    assert len(admitted) == 1
    assert rejected[0][1].error_code == "SKILL_BUSY"
    winner_task_id, winner_owner, winner = admitted[0]
    assert lease.owner is winner_owner
    assert ledger.query(winner_task_id, winner.prepared_request.identity.payload_hash).state == "active"
    assert ledger.query(rejected[0][0], rejected[0][1].prepared_request.identity.payload_hash).state == ""

    assert policy.finalize(winner).state == "terminal"
    retry_owner = ExecutionOwner.skill_command(rejected[0][0])
    retry = policy.admit(_request(task_id=rejected[0][0]), _snapshot(), retry_owner)

    assert retry.admitted
    assert lease.owner is retry_owner
    assert policy.finalize(retry).state == "terminal"
    assert lease.owner is None


@pytest.mark.parametrize(
    "gateway_request",
    [
        _request(task_id=""),
        _request(timeout_sec=0.0),
        _request(motion_distance=-0.1),
        _request(skill_name="unknown"),
    ],
    ids=["empty-task-id", "bad-timeout", "bad-number", "unknown-skill"],
)
def test_policy_returns_skill_rejected_without_mutating_state_for_invalid_requests(gateway_request):
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command(str(gateway_request.task_id))

    evaluated = policy.evaluate(gateway_request, _snapshot(), owner=owner)
    admitted = policy.admit(gateway_request, _snapshot(), owner)

    assert not evaluated.admitted
    assert evaluated.error_code == gateway_policy.SKILL_REJECTED
    assert not admitted.admitted
    assert admitted.error_code == gateway_policy.SKILL_REJECTED
    assert evaluated.error_code != gateway_policy.INVALID_ARGUMENT
    assert ledger.query(str(gateway_request.task_id)) == gateway_policy.LedgerQuery()
    assert lease.owner is None


def test_atomic_admission_rejects_owner_identity_mismatch_without_state_mutation():
    policy, ledger, lease = _atomic_policy()
    request = _request(task_id="task-1")
    prepared = policy.prepare(request)

    decision = policy.admit(request, _snapshot(), ExecutionOwner.skill_command("other-task"))

    assert not decision.admitted
    assert decision.error_code == gateway_policy.SKILL_REJECTED
    assert ledger.query(prepared.identity.task_id, prepared.identity.payload_hash) == gateway_policy.LedgerQuery()
    assert lease.owner is None


def test_prepared_request_payload_is_immutable_after_shared_hashing():
    prepared = _policy().prepare(_request())

    with pytest.raises(TypeError):
        prepared.payload["skill_name"] = "different"

    assert prepared.identity.payload_hash == gateway_policy.skill_request.skill_payload_hash(dict(prepared.payload))


def test_atomic_admission_returns_duplicate_or_conflict_without_new_lease_or_active_record():
    policy, ledger, lease = _atomic_policy()
    owner = ExecutionOwner.skill_command("task-1")
    request = _request()

    first = policy.admit(request, _snapshot(), owner)
    duplicate = policy.admit(request, _snapshot(), owner)
    conflict = policy.admit(_request(motion_distance=0.2), _snapshot(), owner)

    assert first.admitted
    assert not duplicate.admitted
    assert duplicate.error_code == gateway_policy.DUPLICATE_TASK_ID
    assert not conflict.admitted
    assert conflict.error_code == gateway_policy.TASK_ID_CONFLICT
    assert lease.owner is owner
    assert ledger.query("task-1", first.prepared_request.identity.payload_hash).state == "active"
    assert policy.finalize(first).state == "terminal"
