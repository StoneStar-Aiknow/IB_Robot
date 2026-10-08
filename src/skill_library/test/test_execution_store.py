import pytest

from embodied_common.program_contracts import canonicalize_program
from skill_library.execution_store import (
    ExecutionBusyError,
    ExecutionConflictError,
    ExecutionQuarantinedError,
    SQLiteExecutionStore,
)


def _program(**overrides):
    values = {
        "schema_version": 1,
        "request_id": "request-1",
        "caller_scope": "agent",
        "robot_id": "robot",
        "expected_registry_epoch": "epoch",
        "expected_registry_generation": 1,
        "expected_registry_digest": "digest",
        "workflow_steps": [{"schema_version": 1, "skill_name": "wave_hello"}],
        "task_budget_sec": 30.0,
    }
    values.update(overrides)
    return canonicalize_program(**values)


def test_same_request_same_program_is_idempotent_and_payload_change_conflicts(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "executions.sqlite3")
    first, created = store.begin(_program())
    replay, replay_created = store.begin(_program())
    assert created is True
    assert replay_created is False
    assert replay.execution_id == first.execution_id
    with pytest.raises(ExecutionConflictError):
        store.begin(_program(task_budget_sec=31.0))


def test_restart_quarantines_commit_intent_and_blocks_new_motion(tmp_path):
    path = tmp_path / "executions.sqlite3"
    store = SQLiteExecutionStore(path)
    record, _ = store.begin(_program())
    store.mark_committing(record.execution_id)
    store.close()

    recovered = SQLiteExecutionStore(path)
    record = recovered.get(execution_id=record.execution_id)
    assert record.state == "UNKNOWN"
    assert record.error_code == "EXECUTION_RECOVERY_UNKNOWN"
    assert recovered.has_unresolved_unknown() is True
    with pytest.raises(ExecutionQuarantinedError):
        recovered.begin(_program(request_id="request-2"))


def test_restart_fails_precommit_record_without_unknown_quarantine(tmp_path):
    path = tmp_path / "executions.sqlite3"
    store = SQLiteExecutionStore(path)
    record, _ = store.begin(_program())
    store.close()
    recovered = SQLiteExecutionStore(path)
    record = recovered.get(execution_id=record.execution_id)
    assert record.state == "FAILED"
    assert record.error_code == "EXECUTION_RESTARTED_BEFORE_COMMIT"
    assert recovered.has_unresolved_unknown() is False


def test_restart_quarantines_finalizing_barrier(tmp_path):
    path = tmp_path / "executions.sqlite3"
    store = SQLiteExecutionStore(path)
    record, _ = store.begin(_program())
    store.transition(record.execution_id, state="FINALIZING", completed_step_count=1)
    store.close()
    recovered = SQLiteExecutionStore(path)
    assert recovered.get(execution_id=record.execution_id).state == "UNKNOWN"


def test_active_program_serializes_robot_admission(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "executions.sqlite3")
    store.begin(_program())
    with pytest.raises(ExecutionBusyError):
        store.begin(_program(request_id="request-2"))


def test_stop_state_is_monotonic_and_events_are_ordered(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "executions.sqlite3")
    record, _ = store.begin(_program())
    store.mark_committing(record.execution_id)
    first = store.request_stop(record.execution_id)
    second = store.request_stop(record.execution_id)
    assert first.stop_generation == 1
    assert second.stop_generation == 1
    assert second.last_event_sequence == first.last_event_sequence
    assert second.stop_requested is True
    events = store.events_after(record.execution_id)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[-1].state == "STOPPING"


def test_terminal_result_is_immutable(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "executions.sqlite3")
    record, _ = store.begin(_program())
    store.finish(record.execution_id, state="SUCCEEDED", completed_step_count=1)
    assert store.finish(record.execution_id, state="SUCCEEDED", completed_step_count=1).state == "SUCCEEDED"
    with pytest.raises(ExecutionConflictError):
        store.finish(record.execution_id, state="FAILED", completed_step_count=0)


def test_workflow_digest_binding_is_immutable(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "executions.sqlite3")
    record, _ = store.begin(_program())
    assert store.bind_workflow_digest(record.execution_id, "workflow-a").workflow_digest == "workflow-a"
    assert store.bind_workflow_digest(record.execution_id, "workflow-a").workflow_digest == "workflow-a"
    with pytest.raises(ExecutionConflictError):
        store.bind_workflow_digest(record.execution_id, "workflow-b")


def test_stop_before_admission_survives_restart_and_suppresses_late_goal(tmp_path):
    path = tmp_path / "executions.sqlite3"
    store = SQLiteExecutionStore(path)
    assert store.request_stop_by_key("agent", "robot", "request-1") is None
    assert store.request_stop_by_key("agent", "robot", "request-1") is None
    store.close()
    store = SQLiteExecutionStore(path)
    record, created = store.begin(_program())
    assert created
    assert record.state == "CANCELED"
    assert record.stop_requested
    assert record.completed_step_count == 0
    assert store.begin(_program())[1] is False
    with pytest.raises(ExecutionConflictError):
        store.begin(_program(task_budget_sec=1.0))
    # A genuinely new request is not accidentally canceled.
    assert store.begin(_program(request_id="request-2"))[0].state == "RECEIVED"
    store.close()


def test_reconcile_preserves_unknown_result_and_audits_release(tmp_path):
    path = tmp_path / "executions.sqlite3"
    store = SQLiteExecutionStore(path)
    original, _ = store.begin(_program())
    store.mark_committing(original.execution_id)
    store.close()
    store = SQLiteExecutionStore(path)
    with pytest.raises(ValueError):
        store.resolve_unknown(original.execution_id, operator_scope="operator", evidence="")
    assert store.has_unresolved_unknown()
    store.resolve_unknown(original.execution_id, operator_scope="operator", evidence="drives stopped; queues cleared")
    assert not store.has_unresolved_unknown()
    assert store.get(execution_id=original.execution_id).state == "UNKNOWN"
    assert store.events_after(original.execution_id)[-1].error_code == "EXECUTION_RECONCILED"
    assert store.begin(_program())[0].state == "UNKNOWN"
    new, _ = store.begin(_program(request_id="request-2"))
    assert new.state == "RECEIVED"
    store.close()


def test_second_writer_cannot_recover_a_live_gateway(tmp_path):
    path = tmp_path / "executions.sqlite3"
    first = SQLiteExecutionStore(path)
    record, _ = first.begin(_program())
    first.mark_active(record.execution_id)
    with pytest.raises(RuntimeError, match="active writer"):
        SQLiteExecutionStore(path)
    assert first.get(execution_id=record.execution_id).state == "ACTIVE"
    first.close()
    recovered = SQLiteExecutionStore(path)
    assert recovered.get(execution_id=record.execution_id).state == "UNKNOWN"
    recovered.close()
