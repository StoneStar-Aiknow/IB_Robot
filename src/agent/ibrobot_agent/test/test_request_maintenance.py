import json
import sqlite3
from datetime import datetime, timezone

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import AgentRequest, Presentation, TaskRef, request_hash_text
from ibrobot_agent.deployment_lock import DeploymentLock
from ibrobot_agent.request_maintenance import reconcile
from ibrobot_agent.request_store import SQLiteRequestStore


def request(request_id="old"):
    return AgentRequest(
        1, request_id, "session", "test", "operator", "so101_single_arm", "hello", datetime.now(timezone.utc)
    )


def seed(store, value):
    key = value.to_key()
    store.admit(key, input_hash=request_hash_text(value), session_id=value.session_id)
    store.begin_planning(key, expected_generation=0)
    store.mark_proposal_ready(key, expected_generation=1)
    store.mark_preparing(key, expected_generation=1)
    ref = TaskRef("task-" + value.request_id, "plan", "pdig", "epoch", 1, "rdig", 1)
    presentation = Presentation(
        ref, 1, (CanonicalWorkflowStep(1, "wave_hello"),), "immediate_after_presentation", 30, "wave"
    )
    store.bind_task(key, expected_generation=1, task_ref=ref, presentation=presentation, proposal_json="{}")
    return store.mark_submitted(key, expected_generation=1, task_ref=ref)


@pytest.fixture
def ledgers(tmp_path):
    requests, executions, lock = (tmp_path / n for n in ("requests.sqlite3", "executions.sqlite3", "agent.lock"))
    lock.touch()
    value = request()
    store = SQLiteRequestStore(requests)
    record = seed(store, value)
    record = store.quarantine(value.to_key(), expected_generation=1, reason="message too long")
    original = record.terminal
    store.close()
    terminal = {
        "execution_id": "execution-old",
        "request_id": value.request_id,
        "caller_scope": "agent_incubation:operator",
        "robot_id": value.robot_scope,
        "program_digest": "pdig",
        "state": "FAILED",
        "completed_step_count": 0,
        "actual_registry_epoch": "epoch",
        "actual_registry_generation": 1,
        "actual_registry_digest": "rdig",
        "error_code": "HAND_PRESERVED_MOVED",
        "message": "drift",
    }
    with sqlite3.connect(executions) as db:
        fields = ",".join(f"{k} {'INTEGER' if isinstance(v, int) else 'TEXT'}" for k, v in terminal.items())
        db.execute(f"CREATE TABLE executions ({fields})")
        db.execute("INSERT INTO executions VALUES (" + ",".join("?" for _ in terminal) + ")", tuple(terminal.values()))
    return requests, executions, lock, value.to_key(), original


def run(data):
    requests, executions, lock, key, _ = data
    reconcile(
        requests,
        executions,
        lock,
        key,
        caller_scope="agent_incubation:operator",
        evidence="exact saved Gateway terminal",
    )


def test_reconciliation_keeps_original_terminal_and_audit_across_restart(ledgers):
    run(ledgers)
    run(ledgers)  # Identical evidence is idempotent.
    requests, executions, _, key, original = ledgers
    with sqlite3.connect(executions) as db:
        assert db.execute("SELECT state FROM executions").fetchone()[0] == "FAILED"
    store = SQLiteRequestStore(requests)
    try:
        assert store.get_request(key).state == "UNKNOWN"
        assert store.get_request(key).terminal == original
        assert not store.is_robot_quarantined(key.robot_scope)
        proof = store.latest_event_detail(key, "operator_reconciled")
        assert proof["gateway_terminal"]["error_code"] == "HAND_PRESERVED_MOVED"
        assert proof["operator_scope"].startswith("os:")
        # Original request is still deduplicated, never converted into a new execution.
        assert store.admit(key, input_hash=store.get_request(key).input_hash, session_id="session").state == "UNKNOWN"
    finally:
        store.close()
    with sqlite3.connect(requests) as db:
        assert db.execute("SELECT count(*) FROM request_resolutions").fetchone()[0] == 1
        assert json.loads(db.execute("SELECT proof_json FROM request_resolutions").fetchone()[0])["evidence"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("program_digest", "wrong"),
        ("actual_registry_epoch", "wrong"),
        ("actual_registry_generation", 2),
        ("actual_registry_digest", "wrong"),
        ("state", "UNKNOWN"),
        ("state", "RUNNING"),
        ("request_id", "other"),
        ("caller_scope", "other"),
        ("robot_id", "other"),
        ("completed_step_count", 2),
        ("state", "SUCCEEDED"),
    ],
)
def test_unproven_terminal_cannot_clear_quarantine(ledgers, field, value):
    with sqlite3.connect(ledgers[1]) as db:
        db.execute(f"UPDATE executions SET {field}=?", (value,))
    with pytest.raises(ValueError):
        run(ledgers)
    store = SQLiteRequestStore(ledgers[0])
    try:
        assert store.is_robot_quarantined(ledgers[3].robot_scope)
    finally:
        store.close()


@pytest.mark.parametrize("owner", ["agent", "gateway"])
def test_live_owner_prevents_maintenance(ledgers, owner):
    path = ledgers[2] if owner == "agent" else ledgers[1].with_suffix(".sqlite3.lock")
    lock = DeploymentLock(path)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            run(ledgers)
    finally:
        lock.close()


def test_one_resolution_does_not_clear_another_unknown_request(ledgers):
    store = SQLiteRequestStore(ledgers[0])
    key = request("other").to_key()
    seed(store, request("other"))
    store.quarantine(key, expected_generation=1, reason="unrelated unknown")
    store.close()
    run(ledgers)
    store = SQLiteRequestStore(ledgers[0])
    try:
        assert store.is_robot_quarantined(key.robot_scope)
    finally:
        store.close()
