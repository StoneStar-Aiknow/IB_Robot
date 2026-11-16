"""Admission ledger: observation is read-only, recovery is explicit.

The client keeps two local ledgers:

* ``_submitted`` maps a preview to the reference this client actually sent;
* ``_unresolved`` is the admission lock for a reference whose outcome this
  client does not know yet.

The contract exercised here:

* a read-only ``get_execution`` never turns a foreign reference into a lock;
* a real ``submit_prepared`` keeps the lock until a terminal or reconciled
  observation releases it;
* ``GOAL_REJECTED`` clears the submitted and unresolved entries synchronously,
  and a poll that was already in flight when the rejection arrived cannot
  revive the lock;
* ``restore_ref`` is the only explicit way to adopt a reference persisted by a
  previous process, so a Gateway/Web restart can recover a real unknown
  execution without letting an arbitrary GET deny admission.

The transport boundary is faked; the ledger logic is real. No ROS graph, no
robot and no motion is involved, and the concurrency case uses an explicit
barrier instead of sleeps.
"""

from __future__ import annotations

import copy
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from robot_client import workflow
from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transport import gateway_transport


def transport(**overrides):
    values = {"robot_id": "g1", "robot_name": "g1"}
    values.update(overrides)
    return gateway_transport({}, **values)


class Bridge:
    """Transport stub with deterministic send/observe barriers."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = 0
        self.goals = []
        self.receipt = True
        self.accepted = True
        self.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}
        self.events = {"found": True, "execution_id": "", "events": [], "last_event_sequence": 0, "has_more": False}
        self.get_calls = []
        self.events_calls = []
        self.get_started = threading.Event()
        self.get_release = threading.Event()
        self.block_get = False
        self.send_started = threading.Event()
        self.send_release = threading.Event()
        self.block_send = False

    def start(self):
        self.started += 1
        return True

    def close(self):
        pass

    def get_status(self, **kwargs):
        return {
            "robot_name": "g1",
            "registry_epoch": "epoch-1",
            "registry_generation": 1,
            "registry_digest": "registry-digest",
            "capability_digest": "a" * 64,
            "control_plane_ready": False,
        }

    def wait_for_program_server(self, **kwargs):
        return True

    def send_program_goal(self, **kwargs):
        self.goals.append(copy.deepcopy(kwargs))
        self.send_started.set()
        if self.block_send:
            assert self.send_release.wait(5.0), "the goal send barrier was not released"
        future = Future()
        future.set_result(SimpleNamespace(accepted=self.accepted))
        return future

    def wait_future(self, future, **kwargs):
        return self.receipt

    def get_execution(self, **kwargs):
        self.get_calls.append(copy.deepcopy(kwargs))
        self.get_started.set()
        if self.block_get:
            assert self.get_release.wait(5.0), "the observation barrier was not released"
        return copy.deepcopy(self.execution)

    def get_execution_events(self, **kwargs):
        self.events_calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(self.events)

    def cancel_execution(self, **kwargs):
        return {"accepted": True}


class Controller:
    """Replace the plan pipeline; the client still owns presentation and send."""

    def __init__(self, bridge, **kwargs):
        self.bridge = bridge

    def load_catalog(self):
        return None

    def prepare_workflow(self, raw, steps, *, request_id):
        return {
            "steps": copy.deepcopy(steps),
            "registry_identity": {
                "registry_epoch": "epoch-1",
                "registry_generation": 1,
                "registry_digest": "registry-digest",
            },
            "plan_id": "plan-1",
            "proposed_task_budget_sec": 30.0,
        }

    def confirm_plan(self):
        return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": False}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(workflow, "InteractiveController", Controller)
    instance = RobotClient(transport(), transport_factory=Bridge)
    try:
        yield instance
    finally:
        instance.close()


def preview(instance, request_id="planner-request"):
    return instance.prepare_workflow(
        {
            "workflow_steps": [{"skill_name": "wave_hand", "schema_version": 1}],
            "raw_command": "wave the hand",
            "request_id": request_id,
        }
    )


def reference(instance, request_id, execution_id=""):
    session = instance.session
    return {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": request_id,
        "execution_id": execution_id,
    }


def submit(instance, shown, *, receipt=True, accepted=True):
    bridge = instance._bridge_for_operations()
    bridge.receipt = receipt
    bridge.accepted = accepted
    return instance.submit_prepared(
        shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
    )


def assert_admission_blocked(instance):
    with pytest.raises(ClientError) as raised:
        preview(instance, request_id="blocked-request")
    assert raised.value.code == "EXECUTION_UNRESOLVED"


def test_a_missing_reference_query_is_read_only(client):
    bridge = client._bridge_for_operations()
    bridge.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}

    observed = client.get_execution(reference(client, "never-submitted"))

    assert observed["observation_unknown"] is True
    assert client._unresolved == {}
    assert client._submitted == {}
    assert preview(client, request_id="admission-open")["request_id"] == "admission-open"


@pytest.mark.parametrize(
    "execution",
    [
        {"found": True, "state": "RUNNING", "execution_id": "execution-9"},
        {"found": True, "state": "UNKNOWN", "reconciled": False, "execution_id": "execution-9"},
    ],
)
def test_a_foreign_nonterminal_reference_does_not_lock_this_client(client, execution):
    bridge = client._bridge_for_operations()
    bridge.execution = execution

    observed = client.get_execution(reference(client, "another-client-request"))

    assert observed["state"] in {"RUNNING", "UNKNOWN"}
    assert client._unresolved == {}
    assert client._submitted == {}
    preview(client, request_id="admission-open")


def test_a_real_submission_stays_locked_until_reconciled(client):
    shown = preview(client, request_id="real-submit")
    with pytest.raises(ClientError) as raised:
        submit(client, shown, receipt=False)
    assert raised.value.code == "OBSERVATION_UNKNOWN"

    ref = reference(client, shown["request_id"])
    assert list(client._unresolved) == [shown["request_id"]]
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}
    client.get_execution(ref)
    assert list(client._unresolved) == [shown["request_id"]]

    bridge.execution = {"found": True, "state": "UNKNOWN", "reconciled": False, "execution_id": "execution-1"}
    client.get_execution(ref)
    assert list(client._unresolved) == [shown["request_id"]]
    assert_admission_blocked(client)

    bridge.execution = {"found": True, "state": "UNKNOWN", "reconciled": True, "execution_id": "execution-1"}
    observed = client.get_execution(ref)

    assert observed["observation_unknown"] is False
    assert client._unresolved == {}
    assert client._submitted[shown["preview_id"]]["execution_id"] == "execution-1"
    preview(client, request_id="after-reconcile")


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELED"])
def test_a_terminal_observation_releases_a_real_submission(client, state):
    shown = preview(client, request_id=f"terminal-{state.lower()}")
    with pytest.raises(ClientError):
        submit(client, shown, receipt=False)
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.execution = {"found": True, "state": state, "execution_id": "execution-1"}
    observed = client.get_execution(reference(client, shown["request_id"]))

    assert observed["observation_unknown"] is False
    assert client._unresolved == {}
    preview(client, request_id=f"after-{state.lower()}")


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELED"])
def test_a_terminal_ledger_event_releases_an_adopted_reference(client, state):
    ref = reference(client, f"events-terminal-{state.lower()}")
    client.restore_ref(ref)
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [
            {"execution_id": "execution-1", "sequence": 1, "state": "RUNNING"},
            {"execution_id": "execution-1", "sequence": 2, "state": state},
        ],
        "last_event_sequence": 2,
        "has_more": False,
    }
    result = client.get_execution_events(ref)

    assert result["events"][-1]["state"] == state
    assert client._unresolved == {}
    preview(client, request_id=f"after-events-{state.lower()}")


def test_a_terminal_ledger_event_releases_a_real_submission(client):
    shown = preview(client, request_id="events-real-submit")
    with pytest.raises(ClientError):
        submit(client, shown, receipt=False)
    ref = reference(client, shown["request_id"])
    assert list(client._unresolved) == [shown["request_id"]]
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [
            {"execution_id": "execution-1", "sequence": 1, "state": "ACCEPTED"},
            {"execution_id": "execution-1", "sequence": 2, "state": "SUCCEEDED"},
        ],
        "last_event_sequence": 2,
        "has_more": False,
    }
    client.get_execution_events(ref)

    assert client._unresolved == {}
    assert client._submitted[shown["preview_id"]]["execution_id"] == "execution-1"
    preview(client, request_id="after-events-real-submit")


@pytest.mark.parametrize("state", ["ACCEPTED", "RUNNING"])
def test_nonterminal_ledger_events_keep_the_adopted_lock(client, state):
    ref = reference(client, f"events-{state.lower()}")
    client.restore_ref(ref)
    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [{"execution_id": "execution-1", "sequence": 1, "state": state}],
        "last_event_sequence": 1,
        "has_more": False,
    }

    client.get_execution_events(ref)

    assert client._unresolved == {ref["request_id"]: ref}
    assert_admission_blocked(client)


def test_conflicting_terminal_ledger_events_keep_the_lock(client):
    ref = reference(client, "events-conflict")
    client.restore_ref(ref)
    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [
            {"execution_id": "execution-1", "sequence": 1, "state": "SUCCEEDED"},
            {"execution_id": "execution-1", "sequence": 2, "state": "FAILED"},
        ],
        "last_event_sequence": 2,
        "has_more": False,
    }

    client.get_execution_events(ref)

    assert client._unresolved == {ref["request_id"]: ref}
    assert_admission_blocked(client)


def test_a_foreign_terminal_event_page_does_not_adopt(client):
    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-9",
        "events": [{"execution_id": "execution-9", "sequence": 1, "state": "SUCCEEDED"}],
        "last_event_sequence": 1,
        "has_more": False,
    }

    client.get_execution_events(reference(client, "never-submitted"))

    assert client._unresolved == {}
    assert client._submitted == {}
    preview(client, request_id="event-page-read-only")


def test_a_late_terminal_event_cannot_relock_a_rejected_submission(client):
    shown = preview(client, request_id="rejected-events")
    ref = reference(client, shown["request_id"])
    with pytest.raises(ClientError) as raised:
        submit(client, shown, accepted=False)
    assert raised.value.code == "GOAL_REJECTED"
    assert client._unresolved == {}
    assert client._submitted == {}

    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [{"execution_id": "execution-1", "sequence": 1, "state": "SUCCEEDED"}],
        "last_event_sequence": 1,
        "has_more": False,
    }
    client.get_execution_events(ref)

    assert client._unresolved == {}
    assert client._submitted == {}
    preview(client, request_id="after-rejected-events")


def test_a_terminal_event_for_another_execution_identity_keeps_the_lock(client):
    ref = reference(client, "events-other-execution", execution_id="execution-other")
    client.restore_ref(ref)
    bridge = client._bridge_for_operations()
    bridge.events = {
        "found": True,
        "execution_id": "execution-1",
        "events": [
            {"execution_id": "execution-1", "sequence": 1, "state": "SUCCEEDED"},
        ],
        "last_event_sequence": 1,
        "has_more": False,
    }

    client.get_execution_events(ref)

    assert client._unresolved == {ref["request_id"]: ref}
    assert_admission_blocked(client)


def test_rejection_clears_both_ledgers_and_a_late_poll_cannot_relock(client):
    bridge = client._bridge_for_operations()
    shown = preview(client, request_id="rejected-request")
    ref = reference(client, shown["request_id"])
    bridge.block_send = True
    bridge.block_get = True
    bridge.accepted = False
    outcomes = []

    def run_submit():
        try:
            client.submit_prepared(
                shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
            )
        except ClientError as exc:
            outcomes.append(exc)

    submitter = threading.Thread(target=run_submit, name="rejecting-submit", daemon=True)
    submitter.start()
    assert bridge.send_started.wait(5.0), "the submission never reached the send barrier"
    # The reference is recorded before the goal leaves the client.
    assert list(client._unresolved) == [shown["request_id"]]

    observer = threading.Thread(target=lambda: client.get_execution(ref), name="late-poll", daemon=True)
    observer.start()
    assert bridge.get_started.wait(5.0), "the observation never reached the Gateway barrier"

    bridge.send_release.set()
    submitter.join(5.0)
    assert not submitter.is_alive()
    assert [error.code for error in outcomes] == ["GOAL_REJECTED"]
    assert client._unresolved == {}
    assert client._submitted == {}

    bridge.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}
    bridge.get_release.set()
    observer.join(5.0)
    assert not observer.is_alive()
    assert client._unresolved == {}
    assert client._submitted == {}
    preview(client, request_id="after-rejection")


def test_restore_ref_is_the_explicit_recovery_path(client):
    ref = reference(client, "persisted-unknown", execution_id="execution-7")

    adopted = client.restore_ref(ref)

    assert adopted == ref
    assert client._unresolved == {ref["request_id"]: ref}
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.execution = {"found": True, "state": "UNKNOWN", "reconciled": False, "execution_id": "execution-7"}
    client.get_execution(ref)
    assert list(client._unresolved) == [ref["request_id"]]

    bridge.execution = {"found": True, "state": "UNKNOWN", "reconciled": True, "execution_id": "execution-7"}
    client.get_execution(ref)

    assert client._unresolved == {}
    preview(client, request_id="after-restore")


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("robot_id", "g2", "WRONG_TARGET"),
        ("target_key", "another-deployment", "WRONG_TARGET"),
        ("caller_scope", "somebody-else", "INVALID_REFERENCE"),
        ("legacy_task_id", "legacy-task", "INVALID_REFERENCE"),
    ],
)
def test_restore_ref_is_scoped_to_the_bound_deployment(client, field, value, code):
    ref = reference(client, "persisted")
    ref[field] = value

    with pytest.raises(ClientError) as raised:
        client.restore_ref(ref)

    assert raised.value.code == code
    assert client._unresolved == {}


def test_restoring_more_than_one_reference_keeps_each_real_lock(client):
    first = reference(client, "history-canceled", execution_id="execution-a")
    second = reference(client, "history-unknown", execution_id="execution-b")
    client.restore_ref(first)
    client.restore_ref(second)
    assert set(client._unresolved) == {first["request_id"], second["request_id"]}

    bridge = client._bridge_for_operations()
    bridge.execution = {"found": True, "state": "CANCELED", "execution_id": "execution-a"}
    client.get_execution(first)
    assert set(client._unresolved) == {second["request_id"]}
    assert_admission_blocked(client)

    bridge.execution = {"found": True, "state": "SUCCEEDED", "execution_id": "execution-b"}
    client.get_execution(second)
    assert client._unresolved == {}
    preview(client, request_id="after-history")


def test_rejection_clears_only_the_rejected_submission(client):
    first = preview(client, request_id="accepted-1")
    submit(client, first)
    assert_admission_blocked(client)

    bridge = client._bridge_for_operations()
    bridge.execution = {"found": True, "state": "SUCCEEDED", "execution_id": "execution-1"}
    client.get_execution(reference(client, first["request_id"]))
    assert client._unresolved == {}

    second = preview(client, request_id="rejected-2")
    with pytest.raises(ClientError) as raised:
        submit(client, second, accepted=False)
    assert raised.value.code == "GOAL_REJECTED"

    assert first["preview_id"] in client._submitted
    assert second["preview_id"] not in client._submitted
    assert client._unresolved == {}
    preview(client, request_id="after-rejection-2")


def test_rejection_keeps_an_adopted_history_reference_locked(client):
    bridge = client._bridge_for_operations()
    shown = preview(client, request_id="rejected-with-history")
    history = reference(client, "history-canceled", execution_id="execution-history")
    bridge.block_send = True
    bridge.accepted = False
    outcomes = []

    def run_submit():
        try:
            client.submit_prepared(
                shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
            )
        except ClientError as exc:
            outcomes.append(exc)

    submitter = threading.Thread(target=run_submit, name="rejecting-submit", daemon=True)
    submitter.start()
    assert bridge.send_started.wait(5.0), "the submission never reached the send barrier"
    client.restore_ref(history)
    assert set(client._unresolved) == {shown["request_id"], history["request_id"]}

    bridge.send_release.set()
    submitter.join(5.0)
    assert not submitter.is_alive()
    assert [error.code for error in outcomes] == ["GOAL_REJECTED"]
    assert set(client._unresolved) == {history["request_id"]}
    assert client._submitted == {}
    assert_admission_blocked(client)

    bridge.execution = {"found": True, "state": "CANCELED", "execution_id": "execution-history"}
    client.get_execution(history)
    assert client._unresolved == {}
    preview(client, request_id="after-history-rejection")
