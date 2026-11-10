"""Edge cases of the config-bound client: rejection, transport failures, cleanup."""

from __future__ import annotations

import copy
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transport import gateway_transport


def transport(**overrides):
    values = {"robot_id": "g1", "robot_name": "g1"}
    values.update(overrides)
    return gateway_transport({}, **values)


class Bridge:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.goals = []
        self.program_ready = True
        self.accepted = True
        self.closed = 0
        self.canceled = []
        self.start_error = ""
        self.started = 0

    def start(self):
        self.started += 1
        return True

    def close(self):
        self.closed += 1

    def get_status(self, **kwargs):
        return {
            "robot_name": "g1",
            "registry_epoch": "epoch-1",
            "registry_generation": 1,
            "registry_digest": "registry-digest",
            "capability_digest": "a" * 64,
            "control_plane_ready": True,
        }

    def wait_for_program_server(self, **kwargs):
        return self.program_ready

    def send_program_goal(self, **kwargs):
        self.goals.append(copy.deepcopy(kwargs))
        future = Future()
        future.set_result(SimpleNamespace(accepted=self.accepted))
        return future

    def wait_future(self, future, **kwargs):
        return True

    def get_execution(self, **kwargs):
        return {"found": True, "state": "ACTIVE", "execution_id": ""}

    def get_execution_events(self, **kwargs):
        return {"events": []}

    def cancel_execution(self, **kwargs):
        self.canceled.append(copy.deepcopy(kwargs))
        return {"accepted": True, "state": "STOPPING"}


class Controller:
    def __init__(self, bridge, **kwargs):
        self.bridge = bridge

    def load_catalog(self):
        pass

    def prepare_workflow(self, raw, steps, *, request_id):
        raise ClientError("INVALID_REQUEST", "planner rejected the request")

    def confirm_plan(self):
        raise AssertionError("must not confirm a rejected plan")


@pytest.fixture
def client(monkeypatch):
    from robot_client import workflow

    monkeypatch.setattr(workflow, "InteractiveController", Controller)
    instance = RobotClient(transport(), transport_factory=Bridge)
    yield instance
    instance.close()


def test_planner_rejection_leaves_no_preview(client):
    with pytest.raises(ClientError) as raised:
        client.prepare_workflow({"workflow_steps": [{"skill_name": "wave"}], "raw_command": "wave"})
    assert raised.value.code == "INVALID_REQUEST"
    assert client._previews == {}


def test_submit_requires_a_known_preview(client):
    with pytest.raises(ClientError) as raised:
        client.submit_prepared("missing", presenter=lambda value: None, persist_reference=lambda value: None)
    assert raised.value.code == "STALE_PREVIEW"


def test_reference_validation_rejects_unknown_or_unbounded_fields(client):
    base = {
        "request_id": "r1",
        "target_key": client.session["target_key"],
        "robot_id": "g1",
        "caller_scope": "robot_skill_cli",
    }
    with pytest.raises(ClientError) as raised:
        client.cancel_execution({**base, "unexpected": "x"})
    assert raised.value.code == "INVALID_REFERENCE"
    with pytest.raises(ClientError) as raised:
        client.cancel_execution({**base, "request_id": "x" * 257})
    assert raised.value.code == "INVALID_REFERENCE"
    with pytest.raises(ClientError) as raised:
        client.cancel_execution({**base, "execution_id": 5})
    assert raised.value.code == "INVALID_REFERENCE"


def test_cancel_is_forwarded_for_the_bound_deployment(client):
    ref = {
        "request_id": "r1",
        "target_key": client.session["target_key"],
        "robot_id": "g1",
        "caller_scope": "robot_skill_cli",
    }
    assert client.cancel_execution(ref)["accepted"] is True
    bridge = client._bridge_for_operations()
    assert bridge.canceled[0]["request_id"] == "r1"
    assert bridge.canceled[0]["execution_id"] == ""


def test_close_is_idempotent_and_rebuilds_on_demand(client):
    bridge = client._bridge_for_operations()
    client.close()
    client.close()
    assert bridge.closed == 1
    client._bridge_for_operations()
    assert client._bridge is not None


def test_missing_program_action_is_reported_before_display(monkeypatch):
    from robot_client import workflow

    class Preparing(Controller):
        def prepare_workflow(self, raw, steps, *, request_id):
            return {
                "steps": copy.deepcopy(steps),
                "registry_identity": {"registry_epoch": "epoch", "registry_generation": 1, "registry_digest": "d"},
                "plan_id": "plan-1",
                "proposed_task_budget_sec": 30.0,
            }

        def confirm_plan(self):
            return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": False}

    monkeypatch.setattr(workflow, "InteractiveController", Preparing)
    instance = RobotClient(transport(), transport_factory=Bridge)
    shown = instance.prepare_workflow({"workflow_steps": [{"skill_name": "wave"}], "raw_command": "wave"})
    instance._bridge_for_operations().program_ready = False
    presented = []
    with pytest.raises(ClientError) as raised:
        instance.submit_prepared(shown["preview_id"], presenter=presented.append, persist_reference=lambda value: None)
    assert raised.value.code == "SERVER_UNAVAILABLE"
    assert presented
    assert instance._bridge_for_operations().goals == []


def test_rejected_goal_releases_the_target_and_reports_the_reference(monkeypatch):
    from robot_client import workflow

    class Preparing(Controller):
        def prepare_workflow(self, raw, steps, *, request_id):
            return {
                "steps": copy.deepcopy(steps),
                "registry_identity": {"registry_epoch": "epoch", "registry_generation": 1, "registry_digest": "d"},
                "plan_id": "plan-1",
                "proposed_task_budget_sec": 30.0,
            }

        def confirm_plan(self):
            return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": False}

    monkeypatch.setattr(workflow, "InteractiveController", Preparing)
    instance = RobotClient(transport(), transport_factory=Bridge)
    instance._bridge_for_operations().accepted = False
    shown = instance.prepare_workflow({"workflow_steps": [{"skill_name": "wave"}], "raw_command": "wave"})
    persisted = []
    with pytest.raises(ClientError) as raised:
        instance.submit_prepared(shown["preview_id"], presenter=lambda value: None, persist_reference=persisted.append)
    assert raised.value.code == "GOAL_REJECTED"
    assert raised.value.reference["target_key"] == instance.session["target_key"]
    assert persisted[0]["target_key"] == instance.session["target_key"]
    assert instance._unresolved == {}
