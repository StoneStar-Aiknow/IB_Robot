"""Behaviour of the config-bound client: binding, identity, previews, recovery."""

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
        self.receipt = True
        self.accepted = True
        self.stopped = False
        self.execution_id = "execution-1"
        self.state = "RUNNING"
        self.epoch = "epoch-1"
        self.generation = 1
        self.capability_digest = "a" * 64
        self.robot_name = "g1"
        self.closed = False
        self.closed_count = 0
        self.start_error = ""
        self.started = 0

    def start(self):
        self.started += 1
        return True

    def close(self):
        self.closed = True
        self.closed_count += 1

    def get_status(self, **kwargs):
        return {
            "robot_name": self.robot_name,
            "registry_epoch": self.epoch,
            "registry_generation": self.generation,
            "registry_digest": "registry-digest",
            "capability_digest": self.capability_digest,
            "control_plane_ready": False,
        }

    def wait_for_program_server(self, **kwargs):
        return True

    def send_program_goal(self, **kwargs):
        self.goals.append(copy.deepcopy(kwargs))
        future = Future()
        future.set_result(SimpleNamespace(accepted=self.accepted))
        return future

    def wait_future(self, future, **kwargs):
        return self.receipt

    def get_execution(self, **kwargs):
        return {"found": True, "state": self.state, "execution_id": self.execution_id}

    def get_execution_events(self, **kwargs):
        return {"events": []}

    def cancel_execution(self, **kwargs):
        return {"accepted": True, "state": "STOPPING"}


class Controller:
    def __init__(self, bridge, **kwargs):
        self.bridge = bridge
        self.stopped = False

    def discover(self):
        pass

    def prepare_workflow(self, raw, steps, *, request_id):
        return {
            "steps": copy.deepcopy(steps),
            "registry_identity": {"registry_epoch": "epoch", "registry_generation": 1, "registry_digest": "digest"},
            "plan_id": "plan-1",
            "proposed_task_budget_sec": 30.0,
        }

    def confirm_plan(self):
        return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": self.stopped}


@pytest.fixture
def client(monkeypatch):
    from robot_client import workflow

    monkeypatch.setattr(workflow, "InteractiveController", Controller)
    instance = RobotClient(transport(), transport_factory=Bridge)
    yield instance
    instance.close()


def preview(instance):
    return instance.prepare_workflow(
        {
            "workflow_steps": [{"skill_name": "wave_hand", "schema_version": 1}],
            "raw_command": "wave",
            "request_id": "caller-request",
        }
    )


def test_session_is_bound_to_the_resolved_deployment(client):
    session = client.session
    assert session["robot_id"] == "g1"
    assert session["robot_name"] == "g1"
    assert session["caller_scope"] == "robot_skill_cli"
    assert session["connection_version"] == 1
    assert session["target_key"].endswith(client.transport.identity_digest())


def test_the_first_operation_starts_the_transport_exactly_once():
    instance = RobotClient(transport(), transport_factory=Bridge)
    try:
        # Binding is local-only: the transport is not built until an operation
        # needs it, so constructing a client never initializes ROS.
        assert instance._bridge is None
        assert instance.get_status()["robot_name"] == "g1"
        bridge = instance._bridge
        assert bridge.started == 1
        instance.get_status()
        assert instance._bridge_for_operations() is bridge
        assert bridge.started == 1
    finally:
        instance.close()


def test_bridge_receives_the_resolved_endpoints_without_a_catalog(client):
    bridge = client._bridge_for_operations()
    assert bridge.kwargs["status_service"] == "/embodied/get_skill_gateway_status"
    assert bridge.kwargs["execute_program_action"] == "/embodied/execute_program"
    assert bridge.kwargs["robot_id"] == "g1"


def test_status_does_not_need_a_local_catalog(client):
    assert client.get_status()["robot_name"] == "g1"


def test_status_rejects_a_gateway_for_another_robot(client, monkeypatch):
    class Other(Bridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.robot_name = "so101"

    other = RobotClient(transport(), transport_factory=Other)
    with pytest.raises(ClientError) as raised:
        other.get_status()
    assert raised.value.code == "TARGET_IDENTITY_CHANGED"


def test_presentation_and_persistence_precede_one_submission(client):
    events = []
    shown = preview(client)
    ref = client.submit_prepared(
        shown["preview_id"],
        presenter=lambda value: events.append(("presented", value["preview_id"])),
        persist_reference=lambda value: events.append(("persisted", value["target_key"])),
    )
    bridge = client._bridge_for_operations()
    assert events[0] == ("presented", shown["preview_id"])
    assert events[1] == ("persisted", ref["target_key"])
    assert len(bridge.goals) == 1
    assert (
        client.submit_prepared(
            shown["preview_id"],
            presenter=lambda value: events.append(("again", value)),
            persist_reference=lambda v: None,
        )
        == ref
    )
    assert len(bridge.goals) == 1


def test_reference_is_scoped_to_the_canonical_transport_identity(client):
    shown = preview(client)
    ref = client.submit_prepared(
        shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
    )
    assert ref["target_key"] == client.session["target_key"]
    assert ref["robot_id"] == "g1"


@pytest.mark.parametrize(
    "change",
    [
        {"epoch": "epoch-2"},
        {"generation": 2},
        {"capability_digest": "b" * 64},
    ],
)
def test_gateway_restart_or_catalog_change_invalidates_preview(client, change):
    shown = preview(client)
    bridge = client._bridge_for_operations()
    for key, value in change.items():
        setattr(bridge, key, value)
    presented = []
    with pytest.raises(ClientError) as raised:
        client.submit_prepared(shown["preview_id"], presenter=presented.append, persist_reference=lambda value: None)
    assert raised.value.code == "STALE_PREVIEW"
    assert presented == []
    assert bridge.goals == []


def test_failure_before_send_produces_no_goal(client, monkeypatch):
    class Stopping(Controller):
        def confirm_plan(self):
            return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": True}

    from robot_client import workflow

    monkeypatch.setattr(workflow, "InteractiveController", Stopping)
    shown = preview(client)
    with pytest.raises(ClientError) as raised:
        client.submit_prepared(shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None)
    assert raised.value.code == "STOPPED_BEFORE_EXECUTION"
    assert client._bridge_for_operations().goals == []


def test_receipt_loss_never_replays_and_keeps_the_deployment_locked(client):
    bridge = client._bridge_for_operations()
    bridge.receipt = False
    shown = preview(client)
    persisted = []
    with pytest.raises(ClientError) as raised:
        client.submit_prepared(shown["preview_id"], presenter=lambda value: None, persist_reference=persisted.append)
    assert raised.value.code == "OBSERVATION_UNKNOWN"
    assert raised.value.observation_unknown is True
    assert persisted and persisted[0]["target_key"] == client.session["target_key"]
    with pytest.raises(ClientError) as blocked:
        preview(client)
    assert blocked.value.code == "EXECUTION_UNRESOLVED"
    assert len(bridge.goals) == 1


def test_resolving_the_execution_releases_the_lock(client):
    bridge = client._bridge_for_operations()
    bridge.receipt = False
    shown = preview(client)
    with pytest.raises(ClientError):
        client.submit_prepared(shown["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None)
    ref = next(iter(client._unresolved.values()))
    bridge.state = "SUCCEEDED"
    observed = client.get_execution(ref)
    assert observed["observation_unknown"] is False
    assert client._unresolved == {}
    preview(client)


def test_reference_cannot_be_routed_to_another_deployment(client):
    other = RobotClient(transport(robot_id="g2", robot_name="g2"), transport_factory=Bridge)
    ref = client.submit_prepared(
        preview(client)["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
    )
    with pytest.raises(ClientError) as raised:
        other.get_execution(ref)
    assert raised.value.code == "WRONG_TARGET"


def test_reference_from_another_ros_scope_is_rejected(client, monkeypatch):
    ref = client.submit_prepared(
        preview(client)["preview_id"], presenter=lambda value: None, persist_reference=lambda value: None
    )
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    moved = RobotClient(transport(), transport_factory=Bridge)
    assert moved.session["target_key"] != ref["target_key"]
    with pytest.raises(ClientError) as raised:
        moved.get_execution(ref)
    assert raised.value.code == "WRONG_TARGET"


def test_identity_distinguishes_endpoints_and_robot_names():
    base = transport()
    other_endpoints = gateway_transport(
        {"validate_skill_service": "/lab/validate_skill"}, robot_id="g1", robot_name="g1"
    )
    other_robot = transport(robot_name="g2")
    assert base.identity_digest() != other_endpoints.identity_digest()
    assert base.identity_digest() != other_robot.identity_digest()
    assert base.identity_digest() == transport().identity_digest()


def test_a_transport_is_required():
    with pytest.raises(ClientError) as raised:
        RobotClient(object())
    assert raised.value.code == "INVALID_TRANSPORT"
