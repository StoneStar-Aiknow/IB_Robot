"""Exercise the asynchronous adapter with no ROS graph or robot connection."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

import pytest

from manipulation_execution.mirror_session import preset_ros
from manipulation_execution.mirror_session.endpoints import SessionBindingError
from manipulation_execution.mirror_session.hooks import UpperBodyCommand
from manipulation_execution.mirror_session.preset import PresetEpoch


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


class Message:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Publisher:
    def __init__(self):
        self.messages = []
        self.error = None

    def publish(self, message):
        if self.error is not None:
            raise self.error
        self.messages.append(json.loads(message.data))


class Node:
    def __init__(self):
        self.publisher = Publisher()
        self.bindings = []

    def create_publisher(self, message_type, endpoint, qos):
        self.bindings.append(("publisher", message_type, endpoint, qos))
        return self.publisher

    def create_subscription(self, message_type, endpoint, callback, qos, **kwargs):
        self.bindings.append(("subscription", message_type, endpoint, qos, kwargs))
        self.ack_callback = callback
        return callback


class ActionClient:
    def __init__(self, node, action_type, endpoint, **kwargs):
        node.client = self
        node.bindings.append(("action", action_type, endpoint, kwargs))
        self.calls = []
        self.error = None

    def send_goal_async(self, goal, *, feedback_callback, goal_uuid):
        if self.error is not None:
            raise self.error
        response = Future()
        self.calls.append(SimpleNamespace(goal=goal, feedback=feedback_callback, uuid=goal_uuid, response=response))
        return response


class Handle:
    def __init__(self, identifier, accepted=True):
        self.accepted = accepted
        self.goal_id = identifier
        self.result = Future()
        self.cancellation = Future()
        self.cancel_calls = 0
        self.cancel_error = None
        self.result_error = None

    def get_result_async(self):
        if self.result_error:
            raise self.result_error
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        if self.cancel_error:
            raise self.cancel_error
        return self.cancellation


def description():
    return {
        "interfaces": {
            "motion.named": {
                "kind": "action",
                "direction": "serve",
                "message_type": "ibrobot_msgs/action/ExecuteNamedMotion",
                "endpoint": "/isolated/motion",
            },
            "motion.named_hold_lease": {
                "kind": "topic",
                "direction": "subscribe",
                "message_type": "std_msgs/msg/String",
                "endpoint": "/isolated/lease",
                "qos": {"depth": 3},
            },
            "motion.named_hold_lease_ack": {
                "kind": "topic",
                "direction": "publish",
                "message_type": "std_msgs/msg/String",
                "endpoint": "/isolated/ack",
                "qos": {"depth": 7},
            },
        }
    }


@pytest.fixture
def ros(monkeypatch):
    result_type = SimpleNamespace(NONE=0, CANCELLED=9, INTERNAL_ERROR=255)
    types = SimpleNamespace(
        ActionClient=ActionClient,
        ExecuteNamedMotion=SimpleNamespace(Goal=Message, Result=result_type),
        GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6),
        String=Message,
        UUID=Message,
        qos=lambda value: ("qos", value) if value else None,
    )
    monkeypatch.setattr(preset_ros, "_ros_dependencies", lambda: types)
    return types


@pytest.fixture
def rig(ros):
    node, clock = Node(), Clock()
    port = preset_ros.RosPresetPort(node, description(), callback_group="callbacks", clock=clock)
    return SimpleNamespace(node=node, clock=clock, port=port, ros=ros, epoch=PresetEpoch(7, 1))


def begin(rig):
    rig.port.begin("overhead_heart", rig.epoch)
    return rig.node.publisher.messages[-1]


def ack(rig, payload=None):
    payload = rig.node.publisher.messages[-1] if payload is None else payload
    rig.node.ack_callback(Message(data=json.dumps(payload)))


def accept(rig, *, accepted=True):
    if not rig.node.client.calls:
        ack(rig)
    call = rig.node.client.calls[-1]
    handle = Handle(call.uuid, accepted=accepted)
    call.response.set_result(handle)
    return handle


def command(value=0.1):
    return UpperBodyCommand((value,) * 14, (value + 0.2,) * 20)


def digest(target):
    values = list(map(float, target.arm + target.hand))
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finish(handle, *, status=4, success=True, code=0, motion="overhead_heart", target=None, message=None):
    if message is None:
        message = json.dumps({"motion": motion, "exit_target_sha256": digest(target) if target is not None else ""})
    handle.result.set_result(
        SimpleNamespace(status=status, result=SimpleNamespace(success=success, error_code=code, message=message))
    )


def test_constructor_resolves_every_endpoint_and_qos_without_waiting(rig):
    bindings = rig.node.bindings
    assert bindings[0][2:] == ("/isolated/lease", ("qos", {"depth": 3}))
    assert bindings[1][2:] == ("/isolated/motion", {"callback_group": "callbacks"})
    assert bindings[2][2:] == ("/isolated/ack", ("qos", {"depth": 7}), {"callback_group": "callbacks"})
    assert rig.port.snapshot().epoch is None
    assert rig.node.client.calls == []
    assert rig.node.publisher.messages == []


@pytest.mark.parametrize("empty_qos", ["missing", "empty_mapping"])
def test_constructor_uses_ros_depth_default_when_topic_qos_parser_returns_none(ros, empty_qos):
    value = description()
    for interface in ("motion.named_hold_lease", "motion.named_hold_lease_ack"):
        if empty_qos == "missing":
            value["interfaces"][interface].pop("qos")
        else:
            value["interfaces"][interface]["qos"] = {}
    node = Node()
    preset_ros.RosPresetPort(node, value)
    assert node.bindings[0][3] == 10
    assert node.bindings[2][3] == 10


@pytest.mark.parametrize("interface", ["motion.named", "motion.named_hold_lease", "motion.named_hold_lease_ack"])
@pytest.mark.parametrize("defect", ["missing", "kind", "direction", "message_type", "relative", "qos"])
def test_missing_or_mismatched_bindings_fail_before_loading_ros(monkeypatch, interface, defect):
    value = description()
    if defect == "missing":
        value["interfaces"].pop(interface)
    elif defect == "relative":
        value["interfaces"][interface]["endpoint"] = "relative/topic"
    else:
        value["interfaces"][interface][defect] = [] if defect == "qos" else "wrong"
    monkeypatch.setattr(preset_ros, "_ros_dependencies", lambda: pytest.fail("invalid bindings must not load ROS"))
    node = Node()
    with pytest.raises(SessionBindingError):
        preset_ros.RosPresetPort(node, value)
    assert node.bindings == []


@pytest.mark.parametrize("endpoint", ["/", "/x/", "//x", "/x//y", "/has space", "/x/{id}", "/9x", "/x\n"])
def test_nonabsolute_or_noncanonical_endpoint_is_rejected(ros, endpoint):
    value = description()
    value["interfaces"]["motion.named"]["endpoint"] = endpoint
    with pytest.raises(SessionBindingError):
        preset_ros.RosPresetPort(Node(), value)


def test_begin_publishes_uuid_lease_before_same_ack_dispatch_and_is_single_flight(rig):
    lease = begin(rig)
    state = rig.port.snapshot()
    assert set(lease) == {"goal_id", "motion"}
    assert len(lease["goal_id"]) == 32
    assert state.epoch == rig.epoch and state.pending and not state.accepted
    assert state.goal_id == lease["goal_id"]
    assert not rig.node.client.calls
    for payload in ({}, {**lease, "goal_id": "0" * 32}, {**lease, "motion": "different"}, []):
        ack(rig, payload)
    for raw in ("{", "null", "x" * 2049):
        rig.node.ack_callback(Message(data=raw))
    assert not rig.node.client.calls
    ack(rig, lease)
    ack(rig, lease)
    assert len(rig.node.client.calls) == 1
    call = rig.node.client.calls[0]
    assert bytes(call.uuid.uuid).hex() == lease["goal_id"]
    assert call.goal.__dict__ == {"name": "overhead_heart", "target": "both", "interrupt": False, "resume_token": ""}
    with pytest.raises(RuntimeError):
        rig.port.begin("overhead_heart", PresetEpoch(8, 2))


def test_cancel_before_ack_prevents_dispatch_and_refresh(rig):
    lease = begin(rig)
    rig.port.cancel("stop")
    ack(rig, lease)
    rig.port.refresh(command())
    state = rig.port.snapshot()
    assert not rig.node.client.calls
    assert len(rig.node.publisher.messages) == 1
    assert state.terminal and state.release_confirmed and not state.accepted and not state.pending


def test_pending_acceptance_is_cancelled_when_handle_arrives_and_cancel_ack_is_not_release(rig):
    begin(rig)
    ack(rig)
    rig.port.cancel("generation_changed")
    handle = accept(rig)
    assert handle.cancel_calls == 1
    state = rig.port.snapshot()
    assert state.accepted and not state.pending and not state.terminal and not state.release_confirmed
    handle.cancellation.set_result(SimpleNamespace(goals_canceling=[handle.goal_id]))
    assert not rig.port.snapshot().release_confirmed
    finish(handle, status=5, success=False, code=9)
    assert rig.port.snapshot().release_confirmed
    assert not rig.port.snapshot().success


def test_rejected_goal_proves_no_claim_and_allows_new_epoch(rig):
    begin(rig)
    accept(rig, accepted=False)
    state = rig.port.snapshot()
    assert state.terminal and state.release_confirmed and not state.accepted
    rig.port.begin("overhead_heart", PresetEpoch(8, 2))
    assert rig.port.snapshot().epoch == PresetEpoch(8, 2)


def test_feedback_uses_receipt_clock_and_snapshot_refresh_and_duplicate_ack_do_not_refresh_it(rig):
    begin(rig)
    accept(rig)
    call = rig.node.client.calls[-1]
    rig.clock.now = 12.5
    call.feedback(Message(goal_id=call.uuid, feedback=Message(state="holding", progress=1.0)))
    assert rig.port.snapshot().stage == "holding"
    assert rig.port.snapshot().observed_at == 12.5
    rig.clock.now = 15.0
    rig.port.refresh()
    ack(rig)
    assert rig.port.snapshot().observed_at == 12.5
    call.feedback(Message(goal_id=Message(uuid=[0] * 16), feedback=Message(state="exiting")))
    assert rig.port.snapshot().observed_at == 12.5


def test_success_returns_exact_older_submitted_exit_not_latest_and_requires_server_hash(rig):
    lease = begin(rig)
    handle = accept(rig)
    first, latest = command(0.1), command(0.4)
    rig.port.refresh(first)
    rig.port.refresh(latest)
    assert rig.node.publisher.messages[-1] == {
        **lease,
        "exit_target": list(latest.arm + latest.hand),
    }
    rig.clock.now = 20.0
    finish(handle, target=first)
    state = rig.port.snapshot()
    assert state.success and state.terminal and state.accepted and state.release_confirmed
    assert state.final_target_verified and state.final_target == first and state.final_target != latest
    assert state.final_target_at == state.observed_at == 20.0


@pytest.mark.parametrize(
    "result",
    [
        {"status": 6},
        {"status": 0},
        {"success": False},
        {"code": 255},
        {"motion": "wrong"},
        {"target": command(0.9)},
        {"message": "{}"},
        {"message": "not-json"},
        {"message": "null"},
        {"message": json.dumps({"motion": "overhead_heart", "exit_target_sha256": []})},
    ],
)
def test_unverified_or_failed_result_never_proves_release_or_resume(rig, result):
    begin(rig)
    handle = accept(rig)
    target = command()
    rig.port.refresh(target)
    values = {"target": target, **deepcopy(result)}
    finish(handle, **values)
    state = rig.port.snapshot()
    assert not state.success and not state.release_confirmed and not state.final_target_verified
    assert state.final_target is None and state.final_target_at is None and state.fault
    with pytest.raises(RuntimeError):
        rig.port.begin("overhead_heart", PresetEpoch(8, 2))


@pytest.mark.parametrize("status,success,code", [(4, False, 9), (5, True, 9), (5, False, 0), (6, False, 9)])
def test_only_explicit_cancelled_terminal_result_proves_cancel_release(rig, status, success, code):
    begin(rig)
    handle = accept(rig)
    rig.port.cancel("stop")
    finish(handle, status=status, success=success, code=code)
    assert not rig.port.snapshot().release_confirmed


def test_target_evidence_is_bounded_and_evicted_hash_cannot_be_used(rig):
    begin(rig)
    handle = accept(rig)
    targets = [command(value / 100) for value in range(9)]
    for target in targets:
        rig.port.refresh(target)
    assert len(rig.port._flight.submitted) == 8
    finish(handle, target=targets[0])
    assert not rig.port.snapshot().final_target_verified
    assert not rig.port.snapshot().release_confirmed


def test_same_target_can_be_resubmitted_without_growing_evidence(rig):
    begin(rig)
    handle = accept(rig)
    for _ in range(30):
        rig.port.refresh(command())
    assert len(rig.port._flight.submitted) == 1
    finish(handle, target=command())
    assert rig.port.snapshot().final_target_verified


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.1"])
def test_invalid_exit_target_is_not_published_or_recorded(rig, value):
    begin(rig)
    with pytest.raises(ValueError):
        rig.port.refresh(UpperBodyCommand((value,) * 14, (0.0,) * 20))
    assert len(rig.node.publisher.messages) == 1
    assert not rig.port._flight.submitted


@pytest.mark.parametrize("target", [UpperBodyCommand((0.0,) * 13, (0.0,) * 20), None, []])
def test_bad_target_shape_is_rejected(rig, target):
    begin(rig)
    if target is None:
        rig.port.refresh(target)
        assert len(rig.node.publisher.messages) == 2
    else:
        with pytest.raises(ValueError):
            rig.port.refresh(target)


def test_old_epoch_callbacks_cannot_overwrite_new_snapshot_and_late_handle_is_cancelled(rig):
    old_lease = begin(rig)
    ack(rig)
    old_call = rig.node.client.calls[-1]
    old_flight = rig.port._flight
    old_call.response.set_result(Handle(old_call.uuid, accepted=False))
    rig.port.begin("overhead_heart", PresetEpoch(8, 2))
    new_state = rig.port.snapshot()
    ack(rig, old_lease)
    old_call.feedback(Message(goal_id=old_call.uuid, feedback=Message(state="holding")))
    late = Handle(old_call.uuid)
    done = Future()
    done.set_result(late)
    rig.port._on_goal_response(old_flight, done)
    assert late.cancel_calls == 1
    assert rig.port.snapshot() == new_state
    late.cancellation.set_result(SimpleNamespace(goals_canceling=[late.goal_id]))
    old_result = Future()
    old_result.set_result(SimpleNamespace(status=5, result=SimpleNamespace(success=False, error_code=9, message="")))
    rig.port._on_result(old_flight, old_result)
    assert rig.port.snapshot() == new_state


@pytest.mark.parametrize("operation", ["lease", "send", "response", "none_response", "result", "none_result", "cancel"])
def test_transport_failure_or_missing_response_never_becomes_release_proof(rig, operation):
    if operation == "lease":
        rig.node.publisher.error = RuntimeError("delivery uncertain")
    rig.port.begin("overhead_heart", rig.epoch)
    if operation == "send":
        rig.node.client.error = RuntimeError("delivery uncertain")
    if operation != "lease":
        ack(rig)
    if operation == "response":
        rig.node.client.calls[-1].response.set_exception(RuntimeError("response lost"))
    elif operation == "none_response":
        rig.node.client.calls[-1].response.set_result(None)
    elif operation in {"result", "none_result", "cancel"}:
        handle = accept(rig)
        if operation == "result":
            handle.result.set_exception(RuntimeError("result lost"))
        elif operation == "none_result":
            handle.result.set_result(None)
        else:
            handle.cancel_error = RuntimeError("cancel delivery uncertain")
            rig.port.cancel("stop")
    state = rig.port.snapshot()
    assert state.fault and not state.success and not state.release_confirmed
    rig.port.cancel("stop_after_fault")
    assert not rig.port.snapshot().release_confirmed
    with pytest.raises(RuntimeError):
        rig.port.begin("overhead_heart", PresetEpoch(8, 2))


def test_wrong_accepted_uuid_is_cancelled_without_release_proof(rig):
    begin(rig)
    ack(rig)
    handle = Handle(Message(uuid=[0] * 16))
    rig.node.client.calls[-1].response.set_result(handle)
    assert handle.cancel_calls == 1
    state = rig.port.snapshot()
    assert state.fault and not state.release_confirmed and state.accepted
