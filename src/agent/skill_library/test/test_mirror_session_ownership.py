"""A successful start action must not release the persistent session's arms."""

import json
from threading import RLock
from types import SimpleNamespace

import pytest
from std_msgs.msg import String

from skill_library.skill_executor_node import SkillExecutorNode


@pytest.fixture
def node():
    node = object.__new__(SkillExecutorNode)
    node._mirror_session_enabled = True
    node._mirror_session_state_lock = RLock()
    node._mirror_session_state = None
    node._mirror_session_start_latched = False
    node._mirror_session_seen_holding = False
    node._skill_templates = {
        "mirror_start": {"executor": "mirror_session", "binding": {"command": "start"}},
        "mirror_stop": {"executor": "mirror_session", "binding": {"command": "stop"}},
    }
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))
    node._runtime_idle_for_admission = lambda: False
    return node


def observe(node, *, state="WAIT", holding=False, releasing=False, stamp=100.0):
    node._on_mirror_session_state(
        String(data=json.dumps({"state": state, "holding": holding, "releasing": releasing, "stamp": stamp}))
    )


@pytest.mark.parametrize("skill", ["wave_hand", "pick_object", "return_to_task_pose", "__primitive__:moveit_planning"])
@pytest.mark.parametrize("state", ["STANDBY", "MIRRORING"])
def test_session_blocks_other_producers_even_when_runtime_mode_is_shared(node, skill, state):
    observe(node, state=state, holding=True)
    ready, reason = node._ensure_skill_control_mode(skill, None)
    assert not ready
    assert "mirror_stop" in reason


@pytest.mark.parametrize("skill", ["mirror_start", "mirror_stop"])
def test_session_commands_remain_available_without_a_state_heartbeat(node, skill):
    assert node._ensure_skill_control_mode(skill, None) == (True, "")


@pytest.mark.parametrize("stamp", [97.9, 101.0, float("nan"), True])
def test_stale_future_or_malformed_state_cannot_authorize_another_producer(node, stamp):
    observe(node, stamp=stamp)
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]


def test_unknown_state_and_releasing_wait_fail_closed(node):
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]
    observe(node, releasing=True)
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]
    node._on_mirror_session_state(String(data="[]"))
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]


def test_fresh_unheld_wait_preserves_existing_urs_gestures(node):
    observe(node)
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")


def test_old_wait_after_start_reply_cannot_release_the_start_latch(node):
    node._mirror_session_start_latched = True
    observe(node)
    node._runtime_idle_for_admission = lambda: True
    assert not node._ensure_skill_control_mode("pick_object", None)[0]


@pytest.mark.parametrize("skill", ["pick_object", "return_to_task_pose"])
def test_observed_session_release_requires_idle_before_allowing_grasp(node, skill):
    node._mirror_session_start_latched = True
    observe(node, state="STANDBY", holding=True)
    observe(node)
    assert not node._ensure_skill_control_mode(skill, None)[0]
    node._runtime_idle_for_admission = lambda: True
    assert node._ensure_skill_control_mode(skill, None) == (True, "")


def test_deployment_without_mirror_keeps_existing_behavior(node):
    node._mirror_session_enabled = False
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")


@pytest.mark.parametrize("reply_arrives", [True, False])
def test_start_latches_before_the_service_reply_and_timeout_cannot_free_the_arms(node, reply_arrives):
    observe(node)
    node._rpc_timeout = 0.1
    node._set_result_catalog_identity = lambda _result: None
    node._wait_for_future = lambda *_args, **_kwargs: reply_arrives
    response = SimpleNamespace(success=True, message="started")

    def dispatch(request):
        assert request.data
        assert not node._ensure_skill_control_mode("pick_object", None)[0]
        return SimpleNamespace(result=lambda: response)

    node._mirror_session_client = SimpleNamespace(wait_for_service=lambda **_kwargs: True, call_async=dispatch)
    handle = SimpleNamespace(
        request=SimpleNamespace(skill_name="mirror_start", timeout_sec=5.0),
        is_cancel_requested=False,
        succeed=lambda: None,
        abort=lambda: None,
    )
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_start"])
    assert result.success is reply_arrives
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]


class StopHandle:
    def __init__(self):
        self.request = SimpleNamespace(skill_name="mirror_stop", timeout_sec=0.2)
        self.is_cancel_requested = False
        self.terminal = None

    def succeed(self):
        self.terminal = "succeeded"

    def abort(self):
        self.terminal = "aborted"

    def canceled(self):
        self.terminal = "canceled"


def prepare_stop(node, *, success=True):
    node._mirror_session_start_latched = True
    node._rpc_timeout = 0.1
    node._set_result_catalog_identity = lambda _result: None
    node._wait_for_future = lambda *_args, **_kwargs: True
    node._runtime_idle_for_admission = lambda: True
    observe(node, state="MIRRORING", holding=True)
    calls = []

    def dispatch(request):
        calls.append(request.data)
        return SimpleNamespace(result=lambda: SimpleNamespace(success=success, message="stop reply"))

    node._mirror_session_client = SimpleNamespace(wait_for_service=lambda **_kwargs: True, call_async=dispatch)
    return StopHandle(), calls


def test_stop_waits_for_delayed_release_before_authorizing_wave(node, monkeypatch):
    handle, calls = prepare_stop(node)
    observations = []

    def release(_duration):
        observations.append(node._ensure_skill_control_mode("wave_hand", None)[0])
        observe(node)

    monkeypatch.setattr("skill_library.skill_executor_node.time.sleep", release)
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_stop"])
    assert result.success and handle.terminal == "succeeded"
    assert calls == [False] and observations == [False]
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")


@pytest.mark.parametrize("failure", ["rejected", "timeout", "cancel"])
def test_stop_failure_never_authorizes_following_wave(node, monkeypatch, failure):
    handle, calls = prepare_stop(node, success=failure != "rejected")
    if failure == "cancel":
        monkeypatch.setattr(
            "skill_library.skill_executor_node.time.sleep",
            lambda _duration: setattr(handle, "is_cancel_requested", True),
        )
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_stop"])
    assert not result.success and calls == [False]
    assert handle.terminal == ("canceled" if failure == "cancel" else "aborted")
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]
    if failure == "timeout":
        assert result.execution_detail.reason_code == "MIRROR_SESSION_RELEASE_UNCONFIRMED"


@pytest.mark.parametrize("stamp", [99.9, 101.0])
def test_stop_does_not_accept_pre_stop_or_future_wait(node, stamp):
    handle, _calls = prepare_stop(node)
    observe(node, stamp=stamp)
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_stop"])
    assert not result.success
    assert result.execution_detail.reason_code == "MIRROR_SESSION_RELEASE_UNCONFIRMED"


@pytest.mark.parametrize("left_idle", [False, True])
@pytest.mark.parametrize("cancel_at", ["entry", "discovery", "reply"])
def test_canceled_start_returns_only_its_own_mode_without_creating_an_undispatched_latch(node, left_idle, cancel_at):
    mode = {"active": "policy_stream"}
    changes, calls = [], []
    handle = StopHandle()
    handle.request.skill_name = "mirror_start"
    handle.is_cancel_requested = cancel_at == "entry"
    node._rpc_timeout = 0.1
    node._set_result_catalog_identity = lambda _result: None
    node._required_control_mode_for_skill = lambda _skill: "model_inference"
    node._runtime_mode_for = lambda _mode: "policy_stream"
    node._motion_mode_switch_lock = RLock()
    node._runtime_status_for_admission = lambda: (SimpleNamespace(active_mode=mode["active"]), "")
    node._runtime_idle_for_admission = lambda: mode["active"] == "idle"
    if left_idle:
        observe(node)
    else:
        node._mirror_session_start_latched = True
        observe(node, state="MIRRORING", holding=True)

    def change(requested):
        changes.append(requested)
        mode["active"] = requested
        return True, ""

    def available(**_kwargs):
        if cancel_at == "discovery":
            handle.is_cancel_requested = True
        return True

    def dispatch(request):
        calls.append(request.data)
        return SimpleNamespace(result=lambda: SimpleNamespace(success=True, message="started"))

    def reply(*_args, **_kwargs):
        handle.is_cancel_requested = True
        return True

    node._request_runtime_mode = change
    node._mirror_session_client = SimpleNamespace(wait_for_service=available, call_async=dispatch)
    node._wait_for_future = reply
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_start"], left_idle=left_idle)
    assert not result.success and handle.terminal == "canceled"
    assert result.error_code == "SKILL_CANCELLED"
    assert changes == (["idle"] if left_idle else [])
    assert mode["active"] == ("idle" if left_idle else "policy_stream")
    assert calls == ([True] if cancel_at == "reply" else [])
    if cancel_at != "reply":
        assert node._mirror_session_start_latched is (not left_idle)
        assert node._ensure_skill_control_mode("wave_hand", None)[0] is left_idle


def test_expired_undispatched_start_returns_its_mode_without_latching(node, monkeypatch):
    clock = {"now": 0.0}
    calls, returned = [], []
    handle = StopHandle()
    handle.request.skill_name = "mirror_start"
    node._rpc_timeout = 0.1
    node._set_result_catalog_identity = lambda _result: None
    observe(node)

    def available(**_kwargs):
        clock["now"] = 0.3
        return True

    node._mirror_session_client = SimpleNamespace(wait_for_service=available, call_async=calls.append)
    node._give_back_mirror_mode = lambda name: returned.append(name) or "runtime returned to idle"
    monkeypatch.setattr("skill_library.skill_executor_node.time.monotonic", lambda: clock["now"])
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_start"], left_idle=True)
    assert not result.success and handle.terminal == "aborted"
    assert result.error_code == "MIRROR_SESSION_TIMEOUT"
    assert calls == [] and returned == ["mirror_start"]
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")


@pytest.mark.parametrize("released", [False, True])
def test_discovery_rpc_and_release_share_one_stop_deadline(node, monkeypatch, released):
    handle, calls = prepare_stop(node)
    handle.request.timeout_sec = 1.0
    node._rpc_timeout = 0.3
    clock = {"now": 0.0}
    waits = []

    def available(**_kwargs):
        clock["now"] += 0.2
        return True

    def reply(_future, timeout, **_kwargs):
        waits.append(timeout)
        clock["now"] += 0.7
        return True

    def pause(duration):
        clock["now"] += duration
        if released and clock["now"] >= 0.95:
            observe(node)

    node._mirror_session_client.wait_for_service = available
    node._wait_for_future = reply
    monkeypatch.setattr("skill_library.skill_executor_node.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("skill_library.skill_executor_node.time.sleep", pause)
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_stop"])
    assert calls == [False] and waits == [pytest.approx(0.8)]
    assert clock["now"] <= 1.0
    assert result.success is released
    assert node._ensure_skill_control_mode("wave_hand", None)[0] is released


def test_stop_from_idle_with_unconfirmed_session_calls_stop_without_taking_control(node):
    handle, calls = prepare_stop(node)
    node._runtime_status_for_admission = lambda: (SimpleNamespace(active_mode="idle"), "")

    def forbidden(*_args, **_kwargs):
        pytest.fail("mirror_stop must not switch into the producer's mode")

    def dispatch(request):
        calls.append(request.data)
        observe(node)
        return SimpleNamespace(result=lambda: SimpleNamespace(success=True, message="released"))

    node._ensure_skill_control_mode = forbidden
    node._mirror_session_client.call_async = dispatch
    result = node._execute_skill_child(handle, validation_done=True)
    assert result.success and handle.terminal == "succeeded"
    assert calls == [False]


def test_already_confirmed_stop_does_not_call_the_session_or_switch_mode(node):
    handle = StopHandle()
    node._set_result_catalog_identity = lambda _result: None
    node._runtime_idle_for_admission = lambda: True
    observe(node)
    result = node._execute_skill_child(handle, validation_done=True)
    assert result.success and handle.terminal == "succeeded"
    assert result.executed_primitives == []


def test_fresh_idle_wait_does_not_skip_stop_after_an_unconfirmed_start(node):
    handle, calls = prepare_stop(node)
    observe(node)
    node._mirror_session_seen_holding = False
    node._runtime_status_for_admission = lambda: (SimpleNamespace(active_mode="idle"), "")
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]

    def dispatch(request):
        calls.append(request.data)
        observe(node)
        return SimpleNamespace(result=lambda: SimpleNamespace(success=True, message="released"))

    node._mirror_session_client.call_async = dispatch
    result = node._execute_skill_child(handle, validation_done=True)
    assert result.success and calls == [False]
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")


@pytest.mark.parametrize(
    "reason", ["MIRROR_SESSION_BUSY", "MIRROR_SESSION_STATE_UNKNOWN", "MIRROR_SESSION_RELEASE_UNCONFIRMED"]
)
def test_child_admission_reports_mirror_reason_before_starting_grasp(node, reason):
    handle = StopHandle()
    handle.request.skill_name = "pick_object"
    node._set_result_catalog_identity = lambda _result: None
    if reason == "MIRROR_SESSION_BUSY":
        observe(node, state="MIRRORING", holding=True)
    elif reason == "MIRROR_SESSION_RELEASE_UNCONFIRMED":
        observe(node)
        node._mirror_session_start_latched = True
    result = node._execute_skill_child(handle, validation_done=True)
    assert not result.success and handle.terminal == "aborted"
    assert result.execution_detail.reason_code == reason
    assert result.execution_detail.recovery_state == "NOT_NEEDED"
    assert result.executed_primitives == []
