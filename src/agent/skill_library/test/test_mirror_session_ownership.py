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


def test_successful_stop_clears_takeover_uncertainty_only_after_observed_idle(node):
    node._mirror_session_start_latched = True
    node._rpc_timeout = 0.1
    node._set_result_catalog_identity = lambda _result: None
    node._wait_for_future = lambda *_args, **_kwargs: True
    node._runtime_idle_for_admission = lambda: True

    def dispatch(request):
        assert not request.data
        return SimpleNamespace(result=lambda: SimpleNamespace(success=True, message="released"))

    node._mirror_session_client = SimpleNamespace(wait_for_service=lambda **_kwargs: True, call_async=dispatch)
    handle = SimpleNamespace(request=SimpleNamespace(skill_name="mirror_stop", timeout_sec=5.0), succeed=lambda: None)
    result = node._execute_mirror_session_skill(handle, node._skill_templates["mirror_stop"])
    assert result.success
    # Successful stop does not replace a fresh, unheld WAIT heartbeat.
    assert not node._ensure_skill_control_mode("wave_hand", None)[0]
    observe(node)
    assert node._ensure_skill_control_mode("wave_hand", None) == (True, "")
