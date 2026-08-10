"""Scheduler-specific recording reset regressions."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from dataset_tools.record_cli import RecordCLI


def test_send_goal_does_not_start_recording_after_session_restart_failure():
    node = object.__new__(RecordCLI)
    node._goal_started_evt = threading.Event()
    node._goal_rejected_evt = threading.Event()
    node._episode_finished_evt = threading.Event()
    node._last_result_success = True
    node._last_result_message = "old"
    node._should_reset_before_episode = lambda: True
    node.prepare_new_episode = lambda: False
    node._action_client = SimpleNamespace(
        send_goal_async=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recording must not start after restart_session failure")
        )
    )

    RecordCLI.send_goal(node, "pick")

    assert node._goal_rejected_evt.is_set()
    assert node._episode_finished_evt.is_set()
    assert not node._goal_started_evt.is_set()
    assert node._last_result_success is False
    assert node._last_result_message == "inference session restart failed"


def test_legacy_prepare_new_episode_keeps_dispatcher_then_policy_reset_fallback():
    node = object.__new__(RecordCLI)
    node._restart_session_client = None
    calls: list[str] = []
    node._reset_dispatcher_state = lambda: calls.append("dispatcher") or False
    node._reset_policy_state = lambda: calls.append("policy")

    assert RecordCLI.prepare_new_episode(node)
    assert calls == ["dispatcher", "policy"]
