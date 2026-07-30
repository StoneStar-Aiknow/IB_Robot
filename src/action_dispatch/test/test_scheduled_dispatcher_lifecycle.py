"""Scheduled dispatcher lifecycle race regressions."""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from action_dispatch import scheduled_action_dispatcher_node as dispatcher_module
from action_dispatch.scheduled_action_dispatcher_node import (
    DispatcherState,
    ScheduledActionDispatcherNode,
)


class _Future:
    def __init__(self, value) -> None:
        self._value = value

    def result(self):
        return self._value


class _PendingFuture:
    def add_done_callback(self, callback) -> None:
        self.callback = callback


def test_joint_snapshot_uses_local_monotonic_receive_time(monkeypatch):
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._safe_stop_plan = SimpleNamespace(joint_order=["1", "2"])
    node._joint_snapshot = None
    monkeypatch.setattr(dispatcher_module.time, "monotonic_ns", lambda: 123456789)
    message = SimpleNamespace(
        name=["2", "1"],
        position=[2.0, 1.0],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=9_999_999_999, nanosec=0)),
    )

    ScheduledActionDispatcherNode._joint_cb(node, message)

    assert node._joint_snapshot.valid
    assert node._joint_snapshot.positions == [1.0, 2.0]
    assert node._joint_snapshot.received_monotonic_ns == 123456789


def test_late_open_success_while_closing_triggers_compensating_close():
    session_id = "00112233-4455-4677-8899-aabbccddeeff"
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.CLOSING
    node._session_id = session_id
    node._session_generation = 0
    node._pending_open_session_id = session_id
    node._pending_open_completion = threading.Event()
    node._close_after_open = True
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = []
    node._inference_priority = 0
    close_calls: list[int] = []
    node._begin_close_session = lambda: close_calls.append(node._session_generation)
    result = SimpleNamespace(
        success=True,
        session_id=session_id,
        session_generation=7,
    )

    ScheduledActionDispatcherNode._open_result_callback(
        node,
        _Future(SimpleNamespace(result=result)),
        session_id,
        None,
    )

    assert node._session_generation == 7
    assert node._state == DispatcherState.CLOSING
    assert node._pending_open_completion.is_set()
    assert close_calls == [7]


def test_open_sends_only_logical_session_identity_and_deadline():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.STOPPED
    node._received_results = set()
    node._queue = deque()
    node._smoother = SimpleNamespace(reset=lambda: None)
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = ["backup"]
    node._inference_priority = 4
    node._default_open_timeout_ns = 1_000_000_000
    goals = []
    node._open_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda goal: goals.append(goal) or _PendingFuture(),
    )

    ScheduledActionDispatcherNode._open_new_session(node)

    assert len(goals) == 1
    assert goals[0].session_id
    assert goals[0].deadline.sec > 0


def test_shutdown_wait_can_drive_action_callbacks_with_spin_once():
    completed = threading.Event()
    spin_calls = 0

    def spin_once(*, timeout_sec):
        nonlocal spin_calls
        assert timeout_sec > 0
        spin_calls += 1
        completed.set()

    assert ScheduledActionDispatcherNode._wait_for_event(completed, 1_000_000_000, spin_once=spin_once)
    assert spin_calls == 1


def test_control_loop_uses_smoothed_plan_length_for_waterline_and_publishes_size():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._smoothing_enabled = True
    node._smoother = SimpleNamespace(plan_length=3)
    node._watermark = 3
    node._inflight_request_id = ""
    published: list[int] = []
    dispatches: list[bool] = []
    node._queue_size_pub = SimpleNamespace(publish=lambda message: published.append(message.data))
    node._request_dispatch = lambda: dispatches.append(True)
    node._execute_next_action = lambda: None

    ScheduledActionDispatcherNode._control_loop(node)

    assert published == [3]
    assert dispatches == [True]


def test_enqueue_chunk_skips_actions_executed_during_inference():
    from tensormsg.converter import TensorMsgConverter

    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._smoothing_enabled = False
    node._queue = deque()
    node._queue_size = 4
    node._safe_stop_plan = SimpleNamespace(total_positions=2)
    action_chunk = TensorMsgConverter.to_variant(
        {"action": np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])}
    )

    ScheduledActionDispatcherNode._enqueue_chunk(
        node,
        action_chunk,
        reported_chunk_size=4,
        actions_executed=2,
    )

    assert np.array_equal(np.asarray(node._queue), np.asarray([[4.0, 5.0], [6.0, 7.0]]))


def test_enqueue_chunk_rejects_overflow_without_truncating_actions():
    from tensormsg.converter import TensorMsgConverter

    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._smoothing_enabled = False
    node._queue = deque([np.asarray([9.0, 9.0])], maxlen=2)
    node._queue_size = 2
    node._safe_stop_plan = SimpleNamespace(total_positions=2)
    action_chunk = TensorMsgConverter.to_variant({"action": np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])})

    with pytest.raises(ValueError, match="queue capacity"):
        ScheduledActionDispatcherNode._enqueue_chunk(
            node,
            action_chunk,
            reported_chunk_size=3,
            actions_executed=0,
        )

    assert np.array_equal(np.asarray(node._queue), np.asarray([[9.0, 9.0]]))


def test_enqueue_chunk_rejects_reported_chunk_size_mismatch():
    from tensormsg.converter import TensorMsgConverter

    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._smoothing_enabled = False
    node._queue = deque()
    node._queue_size = 4
    node._safe_stop_plan = SimpleNamespace(total_positions=2)
    action_chunk = TensorMsgConverter.to_variant({"action": np.asarray([[0.0, 1.0], [2.0, 3.0]])})

    with pytest.raises(ValueError, match="reported chunk_size=1"):
        ScheduledActionDispatcherNode._enqueue_chunk(
            node,
            action_chunk,
            reported_chunk_size=1,
            actions_executed=0,
        )

    assert not node._queue


def test_smoothed_empty_plan_holds_last_action():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._smoothing_enabled = True
    node._smoother = SimpleNamespace(plan_length=0)
    node._last_action = np.asarray([1.0, 2.0])
    executed: list[np.ndarray] = []
    node._executor = SimpleNamespace(execute=lambda action: executed.append(action.copy()))

    ScheduledActionDispatcherNode._execute_next_action(node)

    assert len(executed) == 1
    assert np.array_equal(executed[0], np.asarray([1.0, 2.0]))


@pytest.mark.parametrize("code", ["unsupported_priority", "hardware_priority_unavailable"])
def test_backend_priority_rejections_are_never_retried(code):
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._inflight_request_id = "request"
    node._retry_max_attempts = 3
    node._retry_initial_ms = 10
    node._retry_max_ms = 100
    failures: list[str] = []
    node._fail_and_close = failures.append
    node.create_timer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("unsupported backend priorities must not schedule a retry")
    )

    ScheduledActionDispatcherNode._retry_dispatch_not_started(node, "request", 0, code)

    assert failures == [f"scheduled dispatch rejected: {code}"]


def test_not_started_retry_limit_counts_retries_after_initial_request():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._inflight_request_id = "request"
    node._retry_max_attempts = 3
    node._retry_initial_ms = 10
    node._retry_max_ms = 100
    failures: list[str] = []
    node._fail_and_close = failures.append
    node.create_timer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("retry index 3 must exhaust a three-retry budget")
    )

    ScheduledActionDispatcherNode._retry_dispatch_not_started(node, "request", 3)

    assert failures == ["scheduled dispatch retries exhausted"]


def test_dispatch_retry_reuses_one_observation_snapshot():
    node = object.__new__(ScheduledActionDispatcherNode)
    node._state_lock = threading.RLock()
    node._state = DispatcherState.ACTIVE
    node._session_id = "00112233-4455-4677-8899-aabbccddeeff"
    node._session_generation = 3
    node._inflight_request_id = ""
    node._inflight_deadline_utc_ns = 0
    node._inflight_observation_time_ns = 0
    node._inflight_goal_handle = None
    node._default_request_timeout_ns = 2_000_000_000
    node._inference_pipeline = "policy"
    node._inference_fallback_chain = ["fallback"]
    node._inference_priority = 0
    node._inference_prompt = ""
    node._current_plan_length_locked = lambda: 0
    observation_time_ns = 123_456_789_012
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=observation_time_ns))
    goals = []
    node._dispatch_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda goal: goals.append(goal) or _PendingFuture(),
    )

    ScheduledActionDispatcherNode._request_dispatch(node)
    first_request_id = node._inflight_request_id
    observation_time_ns += 99_000_000
    ScheduledActionDispatcherNode._request_dispatch(node, attempt=1, replace_request_id=first_request_id)

    assert len(goals) == 2
    first_stamp = goals[0].obs_timestamp.sec * 1_000_000_000 + goals[0].obs_timestamp.nanosec
    second_stamp = goals[1].obs_timestamp.sec * 1_000_000_000 + goals[1].obs_timestamp.nanosec
    assert first_stamp == second_stamp == 123_456_789_012
    assert goals[0].fallback_chain == goals[1].fallback_chain == ["fallback"]
    assert goals[1].deadline == goals[0].deadline
    assert goals[1].request_id != goals[0].request_id
