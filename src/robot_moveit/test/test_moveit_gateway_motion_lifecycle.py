#!/usr/bin/env python3
"""Unit tests for MoveIt gateway motion ownership and cancellation."""

from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

from geometry_msgs.msg import Pose

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from moveit_gateway import MoveIt2State, MoveItGateway  # noqa: E402


class FakeLogger:
    def __init__(self):
        self.messages = []

    def _record(self, level, message):
        self.messages.append((level, message))

    def debug(self, message):
        self._record("debug", message)

    def info(self, message):
        self._record("info", message)

    def warning(self, message):
        self._record("warning", message)

    warn = warning

    def error(self, message):
        self._record("error", message)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


class FakeMoveIt2:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = MoveIt2State.IDLE
        self.motion_suceeded = False
        self.max_velocity = 0.8
        self.cancel_calls = 0
        self.cancel_called = threading.Event()
        self.dispatch = True

    def query_state(self):
        with self._lock:
            return self._state

    def set_state(self, state):
        with self._lock:
            self._state = state

    def cancel_execution(self):
        self.cancel_calls += 1
        self.cancel_called.set()

    def get_last_execution_error_code(self):
        return None

    def clear_goal_constraints(self):
        pass

    def move_to_configuration(self, _joint_positions):
        if self.dispatch:
            self.set_state(MoveIt2State.REQUESTING)


class RaisingTfBuffer:
    def lookup_transform(self, *_args, **_kwargs):
        raise RuntimeError("transform unavailable in unit test")


def make_gateway():
    gateway = MoveItGateway.__new__(MoveItGateway)
    gateway._initialize_motion_coordinator()
    gateway._motion_start_timeout_s = 0.05
    gateway._motion_execution_timeout_s = 0.0
    gateway._motion_cancel_timeout_s = 0.2
    gateway._motion_status_hold_s = 0.0
    gateway._motion_status = "idle"
    gateway.motion_status_pub = FakePublisher()
    gateway.moveit2 = FakeMoveIt2()
    gateway.joint_names = ["1", "2"]
    gateway.latest_joint_state = None
    gateway.base_link = "base"
    gateway.ee_link = "gripper"
    gateway.shoulder_link = "shoulder"
    gateway.tf_buffer = RaisingTfBuffer()
    logger = FakeLogger()
    gateway.get_logger = lambda: logger
    return gateway


def make_configuration_request():
    return SimpleNamespace(
        target_joint_state=SimpleNamespace(name=["1", "2"], position=[0.1, 0.2]),
        velocity_scaling=0.0,
    )


def make_response():
    return SimpleNamespace(success=None, message="", execution_time_s=0.0)


def test_concurrent_service_request_returns_busy_without_dispatch():
    gateway = make_gateway()
    entered = threading.Event()
    release = threading.Event()
    dispatch_count = 0

    def blocking_move(_joint_positions):
        nonlocal dispatch_count
        dispatch_count += 1
        entered.set()
        assert release.wait(timeout=1.0)
        return False

    gateway.move_to_joint = blocking_move
    first_response = make_response()
    first_thread = threading.Thread(
        target=gateway._move_to_configuration_service_cb,
        args=(make_configuration_request(), first_response),
    )
    first_thread.start()
    assert entered.wait(timeout=1.0)
    statuses_before = list(gateway.motion_status_pub.messages)

    second_response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert second_response.success is False
    assert second_response.message == "MoveIt gateway is busy"
    assert dispatch_count == 1
    assert gateway.motion_status_pub.messages == statuses_before
    release.set()
    first_thread.join(timeout=1.0)
    assert not first_thread.is_alive()


def test_service_is_rejected_while_cmd_pose_owns_motion():
    gateway = make_gateway()
    entered = threading.Event()
    release = threading.Event()

    def blocking_strategy(_position, _orientation):
        entered.set()
        assert release.wait(timeout=1.0)
        gateway.moveit2.set_state(MoveIt2State.EXECUTING)
        return True

    gateway._move_with_strategies = blocking_strategy
    cmd_thread = threading.Thread(target=gateway.cmd_pose_callback, args=(Pose(),))
    cmd_thread.start()
    assert entered.wait(timeout=1.0)
    statuses_before = list(gateway.motion_status_pub.messages)

    response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert response.success is False
    assert response.message == "MoveIt gateway is busy"
    assert gateway.motion_status_pub.messages == statuses_before
    release.set()
    cmd_thread.join(timeout=1.0)
    assert not cmd_thread.is_alive()

    gateway.moveit2.motion_suceeded = True
    gateway.moveit2.set_state(MoveIt2State.IDLE)
    gateway._motion_watchdog_callback()
    assert gateway._active_motion_token is None


def test_timeout_restores_velocity_and_publishes_idle_only_after_cancel_is_confirmed():
    gateway = make_gateway()
    token = gateway._claim_motion("MoveToConfiguration")
    gateway._prepare_motion(token)
    gateway._set_motion_velocity(token, 0.2)
    gateway.moveit2.set_state(MoveIt2State.EXECUTING)
    result = None

    def wait_for_motion():
        nonlocal result
        result = gateway._wait_for_motion_completion(token, "MoveToConfiguration")

    wait_thread = threading.Thread(target=wait_for_motion)
    wait_thread.start()
    assert gateway.moveit2.cancel_called.wait(timeout=1.0)

    assert gateway.motion_status_pub.messages == ["executing"]
    assert gateway.moveit2.max_velocity == 0.2
    assert gateway._active_motion_token == token

    gateway.moveit2.set_state(MoveIt2State.IDLE)
    wait_thread.join(timeout=1.0)
    assert not wait_thread.is_alive()
    assert result[0] is False
    assert result[2] is True

    gateway._finalize_motion(token, success=False)
    assert gateway.moveit2.max_velocity == 0.8
    assert gateway.motion_status_pub.messages == ["executing", "failed", "idle"]
    assert gateway._active_motion_token is None


def test_unconfirmed_cancellation_keeps_gateway_busy_until_watchdog_sees_idle():
    gateway = make_gateway()
    gateway._motion_cancel_timeout_s = 0.0
    token = gateway._claim_motion("MoveToConfiguration")
    gateway._prepare_motion(token)
    gateway._set_motion_velocity(token, 0.2)
    gateway.moveit2.set_state(MoveIt2State.EXECUTING)

    success, message, terminal_confirmed = gateway._wait_for_motion_completion(token, "MoveToConfiguration")

    assert success is False
    assert "cancellation is still pending" in message
    assert terminal_confirmed is False
    assert gateway._claim_motion("new request") is None
    assert gateway.motion_status_pub.messages == ["executing"]
    assert gateway.moveit2.max_velocity == 0.2

    gateway._motion_watchdog_callback()
    assert gateway._active_motion_token == token
    gateway.moveit2.set_state(MoveIt2State.IDLE)
    gateway._motion_watchdog_callback()
    assert gateway.moveit2.max_velocity == 0.8
    assert gateway.motion_status_pub.messages == ["executing", "failed", "idle"]


def test_old_token_cannot_finalize_a_new_motion():
    gateway = make_gateway()
    old_token = gateway._claim_motion("old")
    gateway._prepare_motion(old_token)
    assert gateway._finalize_motion(old_token, success=True)

    new_token = gateway._claim_motion("new")
    gateway._prepare_motion(new_token)
    statuses_before = list(gateway.motion_status_pub.messages)

    assert gateway._finalize_motion(old_token, success=False) is False
    assert gateway._active_motion_token == new_token
    assert gateway.motion_status_pub.messages == statuses_before


def test_skipped_moveit_dispatch_is_not_reported_as_successful():
    gateway = make_gateway()
    gateway.moveit2.dispatch = False

    assert gateway.move_to_joint([0.1, 0.2]) is False

    class DoneFuture:
        @staticmethod
        def done():
            return True

    gateway.moveit2.compute_ik_async = lambda **_kwargs: DoneFuture()
    gateway.moveit2.get_compute_ik_result = lambda _future: SimpleNamespace(
        name=["1", "2"],
        position=[0.1, 0.2],
    )
    assert gateway.solve_and_move(Pose()) is False
