"""Executor teardown contract: bounded wait and fail-closed cleanup.

``bounded_executor_teardown`` has exactly one success outcome: the executor
quiesced, the SDK closed and the node was destroyed. A timeout, a failed SDK
close and a failed node destroy each return a distinct reason; only the timeout
leaves both resources open, a close failure must not continue to ``destroy_node``
(its handlers may still be reached by the SDK reader threads), and a destroy
failure must not be reported as a completed cleanup.

The process tests run the real ``main()`` with a child node:
``timeout`` uses a 30 s callback against the production 2 s wait;
``close_error``/``destroy_error`` make the real teardown step raise and assert
the process exits with a diagnostic naming that step. Sentinel prints make a
wrongly executed close/destroy a loud failure instead of a silently accepted
race.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import rclpy
from mock_process import vendor_test_domain
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from unitree_g1_robot import vendor_gateway as _vg
from unitree_g1_robot.teardown import (
    TEARDOWN_CLOSE_FAILED,
    TEARDOWN_DESTROY_FAILED,
    TEARDOWN_FAILURE_EXIT_CODE,
    TEARDOWN_TIMEOUT,
    TEARDOWN_TIMEOUT_MARKER,
    bounded_executor_teardown,
    teardown_failure_message,
)

TEST_DOMAIN_ID = vendor_test_domain()
CHILD_SCRIPT = Path(__file__).with_name("teardown_timeout_child.py")

#: Bound for killing the child if the bounded-exit contract regresses. The
#: timeout callback sleeps 30 s, so anything near that means the process was
#: held by the executor worker instead of the 2 s teardown wait.
EXIT_BOUND_S = 15.0


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Own the ROS context on this suite's private domain, if nothing else does."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init(domain_id=TEST_DOMAIN_ID)
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


class _RecordingNode(Node):
    """A node that records destroy_node; the callback timing is per test."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.destroyed = False

    def destroy_node(self) -> bool:
        self.destroyed = True
        return super().destroy_node()


class _FailingDestroyNode(_RecordingNode):
    """A node whose destroy_node raises after recording the attempt."""

    def destroy_node(self) -> bool:
        self.destroyed = True
        raise RuntimeError("fake node destroy failure")


def _spinning(executor: MultiThreadedExecutor) -> threading.Thread:
    thread = threading.Thread(target=executor.spin, name="teardown-test-spin", daemon=True)
    thread.start()
    return thread


def test_bounded_teardown_closes_and_destroys_after_quiesce():
    node = _RecordingNode("teardown_quiesced")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = _spinning(executor)
    closed: list[bool] = []

    result = bounded_executor_teardown(executor, node, lambda: closed.append(True))

    thread.join(5.0)
    assert result.ok is True
    assert closed == [True], "a quiesced executor must still run the SDK close"
    assert node.destroyed is True, "a quiesced executor must still destroy the node"


def test_bounded_teardown_timeout_leaves_node_and_sdk_open():
    node = _RecordingNode("teardown_timed_out")
    entered = threading.Event()
    finished = threading.Event()

    def slow_callback() -> None:
        if entered.is_set():
            return
        entered.set()
        time.sleep(0.5)
        finished.set()

    node.create_timer(0.02, slow_callback)
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(node)
    thread = _spinning(executor)
    assert entered.wait(5.0), "the slow callback never started"

    closed: list[bool] = []
    result = bounded_executor_teardown(executor, node, lambda: closed.append(True), timeout_s=0.1)

    assert result.ok is False
    assert result.reason == TEARDOWN_TIMEOUT
    assert TEARDOWN_TIMEOUT_MARKER in teardown_failure_message("test", result, timeout_s=0.1)
    assert closed == [], "the timeout branch must not close an SDK a callback may still use"
    assert node.destroyed is False, "the timeout branch must not destroy a node a callback may still use"
    assert finished.wait(5.0), "the slow callback did not finish"
    thread.join(5.0)

    # The callback is gone now; the node can be destroyed by the test itself.
    node.destroy_node()


def test_bounded_teardown_close_failure_keeps_the_node_alive():
    node = _RecordingNode("teardown_close_failed")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = _spinning(executor)

    def failing_close() -> None:
        raise RuntimeError("fake SDK close failure")

    result = bounded_executor_teardown(executor, node, failing_close)

    thread.join(5.0)
    assert result.ok is False
    assert result.reason == TEARDOWN_CLOSE_FAILED
    assert "fake SDK close failure" in result.detail
    assert node.destroyed is False, "a failed SDK close must not continue to destroy the node"
    message = teardown_failure_message("test", result)
    assert TEARDOWN_CLOSE_FAILED in message
    assert "the node was not destroyed" in message
    assert "fake SDK close failure" in message

    # destroy_node was not called by the helper; the test owns the cleanup now.
    node.destroy_node()


def test_bounded_teardown_destroy_failure_is_reported():
    node = _FailingDestroyNode("teardown_destroy_failed")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = _spinning(executor)
    closed: list[bool] = []

    result = bounded_executor_teardown(executor, node, lambda: closed.append(True))

    thread.join(5.0)
    assert closed == [True], "the SDK close still ran before the destroy failure"
    assert result.ok is False
    assert result.reason == TEARDOWN_DESTROY_FAILED
    assert "fake node destroy failure" in result.detail
    message = teardown_failure_message("test", result)
    assert TEARDOWN_DESTROY_FAILED in message
    assert "fake node destroy failure" in message

    # The helper reported the failure instead of claiming cleanup; clean up the
    # node object through the base implementation the failing override replaced.
    Node.destroy_node(node)


@pytest.mark.skipif(not _vg.VENDOR_AVAILABLE, reason="unitree_sdk2py is not installed")
def test_main_teardown_timeout_exits_three_without_closing_resources():
    log = Path(tempfile.gettempdir()) / f"g1_teardown_timeout_{os.getpid()}.log"
    env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN_ID), PYTHONUNBUFFERED="1")
    with open(log, "w", encoding="utf-8") as stream:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(CHILD_SCRIPT)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                text = log.read_text(errors="replace") if log.exists() else ""
                if "CALLBACK-ENTERED" in text:
                    break
                if proc.poll() is not None:
                    raise AssertionError(f"child exited before the callback started; output: {text[-2000:]}")
                time.sleep(0.05)
            else:
                raise AssertionError("the slow callback never announced itself")

            started = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            exit_code = proc.wait(timeout=EXIT_BOUND_S)
            elapsed = time.monotonic() - started
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)

    output = log.read_text(errors="replace")
    assert exit_code == TEARDOWN_FAILURE_EXIT_CODE, f"expected the teardown-failure status; output: {output[-2000:]}"
    assert TEARDOWN_TIMEOUT_MARKER in output, f"the timeout must be observable; output: {output[-2000:]}"
    assert "SDK-CLOSED" not in output, "the timeout branch closed the SDK a callback may still use"
    assert "NODE-DESTROYED" not in output, "the timeout branch destroyed the node a callback may still use"
    assert elapsed < EXIT_BOUND_S, f"the process was held for {elapsed:.1f}s despite exiting {exit_code}"


@pytest.mark.parametrize(
    ("mode", "reason", "detail"),
    [
        ("close_error", TEARDOWN_CLOSE_FAILED, "fake SDK close failure"),
        ("destroy_error", TEARDOWN_DESTROY_FAILED, "fake node destroy failure"),
    ],
)
@pytest.mark.skipif(not _vg.VENDOR_AVAILABLE, reason="unitree_sdk2py is not installed")
def test_main_reports_the_failed_teardown_step_and_exits_three(mode: str, reason: str, detail: str):
    log = Path(tempfile.gettempdir()) / f"g1_teardown_{mode}_{os.getpid()}.log"
    env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN_ID), PYTHONUNBUFFERED="1", TEARDOWN_CHILD_MODE=mode)
    with open(log, "w", encoding="utf-8") as stream:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(CHILD_SCRIPT)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                text = log.read_text(errors="replace") if log.exists() else ""
                if "READY" in text:
                    break
                if proc.poll() is not None:
                    raise AssertionError(f"child exited before READY; output: {text[-2000:]}")
                time.sleep(0.05)
            else:
                raise AssertionError("the child never announced READY")
            proc.send_signal(signal.SIGTERM)
            exit_code = proc.wait(timeout=EXIT_BOUND_S)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)

    output = log.read_text(errors="replace")
    assert exit_code == TEARDOWN_FAILURE_EXIT_CODE, f"expected exit 3; output: {output[-2000:]}"
    assert f"executor teardown failed: {reason}" in output, f"the failed step must be named; output: {output[-2000:]}"
    assert detail in output, f"the failure detail must be reported; output: {output[-2000:]}"
    if mode == "close_error":
        assert "NODE-DESTROYED" not in output, "a failed SDK close must not continue to destroy the node"
    else:
        assert "SDK-CLOSED" in output, "the destroy failure happens after a successful SDK close"
        assert "NODE-DESTROYED" in output, "the destroy attempt must be visible"
