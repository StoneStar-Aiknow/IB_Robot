import threading
import time
from types import SimpleNamespace

from robot_navigation.navigation_command_server import NavigationCommandServer


class _PendingFuture:
    def add_done_callback(self, _callback):
        return None

    def result(self):
        raise AssertionError("a pending future must not be read")


class _CompletedFuture:
    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return "done"


def test_wait_future_has_a_finite_timeout():
    started = time.monotonic()

    assert NavigationCommandServer._wait_future(_PendingFuture(), 0.01) is None
    assert time.monotonic() - started < 0.5


def test_cancel_response_is_accepted_only_when_a_goal_is_listed():
    assert NavigationCommandServer._cancel_response_accepted(SimpleNamespace(goals_canceling=[object()]))
    assert not NavigationCommandServer._cancel_response_accepted(SimpleNamespace(goals_canceling=[]))


def test_result_wait_uses_cancel_terminal_timeout_after_cancel_request():
    cancel_requested = threading.Event()
    cancel_requested.set()
    started = time.monotonic()

    result, cancel_terminal_timeout = NavigationCommandServer._wait_result_future(
        _PendingFuture(),
        result_timeout=1.0,
        cancel_timeout=0.01,
        cancel_requested=cancel_requested,
    )

    assert result is None
    assert cancel_terminal_timeout is True
    assert time.monotonic() - started < 0.5


def test_result_wait_returns_a_completed_future_without_timeout():
    result, cancel_terminal_timeout = NavigationCommandServer._wait_result_future(
        _CompletedFuture(),
        result_timeout=1.0,
        cancel_timeout=0.01,
        cancel_requested=threading.Event(),
    )

    assert result == "done"
    assert cancel_terminal_timeout is False
