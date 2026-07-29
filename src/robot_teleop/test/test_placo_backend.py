from types import SimpleNamespace

from robot_teleop.cartesian_backend.placo_servo import PlacoServoBackend


def test_late_start_success_is_stopped_after_release():
    stop_requests = []
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._requested_enabled = False
    backend._active_enabled = False
    backend._start_request_inflight = True
    backend.disable = lambda: stop_requests.append(True)
    backend._node = SimpleNamespace(
        get_logger=lambda: SimpleNamespace(info=lambda _message: None, error=lambda _message: None)
    )
    future = SimpleNamespace(result=lambda: SimpleNamespace(success=True, message="started"))

    backend._on_start_response(future)

    assert backend._active_enabled is False
    assert backend._start_request_inflight is False
    assert len(stop_requests) == 1


def test_pending_stop_blocks_reenable():
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._stop_pending = True
    backend._requested_enabled = False

    assert backend.enable() is False
    assert backend._requested_enabled is False


def test_released_inflight_start_blocks_new_enable_until_late_response():
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._stop_pending = False
    backend._start_request_inflight = True
    backend._requested_enabled = False

    assert backend.enable() is False
    assert backend._requested_enabled is False


def test_start_retry_does_not_send_duplicate_request_while_inflight():
    requests = []
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._requested_enabled = True
    backend._stop_pending = False
    backend._start_request_inflight = True
    backend._start_cli = SimpleNamespace(call_async=lambda request: requests.append(request))
    backend._cancel_start_retry = lambda: None

    backend._send_start_request()

    assert requests == []


def _pending_home_backend():
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._home_pending = True
    backend._home_result = None
    backend._home_goal_handle = None
    backend._home_goal_generation = None
    backend._lifecycle_generation = 3
    backend._node = SimpleNamespace(
        get_logger=lambda: SimpleNamespace(info=lambda _message: None, error=lambda _message: None)
    )
    return backend


class _GoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result_future = SimpleNamespace(
            add_done_callback=lambda callback: setattr(self, "result_callback", callback)
        )
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1


def test_home_action_success_is_consumed_once():
    backend = _pending_home_backend()
    goal_handle = _GoalHandle()
    backend._on_home_goal_response(SimpleNamespace(result=lambda: goal_handle), generation=3)
    result = SimpleNamespace(success=True, error_code="", message="home reached")

    backend._on_home_result(SimpleNamespace(result=lambda: SimpleNamespace(result=result)), generation=3)

    assert backend.consume_home_result() is True
    assert backend.consume_home_result() is None


def test_home_action_abort_is_reported():
    backend = _pending_home_backend()
    result = SimpleNamespace(success=False, error_code="TIMEOUT", message="timed out")

    backend._on_home_result(SimpleNamespace(result=lambda: SimpleNamespace(result=result)), generation=3)

    assert backend.consume_home_result() is False


def test_late_home_goal_is_canceled_after_disable_generation():
    backend = _pending_home_backend()
    goal_handle = _GoalHandle()

    backend._on_home_goal_response(SimpleNamespace(result=lambda: goal_handle), generation=2)

    assert goal_handle.cancel_calls == 1
    assert backend._home_goal_handle is None


def test_keepalive_publishes_command_lease():
    published = []
    backend = PlacoServoBackend.__new__(PlacoServoBackend)
    backend._lease_pub = SimpleNamespace(publish=published.append)

    backend.keepalive()

    assert len(published) == 1
