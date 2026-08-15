from types import SimpleNamespace

from robot_navigation.navigation_lifecycle_coordinator import (
    NavigationLifecycleCoordinator,
    join_service_name,
)


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def info(self, _message):
        return None


class _Timer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, ready=False):
        self.ready = ready

    def service_is_ready(self):
        return self.ready


class _Future:
    def __init__(self, *, response=None, error=None):
        self._response = response
        self._error = error

    def done(self):
        return True

    def result(self):
        if self._error is not None:
            raise self._error
        return self._response


def _coordinator(monkeypatch, *, now=10.0):
    coordinator = object.__new__(NavigationLifecycleCoordinator)
    coordinator._request_future = None
    coordinator._request_deadline = 0.0
    coordinator._next_attempt_at = 0.0
    coordinator._attempt_started_at = 0.0
    coordinator._service_wait_timeout = 2.0
    coordinator._retry_count = 3
    coordinator._retry_interval = 1.0
    coordinator._attempt = 0
    coordinator._client = _Client()
    coordinator._timer = _Timer()
    logger = _Logger()
    monkeypatch.setattr(NavigationLifecycleCoordinator, "get_logger", lambda _self: logger)
    monkeypatch.setattr("robot_navigation.navigation_lifecycle_coordinator.time.monotonic", lambda: now)
    return coordinator, logger


def test_join_service_name_preserves_default_root_service():
    assert join_service_name("", "/lifecycle_manager_navigation/manage_nodes") == (
        "/lifecycle_manager_navigation/manage_nodes"
    )


def test_join_service_name_normalizes_an_optional_namespace():
    assert join_service_name("/robot1/", "/lifecycle_manager_navigation/manage_nodes") == (
        "/robot1/lifecycle_manager_navigation/manage_nodes"
    )


def test_missing_service_schedules_a_bounded_retry(monkeypatch):
    coordinator, logger = _coordinator(monkeypatch)

    coordinator._tick()

    assert coordinator._attempt == 1
    assert coordinator._next_attempt_at == 11.0
    assert logger.errors == ["Lifecycle manager service is unavailable"]


def test_failed_response_schedules_a_retry(monkeypatch):
    coordinator, logger = _coordinator(monkeypatch)
    coordinator._request_future = _Future(response=SimpleNamespace(success=False))

    coordinator._tick()

    assert coordinator._attempt == 1
    assert logger.errors == ["Lifecycle manager rejected startup request"]


def test_future_exception_schedules_a_retry(monkeypatch):
    coordinator, logger = _coordinator(monkeypatch)
    coordinator._request_future = _Future(error=RuntimeError("service failed"))

    coordinator._tick()

    assert coordinator._attempt == 1
    assert logger.errors == ["Lifecycle startup failed: service failed"]


def test_retry_exhaustion_stops_the_coordinator(monkeypatch):
    coordinator, logger = _coordinator(monkeypatch)
    coordinator._attempt = coordinator._retry_count + 1

    coordinator._tick()

    assert coordinator._timer.cancelled is True
    assert logger.errors == ["Lifecycle startup retries exhausted"]
