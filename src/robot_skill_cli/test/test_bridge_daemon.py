from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_skill_cli import bridge_daemon
from robot_skill_cli.ros_bridge import BridgeError


class _Future:
    def __init__(self) -> None:
        self._done = False
        self._value = None
        self._callbacks = []
        self.cancelled = False

    def set_result(self, value) -> None:
        self._value = value
        self._done = True
        for callback in self._callbacks:
            callback(self)

    def add_done_callback(self, callback) -> None:
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._value

    def cancel(self) -> None:
        self.cancelled = True


class _Message:
    """Minimal stand-in for a generated ROS message."""

    _fields = {}

    def __init__(self, **values) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    def get_fields_and_field_types(self):
        return {name: "" for name in vars(self)}


class _Handle:
    def __init__(self, result) -> None:
        self.accepted = True
        self.status = 2
        self.result_future = _Future()
        self._result = result
        self.cancel_requests = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_requests += 1
        future = _Future()
        future.set_result(_Message(return_code=0))
        return future


class _FakeBridge:
    instances: list[_FakeBridge] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.start_error = ""
        self.started = 0
        self.handles = []
        _FakeBridge.instances.append(self)

    def start(self) -> bool:
        self.started += 1
        return self.kwargs.get("status_service") != "/broken"

    def close(self) -> None:
        pass

    def get_status(self, *, timeout_sec: float):
        self.status_calls = getattr(self, "status_calls", 0) + 1
        if self.status_calls <= getattr(self, "lost_status_responses", 0):
            raise BridgeError("RESULT_TIMEOUT", "service response timed out", exit_code=5)
        return {"control_plane_ready": True, "timeout": timeout_sec, "steps": [1, 2]}

    def validate_skill(self, payload, *, timeout_sec: float):
        raise BridgeError("SKILL_SCHEMA_INVALID", f"bad {payload['skill_name']}", exit_code=7)

    def send_skill_goal(self, payload, *, task_id: str, feedback_callback=None):
        handle = _Handle(_Message(success=True, message="done"))
        self.handles.append(handle)
        goal_future = _Future()

        def run() -> None:
            goal_future.set_result(handle)
            if feedback_callback is not None:
                feedback_callback({"state": "executing", "detail": task_id})
            time.sleep(0.05)
            handle.result_future.set_result(_Message(status=4, result=_Message(success=True, message="done")))

        threading.Thread(target=run, daemon=True).start()
        return goal_future


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    _FakeBridge.instances.clear()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "1")
    path = Path(tmp_path) / "bridge.sock"
    server = bridge_daemon.BridgeDaemon(path, bridge_factory=_FakeBridge, idle_timeout_sec=30.0)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", lambda spawn=True: bridge_daemon._connect(path))
    yield server
    server.request_stop()
    thread.join(timeout=3.0)


def _client(**kwargs) -> bridge_daemon.ResidentBridge:
    bridge = bridge_daemon.ResidentBridge(status_service="/status", **kwargs)
    assert bridge.start() is True
    assert bridge.resident is True
    return bridge


def test_resident_bridge_forwards_calls_and_errors_and_shares_one_ros_bridge(daemon) -> None:
    first = _client()
    second = _client()
    try:
        assert first.get_status(timeout_sec=2.5) == {"control_plane_ready": True, "timeout": 2.5, "steps": [1, 2]}
        with pytest.raises(BridgeError) as error:
            second.validate_skill({"skill_name": "nod_yes"}, timeout_sec=1.0)
        assert (error.value.code, error.value.exit_code, str(error.value)) == ("SKILL_SCHEMA_INVALID", 7, "bad nod_yes")
    finally:
        first.close()
        second.close()
    # Both invocations reused the participant started by the first one.
    assert [bridge.started for bridge in _FakeBridge.instances] == [1]


def test_resident_bridge_proxies_goal_futures_handles_and_feedback(daemon) -> None:
    bridge = _client()
    feedback = []
    try:
        goal_future = bridge.send_skill_goal(
            {"skill_name": "nod_yes"}, task_id="t-1", feedback_callback=feedback.append
        )
        assert bridge.wait_future(goal_future, timeout_sec=2.0)
        handle = goal_future.result()
        assert handle.accepted is True
        result_future = handle.get_result_async()
        assert bridge.wait_future(result_future, timeout_sec=2.0)
        wrapped = result_future.result()
        assert (wrapped.status, wrapped.result.success, wrapped.result.message) == (4, True, "done")
        cancel = handle.cancel_goal_async()
        assert bridge.wait_future(cancel, timeout_sec=2.0)
        assert cancel.result().return_code == 0
    finally:
        bridge.close()
    deadline = time.monotonic() + 1.0
    while not feedback and time.monotonic() < deadline:
        time.sleep(0.01)
    assert feedback == [{"state": "executing", "detail": "t-1"}]
    assert _FakeBridge.instances[0].handles[0].cancel_requests == 1


def test_disconnecting_client_does_not_cancel_its_running_goal(daemon) -> None:
    bridge = _client()
    goal_future = bridge.send_skill_goal({"skill_name": "wave_hello"}, task_id="t-2")
    assert bridge.wait_future(goal_future, timeout_sec=2.0)
    bridge.close()
    time.sleep(0.2)
    assert _FakeBridge.instances[0].handles[0].cancel_requests == 0


def test_bridge_start_failure_is_reported_without_falling_back(daemon) -> None:
    bridge = bridge_daemon.ResidentBridge(status_service="/broken")
    try:
        assert bridge.start() is False
        assert bridge.resident is False
    finally:
        bridge.close()


def test_unreachable_or_disabled_daemon_falls_back_to_in_process_bridge(monkeypatch) -> None:
    created = []

    class _LocalBridge(_FakeBridge):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setattr("robot_skill_cli.ros_bridge.RosBridge", _LocalBridge)

    def unreachable(spawn=True):
        raise bridge_daemon.DaemonUnavailable("no daemon")

    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", unreachable)
    bridge = bridge_daemon.ResidentBridge(status_service="/status")
    assert bridge.start() is True
    assert bridge.resident is False
    assert bridge.get_status(timeout_sec=1.0)["control_plane_ready"] is True

    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "0")
    monkeypatch.setattr(
        bridge_daemon, "connect_or_spawn", lambda spawn=True: pytest.fail("disabled daemon was contacted")
    )
    disabled = bridge_daemon.ResidentBridge(status_service="/status")
    assert disabled.start() is True
    assert disabled.resident is False
    assert len(created) == 2


def test_socket_identity_follows_the_ros_transport_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    base = bridge_daemon.socket_path()
    assert base == bridge_daemon.socket_path()
    assert (tmp_path / "ibrobot").stat().st_mode & 0o777 == 0o700

    monkeypatch.setenv("ROS_DOMAIN_ID", "43")
    assert bridge_daemon.socket_path() != base
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    assert bridge_daemon.socket_path() != base


def test_ros_messages_round_trip_as_attribute_namespaces() -> None:
    encoded = bridge_daemon._encode_plain(_Message(status=4, result=_Message(names=("a", "b"), raw=b"\x01")))
    decoded = bridge_daemon._decode_plain(encoded)
    assert isinstance(decoded, SimpleNamespace)
    assert decoded.status == 4
    assert decoded.result.names == ["a", "b"]
    assert decoded.result.raw == [1]


def test_fresh_daemon_bridge_retries_read_only_status_until_the_gateway_answers(monkeypatch) -> None:
    # The first responses can be lost while the Gateway is still discovering a
    # new participant; the daemon absorbs that before serving any request.
    class _Lossy(_FakeBridge):
        lost_status_responses = 2

    monkeypatch.setattr(bridge_daemon, "WARM_UP_ATTEMPT_SEC", 0.01)
    server = bridge_daemon.BridgeDaemon(Path("/nonexistent.sock"), bridge_factory=_Lossy)
    bridge, error = server.bridge_for({"status_service": "/status"})
    assert error == ""
    assert bridge.status_calls == 3
    assert server.bridge_for({"status_service": "/status"})[0] is bridge
    assert bridge.status_calls == 3
