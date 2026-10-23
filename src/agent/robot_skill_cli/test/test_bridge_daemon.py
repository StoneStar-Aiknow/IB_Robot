from __future__ import annotations

import os
import socket
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
    monkeypatch.setenv(bridge_daemon.DIRECTORY_ENV, str(tmp_path / "ibrobot"))
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


@pytest.mark.parametrize(
    ("mode", "spawn"),
    [(None, False), ("1", False), ("connect", False), ("spawn", True), ("SPAWN", True), ("auto", True)],
)
def test_robot_skill_starts_a_daemon_only_when_opted_in(monkeypatch, mode, spawn) -> None:
    """The launch owns the daemon; a plain robot-skill call never leaves one behind."""
    monkeypatch.setattr("robot_skill_cli.ros_bridge.RosBridge", _FakeBridge)
    if mode is None:
        monkeypatch.delenv(bridge_daemon.DISABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(bridge_daemon.DISABLE_ENV, mode)
    requested = []

    def no_daemon(*, spawn=True):
        requested.append(spawn)
        raise bridge_daemon.DaemonUnavailable("no daemon")

    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", no_daemon)
    bridge = bridge_daemon.ResidentBridge(status_service="/status")
    try:
        assert bridge.start() is True
        assert bridge.resident is False
    finally:
        bridge.close()
    assert requested == [spawn]


def test_connect_only_mode_does_not_spawn_when_no_daemon_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(bridge_daemon.DIRECTORY_ENV, str(tmp_path / "ibrobot"))
    monkeypatch.setattr(bridge_daemon, "_spawn", lambda path: pytest.fail("connect-only mode spawned a daemon"))
    with pytest.raises(bridge_daemon.DaemonUnavailable):
        bridge_daemon.connect_or_spawn(spawn=False)


def test_socket_identity_follows_the_ros_graph_not_participant_tuning(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(bridge_daemon.DIRECTORY_ENV, str(tmp_path / "ibrobot"))
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("FASTDDS_BUILTIN_TRANSPORTS", raising=False)
    base = bridge_daemon.socket_path()
    assert base == bridge_daemon.socket_path()
    assert (tmp_path / "ibrobot").stat().st_mode & 0o777 == 0o700

    # The embodied launch forces UDPv4 for its nodes; the user's shell does not.
    monkeypatch.setenv("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    monkeypatch.setenv("PYTHONPATH", "/another/spelling")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "elsewhere"))
    assert bridge_daemon.socket_path() == base

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


class _TokenBridge(_FakeBridge):
    def for_caller(self, caller_token: str):
        view = SimpleNamespace(caller_token=caller_token, shared=self)
        view.get_execution = lambda **_kwargs: {"token": view.caller_token, "participant": id(self)}
        return view


def _serve(path: Path, factory, **kwargs):
    server = bridge_daemon.BridgeDaemon(path, bridge_factory=factory, **kwargs)
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_callers_with_different_tokens_share_one_participant(tmp_path, monkeypatch) -> None:
    _FakeBridge.instances.clear()
    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "1")
    path = tmp_path / "token.sock"
    server, thread = _serve(path, _TokenBridge)
    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", lambda spawn=True: bridge_daemon._connect(path))
    try:
        operator = bridge_daemon.ResidentBridge(status_service="/status", caller_token="secret-a")
        monkeypatch.setenv("TOKEN_FROM_ENV", "secret-b")
        hermes = bridge_daemon.ResidentBridge(status_service="/status", caller_token_env="TOKEN_FROM_ENV")
        assert operator.start() and hermes.start()
        a, b = operator.get_execution(request_id="r"), hermes.get_execution(request_id="r")
        assert (a["token"], b["token"]) == ("secret-a", "secret-b")
        assert a["participant"] == b["participant"]
        assert len(_FakeBridge.instances) == 1
        operator.close()
        hermes.close()
    finally:
        server.request_stop()
        thread.join(timeout=3.0)


def test_prewarmed_bridge_is_the_one_a_later_cli_call_receives(tmp_path, monkeypatch) -> None:
    _FakeBridge.instances.clear()
    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "1")
    path = tmp_path / "prewarm.sock"
    server, thread = _serve(path, _FakeBridge, idle_timeout_sec=0)
    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", lambda spawn=True: bridge_daemon._connect(path))
    try:
        server.prewarm({"status_service": "/status", "robot_id": "so101"}).join(timeout=3.0)
        assert [bridge.status_calls for bridge in _FakeBridge.instances] == [1]
        client = bridge_daemon.ResidentBridge(status_service="/status", robot_id="so101", caller_token="t")
        assert client.start() is True
        client.close()
        assert len(_FakeBridge.instances) == 1
        # idle_timeout_sec=0 never exits on its own.
        time.sleep(0.2)
        assert thread.is_alive()
    finally:
        server.request_stop()
        thread.join(timeout=3.0)
    assert not thread.is_alive()


def test_replace_takes_the_socket_over_from_a_running_daemon(tmp_path) -> None:
    path = tmp_path / "replace.sock"
    old, old_thread = _serve(path, _FakeBridge)
    bridge_daemon._replace_running_daemon(path, timeout_sec=3.0)
    old_thread.join(timeout=3.0)
    assert not old_thread.is_alive()
    new, new_thread = _serve(path, _FakeBridge)
    try:
        bridge_daemon._connect(path).close()
    finally:
        new.request_stop()
        new_thread.join(timeout=3.0)


def test_warm_up_stops_probing_when_the_daemon_is_stopping() -> None:
    class _NeverAnswers(_FakeBridge):
        lost_status_responses = 10**6

    stop = threading.Event()
    stop.set()
    started = time.monotonic()
    assert bridge_daemon.BridgeDaemon._warm_up(_NeverAnswers(), 60.0, stop) is False
    assert time.monotonic() - started < 1.0


def test_ros_bridge_caller_view_shares_clients_and_never_closes_the_shared_node() -> None:
    from robot_skill_cli.ros_bridge import RosBridge

    base = RosBridge(status_service="/status", validate_skill_service="/validate", skill_action="/skill")
    sentinel = object()
    base._node = sentinel
    view = base.for_caller("token-b")
    assert view._caller_token == "token-b"
    assert base._caller_token == ""
    assert view._node is sentinel
    view.close()
    assert base._node is sentinel


def test_daemon_rejects_unknown_parameters_methods_and_caller_identity_fields(daemon) -> None:
    connection = bridge_daemon._Connection(bridge_daemon._connect(daemon.path))
    try:
        for bridge in ({"status_service": "/s", "caller_token_env": "IBROBOT_EXECUTION_TOKEN_X"}, {"bogus": "x"}):
            with pytest.raises(ValueError, match="unsupported"):
                connection.request({"op": "open", "version": bridge_daemon.PROTOCOL_VERSION, "bridge": bridge})
        connection.request(
            {"op": "open", "version": bridge_daemon.PROTOCOL_VERSION, "bridge": {"status_service": "/status"}}
        )
        for method in ("close", "start", "for_caller", "_call_service"):
            with pytest.raises(ValueError, match="not available remotely"):
                connection.call(method, (), {})
    finally:
        connection.close()
    assert not hasattr(bridge_daemon.ResidentBridge(status_service="/s"), "for_caller")


def test_daemon_drops_private_environment_it_inherited(monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI", "secret")
    monkeypatch.setenv("IBROBOT_STOP_FILE", "/x")
    monkeypatch.setenv("ROS_DOMAIN_ID", "52")
    bridge_daemon._scrub_private_environment()
    assert "IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI" not in os.environ
    assert "IBROBOT_STOP_FILE" not in os.environ
    assert os.environ["ROS_DOMAIN_ID"] == "52"


def test_peer_without_credentials_is_refused() -> None:
    class _NoCreds:
        def getsockopt(self, *_args):
            raise OSError("unsupported")

    assert bridge_daemon._peer_uid(_NoCreds()) is None


def test_stopping_daemon_keeps_serving_a_connected_client_until_it_leaves(tmp_path, monkeypatch) -> None:
    # A stop or --replace must not cut off a command that is still running.
    _FakeBridge.instances.clear()
    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "1")
    path = tmp_path / "drain.sock"
    server, thread = _serve(path, _FakeBridge)
    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", lambda spawn=True: bridge_daemon._connect(path))
    client = bridge_daemon.ResidentBridge(status_service="/status")
    assert client.start()
    bridge_daemon._replace_running_daemon(path, timeout_sec=3.0)
    assert not path.exists()  # released for the replacing daemon
    assert thread.is_alive()
    assert client.get_status(timeout_sec=1.0)["control_plane_ready"] is True
    client.close()
    thread.join(timeout=3.0)
    assert not thread.is_alive()


def test_lost_daemon_fails_goal_futures_and_recovers_task_id_calls_in_process(tmp_path, monkeypatch) -> None:
    """Killing the daemon mid-goal must still let the command cancel by task id."""
    _FakeBridge.instances.clear()
    monkeypatch.setenv(bridge_daemon.DISABLE_ENV, "1")
    path = tmp_path / "lost.sock"
    server, thread = _serve(path, _FakeBridge)
    monkeypatch.setattr(bridge_daemon, "connect_or_spawn", lambda spawn=True: bridge_daemon._connect(path))
    local_calls = []

    class _LocalBridge(_FakeBridge):
        def cancel_task(self, task_id, *, timeout_sec):
            local_calls.append(("cancel_task", task_id))
            return {"accepted": True}

    monkeypatch.setattr("robot_skill_cli.ros_bridge.RosBridge", _LocalBridge)
    monkeypatch.setattr(bridge_daemon, "FALLBACK_WARM_UP_SEC", 0.1)

    def never_finishing_goal(self, payload, *, task_id, feedback_callback=None):
        handle = _Handle(None)
        self.handles.append(handle)
        future = _Future()
        future.set_result(handle)
        return future

    monkeypatch.setattr(_FakeBridge, "send_skill_goal", never_finishing_goal)
    client = bridge_daemon.ResidentBridge(status_service="/status", caller_token="tok")
    try:
        assert client.start()
        goal = client.send_skill_goal({"skill_name": "wave_hello"}, task_id="t-9")
        assert client.wait_future(goal, timeout_sec=2.0)
        result_future = goal.result().get_result_async()
        # The daemon disappears while the goal runs.
        server.request_stop()
        server.shutdown()
        thread.join(timeout=3.0)
        connection = client._connection
        connection._sock.shutdown(socket.SHUT_RDWR)
        deadline = time.monotonic() + 2.0
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert result_future.done()
        with pytest.raises(BridgeError) as error:
            result_future.result()
        assert error.value.code == "ROS_UNAVAILABLE"
        # cancel_goal reports the finished future; the caller then cancels by id.
        assert client.cancel_goal(goal.result(), result_future, timeout_sec=1.0) is True
        assert client.cancel_task("t-9", timeout_sec=1.0) == {"accepted": True}
        assert local_calls == [("cancel_task", "t-9")]
        assert client._fallback.kwargs["caller_token"] == "tok"
        # Non-recoverable calls fail instead of silently starting new goals in-process.
        with pytest.raises(BridgeError):
            client.send_skill_goal({"skill_name": "nod_yes"}, task_id="t-10")
    finally:
        client.close()


def test_open_does_not_wait_for_a_gateway_that_never_answers(tmp_path) -> None:
    class _Silent(_FakeBridge):
        lost_status_responses = 10**9

    server = bridge_daemon.BridgeDaemon(tmp_path / "x.sock", bridge_factory=_Silent, open_warm_wait_sec=0.2)
    started = time.monotonic()
    bridge, error = server.bridge_for({"status_service": "/status"})
    assert bridge is not None and error == ""
    assert time.monotonic() - started < 1.0
    server.shutdown()
