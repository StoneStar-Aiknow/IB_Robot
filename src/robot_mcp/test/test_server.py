from __future__ import annotations

import asyncio
import importlib.util
import queue
import time
import types
from pathlib import Path

from launch.actions import DeclareLaunchArgument

from robot_mcp import server
from robot_mcp.catalog import Catalog
from robot_mcp.ros_bridge import RosBridge
from robot_mcp.server import _build_arg_parser


class _Bridge:
    ros_available = True

    def __init__(self) -> None:
        self.validate_calls = 0
        self.wait_calls = 0

    def validate_skill(self, **_kwargs):
        self.validate_calls += 1
        return {"allowed": True, "reason": ""}

    def wait_for_skill_server(self, **_kwargs):
        self.wait_calls += 1
        return False


class _FakeFuture:
    def __init__(self, result=None, *, done=False):
        self._callbacks = []
        self._done = done
        self._result = result

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def done(self):
        return self._done

    def result(self):
        return self._result

    def set_result(self, result):
        self._result = result
        self._done = True
        for callback in self._callbacks:
            callback(self)


class _SkillResult:
    def __init__(self, success=True):
        self.success = success
        self.error_code = "" if success else "CANCELED"
        self.message = "skill complete" if success else "skill canceled"
        self.executed_primitives = []


class _ResultEnvelope:
    def __init__(self, result):
        self.result = result


class _FakeGoalHandle:
    accepted = True

    def __init__(self, result_future):
        self._cancel_future = _FakeFuture(done=True)
        self._result_future = result_future
        self.cancel_calls = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self._cancel_future

    def get_result_async(self):
        return self._result_future


class _LifecycleBridge(RosBridge):
    def __init__(self, send_futures):
        super().__init__()
        self._ros_available = True
        self._send_futures = list(send_futures)

    def build_skill_goal(self, **_kwargs):
        return object()

    def send_skill_goal(self, _goal):
        return self._send_futures.pop(0), queue.Queue()

    def wait_for_skill_server(self, **_kwargs):
        return True


def _complete_goal(success=True):
    result_future = _FakeFuture(_ResultEnvelope(_SkillResult(success)), done=True)
    return _FakeFuture(_FakeGoalHandle(result_future), done=True)


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _expiring_clock():
    current = 0.0

    def _clock():
        nonlocal current
        current += 10.0
        return current

    return _clock


def _catalog() -> Catalog:
    return Catalog(robot_name="test", config_path="test.yaml", skills=[{"name": "enabled_skill"}])


def test_http_cli_defaults_to_loopback():
    args = _build_arg_parser().parse_args(["--transport", "streamable-http"])

    assert args.host == "127.0.0.1"


def test_http_launch_defaults_to_loopback():
    launch_path = Path(__file__).parents[1] / "launch" / "robot_mcp.launch.py"
    spec = importlib.util.spec_from_file_location("robot_mcp_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    host_argument = next(
        entity for entity in description.entities if isinstance(entity, DeclareLaunchArgument) and entity.name == "host"
    )

    assert host_argument.default_value[0].text == "127.0.0.1"


def test_validate_skill_rejects_skill_missing_from_catalog(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(server, "_catalog", _catalog())
    monkeypatch.setattr(server, "_bridge", bridge)

    result = server.validate_skill("disabled_demo")

    assert result == {"allowed": False, "reason": "unsupported skill: disabled_demo"}
    assert bridge.validate_calls == 0


def test_execute_skill_rejects_skill_missing_from_catalog(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(server, "_catalog", _catalog())
    monkeypatch.setattr(server, "_bridge", bridge)

    result = asyncio.run(server.execute_skill("disabled_demo"))

    assert result["success"] is False
    assert result["error_code"] == "UNSUPPORTED_SKILL"
    assert result["message"] == "unsupported skill: disabled_demo"
    assert bridge.wait_calls == 0


def test_execute_skill_holds_admission_after_late_goal_acceptance(monkeypatch):
    late_send_future = _FakeFuture()
    late_result_future = _FakeFuture()
    late_goal = _FakeGoalHandle(late_result_future)
    bridge = _LifecycleBridge([late_send_future, _complete_goal()])
    monkeypatch.setattr(server, "_catalog", _catalog())
    monkeypatch.setattr(server, "_bridge", bridge)
    monkeypatch.setattr(server, "_ACCEPT_TIMEOUT_SEC", 0.0, raising=False)
    monkeypatch.setattr(server, "_CANCEL_DRAIN_TIMEOUT_SEC", 0.0, raising=False)
    monkeypatch.setattr(server, "time", types.SimpleNamespace(monotonic=_expiring_clock()))

    timeout = asyncio.run(server.execute_skill("enabled_skill"))

    assert timeout["error_code"] == "ACCEPT_TIMEOUT"
    assert asyncio.run(server.execute_skill("enabled_skill"))["error_code"] == "MOTION_RECOVERY_IN_PROGRESS"
    late_send_future.set_result(late_goal)
    assert _wait_until(lambda: late_goal.cancel_calls == 1)
    late_result_future.set_result(_ResultEnvelope(_SkillResult(success=False)))
    assert _wait_until(lambda: asyncio.run(server.execute_skill("enabled_skill"))["success"] is True)


def test_execute_skill_keeps_recovery_admission_until_timed_out_result_is_terminal(monkeypatch):
    result_future = _FakeFuture()
    goal = _FakeGoalHandle(result_future)
    bridge = _LifecycleBridge([_FakeFuture(goal, done=True), _complete_goal()])
    monkeypatch.setattr(server, "_catalog", _catalog())
    monkeypatch.setattr(server, "_bridge", bridge)
    monkeypatch.setattr(server, "_RESULT_GRACE_SEC", 0.0)
    monkeypatch.setattr(server, "_CANCEL_DRAIN_TIMEOUT_SEC", 0.0, raising=False)
    monkeypatch.setattr(server, "time", types.SimpleNamespace(monotonic=_expiring_clock()))

    timeout = asyncio.run(server.execute_skill("enabled_skill", timeout_sec=0.0))

    assert timeout["error_code"] == "CANCEL_CLEANUP_TIMEOUT"
    assert goal.cancel_calls == 1
    assert asyncio.run(server.execute_skill("enabled_skill"))["error_code"] == "MOTION_RECOVERY_IN_PROGRESS"
    result_future.set_result(_ResultEnvelope(_SkillResult(success=False)))
    assert _wait_until(lambda: asyncio.run(server.execute_skill("enabled_skill"))["success"] is True)
