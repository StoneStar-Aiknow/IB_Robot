"""Tests for the timeout-aware ros2_control controller spawner."""

from types import SimpleNamespace

import pytest

from robot_config import controller_spawner


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class _Node:
    def __init__(self):
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def _controllers(*states):
    return SimpleNamespace(controller=[SimpleNamespace(name=name, state=state) for name, state in states])


def test_spawn_controller_forwards_explicit_timeouts(monkeypatch):
    calls = []
    monkeypatch.setattr(controller_spawner, "list_controllers", lambda *args: _controllers())

    def load(*args):
        calls.append(("load", args))
        return SimpleNamespace(ok=True)

    def configure(*args):
        calls.append(("configure", args))
        return SimpleNamespace(ok=True)

    def switch(*args):
        calls.append(("switch", args))
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(controller_spawner, "load_controller", load)
    monkeypatch.setattr(controller_spawner, "configure_controller", configure)
    monkeypatch.setattr(controller_spawner, "switch_controllers", switch)

    node = _Node()
    controller_spawner.spawn_controller(node, "/controller_manager", "joint_state_broadcaster", 41.0, 42.0, 43.0)

    assert calls[0][1][-2:] == (41.0, 42.0)
    assert calls[1][1][-2:] == (41.0, 42.0)
    assert calls[2][1][-2:] == (43.0, 42.0)


def test_spawn_controller_accepts_already_active_controller(monkeypatch):
    monkeypatch.setattr(
        controller_spawner,
        "list_controllers",
        lambda *args: _controllers(("arm_controller", "active")),
    )
    monkeypatch.setattr(
        controller_spawner,
        "load_controller",
        lambda *args: pytest.fail("active controller must not be loaded again"),
    )
    monkeypatch.setattr(
        controller_spawner,
        "configure_controller",
        lambda *args: pytest.fail("active controller must not be configured again"),
    )
    monkeypatch.setattr(
        controller_spawner,
        "switch_controllers",
        lambda *args: pytest.fail("active controller must not be activated again"),
    )

    controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)


def test_spawn_controller_recovers_when_timed_out_load_completed(monkeypatch):
    states = iter([_controllers(), _controllers(("arm_controller", "inactive"))])
    monkeypatch.setattr(controller_spawner, "list_controllers", lambda *args: next(states))
    monkeypatch.setattr(
        controller_spawner,
        "load_controller",
        lambda *args: (_ for _ in ()).throw(RuntimeError("response timeout")),
    )
    monkeypatch.setattr(
        controller_spawner,
        "configure_controller",
        lambda *args: pytest.fail("inactive controller must not be configured again"),
    )
    activated = []
    monkeypatch.setattr(
        controller_spawner,
        "switch_controllers",
        lambda *args: activated.append(args) or SimpleNamespace(ok=True),
    )

    controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)

    assert len(activated) == 1


def test_spawn_controller_recovers_when_timed_out_configure_completed(monkeypatch):
    states = iter(
        [
            _controllers(("arm_controller", "unconfigured")),
            _controllers(("arm_controller", "inactive")),
        ]
    )
    monkeypatch.setattr(controller_spawner, "list_controllers", lambda *args: next(states))
    monkeypatch.setattr(
        controller_spawner,
        "configure_controller",
        lambda *args: (_ for _ in ()).throw(RuntimeError("response timeout")),
    )
    activated = []
    monkeypatch.setattr(
        controller_spawner,
        "switch_controllers",
        lambda *args: activated.append(args) or SimpleNamespace(ok=True),
    )

    controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)

    assert len(activated) == 1


def test_spawn_controller_recovers_when_timed_out_switch_completed(monkeypatch):
    states = iter(
        [
            _controllers(("arm_controller", "inactive")),
            _controllers(("arm_controller", "active")),
        ]
    )
    monkeypatch.setattr(controller_spawner, "list_controllers", lambda *args: next(states))
    monkeypatch.setattr(
        controller_spawner,
        "switch_controllers",
        lambda *args: (_ for _ in ()).throw(RuntimeError("response timeout")),
    )

    controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)


def test_spawn_controller_rejects_failed_load(monkeypatch):
    monkeypatch.setattr(controller_spawner, "list_controllers", lambda *args: _controllers())
    monkeypatch.setattr(controller_spawner, "load_controller", lambda *args: SimpleNamespace(ok=False))

    with pytest.raises(RuntimeError, match="Failed to load"):
        controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)


@pytest.mark.parametrize("failed_operation", ["configure", "activate"])
def test_spawn_controller_rejects_lifecycle_failure(monkeypatch, failed_operation):
    monkeypatch.setattr(
        controller_spawner,
        "list_controllers",
        lambda *args: _controllers(("arm_controller", "unconfigured")),
    )
    monkeypatch.setattr(
        controller_spawner,
        "configure_controller",
        lambda *args: SimpleNamespace(ok=failed_operation != "configure"),
    )
    monkeypatch.setattr(
        controller_spawner,
        "switch_controllers",
        lambda *args: SimpleNamespace(ok=failed_operation != "activate"),
    )

    with pytest.raises(RuntimeError, match=f"Failed to {failed_operation}"):
        controller_spawner.spawn_controller(_Node(), "/controller_manager", "arm_controller", 10.0, 20.0, 30.0)
