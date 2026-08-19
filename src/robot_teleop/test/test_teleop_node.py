import importlib
import threading
from types import MethodType, SimpleNamespace

import pytest
from std_msgs.msg import Bool

from robot_teleop.teleop_node import TeleopNode, connect_device_or_raise

teleop_node_module = importlib.import_module("robot_teleop.teleop_node")


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class _Device:
    def __init__(self):
        self.estop_calls = 0
        self.estop_release_calls = 0
        self.is_connected = True

    def emergency_stop(self):
        self.estop_calls += 1

    def emergency_stop_released(self):
        self.estop_release_calls += 1


def _estop_harness():
    logger = _Logger()
    node = SimpleNamespace(
        estop_active=False,
        _estop_state_lock=threading.Lock(),
        _estop_stop_pending=False,
        _estop_release_pending=False,
        _device_lock=threading.Lock(),
        device=_Device(),
        get_logger=lambda: logger,
    )
    node._estop_is_active = MethodType(TeleopNode._estop_is_active, node)
    node._try_dispatch_estop = MethodType(TeleopNode._try_dispatch_estop, node)
    return node, logger


def test_device_connection_failure_aborts_node_startup():
    device = SimpleNamespace(connect=lambda: False)

    with pytest.raises(RuntimeError, match="connection failed"):
        connect_device_or_raise(device)


def test_successful_device_connection_returns_normally():
    device = SimpleNamespace(connect=lambda: True)

    connect_device_or_raise(device)


def test_bool_estop_latches_stops_once_and_releases_explicitly():
    node, logger = _estop_harness()

    TeleopNode.estop_callback(node, Bool(data=True))
    TeleopNode.estop_callback(node, Bool(data=True))

    assert node.estop_active is True
    assert node.device.estop_calls == 1

    TeleopNode.estop_callback(node, Bool(data=False))

    assert node.estop_active is False
    assert node.device.estop_release_calls == 1
    assert any("released" in message for message in logger.warnings)


def test_estop_callback_does_not_block_when_control_loop_owns_device_lock():
    node, _logger = _estop_harness()
    node._device_lock.acquire()
    try:
        TeleopNode.estop_callback(node, Bool(data=True))
        TeleopNode.estop_callback(node, Bool(data=False))

        assert node.estop_active is True
        assert node._estop_stop_pending is True
        assert node._estop_release_pending is True
        assert node.device.estop_calls == 0
    finally:
        node._device_lock.release()

    assert node._try_dispatch_estop() is True
    assert node.device.estop_calls == 1
    assert node.device.estop_release_calls == 1
    assert node.estop_active is False


def test_estop_arriving_during_device_read_discards_inflight_command():
    node, _logger = _estop_harness()
    arm_messages = []
    gripper_messages = []
    node.arm_joint_names = ["1"]
    node.gripper_joint_names = ["6"]
    node.arm_cmd_pub = SimpleNamespace(publish=arm_messages.append)
    node.gripper_cmd_pub = SimpleNamespace(publish=gripper_messages.append)
    node.safety_filter = SimpleNamespace(apply_limits=lambda targets: targets)

    def get_joint_targets():
        TeleopNode.estop_callback(node, Bool(data=True))
        return {"1": 0.5, "6": 0.25}

    node.device.get_joint_targets = get_joint_targets

    TeleopNode.control_loop_callback(node)

    assert node.device.estop_calls == 1
    assert node.estop_active is True
    assert arm_messages == []
    assert gripper_messages == []


def test_main_keeps_ros_alive_until_device_stop_is_acknowledged(monkeypatch):
    events = []

    class _ShutdownNode:
        def __init__(self):
            self.stop_complete = False

        def disconnect_device(self):
            events.append("disconnect")

        def device_shutdown_complete(self):
            return self.stop_complete

        def destroy_node(self):
            events.append("destroy")

        def get_logger(self):
            return SimpleNamespace(error=lambda message: events.append(message))

    node = _ShutdownNode()

    def init(**kwargs):
        events.append(("init", kwargs["signal_handler_options"]))

    def spin(_node):
        raise KeyboardInterrupt

    def spin_once(_node, timeout_sec):
        assert timeout_sec == 0.05
        events.append("spin_once")
        node.stop_complete = True

    monkeypatch.setattr(teleop_node_module, "TeleopNode", lambda: node)
    monkeypatch.setattr(teleop_node_module.rclpy, "init", init)
    monkeypatch.setattr(teleop_node_module.rclpy, "spin", spin)
    monkeypatch.setattr(teleop_node_module.rclpy, "spin_once", spin_once)
    monkeypatch.setattr(teleop_node_module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(teleop_node_module.rclpy, "shutdown", lambda: events.append("shutdown"))

    teleop_node_module.main()

    assert events[0] == ("init", teleop_node_module.SignalHandlerOptions.NO)
    assert events.index("disconnect") < events.index("spin_once") < events.index("destroy")
