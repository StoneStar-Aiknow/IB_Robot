"""Diagnostic publication with real ROS message types, without the AimDK SDK."""

from types import SimpleNamespace

import pytest
from aimdk_robot.runtime_node import AimdkRuntimeNode
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus
from rclpy.serialization import deserialize_message, serialize_message


@pytest.mark.parametrize(
    "level, expected",
    [
        (DiagnosticStatus.OK, b"\x00"),
        (DiagnosticStatus.WARN, b"\x01"),
        (DiagnosticStatus.ERROR, b"\x02"),
        (DiagnosticStatus.STALE, b"\x03"),
        (0, b"\x00"),
        (2, b"\x02"),
    ],
)
def test_diagnostics_accept_ros_constants_and_integer_levels(level, expected):
    """Humble byte constants and integer callers must both produce valid wire messages."""
    published = []
    stamp = Time(sec=123)
    clock = SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: stamp))
    node = SimpleNamespace(
        get_clock=lambda: clock,
        _state=SimpleNamespace(name="aimdk_robot"),
        _diag_pub=SimpleNamespace(publish=published.append),
    )

    AimdkRuntimeNode._publish_diagnostics(node, [("fsm_state=4", "4", level)], prefix="aimdk/fsm_state")

    assert len(published) == 1
    array = deserialize_message(serialize_message(published[0]), type(published[0]))
    assert array.header.stamp == stamp
    assert len(array.status) == 1
    status = array.status[0]
    assert status.level == expected
    assert status.name == "aimdk/fsm_state/4"
    assert status.hardware_id == "aimdk_robot"
    assert status.message == "fsm_state=4"
    assert [(entry.key, entry.value) for entry in status.values] == [("code", "4")]
