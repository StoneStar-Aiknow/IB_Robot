import math

import pytest

from ibrobot_msgs.action import ExecuteNavigation
from robot_navigation import nav_cmd
from robot_navigation.nav_cmd import COMMAND_TYPES, _parser
from robot_navigation.navigation_command_core import (
    CommandType,
    GoalValidationError,
    resolve_navigation_target,
)


@pytest.mark.parametrize(
    ("command", "expected_type"),
    [
        ("leftward", ExecuteNavigation.Goal.STRAFE_LEFT),
        ("rightward", ExecuteNavigation.Goal.STRAFE_RIGHT),
    ],
)
def test_lateral_cli_commands_are_direct_and_unambiguous(command, expected_type):
    parsed = _parser().parse_args(["--action-name", "/robot/navigation/execute", command, "0.10"])

    assert parsed.command == command
    assert parsed.value == pytest.approx(0.10)
    assert COMMAND_TYPES[command] == expected_type


@pytest.mark.parametrize("retired_command", ["strafe-left", "strafe-right"])
def test_retired_lateral_cli_commands_are_rejected(retired_command):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--action-name", "/robot/navigation/execute", retired_command, "0.10"])


def test_cli_requires_explicit_navigation_action_name():
    with pytest.raises(SystemExit):
        _parser().parse_args(["status"])

    parsed = _parser().parse_args(["--action-name", "/robot/navigation/execute", "status"])

    assert parsed.action_name == "/robot/navigation/execute"


def test_status_uses_explicit_navigation_action_name(monkeypatch):
    action_names = []

    class _Node:
        def get_logger(self):
            return self

        def error(self, _message):
            pass

        def destroy_node(self):
            pass

    class _ActionClient:
        def __init__(self, _node, _action_type, action_name):
            action_names.append(action_name)

        def wait_for_server(self, *, timeout_sec):
            return False

    monkeypatch.setattr(nav_cmd.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(nav_cmd.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(nav_cmd, "Node", lambda _name: _Node())
    monkeypatch.setattr(nav_cmd, "ActionClient", _ActionClient)

    result = nav_cmd.main(["--action-name", "/robot/navigation/execute", "status"])

    assert result == 0
    assert action_names[0] == "/robot/navigation/execute"


def test_absolute_goal_preserves_map_pose():
    assert resolve_navigation_target(
        command_type=CommandType.ABSOLUTE_POSE,
        value=0.0,
        target_frame="map",
        target_x=1.2,
        target_y=-0.4,
        target_yaw=0.75,
        base_x=0.0,
        base_y=0.0,
        base_yaw=0.0,
    ) == pytest.approx((1.2, -0.4, 0.75))


@pytest.mark.parametrize(
    ("command_type", "value", "expected"),
    [
        (CommandType.FORWARD, 1.0, (2.0, 4.0, math.pi / 2.0)),
        (CommandType.BACKWARD, 1.0, (2.0, 2.0, math.pi / 2.0)),
        (CommandType.STRAFE_LEFT, 1.0, (1.0, 3.0, math.pi / 2.0)),
        (CommandType.STRAFE_RIGHT, 1.0, (3.0, 3.0, math.pi / 2.0)),
        (CommandType.TURN_LEFT, math.pi / 2.0, (2.0, 3.0, math.pi)),
        (CommandType.TURN_RIGHT, math.pi / 2.0, (2.0, 3.0, 0.0)),
    ],
)
def test_relative_goal_uses_current_robot_pose(command_type, value, expected):
    assert resolve_navigation_target(
        command_type=command_type,
        value=value,
        target_frame="",
        target_x=0.0,
        target_y=0.0,
        target_yaw=0.0,
        base_x=2.0,
        base_y=3.0,
        base_yaw=math.pi / 2.0,
    ) == pytest.approx(expected)


def test_large_turn_uses_existing_shortest_final_heading_normalization():
    assert resolve_navigation_target(
        command_type=CommandType.TURN_LEFT,
        value=5.0 * math.pi / 2.0,
        target_frame="",
        target_x=0.0,
        target_y=0.0,
        target_yaw=0.0,
        base_x=0.0,
        base_y=0.0,
        base_yaw=0.0,
    ) == pytest.approx((0.0, 0.0, math.pi / 2.0))


@pytest.mark.parametrize(
    ("command_type", "value", "target_frame"),
    [
        (CommandType.ABSOLUTE_POSE, 0.0, "odom"),
        (CommandType.FORWARD, 0.0, ""),
        (99, 1.0, ""),
    ],
)
def test_invalid_goal_is_rejected_before_nav2(command_type, value, target_frame):
    with pytest.raises(GoalValidationError):
        resolve_navigation_target(
            command_type=command_type,
            value=value,
            target_frame=target_frame,
            target_x=0.0,
            target_y=0.0,
            target_yaw=0.0,
            base_x=0.0,
            base_y=0.0,
            base_yaw=0.0,
        )
