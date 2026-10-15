import math

import pytest

from ibrobot_msgs.action import ExecuteNavigation
from safety_guard.rules import validate_primitive_request, validate_skill_request

NAVIGATION_TEMPLATES = {
    "nav_straight": {
        "primitive_sequence": [
            {
                "primitive_name": "nav_straight",
                "direction_from_request": True,
                "distance_from_request": True,
            }
        ]
    },
    "nav_turn": {
        "primitive_sequence": [
            {
                "primitive_name": "nav_turn",
                "direction_from_request": True,
                "degree_from_request": True,
            }
        ]
    },
    "nav_abs_coordinate": {
        "primitive_sequence": [
            {
                "primitive_name": "nav_abs_coordinate",
                "x_from_request": True,
                "y_from_request": True,
                "yaw_from_request": True,
            }
        ]
    },
}


def _validate_navigation_primitive(primitive_name, **overrides):
    request = {
        "primitive_name": primitive_name,
        "pose_name": "",
        "relative_dx": 0.0,
        "relative_dy": 0.0,
        "relative_dz": 0.0,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "gripper_position": 0.0,
        "named_poses": {},
        "workspace": {},
        "navigation_command_type": ExecuteNavigation.Goal.ABSOLUTE_POSE,
        "navigation_target_frame": "map",
        "navigation_target_x": 0.0,
        "navigation_target_y": 0.0,
        "navigation_target_z": 0.0,
        "navigation_target_qx": 0.0,
        "navigation_target_qy": 0.0,
        "navigation_target_qz": 0.0,
        "navigation_target_qw": 1.0,
        "navigation_value": 0.0,
    }
    request.update(overrides)
    return validate_primitive_request(**request)


@pytest.mark.parametrize(
    ("primitive_name", "command_type", "value"),
    [
        ("nav_straight", ExecuteNavigation.Goal.FORWARD, 1.0),
        ("nav_straight", ExecuteNavigation.Goal.STRAFE_LEFT, 1.0),
        ("nav_turn", ExecuteNavigation.Goal.TURN_LEFT, 5.0 * math.pi / 2.0),
        ("nav_turn", ExecuteNavigation.Goal.TURN_RIGHT, math.pi / 2.0),
    ],
)
def test_relative_navigation_primitive_accepts_positive_finite_values(primitive_name, command_type, value):
    assert _validate_navigation_primitive(
        primitive_name,
        navigation_command_type=command_type,
        navigation_value=value,
    ) == (True, "")


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_relative_navigation_primitive_rejects_non_positive_or_non_finite_values(value):
    allowed, reason = _validate_navigation_primitive(
        "nav_straight",
        navigation_command_type=ExecuteNavigation.Goal.FORWARD,
        navigation_value=value,
    )

    assert not allowed
    assert "positive finite" in reason


def test_navigation_primitive_rejects_command_type_mismatch():
    allowed, reason = _validate_navigation_primitive(
        "nav_turn",
        navigation_command_type=ExecuteNavigation.Goal.FORWARD,
        navigation_value=math.pi / 2.0,
    )

    assert not allowed
    assert "command type" in reason


def test_absolute_navigation_primitive_accepts_map_pose_without_map_limits():
    assert _validate_navigation_primitive(
        "nav_abs_coordinate",
        navigation_target_x=-10000.0,
        navigation_target_y=10000.0,
        navigation_target_qz=-1.0,
        navigation_target_qw=0.0,
    ) == (True, "")


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"navigation_target_frame": "odom"}, "map frame"),
        ({"navigation_target_x": float("nan")}, "finite"),
        ({"navigation_target_qz": 0.0, "navigation_target_qw": 0.0}, "non-zero"),
    ],
)
def test_absolute_navigation_primitive_rejects_invalid_pose(overrides, fragment):
    allowed, reason = _validate_navigation_primitive("nav_abs_coordinate", **overrides)

    assert not allowed
    assert fragment in reason


@pytest.mark.parametrize(
    ("skill_name", "kwargs"),
    [
        ("nav_straight", {"direction": "left", "distance": 1.0}),
        ("nav_turn", {"direction": "right", "degree": 450.0}),
        ("nav_abs_coordinate", {"x": 0.0, "y": -2.5, "yaw": -180.0}),
    ],
)
def test_navigation_skill_rules_accept_public_contract(skill_name, kwargs):
    assert validate_skill_request(
        skill_name,
        "",
        "",
        "",
        0.0,
        {},
        {},
        NAVIGATION_TEMPLATES,
        **kwargs,
    ) == (True, "")
