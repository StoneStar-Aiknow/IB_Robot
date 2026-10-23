import math

import pytest

from ibrobot_msgs.action import ExecuteNavigation
from skill_library.resolver import direction_to_delta, resolve_skill_primitives

SKILL_TEMPLATES = {
    "move_configuration_test": {
        "primitive_sequence": [
            {
                "primitive_name": "move_to_configuration",
                "joint_positions": {"1": 0.1, "2": 0.2},
                "duration_sec": 2.0,
            }
        ]
    },
    "hover_named_target": {
        "initial_gripper_state": "closed",
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "hover_pose"},
        ],
    },
    "release_at_named_pose": {
        "initial_gripper_state": "closed",
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "place_name_from_request": True},
            {"primitive_name": "open_gripper"},
        ],
    },
    "move_relative_ee": {
        "primitive_sequence": [
            {
                "primitive_name": "move_relative_ee",
                "motion_direction_from_request": True,
                "motion_distance_from_request": True,
            }
        ]
    },
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


def test_resolve_move_configuration_sequence():
    primitives = resolve_skill_primitives(
        "move_configuration_test",
        "",
        "",
        "",
        0.0,
        {},
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
        arm_joint_names=["1", "2"],
    )
    assert [primitive.primitive_name for primitive in primitives] == ["move_to_configuration"]
    assert primitives[0].joint_names == ["1", "2"]
    assert primitives[0].joint_positions == [0.1, 0.2]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("forward", (0.03, 0.0, 0.0)),
        ("backward", (-0.03, 0.0, 0.0)),
        ("left", (0.0, 0.03, 0.0)),
        ("right", (0.0, -0.03, 0.0)),
        ("up", (0.0, 0.0, 0.03)),
        ("down", (0.0, 0.0, -0.03)),
    ],
)
def test_direction_to_delta(direction, expected):
    assert direction_to_delta(direction, 0.03) == expected


def test_resolver_rejects_disabled_skill():
    templates = {
        "disabled_skill": {
            "disabled": True,
            "primitive_sequence": [{"primitive_name": "open_gripper"}],
        }
    }

    with pytest.raises(KeyError, match="unsupported skill: disabled_skill"):
        resolve_skill_primitives("disabled_skill", "", "", "", 0.0, {}, 1.0, 0.0, templates)


def test_direction_to_delta_uses_configured_base_mapping():
    mapping = {
        "forward": [0.0, 1.0, 0.0],
        "backward": [0.0, -1.0, 0.0],
        "left": [-1.0, 0.0, 0.0],
        "right": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "down": [0.0, 0.0, -1.0],
    }
    assert direction_to_delta("forward", 0.03, mapping) == (0.0, 0.03, 0.0)
    assert direction_to_delta("left", 0.03, mapping) == (-0.03, 0.0, 0.0)


def test_resolve_relative_motion_sequence():
    primitives = resolve_skill_primitives(
        "move_relative_ee",
        "",
        "",
        "left",
        0.03,
        {},
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
    )
    assert len(primitives) == 1
    assert primitives[0].primitive_name == "move_relative_ee"
    assert primitives[0].relative_dx == 0.0
    assert primitives[0].relative_dy == 0.03
    assert primitives[0].relative_dz == 0.0


def test_resolve_hover_target_sequence():
    primitives = resolve_skill_primitives(
        "hover_named_target",
        "demo_object",
        "",
        "",
        0.0,
        {
            "demo_object": {
                "hover_pose": "demo_hover",
            }
        },
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
    )
    assert [primitive.primitive_name for primitive in primitives] == ["close_gripper", "move_to_named_pose"]
    assert primitives[0].gripper_position == 0.15
    assert primitives[1].pose_name == "demo_hover"


def test_resolve_release_sequence():
    primitives = resolve_skill_primitives(
        "release_at_named_pose",
        "",
        "tray_right",
        "",
        0.0,
        {},
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
    )
    assert [primitive.primitive_name for primitive in primitives] == [
        "close_gripper",
        "move_to_named_pose",
        "open_gripper",
    ]
    assert primitives[0].gripper_position == 0.15
    assert primitives[1].pose_name == "tray_right"


def test_invalid_initial_gripper_state_is_rejected():
    templates = {
        "bad_gripper_state": {
            "initial_gripper_state": "half_open",
            "primitive_sequence": [{"primitive_name": "open_gripper"}],
        }
    }
    with pytest.raises(ValueError, match="initial_gripper_state"):
        resolve_skill_primitives("bad_gripper_state", "", "", "", 0.0, {}, 1.0, 0.15, templates, None)


def test_default_templates_resolve_gripper_rotation_skill():
    primitives = resolve_skill_primitives(
        "rotate_gripper_cw",
        "",
        "",
        "",
        30.0,
        {},
        1.0,
        0.15,
        None,
        None,
    )
    assert [primitive.primitive_name for primitive in primitives] == ["rotate_gripper_cw"]
    assert primitives[0].relative_dz == 30.0


def _resolve_navigation(skill_name, **navigation_parameters):
    goal = resolve_skill_primitives(
        skill_name,
        "",
        "",
        "",
        0.0,
        {},
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
        **navigation_parameters,
    )[0].navigation_goal
    assert goal is not None
    return goal


@pytest.mark.parametrize(
    ("direction", "command_type"),
    [
        ("forward", ExecuteNavigation.Goal.FORWARD),
        ("backward", ExecuteNavigation.Goal.BACKWARD),
        ("left", ExecuteNavigation.Goal.STRAFE_LEFT),
        ("right", ExecuteNavigation.Goal.STRAFE_RIGHT),
    ],
)
def test_resolve_navigation_straight_maps_four_directions(direction, command_type):
    goal = _resolve_navigation("nav_straight", direction=direction, distance=1.25)

    assert goal.command_type == command_type
    assert goal.value == 1.25


@pytest.mark.parametrize(
    ("direction", "command_type"),
    [("left", ExecuteNavigation.Goal.TURN_LEFT), ("right", ExecuteNavigation.Goal.TURN_RIGHT)],
)
def test_resolve_navigation_turn_converts_degrees_to_radians(direction, command_type):
    goal = _resolve_navigation("nav_turn", direction=direction, degree=90.0)

    assert goal.command_type == command_type
    assert goal.value == pytest.approx(math.pi / 2.0)


def test_resolve_navigation_turn_preserves_large_requested_angle():
    goal = _resolve_navigation("nav_turn", direction="left", degree=450.0)

    assert goal.value == pytest.approx(5.0 * math.pi / 2.0)


def test_resolve_absolute_coordinate_preserves_zero_negative_values_and_map_frame():
    goal = _resolve_navigation("nav_abs_coordinate", x=0.0, y=-2.5, yaw=-180.0)

    assert goal.command_type == ExecuteNavigation.Goal.ABSOLUTE_POSE
    assert goal.value == 0.0
    assert goal.target_pose.header.frame_id == "map"
    assert goal.target_pose.pose.position.x == 0.0
    assert goal.target_pose.pose.position.y == -2.5
    assert goal.target_pose.pose.orientation.z == pytest.approx(-1.0)
    assert goal.target_pose.pose.orientation.w == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("missing_field", ["x", "y", "yaw"])
def test_resolve_absolute_coordinate_requires_all_values(missing_field):
    values: dict[str, float | None] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    values[missing_field] = None

    with pytest.raises(ValueError, match=missing_field):
        _resolve_navigation("nav_abs_coordinate", **values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("skill_name", "parameters", "field_name"),
    [
        ("nav_straight", {"direction": "forward", "distance": None}, "distance"),
        ("nav_turn", {"direction": "left", "degree": None}, "degree"),
        ("nav_abs_coordinate", {"x": None, "y": 0.0, "yaw": 0.0}, "x"),
        ("nav_abs_coordinate", {"x": 0.0, "y": None, "yaw": 0.0}, "y"),
        ("nav_abs_coordinate", {"x": 0.0, "y": 0.0, "yaw": None}, "yaw"),
    ],
)
def test_resolve_navigation_rejects_non_finite_values(skill_name, parameters, field_name, value):
    parameters[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        _resolve_navigation(skill_name, **parameters)


@pytest.mark.parametrize(
    ("skill_name", "parameters"),
    [("nav_straight", {"direction": "forward", "distance": 0.0}), ("nav_turn", {"direction": "left", "degree": -1.0})],
)
def test_resolve_relative_navigation_requires_positive_magnitude(skill_name, parameters):
    with pytest.raises(ValueError, match="positive"):
        _resolve_navigation(skill_name, **parameters)
