import pytest

from skill_library.resolver import direction_to_delta, resolve_skill_primitives

SKILL_TEMPLATES = {
    "pick_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "pregrasp_pose"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"},
            {"primitive_name": "close_gripper"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "lift_pose"},
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
}


def test_resolve_pick_sequence():
    primitives = resolve_skill_primitives(
        "pick_named_target",
        "demo_object",
        "",
        "",
        0.0,
        {
            "demo_object": {
                "pregrasp_pose": "demo_pregrasp",
                "hover_pose": "demo_hover",
                "grasp_pose": "demo_grasp",
                "lift_pose": "demo_lift",
            }
        },
        1.0,
        0.15,
        SKILL_TEMPLATES,
        None,
    )
    assert [primitive.primitive_name for primitive in primitives] == [
        "move_to_named_pose",
        "move_to_named_pose",
        "close_gripper",
        "move_to_named_pose",
    ]
    assert primitives[2].gripper_position == 0.15


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
