from safety_guard.rules import validate_primitive_request, validate_skill_request

SKILL_TEMPLATES = {
    "hover_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "hover_pose"},
        ]
    },
    "release_at_named_pose": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "place_name_from_request": True},
            {"primitive_name": "open_gripper"},
        ]
    },
    "pick_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "pregrasp_pose"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"},
            {"primitive_name": "close_gripper"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "lift_pose"},
        ]
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


def test_validate_move_pose_inside_workspace():
    allowed, reason = validate_primitive_request(
        "move_to_named_pose",
        "home",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        {"home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}}},
        {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4]},
    )
    assert allowed
    assert reason == ""


def test_validate_skill_requires_target():
    allowed, reason = validate_skill_request(
        "pick_named_target",
        "missing_target",
        "",
        "",
        0.0,
        {"home": {}, "observe_table": {}, "tray_right": {}},
        {"demo_object": {}},
        SKILL_TEMPLATES,
    )
    assert not allowed
    assert "unknown target" in reason


def test_validate_skill_uses_default_templates_without_override():
    allowed, reason = validate_skill_request(
        "inspect_scene",
        "",
        "",
        "",
        0.0,
        {"observe_table": {}},
        {},
        None,
    )
    assert allowed
    assert reason == ""


def test_validate_relative_skill_direction():
    allowed, reason = validate_skill_request(
        "move_relative_ee",
        "",
        "",
        "forward",
        0.03,
        {"home": {}, "observe_table": {}, "tray_right": {}},
        {"demo_object": {}},
        SKILL_TEMPLATES,
    )
    assert allowed
    assert reason == ""


def test_validate_default_gripper_rotation_skill():
    allowed, reason = validate_skill_request(
        "rotate_gripper_cw",
        "",
        "",
        "",
        30.0,
        {"home": {}},
        {},
        None,
    )
    assert allowed
    assert reason == ""


def test_validate_relative_primitive_target_workspace():
    allowed, reason = validate_primitive_request(
        "move_relative_ee",
        "",
        0.03,
        0.0,
        0.0,
        0.25,
        0.0,
        0.2,
        0.0,
        {"home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}}},
        {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4]},
    )
    assert allowed
    assert reason == ""


def test_validate_hover_skill_requires_target_pose_key():
    allowed, reason = validate_skill_request(
        "hover_named_target",
        "demo_object",
        "",
        "",
        0.0,
        {"home": {}, "observe_table": {}, "tray_right": {}, "demo_hover": {}},
        {"demo_object": {"hover_pose": "demo_hover"}},
        SKILL_TEMPLATES,
    )
    assert allowed
    assert reason == ""


def test_validate_release_skill_requires_place_name():
    allowed, reason = validate_skill_request(
        "release_at_named_pose",
        "",
        "tray_right",
        "",
        0.0,
        {"home": {}, "observe_table": {}, "tray_right": {}},
        {"demo_object": {}},
        SKILL_TEMPLATES,
    )
    assert allowed
    assert reason == ""
