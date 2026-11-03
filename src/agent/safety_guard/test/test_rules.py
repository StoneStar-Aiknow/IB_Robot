import os

import pytest
import rclpy

from embodied_common.primitive_contracts import GRIPPER_APPROACH_EXTENSION, primitive_contract_for_version
from ibrobot_msgs.srv import ValidateSkill
from safety_guard.rules import validate_primitive_request, validate_skill_request
from safety_guard.safety_guard_node import SafetyGuardNode

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
    "pick_object": {
        "executor": "grasp_pipeline",
        "required_args": ["target_name"],
        "timeout_sec": 180.0,
    },
    "place_in_container": {
        "executor": "placement_pipeline",
        "required_args": ["target_name", "container_name"],
        "timeout_sec": 120.0,
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


def _assert_test_ros_environment() -> None:
    expected_domain = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert expected_domain is not None
    assert expected_domain.isdigit()
    assert 1 <= int(expected_domain) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == expected_domain
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    _assert_test_ros_environment()
    rclpy.init()
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


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


def test_validate_named_pose_rejects_non_numeric_coordinate():
    try:
        allowed, reason = validate_primitive_request(
            "move_to_named_pose",
            "bad_pose",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            {"bad_pose": {"position": {"x": "not-a-number", "y": 0.0, "z": 0.0}}},
            {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4]},
        )
    except (TypeError, ValueError) as exc:
        pytest.fail(f"validation must fail closed instead of raising: {exc}")

    assert not allowed
    assert "pose x must be finite" in reason


def test_validate_pick_object_requires_target_query():
    allowed, reason = validate_skill_request(
        "pick_object",
        "",
        "",
        "",
        0.0,
        {"home": {}, "observe_table": {}, "tray_right": {}},
        {"demo_object": {}},
        SKILL_TEMPLATES,
    )
    assert not allowed
    assert "target_name is required" in reason


def test_validate_pick_object_accepts_runtime_target_query():
    allowed, reason = validate_skill_request(
        "pick_object",
        "banana",
        "",
        "",
        0.0,
        {"home": {}},
        {},
        SKILL_TEMPLATES,
    )
    assert allowed
    assert reason == ""


def test_validate_place_in_container_accepts_runtime_target_query():
    allowed, reason = validate_skill_request(
        "place_in_container",
        "marker",
        "",
        "",
        0.0,
        {},
        {},
        SKILL_TEMPLATES,
        container_name="black bowl",
    )
    assert allowed
    assert reason == ""


def test_validate_place_in_container_requires_container_query():
    allowed, reason = validate_skill_request(
        "place_in_container",
        "marker",
        "",
        "",
        0.0,
        {},
        {},
        SKILL_TEMPLATES,
    )

    assert not allowed
    assert reason == "container_name is required"


def test_validate_skill_rejects_unknown_delegated_executor():
    allowed, reason = validate_skill_request(
        "unknown_delegated_skill",
        "marker",
        "",
        "",
        0.0,
        {},
        {},
        {
            "unknown_delegated_skill": {
                "executor": "unknown_pipeline",
                "required_args": ["target_name"],
            }
        },
    )
    assert not allowed
    assert reason == "unsupported skill executor: unknown_pipeline"


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


def test_validate_skill_rejects_default_skill_with_explicitly_empty_templates():
    allowed, reason = validate_skill_request(
        "inspect_scene",
        "",
        "",
        "",
        0.0,
        {"observe_table": {}},
        {},
        {},
    )

    assert allowed is False
    assert "unsupported skill" in reason


def test_safety_guard_node_rejects_validation_without_exact_snapshot_identity():
    _assert_test_ros_environment()
    node = SafetyGuardNode()
    try:
        response = node._handle_validate_skill(ValidateSkill.Request(), ValidateSkill.Response())
    finally:
        node.destroy_node()

    assert response.allowed is False
    assert response.error_code == "SKILL_SCHEMA_INVALID"


def test_validate_skill_rejects_disabled_skill():
    templates = {
        "disabled_skill": {
            "disabled": True,
            "primitive_sequence": [{"primitive_name": "open_gripper"}],
        }
    }

    allowed, reason = validate_skill_request(
        "disabled_skill",
        "",
        "",
        "",
        0.0,
        {},
        {},
        templates,
    )

    assert allowed is False
    assert "unsupported skill" in reason


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


def test_validate_sound_following_toggle_direction():
    templates = {
        "sound_following": {
            "executor": "sound_following",
            "required_args": ["motion_direction"],
            "capability": {"schema_version": 2},
        }
    }

    allowed, reason = validate_skill_request(
        "sound_following",
        "",
        "",
        "forward",
        0.0,
        {},
        {},
        templates,
        schema_version=2,
    )

    assert allowed
    assert reason == ""


def _runtime_named_motion_templates():
    return {
        "handshake": {
            "executor": "runtime_named_motion",
            "binding": {"motion": "handshake"},
            "required_args": [],
            "capability": {"schema_version": 1},
        }
    }


def test_validate_runtime_named_motion_accepts_a_request_without_arguments():
    allowed, reason = validate_skill_request(
        "handshake", "", "", "", 0.0, {}, {}, _runtime_named_motion_templates(), schema_version=1
    )

    assert allowed
    assert reason == ""


@pytest.mark.parametrize(
    ("overrides", "rejected"),
    [
        ({"target_name": "cup"}, "target_name not accepted by handshake"),
        ({"arm_side": "left"}, "arm_side is not accepted by handshake"),
        ({"motion_direction": "up"}, "motion parameters are not accepted by handshake"),
        ({"place_name": "home"}, "place_name is not accepted by handshake"),
        ({"direction": "forward", "distance": 0.5}, "direction, distance not accepted by handshake"),
        ({"x": 1.0, "yaw": 0.3}, "x, yaw not accepted by handshake"),
    ],
)
def test_validate_runtime_named_motion_rejects_every_caller_argument(overrides, rejected):
    request = {
        "skill_name": "handshake",
        "target_name": "",
        "place_name": "",
        "motion_direction": "",
        "motion_distance": 0.0,
        "named_poses": {},
        "named_targets": {},
        "skill_templates": _runtime_named_motion_templates(),
        "schema_version": 1,
        **overrides,
    }

    allowed, reason = validate_skill_request(**request)

    assert not allowed
    assert reason == rejected


def test_validate_relative_skill_rejects_non_finite_distance():
    allowed, reason = validate_skill_request(
        "move_relative_ee",
        "",
        "",
        "forward",
        float("nan"),
        {"home": {}, "observe_table": {}, "tray_right": {}},
        {"demo_object": {}},
        SKILL_TEMPLATES,
    )

    assert not allowed
    assert "motion_distance must be finite" in reason


def test_validate_skill_rejects_non_finite_primitive_duration():
    templates = {
        "bad_duration": {
            "primitive_sequence": [
                {
                    "primitive_name": "move_to_joint_positions",
                    "joint_positions": {"1": 0.0},
                    "duration_sec": float("nan"),
                }
            ]
        }
    }

    allowed, reason = validate_skill_request(
        "bad_duration",
        "",
        "",
        "",
        0.0,
        {},
        {},
        templates,
    )

    assert not allowed
    assert "duration_sec must be finite" in reason


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


@pytest.mark.parametrize(
    ("angle", "fragment"),
    [
        (float("nan"), "rotation angle must be finite"),
        (float("inf"), "rotation angle must be finite"),
        (-30.0, "rotation angle must be non-negative"),
        (0.0, "rotation angle must be greater than zero"),
    ],
)
def test_validate_gripper_rotation_rejects_invalid_angle(angle, fragment):
    allowed, reason = validate_skill_request(
        "rotate_gripper_cw",
        "",
        "",
        "",
        angle,
        {"home": {}},
        {},
        None,
    )
    assert not allowed
    assert fragment in reason

    allowed, reason = validate_skill_request(
        "rotate_gripper_ccw",
        "",
        "",
        "",
        angle,
        {"home": {}},
        {},
        None,
    )
    assert not allowed
    assert fragment in reason


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


@pytest.mark.parametrize(
    ("primitive_name", "overrides"),
    [
        ("move_relative_ee", {"relative_dx": float("inf")}),
        ("move_relative_ee", {"target_x": float("nan")}),
        (
            "move_to_joint_positions",
            {
                "joint_names": ["1"],
                "joint_positions": [float("nan")],
                "arm_joint_names": ["1"],
                "joint_limits": {"1": {"min": -1.0, "max": 1.0}},
                "primitive_duration_sec": 0.4,
            },
        ),
        (
            "move_to_joint_positions",
            {
                "joint_names": ["1"],
                "joint_positions": [0.0],
                "arm_joint_names": ["1"],
                "joint_limits": {"1": {"min": -1.0, "max": 1.0}},
                "primitive_duration_sec": float("nan"),
            },
        ),
        (
            "move_through_joint_positions",
            {
                "joint_names": ["1"],
                "joint_waypoints": [float("inf")],
                "joint_waypoint_count": 1,
                "arm_joint_names": ["1"],
                "joint_limits": {"1": {"min": -1.0, "max": 1.0}},
                "waypoint_duration_sec": 0.1,
            },
        ),
        (
            "move_through_joint_positions",
            {
                "joint_names": ["1"],
                "joint_waypoints": [0.0],
                "joint_waypoint_count": 1,
                "arm_joint_names": ["1"],
                "joint_limits": {"1": {"min": -1.0, "max": 1.0}},
                "waypoint_duration_sec": float("nan"),
            },
        ),
        ("rotate_gripper_cw", {"relative_dz": float("nan")}),
        ("rotate_gripper_ccw", {"relative_dz": float("inf")}),
        ("open_gripper", {"gripper_position": float("nan")}),
        ("open_gripper", {"gripper_position": 10**1000}),
    ],
)
def test_validate_primitive_rejects_non_finite_numbers(primitive_name, overrides):
    request = {
        "primitive_name": primitive_name,
        "pose_name": "",
        "relative_dx": 0.0,
        "relative_dy": 0.0,
        "relative_dz": 0.0,
        "target_x": 0.2,
        "target_y": 0.0,
        "target_z": 0.2,
        "gripper_position": 0.5,
        "named_poses": {},
        "workspace": {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4]},
    }
    request.update(overrides)

    allowed, reason = validate_primitive_request(**request)

    assert not allowed
    assert "finite" in reason


def test_validate_joint_primitive_rejects_numeric_string_position():
    allowed, reason = validate_primitive_request(
        "move_to_joint_positions",
        "",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        {},
        {},
        joint_names=["1"],
        joint_positions=["0.0"],
        arm_joint_names=["1"],
        joint_limits={"1": {"min": -1.0, "max": 1.0}},
        primitive_duration_sec=0.4,
    )

    assert not allowed
    assert "joint target for 1" in reason


def test_validate_dynamic_pose_requires_normalized_quaternion():
    allowed, reason = validate_primitive_request(
        "move_to_pose",
        "",
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.2,
        0.0,
        {},
        {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4]},
        target_qw=0.0,
    )
    assert not allowed
    assert "quaternion must be non-zero" in reason


def test_validate_dynamic_pose_inside_workspace():
    allowed, reason = validate_primitive_request(
        "move_to_pose",
        "",
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.2,
        0.0,
        {},
        {"x": [0.0, 0.4], "y": [-0.2, 0.2], "z": [0.0, 0.4], "max_radius_m": 0.4},
        target_qw=1.0,
        velocity_scaling=0.05,
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


# --- contract v2: multi-arm robots (arm_groups) and per-robot motion limits ------------------

ARM_GROUPS = {"left": "left_arm", "right": "right_arm"}
DUAL_WORKSPACE = {"x": [-0.7, 0.7], "y": [-0.45, 0.3], "z": [0.77, 1.25]}


def _relative_skill(arm_side, arm_groups=None):
    return validate_skill_request(
        "move_relative_ee", "", "", "down", 0.03, {}, {}, SKILL_TEMPLATES, arm_side=arm_side, arm_groups=arm_groups
    )


def test_single_arm_robot_still_rejects_arm_side():
    allowed, reason = _relative_skill("left")
    assert not allowed and "HRI parameters are not accepted" in reason


@pytest.mark.parametrize("arm_side", ["left", "right", "auto"])
def test_multi_arm_robot_accepts_its_arm_sides(arm_side):
    assert _relative_skill(arm_side, ARM_GROUPS) == (True, "")


def test_multi_arm_robot_rejects_an_arm_it_does_not_have():
    allowed, reason = _relative_skill("middle", ARM_GROUPS)
    assert not allowed and "arm_side must be one of ['left', 'right']" in reason


def _relative_primitive(**kwargs):
    return validate_primitive_request(
        "move_relative_ee", "", 0.0, 0.0, -0.03, -0.3, -0.3, 0.85, 0.0, {}, DUAL_WORKSPACE, **kwargs
    )


def test_primitive_target_group_must_be_one_of_the_robots_groups():
    assert _relative_primitive(target_group="left_arm", arm_groups=ARM_GROUPS) == (True, "")
    allowed, reason = _relative_primitive(target_group="left_arm")
    assert not allowed and "unknown target_group" in reason


def test_motion_limit_caps_velocity_scaling():
    limits = {"max_velocity_scaling": 0.6}
    assert _relative_primitive(velocity_scaling=0.5, motion_limits=limits) == (True, "")
    for scaling in (0.0, 0.7):
        allowed, reason = _relative_primitive(velocity_scaling=scaling, motion_limits=limits)
        assert not allowed and "outside this robot's limit (0, 0.6]" in reason


def test_motion_limit_does_not_apply_to_grippers():
    allowed, _ = validate_primitive_request(
        "close_gripper",
        "",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        {},
        DUAL_WORKSPACE,
        motion_limits={"max_velocity_scaling": 0.6},
    )
    assert allowed


# --- gripper approach presets (set_gripper_approach) --------------------------------------------

GRIPPER_APPROACH = {
    "axis": "x",
    "presets": {
        "horizontal": {"orientation": {"x": 0.0, "y": 0.0, "z": 0.7071068, "w": 0.7071068}},
        "top_down": {"orientation": {"x": -0.5, "y": 0.5, "z": 0.5, "w": 0.5}},
    },
}
APPROACH_TEMPLATES = {
    **SKILL_TEMPLATES,
    "set_gripper_approach": {
        "initial_gripper_state": "none",
        "primitive_sequence": [{"primitive_name": "set_gripper_approach", "approach_from_request": True}],
    },
}
APPROACH_DESCRIPTORS = primitive_contract_for_version(1, (GRIPPER_APPROACH_EXTENSION,)).descriptors


def _approach_skill(approach, skill_name="set_gripper_approach", **kwargs):
    return validate_skill_request(
        skill_name,
        "",
        "",
        kwargs.pop("motion_direction", ""),
        kwargs.pop("motion_distance", 0.0),
        {},
        {},
        APPROACH_TEMPLATES,
        primitive_descriptors=APPROACH_DESCRIPTORS,
        gripper_approach=GRIPPER_APPROACH,
        approach=approach,
        **kwargs,
    )


def test_approach_skill_accepts_only_declared_presets():
    assert _approach_skill("top_down", arm_side="right", arm_groups=ARM_GROUPS) == (True, "")
    allowed, reason = _approach_skill("upside_down")
    assert not allowed and "unknown gripper approach 'upside_down'" in reason
    allowed, reason = _approach_skill("")
    assert not allowed and "unknown gripper approach" in reason


def test_approach_is_refused_by_skills_that_do_not_take_it():
    allowed, reason = _approach_skill(
        "top_down", skill_name="move_relative_ee", motion_direction="down", motion_distance=0.03
    )
    assert not allowed and "approach is not accepted by move_relative_ee" in reason


def test_approach_skill_needs_a_robot_with_the_primitive():
    allowed, reason = validate_skill_request(
        "set_gripper_approach", "", "", "", 0.0, {}, {}, APPROACH_TEMPLATES, approach="top_down"
    )
    assert not allowed and "unsupported primitive: set_gripper_approach" in reason


def _approach_primitive(pose_name, quaternion, xyz=(0.3, -0.1, 0.85), **kwargs):
    qx, qy, qz, qw = quaternion
    return validate_primitive_request(
        "set_gripper_approach",
        pose_name,
        0.0,
        0.0,
        0.0,
        *xyz,
        0.0,
        {},
        DUAL_WORKSPACE,
        target_qx=qx,
        target_qy=qy,
        target_qz=qz,
        target_qw=qw,
        velocity_scaling=0.5,
        primitive_descriptors=APPROACH_DESCRIPTORS,
        target_group="right_arm",
        arm_groups=ARM_GROUPS,
        motion_limits={"max_velocity_scaling": 0.6},
        gripper_approach=GRIPPER_APPROACH,
        **kwargs,
    )


def test_approach_primitive_requires_the_declared_orientation_inside_the_workspace():
    assert _approach_primitive("top_down", (-0.5, 0.5, 0.5, 0.5)) == (True, "")
    assert _approach_primitive("top_down", (0.5, -0.5, -0.5, -0.5)) == (True, "")  # same rotation, -q
    allowed, reason = _approach_primitive("top_down", (0.0, 0.0, 0.7071068, 0.7071068))
    assert not allowed and "is not the declared 'top_down' approach" in reason
    allowed, reason = _approach_primitive("sideways", (-0.5, 0.5, 0.5, 0.5))
    assert not allowed and "unknown gripper approach" in reason
    allowed, reason = _approach_primitive("top_down", (-0.5, 0.5, 0.5, 0.5), xyz=(0.3, -0.1, 0.70))
    assert not allowed and "outside workspace" in reason
