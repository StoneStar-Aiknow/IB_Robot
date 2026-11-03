"""Contract v2: optional multi-arm fields of the robot context and the shipped RoboDojo X5 profile."""

from dataclasses import replace
from pathlib import Path

from test_compiler import _context

from embodied_common.catalog_digest import _robot_context_to_preimage
from embodied_common.primitive_contracts import primitive_contract_for_version, primitive_extensions_for_context
from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import SkillCompileError
from skill_catalog.source import DevelopmentStagingSkillSource

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_empty_optional_fields_keep_the_preimage_of_existing_robots():
    robot = _context().robot
    preimage = _robot_context_to_preimage(robot)
    assert "arm_groups" not in preimage and "motion_limits" not in preimage


def test_declared_arm_groups_and_motion_limits_enter_the_preimage():
    robot = replace(
        _context().robot,
        arm_groups={"left": "left_arm", "right": "right_arm"},
        motion_limits={"max_velocity_scaling": 0.6},
    )
    preimage = _robot_context_to_preimage(robot)
    assert preimage["arm_groups"] == {"left": "left_arm", "right": "right_arm"}
    assert preimage["motion_limits"] == {"max_velocity_scaling": 0.6}


APPROACH = {
    "axis": "x",
    "presets": {
        "horizontal": {"orientation": {"x": 0.0, "y": 0.0, "z": 0.7071068, "w": 0.7071068}},
        "top_down": {"orientation": {"x": -0.5, "y": 0.5, "z": 0.5, "w": 0.5}},
        "top_down_rotated": {"orientation": {"x": 0.0, "y": 0.7071068, "z": 0.0, "w": 0.7071068}},
        "tilted_45": {"orientation": {"x": -0.2705981, "y": 0.2705981, "z": 0.6532815, "w": 0.6532815}},
    },
}
X5_SKILLS = {
    "move_relative_ee",
    "open_gripper_skill",
    "close_gripper_skill",
    "rotate_gripper_cw",
    "rotate_gripper_ccw",
    "set_gripper_approach",
}


def _robodojo_context(gripper_approach=APPROACH):
    context = _context()
    robot = replace(
        context.robot,
        robot_name="robodojo_x5",
        timeout_policy={"task_budget_sec": 180.0},
        arm_groups={"left": "left_arm", "right": "right_arm"},
        gripper_approach=gripper_approach,
    )
    contract = primitive_contract_for_version(1, primitive_extensions_for_context(robot))
    return replace(
        context,
        robot=robot,
        primitive_contracts=contract.descriptors,
        primitive_contract_digest=contract.digest,
    )


def test_robodojo_profile_compiles_with_arm_side_on_its_skills():
    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(PACKAGE_ROOT), profile_name="robodojo_x5", context=_robodojo_context()
    )
    skills = snapshot.capability_view
    assert set(skills) == X5_SKILLS
    for skill in skills.values():
        assert list(skill["parameters"]["properties"]["arm_side"]["enum"]) == ["left", "right", "auto"]
    approach = skills["set_gripper_approach"]["parameters"]
    assert list(approach["required"]) == ["approach"]
    assert set(approach["properties"]["approach"]["enum"]) == set(APPROACH["presets"])


def test_approach_primitive_needs_declared_presets():
    snapshot = None
    try:
        snapshot = compile_skill_catalog(
            DevelopmentStagingSkillSource(PACKAGE_ROOT),
            profile_name="robodojo_x5",
            context=_robodojo_context(gripper_approach={"axis": "x"}),
        )
    except SkillCompileError as exc:
        assert "unsupported primitive" in str(exc)
    assert snapshot is None


def test_gripper_approach_enters_the_preimage_only_when_declared():
    assert "gripper_approach" not in _robot_context_to_preimage(_context().robot)
    assert _robot_context_to_preimage(_robodojo_context().robot)["gripper_approach"]["axis"] == "x"
