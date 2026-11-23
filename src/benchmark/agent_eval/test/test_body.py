"""Body description from robot config + runtime profile; no robot names in the code."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_eval.body import BodyConfigError, body_from_documents, load_body
from agent_eval_testkit import FIXTURES, fixture_body


def test_dual_arm_body_comes_from_config_and_profile():
    body = fixture_body()
    assert body.sides == ("left", "right")
    left, right = body.arms
    assert (left.group, left.gripper_joint, left.wrist_joint) == ("left_arm", "left_gripper", "left_joint6")
    assert left.ee_pose_topic == "/robot_status/left_arm/ee_pose"
    assert right.ee_pose_topic == "/robot_status/right_arm/ee_pose"  # default from the group name
    facts = body.facts()
    assert facts.approach_axis == "x"
    assert set(facts.approach_presets) == {"horizontal", "top_down", "top_down_rotated", "tilted_45"}


def test_single_seven_dof_arm_body():
    body = fixture_body("single_arm_fixture")
    assert body.sides == ("main",)
    assert body.arm("main").wrist_joint == "j7"
    assert body.facts().approach_presets == {}


def test_load_body_resolves_the_profile_through_search_dirs():
    body = load_body(
        robot_config_path=FIXTURES / "robots" / "dual_arm_fixture.yaml",
        profile_search_dirs=[FIXTURES / "profiles"],
    )
    assert body.runtime_provider == "fixture_robot"
    assert body.runtime_profile_path.endswith("dual_arm_fixture.yaml")
    with pytest.raises(BodyConfigError, match="not found"):
        load_body(robot_config_path=FIXTURES / "robots" / "dual_arm_fixture.yaml", profile_search_dirs=[FIXTURES])


def test_a_profile_given_as_a_path_is_used_as_is():
    from agent_eval.body import resolve_profile_path

    path = FIXTURES / "profiles" / "dual_arm_fixture.yaml"
    assert resolve_profile_path("fixture_robot", str(path)) == path.resolve()
    with pytest.raises(BodyConfigError, match="not a file"):
        resolve_profile_path("fixture_robot", str(FIXTURES / "profiles" / "missing.yaml"))


def test_shipped_robodojo_config_builds_the_same_body_as_the_fixture():
    # the fixture body must stay the robodojo_x5 body (robot_config resolves the name)
    body = load_body("robodojo_x5", profile_path=FIXTURES / "profiles" / "dual_arm_fixture.yaml")
    fixture = fixture_body()
    assert body.arms == fixture.arms
    assert body.facts() == fixture.facts()


def test_missing_body_facts_are_named():
    robot = {"embodied": {"execution": {"arm_groups": {"left": "left_arm"}}}, "task_dispatch": {"groups": {}}}
    with pytest.raises(BodyConfigError, match="gripper_joint"):
        body_from_documents(robot, {"motion_groups": {"left_arm": {"joints": ["j"]}}})
    robot["task_dispatch"]["groups"]["left_arm"] = {"gripper_joint": "g"}
    with pytest.raises(BodyConfigError, match="joints is empty"):
        body_from_documents(robot, {"motion_groups": {"left_arm": {"joints": []}}})


def test_inherited_robot_configs_resolve_through_the_robot_config_loader(tmp_path):
    import shutil

    shutil.copy(FIXTURES / "robots" / "dual_arm_fixture.yaml", tmp_path / "dual_arm_fixture.yaml")
    (tmp_path / "derived.yaml").write_text(
        "robot:\n  base_config: dual_arm_fixture\n  name: derived_fixture\n", encoding="utf-8"
    )
    body = load_body(
        robot_config_path=tmp_path / "derived.yaml",
        profile_path=FIXTURES / "profiles" / "dual_arm_fixture.yaml",
    )
    base = load_body(
        robot_config_path=tmp_path / "dual_arm_fixture.yaml",
        profile_path=FIXTURES / "profiles" / "dual_arm_fixture.yaml",
    )
    assert body.robot_config == "derived_fixture" and body.arms == base.arms  # facts come from the base
    assert [Path(p).name for p in body.robot_config_sources] == ["dual_arm_fixture.yaml", "derived.yaml"]
