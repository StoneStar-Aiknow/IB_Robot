"""The exhibition catalog: every clip resolves in the deployment and the planner sees the fruit menu."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from test_runtime_named_motion_catalog import _context

from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.source import DevelopmentStagingSkillSource

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT.parents[1] / "robot_config" / "config" / "robots" / "aimdk_x2_summit.yaml"
CLIP_SKILLS = {
    "greet_visitor",
    "self_introduction",
    "introduce_platform",
    "introduce_agent",
    "introduce_in_detail",
    "explain_pick_scope",
    "offer_fruit_menu",
    "mimic_not_ready",
    "invite_visitor",
}


@pytest.fixture(scope="module")
def snapshot():
    context = _context()
    delegated = dict(context.delegated_executors)
    for executor in ("pose_composer", "grasp_pipeline", "runtime_play_audio"):
        delegated.update(_context(executor).delegated_executors)
    delegated["grasp_pipeline"] = replace(
        delegated["grasp_pipeline"],
        model_deployment_name="mock",
        model_fingerprint="0" * 64,
        model_bundle_digest="1" * 64,
    )
    robot = replace(
        context.robot,
        robot_name="aimdk_x2_summit",
        supported_control_modes=("named_motion", "model_inference", "moveit_planning"),
        timeout_policy={"task_budget_sec": 180.0},
    )
    return compile_skill_catalog(
        DevelopmentStagingSkillSource(ROOT),
        profile_name="aimdk_x2_summit",
        context=replace(context, robot=robot, delegated_executors=delegated),
    )


@pytest.fixture(scope="module")
def deployment():
    return yaml.safe_load(ROBOT.read_text(encoding="utf-8"))["robot"]["embodied"]


def test_the_catalog_matches_the_deployment_allowlist(snapshot, deployment):
    enabled = set(snapshot.enabled_skill_names)
    assert enabled == set(deployment["agent"]["test_allowlist"])
    assert enabled >= CLIP_SKILLS
    assert not {"mirror_start", "mirror_stop", "free_pose"} & enabled


def test_every_clip_skill_is_silent_and_resolves_its_clip(snapshot, deployment):
    played = {name for name, template in snapshot.templates.items() if template["executor"] == "runtime_play_audio"}
    assert played == CLIP_SKILLS == set(deployment["agent"]["voice_bridge"]["silent_skills"])
    assert {snapshot.templates[name]["binding"]["clip"] for name in played} == set(
        deployment["runtime_play_audio"]["clips"]
    )


def test_each_social_clip_selects_its_recorded_gesture(snapshot):
    bindings = {
        "greet_visitor": ("greeting", "short_wave"),
        "self_introduction": ("self_intro", "self_intro"),
        "introduce_platform": ("intro_platform", "explain"),
        "introduce_agent": ("intro_agent", "explain"),
        "introduce_in_detail": ("intro_detail", "self_intro"),
        "invite_visitor": ("invitation", "invite"),
    }
    assert {name for name in CLIP_SKILLS if snapshot.capability_view[name]["moves_robot"]} == set(bindings)
    for name, (clip, motion) in bindings.items():
        assert snapshot.templates[name]["binding"] == {"clip": clip, "motion": motion}
        assert snapshot.templates[name]["timeout_sec"] == 90.0
        assert snapshot.templates[name]["implementation_identity"].endswith("#aimdk_x2_summit")


def test_the_planner_is_told_the_three_fruits(snapshot):
    assert snapshot.templates["pick_object"]["implementation_identity"].endswith("#aimdk_x2_summit_fruit")
    assert snapshot.templates["pick_object"]["timeout_sec"] == 150.0
    summary = snapshot.capability_view["pick_object"]["summary"]
    for target in ('"a lemon"', '"an apple"', '"an orange"'):
        assert target in summary
