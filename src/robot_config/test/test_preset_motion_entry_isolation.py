"""Preset matching is opt-in and does not replace grasp or summit skill profiles."""

from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
ROBOTS = PACKAGE / "config" / "robots"


def robot(name):
    return yaml.safe_load((ROBOTS / f"{name}.yaml").read_text())["robot"]


def test_all_entry_enables_only_observation_and_keeps_motion_disabled():
    settings = robot("aimdk_x2_voice_mirror_all")["preset_motions"]
    assert settings["enabled"] is True
    assert settings["motion_enabled"] is False
    assert settings["match_topic"] == "/preset_motions/matched"
    assert settings["state_topic"] == "/preset_motions/state"


def test_grasp_explicitly_disables_both_matching_and_motion():
    settings = robot("aimdk_x2_voice_mirror_grasp")["preset_motions"]
    assert settings["enabled"] is False
    assert settings["motion_enabled"] is False


def test_summit_inherits_the_disabled_grasp_entry():
    settings = robot("aimdk_x2_summit")
    assert settings["base_config"] == "aimdk_x2_voice_mirror_grasp.yaml"
    assert settings.get("preset_motions", {}).get("enabled", False) is False
    assert settings.get("preset_motions", {}).get("motion_enabled", False) is False


def test_existing_skill_catalog_bindings_are_not_replaced():
    assert robot("aimdk_x2_voice_mirror_all")["embodied"]["skill_catalog_profile"] == "aimdk_x2_all"
    assert robot("aimdk_x2_voice_mirror_grasp")["embodied"]["skill_catalog_profile"] == "aimdk_x2_voice_mirror_grasp"
    assert robot("aimdk_x2_summit")["embodied"]["skill_catalog_profile"] == "aimdk_x2_summit"


def test_visual_pose_approval_is_not_a_motion_certificate():
    settings = yaml.safe_load((PACKAGE / "config/preset_motions/overhead_heart.yaml").read_text())
    assert settings["visual_approval"]["approved"] is True
    assert settings["review_state"] == "visual_approved_safety_pending"
    assert settings["validated"] is False and settings["motion_enabled"] is False
    assert settings["required_head_clearance_m"] == 0.20
