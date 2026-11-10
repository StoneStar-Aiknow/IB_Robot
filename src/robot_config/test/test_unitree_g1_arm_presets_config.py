"""Minimal G1 arm preset deployment does not inherit hand or base control."""

from pathlib import Path

import yaml

from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

EXPECTED_ACTIONS = (
    "release_arm",
    "turn_back_wave",
    "blow_kiss_with_both_hands",
    "blow_kiss_with_left_hand",
    "blow_kiss_with_right_hand",
    "arms_horizontal",
    "clamp",
    "high_five",
    "hug",
    "make_heart_with_both_hands",
    "make_heart_with_right_hand",
    "refuse",
    "right_hand_up",
    "ultraman_ray",
    "wave_under_head",
    "wave_above_head",
    "shake_hand",
    "box_left_hand_win",
    "box_right_hand_win",
    "box_both_hand_win",
    "right_hand_on_heart",
    "both_hands_up_deviate_right",
    "forward_push",
)
MODE_MACHINE_LIMITED = {
    "make_heart_with_both_hands",
    "make_heart_with_right_hand",
    "box_left_hand_win",
    "box_right_hand_win",
    "box_both_hand_win",
    "forward_push",
}
EXCLUDED_NAME_ACTIONS = {"throw_money", "spin_discs", "scratch_head", "waist_drum_dance"}


def test_arm_preset_config_is_valid_and_keeps_runtime_idle():
    path = Path(__file__).resolve().parents[1] / "config/robots/unitree_g1_arm_presets.yaml"
    config = load_robot_config_dict(path, defer_interface_binding=True)
    validate_robot_config_dict(config, deferred_interfaces=True)
    assert config["runtime"]["profile"] == "g1_29dof_arm_presets"
    assert config["capabilities"]["requires"] == ["joint.state", "motion.named"]
    assert config["skill_required_control_mode"] == "named_motion"
    assert config["control_modes"]["named_motion"]["runtime_mode"] == "idle"
    assert config["control_modes"]["named_motion"]["executor"]["enabled"] is False
    assert config["embodied"]["arm_surface"] is False
    assert config["embodied"]["runtime_named_motion"]["interface"] == "motion.named"
    assert config["embodied"]["skill_catalog_profile"] == "unitree_g1_arm_presets"


def test_preset_surface_is_one_static_23_action_contract():
    src = Path(__file__).resolve().parents[2]
    runtime_profile = yaml.safe_load(
        (src / "robots/unitree/unitree_g1_robot/profiles/g1_29dof_arm_presets.yaml").read_text(encoding="utf-8")
    )
    catalog_profile = yaml.safe_load(
        (src / "agent/skill_catalog/config/profiles/unitree_g1_arm_presets.yaml").read_text(encoding="utf-8")
    )
    vendor = runtime_profile["vendor"]
    actions = vendor["arm_actions"]
    names = list(runtime_profile["capabilities"]["motion.named"]["names"])
    allowed = vendor["preset_commissioning"]["allowed_arm_actions"]
    enabled = [entry["name"] for entry in catalog_profile["enabled_skills"]]

    assert len(EXPECTED_ACTIONS) == 23
    assert names == list(EXPECTED_ACTIONS)
    assert list(actions) == list(EXPECTED_ACTIONS)
    assert allowed == actions
    assert list(allowed) == list(EXPECTED_ACTIONS)
    assert enabled == list(EXPECTED_ACTIONS)
    assert all(entry["planner_visible"] is True for entry in catalog_profile["enabled_skills"])
    assert EXCLUDED_NAME_ACTIONS.isdisjoint({name.lower() for name in EXPECTED_ACTIONS})

    assert actions["arms_horizontal"]["action_id"] == 15
    assert actions["arms_horizontal"]["action_name"] == "both_hands_up"
    assert actions["turn_back_wave"].get("fsm_ids") == [500, 501]
    for name in MODE_MACHINE_LIMITED:
        assert actions[name].get("mode_machine") == [5, 6]
    constrained = {name for name, entry in actions.items() if "fsm_ids" in entry or "mode_machine" in entry}
    assert constrained == {"turn_back_wave"} | MODE_MACHINE_LIMITED

    skills = src / "agent/skill_catalog/config/skills"
    for name in EXPECTED_ACTIONS:
        manifest = yaml.safe_load((skills / name / "manifest.yaml").read_text(encoding="utf-8"))
        implementation = yaml.safe_load(
            (skills / name / "implementations/g1_preset_v1.yaml").read_text(encoding="utf-8")
        )
        assert manifest["name"] == name
        assert manifest["capability"]["display_name"]
        assert manifest["capability"]["recovery_policy"] == "never_retry"
        assert implementation["executor"] == "runtime_named_motion"
        assert implementation["binding"] == {"motion": name}

    release_manifest = yaml.safe_load((skills / "release_arm/manifest.yaml").read_text(encoding="utf-8"))
    assert release_manifest["semantic_level"] == "atomic_operator"
    assert release_manifest["capability"]["display_name"] == "释放手臂控制"
    assert release_manifest["capability"]["recovery_policy"] == "never_retry"
    release_summary = release_manifest["description"]["summary"].lower()
    assert "control-ownership" in release_summary
    assert "not a normal gesture" in release_summary
    assert "never" in release_summary
