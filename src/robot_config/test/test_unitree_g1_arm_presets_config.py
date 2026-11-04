"""Minimal G1 arm preset deployment does not inherit hand or base control."""

from pathlib import Path

from robot_config.loader import load_robot_config_dict, validate_robot_config_dict


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
