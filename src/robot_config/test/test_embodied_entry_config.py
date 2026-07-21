from pathlib import Path

import pytest

from robot_config.loader import (
    load_robot_config,
    load_robot_config_dict,
    validate_config,
    validate_embodied_launch_dict,
)

GRIPPER_TRAJECTORY_DURATION_SEC = 1.0


@pytest.mark.parametrize(
    "config_name",
    ["so101_single_arm"],
)
def test_loaded_embodied_skill_templates_include_dance_basic(config_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)
    skill_templates = config["embodied"]["skill_templates"]

    assert "dance_basic" in skill_templates
    primitive_sequence = skill_templates["dance_basic"]["primitive_sequence"]
    assert primitive_sequence
    trajectory_step = next(
        step for step in primitive_sequence if step["primitive_name"] == "move_through_joint_positions"
    )
    assert trajectory_step["joint_waypoints"]


def test_embodied_entry_visual_games_typed():
    """entry.visual_games must survive the typed loader (SSOT), not be dropped."""
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    games = config.embodied.entry["visual_games"]
    sorting_hat = games["sorting_hat"]
    assert "enabled" in sorting_hat
    assert sorting_hat["trigger_aliases"]


def test_enabled_game_requires_perception_enabled():
    """An enabled visual game with perception disabled must fail validation."""
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True
    config.embodied.perception = {**config.embodied.perception, "enabled": False}
    config.embodied.entry = {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}}

    errors = validate_config(config)
    assert any("visual_games" in error for error in errors)


def test_disabled_games_do_not_require_perception():
    """All games disabled: perception may stay off without a validation error."""
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True
    config.embodied.perception = {**config.embodied.perception, "enabled": False}
    config.embodied.entry = {"visual_games": {"sorting_hat": {"enabled": False, "trigger_aliases": ["分院帽"]}}}

    errors = validate_config(config)
    assert not any("visual_games" in error for error in errors)


def test_launch_dict_enabled_game_without_perception_is_rejected():
    """The raw-dict launch gate rejects a game enabled + perception disabled."""
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": False},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
        }
    }
    errors = validate_embodied_launch_dict(config)
    assert any("visual_games" in error for error in errors)


def test_launch_dict_enabled_game_with_perception_is_ok():
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
        }
    }
    assert validate_embodied_launch_dict(config) == []


def test_launch_dict_skips_when_embodied_disabled():
    """A disabled embodied stack is never gated on game/perception consistency."""
    config = {
        "embodied": {
            "enabled": False,
            "perception": {"enabled": False},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
        }
    }
    assert validate_embodied_launch_dict(config) == []


def test_embodied_config_keeps_only_supported_direct_skills():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    skill_templates = config["embodied"]["skill_templates"]

    assert "dance_basic" in skill_templates
    assert "pick_named_target" not in skill_templates
    assert "place_named_pose" not in skill_templates
    assert "observe_target_area" not in skill_templates


@pytest.mark.parametrize(
    ("skill_name", "pose_name"),
    [
        ("recover_safe_pose", "home"),
        ("inspect_scene", "observe_table"),
        ("recover_zero_pose", "zero"),
    ],
)
def test_embodied_named_pose_skills_map_to_configured_poses(skill_name, pose_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    skill_templates = config["embodied"]["skill_templates"]

    assert pose_name in config["embodied"]["named_poses"]
    assert skill_templates[skill_name]["primitive_sequence"] == [
        {"primitive_name": "move_to_named_pose", "pose_name": pose_name}
    ]


def test_enabled_embodied_config_uses_configured_default_place_pose():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True

    assert validate_config(config) == []
    assert config.embodied.default_place_name in config.embodied.named_poses


@pytest.mark.parametrize(
    "skill_name",
    ["wave_hello", "nod_yes", "shake_no", "act_cute", "happy_spin_upright"],
)
def test_social_gesture_duration_estimate_covers_configured_motion(skill_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    skill = load_robot_config_dict(config_path)["embodied"]["skill_templates"][skill_name]

    configured_duration = 0.0
    if skill.get("initial_gripper_state") in {"open", "closed"}:
        configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC
    for step in skill["primitive_sequence"]:
        primitive_name = step["primitive_name"]
        if primitive_name == "move_to_joint_positions":
            configured_duration += float(step.get("duration_sec", 0.4))
        elif primitive_name == "move_through_joint_positions":
            configured_duration += len(step["joint_waypoints"]) * float(step["waypoint_duration_sec"])
        elif primitive_name in {"open_gripper", "close_gripper"}:
            configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC

    assert float(skill["description"]["duration_sec_estimate"]) >= configured_duration
