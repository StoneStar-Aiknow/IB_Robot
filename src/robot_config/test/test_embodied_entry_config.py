from pathlib import Path

import pytest
import yaml

from robot_config.loader import (
    load_robot_config,
    load_robot_config_dict,
    validate_config,
    validate_embodied_launch_dict,
)
from robot_skill_cli.catalog import compile_local_snapshot

GRIPPER_TRAJECTORY_DURATION_SEC = 1.0


def _snapshot(config_path: Path):
    return compile_local_snapshot(load_robot_config_dict(config_path), config_path)


@pytest.mark.parametrize(
    "config_name",
    ["so101_single_arm"],
)
def test_compiled_profile_includes_dance_basic(config_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    skill_templates = _snapshot(config_path).templates

    assert "dance_basic" in skill_templates
    primitive_sequence = skill_templates["dance_basic"]["primitive_sequence"]
    assert primitive_sequence
    trajectory_step = next(
        step for step in primitive_sequence if step["primitive_name"] == "move_through_joint_positions"
    )
    assert trajectory_step["joint_waypoints"]


def test_default_loader_uses_robot_config_environment_path(monkeypatch):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    monkeypatch.setenv("ROBOT_CONFIG", str(config_path))

    config = load_robot_config_dict()

    assert config["_config_path"] == str(config_path.resolve())


@pytest.mark.parametrize("required_control_mode", ["unknown_mode", 1])
def test_loader_rejects_invalid_global_skill_required_control_mode(tmp_path, required_control_mode):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "test_robot",
                    "control_modes": {"moveit_planning": {}},
                    "skill_required_control_mode": required_control_mode,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skill_required_control_mode"):
        load_robot_config_dict(config_path)


def test_so101_skill_gateway_control_mode_is_global_and_safety_has_no_motion_authorization():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    embodied = raw_config["embodied"]

    assert raw_config["skill_required_control_mode"] == "moveit_planning"
    assert raw_config["skill_required_control_mode"] in raw_config["control_modes"]
    assert "motion_authorized" not in embodied["safety"]
    assert "skill_templates" not in embodied

    typed_config = load_robot_config(config_path)
    assert not hasattr(typed_config.embodied, "motion_authorized")


def test_compiled_skills_match_profile_enabled_set():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    profile_path = config_path.parents[3] / "skill_catalog" / "config" / "profiles" / "so101_single_arm.yaml"
    expected = {entry["name"] for entry in yaml.safe_load(profile_path.read_text(encoding="utf-8"))["enabled_skills"]}

    assert set(_snapshot(config_path).enabled_skill_names) == expected


def test_production_robot_yaml_has_no_inline_skill_catalog():
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    embodied = yaml.safe_load(source_path.read_text(encoding="utf-8"))["robot"]["embodied"]
    assert "skill_templates" not in embodied
    assert embodied["skill_catalog_profile"] == "so101_single_arm"


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
    skill_templates = _snapshot(config_path).templates

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
    skill_templates = _snapshot(config_path).templates

    assert pose_name in config["embodied"]["named_poses"]
    step = skill_templates[skill_name]["primitive_sequence"][0]
    assert dict(step) == {"primitive_name": "move_to_named_pose", "pose_name": pose_name}


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
    skill = _snapshot(config_path).templates[skill_name]
    manifest_path = config_path.parents[3] / "skill_catalog" / "config" / "skills" / skill_name / "manifest.yaml"
    description = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["description"]

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

    assert float(description["duration_sec_estimate"]) >= configured_duration
