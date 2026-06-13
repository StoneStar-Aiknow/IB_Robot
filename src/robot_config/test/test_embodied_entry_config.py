from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict


@pytest.mark.parametrize(
    "config_name",
    ["so101_single_arm"],
)
def test_embodied_entry_direct_skill_whitelist_includes_move_and_dance(config_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    direct_skills = raw_config["embodied"]["entry"]["direct_skill_whitelist"]

    assert "move_relative_ee" in direct_skills
    assert "dance_basic" in direct_skills


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
    assert primitive_sequence[0]["primitive_name"] == "move_through_joint_positions"
    assert primitive_sequence[0]["joint_waypoints"]


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
