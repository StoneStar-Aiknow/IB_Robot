from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict


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


def test_embodied_config_does_not_expose_unused_entry_routing():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]

    assert "entry" not in raw_config["embodied"]


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
