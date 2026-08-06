import copy

from robot_config.loader import robot_config_digest


def _config():
    return {
        "name": "test_robot",
        "joints": {"arm": ["joint_1"]},
        "skill_required_control_mode": "moveit_planning",
        "teleoperation": {"safety": {"joint_limits": {"joint_1": {"lower": -1.0, "upper": 1.0}}}},
        "embodied": {
            "skill_catalog_profile": "profile_a",
            "skill_catalog_source_mode": "development",
            "skill_catalog_source_root": "/catalog/a",
            "named_poses": {"home": {"joint_positions": {"joint_1": 0.0}}},
            "named_targets": {},
            "safety": {"workspace": {"x": [-1.0, 1.0]}},
            "execution": {"relative_motion_step_m": 0.03},
        },
        "unrelated": {"camera_preview": True},
        "_config_path": "/tmp/a.yaml",
    }


def test_robot_config_digest_excludes_catalog_source_profile_and_unrelated_config():
    original = _config()
    changed = copy.deepcopy(original)
    changed["embodied"].update(
        {
            "skill_catalog_profile": "profile_b",
            "skill_catalog_source_mode": "production",
            "skill_catalog_source_root": "/catalog/b",
        }
    )
    changed["unrelated"] = {"camera_preview": False}
    changed["_config_path"] = "/tmp/b.yaml"

    assert robot_config_digest(original) == robot_config_digest(changed)


def test_robot_config_digest_covers_execution_semantics_and_defaults():
    original = _config()
    explicit_defaults = copy.deepcopy(original)
    explicit_defaults["embodied"]["execution"].update(
        {
            "relative_motion_reference_frame": "base",
            "gripper_open_position": 1.0,
            "gripper_closed_position": 0.0,
        }
    )
    changed = copy.deepcopy(original)
    changed["embodied"]["execution"]["relative_motion_step_m"] = 0.04

    assert robot_config_digest(original) == robot_config_digest(explicit_defaults)
    assert robot_config_digest(original) != robot_config_digest(changed)
