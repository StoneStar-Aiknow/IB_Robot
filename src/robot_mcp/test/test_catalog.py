"""Pure unit tests for robot_mcp.catalog (no ROS required).

The catalog helpers are the SSOT-derived core of Phase 0; they must produce the
same shape the agent tools return. These tests run without a ROS daemon.
Config-backed tests are skipped if robot_config is not importable (e.g. the
workspace overlay is not sourced).
"""

from __future__ import annotations

import pytest

from robot_mcp.catalog import (
    _build_pose_entry,
    _build_skill_entry,
    _normalize_workspace,
    build_catalog,
    resolve_config_path,
    valid_motion_directions,
)

robot_config_available = True
try:
    import robot_config  # noqa: F401
except Exception:  # noqa: BLE001
    robot_config_available = False


# ----------------------------- pure helpers ---------------------------------


def test_skill_entry_shape_and_motion_flag():
    template = {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "pose_name": "home"},
            {
                "primitive_name": "move_relative_ee",
                "motion_direction_from_request": True,
                "motion_distance_from_request": True,
            },
        ]
    }
    entry = _build_skill_entry("move_relative_ee", template)
    assert entry["name"] == "move_relative_ee"
    assert entry["primitives"] == ["move_to_named_pose", "move_relative_ee"]
    assert entry["pose_targets"] == ["home"]
    assert entry["accepts_motion"] is True
    assert entry["vision_only"] is False


def test_inspect_scene_is_vision_only():
    entry = _build_skill_entry("inspect_scene", {"primitive_sequence": []})
    assert entry["vision_only"] is True
    assert entry["primitives"] == []
    assert entry["accepts_motion"] is False


def test_pose_entry_rounds_and_drops_missing():
    raw = {
        "position": {"x": 0.02003073320, "y": -0.11809839, "z": 0.13887055751},
        "orientation": {"x": -0.26790866, "y": -0.26221859, "z": -0.65251338, "w": 0.65855344},
    }
    entry = _build_pose_entry("home", raw)
    assert entry["name"] == "home"
    assert set(entry["position"]) == {"x", "y", "z"}
    assert entry["position"]["x"] == round(0.02003073320, 6)
    assert set(entry["orientation"]) == {"x", "y", "z", "w"}


def test_pose_entry_handles_bad_input():
    assert _build_pose_entry("weird", {})["position"] == {}
    assert _build_pose_entry("weird", "not-a-dict")["orientation"] == {}


def test_normalize_workspace():
    assert _normalize_workspace({"x": [0.1, 0.5], "y": [-0.2, 0.2], "z": [0, 1]}) == {
        "x": [0.1, 0.5],
        "y": [-0.2, 0.2],
        "z": [0.0, 1.0],
    }
    assert _normalize_workspace({}) == {}
    assert _normalize_workspace({"x": [1]}) == {}  # wrong length -> dropped


def test_valid_motion_directions():
    dirs = valid_motion_directions()
    assert dirs == ["backward", "down", "forward", "left", "right", "up"]


# ----------------------- config-backed (need overlay) -----------------------


@pytest.mark.skipif(not robot_config_available, reason="robot_config overlay not sourced")
def test_resolve_config_path_for_default_robot():
    path = resolve_config_path(config_name="so101_single_arm")
    assert path.exists()
    assert path.name == "so101_single_arm.yaml"


@pytest.mark.skipif(not robot_config_available, reason="robot_config overlay not sourced")
def test_build_catalog_matches_so101_ssot():
    catalog = build_catalog(config_name="so101_single_arm")
    skill_names = {s["name"] for s in catalog.skills}
    # The 9 canonical skills (loader.py valid set) are all declared in so101.
    assert {
        "inspect_scene",
        "open_gripper_skill",
        "close_gripper_skill",
        "recover_safe_pose",
        "recover_zero_pose",
        "move_relative_ee",
        "rotate_gripper_cw",
        "rotate_gripper_ccw",
        "dance_basic",
    } <= skill_names

    pose_names = {p["name"] for p in catalog.poses}
    assert {"home", "observe_table", "zero"} <= pose_names

    # ROS interface names come straight from EmbodiedConfig defaults.
    assert catalog.ros_interfaces["status_topic"] == "/embodied/task_status"
    assert catalog.ros_interfaces["skill_action"] == "/embodied/execute_skill"
    assert catalog.ros_interfaces["validate_skill_service"] == "/embodied/validate_skill"

    # A known pose carries a full quaternion.
    home = next(p for p in catalog.poses if p["name"] == "home")
    assert set(home["orientation"]) == {"x", "y", "z", "w"}

    # move_relative_ee must advertise that it accepts motion params.
    rel = next(s for s in catalog.skills if s["name"] == "move_relative_ee")
    assert rel["accepts_motion"] is True
