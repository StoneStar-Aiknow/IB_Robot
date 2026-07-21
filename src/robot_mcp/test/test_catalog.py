"""Pure unit tests for robot_mcp.catalog (no ROS required).

The catalog helpers are the SSOT-derived core of Phase 0; they must produce the
same shape the agent tools return. These tests run without a ROS daemon.
Config-backed tests are skipped if robot_config is not importable (e.g. the
workspace overlay is not sourced).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


SO101 = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"


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


def test_skill_entry_includes_implicit_initial_gripper_primitive():
    template = {
        "initial_gripper_state": "closed",
        "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "home"}],
    }

    entry = _build_skill_entry("recover_safe_pose", template)

    assert entry["primitives"] == ["close_gripper", "move_to_named_pose"]


def test_inspect_scene_is_vision_only():
    entry = _build_skill_entry("inspect_scene", {"primitive_sequence": []})
    assert entry["vision_only"] is True
    assert entry["primitives"] == []
    assert entry["accepts_motion"] is False


def test_skill_entry_surfaces_structured_description_fields():
    template = {
        "primitive_sequence": [{"primitive_name": "move_through_joint_positions"}],
        "description": {
            "summary": "Wave hello or goodbye with the wrist.",
            "category": "social_greeting",
            "when_to_use": ["greet someone", "say hi or bye"],
            "do_not_use": [
                {"condition": "agree or say yes", "instead_use": "nod_yes"},
                {"condition": "say no", "instead_use": "shake_no"},
            ],
            "aliases_zh": ["打招呼", "挥手"],
            "aliases_en": ["hello", "wave"],
            "motion_scope": ["wrist"],
            "anchor_pose": "home",
            "intensity": "moderate",
            "duration_sec_estimate": 2.5,
            "requires_motion_params": False,
        },
    }
    entry = _build_skill_entry("wave_hello", template)

    assert entry["category"] == "social_greeting"
    assert entry["aliases_zh"] == ["打招呼", "挥手"]
    assert entry["motion_scope"] == ["wrist"]
    assert entry["anchor_pose"] == "home"
    assert entry["intensity"] == "moderate"
    assert entry["requires_motion_params"] is False
    assert entry["when_to_use"] == ["greet someone", "say hi or bye"]

    doc = entry["doc"]
    # Synthesized doc must lead with category + summary, list use cases, and --
    # crucially -- spell out the do-not-use redirects toward near-synonyms so an
    # MCP/LLM caller can disambiguate wave_hello from nod_yes / shake_no.
    assert "[social_greeting]" in doc
    assert "Wave hello or goodbye with the wrist." in doc
    assert "greet someone" in doc
    assert "agree or say yes -> nod_yes" in doc
    assert "say no -> shake_no" in doc
    assert "中文" in doc and "打招呼" in doc
    assert "scope=wrist" in doc and "anchor=home" in doc and "intensity=moderate" in doc


def test_skill_entry_without_description_falls_back_to_legacy_docs():
    entry = _build_skill_entry("inspect_scene", {"primitive_sequence": []})
    assert entry["doc"]  # never empty
    assert "category" not in entry  # structured fields only when SSOT block present


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
    # The robot-declared skill set is surfaced without a process-global whitelist.
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
        "wave_hello",
        "nod_yes",
        "shake_no",
        "celebrate",
        "greet_observe_raise",
        "act_cute",
        "happy_spin_upright",
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

    wave = next(s for s in catalog.skills if s["name"] == "wave_hello")
    assert wave["primitives"] == [
        "close_gripper",
        "move_to_joint_positions",
        "move_through_joint_positions",
        "move_to_joint_positions",
    ]
    assert wave["pose_targets"] == []
    assert wave["doc"]
    # Structured description contract is surfaced for agent-side filtering.
    assert wave["category"] == "social_greeting"
    assert "打招呼" in wave["aliases_zh"]
    assert wave["motion_scope"] == ["wrist"]
    assert wave["intensity"] in {"subtle", "moderate", "large"}

    assert "wave_goodbye" not in skill_names

    # Every do_not_use redirect must point to a skill that the catalog actually
    # exposes; otherwise the disambiguation hint would send the agent nowhere.
    for skill in catalog.skills:
        for redirect in skill.get("do_not_use", []):
            assert redirect["instead_use"] in skill_names, (
                f"{skill['name']} redirects to missing skill {redirect['instead_use']}"
            )

    celebrate = next(s for s in catalog.skills if s["name"] == "celebrate")
    assert celebrate["primitives"] == [
        "close_gripper",
        "move_to_named_pose",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_to_named_pose",
    ]
    assert celebrate["pose_targets"] == ["observe_table", "observe_table"]
    assert celebrate["accepts_motion"] is False
    assert "observe_table" in celebrate["doc"]

    greet = next(s for s in catalog.skills if s["name"] == "greet_observe_raise")
    assert greet["primitives"] == [
        "close_gripper",
        "move_to_named_pose",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_relative_ee",
        "move_to_named_pose",
    ]
    assert greet["pose_targets"] == ["observe_table", "observe_table"]
    assert greet["accepts_motion"] is False
    assert "greet" in greet["doc"]

    act_cute = next(s for s in catalog.skills if s["name"] == "act_cute")
    assert act_cute["primitives"] == [
        "close_gripper",
        "move_to_named_pose",
        "move_to_joint_positions",
        "open_gripper",
        "move_through_joint_positions",
        "close_gripper",
        "open_gripper",
        "move_to_joint_positions",
    ]
    assert act_cute["pose_targets"] == ["home"]
    assert act_cute["accepts_motion"] is False

    happy_spin = next(s for s in catalog.skills if s["name"] == "happy_spin_upright")
    assert happy_spin["primitives"] == [
        "close_gripper",
        "move_to_joint_positions",
        "move_through_joint_positions",
        "move_to_joint_positions",
    ]
    assert happy_spin["pose_targets"] == []
    assert happy_spin["accepts_motion"] is False
    assert "upright" in happy_spin["doc"]

    dance = next(s for s in catalog.skills if s["name"] == "dance_basic")
    assert dance["primitives"] == [
        "close_gripper",
        "move_to_joint_positions",
        "move_through_joint_positions",
    ]

    nod = next(s for s in catalog.skills if s["name"] == "nod_yes")
    assert nod["primitives"] == [
        "close_gripper",
        "move_to_joint_positions",
        "move_through_joint_positions",
        "move_to_joint_positions",
    ]


@pytest.mark.skipif(not robot_config_available, reason="robot_config overlay not sourced")
def test_build_catalog_rejects_malformed_skill_description(tmp_path):
    data = yaml.safe_load(SO101.read_text(encoding="utf-8"))
    data["robot"]["embodied"]["skill_templates"]["wave_hello"]["description"]["summary"] = {"text": "wave"}
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="summary must be a non-empty string"):
        build_catalog(config_path=str(config_path))


@pytest.mark.skipif(not robot_config_available, reason="robot_config overlay not sourced")
def test_build_catalog_excludes_disabled_skill(tmp_path):
    data = yaml.safe_load(SO101.read_text(encoding="utf-8"))
    data["robot"]["embodied"]["skill_templates"]["disabled_demo"] = {
        "disabled": True,
        "description": {
            "summary": "Disabled demo.",
            "category": "system",
            "when_to_use": ["never"],
        },
        "primitive_sequence": [{"primitive_name": "open_gripper"}],
    }
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    catalog = build_catalog(config_path=str(config_path))

    assert "disabled_demo" not in {skill["name"] for skill in catalog.skills}
