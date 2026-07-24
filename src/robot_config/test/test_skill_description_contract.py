"""Tests for the MCP-facing skill `description` contract and its validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import _validate_skill_description, load_robot_config, load_robot_config_dict

SO101 = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"

_REQUIRED_FIELDS = ("summary", "category", "when_to_use", "motion_scope", "intensity")


def _description(skill_templates: dict, skill_name: str) -> dict:
    description = skill_templates[skill_name].get("description")
    assert description, f"{skill_name} is missing a description block"
    return description


def _write_config(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(SO101.read_text(encoding="utf-8"))
    mutate(data["robot"]["embodied"]["skill_templates"])
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_so101_skills_all_carry_description_contract():
    config = load_robot_config_dict(SO101)
    skill_templates = config["embodied"]["skill_templates"]

    assert skill_templates, "so101 must define embodied.skill_templates"
    for skill_name in skill_templates:
        description = _description(skill_templates, skill_name)
        for field in _REQUIRED_FIELDS:
            assert field in description, f"{skill_name}.description missing {field}"
        assert isinstance(description["when_to_use"], list) and description["when_to_use"]
        assert description["intensity"] in {"subtle", "moderate", "large"}
        assert description["summary"]


@pytest.mark.parametrize(
    "skill_name",
    ["wave_hello", "nod_yes", "shake_no", "celebrate", "greet_observe_raise", "act_cute", "happy_spin_upright"],
)
def test_social_gestures_declare_disambiguation(skill_name: str):
    config = load_robot_config_dict(SO101)
    description = _description(config["embodied"]["skill_templates"], skill_name)
    do_not_use = description.get("do_not_use")

    assert do_not_use, f"{skill_name} must declare do_not_use to disambiguate from near-synonyms"
    for entry in do_not_use:
        assert entry.get("condition"), f"{skill_name} do_not_use entry needs a condition"
        assert entry.get("instead_use"), f"{skill_name} do_not_use entry needs instead_use"


def test_so101_description_aliases_match_supported_skills():
    """Every instead_use target must be a real skill declared in the same config."""
    config = load_robot_config_dict(SO101)
    skill_templates = config["embodied"]["skill_templates"]
    known_skills = set(skill_templates)

    for skill_name, template in skill_templates.items():
        for entry in template.get("description", {}).get("do_not_use", []):
            target = entry["instead_use"]
            assert target in known_skills, f"{skill_name}.description.do_not_use points to unknown skill '{target}'"
            assert target != skill_name, f"{skill_name}.description.do_not_use points to itself"


def test_skill_disabled_flag_must_be_boolean(tmp_path):
    config_path = _write_config(tmp_path, lambda skills: skills["wave_hello"].update(disabled="yes"))

    with pytest.raises(ValueError, match="disabled must be a boolean"):
        load_robot_config_dict(config_path)


def test_planner_allowlist_rejects_disabled_skill(tmp_path):
    config_path = _write_config(tmp_path, lambda skills: skills["recover_zero_pose"].update(disabled=True))

    with pytest.raises(ValueError, match=r"allowed_skills.*recover_zero_pose"):
        load_robot_config_dict(config_path)


def _collect_errors(description: dict, valid_skills: set[str], named_poses: dict | None = None) -> list[str]:
    errors: list[str] = []
    _validate_skill_description("demo", description, valid_skills, named_poses or {}, errors)
    return errors


def test_validator_accepts_well_formed_description():
    description = {
        "summary": "A well-formed demo skill.",
        "category": "demo",
        "when_to_use": ["demo case"],
        "do_not_use": [{"condition": "other case", "instead_use": "other"}],
        "motion_scope": ["wrist"],
        "anchor_pose": "none",
        "intensity": "subtle",
        "duration_sec_estimate": 1.5,
        "requires_motion_params": False,
    }
    assert _collect_errors(description, {"demo", "other"}) == []


def test_validator_rejects_unknown_instead_use_target():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "do_not_use": [{"condition": "y", "instead_use": "ghost_skill"}],
    }
    errors = _collect_errors(description, {"demo"})
    assert any("ghost_skill" in e for e in errors)


def test_validator_rejects_self_referencing_instead_use():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "do_not_use": [{"condition": "y", "instead_use": "demo"}],
    }
    errors = _collect_errors(description, {"demo"})
    assert any("same skill" in e for e in errors)


def test_validator_rejects_bad_vocab_and_values():
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "motion_scope": ["knee"],
        "intensity": "extreme",
        "duration_sec_estimate": -2.0,
        "anchor_pose": "nonexistent_pose",
        "rule_entry": "yes",
    }
    errors = _collect_errors(description, {"demo"})
    joined = " ".join(errors)
    assert "motion_scope" in joined
    assert "intensity" in joined
    assert "duration_sec_estimate" in joined
    assert "anchor_pose" in joined
    assert "rule_entry" in joined


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", {"text": "demo"}),
        ("category", True),
        ("intensity", False),
        ("anchor_pose", 123),
    ],
)
def test_validator_rejects_non_string_description_fields(field, value):
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "intensity": "subtle",
    }
    description[field] = value

    errors = _collect_errors(description, {"demo"})

    assert any(field in error and "string" in error for error in errors)


@pytest.mark.parametrize("duration", [True, "1.5", float("nan"), float("inf"), float("-inf"), 10**1000])
def test_validator_rejects_non_finite_or_non_numeric_duration(duration):
    description = {
        "summary": "demo",
        "category": "demo",
        "when_to_use": ["x"],
        "duration_sec_estimate": duration,
    }

    errors = _collect_errors(description, {"demo"})

    assert any("duration_sec_estimate must be a finite number" in error for error in errors)


def test_validator_rejects_missing_description():
    errors = _collect_errors(None, {"demo"})
    assert any("description is required" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"].pop(0),
            "must have a preceding move_to_joint_positions arm entry",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][0].update(duration_sec=0.0),
            "duration_sec must be greater than zero",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][0]["joint_positions"].update({"5": 0.0}),
            "must match the first joint waypoint",
        ),
    ],
)
def test_canonical_loader_rejects_unsafe_absolute_trajectory_entry(tmp_path, mutation, message):
    path = _write_config(tmp_path, mutation)

    with pytest.raises(ValueError, match=message):
        load_robot_config_dict(path)


def test_canonical_loader_rejects_unknown_description_redirect(tmp_path):
    def mutate(skills):
        skills["wave_hello"]["description"]["do_not_use"][0]["instead_use"] = "ghost_skill"

    with pytest.raises(ValueError, match="ghost_skill"):
        load_robot_config_dict(_write_config(tmp_path, mutate))


def test_typed_loader_rejects_unknown_description_redirect(tmp_path):
    def mutate(skills):
        skills["wave_hello"]["description"]["do_not_use"][0]["instead_use"] = "ghost_skill"

    with pytest.raises(ValueError, match="ghost_skill"):
        load_robot_config(_write_config(tmp_path, mutate))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][-1]["joint_positions"].update(
                {"5": float("nan")}
            ),
            "joint_positions.*finite",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][0].update(duration_sec=float("inf")),
            "duration_sec must be a finite number",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][0].update(duration_sec=10**1000),
            "duration_sec must be a finite number",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][0]["joint_positions"].update({"1": 10**1000}),
            "joint_positions.*finite",
        ),
        (
            lambda skills: skills["wave_hello"]["primitive_sequence"][1]["trajectory_template"].update(
                waypoint_duration_sec=float("nan")
            ),
            "waypoint_duration_sec must be a finite number",
        ),
    ],
)
def test_canonical_loader_rejects_non_finite_primitive_numbers(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        load_robot_config_dict(_write_config(tmp_path, mutation))


def test_canonical_loader_rejects_unknown_planner_skill(tmp_path):
    data = yaml.safe_load(SO101.read_text(encoding="utf-8"))
    data["robot"]["embodied"]["planner"]["planning_policy"]["allowed_skills"].append("ghost_skill")
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_skills.*ghost_skill"):
        load_robot_config_dict(path)
