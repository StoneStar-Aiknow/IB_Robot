"""The X2 free-pose Agent deployment: the gesture stack plus the pose composer."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robots" / "aimdk_x2_free_pose.yaml"


@pytest.fixture(scope="module")
def config():
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


def test_configuration_is_valid(config):
    validate_robot_config_dict(config, deferred_interfaces=True)


def test_it_adds_free_pose_to_the_gestures_without_an_arm_surface(config):
    assert config["name"] == "aimdk_x2_skills"
    assert config["runtime"]["profile"] == "x2_ultra_omnihand"
    assert config["embodied"]["arm_surface"] is False
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_free_pose"
    assert "free_pose" in config["embodied"]["agent"]["test_allowlist"]


def test_the_composer_model_key_is_never_literal(config):
    llm = config["pose_composer"]["llm"]
    assert config["pose_composer"]["enabled"] is True
    assert "api_key" not in llm


def test_the_catalog_compiles_with_the_pose_composer_executor(config):
    from robot_skill_cli.catalog import compile_local_snapshot

    snapshot = compile_local_snapshot(config, CONFIG)
    assert "free_pose" in snapshot.planner_visible_skill_names
    assert snapshot.templates["free_pose"]["executor"] == "pose_composer"
    assert "pose_composer" in snapshot.delegated_executors


def test_the_pose_memory_is_on_with_seeds_without_a_verdict_phrase_table(config):
    memory = config["pose_composer"]["memory"]
    assert memory["enabled"] is True and memory["seeds"] is True
    assert "feedback" not in memory
