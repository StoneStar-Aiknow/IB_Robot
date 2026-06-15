"""Tests for VLMTaskPlannerNode JSON parameter helpers."""

from __future__ import annotations

import json

import pytest

# Avoid importing the full node (which declares ROS parameters); reach in to
# the static helper directly.  ``_load_allowed_skills`` is independent of rclpy.
from vlm_task_planner.vlm_task_planner_node import VLMTaskPlannerNode


def test_load_allowed_skills_empty_returns_empty_list():
    assert VLMTaskPlannerNode._load_allowed_skills("") == []
    assert VLMTaskPlannerNode._load_allowed_skills("   ") == []


def test_load_allowed_skills_parses_array():
    raw = json.dumps(["inspect_scene", "dance_basic"])
    assert VLMTaskPlannerNode._load_allowed_skills(raw) == [
        "inspect_scene",
        "dance_basic",
    ]


def test_load_allowed_skills_coerces_items_to_strings():
    raw = json.dumps([1, 2, 3])
    assert VLMTaskPlannerNode._load_allowed_skills(raw) == ["1", "2", "3"]


def test_load_allowed_skills_rejects_invalid_json():
    with pytest.raises(ValueError) as exc_info:
        VLMTaskPlannerNode._load_allowed_skills("not-json")
    assert "invalid allowed_skills_json" in str(exc_info.value)


def test_load_allowed_skills_rejects_non_array():
    with pytest.raises(ValueError):
        VLMTaskPlannerNode._load_allowed_skills('{"skill": "pick"}')
