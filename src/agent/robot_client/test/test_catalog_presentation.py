import copy

import pytest

from robot_client.catalog import describe_skill, list_skills, planner_catalog_view


@pytest.fixture
def view():
    return {
        "robot_name": "test_robot",
        "capability_digest": "a" * 64,
        "timeout_policy": {"task_budget_sec": 30.0},
        "skills": [
            {
                "name": "vendor_kiss",
                "display_name": "双手飞吻",
                "tags": ["preset"],
                "summary": "双手飞吻",
                "domain": "social",
                "moves_robot": True,
                "required_control_mode": "named_motion",
                "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                "recovery_policy": "never_retry",
                "timeout_sec": 15.0,
            }
        ],
    }


def test_list_and_describe_preserve_display_metadata_and_stable_id(view):
    listed = list_skills(view)["skills"][0]
    described = describe_skill(view, "vendor_kiss")
    for projected in (listed, described):
        assert projected["name"] == "vendor_kiss"
        assert projected["display_name"] == "双手飞吻"
        assert projected["tags"] == ["preset"]
        projected["tags"].append("modified")
    assert view["skills"][0]["tags"] == ["preset"]
    assert described["timeout_sec"] == 15.0
    assert described["recovery_policy"] == "never_retry"


def test_old_catalog_without_display_metadata_still_projects(view):
    original = copy.deepcopy(view)
    del view["skills"][0]["display_name"]
    del view["skills"][0]["tags"]
    for projected in (list_skills(view)["skills"][0], describe_skill(view, "vendor_kiss")):
        assert "display_name" not in projected
        assert "tags" not in projected
        assert projected["name"] == original["skills"][0]["name"]


def test_display_metadata_does_not_change_planner_contract(view):
    planned = planner_catalog_view(view)["skills"][0]
    assert planned["name"] == "vendor_kiss"
    assert planned["summary"] == "双手飞吻"
    assert "display_name" not in planned
    assert "tags" not in planned
