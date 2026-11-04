"""A deployment may state the control modes its runtime can switch per skill.

The X2 compute pack has no navigation service, so ``nav_stage: hybrid`` (the
only other source of per-skill modes) is not available there while gestures and
the mirror still need two different modes. These cases pin the override, the
untouched hybrid path, and the validation that keeps a typo from becoming a
deployment that quietly loses its skill catalog.
"""

from robot_config.loader import _validate_skill_gateway_config, robot_supported_control_modes

HYBRID_ROBOT = {
    "nav_stage": "hybrid",
    "motion_mode": {
        "manipulation_control_mode": "moveit_planning",
        "navigation_control_mode": "base_navigation",
    },
}


def _override_errors(robot_config: dict) -> list[str]:
    return [e for e in _validate_skill_gateway_config(robot_config) if "skill_supported_control_modes" in e]


def test_override_states_the_modes_without_a_hybrid_navigation_stage():
    robot_config = {
        "control_modes": {"named_motion": {}, "model_inference": {}},
        "skill_supported_control_modes": ["named_motion", "model_inference"],
    }
    assert robot_supported_control_modes(robot_config) == ("named_motion", "model_inference")


def test_override_is_trimmed_and_deduplicated():
    robot_config = {"skill_supported_control_modes": [" named_motion ", "named_motion", "moveit_planning"]}
    assert robot_supported_control_modes(robot_config) == ("named_motion", "moveit_planning")


def test_without_an_override_a_non_hybrid_deployment_states_no_modes():
    assert robot_supported_control_modes({"control_modes": {"named_motion": {}}}) == ()


def test_without_an_override_the_hybrid_pair_still_comes_from_motion_mode():
    assert robot_supported_control_modes(HYBRID_ROBOT) == ("moveit_planning", "base_navigation")


def test_a_valid_override_is_accepted():
    robot_config = {
        "control_modes": {"named_motion": {}, "model_inference": {}},
        "skill_supported_control_modes": ["named_motion", "model_inference"],
    }
    assert _override_errors(robot_config) == []


def test_an_empty_override_is_rejected():
    assert _override_errors({"control_modes": {"named_motion": {}}, "skill_supported_control_modes": []})


def test_an_override_entry_outside_control_modes_is_rejected():
    robot_config = {"control_modes": {"named_motion": {}}, "skill_supported_control_modes": ["named_motion", "typo"]}
    errors = _override_errors(robot_config)
    assert any("typo" in error for error in errors)


def test_an_override_without_the_control_modes_mapping_is_rejected():
    assert _override_errors({"skill_supported_control_modes": ["named_motion"]})
