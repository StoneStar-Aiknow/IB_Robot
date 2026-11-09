"""Optional observer wiring preserves the disabled baseline and rejects unsafe opt-in."""

import json

import pytest

from manipulation_execution.preset_motions.config import launch_parameters, matcher_settings

STREAM = {"frame_topic": "/pear", "actor_state_topic": "/actor"}


def test_absent_or_disabled_feature_adds_no_observer():
    assert launch_parameters({}, {}) is None
    assert launch_parameters({"preset_motions": {"enabled": False}}, {}) is None


def test_enabled_observer_uses_existing_stream_and_disables_motion():
    result = launch_parameters({"preset_motions": {"enabled": True, "matcher": {"enter_confirm_s": 0.5}}}, STREAM)
    assert result["pear_frame_topic"] == "/pear"
    assert result["actor_state_topic"] == "/actor"
    assert result["motion_enabled"] is False
    assert json.loads(result["matcher_json"]) == {"enter_confirm_s": 0.5}


@pytest.mark.parametrize(
    "section", [True, {"enabled": "true"}, {"enabled": True, "motion_enabled": True}, {"enabled": True, "unknown": 1}]
)
def test_unsafe_or_ambiguous_config_is_rejected(section):
    with pytest.raises(ValueError):
        launch_parameters({"preset_motions": section}, STREAM)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_frame_age_s": float("nan")},
        {"unknown": 1},
        {"min_enter_frames": 1},
        {"rule": {"enter": {"wrist_span_max": False}}},
    ],
)
def test_bad_matcher_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        matcher_settings(settings)


def test_stream_endpoint_is_required():
    with pytest.raises(ValueError):
        launch_parameters({"preset_motions": {"enabled": True}}, {})
