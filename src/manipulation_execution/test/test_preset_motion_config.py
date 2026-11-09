"""Optional observer wiring preserves the disabled baseline and rejects unsafe opt-in."""

import hashlib
import json

import pytest

from manipulation_execution.preset_motions.config import launch_parameters, matcher_settings, motion_parameters

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


def test_missing_or_disabled_motion_does_not_read_assets():
    assert motion_parameters({}) is None
    assert (
        motion_parameters({"preset_motions": {"enabled": True, "motion_enabled": False, "template_file": "/missing"}})
        is None
    )


def test_visual_approval_alone_cannot_enable_motion(tmp_path):
    import yaml

    template = tmp_path / "pose.yaml"
    geometry = tmp_path / "geometry.json"
    template.write_text(
        yaml.safe_dump(
            {
                "name": "overhead_heart",
                "validated": False,
                "visual_approval": {"approved": True},
                "required_head_clearance_m": 0.2,
            }
        )
    )
    geometry.write_text(json.dumps({"validated": True}))
    settings = {
        "preset_motions": {
            "enabled": True,
            "motion_enabled": True,
            "template_file": str(template),
            "geometry_file": str(geometry),
        }
    }
    with pytest.raises(ValueError, match="unvalidated"):
        motion_parameters(settings)


def test_motion_requires_unchanged_clearance_and_a_certified_stop_bridge(tmp_path):
    import yaml

    template = tmp_path / "pose.yaml"
    geometry = tmp_path / "geometry.json"
    record = {
        "name": "overhead_heart",
        "validated": True,
        "required_head_clearance_m": 0.2,
        "visual_approval": {"approved": True},
    }
    shape = {
        "name": "overhead_heart",
        "validated": True,
        "required_head_clearance_m": 0.2,
        "head_envelope_covers_allowed_motion": True,
        "complete_articulated_hands": True,
        "continuous_motion_bounds_verified": True,
    }
    shape["sha256"] = hashlib.sha256(json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    geometry.write_text(json.dumps(shape))
    template.write_text(yaml.safe_dump(record))
    settings = {
        "preset_motions": {
            "enabled": True,
            "motion_enabled": True,
            "template_file": str(template),
            "geometry_file": str(geometry),
        }
    }
    with pytest.raises(ValueError, match="stop bridge"):
        motion_parameters(settings)
    record["required_head_clearance_m"] = 0.15
    template.write_text(yaml.safe_dump(record))
    with pytest.raises(ValueError, match="0.20"):
        motion_parameters(settings)
