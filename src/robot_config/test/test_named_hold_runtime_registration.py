"""The feature owns its AimDK registration argument; disabled launches stay empty."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_config.launch_builders.runtime import _named_hold_json


@pytest.mark.parametrize(
    "section", [None, {}, {"enabled": False, "motion_enabled": True}, {"enabled": True, "motion_enabled": False}]
)
def test_disabled_feature_does_not_resolve_or_read_any_assets(section, monkeypatch):
    import robot_config.utils

    monkeypatch.setattr(robot_config.utils, "resolve_ros_path", lambda _path: pytest.fail("disabled asset lookup"))
    config = {} if section is None else {"preset_motions": section}
    assert _named_hold_json(config, "aimdk_robot") == ""


def test_enabled_feature_passes_exact_same_asset_paths_and_name(tmp_path):
    template, geometry = tmp_path / "pose.yaml", tmp_path / "geometry.json"
    template.write_text("not a certificate; provider must validate")
    geometry.write_text("{}")
    config = {
        "preset_motions": {
            "enabled": True,
            "motion_enabled": True,
            "motion_name": "overhead_heart",
            "template_file": str(template),
            "geometry_file": str(geometry),
        }
    }
    assert json.loads(_named_hold_json(config, "aimdk_robot")) == {
        "motion_name": "overhead_heart",
        "template_file": str(template),
        "geometry_file": str(geometry),
    }
    with pytest.raises(ValueError, match="runtime.provider=aimdk_robot"):
        _named_hold_json(config, "so101_robot")


def test_enabled_feature_rejects_missing_assets_instead_of_unknown_runtime_name():
    with pytest.raises(ValueError, match="template_file"):
        _named_hold_json({"preset_motions": {"enabled": True, "motion_enabled": True}}, "aimdk_robot")


@pytest.mark.parametrize("enabled", [False, True])
def test_provider_include_forwards_registration_only_for_the_enabled_feature(tmp_path, monkeypatch, enabled):
    from robot_config.launch_builders import runtime as builder

    template, geometry = tmp_path / "pose.yaml", tmp_path / "geometry.json"
    template.write_text("provider independently checks this asset")
    geometry.write_text("{}")
    monkeypatch.setattr(builder, "resolve_runtime_launch", lambda _provider: "/unused/runtime.launch.py")
    monkeypatch.setattr(builder, "_write_peripherals_fragment", lambda _config: "")
    monkeypatch.setattr(builder, "_cbf_safety_json", lambda _config: "")
    monkeypatch.setattr(builder, "runtime_ready_waiter", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(builder, "IncludeLaunchDescription", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    config = {
        "runtime": {"provider": "aimdk_robot", "profile": "x2_ultra_omnihand"},
        "preset_motions": {
            "enabled": True,
            "motion_enabled": enabled,
            "template_file": str(template),
            "geometry_file": str(geometry),
            "motion_name": "overhead_heart",
        },
    }
    actions, _waiter = builder.generate_runtime_provider_actions(config, use_sim=False)
    arguments = dict(actions[0].launch_arguments)
    assert ("named_hold_json" in arguments) is enabled
    assert arguments["profile"] == "x2_ultra_omnihand"
    if enabled:
        assert json.loads(arguments["named_hold_json"])["geometry_file"] == str(geometry)


def test_committed_recording_is_only_the_frozen_visual_pose():
    path = Path(__file__).resolve().parents[1] / "config/preset_motions/overhead_heart.yaml"
    template = yaml.safe_load(path.read_text())
    recording = json.loads((path.parent / template["recording_file"]).read_text())
    assert template["validated"] is template["motion_enabled"] is False
    assert template["required_head_clearance_m"] == 0.20
    assert recording["validated"] is False
    assert recording["time_from_start_s"] == [0.0, 1.0]
    assert recording["arms"]["positions_rad"] == [template["arm_target_rad"]] * 2
    assert recording["hands"]["positions"] == [template["hand_target_rad"]] * 2
    assert recording["arms"]["joints"] == template["arm_joint_names"]
    assert recording["hands"]["joints"] == [
        f"{'left' if name.startswith('L_') else 'right'}_{name[2:].removesuffix('_joint')}"
        for name in template["hand_joint_names"]
    ]
    assert len(recording["arms"]["joints"]) == 14 and len(recording["hands"]["joints"]) == 20
    assert recording["visual_version"] == template["visual_version"]
    assert recording["source_candidate_sha256"] == template["visual_approval"]["source_candidate_sha256"]
