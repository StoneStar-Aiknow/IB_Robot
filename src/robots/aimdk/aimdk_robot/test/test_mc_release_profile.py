"""Contract checks for the separately delivered MC release, without external assets."""

from pathlib import Path

import pytest
from aimdk_robot import dataplane, projection

from robot_runtime.interface_description import build_description
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.profile import load_profile

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_mc.yaml"


def test_release_profile_does_not_advertise_unsupported_upper_body_control():
    profile = load_profile(PROFILE)
    description = build_description(profile)
    assert "joint.position_stream" not in profile["capabilities"]
    assert "gripper.1d" not in profile["capabilities"]
    assert not any(name.startswith("joint.") and name.endswith("stream") for name in description["interfaces"])
    modes = ModeModel(ModeModelConfig.from_profile(profile))
    assert not modes.can_switch("stream").allowed
    assert not modes.can_switch("head").allowed
    assert projection.mode_action(profile, "locomotion") == "LOCOMOTION_DEFAULT"
    assert projection.mode_action(profile, "step") == "LOCOMOTION_STEP"
    assert not modes.can_switch("prepare").allowed
    assert not modes.can_switch("sit_up").allowed
    assert "runtime.stop" not in profile["capabilities"]
    assert description["interfaces"]["runtime.stop"]["endpoint"] == "/runtime/stop"
    assert not dataplane.plan(profile)["upper_body_command"]["topic"]

    profile["vendor"]["mc_actions"]["stream"] = "UPPERBODY_REMOTE_SPLIT"
    with pytest.raises(projection.ProjectionError):
        projection.mode_action(profile, "stream")


def test_step_mode_is_not_enabled_for_the_original_hardware_profile():
    profile = {"vendor": {"mc_actions": {"step": "LOCOMOTION_STEP"}}}
    with pytest.raises(projection.ProjectionError):
        projection.mode_action(profile, "step")


@pytest.mark.parametrize("use_sim", [False, True])
def test_release_odometry_contract_does_not_assert_a_vendor_frame(use_sim):
    profile = load_profile(PROFILE)
    description = build_description(profile, use_sim=use_sim)
    assert "base.odom" not in profile["capabilities"]
    assert "base.odom" not in description["interfaces"]
    odometry = description["interfaces"]["x2.mc.leg_odometry"]
    assert odometry["endpoint"] == "/aimdk/leg_odometry"
    assert odometry["message_type"] == "nav_msgs/msg/Odometry"
    assert odometry["frame_id"] is None


def test_mc_queries_are_x2_extensions_not_generic_motion_interfaces():
    interfaces = build_description(load_profile(PROFILE))["interfaces"]
    assert "motion.state" not in interfaces
    assert "motion.locomotion_profiles" not in interfaces
    for name, service in (("state", "GetMotionState"), ("locomotion_profiles", "GetLocomotionProfiles")):
        interface = interfaces[f"x2.mc.{name}"]
        assert interface["message_type"] == f"aimdk_interfaces/srv/{service}"
        assert interface["endpoint"].startswith("/aimdk/mc/")


def test_release_advertised_motions_are_dispatchable_and_no_risky_aliases_exist():
    profile = load_profile(PROFILE)
    vendor = profile["vendor"]
    assert set(profile["capabilities"]["motion.named"]["names"]) == set(vendor["preset_motions"])
    assert set(profile["capabilities"]["motion.posture"]["postures"]) == set(vendor["postures"])
    assert "stand_up" not in vendor["postures"]
    assert "ascend_stairs" not in vendor["postures"]
    assert not set(vendor["mc_actions"].values()) & {"HEAD_ONLY", "UPPERBODY_REMOTE_SPLIT"}
