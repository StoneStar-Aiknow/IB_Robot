"""Unit tests for the X2 projection rules.

These are the rules that decide what the robot is told to do, so they are
tested without ROS, without the vendor SDK and without a robot. The rules the
data plane applies at the platform's rate (command payloads, holds, velocity
limits, joint state aggregation) are tested with it: aimdk_dataplane's gtests.
"""

from __future__ import annotations

import pytest
from aimdk_robot import projection
from aimdk_robot.projection import CommandRejected, LocomotionLimits, ProjectionError

# --- locomotion -------------------------------------------------------------


def test_locomotion_limits_from_profile():
    limits = LocomotionLimits.from_profile({"min_linear_mps": 0.3, "max_linear_mps": 0.9})
    assert limits.min_linear_mps == 0.3
    assert limits.max_linear_mps == 0.9
    assert limits.min_angular_radps == 0.1


# --- freshness --------------------------------------------------------------


def test_stamp_outside_the_window_is_stale():
    assert projection.is_fresh(100.0, 100.1, 0.2)
    assert not projection.is_fresh(100.0, 100.5, 0.2)
    # Stamp 0 means "use receive time" on this platform.
    assert projection.is_fresh(0.0, 100.0, 0.2)


# --- modes ------------------------------------------------------------------


def test_mode_action_resolves_from_profile():
    profile = {"vendor": {"mc_actions": {"stream": "UPPERBODY_REMOTE_SPLIT"}}}
    assert projection.mode_action(profile, "stream") == "UPPERBODY_REMOTE_SPLIT"


def test_mode_action_rejects_unknown_mode_and_unknown_action():
    with pytest.raises(ProjectionError):
        projection.mode_action({"vendor": {"mc_actions": {}}}, "stream")
    with pytest.raises(ProjectionError):
        projection.mode_action({"vendor": {"mc_actions": {"stream": "NOT_AN_ACTION"}}}, "stream")


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (0, ""),
        (4, "INVALID_POSTURE"),
        (6, "SAFETY_FORBIDS_TRANSITION"),
        (8, "MOVING_BUSY"),
        (10, "NO_TRANSITION_PATH"),
    ],
)
def test_platform_rejection_codes_are_surfaced_verbatim(code, reason):
    assert projection.mode_rejection_reason(code) == reason


def test_unknown_rejection_code_is_reported_not_swallowed():
    assert projection.mode_rejection_reason(77) == "PLATFORM_CODE_77"


# --- stop -------------------------------------------------------------------


def test_hold_stop_switches_no_mode_when_the_platform_registers_no_hold_action():
    """Closing admission is the hold; no registered mode holds every posture."""
    plan = projection.resolve_stop_plan("HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={})
    assert plan.vendor_action == ""
    assert plan.latch and plan.release_arbitration


def test_hold_stop_uses_a_configured_action_when_one_is_registered():
    plan = projection.resolve_stop_plan(
        "HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={"hold_action": "JOINT_DEFAULT"}
    )
    assert plan.vendor_action == "JOINT_DEFAULT"


def test_stop_actions_the_firmware_does_not_register_are_refused():
    """The McAction enum is wider than the firmware's registered mode set.

    JOINT_FREEZE, SOFT_EMERGENCY_STOP and ZERO_TORQUE_DEFAULT are all in the
    IDL and none are registered; the platform answers code 3 for them. A
    profile naming one must fail here, not on the robot.
    """
    for name in ("JOINT_FREEZE", "SOFT_EMERGENCY_STOP", "ZERO_TORQUE_DEFAULT", "STAND_BODY_CONTROL"):
        assert name not in projection.VENDOR_MOTION_MODES
        with pytest.raises(ProjectionError):
            projection.resolve_stop_plan(
                "HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={"hold_action": name}
            )
        with pytest.raises(ProjectionError):
            projection.resolve_stop_plan(
                "TORQUE_OFF", body_pose=projection.BODY_POSE_STAND, stop_config={"soft_estop_action": name}
            )


def test_torque_off_defaults_to_the_platforms_damping_mode_and_says_so():
    """DAMPING_DEFAULT is registered and "always permitted"; the enum's soft
    e-stop name is neither."""
    plan = projection.resolve_stop_plan("TORQUE_OFF", body_pose=projection.BODY_POSE_STAND, stop_config={})
    assert plan.vendor_action == "DAMPING_DEFAULT"
    assert plan.downgraded_from == "TORQUE_OFF"


def test_torque_off_while_standing_is_refused_when_zero_torque_is_configured():
    """Removing torque from a balancing biped drops it."""
    with pytest.raises(CommandRejected) as excinfo:
        projection.resolve_stop_plan(
            "TORQUE_OFF",
            body_pose=projection.BODY_POSE_STAND,
            stop_config={"torque_off_policy": "zero_torque_when_seated"},
        )
    assert excinfo.value.reason == "TORQUE_OFF_UNSAFE_POSTURE"
    assert "stand" in excinfo.value.detail


@pytest.mark.parametrize(
    "pose",
    [projection.BODY_POSE_SIT, projection.BODY_POSE_SQUAT, projection.BODY_POSE_LIE_FACE_UP],
)
def test_torque_off_is_allowed_in_a_supported_posture(pose):
    plan = projection.resolve_stop_plan(
        "TORQUE_OFF", body_pose=pose, stop_config={"torque_off_policy": "zero_torque_when_seated"}
    )
    assert plan.vendor_action == "PASSIVE_DEFAULT"
    assert plan.downgraded_from == ""


def test_unknown_stop_policy_is_a_configuration_error():
    with pytest.raises(ProjectionError):
        projection.resolve_stop_plan("COAST", body_pose=projection.BODY_POSE_STAND, stop_config={})


# --- interaction helpers ----------------------------------------------------


LEVELS = ((0, 1), (20, 2), (40, 4), (60, 6), (80, 7), (95, 8))


def test_priority_maps_onto_the_platform_levels_as_a_step_function():
    """Platform priorities are an enumeration: only listed values may come out."""
    assert projection.priority_level(0, levels=LEVELS) == 1
    assert projection.priority_level(19, levels=LEVELS) == 1
    assert projection.priority_level(20, levels=LEVELS) == 2
    assert projection.priority_level(50, levels=LEVELS) == 4
    assert projection.priority_level(79, levels=LEVELS) == 6
    assert projection.priority_level(94, levels=LEVELS) == 7
    assert projection.priority_level(100, levels=LEVELS) == 8
    legal = {value for _threshold, value in LEVELS}
    assert all(projection.priority_level(n, levels=LEVELS) in legal for n in range(101))
    # Order is preserved, and the top of the neutral range is the last listed
    # level: nothing above it (the platform's own safety layer) is reachable.
    outputs = [projection.priority_level(n, levels=LEVELS) for n in range(101)]
    assert outputs == sorted(outputs)
    assert max(outputs) == 8


def test_priority_outside_the_neutral_range_is_rejected():
    with pytest.raises(CommandRejected) as excinfo:
        projection.priority_level(120, levels=LEVELS)
    assert excinfo.value.reason == "PRIORITY_RANGE"


def test_priority_levels_must_be_ordered_and_start_at_zero():
    with pytest.raises(ProjectionError):
        projection.priority_level(10, levels=((5, 1), (20, 2)))
    with pytest.raises(ProjectionError):
        projection.priority_level(10, levels=((0, 1), (20, 2), (20, 4)))


def test_resolve_named_refuses_unknown_names_and_lists_the_known_ones():
    with pytest.raises(CommandRejected) as excinfo:
        projection.resolve_named("moonwalk", {"wave": 1002}, reason="UNKNOWN_MOTION")
    assert excinfo.value.reason == "UNKNOWN_MOTION"
    assert "wave" in excinfo.value.detail


# --- vendor hand types ------------------------------------------------------


def test_hand_type_families_cover_the_vendor_enumeration():
    """HandType.msg has six values; only four name an end-effector family."""
    from aimdk_robot import vendor_gateway as vg

    assert vg.hand_type_family(vg.HAND_TYPE_CLAW) == "claw"
    assert vg.hand_type_family(vg.HAND_TYPE_NIMBLE) == "dexterous"
    assert vg.hand_type_family(vg.HAND_TYPE_LEISAI_NIMBLE) == "dexterous"
    assert vg.hand_type_family(vg.HAND_TYPE_LITE_S) == "dexterous"
    # Absence of hand state and a hand fault are not families: they can
    # neither confirm nor contradict what a profile declares.
    assert vg.hand_type_family(vg.HAND_TYPE_NONE) is None
    assert vg.hand_type_family(vg.HAND_TYPE_ERROR) is None
    # A value a newer firmware adds is unknown, not a mismatch.
    assert vg.hand_type_family(0x7E) is None
    assert vg.hand_type_name(vg.HAND_TYPE_NONE) == "none"
    assert vg.hand_type_name(0x7E) == "0x7e"


# --- vendor QoS -------------------------------------------------------------


def test_vendor_qos_takes_the_profile_value_then_the_default():
    config = {"audio_playback": {"reliability": "reliable"}, "mc_state": {"durability": "transient_local", "depth": 5}}
    assert projection.vendor_qos(config, "audio_playback") == ("reliable", "volatile", 10)
    assert projection.vendor_qos(config, "mc_state") == ("best_effort", "transient_local", 5)
    # An endpoint the profile does not name falls back to the caller's default.
    assert projection.vendor_qos(config, "other", reliability="reliable") == ("reliable", "volatile", 10)
    assert projection.vendor_qos(None, "other") == ("best_effort", "volatile", 10)


def test_vendor_qos_rejects_values_that_would_silently_mismatch():
    """A typo must fail loudly: a wrong QoS drops every message with no error."""
    for bad in ({"a": {"reliability": "RELIABLE"}}, {"a": {"durability": "latched"}}, {"a": {"depth": 0}}):
        with pytest.raises(ProjectionError):
            projection.vendor_qos(bad, "a")


# --- audio focus ------------------------------------------------------------


def test_audio_focus_defaults_to_the_platform_priority_band():
    assert projection.audio_focus_request(None) == (6, 0, 1.0, 50)
    assert projection.audio_focus_request({"focus_priority": 8, "focus_release_idle_s": 0.2})[0] == 8
    assert projection.audio_focus_request({"focus_release_idle_s": 0.2})[2] == 0.2


def test_audio_focus_rejects_priorities_the_platform_cannot_express():
    for bad in ({"focus_priority": 0}, {"focus_priority": 11}, {"focus_priority_weight": 100}):
        with pytest.raises(ProjectionError):
            projection.audio_focus_request(bad)
    for bad in ({"focus_release_idle_s": 0.0}, {"focus_buffer_chunks": 0}):
        with pytest.raises(ProjectionError):
            projection.audio_focus_request(bad)


# --- clock agreement --------------------------------------------------------


def test_clock_skew_estimate_is_the_minimum_because_delay_only_adds():
    """A late callback inflates its own sample; it must not move the estimate.

    Each sample is offset + transport/queueing delay, and that delay is always
    positive. One callback stuck behind a burst of joint feedback therefore
    reports a large difference while the clocks are fine — which is how a
    service call on a busy node looked like a clock that had drifted.
    """
    assert projection.clock_skew_estimate([]) is None
    # Clocks agree; one callback waited 1.2 s behind other work.
    assert projection.clock_skew_estimate([0.004, 0.006, 1.2, 0.005]) == pytest.approx(0.004)
    # A real offset survives, because every sample carries it.
    assert projection.clock_skew_estimate([12.4, 12.9, 13.6]) == pytest.approx(12.4)
    # A board that lags reads negative and is just as real.
    assert projection.clock_skew_estimate([-12.4, -12.1, -11.0]) == pytest.approx(-12.4)


def test_a_delayed_callback_alone_never_raises_a_clock_fault():
    """The whole point: busy != skewed."""
    estimate = projection.clock_skew_estimate([0.003, 0.9, 2.5, 0.004])
    assert projection.clock_skew_fault(estimate, limit_s=0.1, window_s=0.2) is None


def test_clock_skew_within_the_limit_is_not_a_fault():
    assert projection.clock_skew_fault(None, limit_s=0.1, window_s=0.2) is None
    assert projection.clock_skew_fault(0.05, limit_s=0.1, window_s=0.2) is None
    assert projection.clock_skew_fault(-0.05, limit_s=0.1, window_s=0.2) is None


def test_clock_skew_beyond_the_limit_names_the_silent_failure_it_causes():
    """The platform drops out-of-window commands without a word; say so."""
    detail = projection.clock_skew_fault(12.4, limit_s=0.1, window_s=0.2)
    assert detail is not None
    assert "+12.400s" in detail
    assert "discarded by the platform" in detail
    # A clock that lags is as fatal as one that leads.
    assert projection.clock_skew_fault(-12.4, limit_s=0.1, window_s=0.2) is not None


# --- interaction priority scales --------------------------------------------


def test_screen_priority_never_hides_a_fault_indication_below_the_top_band():
    """The platform shows its own fault indications at 8-10.

    A neutral priority passed through unmapped would sit above them for most of
    the contract's range and silently hide over-temperature and fall-protection
    indications from the operator. Only the explicit top band may do that.
    """
    vg = pytest.importorskip("aimdk_robot.vendor_gateway")
    levels = vg.SCREEN_PRIORITY_LEVELS
    for neutral in (0, 25, 50, 79, 90, 94):
        assert projection.priority_level(neutral, levels=levels) < 8, neutral
    # Only the deliberate top of the scale overrides a fault indication.
    assert projection.priority_level(95, levels=levels) > 10
    assert projection.priority_level(100, levels=levels) > 10


def test_led_priority_stays_in_a_band_that_leaves_headroom():
    """Each accepted request raises the platform's threshold; 100 must not pin it."""
    vg = pytest.importorskip("aimdk_robot.vendor_gateway")
    levels = vg.LED_PRIORITY_LEVELS
    values = [projection.priority_level(n, levels=levels) for n in (0, 50, 100)]
    assert values == sorted(values)
    assert max(values) <= 10, "a neutral 100 must not ratchet the threshold out of reach"


# --- platform state fields --------------------------------------------------


def test_fsm_diagnostic_preserves_the_raw_vendor_value():
    """FSM values are observations, not lifecycle or command-admission rules."""
    assert projection.fsm_diagnostic(4) == "platform balance controller reported fsm_state=4"


def test_an_unreported_speed_envelope_is_not_usable():
    """All-zero bounds mean "not reported": the runtime passes no envelope to the data plane for them."""
    assert not projection.SpeedEnvelope().usable
    assert projection.SpeedEnvelope(mode=1, max_linear_mps=0.4, max_angular_radps=0.3).usable
