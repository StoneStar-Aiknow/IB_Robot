"""Unit tests for the G1 projection rules.

These are the rules that decide what the robot is told to do, so they are
tested without ROS, without the vendor SDK and without a robot. Every fixture
value is a plain Python value shaped like the vendored unitree_hg /
unitree_api IDL the projection produces and consumes.

Regression coverage demanded by the phase-1 review:

* ``low_state_joint_positions`` must refuse a short LowState vector instead
  of raising ``TypeError`` (``len(seq) < dict`` is unordered comparison).
* ``dex3_handcmd(timeout=True)`` must zero kp AND kd on every slot — a
  released hand with a latched kp still pulls.
* ``is_fresh`` must treat stamp 0 ("never reported") as stale — treating it
  as fresh would let stream entry latch positions that do not exist (D4).
"""

from __future__ import annotations

import json

import pytest
from unitree_g1_robot import projection
from unitree_g1_robot.projection import CommandRejected, LocomotionLimits, ProjectionError


def _frame_latch(value_for: callable) -> dict[str, float]:
    return {name: value_for(index, name) for index, name in enumerate(projection.ARM_SDK_FRAME)}


# --- dex3 naming ---------------------------------------------------------------


def test_dex3_joint_names_prefix_the_urdf_side():
    assert projection.dex3_joint_names("left") == (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
    )
    assert projection.dex3_joint_names("right")[0] == "right_hand_thumb_0_joint"


def test_dex3_joint_names_rejects_unknown_side():
    with pytest.raises(ProjectionError):
        projection.dex3_joint_names("middle")


# --- joint target projection ----------------------------------------------------


def test_in_range_target_passes_unchanged():
    limits = {"lower": -1.047198, "upper": 1.047198}
    projection.check_joint_target("left_hand_thumb_0_joint", 0.0, limits)
    # The bounds themselves are valid targets; nothing is nudged inside.
    projection.check_joint_target("left_hand_thumb_0_joint", limits["upper"], limits)
    projection.check_joint_target("left_hand_thumb_0_joint", limits["lower"], limits)


@pytest.mark.parametrize("value", [-1.05, 1.05, float("inf"), float("-inf")])
def test_out_of_range_target_is_rejected_not_clamped(value):
    """A clamp would execute a motion the caller did not ask for."""
    with pytest.raises(CommandRejected) as excinfo:
        projection.check_joint_target("left_hand_thumb_0_joint", value, {"lower": -1.047198, "upper": 1.047198})
    assert excinfo.value.reason == "JOINT_TARGET_RANGE"


def test_channel_targets_maps_values_onto_joint_names():
    joints = ["left_hand_thumb_0_joint", "left_hand_index_1_joint"]
    limits = {
        "left_hand_thumb_0_joint": {"lower": -1.047198, "upper": 1.047198},
        "left_hand_index_1_joint": {"lower": -1.745329, "upper": 0.0},
    }
    targets = projection.channel_targets(
        [0.1, -0.2], channel_joints=joints, joint_limits=limits, label="hand_left_stream"
    )
    assert targets == {
        "left_hand_thumb_0_joint": 0.1,
        "left_hand_index_1_joint": -0.2,
    }


def test_channel_targets_rejects_wrong_length():
    joints = ["a_joint", "b_joint"]
    limits = {name: {"lower": -1.0, "upper": 1.0} for name in joints}
    with pytest.raises(CommandRejected) as excinfo:
        projection.channel_targets([0.0], channel_joints=joints, joint_limits=limits, label="hand_left_stream")
    assert excinfo.value.reason == "HAND_LEFT_STREAM_LENGTH"


def test_channel_targets_rejects_joints_without_urdf_limits():
    with pytest.raises(ProjectionError):
        projection.channel_targets([0.0], channel_joints=["mystery_joint"], joint_limits={}, label="arm_stream")


@pytest.mark.parametrize("values", [["x", 0.0], [None, 0.0]])
def test_non_numeric_command_is_rejected(values):
    limits = {"a_joint": {"lower": -1.0, "upper": 1.0}, "b_joint": {"lower": -1.0, "upper": 1.0}}
    with pytest.raises(CommandRejected) as excinfo:
        projection.channel_targets(
            values, channel_joints=["a_joint", "b_joint"], joint_limits=limits, label="arm_stream"
        )
    assert excinfo.value.reason == "NON_NUMERIC_COMMAND"


def test_non_finite_command_is_rejected():
    limits = {"a_joint": {"lower": -1.0, "upper": 1.0}, "b_joint": {"lower": -1.0, "upper": 1.0}}
    with pytest.raises(CommandRejected) as excinfo:
        projection.channel_targets(
            [float("nan"), 0.0], channel_joints=["a_joint", "b_joint"], joint_limits=limits, label="arm_stream"
        )
    assert excinfo.value.reason == "NON_FINITE_COMMAND"


# --- latch (design D4) -----------------------------------------------------------


def test_merge_latch_overwrites_exactly_the_updated_slots():
    entry = {"left_shoulder_pitch_joint": 0.1, "waist_yaw_joint": 0.0}
    merged = projection.merge_latch(entry, {"left_shoulder_pitch_joint": 0.5})
    assert merged == {"left_shoulder_pitch_joint": 0.5, "waist_yaw_joint": 0.0}


def test_stale_channel_keeps_the_last_commanded_value_not_the_entry_latch():
    """The confirmed D4 semantics: a stale channel holds its last command.

    The entry latch only seeds slots that never received a command. When a
    channel goes silent its slots must keep the last commanded value —
    republishing the entry value would move a joint back to where it was at
    mode entry, i.e. execute a motion nobody commanded (the same reason there
    is no per-tick measured refill).
    """
    entry = {"left_shoulder_pitch_joint": 0.1, "waist_yaw_joint": 0.0}
    commanded = projection.merge_latch(entry, {"left_shoulder_pitch_joint": 0.5})
    assert commanded["left_shoulder_pitch_joint"] == 0.5
    # Channel goes stale: the empty update republishes the previous latch as-is.
    held = projection.merge_latch(commanded, {})
    assert held == commanded
    assert held["left_shoulder_pitch_joint"] == 0.5


def test_merge_latch_does_not_mutate_its_inputs():
    entry = {"left_shoulder_pitch_joint": 0.1}
    update = {"left_shoulder_pitch_joint": 0.5}
    projection.merge_latch(entry, update)
    assert entry == {"left_shoulder_pitch_joint": 0.1}
    assert update == {"left_shoulder_pitch_joint": 0.5}


def test_capture_hold_targets_requires_every_joint_measured():
    order = ["left_shoulder_pitch_joint", "waist_yaw_joint"]
    measurements = {"left_shoulder_pitch_joint": 0.1, "waist_yaw_joint": 0.2}
    assert projection.capture_hold_targets(measurements, order=order) == [0.1, 0.2]


def test_capture_hold_targets_fails_closed_on_partial_measurement():
    """A partial measurement is no basis for a hold target; entry is refused."""
    order = ["left_shoulder_pitch_joint", "waist_yaw_joint"]
    assert projection.capture_hold_targets({"left_shoulder_pitch_joint": 0.1}, order=order) is None


# --- arm_sdk LowCmd ----------------------------------------------------------------


def test_motor_cmd_defaults_to_vendor_zeros():
    assert projection.motor_cmd() == {"mode": 0, "q": 0.0, "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0}


def test_arm_sdk_lowcmd_rides_the_weight_in_slot_29():
    latch = _frame_latch(lambda index, name: 0.1 * index)
    cmd = projection.arm_sdk_lowcmd(latch, weight=0.7, mode_pr=1, mode_machine=5)
    assert cmd["mode_pr"] == 1
    assert cmd["mode_machine"] == 5
    # The arm_sdk example never computes a CRC and the rt/lowcmd tier (which
    # requires one) stays disabled (D11).
    assert cmd["crc"] == 0
    assert cmd["reserve"] == [0, 0, 0, 0]
    assert len(cmd["motor_cmd"]) == projection.LOW_CMD_MOTOR_COUNT == 35
    weight_slot = cmd["motor_cmd"][projection.ARM_SDK_WEIGHT_INDEX]
    assert weight_slot["q"] == 0.7
    # The weight slot carries only the weight; gains stay zero.
    assert weight_slot["kp"] == 0.0
    assert weight_slot["kd"] == 0.0


def test_arm_sdk_lowcmd_commands_the_frame_at_lowstate_indices():
    latch = _frame_latch(lambda index, name: 0.1 * index)
    cmd = projection.arm_sdk_lowcmd(latch, weight=1.0, mode_pr=1, mode_machine=5)
    for name, index in projection.LOW_STATE_JOINT_INDEX.items():
        slot = cmd["motor_cmd"][index]
        if name in latch:
            assert slot["q"] == latch[name]
            assert slot["kp"] == projection.EXAMPLE_ARM_SDK_KP
            assert slot["kd"] == projection.EXAMPLE_ARM_SDK_KD
            assert slot["dq"] == 0.0
            assert slot["tau"] == 0.0
        else:
            # Leg slots stay at the vendor-default zeros, exactly as the
            # example leaves them: the vendor gates the whole message on the
            # shared weight, not per-joint modes.
            assert slot == projection.motor_cmd()
    # Slots beyond the joint index range carry nothing but the weight at 29.
    for index in range(30, projection.LOW_CMD_MOTOR_COUNT):
        assert cmd["motor_cmd"][index] == projection.motor_cmd()


def test_arm_sdk_lowcmd_refuses_a_partial_latch():
    """All 17 frame slots must carry a target: the weight is shared."""
    latch = _frame_latch(lambda index, name: 0.0)
    del latch["waist_pitch_joint"]
    with pytest.raises(ProjectionError):
        projection.arm_sdk_lowcmd(latch, weight=1.0, mode_pr=1, mode_machine=5)


@pytest.mark.parametrize("weight", [-0.1, 1.01, float("nan")])
def test_arm_sdk_lowcmd_rejects_weight_outside_unit_range(weight):
    latch = _frame_latch(lambda index, name: 0.0)
    with pytest.raises(CommandRejected) as excinfo:
        projection.arm_sdk_lowcmd(latch, weight=weight, mode_pr=1, mode_machine=5)
    assert excinfo.value.reason == "ARM_SDK_WEIGHT_RANGE"


# --- weight ramp -------------------------------------------------------------------


def test_ramp_advances_one_tick_toward_the_target():
    assert projection.ramp_weight(0.5, 1.0, rate_per_s=0.2, dt_s=0.02) == pytest.approx(0.504)


def test_ramp_clamps_at_the_target():
    assert projection.ramp_weight(0.999, 1.0, rate_per_s=0.2, dt_s=0.02) == 1.0


def test_ramp_descends_toward_zero():
    assert projection.ramp_weight(0.6, 0.0, rate_per_s=0.5, dt_s=0.1) == pytest.approx(0.55)


@pytest.mark.parametrize("kwargs", [{"dt_s": -0.01, "rate_per_s": 0.2}, {"dt_s": 0.01, "rate_per_s": -0.2}])
def test_ramp_rejects_negative_tick_or_rate(kwargs):
    with pytest.raises(ProjectionError):
        projection.ramp_weight(0.5, 1.0, **kwargs)


# --- Dex3 ---------------------------------------------------------------------------


def test_ris_mode_packs_id_status_timeout_bits():
    # id:4 | status:3 | timeout:1 (g1_dex3_example.cpp).
    assert projection.ris_mode_byte(3) == 0x13
    assert projection.ris_mode_byte(2, status=0) == 0x02
    assert projection.ris_mode_byte(15, status=7) == 0x7F
    # timeout keeps the default ENABLE status, so bit 4 stays set alongside bit 7.
    assert projection.ris_mode_byte(1, timeout=True) == 0x91
    assert projection.ris_mode_byte(0, timeout=True) == 0x90


def test_ris_mode_rejects_out_of_range_fields():
    with pytest.raises(ProjectionError):
        projection.ris_mode_byte(16)
    with pytest.raises(ProjectionError):
        projection.ris_mode_byte(0, status=8)


def test_dex3_handcmd_commands_only_latched_slots():
    targets = {"left_hand_thumb_0_joint": 0.5, "left_hand_index_1_joint": -0.2}
    cmd = projection.dex3_handcmd(targets, side="left")
    names = projection.dex3_joint_names("left")
    assert len(cmd["motor_cmd"]) == projection.DEX3_MOTORS_PER_HAND == 7
    for index, (name, slot) in enumerate(zip(names, cmd["motor_cmd"], strict=True)):
        commanded = name in targets
        assert slot["q"] == (targets[name] if commanded else 0.0)
        assert slot["kp"] == (projection.EXAMPLE_DEX3_KP if commanded else 0.0)
        assert slot["kd"] == (projection.EXAMPLE_DEX3_KD if commanded else 0.0)
        assert slot["mode"] == projection.ris_mode_byte(index)


def test_dex3_timeout_release_zeroes_gains_on_every_slot():
    """Reviewed bug fix: the example's stopMotors zeroes kp AND kd, all slots.

    A released hand whose latched kp survived would keep pulling on the
    fingers. The mode byte must carry the timeout flag (bit 7) on every slot
    alongside the zeroed gains.
    """
    targets = dict.fromkeys(projection.dex3_joint_names("left"), 0.5)
    cmd = projection.dex3_handcmd(targets, side="left", timeout=True)
    for index, slot in enumerate(cmd["motor_cmd"]):
        assert slot["kp"] == 0.0
        assert slot["kd"] == 0.0
        assert slot["mode"] & 0x80
        assert slot["mode"] == projection.ris_mode_byte(index, timeout=True)
        # Positions stay latched; the release removes effort, not targets.
        assert slot["q"] == 0.5


def test_dex3_handcmd_rejects_other_hand_joint_names():
    with pytest.raises(ProjectionError):
        projection.dex3_handcmd({"left_hand_thumb_0_joint": 0.1}, side="right")


# --- RPC ----------------------------------------------------------------------------


def test_rpc_request_correlates_identity_and_compacts_the_parameter():
    request = projection.rpc_request(7, 7101, {"data": 500})
    assert request["header"]["identity"] == {"id": 7, "api_id": 7101}
    # RequestLease carries only id; priority/noreply are RequestPolicy fields
    # (vendored unitree_api/msg/RequestLease.msg, RequestPolicy.msg).
    assert request["header"]["lease"] == {"id": 0}
    assert request["header"]["policy"] == {"priority": 0, "noreply": False}
    assert request["parameter"] == '{"data":500}'  # the SDK's ToJsonString shape
    assert request["binary"] == []


def test_rpc_request_passes_through_a_prebuilt_json_parameter():
    request = projection.rpc_request(3, 1001, '{"index":0,"speaker_id":0,"text":"hi"}')
    assert json.loads(request["parameter"]) == {"index": 0, "speaker_id": 0, "text": "hi"}


def test_rpc_request_rejects_non_positive_identity():
    with pytest.raises(ProjectionError):
        projection.rpc_request(0, 7101, {"data": 500})


def test_rpc_response_matches_on_identity_and_api_id():
    response = {"header": {"identity": {"id": 7, "api_id": 7101}, "status": {"code": 0}}}
    assert projection.rpc_response_matches(response, identity_id=7, api_id=7101)
    assert not projection.rpc_response_matches(response, identity_id=8, api_id=7101)
    assert not projection.rpc_response_matches(response, identity_id=7, api_id=7102)
    assert not projection.rpc_response_matches({}, identity_id=7, api_id=7101)
    assert projection.rpc_response_code(response) == 0


def test_rpc_response_data_parses_the_json_payload():
    assert projection.rpc_response_data({"data": '{"data":500}'}) == {"data": 500}


def test_rpc_topics_follow_the_rt_api_convention():
    assert projection.rpc_request_topic("sport") == "rt/api/sport/request"
    assert projection.rpc_response_topic("voice") == "rt/api/voice/response"


# --- RPC parameters (SDK Jsonize layouts) ----------------------------------------------


def test_fsm_id_parameter_is_a_jsonized_int():
    assert projection.fsm_id_parameter(projection.FSM_IDS["start"]) == {"data": 500}


def test_velocity_parameter_defaults_to_a_single_step():
    assert projection.velocity_parameter(0.1, 0.0, 0.0) == {"velocity": [0.1, 0.0, 0.0], "duration": 1.0}


def test_velocity_parameter_rejects_non_positive_duration():
    with pytest.raises(CommandRejected) as excinfo:
        projection.velocity_parameter(0.1, 0.0, 0.0, duration_s=0.0)
    assert excinfo.value.reason == "VELOCITY_DURATION"


def test_velocity_parameter_rejects_non_finite_components():
    with pytest.raises(CommandRejected) as excinfo:
        projection.velocity_parameter(float("nan"), 0.0, 0.0)
    assert excinfo.value.reason == "NON_FINITE_COMMAND"


def test_action_and_tts_parameters_shape_the_sdk_layouts():
    assert projection.action_id_parameter(26) == {"action_id": 26}
    assert projection.action_name_parameter("high wave") == {"action_name": "high wave"}
    assert projection.tts_parameter("hello") == {"index": 0, "speaker_id": 0, "text": "hello"}


def test_led_parameter_rejects_out_of_range_channels():
    with pytest.raises(CommandRejected) as excinfo:
        projection.led_parameter(256, 0, 0)
    assert excinfo.value.reason == "LED_RANGE"
    assert projection.led_parameter(0, 128, 255) == {"R": 0, "G": 128, "B": 255}


def test_play_parameters_carry_app_and_stream_ids():
    start = projection.play_stream_parameter("jukebox", "greeting")
    assert start == {"app_name": "jukebox", "stream_id": "greeting"}
    assert projection.play_stop_parameter("jukebox") == {"app_name": "jukebox"}


# --- locomotion ------------------------------------------------------------------------


def test_locomotion_limits_default_envelope_is_conservative_for_a_biped():
    limits = LocomotionLimits()
    assert limits.deadband == 0.005
    assert limits.max_linear_mps == 1.0
    assert limits.max_lateral_mps == 0.4
    assert limits.max_angular_radps == 1.0


def test_locomotion_limits_lateral_falls_back_to_linear_when_absent():
    limits = LocomotionLimits.from_profile({"max_linear_mps": 1.5})
    assert limits.max_linear_mps == 1.5
    assert limits.max_lateral_mps == 1.5


def test_locomotion_limits_rejects_non_positive_maxima_and_negative_deadband():
    with pytest.raises(ProjectionError):
        LocomotionLimits.from_profile({"max_linear_mps": 0.0})
    with pytest.raises(ProjectionError):
        LocomotionLimits.from_profile({"deadband": -0.1})


def test_twist_below_deadband_becomes_exact_zero():
    limits = LocomotionLimits()
    assert projection.twist_to_velocity(0.003, -0.004, 0.002, limits) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("twist", "label"),
    [((1.2, 0.0, 0.0), "linear.x"), ((0.0, 0.5, 0.0), "linear.y"), ((0.0, 0.0, 1.1), "angular.z")],
)
def test_twist_above_maximum_is_rejected_not_clamped(twist, label):
    """A clamp would move the robot faster than the caller asked."""
    with pytest.raises(CommandRejected) as excinfo:
        projection.twist_to_velocity(*twist, LocomotionLimits())
    assert excinfo.value.reason == "VELOCITY_ABOVE_MAX"
    assert label in excinfo.value.detail


def test_twist_within_envelope_passes_through():
    assert projection.twist_to_velocity(0.8, -0.3, 0.9, LocomotionLimits()) == (0.8, -0.3, 0.9)


def test_twist_rejects_non_finite_components():
    with pytest.raises(CommandRejected):
        projection.twist_to_velocity(float("inf"), 0.0, 0.0, LocomotionLimits())


# --- state feedback projection ------------------------------------------------------------


def test_low_state_maps_all_29_joint_indices():
    """Reviewed bug fix: a 35-entry vector is the normal shape and must map."""
    positions = projection.low_state_joint_positions([float(index) for index in range(35)])
    assert len(positions) == 29
    assert positions["left_hip_pitch_joint"] == 0.0
    assert positions["waist_yaw_joint"] == 12.0
    assert positions["left_shoulder_pitch_joint"] == 15.0
    assert positions["right_wrist_yaw_joint"] == 28.0


def test_short_low_state_is_refused_as_a_projection_error():
    """Reviewed bug fix: the old ``len(seq) < dict`` guard raised TypeError.

    A short vector is a configuration/transport defect and must fail closed
    as ProjectionError naming the shortfall, never compare against the dict.
    """
    with pytest.raises(ProjectionError, match="29"):
        projection.low_state_joint_positions([0.0] * 28)


def test_minimal_29_entry_low_state_is_accepted():
    positions = projection.low_state_joint_positions([0.0] * 29)
    assert len(positions) == 29


def test_dex3_joint_positions_maps_seven_motors_per_hand():
    positions = projection.dex3_joint_positions([0.1 * index for index in range(7)], side="right")
    assert positions == dict(
        zip(projection.dex3_joint_names("right"), [0.1 * index for index in range(7)], strict=True)
    )


def test_dex3_joint_positions_rejects_wrong_length():
    with pytest.raises(ProjectionError):
        projection.dex3_joint_positions([0.0] * 6, side="left")


def test_contract_joint_state_omits_unmeasured_joints():
    """A fabricated zero would read as a real measurement downstream."""
    names, positions = projection.contract_joint_state(
        {"left_shoulder_pitch_joint": 0.1, "left_hand_thumb_0_joint": 0.2},
        order=["left_shoulder_pitch_joint", "waist_yaw_joint", "left_hand_thumb_0_joint"],
    )
    assert names == ["left_shoulder_pitch_joint", "left_hand_thumb_0_joint"]
    assert positions == [0.1, 0.2]


def test_imu_payload_maps_wxyz_quaternion_onto_xyzw():
    payload = projection.imu_payload(
        {"quaternion": [1.0, 2.0, 3.0, 4.0], "gyroscope": [0.1, 0.2, 0.3], "accelerometer": [9.8, 0.0, 0.0]}
    )
    assert payload["orientation"] == {"x": 2.0, "y": 3.0, "z": 4.0, "w": 1.0}
    assert payload["angular_velocity"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert payload["linear_acceleration"] == {"x": 9.8, "y": 0.0, "z": 0.0}


def test_imu_payload_rejects_wrong_array_lengths():
    state = {"quaternion": [1.0, 2.0], "gyroscope": [0.0, 0.0, 0.0], "accelerometer": [0.0, 0.0, 0.0]}
    with pytest.raises(ProjectionError):
        projection.imu_payload(state)


def test_power_state_payload_shapes_the_contract_surface():
    assert projection.power_state_payload(24.0, 3.5) == {"battery_voltage": 24.0, "battery_current": 3.5}


def test_power_state_payload_rejects_negative_voltage():
    with pytest.raises(ProjectionError):
        projection.power_state_payload(-0.1, 3.5)


# --- freshness -----------------------------------------------------------------------------


def test_never_reported_source_is_not_fresh():
    """Reviewed bug fix: stamp 0 means "never reported" and must be stale.

    Treating it as fresh would let stream-mode entry latch positions that do
    not exist (design D4): silence is not a measurement.
    """
    assert projection.is_fresh(0.0, 0.0, window_s=0.5) is False


def test_reported_inside_the_window_is_fresh():
    assert projection.is_fresh(100.0, 100.4, window_s=0.5) is True


def test_reported_beyond_the_window_is_stale():
    assert projection.is_fresh(100.0, 100.6, window_s=0.5) is False


# --- mode machine (design D10) ------------------------------------------------------------------


def test_matching_mode_machine_is_not_a_fault():
    assert projection.mode_machine_fault(5, 5) is None


def test_foreign_mode_machine_faults_with_the_generation_reason():
    fault = projection.mode_machine_fault(15, 5)
    assert fault is not None
    assert "15" in fault
    assert "5" in fault


# --- modes and stop (design D5) -------------------------------------------------------------------


def test_mode_weight_targets_leave_internal_control_outside_stream():
    assert projection.MODE_WEIGHT_TARGETS == {"idle": 0.0, "stream": 1.0, "locomotion": 0.0}


def test_hold_stop_freezes_admission_and_ramps_the_weight_down():
    plan = projection.resolve_stop_plan("HOLD", fsm_id=projection.FSM_IDS["start"], stop_config=None)
    assert plan.zero_velocity
    assert plan.weight_target == 0.0
    assert plan.fsm_id is None  # closing admission IS the stop on this platform
    assert plan.latch
    assert plan.downgraded_from == ""


def test_hold_stop_refuses_a_profile_without_zero_velocity_first():
    with pytest.raises(ProjectionError):
        projection.resolve_stop_plan("HOLD", fsm_id=500, stop_config={"zero_velocity_first": False})


def test_torque_off_downgrades_to_damping_in_a_safe_posture():
    """SetFsmId(0) drops a balancing biped; the performed action is damping."""
    stop_config = {"torque_off_safe_fsm_ids": [2, 3], "torque_off_fsm_id": 1}
    plan = projection.resolve_stop_plan("TORQUE_OFF", fsm_id=3, stop_config=stop_config)
    assert plan.fsm_id == projection.FSM_IDS["damping"]
    assert plan.zero_velocity and plan.latch
    assert plan.downgraded_from == "TORQUE_OFF"


def test_torque_off_is_refused_while_standing():
    stop_config = {"torque_off_safe_fsm_ids": [2, 3], "torque_off_fsm_id": 1}
    with pytest.raises(CommandRejected) as excinfo:
        projection.resolve_stop_plan("TORQUE_OFF", fsm_id=projection.FSM_IDS["start"], stop_config=stop_config)
    assert excinfo.value.reason == "TORQUE_OFF_UNSAFE_POSTURE"


def test_unknown_stop_policy_is_a_configuration_error():
    with pytest.raises(ProjectionError):
        projection.resolve_stop_plan("BLAH", fsm_id=500, stop_config=None)


# --- vendor QoS ------------------------------------------------------------------------------------


def test_vendor_qos_defaults_to_the_high_rate_stream_profile():
    assert projection.vendor_qos(None, "low_state") == ("best_effort", "volatile", 10)


def test_vendor_qos_resolves_from_the_profile_entry():
    qos = {"low_state": {"reliability": "reliable", "durability": "volatile", "depth": 5}}
    assert projection.vendor_qos(qos, "low_state") == ("reliable", "volatile", 5)


def test_vendor_qos_rejects_unknown_reliability_durability_and_depth():
    with pytest.raises(ProjectionError):
        projection.vendor_qos({"low_state": {"reliability": "maybe"}}, "low_state")
    with pytest.raises(ProjectionError):
        projection.vendor_qos({"low_state": {"durability": "immortal"}}, "low_state")
    with pytest.raises(ProjectionError):
        projection.vendor_qos({"low_state": {"depth": 0}}, "low_state")


# --- expert tier (design D11) ------------------------------------------------------------------------


def test_expert_tier_is_not_implemented():
    assert projection.EXPERT_TIER_IMPLEMENTED is False


def test_expert_tier_gate_accepts_the_disabled_default():
    projection.expert_tier_gate(False)  # must not raise


def test_expert_tier_gate_refuses_an_enabled_profile():
    """A profile flipping rt/lowcmd on must not publish un-CRC'd frames."""
    with pytest.raises(ProjectionError, match="D11"):
        projection.expert_tier_gate(True)
