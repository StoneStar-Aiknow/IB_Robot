"""Profile-level contract checks for the G1 runtime profile.

These run without ROS, without the Unitree SDK and without a robot: they assert
that the shipped profile is a valid runtime profile, that it projects the G1
the way `design.md` D3-D11 and user decisions U4/U5 decided, and that the
description it produces is accepted by the contract layer without any public
model (D6). The projection cross-checks pin the profile's vendor facts to the
cited SDK values, so the profile and the pure projection cannot drift apart
silently between reviews.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unitree_g1_robot import projection

from robot_runtime.capabilities import all_capabilities
from robot_runtime.interface_description import build_description, validate_description
from robot_runtime.profile import load_profile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"

# The profile is the audit surface: yaml.safe_load drops comments, so the
# unverified-marker test reads the raw text instead of the parsed mapping.
PROFILE_TEXT = PROFILE_PATH.read_text(encoding="utf-8")

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


@pytest.fixture(scope="module")
def profile() -> dict:
    return load_profile(PROFILE_PATH)


def _channel(profile: dict, name: str) -> dict:
    for channel in profile["command_channels"]:
        if channel["channel"] == name:
            return channel
    raise AssertionError(f"profile declares no command channel {name!r}")


def _comment_block_above(line: str) -> str:
    """The comment lines immediately above the profile line equal to ``line``."""
    lines = PROFILE_TEXT.splitlines()
    for index, candidate in enumerate(lines):
        if candidate.strip() == line:
            block: list[str] = []
            cursor = index - 1
            while cursor >= 0 and (not lines[cursor].strip() or lines[cursor].lstrip().startswith("#")):
                if lines[cursor].lstrip().startswith("#"):
                    block.append(lines[cursor])
                cursor -= 1
            return "\n".join(reversed(block))
    raise AssertionError(f"profile has no line {line!r}")


# --- shape and registration ------------------------------------------------------


def test_profile_loads_and_capabilities_are_registered(profile):
    assert set(profile["capabilities"]) <= all_capabilities()


def test_the_profile_serves_the_physical_vendor_tier(profile):
    assert profile["simulated"] is False
    assert profile["joint_state_topic"] == "/joint_states"
    assert profile["runtime"] == {"name": "unitree_g1_robot", "version": "0.1.0", "type": "g1_29dof_dex3"}


def test_joint_inventory_is_the_urdf_43_in_declaration_order(profile):
    joints = profile["joints"]
    assert len(joints) == 43
    assert len(set(joints)) == 43
    # 12 legs + 3 waist + 14 arms in the vendor LowState index order, then the
    # 14 Dex3 fingers in URDF declaration order (thumb, middle, index per hand).
    assert joints[:29] == list(projection.LOW_STATE_JOINT_INDEX)
    fingers = set(joints[29:])
    assert fingers == set(projection.dex3_joint_names("left")) | set(projection.dex3_joint_names("right"))
    assert len(fingers) == 14


def test_joint_capability_counts_match_the_command_surface(profile):
    caps = profile["capabilities"]
    streamable = sum(len(channel["joints"]) for channel in profile["command_channels"] if channel["type"] != "twist")
    assert caps["joint.state"]["joint_count"] == len(profile["joints"]) == 43
    assert caps["joint.position_stream"]["joint_count"] == streamable == 31
    assert caps["joint.state"]["rate_hz"] == caps["joint.position_stream"]["rate_hz"] == 50.0


# --- joint groups (D3, U4, U5) -----------------------------------------------------


def test_arm_group_is_fourteen_joints_without_the_waist(profile):
    """U4: the waist stays out of the public arm group."""
    arm = profile["arm_joints"]
    assert len(arm) == 14
    assert not any(name.startswith("waist_") for name in arm)
    assert not any("_hand_" in name for name in arm)


def test_arm_sdk_vendor_frame_is_the_arm_group_plus_the_waist(profile):
    """D4/U4: the shared weight spans 17 joints, so the waist rides the vendor
    frame without joining the public arm group."""
    assert len(projection.ARM_SDK_FRAME) == 17
    assert set(projection.ARM_SDK_FRAME) == set(profile["arm_joints"]) | set(WAIST_JOINTS)


def test_hand_is_multi_joint_not_a_scalar_gripper(profile):
    """D3/U5: a Dex3 finger is not a 0..100 aperture; no openness is synthesised."""
    caps = profile["capabilities"]
    assert "gripper.1d" not in caps
    assert caps["hand.multi_joint"] == {"count": 2, "joints_per_hand": 7}
    assert profile["gripper_joints"] == []


# --- command channels -----------------------------------------------------------------


def test_channels_declare_the_contract_side_joint_orders(profile):
    assert _channel(profile, "arm_stream")["joints"] == profile["arm_joints"]
    assert _channel(profile, "waist_stream")["joints"] == list(WAIST_JOINTS)
    for side in ("left", "right"):
        # The hand channels carry the vendor motor order (thumb 0-2, index 3-4,
        # middle 5-6); the projection maps it onto the URDF declaration order.
        assert _channel(profile, f"hand_{side}_stream")["joints"] == list(projection.dex3_joint_names(side))
    base = _channel(profile, "base_velocity")
    assert base["type"] == "twist"
    assert base["modes"] == ["locomotion"]


def test_channel_modes_exist_and_match_their_mode_gates(profile):
    modes = profile["modes"]
    for channel in profile["command_channels"]:
        for mode in channel.get("modes", []):
            assert mode in modes, f"channel {channel['channel']} names unknown mode {mode}"
        if channel["type"] == "twist":
            assert channel["modes"] == ["locomotion"]
            assert modes["locomotion"]["allows_base"] is True
        else:
            assert channel["modes"] == ["stream"]
            assert modes["stream"]["allows_stream"] is True


# --- modes (D5) -------------------------------------------------------------------------


def test_modes_partition_streaming_from_locomotion(profile):
    """D5: stream admits the 17-joint frame; in locomotion the arms stay under
    internal control and only the base is commandable."""
    modes = profile["modes"]
    assert modes["initial"] == "idle"
    declared = {name for name in modes if name != "initial"}
    assert declared == {"idle", "stream", "locomotion"}
    assert modes["stream"]["allows_stream"] is True
    assert "allows_stream" not in modes["locomotion"]
    assert modes["locomotion"]["allows_base"] is True
    assert "allows_base" not in modes["stream"] and "allows_base" not in modes["idle"]
    for name in declared:
        mode = modes[name]
        assert mode["controllers"] == []  # the bridge switches, no controller manager
        assert set(mode["transitions"]) <= declared
        assert name not in mode["transitions"]


def test_no_ros2_control_stack_is_configured(profile):
    """D1: the bridge talks to the vendor DDS tier; there is no controller
    manager to configure."""
    assert "ros2_control" not in profile
    assert profile["hardware_components"] == []
    assert profile["trajectory_actions"] == []


# --- vendor tier surface -------------------------------------------------------------------


def test_vendor_topics_name_the_dds_streams_the_bridge_translates(profile):
    topics = profile["vendor"]["topics"]
    assert topics["low_state"] == "rt/lowstate"
    assert topics["arm_sdk"] == "rt/arm_sdk"
    assert topics["dex3_left_cmd"] == "rt/dex3/left/cmd"
    assert topics["dex3_right_cmd"] == "rt/dex3/right/cmd"
    assert topics["arm_action_state"] == "rt/arm/action/state"


def test_rpc_surface_matches_the_projection_convention(profile):
    rpc = profile["vendor"]["rpc"]
    assert rpc["request_topics"] == {
        "loco": projection.rpc_request_topic("sport"),
        "arm": projection.rpc_request_topic("arm"),
        "audio": projection.rpc_request_topic("voice"),
    }
    assert rpc["response_topics"] == {
        "loco": projection.rpc_response_topic("sport"),
        "arm": projection.rpc_response_topic("arm"),
        "audio": projection.rpc_response_topic("voice"),
    }
    assert rpc["timeout_s"] > 0.0


def test_api_ids_match_the_vendored_sdk_headers(profile):
    api_ids = profile["vendor"]["api_ids"]
    assert api_ids["loco"] == projection.LOCO_API_IDS
    assert api_ids["arm"] == projection.ARM_ACTION_API_IDS
    assert api_ids["audio"] == projection.AUDIO_API_IDS


def test_tts_led_and_audio_api_ids_match_the_vendored_headers(profile):
    vendor = profile["vendor"]
    assert vendor["tts"]["api_id"] == projection.AUDIO_API_IDS["tts"]
    assert vendor["led"]["api_id"] == projection.AUDIO_API_IDS["set_rgb_led"]
    assert vendor["audio"]["start_play_api_id"] == projection.AUDIO_API_IDS["start_play"]
    assert vendor["audio"]["stop_play_api_id"] == projection.AUDIO_API_IDS["stop_play"]


def test_postures_map_onto_sdk_fsm_ids(profile):
    vendor = profile["vendor"]
    scalars = {name: value for name, value in vendor["fsm_ids"].items() if not isinstance(value, dict)}
    assert scalars == projection.FSM_IDS
    assert vendor["fsm_ids"]["internal_ctrl"] == projection.INTERNAL_FSM_MODES
    assert set(vendor["postures"]) == set(profile["capabilities"]["motion.posture"]["postures"])
    for posture, entry in vendor["postures"].items():
        assert entry["fsm_id"] == projection.FSM_IDS[posture]


def test_named_motions_map_onto_unique_vendor_action_ids(profile):
    vendor = profile["vendor"]
    actions = vendor["arm_actions"]
    assert set(actions) == set(profile["capabilities"]["motion.named"]["names"])
    ids = [entry["action_id"] for entry in actions.values()]
    # The vendor's own action table carries a duplicate id 12; the shipped
    # subset must not.
    assert len(ids) == len(set(ids))


def test_arm_sdk_parameters_carry_the_cited_example_values(profile):
    arm_sdk = profile["vendor"]["arm_sdk"]
    assert arm_sdk["weight_index"] == projection.ARM_SDK_WEIGHT_INDEX == 29
    assert arm_sdk["rate_hz"] == 50.0
    assert arm_sdk["default_kp"] == projection.EXAMPLE_ARM_SDK_KP
    assert arm_sdk["default_kd"] == projection.EXAMPLE_ARM_SDK_KD
    assert arm_sdk["weight_ramp_rate"] == projection.EXAMPLE_WEIGHT_RAMP_RATE
    assert arm_sdk["stream_staleness_s"] > 0.0
    assert arm_sdk["feedback_staleness_s"] > 0.0


def test_dex3_parameters_carry_the_cited_example_values(profile):
    dex3 = profile["vendor"]["dex3"]
    assert dex3["joints_per_hand"] == projection.DEX3_MOTORS_PER_HAND == 7
    assert dex3["press_sensors_per_hand"] == 9
    assert dex3["default_kp"] == projection.EXAMPLE_DEX3_KP
    assert dex3["default_kd"] == projection.EXAMPLE_DEX3_KD


def test_locomotion_envelope_is_shared_by_capability_and_vendor_section(profile):
    caps = profile["capabilities"]["base.cmd_vel"]
    vendor = profile["vendor"]["locomotion"]
    assert (caps["max_vx"], caps["max_vy"], caps["max_wz"]) == (
        vendor["max_linear_mps"],
        vendor["max_lateral_mps"],
        vendor["max_angular_radps"],
    )
    assert projection.LocomotionLimits.from_profile(vendor) == projection.LocomotionLimits()
    assert vendor["command_duration_s"] == projection.VELOCITY_STEP_DURATION_S
    assert vendor["continuous_duration_s"] == projection.VELOCITY_CONTINUOUS_DURATION_S


def test_joint_limits_cover_every_joint_and_bracket_their_targets(profile):
    limits = profile["vendor"]["joint_limits"]
    assert set(limits) == set(profile["joints"])
    for name, entry in limits.items():
        assert entry["lower"] < entry["upper"], name
    # The projection refuses rather than clamps against these very entries.
    joint = profile["arm_joints"][0]
    with pytest.raises(projection.CommandRejected) as excinfo:
        projection.channel_targets(
            [limits[joint]["upper"] + 1.0], channel_joints=[joint], joint_limits=limits, label="arm_stream"
        )
    assert excinfo.value.reason == "JOINT_TARGET_RANGE"


def test_vendor_qos_entries_resolve_to_valid_profiles(profile):
    qos = profile["vendor"]["qos"]
    assert projection.vendor_qos(qos, "low_state") == ("best_effort", "volatile", 10)
    assert projection.vendor_qos(qos, "rpc_request") == ("reliable", "volatile", 10)
    for key in qos:  # every entry resolves without raising
        projection.vendor_qos(qos, key)


# --- stop (D5) -------------------------------------------------------------------------------


def test_stop_policy_is_hold_and_torque_off_fails_closed(profile):
    stop = profile["vendor"]["stop"]
    assert profile["stop_default_policy"] == "HOLD"
    assert stop["zero_velocity_first"] is True
    assert stop["torque_off_fsm_id"] == projection.FSM_IDS["damping"]
    assert set(stop["torque_off_safe_fsm_ids"]) == {projection.FSM_IDS["sit"], projection.FSM_IDS["squat"]}
    hold = projection.resolve_stop_plan("HOLD", fsm_id=projection.FSM_IDS["start"], stop_config=stop)
    assert hold.fsm_id is None  # closing admission IS the stop on this platform
    assert hold.latch and hold.weight_target == 0.0
    seated = projection.resolve_stop_plan("TORQUE_OFF", fsm_id=projection.FSM_IDS["sit"], stop_config=stop)
    assert seated.fsm_id == projection.FSM_IDS["damping"]
    assert seated.downgraded_from == "TORQUE_OFF"
    with pytest.raises(projection.CommandRejected):
        projection.resolve_stop_plan("TORQUE_OFF", fsm_id=projection.FSM_IDS["start"], stop_config=stop)


def test_stop_bounds_cover_the_ramp_down_and_the_rpc_round_trip(profile):
    """The declared bounds must be no smaller than what the implementation can
    take: idle covers the weight ramp-down from 1.0 at stop.weight_ramp_down_rate
    plus the bounded-wait margin (weight/rate + 1 s); torque_off adds one
    vendor RPC round trip (rpc.timeout_s) for the damping switch."""
    stop = profile["vendor"]["stop"]
    bounds = profile["capabilities"]["runtime.stop"]
    ramp_s = 1.0 / stop["weight_ramp_down_rate"]
    assert bounds["idle_bound_s"] >= ramp_s + 1.0
    assert bounds["torque_off_bound_s"] >= bounds["idle_bound_s"] + profile["vendor"]["rpc"]["timeout_s"]


# --- mode machine (D10) ------------------------------------------------------------------------


def test_mode_machine_generation_is_pinned_and_checked_at_startup(profile):
    """D10: a unit reporting another generation has no matching joint model."""
    expected = profile["vendor"]["mode_machine"]
    assert expected == 5
    assert projection.mode_machine_fault(expected, expected) is None
    fault = projection.mode_machine_fault(15, expected)
    assert fault is not None and "15" in fault


# --- expert tier (D11) ---------------------------------------------------------------------------


def test_expert_joint_tier_stays_disabled(profile):
    """D11: rt/lowcmd publishing (and its CRC requirement) is not implemented."""
    enabled = profile["vendor"]["expert_joint_tier"]["enabled"]
    assert enabled is False
    projection.expert_tier_gate(enabled)  # must not raise


# --- description projection (D6, D8) ----------------------------------------------------------------


def test_description_builds_and_publishes_no_model(profile):
    """D6: no URDF is passed to the description; consumers needing
    normalization metadata fail closed on the G1."""
    description = build_description(profile, use_sim=True)
    validate_description(description)
    assert "model" not in description
    assert "robot.description" not in description["interfaces"]
    assert not [name for name in description["interfaces"] if name.startswith("motion.arm")]
    assert profile["teleoperation"]["enabled"] is False


def test_command_surface_is_bridge_owned_neutral_topics(profile):
    interfaces = build_description(profile, use_sim=True)["interfaces"]
    assert interfaces["joint.arm_stream"]["endpoint"] == "/unitree_g1/arm/commands"
    assert interfaces["joint.waist_stream"]["endpoint"] == "/unitree_g1/waist/commands"
    assert interfaces["joint.hand_left_stream"]["endpoint"] == "/unitree_g1/hand_left/commands"
    assert interfaces["joint.hand_right_stream"]["endpoint"] == "/unitree_g1/hand_right/commands"
    assert interfaces["joint.hand_left_stream"]["joint_names"] == list(projection.dex3_joint_names("left"))
    base = interfaces["base.cmd_vel"]
    assert base["endpoint"] == "/cmd_vel"
    assert base["limits"] == {"max_vx": 1.0, "max_vy": 0.4, "max_wz": 1.0}
    assert base["modes"] == ["locomotion"]


def test_joint_state_exposes_the_full_43_joint_projection(profile):
    interfaces = build_description(profile, use_sim=True)["interfaces"]
    state = interfaces["joint.state"]
    assert state["endpoint"] == "/joint_states"
    assert state["joint_names"] == profile["joints"]
    assert state["units"] == {"position": "rad", "velocity": "rad/s"}


def test_peripheral_drivers_serve_the_declared_capability_topics(profile):
    """D8: the runtime composes shared drivers; the endpoints are the ones
    peripheral_nodes serves, not vendor SDK topics."""
    interfaces = build_description(profile, use_sim=True)["interfaces"]
    endpoints = {spec["endpoint"] for spec in interfaces.values()}
    for topic in profile["capabilities"]["perception.camera"]["topics"]:
        assert topic in endpoints
    for topic in profile["capabilities"]["perception.lidar"]["topics"]:
        assert topic in endpoints


def test_public_interfaces_serve_neutral_message_types(profile):
    """No generic consumer links unitree_hg or unitree_api: every vendor-facing
    surface is re-served with a neutral type."""
    interfaces = build_description(profile, use_sim=True)["interfaces"]
    for name, spec in profile["interfaces"].items():
        assert spec["capability"] in profile["capabilities"], name
        assert name in interfaces
    assert interfaces["imu.body"]["message_type"] == "sensor_msgs/msg/Imu"
    assert interfaces["speech.speak"]["message_type"] == "ibrobot_msgs/srv/SpeakText"
    assert interfaces["motion.named"]["message_type"] == "ibrobot_msgs/action/ExecuteNamedMotion"
    assert interfaces["led.set_pattern"]["message_type"] == "ibrobot_msgs/srv/SetLedPattern"


def test_unsourced_capabilities_stay_undeclared(profile):
    """interaction.audio_playback and power.state have no verified vendor
    source (gate 6.1); declaring either without one would fail conformance
    against an endpoint nothing serves. They return with the gate closed."""
    caps = profile["capabilities"]
    assert "interaction.audio_playback" not in caps
    assert "power.state" not in caps
    interfaces = build_description(profile, use_sim=True)["interfaces"]
    assert "audio.play" not in interfaces
    assert "power.state" not in interfaces


def test_led_capability_declares_no_priority_levels(profile):
    """SetRgbLed carries no priority parameter; the deviation from D7's
    X2-style priority list is recorded in the profile comment."""
    assert "priority_levels" not in profile["capabilities"]["interaction.led"]


def test_lidar_odometry_stays_disabled_until_the_network_facts_are_confirmed(profile):
    assert profile["fast_lio"]["enabled"] is False


# --- unverified marking ------------------------------------------------------------------------------


UNVERIFIED_LINES = [
    ("base.cmd_vel: {max_vx: 1.0, max_vy: 0.4, max_wz: 1.0, staleness_s: 0.5}", "velocity envelope"),
    ("runtime.stop: {cancel_bound_s: 1.0, idle_bound_s: 6.0, torque_off_bound_s: 11.0}", "stop bounds"),
    ("mode_machine: 5", "hardware generation"),
    ("mode_pr: 1", "lowcmd mode_pr"),
    ("dex3_left_state: rt/lf/dex3/left/state", "dex3 state stream"),
    ("bms_state: rt/bmsstate", "battery topic"),
    ("rpc:", "rpc topic naming"),
    ("qos:", "endpoint qos"),
    ("optical_frame_id: d435_link", "camera optical frame"),
    ("lidar_ip: 192.168.1.100", "lidar network"),
    ("weight_ramp_rate: 0.2", "arm_sdk ramp rate"),
    ("default_kp: 60.0", "arm_sdk gains"),
    ("rate_hz: 100.0", "dex3 rate"),
    ("default_kp: 1.5", "dex3 gains"),
    ("max_linear_mps: 1.0", "locomotion envelope"),
    ("torque_off_fsm_id: 1", "torque-off safe set"),
    ("speaker_id: 0", "tts defaults"),
    ("- channel: hand_left_stream", "dex3 motor split"),
    ("rate_hz: 500.0", "imu frame"),
]


@pytest.mark.parametrize(("line", "what"), UNVERIFIED_LINES)
def test_unverifiable_values_carry_their_unverified_note(line, what):
    """Every value that cannot be established from the vendored material names
    its hardware gate in the profile instead of passing as fact."""
    assert "unverified" in _comment_block_above(line).lower(), what
