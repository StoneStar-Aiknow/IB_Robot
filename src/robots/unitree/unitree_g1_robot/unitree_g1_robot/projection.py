"""Pure projection between the public runtime contract and the Unitree G1 DDS tier.

No ROS and no vendor imports live here, so every conversion, threshold and
rejection rule is testable without a robot, without the Unitree SDK and without
a DDS participant. ``vendor_gateway`` (phase 2) owns the vendor message types;
this module only produces and consumes plain Python values.

Every vendor constant below is cited to the material vendored in this
repository (``unitree/unitree_sdk2`` and ``unitree/unitree_ros2``):

* joint indices, the 17-joint arm_sdk frame and the weight slot:
  ``unitree_sdk2/include/unitree/dds_wrapper/robots/g1/defines.h`` and
  ``example/g1/high_level/g1_arm7_sdk_dds_example.cpp``
* Dex3 motor order and the RIS_Mode bitfield:
  ``example/g1/dex3/g1_dex3_example.cpp``
* service names and api ids: ``include/unitree/robot/g1/loco/g1_loco_api.hpp``,
  ``g1/arm/g1_arm_action_api.hpp``, ``g1/audio/g1_audio_api.hpp``
* FSM ids: the high-level methods of ``g1_loco_client.hpp``
  (Damp/Start/Squat/Sit/StandUp/ZeroTorque are SetFsmId calls)
* RPC request/response shape: ``unitree_ros2`` msg ``unitree_api/Request.msg``
  and ``Response.msg``
* payload field layouts: ``unitree_ros2`` msg ``unitree_hg/{LowCmd,LowState,
  MotorCmd,MotorState,HandCmd,HandState,IMUState}.msg``

Values that cannot be established from that material are marked ``unverified``
and name the hardware gate (tasks 6.1-6.4) that will confirm them; each of them
also carries a note in the shipped profile.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# --- vendor joint indices ----------------------------------------------------
# JointIndex in defines.h: 0-5 left leg, 6-11 right leg, 12-14 waist,
# 15-21 left arm, 22-28 right arm. These index both LowState.motor_state and
# LowCmd.motor_cmd. Names are the vendored URDF's (g1_29dof_with_hand_rev_1_0).
LOW_STATE_JOINT_INDEX: dict[str, int] = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

# The 17-joint frame the arm_sdk topic carries (defines.h ArmJoints and the
# arm_sdk example's arm_joints array): left arm 7, right arm 7, then the three
# waist joints last. The waist IS part of this vendor frame even though it is
# NOT part of the contract's arm group (user decision U4): whenever any one of
# the 17 streams, every slot must carry a target, because the vendor blends the
# whole frame against the one shared weight.
ARM_SDK_FRAME: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ARM_SDK_FRAME_SIZE = len(ARM_SDK_FRAME)

# The first slot after JointIndex 0-28. Its q field is the shared weight the
# vendor blends against internal control for the whole 17-joint frame; the
# example writes it as msg.motor_cmd().at(kNotUsedJoint).q(weight).
ARM_SDK_WEIGHT_INDEX = 29

# LowCmd.motor_cmd is fixed at 35 entries (unitree_hg/LowCmd.msg).
LOW_CMD_MOTOR_COUNT = 35

# Dex3 motor order per hand, from the SDK dex3 example's limits arrays
# (maxLimits_left/minLimits_left match the vendored URDF thumb/index/middle
# ranges in this order). unverified: the example's ranges cannot distinguish
# index from middle motor positions (identical limits); confirmed on hardware
# (gate 6.3) before the split matters operationally.
DEX3_FINGER_NAMES: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
)
DEX3_MOTORS_PER_HAND = len(DEX3_FINGER_NAMES)

#: MotorCmd gains the SDK examples pair with joint targets on arm_sdk
#: (kp 60, kd 1.5). unverified: tuned on a gantry (gate 6.2); the profile
#: carries them as data, this is only the cited example value.
EXAMPLE_ARM_SDK_KP = 60.0
EXAMPLE_ARM_SDK_KD = 1.5

#: The weight ramp rate the arm_sdk example uses (weight_rate = 0.2 with a
#: 20 ms tick, i.e. 0.2/s). unverified: tuned on a gantry (gate 6.2).
EXAMPLE_WEIGHT_RAMP_RATE = 0.2

#: Dex3 gains the example's grip path uses (kp 1.5, kd 0.1). unverified
#: (gate 6.3); the profile carries them as data.
EXAMPLE_DEX3_KP = 1.5
EXAMPLE_DEX3_KD = 0.1


def dex3_joint_names(side: str) -> tuple[str, ...]:
    """URDF joint names of one hand's 7 motors, in vendor motor order.

    ``side`` is "left" or "right" (the URDF prefixes each finger joint with
    ``<side>_hand_``).
    """
    if side not in ("left", "right"):
        raise ProjectionError(f"unknown dex3 side {side!r}; expected 'left' or 'right'")
    return tuple(f"{side}_hand_{finger}_joint" for finger in DEX3_FINGER_NAMES)


# --- vendor services, api ids and FSM ids ------------------------------------
# Service names from the SDK headers; RPC topics follow the rt/api/<service>/
# {request,response} convention (unverified on hardware, gate 6.1; the profile
# carries the same strings as data).
RPC_SERVICES: tuple[str, ...] = ("sport", "arm", "voice")

# g1_loco_api.hpp. api 7006 (GetPhase) is deprecated and 7106 (SetArmTask)
# is deliberately absent: wave/shake whole-body tasks are dispatched through
# the arm action service instead, not duplicated here.
LOCO_API_IDS: dict[str, int] = {
    "get_fsm_id": 7001,
    "get_fsm_mode": 7002,
    "get_balance_mode": 7003,
    "get_swing_height": 7004,
    "get_stand_height": 7005,
    "fsm_api": 7100,
    "set_fsm_id": 7101,
    "set_balance_mode": 7102,
    "set_swing_height": 7103,
    "set_stand_height": 7104,
    "set_velocity": 7105,
    "set_speed_mode": 7107,
    "switch_to_user_ctrl": 7110,
    "switch_to_internal_ctrl": 7111,
}

# g1_arm_action_api.hpp.
ARM_ACTION_API_IDS: dict[str, int] = {
    "execute_action": 7106,
    "get_action_list": 7107,
    "execute_custom_action": 7108,
    "stop_custom_action": 7113,
}

# g1_audio_api.hpp.
AUDIO_API_IDS: dict[str, int] = {
    "tts": 1001,
    "asr": 1002,
    "start_play": 1003,
    "stop_play": 1004,
    "get_volume": 1005,
    "set_volume": 1006,
    "set_rgb_led": 1010,
}

# The FSM ids the SDK's own high-level LocoClient methods pass to SetFsmId.
# Their acceptance latencies and preconditions are a hardware gate (6.2/6.4);
# the ids themselves are SDK facts. "start" (500) enters the locomotion FSM.
FSM_IDS: dict[str, int] = {
    "zero_torque": 0,
    "damping": 1,
    "squat": 2,
    "sit": 3,
    "stand_up": 4,
    "start": 500,
}

# SwitchToInternalCtrl's InternalFsmMode values (g1_loco_client.hpp).
INTERNAL_FSM_MODES: dict[str, int] = {"last": 0, "passive": 1, "walkrun": 2}

# SetVelocity's own semantics (g1_loco_client.hpp): Move() passes 864000 s for
# continuous motion and 1 s for a single step; the bridge re-issues at the
# profile's command_duration_s while fresh Twists keep arriving.
VELOCITY_STEP_DURATION_S = 1.0
VELOCITY_CONTINUOUS_DURATION_S = 864000.0


def rpc_request_topic(service: str) -> str:
    """DDS topic carrying requests for one vendor RPC service.

    unverified: the rt/api/<service>/{request,response} convention is assumed
    from the SDK's channel naming and confirmed on hardware (gate 6.1); the
    profile carries the same strings.
    """
    return f"rt/api/{service}/request"


def rpc_response_topic(service: str) -> str:
    """DDS topic carrying responses for one vendor RPC service."""
    return f"rt/api/{service}/response"


# --- errors ------------------------------------------------------------------


class ProjectionError(ValueError):
    """Static (configuration) error: the profile cannot be projected."""


@dataclass(frozen=True, slots=True)
class CommandRejected(Exception):
    """Dynamic error: a well-formed request the runtime must refuse.

    ``reason`` is machine-readable and appears verbatim in RuntimeStatus and in
    service responses, so an operator can tell *why* a command did nothing.
    """

    reason: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


def _finite(values: Iterable[Any], label: str) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise CommandRejected("NON_NUMERIC_COMMAND", f"{label}: {value!r}") from exc
        if not math.isfinite(number):
            raise CommandRejected("NON_FINITE_COMMAND", f"{label}: {number!r}")
        out.append(number)
    return out


# --- joint command projection ------------------------------------------------


def check_joint_target(name: str, value: float, limits: Mapping[str, Any]) -> None:
    """Refuse a target outside the URDF limits; nothing is clamped.

    A clamp would execute a motion the caller did not ask for. ``limits`` is
    one profile ``vendor.joint_limits`` entry ({lower, upper, ...}).
    """
    lower = float(limits["lower"])
    upper = float(limits["upper"])
    if not lower <= value <= upper:
        raise CommandRejected(
            "JOINT_TARGET_RANGE",
            f"{name} target {value:.4f} outside URDF limits [{lower:.4f}, {upper:.4f}]",
        )


def channel_targets(
    values: Sequence[float],
    *,
    channel_joints: Sequence[str],
    joint_limits: Mapping[str, Mapping[str, float]],
    label: str,
) -> dict[str, float]:
    """Project one streaming channel's payload into a name -> target mapping.

    ``channel_joints`` is the profile's declared order for the channel
    (Float64MultiArray.data order); the result is keyed by joint name so the
    caller never depends on two orders agreeing. Each target is checked
    against the URDF limits and rejected, never clamped.
    """
    numbers = _finite(values, label)
    if len(numbers) != len(channel_joints):
        raise CommandRejected(
            f"{label.upper()}_LENGTH",
            f"expected {len(channel_joints)} values, got {len(numbers)}",
        )
    targets: dict[str, float] = {}
    for name, value in zip(channel_joints, numbers, strict=True):
        limit = joint_limits.get(name)
        if limit is None:
            raise ProjectionError(f"joint {name!r} has no URDF limits declared")
        check_joint_target(name, value, limit)
        targets[name] = value
    return targets


def merge_latch(latched: Mapping[str, float], update: Mapping[str, float]) -> dict[str, float]:
    """Immutably merge fresh channel targets into a latch.

    Design D4: entering stream mode latches measured positions as hold
    targets, and each fresh channel update overwrites exactly its own slots.
    A stale channel contributes nothing — the previous latch is republished
    as-is. There is deliberately no per-tick measured refill: republishing the
    measured position each tick would command a sink under gravity whenever
    the waist's measurement advances while the channel is silent, i.e. the
    robot would execute a motion nobody commanded.
    """
    merged = dict(latched)
    merged.update(update)
    return merged


def capture_hold_targets(measurements: Mapping[str, float], *, order: Sequence[str]) -> list[float] | None:
    """Measured positions in ``order``, or None if any joint is unmeasured.

    Used at stream-mode entry (design D4): the latch is seeded from measured
    positions so the first weight-blended frame holds rather than snaps. A
    partial measurement is no basis for a hold target — entry is refused
    (fail closed) rather than partially seeded. Freshness of the measurements
    (profile feedback_staleness_s) is the bridge's check; this function only
    demands presence.
    """
    targets: list[float] = []
    for name in order:
        if name not in measurements:
            return None
        targets.append(float(measurements[name]))
    return targets


def motor_cmd(
    *, q: float = 0.0, dq: float = 0.0, tau: float = 0.0, kp: float = 0.0, kd: float = 0.0, mode: int = 0
) -> dict[str, float | int]:
    """One unitree_hg/MotorCmd-shaped plain dict (MotorCmd.msg field names)."""
    return {"mode": int(mode), "q": float(q), "dq": float(dq), "tau": float(tau), "kp": float(kp), "kd": float(kd)}


def arm_sdk_lowcmd(
    latched: Mapping[str, float],
    *,
    weight: float,
    mode_pr: int,
    mode_machine: int,
    kp: float = EXAMPLE_ARM_SDK_KP,
    kd: float = EXAMPLE_ARM_SDK_KD,
) -> dict[str, Any]:
    """Project the latched 17-joint frame onto a unitree_hg/LowCmd plain dict.

    The weight rides in motor_cmd[ARM_SDK_WEIGHT_INDEX].q — the slot after
    JointIndex 0-28 that the arm_sdk example writes through kNotUsedJoint. Only
    the 17 frame joints carry targets and gains; the 12 leg slots stay at
    vendor-default zeros, exactly as the example leaves them, because the
    vendor gates the whole message on the shared weight, not per-joint modes.

    ``crc`` is emitted as 0: the arm_sdk example never computes one, and the
    CRC-requiring rt/lowcmd tier stays disabled (design D11).
    """
    if len(latched) != ARM_SDK_FRAME_SIZE or any(name not in latched for name in ARM_SDK_FRAME):
        raise ProjectionError(
            f"arm_sdk latch must hold exactly the {ARM_SDK_FRAME_SIZE} frame joints, got {sorted(latched)}"
        )
    if not 0.0 <= float(weight) <= 1.0:
        raise CommandRejected("ARM_SDK_WEIGHT_RANGE", f"weight {weight} outside [0, 1]")
    motor_cmd_slots: list[dict[str, float | int]] = [motor_cmd() for _ in range(LOW_CMD_MOTOR_COUNT)]
    for name, index in LOW_STATE_JOINT_INDEX.items():
        if name in latched:
            motor_cmd_slots[index] = motor_cmd(q=latched[name], dq=0.0, tau=0.0, kp=kp, kd=kd)
    motor_cmd_slots[ARM_SDK_WEIGHT_INDEX]["q"] = float(weight)
    return {
        "mode_pr": int(mode_pr),
        "mode_machine": int(mode_machine),
        "motor_cmd": motor_cmd_slots,
        "reserve": [0, 0, 0, 0],
        "crc": 0,
    }


def ramp_weight(current: float, target: float, *, rate_per_s: float, dt_s: float) -> float:
    """Advance the shared weight one tick toward ``target``.

    The example ramps at weight_rate * control_dt per 20 ms tick and clamps
    to [0, 1]; the ramp is how the vendor's internal controller and the
    streamed targets are superposed, so the rate is a safety parameter, taken
    from the profile (unverified until gate 6.2), never invented here.
    """
    if dt_s < 0.0:
        raise ProjectionError(f"tick must be non-negative, got {dt_s}")
    if rate_per_s < 0.0:
        raise ProjectionError(f"ramp rate must be non-negative, got {rate_per_s}")
    step = rate_per_s * dt_s
    moved = current + step if target > current else current - step
    return min(max(moved, min(current, target)), max(current, target))


# --- Dex3 --------------------------------------------------------------------

#: The status value the example's rotate/grip/stop paths all send (0x01).
DEX3_STATUS_ENABLE = 0x01


def ris_mode_byte(motor_id: int, *, status: int = DEX3_STATUS_ENABLE, timeout: bool = False) -> int:
    """Pack the Dex3 RIS_Mode bitfield byte (g1_dex3_example.cpp).

    The byte is ``id:4 | status:3 | timeout:1``: low nibble = motor id, next
    three bits = status, MSB = the timeout flag the example's stopMotors sets
    to release the hand (kp/kd zeroed alongside).
    """
    if not 0 <= int(motor_id) <= 15:
        raise ProjectionError(f"ris motor id must fit 4 bits, got {motor_id}")
    if not 0 <= int(status) <= 7:
        raise ProjectionError(f"ris status must fit 3 bits, got {status}")
    mode = int(motor_id) & 0x0F
    mode |= (int(status) & 0x07) << 4
    mode |= (1 if timeout else 0) << 7
    return mode


def dex3_handcmd(
    targets: Mapping[str, float],
    *,
    side: str,
    kp: float = EXAMPLE_DEX3_KP,
    kd: float = EXAMPLE_DEX3_KD,
    timeout: bool = False,
) -> dict[str, Any]:
    """Project one hand's latched targets onto a unitree_hg/HandCmd plain dict.

    ``targets`` maps URDF joint names (from :func:`dex3_joint_names`) to
    positions; slots without a latched target stay at the HandCmd default
    zero. ``timeout=True`` reproduces the example's stopMotors release: mode
    byte gets the timeout flag and gains go to zero.
    """
    names = dex3_joint_names(side)
    if not set(targets) <= set(names):
        unknown = sorted(set(targets) - set(names))
        raise ProjectionError(f"{side} hand targets name unknown joints {unknown}")
    motor_cmd_slots: list[dict[str, float | int]] = []
    for index, name in enumerate(names):
        commanded = name in targets and not timeout
        motor_cmd_slots.append(
            motor_cmd(
                q=float(targets.get(name, 0.0)),
                dq=0.0,
                tau=0.0,
                kp=kp if commanded else 0.0,
                kd=kd if commanded else 0.0,
                mode=ris_mode_byte(index, timeout=timeout),
            )
        )
    return {"motor_cmd": motor_cmd_slots, "reserve": [0, 0, 0, 0]}


# --- RPC ---------------------------------------------------------------------


def _compact_json(parameter: Any) -> str:
    """Serialize an RPC parameter the way the SDK's ToJsonString does.

    The vendor's Jsonize helpers emit compact objects ({"data":123}); the RPC
    server parses JSON, so whitespace is not contractually significant, but
    matching the vendor's own output keeps byte-level comparisons in the
    conformance suite honest.
    """
    return json.dumps(parameter, separators=(",", ":"), ensure_ascii=False)


def rpc_request(
    identity_id: int,
    api_id: int,
    parameter: Any,
    *,
    binary: bytes = b"",
    lease_id: int = 0,
    priority: int = 0,
    noreply: bool = False,
) -> dict[str, Any]:
    """Build a unitree_api/Request-shaped plain dict.

    ``identity_id`` is the correlation id the caller allocates monotonically;
    the response echoes it in header.identity.id together with header.api_id.
    ``parameter`` is a plain dict (see the *_parameter builders) or an
    already-serialized JSON string.

    Field layout follows the vendored IDL exactly (unitree_api/msg/): the
    lease carries only ``id`` (RequestLease.msg), while ``priority`` and
    ``noreply`` are RequestPolicy.msg fields — the SDK examples leave them
    default (0/0/false) and so does this builder.
    """
    if identity_id <= 0:
        raise ProjectionError(f"rpc identity id must be positive and monotonic, got {identity_id}")
    if isinstance(parameter, str):
        parameter_json = parameter
    else:
        parameter_json = _compact_json(parameter)
    return {
        "header": {
            "identity": {"id": int(identity_id), "api_id": int(api_id)},
            "lease": {"id": int(lease_id)},
            "policy": {"priority": int(priority), "noreply": bool(noreply)},
        },
        "parameter": parameter_json,
        "binary": list(binary),
    }


def rpc_response_matches(response: Mapping[str, Any], *, identity_id: int, api_id: int) -> bool:
    """Whether a Response is the answer to our Request (identity correlation)."""
    try:
        identity = response["header"]["identity"]
    except (KeyError, TypeError):
        return False
    return identity.get("id") == int(identity_id) and identity.get("api_id") == int(api_id)


def rpc_response_code(response: Mapping[str, Any]) -> int:
    """ResponseStatus.code; 0 is success, anything else is a vendor error."""
    return int(response["header"]["status"]["code"])


def rpc_response_data(response: Mapping[str, Any]) -> Any:
    """Parse a Response's JSON ``data`` payload."""
    return json.loads(response["data"])


# --- RPC parameters (SDK Jsonize layouts) -------------------------------------


def fsm_id_parameter(fsm_id: int) -> dict[str, int]:
    """SetFsmId parameter (JsonizeDataInt): {"data": fsm_id}."""
    return {"data": int(fsm_id)}


def velocity_parameter(
    vx: float, vy: float, omega: float, duration_s: float = VELOCITY_STEP_DURATION_S
) -> dict[str, Any]:
    """SetVelocity parameter (JsonizeVelocityCommand)."""
    numbers = _finite((vx, vy, omega), "velocity command")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise CommandRejected("VELOCITY_DURATION", f"duration {duration!r} must be positive")
    return {"velocity": numbers, "duration": duration}


def action_id_parameter(action_id: int) -> dict[str, int]:
    """ExecuteAction parameter (G1ArmActionClient::ExecuteAction(int))."""
    return {"action_id": int(action_id)}


def action_name_parameter(action_name: str) -> dict[str, str]:
    """ExecuteCustomAction parameter (G1ArmActionClient::ExecuteAction(string))."""
    return {"action_name": str(action_name)}


def tts_parameter(text: str, *, index: int = 0, speaker_id: int = 0) -> dict[str, Any]:
    """TtsMakerParameter. unverified: index/speaker_id defaults (gate 6.1)."""
    return {"index": int(index), "speaker_id": int(speaker_id), "text": str(text)}


def led_parameter(red: int, green: int, blue: int) -> dict[str, int]:
    """SetRgbLed parameter (LedControlParameter R/G/B)."""
    channels = {"R": int(red), "G": int(green), "B": int(blue)}
    for name, value in channels.items():
        if not 0 <= value <= 255:
            raise CommandRejected("LED_RANGE", f"{name}={value} outside [0, 255]")
    return channels


def play_stream_parameter(app_name: str, stream_id: str) -> dict[str, str]:
    """StartPlay parameter (PlayStreamParameter app_name/stream_id).

    unverified: the voice service plays named streams, not a documented PCM
    sink; the mapping from a PCM topic onto a stream is a hardware gate (6.1).
    """
    return {"app_name": str(app_name), "stream_id": str(stream_id)}


def play_stop_parameter(app_name: str) -> dict[str, str]:
    """StopPlay parameter (PlayStopParameter app_name)."""
    return {"app_name": str(app_name)}


# --- locomotion ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocomotionLimits:
    """Velocity envelope for SetVelocity, from the profile (vendor.locomotion).

    unverified: the envelope and the deadband are measured on hardware (gate
    6.3); the profile's lateral maximum is deliberately conservative for a
    biped. Unlike the X2 there is no documented minimum effective speed, so a
    sub-deadband component becomes an exact zero and anything else passes.
    """

    deadband: float = 0.005
    max_linear_mps: float = 1.0
    max_lateral_mps: float = 0.4
    max_angular_radps: float = 1.0

    @classmethod
    def from_profile(cls, config: dict[str, Any] | None) -> LocomotionLimits:
        entry = config or {}
        limits = cls(
            deadband=float(entry.get("deadband", 0.005)),
            max_linear_mps=float(entry.get("max_linear_mps", 1.0)),
            max_lateral_mps=float(entry.get("max_lateral_mps", entry.get("max_linear_mps", 0.4))),
            max_angular_radps=float(entry.get("max_angular_radps", 1.0)),
        )
        if limits.deadband < 0.0:
            raise ProjectionError(f"deadband must be non-negative, got {limits.deadband}")
        if limits.max_linear_mps <= 0.0 or limits.max_lateral_mps <= 0.0 or limits.max_angular_radps <= 0.0:
            raise ProjectionError("locomotion maxima must be positive")
        return limits


def twist_to_velocity(
    linear_x: float, linear_y: float, angular_z: float, limits: LocomotionLimits
) -> tuple[float, float, float]:
    """Convert a contract Twist into the SetVelocity (vx, vy, omega) triple.

    A component below the deadband becomes an exact zero (a drift on a
    joystick is not a command); a component above the profile maximum is
    rejected rather than clamped, for the same reason a joint target is:
    a clamp would move the robot faster than asked.
    """
    forward, lateral, angular = _finite((linear_x, linear_y, angular_z), "twist")
    out: list[float] = []
    for value, maximum, label in (
        (forward, limits.max_linear_mps, "linear.x"),
        (lateral, limits.max_lateral_mps, "linear.y"),
        (angular, limits.max_angular_radps, "angular.z"),
    ):
        if abs(value) < limits.deadband:
            out.append(0.0)
            continue
        if abs(value) > maximum:
            raise CommandRejected("VELOCITY_ABOVE_MAX", f"{label}={value} exceeds {maximum}")
        out.append(value)
    return out[0], out[1], out[2]


# --- state feedback projection ------------------------------------------------


def low_state_joint_positions(motor_state_q: Sequence[float]) -> dict[str, float]:
    """Map LowState.motor_state[0..28].q onto URDF joint names.

    ``motor_state_q`` is the 35-entry q vector (indices 29-34 carry no joint).
    Slots the LowState feed leaves at zero are still reported — LowState is a
    fixed-layout array where every slot always carries a value; separating a
    real zero from a missing sample is the freshness layer's job.
    """
    if len(motor_state_q) < 29:
        raise ProjectionError(f"low state carries {len(motor_state_q)} motor positions; the joint index range needs 29")
    return {name: float(motor_state_q[index]) for name, index in LOW_STATE_JOINT_INDEX.items()}


def dex3_joint_positions(motor_state_q: Sequence[float], *, side: str) -> dict[str, float]:
    """Map one HandState.motor_state[].q onto that hand's URDF joint names."""
    numbers = _finite(motor_state_q, f"{side} hand state")
    if len(numbers) != DEX3_MOTORS_PER_HAND:
        raise ProjectionError(
            f"{side} hand state must carry {DEX3_MOTORS_PER_HAND} motor positions, got {len(numbers)}"
        )
    return dict(zip(dex3_joint_names(side), numbers, strict=True))


def contract_joint_state(measurements: Mapping[str, float], *, order: Sequence[str]) -> tuple[list[str], list[float]]:
    """Assemble the contract /joint_states arrays from all vendor measurements.

    Joints named in ``order`` but absent from the vendor feeds are omitted,
    not zero-filled: a fabricated zero reads as a real measurement downstream.
    """
    names: list[str] = []
    positions: list[float] = []
    for name in order:
        if name in measurements:
            names.append(name)
            positions.append(float(measurements[name]))
    return names, positions


def imu_payload(imu_state: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Project LowState.imu_state onto a sensor_msgs/Imu-shaped plain dict.

    unitree_hg/IMUState carries quaternion[4], gyroscope[3], accelerometer[3],
    rpy[3]. unverified: the SDK does not document the quaternion component
    order; (w, x, y, z) is assumed from the unitree_go IMUState convention and
    confirmed on hardware (gate 6.1) before the frame is trusted for balance.
    """
    quaternion = _finite(imu_state["quaternion"], "imu quaternion")
    gyroscope = _finite(imu_state["gyroscope"], "imu gyroscope")
    accelerometer = _finite(imu_state["accelerometer"], "imu accelerometer")
    if len(quaternion) != 4 or len(gyroscope) != 3 or len(accelerometer) != 3:
        raise ProjectionError("imu state arrays must be quaternion[4], gyroscope[3], accelerometer[3]")
    w, x, y, z = quaternion
    return {
        "orientation": {"x": x, "y": y, "z": z, "w": w},
        "angular_velocity": {"x": gyroscope[0], "y": gyroscope[1], "z": gyroscope[2]},
        "linear_acceleration": {"x": accelerometer[0], "y": accelerometer[1], "z": accelerometer[2]},
    }


def power_state_payload(power_v: float, power_a: float) -> dict[str, float]:
    """Shape one vendor power reading for the ibrobot_msgs/PowerState surface.

    unverified: which vendor topic publishes the G1's pack state (LowState
    carries no battery fields; HandState.power_* is the hand board's rail);
    resolved in gate 6.1. The contract endpoint is bridge-owned (/power_state)
    regardless of the source.
    """
    voltage, current = _finite((power_v, power_a), "power state")
    if voltage < 0.0:
        raise ProjectionError(f"battery voltage must be non-negative, got {voltage}")
    return {"battery_voltage": voltage, "battery_current": current}


# --- freshness ----------------------------------------------------------------


def is_fresh(command_stamp_s: float, now_s: float, window_s: float) -> bool:
    """Whether a source last reported inside its acceptance window.

    The bridge stamps at receive time, so a stamp of 0 means "never
    reported" — exactly the case at stream-mode entry before the first
    LowState arrives. Treating that as fresh would let entry latch positions
    that do not exist (design D4), so never-reported sources are stale:
    silence is not a measurement.
    """
    if command_stamp_s <= 0.0:
        return False
    return abs(now_s - command_stamp_s) <= window_s


# --- mode machine (design D10) ------------------------------------------------


def mode_machine_fault(observed: int, expected: int) -> str | None:
    """Describe a LowState.mode_machine the profile does not declare, or None.

    The vendored profile and URDF describe one hardware generation
    (rev_1_0, 4010 wrist motors). A unit reporting a different mode_machine
    has no matching joint model in this repository; streaming it anyway would
    command joints that do not exist at the declared indices. The check fails
    closed at startup (unverified value itself: gate 6.1).
    """
    if int(observed) == int(expected):
        return None
    return (
        f"robot reports mode_machine {observed}, profile declares {expected}: this profile is "
        "built for the rev_1_0 generation; do not stream commands at a joint model "
        "the unit does not match"
    )


# --- modes and stop (design D5) -----------------------------------------------

#: Shared-weight target per runtime mode: idle and locomotion leave the
#: 17-joint frame under internal control (weight 0); stream ramps to 1.
#: Leaving stream first ramps down — the sequence is the bridge's, phase 2.
MODE_WEIGHT_TARGETS: dict[str, float] = {"idle": 0.0, "stream": 1.0, "locomotion": 0.0}


@dataclass(frozen=True, slots=True)
class StopPlan:
    """What StopRuntime will actually do for the requested policy.

    ``fsm_id`` is None when the plan switches no FSM: for HOLD, closing
    admission *is* the stop — the velocity command is zeroed, streaming
    targets freeze at their latch, and the shared weight ramps back to 0 so
    the vendor's internal controller takes the frame over. That is not a
    gap; it is the honest answer for a platform whose only "hold" is the
    vendor's own balance controller.
    """

    zero_velocity: bool
    weight_target: float
    fsm_id: int | None
    latch: bool
    downgraded_from: str = ""


def resolve_stop_plan(policy: str, *, fsm_id: int, stop_config: Mapping[str, Any] | None) -> StopPlan:
    """Decide the vendor actions for a stop policy, or refuse it.

    TORQUE_OFF is refuse-while-standing, fail-closed: the posture test uses
    the currently reported FSM id (LowState carries no body-pose field), and
    only the profile's safe set — sit and squat until gate 6.4 widens it —
    may release torque. Even then the performed action is the damping FSM
    (SetFsmId(1)), never the zero-torque FSM a caller might imagine: on a
    balancing biped, SetFsmId(0) drops the robot, and reporting TORQUE_OFF as
    performed while damping would be a lie. The plan says so via
    ``downgraded_from``.
    """
    config = dict(stop_config or {})
    if policy == "HOLD":
        if not config.get("zero_velocity_first", True):
            raise ProjectionError("profile stop.zero_velocity_first must be true; the G1 has no registered freeze")
        return StopPlan(zero_velocity=True, weight_target=0.0, fsm_id=None, latch=True)

    if policy != "TORQUE_OFF":
        raise ProjectionError(f"unsupported stop policy {policy!r}")

    safe_ids = {int(value) for value in config.get("torque_off_safe_fsm_ids", ())}
    if int(fsm_id) not in safe_ids:
        raise CommandRejected(
            "TORQUE_OFF_UNSAFE_POSTURE",
            f"reported fsm_id {fsm_id} is not in the safe set {sorted(safe_ids)}; removing torque would drop the robot",
        )
    return StopPlan(
        zero_velocity=True,
        weight_target=0.0,
        fsm_id=int(config.get("torque_off_fsm_id", FSM_IDS["damping"])),
        latch=True,
        downgraded_from="TORQUE_OFF",
    )


# --- vendor QoS ---------------------------------------------------------------

QOS_RELIABILITIES: frozenset[str] = frozenset({"reliable", "best_effort"})
QOS_DURABILITIES: frozenset[str] = frozenset({"volatile", "transient_local"})


def vendor_qos(
    qos_config: Mapping[str, Any] | None,
    key: str,
    *,
    reliability: str = "best_effort",
    durability: str = "volatile",
    depth: int = 10,
) -> tuple[str, str, int]:
    """Resolve the QoS for one vendor endpoint, as the profile declares it.

    QoS is part of a vendor endpoint's contract, not a stylistic choice: a
    BEST_EFFORT publisher cannot serve a RELIABLE subscription — every
    message is silently dropped. unverified: the G1 endpoint profiles are
    assumed from DDS high-rate-stream conventions and measured with
    `ros2 topic info -v` on hardware (gate 6.1) before conformance.
    """
    entry = dict((qos_config or {}).get(key) or {})
    resolved_reliability = str(entry.get("reliability", reliability))
    resolved_durability = str(entry.get("durability", durability))
    if resolved_reliability not in QOS_RELIABILITIES:
        raise ProjectionError(f"vendor qos {key!r}: unknown reliability {resolved_reliability!r}")
    if resolved_durability not in QOS_DURABILITIES:
        raise ProjectionError(f"vendor qos {key!r}: unknown durability {resolved_durability!r}")
    resolved_depth = int(entry.get("depth", depth))
    if resolved_depth < 1:
        raise ProjectionError(f"vendor qos {key!r}: depth must be positive, got {resolved_depth}")
    return resolved_reliability, resolved_durability, resolved_depth


# --- expert tier (design D11) ---------------------------------------------------

#: The rt/lowcmd tier (raw MotorCmd broadcast + CRC32 requirement) is disabled
#: by default and not implemented in this change. The constant exists so the
#: phase-2 bridge can fail closed if a profile ever enables it without the
#: CRC implementation landing first.
EXPERT_TIER_IMPLEMENTED = False


def expert_tier_gate(enabled: bool) -> None:
    """Refuse an expert tier this codebase cannot serve.

    Design D11 keeps the unprotected low-level joint command tier off; a
    profile flipping it on must not silently publish un-CRC'd lowcmd frames.
    """
    if enabled and not EXPERT_TIER_IMPLEMENTED:
        raise ProjectionError(
            "vendor expert_joint_tier.enabled is true, but the rt/lowcmd tier is not implemented "
            "in this change (design D11); refusing to start"
        )
