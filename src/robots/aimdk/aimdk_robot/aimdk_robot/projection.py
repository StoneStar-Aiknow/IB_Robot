"""Pure projection between the public runtime contract and the AgiBot X2 MC tier.

No ROS and no vendor imports live here, so every conversion, threshold and
rejection rule is testable without a robot, without the AimDK SDK and without a
DDS participant. ``vendor_gateway`` owns the vendor message types; this module
only produces and consumes plain Python values.

The rules applied at the platform's rate -- command channel payloads, holds,
velocity limits, joint state aggregation -- are the data plane's and live in
``aimdk_dataplane`` (C++, ``projection.hpp``) with their tests; see
``aimdk_robot.dataplane``.

Every vendor constant below is cited to the shipped SDK (``aimdk
v0.0.0-gcdc995c``, docs ``AimDK_X2 1.1.0``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# --- vendor payload layouts -------------------------------------------------
# UpperBodyCommandArray.arm_pos: left 7 then right 7
# (Interface/control_mod/upper_body_control.html).
VENDOR_ARM_ORDER: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
# UpperBodyCommandArray.head_pos: [yaw, pitch]
VENDOR_HEAD_ORDER: tuple[str, ...] = ("head_yaw_joint", "head_pitch_joint")

# UpperBodyCommandArray.hand_sub_mode
HAND_SUB_MODE_CLAW = 1
HAND_SUB_MODE_DEXTEROUS_JOINT = 2
HAND_SUB_MODE_DEXTEROUS_GESTURE = 3

# Dexterous hand joint order per hand, from the SDK example's documented order.
DEXTEROUS_JOINT_ORDER: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)
DEXTEROUS_JOINTS_PER_HAND = len(DEXTEROUS_JOINT_ORDER)

# Vendor MC motion modes. This is the set the firmware *registers*, which is
# the documented mode table (Interface/control_mod/modeswitch.html) and the set
# the vendor's own set_mc_action example offers — not the McAction IDL
# enumeration, which still carries names the firmware dropped. The distinction
# is not academic: `SetMcAction` matches on `action_desc` (the `action` enum
# field is "v0.8.2开始不再使用") and answers code 3, "动作未在配置中登记", for a
# name that is in the enum but not in the configuration. The documentation says
# it outright — "请勿使用本节未提及的运动模式和模式描述设置".
VENDOR_MOTION_MODES: frozenset[str] = frozenset(
    {
        "PASSIVE_DEFAULT",  # zero torque, free joints; always permitted
        "DAMPING_DEFAULT",  # damped joints, safe takeover; always permitted
        "JOINT_DEFAULT",  # position-control stand, joints locked; posture-limited
        "STAND_DEFAULT",  # force-control stand with active balance
        "LOCOMOTION_DEFAULT",  # walk/run (unified with STAND_DEFAULT since v0.8.0)
        "HEAD_ONLY",
        "UPPERBODY_REMOTE_SPLIT",
    }
)
# The motion mode whose upper-body commands carry the hands: every command in it
# needs a hand_pos that fits its hand_sub_mode, or the platform drops it whole.
UPPER_BODY_HANDS_ACTION = "UPPERBODY_REMOTE_SPLIT"

# Posture skills, dispatched through the same service but absent from the mode
# table. Their registration has not been confirmed against hardware (the unit
# was on a gantry), so a deployment that uses them should verify each one.
VENDOR_SKILL_ACTIONS: frozenset[str] = frozenset(
    {
        "SIT_DOWN_DEFAULT",
        "CROUCH_DOWN_DEFAULT",
        "LIE_DOWN_DEFAULT",
        "STAND_UP_DEFAULT",
        "ASCEND_STAIRS",
        "DESCEND_STAIRS",
    }
)

VENDOR_ACTIONS: frozenset[str] = VENDOR_MOTION_MODES | VENDOR_SKILL_ACTIONS

# (motion, area) pairs the platform actually maps to an animation
# (Interface/control_mod/preset_motion.html). Since v0.8.0 `area` is no longer
# a body region — "area 原有的分区概念已经弱化, 仅和 motion 联合使用映射具体
# 动作" — so only these combinations exist, and neither the McPresetMotion nor
# the McControlArea enumeration is a guide to what is valid: the enums list
# values (3015 HITCLAP, area 4 HEAD, area 8 WAIST) that appear in no pair, and
# the table lists pairs (3017/11, 3031/11) that appear in no enum.
PRESET_MOTION_AREAS: dict[int, tuple[int, ...]] = {
    1001: (1, 2),  # raise hand, left / right
    1002: (1, 2),  # wave
    1003: (1, 2),  # handshake
    1004: (1, 2),  # blow kiss
    1007: (1, 2, 3),  # heart, one hand or both
    1008: (1, 2),  # clap hands (the single-hand animation)
    1010: (1, 2, 3),  # arms out
    1011: (1, 2),  # wave at chest height
    1013: (1, 2),  # salute
    3001: (11,),  # bow
    3007: (11,),  # light wave
    3008: (11,),  # hug
    3009: (11,),  # arms crossed
    3011: (11,),  # cheer
    3017: (11,),  # applause, two-handed
    3024: (11,),  # scratch head
    3025: (11,),  # scratch backside
    3031: (11,),  # bye bye
}

PRESET_MOTION_PAIRS: frozenset[tuple[int, int]] = frozenset(
    (motion, area) for motion, areas in PRESET_MOTION_AREAS.items() for area in areas
)

# SetMcAction response.header.code -> contract-visible reason
# (Interface/control_mod/modeswitch.html).
MODE_REJECTION_REASONS: dict[int, str] = {
    0: "",
    2: "INVALID_REQUEST",
    3: "UNKNOWN_ACTION",
    4: "INVALID_POSTURE",
    5: "IN_RECOVERY",
    6: "SAFETY_FORBIDS_TRANSITION",
    7: "PLATFORM_STARTING",
    8: "MOVING_BUSY",
    9: "MOTION_BUSY",
    10: "NO_TRANSITION_PATH",
}

# McCommonState.body_status values (mc/data/msg/common/McBodyPoseStatus.msg).
BODY_POSE_UNKNOWN = 0
BODY_POSE_STAND = 1
BODY_POSE_LIE_FACE_UP = 2
BODY_POSE_LIE_FACE_DOWN = 3
BODY_POSE_SIT = 4
BODY_POSE_SQUAT = 5
BODY_POSE_NAMES: dict[int, str] = {
    BODY_POSE_UNKNOWN: "unknown",
    BODY_POSE_STAND: "stand",
    BODY_POSE_LIE_FACE_UP: "lie_face_up",
    BODY_POSE_LIE_FACE_DOWN: "lie_face_down",
    BODY_POSE_SIT: "sit",
    BODY_POSE_SQUAT: "squat",
}
# Postures in which removing joint torque does not drop the robot.
TORQUE_OFF_SAFE_POSES: frozenset[int] = frozenset(
    {BODY_POSE_LIE_FACE_UP, BODY_POSE_LIE_FACE_DOWN, BODY_POSE_SIT, BODY_POSE_SQUAT}
)

# CommonTaskResponse.state (common/msg/CommonState.msg). GetMcPresetMotionState
# reports RUNNING while a preset motion executes and SUCCESS once it has
# finished (Interface/control_mod/preset_motion.html); the other values are
# terminal failures.
TASK_STATE_UNKNOWN = 0
TASK_STATE_SUCCESS = 1
TASK_STATE_FAILURE = 2
TASK_STATE_ABORTED = 3
TASK_STATE_TIMEOUT = 4
TASK_STATE_INVALID = 5
TASK_STATE_IN_MANUAL = 6
TASK_STATE_NOT_READY = 100
TASK_STATE_PENDING = 200
TASK_STATE_CREATED = 300
TASK_STATE_RUNNING = 400
TASK_STATE_NAMES: dict[int, str] = {
    TASK_STATE_UNKNOWN: "UNKNOWN",
    TASK_STATE_SUCCESS: "SUCCESS",
    TASK_STATE_FAILURE: "FAILURE",
    TASK_STATE_ABORTED: "ABORTED",
    TASK_STATE_TIMEOUT: "TIMEOUT",
    TASK_STATE_INVALID: "INVALID",
    TASK_STATE_IN_MANUAL: "IN_MANUAL",
    TASK_STATE_NOT_READY: "NOT_READY",
    TASK_STATE_PENDING: "PENDING",
    TASK_STATE_CREATED: "CREATED",
    TASK_STATE_RUNNING: "RUNNING",
}
TASK_STATES_IN_PROGRESS: frozenset[int] = frozenset(
    {TASK_STATE_NOT_READY, TASK_STATE_PENDING, TASK_STATE_CREATED, TASK_STATE_RUNNING}
)

# Balance finite-state machine (``McFsmState``). Keep the raw value for
# diagnostics only: the SDK message and generated documentation disagree on
# the numeric meanings, and the vendor examples do not gate commands on it.
FSM_STATE_UNKNOWN = 0
FSM_STATE_STARTING = 1
FSM_STATE_STABLE = 2
FSM_STATE_MOVING = 3
FSM_STATE_SAFE = 4
FSM_STATE_SPECIAL = 5
FSM_STATE_TEST = 6


def fsm_diagnostic(state: int) -> str:
    """Format the vendor FSM value without assigning it control semantics."""
    return f"platform balance controller reported fsm_state={int(state)}"


# Upper-body motion player (``McPlayerState``), pushed with every state
# message. Faster and cheaper than polling GetMcPresetMotionState, which stays
# as the authority on a specific task id.
PLAYER_STATE_IDLE = 0
PLAYER_STATE_PRE_PLAYING = 1
PLAYER_STATE_PLAYING = 2
PLAYER_STATE_INTERRUPTING = 3
PLAYER_STATE_ERROR = 4
PLAYER_STATES_BUSY: frozenset[int] = frozenset(
    {PLAYER_STATE_PRE_PLAYING, PLAYER_STATE_PLAYING, PLAYER_STATE_INTERRUPTING}
)

# McHandStatus values (``McRuntimeModel``): a second, continuous source for
# what GetHandType answers once at startup.
HAND_STATUS_OMNI_HAND = 1000
HAND_STATUS_OMNI_PICKER = 2000
HAND_STATUS_FAMILIES: dict[int, str] = {
    HAND_STATUS_OMNI_HAND: "dexterous",
    HAND_STATUS_OMNI_PICKER: "claw",
}


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


# --- locomotion -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocomotionLimits:
    """Vendor velocity envelope (py_examples/mc_locomotion_velocity.py)."""

    deadband: float = 0.005
    min_linear_mps: float = 0.2
    min_angular_radps: float = 0.1
    max_linear_mps: float = 1.0
    max_angular_radps: float = 1.0

    @classmethod
    def from_profile(cls, config: dict[str, Any] | None) -> LocomotionLimits:
        config = config or {}
        return cls(
            deadband=float(config.get("deadband", 0.005)),
            min_linear_mps=float(config.get("min_linear_mps", 0.2)),
            min_angular_radps=float(config.get("min_angular_radps", 0.1)),
            max_linear_mps=float(config.get("max_linear_mps", 1.0)),
            max_angular_radps=float(config.get("max_angular_radps", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class SpeedEnvelope:
    """The platform's currently reported speed bounds (``McLocomotionSpeedStatus``)."""

    mode: int = 0
    max_linear_mps: float = 0.0
    max_angular_radps: float = 0.0

    @property
    def usable(self) -> bool:
        """Whether the platform actually reported an envelope.

        All-zero bounds mean "no envelope reported" (the zero-speed mode, or a
        firmware that does not populate it), not "this robot may not move": a
        zero maximum would refuse every command, so it is ignored rather than
        enforced.
        """
        return self.max_linear_mps > 0.0 and self.max_angular_radps > 0.0


# --- freshness --------------------------------------------------------------


def is_fresh(command_stamp_s: float, now_s: float, window_s: float) -> bool:
    """The vendor drops upper-body commands stamped outside its window (200 ms)."""
    if command_stamp_s <= 0.0:
        # Vendor rule: stamp 0 means "use receive time".
        return True
    return abs(now_s - command_stamp_s) <= window_s


# --- modes and stop ---------------------------------------------------------


def mode_action(profile: dict[str, Any], mode: str) -> str:
    """Vendor MC motion mode required by a runtime mode."""
    actions = (profile.get("vendor") or {}).get("mc_actions") or {}
    try:
        action = str(actions[mode])
    except KeyError as exc:
        raise ProjectionError(f"profile declares no vendor MC action for mode {mode!r}") from exc
    if action not in VENDOR_MOTION_MODES:
        raise ProjectionError(
            f"mode {mode!r} maps to {action!r}, which the platform does not register as a motion mode"
        )
    return action


def mode_rejection_reason(code: int) -> str:
    """Vendor mode-switch code -> contract reason string."""
    return MODE_REJECTION_REASONS.get(int(code), f"PLATFORM_CODE_{int(code)}")


@dataclass(frozen=True, slots=True)
class StopPlan:
    """What StopRuntime will actually do for the requested policy.

    An empty ``vendor_action`` means no mode switch: closing admission *is* the
    stop. That is not a gap, it is the honest answer for a platform with no
    registered "freeze" mode — see ``resolve_stop_plan``.
    """

    vendor_action: str
    release_arbitration: bool
    latch: bool
    downgraded_from: str = ""


def resolve_stop_plan(policy: str, *, body_pose: int, stop_config: dict[str, Any] | None) -> StopPlan:
    """Decide the vendor action for a stop policy, or refuse it.

    A balancing biped cannot honour TORQUE_OFF while standing: removing torque
    drops it. The runtime refuses instead of quietly performing a hold and
    reporting success.

    HOLD has no explicit action on this platform, and the profile says so by
    leaving ``hold_action`` empty. No registered mode holds the robot from
    *every* posture: leaving STAND_DEFAULT would trade active balance for a
    position-held stand, JOINT_DEFAULT is refused from a seated posture, and
    STAND_DEFAULT from a seated posture is a stand-up — motion, which is the
    opposite of a stop. Closing admission holds the robot in whatever mode it
    is already in, and the platform's own input timeout zeroes the stream.
    """
    config = stop_config or {}
    hold_action = str(config.get("hold_action", ""))
    if policy == "HOLD":
        if hold_action and hold_action not in VENDOR_MOTION_MODES:
            raise ProjectionError(
                f"profile stop.hold_action {hold_action!r} is not a registered vendor motion mode; "
                f"the platform answers UNKNOWN_ACTION for names it does not register"
            )
        return StopPlan(vendor_action=hold_action, release_arbitration=True, latch=True)
    if policy != "TORQUE_OFF":
        raise ProjectionError(f"unsupported stop policy {policy!r}")

    torque_off_policy = str(config.get("torque_off_policy", "soft_estop"))
    if torque_off_policy == "soft_estop":
        return StopPlan(
            vendor_action=_stop_action(config, "soft_estop_action", "DAMPING_DEFAULT"),
            release_arbitration=True,
            latch=True,
            downgraded_from="TORQUE_OFF",
        )
    if torque_off_policy == "zero_torque_when_seated":
        if body_pose in TORQUE_OFF_SAFE_POSES:
            return StopPlan(
                vendor_action=_stop_action(config, "zero_torque_action", "PASSIVE_DEFAULT"),
                release_arbitration=True,
                latch=True,
            )
        raise CommandRejected(
            "TORQUE_OFF_UNSAFE_POSTURE",
            f"body pose is {BODY_POSE_NAMES.get(body_pose, body_pose)}; removing torque would drop the robot",
        )
    raise ProjectionError(f"unsupported stop.torque_off_policy {torque_off_policy!r}")


def _stop_action(config: dict[str, Any], key: str, default: str) -> str:
    """The vendor mode a stop policy maps onto, as the profile declares it."""
    action = str(config.get(key, default))
    if action not in VENDOR_MOTION_MODES:
        raise ProjectionError(
            f"profile stop.{key} {action!r} is not a registered vendor motion mode; "
            f"the platform answers UNKNOWN_ACTION for names it does not register"
        )
    return action


# --- vendor QoS -------------------------------------------------------------

QOS_RELIABILITIES: frozenset[str] = frozenset({"reliable", "best_effort"})
QOS_DURABILITIES: frozenset[str] = frozenset({"volatile", "transient_local"})


def vendor_qos(
    qos_config: dict[str, Any] | None,
    key: str,
    *,
    reliability: str = "best_effort",
    durability: str = "volatile",
    depth: int = 10,
) -> tuple[str, str, int]:
    """Resolve the QoS the vendor documents for one endpoint.

    QoS is part of a vendor endpoint's contract, not a stylistic choice: a
    BEST_EFFORT publisher cannot serve the platform's RELIABLE subscription —
    every message is silently dropped — and a VOLATILE subscription forgoes the
    latched last sample a TRANSIENT_LOCAL publisher offers. Each endpoint
    therefore carries the reliability and durability from the vendor's own
    interface tables, in the profile, where a firmware change can correct it
    without touching code.
    """
    entry = (qos_config or {}).get(key) or {}
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


def joint_state_rate_hz(profile: dict[str, Any]) -> float:
    """The rate the profile declares for ``/joint_states``: what the joint state views publish at."""
    return float(((profile.get("capabilities") or {}).get("joint.state") or {}).get("rate_hz", 50.0))


# --- interaction priorities -------------------------------------------------


def priority_level(neutral: int, *, levels: Sequence[tuple[int, int]]) -> int:
    """Map a neutral 0..100 priority onto a platform's discrete priority levels.

    ``levels`` is an ordered sequence of ``(threshold, platform_value)``; the
    entry with the highest threshold not above ``neutral`` wins. Platform
    priorities are enumerations with scheduling meaning, not a numeric range,
    so the mapping is a step function whose only outputs are the listed
    values: nothing in between and nothing beyond the last entry can be
    produced from the contract.
    """
    if not 0 <= int(neutral) <= 100:
        raise CommandRejected("PRIORITY_RANGE", f"priority {neutral} outside [0, 100]")
    if not levels or int(levels[0][0]) != 0:
        raise ProjectionError("priority levels must start at threshold 0")
    chosen = int(levels[0][1])
    previous = -1
    for threshold, value in levels:
        if int(threshold) <= previous:
            raise ProjectionError("priority level thresholds must be strictly increasing")
        previous = int(threshold)
        if int(neutral) >= int(threshold):
            chosen = int(value)
    return chosen


# --- clock agreement --------------------------------------------------------


def clock_skew_estimate(samples: Sequence[float]) -> float | None:
    """Best estimate of the clock offset from one-way stamp differences.

    Each sample is ``local_receive - remote_send``, which is the true offset
    *plus* a transport and queueing delay. That delay is always positive and
    highly variable — a callback that waited behind a burst of other work
    inflates its sample by however long it waited — so a single sample cannot
    distinguish "the clocks disagree" from "this callback was late". The
    minimum over a window can: delay only ever adds, so the smallest difference
    observed is the closest to the true offset, and one slow callback cannot
    drag it anywhere.

    Returns None when there is no evidence, which is not the same as agreement.
    """
    fresh = [float(sample) for sample in samples]
    return min(fresh) if fresh else None


def clock_skew_fault(skew_s: float | None, *, limit_s: float, window_s: float) -> str | None:
    """Describe a clock disagreement large enough to disable every command.

    The platform drops any command whose stamp falls outside its acceptance
    window, and says nothing when it does. A compute pack whose clock differs
    from the robot's by more than that window therefore streams perfectly valid
    commands into a silent discard — the robot simply does not move, no error
    is raised anywhere, and no log on either side names the cause. Returns the
    fault text, or None while the clocks agree closely enough to be usable.
    """
    if skew_s is None:
        return None
    if abs(skew_s) <= abs(limit_s):
        return None
    return (
        f"clock disagrees with the platform by {skew_s:+.3f}s (limit {limit_s:.3f}s, "
        f"platform command window {window_s:.3f}s): every streamed command would be "
        "discarded by the platform without an error. Synchronise this host's clock "
        "with the robot."
    )


#: The platform's audio focus priority band (``FocusRequester.msg``). Its
#: default is 6; anything outside 1..10 is not a focus priority at all.
AUDIO_FOCUS_PRIORITIES = range(1, 11)

#: ``priority + priority_weight%`` is the final focus priority, so the weight
#: only ever subdivides one band.
AUDIO_FOCUS_WEIGHTS = range(0, 100)


def audio_focus_request(config: dict[str, Any] | None) -> tuple[int, int, float, int]:
    """Resolve the audio focus parameters for raw playback.

    The platform does not take focus on a client's behalf: ``hal_audio``'s
    subscription to the playback topic never requests it, so a publisher that
    skips ``RequestAudioFocus`` contends with whatever else is playing instead
    of being mixed. Only the file-playback service manages focus internally.
    Returns ``(priority, weight, release_idle_s, buffer_chunks)``.
    """
    entry = config or {}
    priority = int(entry.get("focus_priority", 6))
    weight = int(entry.get("focus_priority_weight", 0))
    release_idle_s = float(entry.get("focus_release_idle_s", 1.0))
    buffer_chunks = int(entry.get("focus_buffer_chunks", 50))
    if priority not in AUDIO_FOCUS_PRIORITIES:
        raise ProjectionError(f"audio focus priority {priority} outside [1, 10]")
    if weight not in AUDIO_FOCUS_WEIGHTS:
        raise ProjectionError(f"audio focus priority weight {weight} outside [0, 99]")
    if release_idle_s <= 0.0:
        raise ProjectionError(f"audio focus release idle must be positive, got {release_idle_s}")
    if buffer_chunks < 1:
        raise ProjectionError(f"audio focus buffer must hold at least one chunk, got {buffer_chunks}")
    return priority, weight, release_idle_s, buffer_chunks


def resolve_named(name: str, table: dict[str, Any], *, reason: str) -> Any:
    """Look up a semantic name in a profile table, refusing unknown names."""
    try:
        return table[name]
    except KeyError as exc:
        raise CommandRejected(reason, f"unknown name {name!r}; declared: {sorted(table)}") from exc
