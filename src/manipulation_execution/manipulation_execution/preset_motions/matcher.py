"""Match an overhead heart from the selected actor's raw PEAR body pose.

No ROS objects, model inference or robot commands are created here. One caller
must serialize access: NeutralFK reuses internal buffers. Rule scores describe
geometry, not calibrated probabilities or precise finger-heart recognition.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from numbers import Integral, Real

import numpy as np

from manipulation_execution.imitation_retargeting.rotations import decode_rotation_6d
from manipulation_execution.imitation_retargeting.x2._vendor.neutral_fk import NeutralFK
from manipulation_execution.imitation_retargeting.x2._vendor.smplx2robot.source import JOINT, torso_coordinates

_NS = 1_000_000_000
_EPS = 1e-8
_SCHEMA = "pear_parameter_v1"


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return number


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _stamp_ns(message) -> int:
    stamp = message.header.stamp
    sec = _integer(stamp.sec, "header.stamp.sec")
    nanosec = _integer(stamp.nanosec, "header.stamp.nanosec")
    if nanosec >= _NS:
        raise ValueError("header.stamp.nanosec must be less than one second")
    return sec * _NS + nanosec


@dataclass(frozen=True)
class HeartBounds:
    """Body-frame distances are in shoulder widths, angles in degrees."""

    wrist_height_min: float = 0.50
    elbow_height_min: float = -0.15
    wrist_span_max: float = 1.45
    wrist_center_offset_max: float = 0.65
    forearm_inward_min: float = 0.10
    elbow_angle_min_deg: float = 35.0
    elbow_angle_max_deg: float = 150.0

    def __post_init__(self):
        for item in fields(self):
            _finite(getattr(self, item.name), item.name)
        if self.wrist_span_max <= 0 or self.wrist_center_offset_max < 0 or self.forearm_inward_min < 0:
            raise ValueError("heart span, offset and inward bounds must have valid signs")
        if not 0 <= self.elbow_angle_min_deg < self.elbow_angle_max_deg <= 180:
            raise ValueError("heart elbow bounds must satisfy 0 <= minimum < maximum <= 180")


@dataclass(frozen=True)
class HeartRuleConfig:
    enter: HeartBounds = field(default_factory=HeartBounds)
    keep: HeartBounds = field(default_factory=lambda: HeartBounds(0.30, -0.30, 1.70, 0.85, 0.0, 20.0, 165.0))

    def __post_init__(self):
        if not isinstance(self.enter, HeartBounds) or not isinstance(self.keep, HeartBounds):
            raise ValueError("heart enter and keep must be HeartBounds")
        minimums = ("wrist_height_min", "elbow_height_min", "forearm_inward_min", "elbow_angle_min_deg")
        maximums = ("wrist_span_max", "wrist_center_offset_max", "elbow_angle_max_deg")
        if any(getattr(self.keep, key) > getattr(self.enter, key) for key in minimums) or any(
            getattr(self.keep, key) < getattr(self.enter, key) for key in maximums
        ):
            raise ValueError("heart keep bounds must contain the enter region")


@dataclass(frozen=True)
class MatcherConfig:
    rule: HeartRuleConfig = field(default_factory=HeartRuleConfig)
    enter_confirm_s: float = 0.5
    exit_confirm_s: float = 0.4
    min_enter_frames: int = 3
    max_frame_age_s: float = 0.8
    max_receive_gap_s: float = 0.35
    actor_max_age_s: float = 1.0
    future_tolerance_s: float = 0.1
    clock_jump_tolerance_s: float = 1.0

    def __post_init__(self):
        if not isinstance(self.rule, HeartRuleConfig):
            raise ValueError("rule must be HeartRuleConfig")
        for name in ("enter_confirm_s", "exit_confirm_s", "future_tolerance_s"):
            _finite(getattr(self, name), name, minimum=0.0)
        for name in ("max_frame_age_s", "max_receive_gap_s", "actor_max_age_s", "clock_jump_tolerance_s"):
            if _finite(getattr(self, name), name, minimum=0.0) <= 0:
                raise ValueError(f"{name} must be positive")
        _integer(self.min_enter_frames, "min_enter_frames", minimum=2)


@dataclass(frozen=True)
class MatchResult:
    template_name: str
    score: float
    active: bool
    entered: bool
    exited: bool
    reason: str
    actor_id: int | None
    source_stamp_ns: int | None
    sequence: int | None
    session_epoch: int
    features: Mapping[str, float]


@dataclass(frozen=True)
class _Actor:
    stamp_ns: int
    received_s: float
    actor_id: int
    present: bool
    generation: int


def pose_features(points) -> dict[str, float]:
    """Extract geometry independently of body scale, translation and root tilt."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (145, 3) or not np.isfinite(points).all():
        raise ValueError("body FK must contain 145 finite 3D points")
    left_shoulder = points[JOINT["l_shoulder"]]
    right_shoulder = points[JOINT["r_shoulder"]]
    width = float(np.linalg.norm(left_shoulder - right_shoulder))
    up = points[JOINT["neck"]] - points[JOINT["pelvis"]]
    across = points[JOINT["l_collar"]] - points[JOINT["r_collar"]]
    if width < _EPS or np.linalg.norm(up) < _EPS:
        raise ValueError("degenerate shoulder width or torso height")
    up = up / np.linalg.norm(up)
    if np.linalg.norm(across - np.dot(across, up) * up) < _EPS:
        raise ValueError("degenerate torso lateral axis")
    frame = torso_coordinates(points[None])[0]
    if not np.isfinite(frame).all() or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-5):
        raise ValueError("invalid torso coordinate frame")
    local = (points - (left_shoulder + right_shoulder) / 2) @ frame.T / width
    features = {"shoulder_width_m": width, "head_height": float(local[JOINT["head"], 2])}
    for side, prefix in (("left", "l"), ("right", "r")):
        shoulder, elbow, wrist = (local[JOINT[f"{prefix}_{name}"]] for name in ("shoulder", "elbow", "wrist"))
        upper, lower = shoulder - elbow, wrist - elbow
        denominator = float(np.linalg.norm(upper) * np.linalg.norm(lower))
        if denominator < _EPS:
            raise ValueError("degenerate arm segment")
        features[f"{side}_wrist_height"] = float(wrist[2])
        features[f"{side}_elbow_height"] = float(elbow[2])
        features[f"{side}_forearm_inward"] = float((elbow[1] - wrist[1]) * (1 if side == "left" else -1))
        features[f"{side}_elbow_angle_deg"] = math.degrees(
            math.acos(float(np.clip(np.dot(upper, lower) / denominator, -1.0, 1.0)))
        )
        features[f"{side}_wrist_forward"] = float(wrist[0])
    wrists = local[[JOINT["l_wrist"], JOINT["r_wrist"]]]
    features["wrist_span"] = abs(float(wrists[0, 1] - wrists[1, 1]))
    features["wrist_center_offset"] = abs(float(np.mean(wrists[:, 1])))
    return features


def _evaluate(features: Mapping[str, float], bounds: HeartBounds) -> tuple[bool, float]:
    margins = []
    for side in ("left", "right"):
        margins.extend(
            (
                (features[f"{side}_wrist_height"] - bounds.wrist_height_min) / 0.5,
                (features[f"{side}_elbow_height"] - bounds.elbow_height_min) / 0.5,
                (features[f"{side}_forearm_inward"] - bounds.forearm_inward_min) / 0.5,
                (features[f"{side}_elbow_angle_deg"] - bounds.elbow_angle_min_deg) / 60.0,
                (bounds.elbow_angle_max_deg - features[f"{side}_elbow_angle_deg"]) / 60.0,
            )
        )
    margins.extend(
        (
            (bounds.wrist_span_max - features["wrist_span"]) / 0.75,
            (bounds.wrist_center_offset_max - features["wrist_center_offset"]) / 0.5,
        )
    )
    worst = min(margins)
    return worst >= -1e-9, float(np.clip(0.5 + worst, 0.0, 1.0))


class PoseMatcher:
    """One serialized matcher instance for one stream and one mirror session.

    set_session arms a new observation barrier. The next tick/PEAR callback
    establishes its ROS time; camera frames captured before that point cannot
    reactivate a stopped round. Call tick before feeding the first new frame
    when the adapter already has a current ROS clock.
    """

    def __init__(self, config: MatcherConfig | None = None):
        self.config = config or MatcherConfig()
        if not isinstance(self.config, MatcherConfig):
            raise ValueError("config must be MatcherConfig")
        self._fk = NeutralFK()
        self._actors: deque[_Actor] = deque(maxlen=64)
        self._actor_generation = 0
        self._epoch = 0
        self._enabled = False
        self._active = False
        self._exit_event = False
        self._entered_event = False
        self._reason = "inactive"
        self._score = 0.0
        self._features: dict[str, float] = {}
        self._actor_id: int | None = None
        self._source_ns: int | None = None
        self._sequence: int | None = None
        self._last_seen_ns: int | None = None
        self._last_seen_sequence: int | None = None
        self._minimum_source_ns = -1
        self._arm_ros_ns: int | None = None
        self._last_ros_s: float | None = None
        self._last_mono_s: float | None = None
        self._last_valid_receive_s: float | None = None
        self._pending_source_ns: int | None = None
        self._pending_receive_s: float | None = None
        self._pending_frames = 0
        self._bad_source_ns: int | None = None
        self._bad_receive_s: float | None = None

    def _clear_confirmation(self):
        self._pending_source_ns = self._pending_receive_s = None
        self._pending_frames = 0
        self._bad_source_ns = self._bad_receive_s = None

    def _deactivate(self, reason: str):
        self._exit_event |= self._active
        self._active = False
        self._clear_confirmation()
        self._reason = reason

    def _result(self) -> MatchResult:
        result = MatchResult(
            "overhead_heart",
            self._score,
            self._active,
            self._entered_event,
            self._exit_event,
            self._reason,
            self._actor_id,
            self._source_ns,
            self._sequence,
            self._epoch,
            dict(self._features),
        )
        self._entered_event = self._exit_event = False
        return result

    def set_session(self, *, epoch: int, active: bool) -> MatchResult:
        epoch = _integer(epoch, "epoch")
        if type(active) is not bool:
            raise ValueError("active must be a boolean")
        if epoch == self._epoch and active == self._enabled:
            return self._result()
        self._deactivate("session_changed" if active else "stopped")
        self._epoch, self._enabled = epoch, active
        self._actors.clear()
        self._actor_generation += 1
        self._minimum_source_ns = max(self._minimum_source_ns, self._last_seen_ns or -1)
        self._arm_ros_ns = None
        self._last_valid_receive_s = None
        self._source_ns = self._sequence = self._actor_id = None
        self._score, self._features = 0.0, {}
        return self._result()

    def observe_actor(self, message: object, *, now_mono_s: float) -> None:
        try:
            now = _finite(now_mono_s, "now_mono_s")
            stamp = _stamp_ns(message)
            actor_id = int(message.actor_id)
            if isinstance(message.actor_id, bool) or not isinstance(message.actor_id, Integral):
                raise ValueError("actor_id must be an integer")
            if type(message.actor_present) is not bool or type(message.start_ready) is not bool:
                raise ValueError("actor presence fields must be boolean")
            present = message.actor_present and message.start_ready and actor_id >= 0
        except (AttributeError, TypeError, ValueError, OverflowError):
            self._actors.clear()
            self._actor_generation += 1
            self._deactivate("invalid_actor")
            return
        previous = self._actors[-1] if self._actors else None
        if previous and stamp <= previous.stamp_ns:
            return  # Repeated cached messages must not refresh the heartbeat.
        if previous and (
            present != previous.present
            or actor_id != previous.actor_id
            or now - previous.received_s > self.config.actor_max_age_s
            or now < previous.received_s
        ):
            self._actor_generation += 1
            self._deactivate("actor_changed")
        self._actors.append(_Actor(stamp, now, actor_id, present, self._actor_generation))
        if not present:
            self._deactivate("no_actor")

    def _clock(self, now_ros_s: float, now_mono_s: float) -> tuple[float, float, bool]:
        ros, mono = _finite(now_ros_s, "now_ros_s", minimum=0), _finite(now_mono_s, "now_mono_s")
        changed = self._last_mono_s is not None and (
            mono < self._last_mono_s
            or ros < self._last_ros_s - self.config.future_tolerance_s
            or abs((ros - self._last_ros_s) - (mono - self._last_mono_s)) > self.config.clock_jump_tolerance_s
        )
        if changed:
            self._deactivate("clock_reset")
            self._actors.clear()
            self._actor_generation += 1
            self._last_seen_ns = self._last_seen_sequence = None
            self._minimum_source_ns = -1
            self._last_valid_receive_s = None
            self._arm_ros_ns = int(ros * _NS)
        if self._enabled and self._arm_ros_ns is None:
            self._arm_ros_ns = int(ros * _NS)
        self._last_ros_s, self._last_mono_s = ros, mono
        return ros, mono, not changed

    def _current_actor(self, ros: float, mono: float) -> _Actor | None:
        if not self._actors:
            return None
        actor = self._actors[-1]
        source_age = ros - actor.stamp_ns / _NS
        if (
            not actor.present
            or not 0 <= mono - actor.received_s <= self.config.actor_max_age_s
            or not -self.config.future_tolerance_s <= source_age <= self.config.actor_max_age_s
        ):
            return None
        return actor

    def _reject(self, reason: str, *, revoke: bool = True) -> MatchResult:
        if revoke:
            self._deactivate(reason)
            self._score = 0.0
        else:
            self._reason = reason
        return self._result()

    def observe_pear(self, message: object, *, now_ros_s: float, now_mono_s: float) -> MatchResult:
        ros, mono, clock_ok = self._clock(now_ros_s, now_mono_s)
        if not self._enabled:
            return self._reject("inactive")
        if not clock_ok:
            return self._result()
        try:
            stamp = _stamp_ns(message)
            sequence = _integer(message.sequence, "sequence")
            actor_id = _integer(message.actor_id, "actor_id")
            if message.schema != _SCHEMA or message.source != "yolox":
                return self._reject("invalid_source")
        except (AttributeError, TypeError, ValueError, OverflowError):
            return self._reject("invalid_metadata")
        age = ros - stamp / _NS
        if age > self.config.max_frame_age_s:
            return self._reject("source_stale")
        if age < -self.config.future_tolerance_s:
            return self._reject("source_future")
        if stamp < self._arm_ros_ns or stamp <= self._minimum_source_ns:
            return self._reject("source_before_session")
        if self._last_seen_ns is not None and stamp <= self._last_seen_ns:
            return self._reject("duplicate_or_out_of_order", revoke=False)
        if self._last_seen_sequence is not None and sequence <= self._last_seen_sequence:
            # A strictly newer source frame with a reset sequence identifies a
            # new producer generation. The first packet need not be received.
            self._deactivate("producer_restart")
        previous_source = self._last_seen_ns
        self._last_seen_ns, self._last_seen_sequence = stamp, sequence
        current_actor = self._current_actor(ros, mono)
        if current_actor is None:
            return self._reject("actor_stale")
        paired = next((actor for actor in reversed(self._actors) if actor.stamp_ns <= stamp), None)
        if (
            paired is None
            or not paired.present
            or paired.actor_id != actor_id
            or current_actor.actor_id != actor_id
            or paired.generation != current_actor.generation
            or stamp - paired.stamp_ns > self.config.actor_max_age_s * _NS
        ):
            return self._reject("actor_mismatch")
        try:
            raw = np.asarray(message.smplx_pose_raw, dtype=np.float64)
            if raw.shape != (312,) or not np.isfinite(raw).all():
                raise ValueError("PEAR must provide 312 finite values")
            rotations = decode_rotation_6d(raw)
            if np.max(np.abs(np.linalg.det(rotations) - 1.0)) > 1e-3 or not np.allclose(
                rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-3
            ):
                raise ValueError("degenerate 6D rotation")
            points = self._fk(
                {
                    "global_orient": rotations[0:1],
                    "body_pose": rotations[1:22],
                    "left_hand_pose": rotations[22:37],
                    "right_hand_pose": rotations[37:52],
                }
            )
            features = pose_features(points)
        except (AttributeError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            return self._reject("invalid_pose")
        if (
            not self._active
            and self._last_valid_receive_s is not None
            and (
                mono - self._last_valid_receive_s > self.config.max_receive_gap_s
                or previous_source is not None
                and (stamp - previous_source) / _NS > self.config.max_receive_gap_s
            )
        ):
            # Missing observations break entry confirmation. Once active,
            # valid negative observations must still accumulate exit time;
            # otherwise a slower stream could hold a released pose forever.
            self._clear_confirmation()
        self._last_valid_receive_s = mono
        self._source_ns, self._sequence, self._actor_id, self._features = stamp, sequence, actor_id, features
        enter, self._score = _evaluate(features, self.config.rule.enter)
        keep, _ = _evaluate(features, self.config.rule.keep)
        if not self._active:
            if not enter:
                self._clear_confirmation()
                self._reason = "pose_not_matched"
            else:
                if self._pending_source_ns is None:
                    self._pending_source_ns, self._pending_receive_s = stamp, mono
                self._pending_frames += 1
                ready = (
                    self._pending_frames >= self.config.min_enter_frames
                    and (stamp - self._pending_source_ns) / _NS + 1e-9 >= self.config.enter_confirm_s
                    and mono - self._pending_receive_s + 1e-9 >= self.config.enter_confirm_s
                )
                self._active = ready
                self._entered_event = ready
                self._reason = "matched" if ready else "confirming"
                if ready:
                    self._clear_confirmation()
        elif keep:
            self._bad_source_ns = self._bad_receive_s = None
            self._reason = "holding"
        else:
            if self._bad_source_ns is None:
                self._bad_source_ns, self._bad_receive_s = stamp, mono
            if (
                (stamp - self._bad_source_ns) / _NS + 1e-9 >= self.config.exit_confirm_s
                and mono - self._bad_receive_s + 1e-9 >= self.config.exit_confirm_s
            ):
                self._deactivate("pose_exited")
            else:
                self._reason = "exit_confirming"
        return self._result()

    def tick(self, *, now_ros_s: float, now_mono_s: float) -> MatchResult:
        ros, mono, clock_ok = self._clock(now_ros_s, now_mono_s)
        if not self._enabled:
            return self._reject("inactive")
        if not clock_ok:
            return self._result()
        if self._current_actor(ros, mono) is None:
            return self._reject("actor_stale")
        if self._last_valid_receive_s is None or self._source_ns is None:
            self._reason = "awaiting_pose"
            return self._result()
        if (
            mono - self._last_valid_receive_s > self.config.max_frame_age_s
            or ros - self._source_ns / _NS > self.config.max_frame_age_s
        ):
            return self._reject("pose_stale")
        if not self._active and mono - self._last_valid_receive_s > self.config.max_receive_gap_s:
            self._clear_confirmation()
            self._reason = "confirmation_gap"
        return self._result()
