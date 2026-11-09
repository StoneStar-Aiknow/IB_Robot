"""Named motions and postures exposed as one neutral action server.

Three mechanisms sit behind ``/motion/execute_named``:

* preset motions (``SetMcPresetMotion``) — wave, handshake, clap, …
* posture MC actions (``SetMcAction``) — sit, squat, lie down, stand up, stairs
* recorded arm trajectories — the preset gestures replayed on the upper-body
  stream, for the modes in which the platform will not play a preset

Which one runs follows the runtime mode: a trajectory mode replays, every other
admitted mode uses the vendor mechanisms.

The same replay serves ``joint.trajectory``: a ``FollowJointTrajectory`` goal
the caller computed (a composed pose, a primitive) is validated against the
profile's arm limits and speed ceiling and replayed exactly like a recording,
with the same admission, ease-in, cancel-and-hold and follow check. Named
motions and trajectory goals share one executor: one arm motion at a time.

All are "run this named thing"; the caller does not need to know which
mechanism a given name uses. Names come from the profile, so an unknown name is
rejected instead of being mapped onto a neighbouring motion.

Admission is the runtime's, not the platform's: ``SetMcPresetMotion`` has no
priority protection and ``interrupt=true`` pre-empts whatever is playing
without comparing priorities (Interface/control_mod/preset_motion.html), so
this entry point applies the same lifecycle, arbitration and mode rules as the
streaming channels before anything reaches the vendor.

Completion is the platform's, not the dispatch acknowledgement's: a preset
motion is reported done when ``GetMcPresetMotionState`` says so, a posture
when the platform's own body-pose report shows it.

Only a replayed trajectory can be cancelled: it stops between ticks, holds the
arm where it is, and hands back a resume token. A later goal carrying
that token continues the motion from where it stopped, provided nothing has
moved the arm since; otherwise it plays from the start and says why.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from std_msgs.msg import String

from aimdk_robot import projection
from aimdk_robot import trajectory as traj
from aimdk_robot import vendor_gateway as vg
from aimdk_robot.display_completion import (
    display_completion_policy,
    final_pose_evidence,
    final_pose_reached,
    goal_acceptance_disabled,
)
from aimdk_robot.projection import CommandRejected
from ibrobot_msgs.action import ExecuteNamedMotion
from robot_runtime.contract import LIFECYCLE_ACTIVE

NAMED_MOTION_ACTION = "/motion/execute_named"

# McControlArea values usable with preset motions. Since v0.8.0 the vendor has
# "weakened" the area concept: it is no longer a body region but half of a
# (motion, area) pair that maps to one specific animation, and only the pairs
# the documentation tabulates exist. The usable areas there are 1 (left hand),
# 2 (right hand), 3 (both hands) and 11 (whole upper body, for the two-handed
# animations like clap, bow and hug). HEAD (4) and WAIST (8) are McControlArea
# members but appear in no documented preset pair, so they are not offered as
# targets: a caller asking for one would get a request the platform has no
# animation for.
CONTROL_AREAS: dict[str, int] = {"": 0, "left": 1, "right": 2, "both": 3, "body": 11}

# Below this commanded excursion the follow check is skipped: measurement noise
# would decide it, which happens when a resumed motion has little left to play.
FOLLOW_CHECK_MIN_RAD = 0.05


class _Superseded(Exception):
    """Raised inside a wait loop when a later goal or a stop took over."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Cancelled(Exception):
    """Raised inside a replay when its goal was cancelled; carries the resume point."""

    def __init__(self, message: str, token: str, progress: float) -> None:
        super().__init__(message)
        self.message = message
        self.token = token
        self.progress = progress


class _StopRequested(Exception):
    """Interrupt a leased pose with its first UUID-bound graceful stop target."""

    def __init__(self, target: tuple[float, ...]) -> None:
        super().__init__("named hold graceful stop requested")
        self.target = target


@dataclass(frozen=True)
class _Checkpoint:
    """Where a cancelled replay stopped, and what must not have changed to resume it."""

    token: str
    name: str
    mode: str
    offset_s: float
    stop_epoch: int
    arm_epoch: int
    stamp: float


class NamedMotionServer:
    """Serve ExecuteNamedMotion from the profile's motion tables."""

    def __init__(self, node: Any) -> None:
        self._node = node
        vendor = node.vendor_config
        self._presets = {str(k): dict(v) for k, v in (vendor.get("preset_motions") or {}).items()}
        self._postures = {str(k): str(v) for k, v in (vendor.get("postures") or {}).items()}
        config = vendor.get("named_motion") or {}
        self._allowed_modes = [str(mode) for mode in config.get("allowed_modes", ["idle"])]
        self._preset_timeout_s = float(config.get("preset_timeout_s", 30.0))
        self._poll_s = float(config.get("poll_s", 0.2))
        # How long a preset motion may take to show up in the player state before
        # "accepted but never played" is reported (T6 firmware path).
        self._start_grace_s = float(config.get("start_grace_s", 2.0))
        # "执行预设动作(请先进入稳定站立模式)": every documented preset pair is
        # annotated "稳定站立模式下执行". The runtime's own mode being right is
        # not the same as the platform being in that action — a robot on a
        # gantry may never reach stable stand — so the platform's own report is
        # what is checked.
        self._required_action = str(config.get("required_action", ""))
        # Modes in which named motions are replayed from recorded trajectories
        # on the arm channel instead of being handed to the platform. Files are
        # resolved relative to the profile and validated at startup.
        self._trajectory_modes = [str(mode) for mode in config.get("trajectory_modes", [])]
        # Caller-planned trajectories use the public mode capability. Keep
        # recorded gesture replay modes independent of policy-stream preparation.
        self._planned_trajectory_modes = [
            str(name)
            for name, mode in (node.profile.get("modes") or {}).items()
            if isinstance(mode, dict) and mode.get("allows_trajectory") is True
        ]
        self._blend_speed = float(config.get("trajectory_blend_speed_radps", 0.5))
        self._blend_min_s = float(config.get("trajectory_blend_min_s", 0.5))
        # A replay only succeeds if the measured arm followed it: the platform
        # drops upper-body commands it cannot use without saying so.
        self._follow_min_ratio = float(config.get("trajectory_follow_min_ratio", 0.3))
        # How long a cancelled replay can still be resumed from where it stopped.
        self._resume_ttl_s = float(config.get("trajectory_resume_ttl_s", 120.0))
        # How far the measured arm trails the commanded one. A cancel holds the
        # target sent this long ago: the target the arm is realizing now,
        # including the error the platform needs to hold it against gravity.
        self._hold_lag_s = float(config.get("trajectory_hold_lag_s", 0.05))
        self._rate_hz = float((vendor.get("command") or {}).get("rate_hz", 50.0))
        profile_dir = Path(str(node.get_parameter("profile").value)).parent
        max_speed = float(config.get("trajectory_max_speed_radps", 3.0))
        self._trajectories = {
            str(name): traj.load_trajectory(profile_dir / str(path), max_speed_radps=max_speed)
            for name, path in (config.get("trajectories") or {}).items()
        }
        return_names = config.get("cancel_return_motions", [])
        if not isinstance(return_names, list) or any(not isinstance(name, str) or not name for name in return_names):
            raise ValueError("cancel_return_motions must be a list of non-empty recorded motion names")
        self._return_motions = set(return_names)
        if self._return_motions - set(self._trajectories):
            raise ValueError("cancel_return_motions must name recorded trajectories")
        self._return_timeout_s = float(config.get("trajectory_cancel_return_timeout_s", 60.0))
        self._return_settle_s = float(config.get("trajectory_cancel_return_settle_s", 2.0))
        self._return_tolerance = float(config.get("trajectory_cancel_return_tolerance_rad", 0.03))
        self._returning = False
        self._lock = threading.Lock()
        # (name, generation) of the goal currently executing; a later goal
        # with interrupt=true replaces it and the earlier wait loop notices.
        self._running: tuple[str, int] | None = None
        # Goals between entering _execute and returning their result. Unlike
        # _running it is taken before any vendor call, so it is what an
        # interrupt=false goal is refused against.
        self._executing = 0
        self._generation = 0
        # The latest cancelled replay; one slot, consumed by the goal that tries it.
        self._checkpoint: _Checkpoint | None = None
        self._server = ActionServer(
            node,
            ExecuteNamedMotion,
            NAMED_MOTION_ACTION,
            execute_callback=self._execute,
            goal_callback=self._accept,
            cancel_callback=self._on_cancel,
            callback_group=node.action_callback_group,
        )
        # joint.trajectory: caller-computed arm trajectories on the same replay.
        self._max_speed = max_speed
        arm = vendor.get("arm") or {}
        self._arm_limits = {str(k): (float(v[0]), float(v[1])) for k, v in (arm.get("limits") or {}).items()}
        self._display_completion = display_completion_policy(config.get("display_completion"), self._trajectories)
        self._init_hold(config)
        if self._return_motions:
            values = [
                self._return_timeout_s,
                self._return_settle_s,
                self._return_tolerance,
                self._blend_speed,
                max_speed,
                self._rate_hz,
            ]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("cancel return budgets, tolerance, speeds and rate must be finite and positive")
            if set(projection.VENDOR_ARM_ORDER) - set(self._arm_limits):
                raise ValueError("cancel return requires existing limits for every arm joint")
        self._trajectory_server = None
        endpoints = [str(e) for e in (node.profile.get("trajectory_actions") or [])]
        if endpoints:
            missing = sorted(set(projection.VENDOR_ARM_ORDER) - set(self._arm_limits))
            if missing or not self._planned_trajectory_modes:
                raise ValueError(
                    "trajectory_actions needs vendor.arm.limits for every arm joint and a trajectory mode; "
                    f"missing limits {missing}"
                )
            self._trajectory_server = ActionServer(
                node,
                FollowJointTrajectory,
                endpoints[0],
                execute_callback=self._execute_trajectory,
                goal_callback=self._accept_trajectory,
                # Caller-planned arm paths can always cancel between ticks.
                # Keep the stricter preset/posture cancel policy separate.
                cancel_callback=lambda _goal: CancelResponse.ACCEPT,
                callback_group=node.action_callback_group,
            )

    def _init_hold(self, config: dict) -> None:
        """Opt in bounded, leased poses without changing ordinary replay."""
        names = config.get("hold_motions", [])
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("hold_motions must be a list of non-empty motion names")
        if len(names) != len(set(names)):
            raise ValueError("hold_motions must be distinct")
        self._hold_names = frozenset(names)
        self._executing_hold = False
        self._hold_leases: dict[tuple[str, str], float] = {}
        self._hold_exit_targets: dict[tuple[str, str], tuple[float, ...]] = {}
        self._hold_stop_targets: dict[tuple[str, str], tuple[float, ...]] = {}
        self._hold_lease_lock = threading.Lock()
        self._hold_active: tuple[str, str] | None = None
        self._hold_subscription = None
        self._hold_lease_ack = None
        if not names:
            return
        if self._hold_names - self._trajectories.keys() or self._hold_names & self._return_motions:
            raise ValueError("hold_motions require recordings and cannot use cancel_return_motions")
        if self._display_completion and self._hold_names & set(self._display_completion["motions"]):
            raise ValueError("hold_motions require measured arrival, not playback-only completion")
        if set(projection.VENDOR_ARM_ORDER) - self._arm_limits.keys():
            raise ValueError("hold_motions require every native arm limit")
        for name in names:
            recording = self._trajectories[name]
            if not recording.hands or recording.target != "both":
                raise ValueError("hold_motions require a both-arm recording with 20 hand joints")
            if any(row != recording.positions[0] for row in recording.positions) or any(
                row != recording.hands[0] for row in recording.hands
            ):
                raise ValueError("hold_motions must describe one fixed arm-and-hand pose")
        self._hold_lease_timeout_s = float(config.get("hold_lease_timeout_s", 1.0))
        self._hold_max_duration_s = float(config.get("hold_max_duration_s", 60.0))
        self._hold_release_timeout_s = float(config.get("hold_release_timeout_s", 0.3))
        self._hold_feedback_skew_s = float(config.get("hold_feedback_skew_s", 0.04))
        values = [
            self._hold_lease_timeout_s,
            self._hold_max_duration_s,
            self._hold_release_timeout_s,
            self._hold_feedback_skew_s,
            self._blend_speed,
            self._blend_min_s,
            self._max_speed,
            self._rate_hz,
        ]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("hold timing and speed values must be finite and positive")
        if self._hold_max_duration_s > 60.0 or self._hold_lease_timeout_s > 1.0:
            raise ValueError("hold action cap cannot exceed 60 s and lease timeout cannot exceed 1 s")
        bands = config.get("hold_hand_limits")
        if not isinstance(bands, list) or len(bands) != 20:
            raise ValueError("hold_hand_limits must contain 20 native-radian limit pairs")
        try:
            self._hold_hand_limits = tuple((float(pair[0]), float(pair[1])) for pair in bands if len(pair) == 2)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("hold_hand_limits must contain finite ordered limit pairs") from exc
        if len(self._hold_hand_limits) != 20 or any(
            not math.isfinite(low) or not math.isfinite(high) or low > high for low, high in self._hold_hand_limits
        ):
            raise ValueError("hold_hand_limits must contain finite ordered limit pairs")
        speeds = config.get("hold_hand_speed_radps", self._blend_speed)
        speeds = [speeds] * 20 if isinstance(speeds, int | float) and not isinstance(speeds, bool) else speeds
        if not isinstance(speeds, list) or len(speeds) != 20:
            raise ValueError("hold_hand_speed_radps must be a positive scalar or 20 speeds")
        self._hold_hand_speeds = tuple(float(speed) for speed in speeds)
        if any(not math.isfinite(speed) or speed <= 0 for speed in self._hold_hand_speeds):
            raise ValueError("hold_hand_speed_radps must be finite and positive")
        topic = config.get("hold_lease_topic", "/runtime/named_motion_hold_lease")
        if not isinstance(topic, str) or not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("hold_lease_topic must be an absolute ROS topic")
        ack_topic = config.get("hold_lease_ack_topic", "/runtime/named_motion_hold_lease_ack")
        if (
            not isinstance(ack_topic, str)
            or not ack_topic.startswith("/")
            or any(char.isspace() for char in ack_topic)
            or ack_topic == topic
        ):
            raise ValueError("hold_lease_ack_topic must be a distinct absolute ROS topic")
        configure = getattr(self._node, "configure_named_hold_guard", None)
        if configure is not None:
            configure(config)
        self._hold_lease_ack = self._node.create_publisher(String, ack_topic, 10)
        self._hold_subscription = self._node.create_subscription(
            String, topic, self._on_hold_lease, 10, callback_group=self._node.action_callback_group
        )

    @staticmethod
    def _hold_goal_id(goal_handle) -> str:
        try:
            raw = bytes(goal_handle.goal_id.uuid)
            return uuid.UUID(bytes=raw).hex if len(raw) == 16 and any(raw) else ""
        except (AttributeError, TypeError, ValueError):
            return ""

    def _on_hold_lease(self, message: String) -> None:
        """Accept only a recent lease for an explicitly configured name/UUID."""
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict) or not {"goal_id", "motion"} <= set(value) <= {
                "goal_id",
                "motion",
                "exit_target",
                "stop_target",
            }:
                return
            if "exit_target" in value and "stop_target" in value:
                return
            motion = value["motion"]
            if not isinstance(motion, str) or motion not in self._hold_names:
                return
            goal_id = uuid.UUID(value["goal_id"]).hex
            if int(goal_id, 16) == 0:
                return
            exit_target = value.get("exit_target")
            stop_target = value.get("stop_target")
            for field in ("exit_target", "stop_target"):
                if field in value and (
                    not isinstance(value[field], list)
                    or len(value[field]) != 34
                    or any(type(item) not in (int, float) or not math.isfinite(item) for item in value[field])
                ):
                    return
        except (TypeError, ValueError, AttributeError, OverflowError):
            return
        now = time.monotonic()
        with self._hold_lease_lock:
            key = (goal_id, motion)
            self._hold_leases = {
                item: stamp
                for item, stamp in self._hold_leases.items()
                if 0 <= now - stamp <= self._hold_lease_timeout_s
            }
            if len(self._hold_leases) >= 32 and key not in self._hold_leases:
                oldest = min(
                    (item for item in self._hold_leases if item != self._hold_active),
                    key=self._hold_leases.__getitem__,
                    default=None,
                )
                if oldest is None:
                    return
                del self._hold_leases[oldest]
            self._hold_exit_targets = {
                item: target for item, target in self._hold_exit_targets.items() if item in self._hold_leases
            }
            self._hold_stop_targets = {
                item: target for item, target in self._hold_stop_targets.items() if item in self._hold_leases
            }
            self._hold_leases[key] = now
            if stop_target is not None:
                self._hold_stop_targets.setdefault(key, tuple(float(item) for item in stop_target))
                self._hold_exit_targets.pop(key, None)
            elif exit_target is not None and key not in self._hold_stop_targets:
                self._hold_exit_targets[key] = tuple(float(item) for item in exit_target)
        acknowledgement = getattr(self, "_hold_lease_ack", None)
        if acknowledgement is not None:
            acknowledgement.publish(String(data=json.dumps({"goal_id": goal_id, "motion": motion})))

    def _hold_lease_fresh(self, key: tuple[str, str], now: float) -> bool:
        with self._hold_lease_lock:
            stamp = self._hold_leases.get(key)
        return stamp is not None and 0.0 <= now - stamp <= self._hold_lease_timeout_s

    def _hold_limits(self, values) -> bool:
        if len(values) != 34 or any(not math.isfinite(value) for value in values):
            return False
        bands = [self._arm_limits[name] for name in projection.VENDOR_ARM_ORDER] + list(self._hold_hand_limits)
        return all(low <= value <= high for value, (low, high) in zip(values, bands, strict=True))

    def _hold_duration(self, start, target) -> float:
        speeds = [min(self._blend_speed, self._max_speed)] * 14 + list(self._hold_hand_speeds)
        # Cosine interpolation reaches pi/2 times its mean velocity.
        return max(
            self._blend_min_s,
            max(math.pi * abs(b - a) / (2 * speed) for a, b, speed in zip(start, target, speeds, strict=True)),
        )

    def _hold_validate(self, name, times, points) -> tuple[bool, str]:
        if any(not self._hold_limits(point) for point in points):
            return False, "named hold target violates native 34-joint limits"
        validate = getattr(self._node, "validate_named_hold_path", None)
        if validate is None:
            return False, "named hold geometry guard is unavailable"
        try:
            answer = validate(name, times, points)
            if not isinstance(answer, tuple) or len(answer) != 2 or type(answer[0]) is not bool:
                return False, "invalid named hold geometry response"
            return answer[0], str(answer[1])
        except Exception as exc:
            return False, f"named hold geometry failed: {exc}"

    def _hold_sample(self):
        sample = getattr(self._node, "measured_named_hold_sample", None)
        sample = None if sample is None else sample(max_skew_s=self._hold_feedback_skew_s)
        if sample is None or not self._hold_limits(sample[0]):
            raise _Superseded(
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                "named hold requires fresh simultaneous double-hand and arm feedback",
            )
        return list(sample[0])

    def _hold_feedback(self, goal_handle, state, progress):
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = state
        feedback.progress = progress
        goal_handle.publish_feedback(feedback)

    def _hold_check(self, key, mode, generation, epoch, started, *, allow_stop=True):
        self._still_current(generation, epoch)
        if self._node.current_mode() != mode or not self._node.output_admitted():
            raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "named hold lost runtime admission")
        now = time.monotonic()
        if now - started >= self._hold_max_duration_s:
            raise _Superseded(ExecuteNamedMotion.Result.TIMEOUT, "named hold reached its absolute duration limit")
        if not self._hold_lease_fresh(key, now):
            raise _Superseded(ExecuteNamedMotion.Result.TIMEOUT, "named hold lease expired")
        actual = self._hold_sample()
        ok, reason = self._hold_validate(key[1], [0.0, 1.0 / self._rate_hz], [actual, actual])
        if not ok:
            raise _Superseded(
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, f"measured named hold geometry rejected: {reason}"
            )
        if allow_stop:
            with self._hold_lease_lock:
                stop_target = self._hold_stop_targets.get(key)
            if stop_target is not None:
                raise _StopRequested(stop_target)
        return actual

    def _hold_command(self, values, generation, epoch):
        if not self._node.command_recorded(values[:14], values[14:], owner=generation, epoch=epoch):
            raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "named hold output was refused")
        sent = getattr(self, "_hold_sent", None)
        if sent is not None:
            now = time.monotonic()
            sent.append((now, list(values)))
            while len(sent) > 1 and sent[1][0] <= now - getattr(self, "_hold_lag_s", 0.05):
                sent.popleft()

    def _hold_cancel(self, name, values, goal_handle, generation, epoch):
        # Match the ordinary replay's tracking-lag hold, retaining all hands.
        sent = getattr(self, "_hold_sent", ())
        hold = list(sent[0][1]) if sent else list(values)
        actual = self._hold_sample()
        ok, reason = self._hold_validate(name, [0.0, self._hold_duration(actual, hold)], [actual, hold])
        if not ok:
            raise _Superseded(
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, f"named hold cancel target rejected: {reason}"
            )
        self._hold_command(hold, generation, epoch)
        raise _Cancelled(f"{name} canceled; complete arm-and-hand target held", "", 0.0)

    def _hold_move(self, name, mode, start, target, goal_handle, key, generation, epoch, started, state):
        duration = self._hold_duration(start, target)
        ok, reason = self._hold_validate(name, [0.0, duration], [start, target])
        if not ok:
            return False, reason, list(start)
        begin = time.monotonic()
        current = list(start)
        last_feedback = -math.inf
        while True:
            if goal_handle.is_cancel_requested:
                self._hold_cancel(name, current, goal_handle, generation, epoch)
            self._hold_check(key, mode, generation, epoch, started, allow_stop=state != "stopping")
            elapsed = time.monotonic() - begin
            current = traj.blend(start, target, elapsed / duration)
            self._hold_command(current, generation, epoch)
            if elapsed - last_feedback >= 0.1:
                self._hold_feedback(goal_handle, state, min(elapsed / duration, 1.0))
                last_feedback = elapsed
            if elapsed >= duration:
                return True, "named hold bridge completed", list(target)
            time.sleep(min(1.0 / self._rate_hz, max(0.0, duration - elapsed)))

    def _hold_wait_arrival(self, name, mode, point, goal_handle, key, generation, epoch, started, state="entering"):
        issued_at = time.monotonic()
        deadline = issued_at + self._return_settle_s
        last_feedback = -math.inf
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                self._hold_cancel(name, point, goal_handle, generation, epoch)
            self._hold_check(key, mode, generation, epoch, started, allow_stop=state != "stopping")
            self._hold_command(point, generation, epoch)
            now = time.monotonic()
            if now - last_feedback >= 0.1:
                self._hold_feedback(goal_handle, state, 1.0)
                last_feedback = now
            sample = self._node.measured_named_hold_sample(max_skew_s=self._hold_feedback_skew_s)
            if (
                sample is not None
                and sample[1] >= issued_at
                and max(abs(a - b) for a, b in zip(sample[0], point, strict=True)) <= self._return_tolerance
            ):
                return True
            time.sleep(1.0 / self._rate_hz)
        return False

    @staticmethod
    def _hold_target_message(name, final, field):
        identity = hashlib.sha256(
            json.dumps([float(value) for value in final], separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return json.dumps({"motion": name, field: identity}, separators=(",", ":"))

    def _run_hold_pose(self, name, mode, target, resume_token, goal_handle, generation, epoch):
        """Run only an explicitly leased, geometrically certified fixed pose."""
        if resume_token:
            raise CommandRejected("INVALID_TARGET", "held poses cannot resume an old execution")
        if target not in ("", "both"):
            raise CommandRejected("INVALID_TARGET", "held poses require both arms and both hands")
        required = str((self._node.vendor_config.get("mc_actions") or {}).get(mode, ""))
        if required and str(self._node.platform_action()) != required:
            return False, ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "platform action does not permit named hold"
        if not self._node.ensure_input_source():
            return False, ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "named hold input source is unavailable"
        key = (self._hold_goal_id(goal_handle), name)
        if not key[0] or not self._hold_lease_fresh(key, time.monotonic()):
            return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, "named hold requires its UUID-bound lease"
        start = self._hold_sample()
        recording = self._trajectories[name]
        pose = list(recording.positions[0]) + list(recording.hands[0])
        with self._hold_lease_lock:
            stop_target = self._hold_stop_targets.get(key)
        first_target = list(stop_target) if stop_target is not None else pose
        duration = self._hold_duration(start, first_target)
        ok, reason = self._hold_validate(name, [0.0, duration], [start, first_target])
        if not ok:
            if stop_target is not None:
                self._node.fail_named_hold(f"named hold stop target rejected: {reason}")
            return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, reason
        self._take_over(name, generation)
        if not self._node.claim_arm(generation):
            return False, ExecuteNamedMotion.Result.BUSY, "public stream has not released the arm channel"
        with self._hold_lease_lock:
            self._hold_active = key
        started = time.monotonic()
        current = start
        self._hold_sent = deque([(started, list(start))])
        normal_exit = canceled = False
        failure_reason = "named hold execution failed or arrival was unconfirmed"
        try:
            try:
                # Claim confirmation can take several feedback cycles. Rebuild
                # the entry from actual state, and honour a stop already leased
                # before acceptance without first sending the template pose.
                if goal_handle.is_cancel_requested:
                    self._hold_cancel(name, current, goal_handle, generation, epoch)
                start = self._hold_check(key, mode, generation, epoch, started)
                current = start
                ok, reason, current = self._hold_move(
                    name, mode, start, pose, goal_handle, key, generation, epoch, started, "entering"
                )
                if not ok:
                    return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, reason
                if not self._hold_wait_arrival(name, mode, current, goal_handle, key, generation, epoch, started):
                    return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, "named hold arrival unconfirmed"
                last_feedback = -math.inf
                while True:
                    if goal_handle.is_cancel_requested:
                        self._hold_cancel(name, current, goal_handle, generation, epoch)
                    actual = self._hold_check(key, mode, generation, epoch, started)
                    self._hold_command(current, generation, epoch)
                    now = time.monotonic()
                    if now - last_feedback >= 0.1:
                        self._hold_feedback(goal_handle, "holding", 1.0)
                        last_feedback = now
                    with self._hold_lease_lock:
                        exit_target = self._hold_exit_targets.get(key)
                    if exit_target is not None:
                        ok, reason, final = self._hold_move(
                            name,
                            mode,
                            actual,
                            list(exit_target),
                            goal_handle,
                            key,
                            generation,
                            epoch,
                            started,
                            "exiting",
                        )
                        if ok:
                            if not self._hold_wait_arrival(
                                name, mode, final, goal_handle, key, generation, epoch, started, state="exiting"
                            ):
                                return (
                                    False,
                                    ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                                    "named hold exit arrival unconfirmed",
                                )
                            normal_exit = True
                            return (
                                True,
                                ExecuteNamedMotion.Result.NONE,
                                self._hold_target_message(name, final, "exit_target_sha256"),
                            )
                        self._hold_feedback(goal_handle, "exit_blocked", 1.0)
                        # Reject this request without losing the safe holding pose.
                        with self._hold_lease_lock:
                            self._hold_exit_targets.pop(key, None)
                    time.sleep(1.0 / self._rate_hz)
            except _StopRequested as stop:
                # This bridge keeps the named owner's lease, all runtime guards
                # and measured arrival. It is a successful execution, not an
                # ActionCancel transition on a goal still in EXECUTING state.
                actual = self._hold_check(key, mode, generation, epoch, started, allow_stop=False)
                ok, reason, final = self._hold_move(
                    name, mode, actual, list(stop.target), goal_handle, key, generation, epoch, started, "stopping"
                )
                if not ok:
                    failure_reason = f"named hold stop target rejected: {reason}"
                    return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, failure_reason
                if not self._hold_wait_arrival(
                    name, mode, final, goal_handle, key, generation, epoch, started, state="stopping"
                ):
                    failure_reason = "named hold stop arrival unconfirmed"
                    return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, failure_reason
                normal_exit = True
                return (
                    True,
                    ExecuteNamedMotion.Result.NONE,
                    self._hold_target_message(name, final, "stop_target_sha256"),
                )
        except _Cancelled:
            canceled = True
            raise
        except Exception as exc:
            failure_reason = str(exc)
            raise
        finally:
            if not normal_exit and not canceled:
                self._node.fail_named_hold(failure_reason)
            with self._hold_lease_lock:
                self._hold_active = None
                self._hold_leases.pop(key, None)
                self._hold_exit_targets.pop(key, None)
                self._hold_stop_targets.pop(key, None)
            self._hold_sent = None
            release = getattr(self._node, "release_arm_confirmed", None)
            if release is None or not release(generation, wait_s=self._hold_release_timeout_s):
                fail = getattr(self._node, "fail_named_hold", None)
                if fail is not None:
                    fail("named hold owner release was not confirmed")
                raise _Superseded(
                    ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                    "named hold owner release was not confirmed; output remains stopped",
                )

    # --- goal handling ------------------------------------------------------

    def _accept(self, goal) -> GoalResponse:
        name = str(goal.name)
        if name not in self._presets and name not in self._postures and name not in self._trajectories:
            return GoalResponse.REJECT
        # Early answer only: two goals can both pass here before either has
        # started executing. _execute makes the check that holds.
        with self._lock:
            if (
                self._returning
                or getattr(self, "_executing_hold", False)
                or (self._executing and not bool(goal.interrupt))
            ):
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept a cancel only where it is honoured: a replay in a trajectory mode.

        A replayed trajectory stops between ticks. A posture transition is not
        interruptible (stopping half-way through sitting down is a fall), and
        the SDK exposes no "stop preset motion" call; accepting a cancel there
        and then ignoring it would be a lie, so it is refused and
        ``/runtime/stop`` named instead, which does interrupt everything.
        """
        if str(self._node.current_mode()) in self._trajectory_modes:
            return CancelResponse.ACCEPT
        self._node.get_logger().warning(
            "named motion cancel refused: the platform cannot cancel a running motion; use /runtime/stop"
        )
        return CancelResponse.REJECT

    def _admission_failure(self, *, planned: bool = False) -> tuple[int, str] | None:
        """Why a named motion may not be dispatched right now, or None."""
        state = self._node.runtime_state
        if state.stop_latched:
            return ExecuteNamedMotion.Result.STOP_LATCHED, "stop latched"
        if state.lifecycle != LIFECYCLE_ACTIVE:
            faults = "; ".join(state.faults) or "no detail"
            return ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, f"runtime is {state.lifecycle}: {faults}"
        if not self._node.holds_arbitration():
            return (
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                "runtime does not hold platform command arbitration",
            )
        mode = str(self._node.current_mode())
        if planned:
            if mode in self._planned_trajectory_modes:
                return None
            return (
                ExecuteNamedMotion.Result.MODE_NOT_ALLOWED,
                f"mode {mode!r} does not replay trajectories; use {self._planned_trajectory_modes}",
            )
        if mode not in self._allowed_modes and mode not in self._trajectory_modes:
            return (
                ExecuteNamedMotion.Result.MODE_NOT_ALLOWED,
                f"mode {mode!r} does not admit named motions; allowed: {self._allowed_modes + self._trajectory_modes}",
            )
        return None

    def _execute(self, goal_handle):
        """Refuse an interrupt=false goal while another executes, atomically with entering.

        The check is made here and not by reserving in the goal callback:
        rclpy can accept a goal and never execute it (the client went away
        before the response), and a reservation taken there would never be
        released.
        """
        request = goal_handle.request
        is_hold = str(request.name) in getattr(self, "_hold_names", ())
        with self._lock:
            busy = (
                self._returning
                or getattr(self, "_executing_hold", False)
                or (self._executing > 0 and not bool(request.interrupt))
            )
            if not busy:
                self._executing += 1
                if is_hold:
                    self._executing_hold = True
        if busy:
            return self._fail(
                goal_handle,
                ExecuteNamedMotion.Result(),
                ExecuteNamedMotion.Result.BUSY,
                "another named motion is executing and interrupt is false",
            )
        try:
            return self._execute_admitted(goal_handle)
        finally:
            with self._lock:
                self._executing -= 1
                if is_hold:
                    self._executing_hold = False

    # --- joint.trajectory ---------------------------------------------------

    def _accept_trajectory(self, goal) -> GoalResponse:
        """Refuse a goal that names no arm joint, an unknown one, or no point."""
        names = list(goal.trajectory.joint_names)
        if not names or not set(names) <= set(projection.VENDOR_ARM_ORDER) or len(set(names)) != len(names):
            return GoalResponse.REJECT
        if not goal.trajectory.points:
            return GoalResponse.REJECT
        with self._lock:
            if self._executing:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        with self._lock:
            busy = self._executing > 0
            if not busy:
                self._executing += 1
        if busy:
            return self._fail_trajectory(goal_handle, result, "another arm motion is executing")
        try:
            return self._execute_trajectory_admitted(goal_handle, result)
        finally:
            with self._lock:
                self._executing -= 1

    def _execute_trajectory_admitted(self, goal_handle, result):
        failure = self._admission_failure(planned=True)
        if failure is not None:
            return self._fail_trajectory(goal_handle, result, failure[1])
        mode = str(self._node.current_mode())
        if mode not in self._planned_trajectory_modes:
            return self._fail_trajectory(
                goal_handle, result, f"mode {mode!r} does not replay trajectories; use {self._planned_trajectory_modes}"
            )
        start = self._node.measured_arm_positions()
        if start is None:
            return self._fail_trajectory(goal_handle, result, "no fresh arm measurement to start the motion from")
        # Start from what the platform is driving: see arm_takeover_start.
        origin = self._node.arm_takeover_start(start)
        goal = goal_handle.request.trajectory
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in goal.points]
        try:
            plan = traj.planned_trajectory(
                "trajectory",
                origin,
                list(goal.joint_names),
                times,
                [list(p.positions) for p in goal.points],
                limits=self._arm_limits,
                max_speed_radps=self._max_speed,
                rate_hz=self._rate_hz,
            )
        except traj.TrajectoryError as exc:
            return self._fail_trajectory(goal_handle, result, str(exc))
        with self._lock:
            self._generation += 1
            generation = self._generation
        epoch = self._node.runtime_state.stop_epoch

        def on_progress(_progress: float, point: list[float]) -> None:
            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = list(projection.VENDOR_ARM_ORDER)
            feedback.desired.positions = [float(v) for v in point]
            goal_handle.publish_feedback(feedback)

        try:
            ok, _code, message = self._run_trajectory(
                "trajectory",
                mode,
                "",
                "",
                goal_handle,
                generation,
                epoch,
                recorded=plan,
                origin=origin,
                on_progress=on_progress,
                verify_completion=not (
                    self._node.vendor_config.get("named_motion", {}).get("allow_unverified_trajectories") is True
                    and goal_acceptance_disabled(goal_handle.request.goal_tolerance, projection.VENDOR_ARM_ORDER)
                ),
            )
        except _Superseded as superseded:
            return self._fail_trajectory(goal_handle, result, superseded.message)
        except _Cancelled as cancelled:
            goal_handle.canceled()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = cancelled.message
            return result
        finally:
            with self._lock:
                if self._running == ("trajectory", generation):
                    self._running = None
        if not ok:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = message
            goal_handle.abort()
            return result
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = message
        return result

    def _fail_trajectory(self, goal_handle, result, message: str):
        goal_handle.abort()
        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
        result.error_string = message
        return result

    def _execute_admitted(self, goal_handle):
        request = goal_handle.request
        result = ExecuteNamedMotion.Result()
        name = str(request.name)

        failure = self._admission_failure()
        if failure is not None:
            return self._fail(goal_handle, result, *failure)
        # A generation identifies this goal, but ownership of "the running
        # motion" is only taken over in _take_over, once the platform has
        # accepted the request: a replacement that fails validation or is
        # refused by the platform leaves the earlier goal's tracking intact,
        # because the earlier motion is still the one executing.
        with self._lock:
            self._generation += 1
            generation = self._generation
        epoch = self._node.runtime_state.stop_epoch
        try:
            feedback = ExecuteNamedMotion.Feedback()
            feedback.state = "accepted"
            feedback.progress = 0.0
            goal_handle.publish_feedback(feedback)

            # Re-check right before the vendor call: admission can change
            # between goal acceptance and this thread getting scheduled.
            failure = self._admission_failure()
            if failure is not None:
                return self._fail(goal_handle, result, *failure)

            mode = str(self._node.current_mode())
            if mode in self._trajectory_modes:
                outcome = self._run_trajectory(
                    name, mode, str(request.target), str(request.resume_token), goal_handle, generation, epoch
                )
            elif name in self._postures:
                outcome = self._run_posture(name, goal_handle, generation, epoch)
            else:
                outcome = self._run_preset(
                    name, str(request.target), bool(request.interrupt), goal_handle, generation, epoch
                )
        except CommandRejected as rejected:
            code = {
                "TORQUE_OFF_UNSAFE_POSTURE": ExecuteNamedMotion.Result.UNSAFE_POSTURE,
                "UNKNOWN_MOTION": ExecuteNamedMotion.Result.UNKNOWN_MOTION,
                "INVALID_TARGET": ExecuteNamedMotion.Result.INVALID_TARGET,
            }.get(rejected.reason, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM)
            return self._fail(goal_handle, result, code, str(rejected))
        except _Superseded as superseded:
            return self._fail(goal_handle, result, superseded.code, superseded.message)
        except _Cancelled as cancelled:
            goal_handle.canceled()
            result.success = False
            result.error_code = ExecuteNamedMotion.Result.CANCELLED
            result.message = cancelled.message
            result.resume_token = cancelled.token
            result.progress = cancelled.progress
            return result
        finally:
            with self._lock:
                if self._running == (name, generation):
                    self._running = None

        ok, code, message = outcome
        if not ok:
            return self._fail(goal_handle, result, code, message)
        feedback.state = "completing"
        feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result.success = True
        result.error_code = ExecuteNamedMotion.Result.NONE
        result.message = message
        result.progress = 1.0
        return result

    def _fail(self, goal_handle, result, code: int, message: str):
        goal_handle.abort()
        result.success = False
        result.error_code = code
        result.message = message
        return result

    def _take_over(self, name: str, generation: int) -> None:
        """Transfer ownership to this goal; the platform has accepted its request."""
        with self._lock:
            if self._returning and self._running != (name, generation):
                raise _Superseded(ExecuteNamedMotion.Result.BUSY, "cancel return owns the arm channel")
            self._running = (name, generation)

    def _still_current(self, generation: int, epoch: int) -> None:
        """Raise if a stop engaged or a later goal took over while waiting."""
        if self._node.runtime_state.stop_epoch != epoch:
            raise _Superseded(ExecuteNamedMotion.Result.STOP_LATCHED, "stop engaged while the motion was running")
        with self._lock:
            running = self._running
        if running is None or running[1] != generation:
            later = running[0] if running else "a later goal"
            raise _Superseded(ExecuteNamedMotion.Result.CANCELLED, f"interrupted by {later!r}")

    # --- vendor mechanisms --------------------------------------------------

    def _run_preset(
        self, name: str, target: str, interrupt: bool, goal_handle, generation: int, epoch: int
    ) -> tuple[bool, int, str]:
        entry = projection.resolve_named(name, self._presets, reason="UNKNOWN_MOTION")
        area_name = target or str(entry.get("target", ""))
        if area_name not in CONTROL_AREAS and not str(area_name).isdigit():
            raise CommandRejected("INVALID_TARGET", f"unknown target {area_name!r}")
        area = int(area_name) if str(area_name).isdigit() else CONTROL_AREAS[area_name]
        area = int(entry.get("area", area))
        # The platform executes preset motions only from stable stand. Checking
        # its own reported action turns "accepted, nothing happened" into a
        # refusal that names the reason.
        if self._required_action:
            current = str(self._node.platform_action())
            if current != self._required_action:
                return (
                    False,
                    ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                    f"platform is in {current or 'an unreported action'}; preset motions require "
                    f"{self._required_action}",
                )
        source = self._node.vendor_config.get("input_source") or {}
        # Re-assert the claim immediately before dispatching, as the vendor's
        # own preset client does (ADD, falling back to ENABLE). A registration
        # made at startup can have lapsed — a stop releases it — and the
        # platform discards requests from a source that is not enabled.
        if not self._node.ensure_input_source():
            return (
                False,
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                "vendor input source could not be registered; the platform would discard this motion",
            )
        request = vg.preset_motion_request(
            self._node,
            motion_value=int(entry["motion"]),
            area_value=area,
            interrupt=interrupt,
            source=self._node.input_source,
            priority=int(source.get("priority", 0)),
            timeout_ms=int(source.get("timeout_ms", 0)),
        )
        outcome = self._node.preset_motion_caller.call(request)
        task_id, state = vg.task_outcome(outcome.response) if outcome.response is not None else (0, 0)
        if not outcome.ok:
            if state == projection.TASK_STATE_RUNNING:
                # First come, first served: another motion is playing and
                # interrupt was false (preset_motion.html).
                return False, ExecuteNamedMotion.Result.BUSY, "platform is already playing a motion"
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                outcome.message or f"platform rejected preset motion {name} (code {outcome.code})",
            )
        if task_id <= 0:
            # T6 firmware accepts the preset motion and plays it but answers
            # task_id 0, so GetMcPresetMotionState has no handle to poll. Polling
            # a task that was never created would answer SUCCESS immediately, so
            # the playback is observed on the platform's own player state instead
            # (see _await_preset_without_task).
            return self._await_preset_without_task(name, goal_handle, generation, epoch)
        self._take_over(name, generation)
        return self._await_preset(name, task_id, goal_handle, generation, epoch)

    def _await_preset_without_task(self, name: str, goal_handle, generation: int, epoch: int) -> tuple[bool, int, str]:
        """Hold the goal while the platform plays a motion it answered without a task.

        T6 firmware accepts the preset motion and plays it but returns task_id 0,
        so GetMcPresetMotionState has nothing to report. The platform publishes
        its own player state on mc_state: wait for it to go busy and then idle
        again, and refuse to claim a motion the platform never reported playing.
        """
        self._take_over(name, generation)
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = "running"
        feedback.progress = 0.5
        goal_handle.publish_feedback(feedback)
        started = time.monotonic()
        deadline = started + self._preset_timeout_s
        seen_busy = False
        idle_since: float | None = None
        while time.monotonic() < deadline:
            self._still_current(generation, epoch)
            if self._node.player_state() in projection.PLAYER_STATES_BUSY:
                seen_busy = True
                idle_since = None
            elif seen_busy:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= 0.5:
                    return (
                        True,
                        ExecuteNamedMotion.Result.NONE,
                        f"preset motion {name} completed (player state; platform returned no task)",
                    )
            elif time.monotonic() - started >= self._start_grace_s:
                return (
                    False,
                    ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                    f"platform accepted preset motion {name} but never reported playing it",
                )
            time.sleep(self._poll_s)
        return (
            False,
            ExecuteNamedMotion.Result.TIMEOUT,
            f"preset motion {name} (no platform task) not reported complete within {self._preset_timeout_s:.0f}s",
        )

    def _await_preset(self, name: str, task_id: int, goal_handle, generation: int, epoch: int) -> tuple[bool, int, str]:
        """Hold the goal until the platform reports the motion's terminal state."""
        if not self._node.preset_state_caller.ready(1.0):
            return (
                False,
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                f"preset motion {name} dispatched (task {task_id}) but GetMcPresetMotionState is unavailable, "
                "so completion cannot be confirmed",
            )
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = "running"
        feedback.progress = 0.5
        goal_handle.publish_feedback(feedback)
        deadline = time.monotonic() + self._preset_timeout_s
        while time.monotonic() < deadline:
            self._still_current(generation, epoch)
            answer = self._node.preset_state_caller.call(vg.preset_motion_state_request(self._node, task_id=task_id))
            if answer.response is not None:
                _task, state = vg.task_outcome(answer.response)
                if state == projection.TASK_STATE_SUCCESS:
                    return True, ExecuteNamedMotion.Result.NONE, f"preset motion {name} completed (task {task_id})"
                if state not in projection.TASK_STATES_IN_PROGRESS:
                    return (
                        False,
                        ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                        f"preset motion {name} ended in {projection.TASK_STATE_NAMES.get(state, state)} "
                        f"(task {task_id})",
                    )
            time.sleep(self._poll_s)
        return (
            False,
            ExecuteNamedMotion.Result.TIMEOUT,
            f"preset motion {name} (task {task_id}) not reported complete within {self._preset_timeout_s:.0f}s",
        )

    def _cancel_replay(self, name: str, mode: str, recorded, offset_s: float, epoch: int) -> _Cancelled:
        """Record where a cancelled replay stopped and return the exception that ends its goal."""
        token = uuid.uuid4().hex
        with self._lock:
            self._checkpoint = _Checkpoint(
                token=token,
                name=name,
                mode=mode,
                offset_s=offset_s,
                stop_epoch=epoch,
                arm_epoch=self._node.arm_motion_epoch(),
                stamp=time.monotonic(),
            )
        progress = min(offset_s / recorded.duration, 1.0)
        return _Cancelled(
            f"{name} cancelled at {offset_s:.1f}s of {recorded.duration:.1f}s; arm held where it is",
            token,
            progress,
        )

    def _resume_offset(self, token: str, name: str, mode: str, epoch: int) -> tuple[float, str]:
        """Where a replay starts in its recording, and a note for the result message.

        The checkpoint is taken whatever the outcome: a token resumes once.
        Before the arm channel is claimed, so this goal's own claim does not
        count as the arm having moved.
        """
        if not token:
            return 0.0, ""
        with self._lock:
            checkpoint, self._checkpoint = self._checkpoint, None
        reason = ""
        if checkpoint is None or checkpoint.token != token:
            reason = "unknown or already used"
        elif checkpoint.name != name:
            reason = f"it was taken for {checkpoint.name!r}"
        elif checkpoint.mode != mode or checkpoint.stop_epoch != epoch:
            reason = "the runtime changed mode or stopped since"
        elif checkpoint.arm_epoch != self._node.arm_motion_epoch():
            reason = "the arm was commanded since"
        elif time.monotonic() - checkpoint.stamp > self._resume_ttl_s:
            reason = f"older than {self._resume_ttl_s:.0f}s"
        if reason:
            return 0.0, f"; resume checkpoint not usable ({reason}), played from the start"
        return checkpoint.offset_s, f"; resumed from {checkpoint.offset_s:.1f}s"

    def _return_cancelled(self, name, mode, start, history, goal_handle, generation, epoch, progress):
        """Reverse this execution's sent path, including its ease-in, with ownership held."""
        with self._lock:
            if self._running != (name, generation):
                raise _Superseded(ExecuteNamedMotion.Result.BUSY, "motion superseded before cancel return")
            self._returning = True
            self._checkpoint = None
        period = 1.0 / self._rate_hz
        speed = min(self._blend_speed, self._max_speed)
        required = str((self._node.vendor_config.get("mc_actions") or {}).get(mode, ""))

        def check():
            self._still_current(generation, epoch)
            if self._node.current_mode() != mode or (required and self._node.platform_action() != required):
                raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "mode changed during cancel return")
            if not self._node.output_admitted():
                raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "output disabled during cancel return")
            sample = self._node.measured_arm_sample()
            if len(start) == 34:
                hand = self._node.measured_hand_sample()
                if sample is None or hand is None:
                    sample = None
                else:
                    sample = (sample[0] + hand[0], min(sample[1], hand[1]))
            if (
                sample is None
                or len(sample[0]) != len(start)
                or not math.isfinite(sample[1])
                or any(not math.isfinite(value) for value in sample[0])
            ):
                raise _Superseded(
                    ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "stale arm feedback during cancel return"
                )
            return sample

        check()
        # Begin at the last commanded point; reverse every point actually sent,
        # rather than sampling the unexecuted remainder of the recording.
        knots = [*reversed(history), list(start)]
        for point in knots:
            for joint, value in zip(projection.VENDOR_ARM_ORDER, point[:14], strict=True):
                low, high = self._arm_limits[joint]
                if not math.isfinite(value) or not low <= value <= high:
                    raise _Superseded(
                        ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, f"cancel return outside {joint} limits"
                    )
        durations = [
            max(period, max(abs(b - a) for a, b in zip(left, right, strict=True)) / speed)
            for left, right in zip(knots, knots[1:], strict=False)
        ]
        budget = sum(durations) + self._return_settle_s
        if budget > self._return_timeout_s:
            raise _Superseded(
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"cancel return needs {budget:.2f}s, exceeds {self._return_timeout_s:.2f}s budget",
            )
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state, feedback.progress = "returning", progress
        goal_handle.publish_feedback(feedback)
        deadline = time.monotonic() + self._return_timeout_s

        def send(point):
            check()
            if time.monotonic() >= deadline:
                raise _Superseded(ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, "cancel return timed out")
            accepted = (
                self._node.command_recorded(point[:14], point[14:], owner=generation, epoch=epoch)
                if len(point) == 34
                else self._node.command_arm(point, owner=generation, epoch=epoch)
            )
            if not accepted:
                self._still_current(generation, epoch)
                raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "cancel return command refused")

        for left, right, duration in zip(knots, knots[1:], durations, strict=False):
            steps = max(1, math.ceil(duration / period))
            began = time.monotonic()
            for tick in range(1, steps + 1):
                point = [a + (b - a) * tick / steps for a, b in zip(left, right, strict=True)]
                time.sleep(max(0.0, began + tick * duration / steps - time.monotonic()))
                send(point)
        final_at = time.monotonic()
        send(start)
        settle_deadline = min(deadline, final_at + self._return_settle_s)
        while time.monotonic() < settle_deadline:
            measured, sampled_at = check()
            if (
                sampled_at >= final_at
                and max(abs(a - b) for a, b in zip(measured, start, strict=True)) <= self._return_tolerance
            ):
                check()
                raise _Cancelled(
                    f"{name} cancelled; returned to measured pre-replay pose ({budget:.2f}s budget)", "", progress
                )
            send(start)
            time.sleep(period)
        raise _Superseded(
            ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
            "cancel return did not reach measured pre-replay pose before settle timeout",
        )

    def _run_trajectory(
        self,
        name: str,
        mode: str,
        target: str,
        resume_token: str,
        goal_handle,
        generation: int,
        epoch: int,
        *,
        recorded: traj.ArmTrajectory | None = None,
        origin: list[float] | None = None,
        on_progress=None,
        verify_completion=True,
    ) -> tuple[bool, int, str]:
        """Replay a trajectory on the arm channel, easing in from the measured pose.

        ``recorded`` defaults to the profile's recording for ``name``; a
        ``joint.trajectory`` goal passes its own validated plan instead.
        """
        planned = recorded is not None
        if not planned and name in getattr(self, "_hold_names", ()):
            return self._run_hold_pose(name, mode, target, resume_token, goal_handle, generation, epoch)
        recorded = recorded or self._trajectories.get(name)
        display = self._display_completion
        if not planned and display and name in display["motions"]:
            verify_completion = False
        if recorded is None:
            return (
                False,
                ExecuteNamedMotion.Result.MODE_NOT_ALLOWED,
                f"mode {mode!r} replays only recorded motions {sorted(self._trajectories)}; {name!r} has none",
            )
        if on_progress is None:
            named_feedback = ExecuteNamedMotion.Feedback()
            named_feedback.state = "running"

            def on_progress(progress: float, _point: list[float]) -> None:
                named_feedback.progress = progress
                goal_handle.publish_feedback(named_feedback)

        # A recording drives exactly the side it was recorded on (the loader
        # refuses one that does not say which), so a request for any other side
        # is refused rather than answered with that side's motion.
        if not planned and target and target != recorded.target:
            raise CommandRejected("INVALID_TARGET", f"{name!r} was recorded for {recorded.target!r}, not {target!r}")
        # Arm targets only move the robot in the mode's own platform action;
        # anywhere else they would be accepted and silently dropped.
        required = str((self._node.vendor_config.get("mc_actions") or {}).get(mode, ""))
        current = str(self._node.platform_action())
        if required and current != required:
            return (
                False,
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                f"platform is in {current or 'an unreported action'}; replaying {name} requires {required}",
            )
        if not self._node.ensure_input_source():
            return (
                False,
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                "vendor input source could not be registered; the platform would discard this motion",
            )
        start = self._node.measured_arm_positions()
        if start is None:
            return (
                False,
                ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                "no fresh arm measurement to start the motion from",
            )
        if not planned and name in self._return_motions:
            for point in [start, *recorded.positions]:
                for joint, value in zip(projection.VENDOR_ARM_ORDER, point, strict=True):
                    low, high = self._arm_limits[joint]
                    if not math.isfinite(value) or not low <= value <= high:
                        raise CommandRejected("INVALID_TARGET", f"{joint} outside limits for cancel-return motion")
        offset, resume_note = self._resume_offset(resume_token, name, mode, epoch)
        return_on_cancel = not planned and name in self._return_motions
        if goal_handle.is_cancel_requested and not return_on_cancel:
            raise self._cancel_replay(name, mode, recorded, offset, epoch)
        hand_start = None
        if recorded.hands:
            sample = self._node.measured_hand_sample()
            if sample is None:
                raise CommandRejected("HAND_HOLD_UNAVAILABLE", "no fresh hand measurement for recorded entry")
            hand_start = sample[0]
        first = recorded.sample(offset)
        # Where the command eases from: the measured pose unless the caller knows the platform's target.
        ease_from = start if origin is None else origin
        ease = traj.blend_duration(ease_from, first, speed_radps=self._blend_speed, minimum_s=self._blend_min_s)
        if recorded.hands:
            ease = max(
                ease,
                1.5,
                traj.blend_duration(
                    hand_start, recorded.sample_hands(offset), speed_radps=self._blend_speed, minimum_s=1.5
                ),
            )
        total = ease + recorded.duration - offset
        period = 1.0 / self._rate_hz
        self._take_over(name, generation)
        if not self._node.claim_arm(generation):
            return (
                False,
                ExecuteNamedMotion.Result.BUSY,
                f"another source is streaming on the arm channel; {name} does not preempt it",
            )
        commanded_peak = [0.0] * len(start)
        measured_peak = [0.0] * len(start)
        # (t, target) sent within the last hold lag, oldest first.
        sent: deque[tuple[float, list[float]]] = deque()
        history = [list(start) + (hand_start or [])]
        try:
            started = time.monotonic()
            tick = 0
            while True:
                t = tick * period
                self._still_current(generation, epoch)
                if self._node.current_mode() != mode:
                    raise _Superseded(ExecuteNamedMotion.Result.CANCELLED, f"runtime left {mode!r} during {name}")
                reached = offset + max(0.0, t - ease)
                if goal_handle.is_cancel_requested:
                    # Hold where the arm is. Not at the last target: the arm
                    # trails it and would carry on. Not by commanding the
                    # measured pose either: the platform holds a pose only with
                    # an error against gravity, so the arm would sag by it. The
                    # target sent one tracking lag ago is the one being realized.
                    if return_on_cancel:
                        joint = max(range(len(start)), key=commanded_peak.__getitem__)
                        if (
                            commanded_peak[joint] >= FOLLOW_CHECK_MIN_RAD
                            and measured_peak[joint] < self._follow_min_ratio * commanded_peak[joint]
                        ):
                            raise _Superseded(
                                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                                "platform did not follow motion before cancel return",
                            )
                        self._return_cancelled(
                            name,
                            mode,
                            start + (hand_start or []),
                            history,
                            goal_handle,
                            generation,
                            epoch,
                            min(reached / recorded.duration, 1.0),
                        )
                    self._node.command_arm(sent[0][1] if sent else start, owner=generation, epoch=epoch)
                    raise self._cancel_replay(name, mode, recorded, reached, epoch)
                point = traj.blend(ease_from, first, t / ease) if t < ease else recorded.sample(reached)
                sent.append((t, point))
                while len(sent) > 1 and sent[1][0] <= t - self._hold_lag_s:
                    sent.popleft()
                hands = (
                    (
                        traj.blend(hand_start, recorded.sample_hands(offset), t / ease)
                        if t < ease
                        else recorded.sample_hands(reached)
                    )
                    if recorded.hands
                    else []
                )
                accepted = (
                    self._node.command_recorded(point, hands, owner=generation, epoch=epoch)
                    if recorded.hands
                    else self._node.command_arm(point, owner=generation, epoch=epoch)
                )
                if not accepted:
                    self._still_current(generation, epoch)
                    return (
                        False,
                        ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                        f"runtime stopped admitting arm commands during {name}",
                    )
                if return_on_cancel:
                    types = self._node._dataplane.hand_types if hands else []
                    sent_hands = [
                        v if vg.hand_type_family(types[k // 10]) == "dexterous" else 0.0 for k, v in enumerate(hands)
                    ]
                    history.append(list(point) + sent_hands)
                measured = self._node.measured_arm_positions() or start
                for k, (origin, target, actual) in enumerate(zip(start, point, measured, strict=True)):
                    commanded_peak[k] = max(commanded_peak[k], abs(target - origin))
                    measured_peak[k] = max(measured_peak[k], abs(actual - origin))
                if t >= total:
                    break
                if tick % int(self._rate_hz) == 0:
                    on_progress(min(reached / recorded.duration, 1.0), point)
                tick += 1
                time.sleep(max(0.0, started + tick * period - time.monotonic()))
            self._still_current(generation, epoch)
            if goal_handle.is_cancel_requested:
                if return_on_cancel:
                    self._return_cancelled(
                        name, mode, start + (hand_start or []), history, goal_handle, generation, epoch, 1.0
                    )
                raise self._cancel_replay(name, mode, recorded, recorded.duration, epoch)
            if not self._node.output_admitted() or self._node.current_mode() != mode:
                raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "runtime unavailable at motion end")
            if recorded.hands and verify_completion:
                # Success requires new feedback acquired after the final target was sent.
                final_at = time.monotonic()
                deadline = final_at + self._return_settle_s
                arm_tolerance = hand_tolerance = self._return_tolerance
                while True:
                    self._still_current(generation, epoch)
                    if not self._node.output_admitted() or self._node.current_mode() != mode:
                        raise _Superseded(
                            ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, "runtime unavailable at motion end"
                        )
                    arm_sample, hand_sample = self._node.measured_arm_sample(), self._node.measured_hand_sample()
                    types = self._node._dataplane.hand_types
                    desired_hands = [
                        v if vg.hand_type_family(types[k // 10]) == "dexterous" else 0.0 for k, v in enumerate(hands)
                    ]
                    evidence = final_pose_evidence(
                        arm_sample, hand_sample, point, desired_hands, final_at, time.monotonic()
                    )
                    evidence.update(
                        motion=name,
                        policy="strict",
                        arm_tolerance_rad=arm_tolerance,
                        hand_tolerance_rad=hand_tolerance,
                    )
                    if arm_sample and any(
                        not self._arm_limits[joint][0] <= value <= self._arm_limits[joint][1]
                        for joint, value in zip(projection.VENDOR_ARM_ORDER, arm_sample[0], strict=True)
                    ):
                        raise CommandRejected("FINAL_POSE_UNCONFIRMED", "measured arm outside joint limits")
                    if final_pose_reached(evidence, arm_tolerance, hand_tolerance):
                        self._node.get_logger().info(f"recorded completion: {json.dumps(evidence, sort_keys=True)}")
                        break
                    if time.monotonic() >= deadline:
                        self._node.get_logger().error(
                            f"recorded completion failed: {json.dumps(evidence, sort_keys=True)}"
                        )
                        raise CommandRejected("FINAL_POSE_UNCONFIRMED", json.dumps(evidence, sort_keys=True))
                    if goal_handle.is_cancel_requested:
                        self._return_cancelled(
                            name, mode, start + hand_start, history, goal_handle, generation, epoch, 1.0
                        )
                    if not self._node.command_recorded(point, hands, owner=generation, epoch=epoch):
                        raise CommandRejected("FINAL_POSE_UNCONFIRMED", "final target refused")
                    time.sleep(period)
        except _Cancelled:
            raise
        except Exception as exc:
            if (return_on_cancel and self._returning) or (
                recorded.hands and isinstance(exc, CommandRejected) and exc.reason == "FINAL_POSE_UNCONFIRMED"
            ):
                # Unconfirmed return must not admit fresh work after releasing ownership.
                phase = "cancel return" if self._returning else "recorded completion"
                self._node._set_fault(
                    "cancel_return" if self._returning else "recorded_completion", f"{phase} unconfirmed: {exc}"
                )
                self._node._push_control()
                if not isinstance(exc, _Superseded):
                    raise _Superseded(ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE, f"{phase} failed: {exc}") from exc
            raise
        finally:
            self._node.release_arm(generation)
            with self._lock:
                if self._running == (name, generation):
                    self._returning = False
        if not verify_completion:
            message = f"display playback finished: {name}; final pose not verified ({total:.1f}s){resume_note}"
            self._node.get_logger().info(message)
            return True, ExecuteNamedMotion.Result.NONE, message
        joint = max(range(len(start)), key=commanded_peak.__getitem__)
        if (
            commanded_peak[joint] >= FOLLOW_CHECK_MIN_RAD
            and measured_peak[joint] < self._follow_min_ratio * commanded_peak[joint]
        ):
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"platform did not follow {name}: {projection.VENDOR_ARM_ORDER[joint]} was commanded "
                f"{commanded_peak[joint]:.2f} rad and moved {measured_peak[joint]:.2f} rad",
            )
        return True, ExecuteNamedMotion.Result.NONE, f"replayed {name} in {mode} ({total:.1f}s){resume_note}"

    def _run_posture(self, name: str, goal_handle, generation: int, epoch: int) -> tuple[bool, int, str]:
        action = projection.resolve_named(name, self._postures, reason="UNKNOWN_MOTION")
        if action not in projection.VENDOR_ACTIONS:
            raise CommandRejected("UNKNOWN_MOTION", f"posture {name!r} maps to unknown platform action {action!r}")
        outcome = self._node.action_caller.call(
            vg.set_action_request(self._node, source=self._node.input_source, action_name=action)
        )
        if not outcome.ok:
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"platform rejected {action}: {projection.mode_rejection_reason(outcome.code)}",
            )
        self._take_over(name, generation)
        # Posture changes are asynchronous; report when the platform's own body
        # pose reflects the request, or say plainly that it did not.
        expected = (self._node.vendor_config.get("posture_poses") or {}).get(name)
        if expected is None:
            return True, ExecuteNamedMotion.Result.NONE, f"posture action {action} accepted"
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = "running"
        feedback.progress = 0.5
        goal_handle.publish_feedback(feedback)
        deadline = time.monotonic() + float(self._node.vendor_config.get("posture_timeout_s", 20.0))
        while time.monotonic() < deadline:
            self._still_current(generation, epoch)
            if projection.BODY_POSE_NAMES.get(self._node.current_body_pose()) == str(expected):
                return True, ExecuteNamedMotion.Result.NONE, f"posture {name} reached"
            time.sleep(0.1)
        return (
            False,
            ExecuteNamedMotion.Result.TIMEOUT,
            f"posture {name} not confirmed by the platform within the timeout",
        )
