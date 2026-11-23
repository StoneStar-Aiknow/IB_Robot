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

import contextlib
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from aimdk_robot import projection
from aimdk_robot import trajectory as traj
from aimdk_robot import vendor_gateway as vg
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
#: How long a replay keeps sending its final target for the data plane to echo it (s).
FINAL_TARGET_CONFIRM_S = 0.25
#: How close the echo must be to the final target to confirm it (rad); the
#: platform may carry targets in single precision.
FINAL_TARGET_TOLERANCE_RAD = 1e-4


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


def _unsupported_trajectory_request(request) -> str:
    """Why a FollowJointTrajectory goal asks for what this runtime does not do, or "".

    Refused rather than ignored: a scheduled start, a path tolerance or a
    velocity/acceleration goal tolerance would each be a promise the replay
    does not keep. Position goal tolerances and goal_time_tolerance are kept.
    """
    stamp = request.trajectory.header.stamp
    if stamp.sec or stamp.nanosec:
        return "trajectory.header.stamp must be zero: a scheduled start is not supported"
    if any(t.position > 0.0 or t.velocity > 0.0 or t.acceleration > 0.0 for t in request.path_tolerance):
        return "path_tolerance is not supported: the arm trails its command by the platform's tracking lag"
    if any(t.velocity > 0.0 or t.acceleration > 0.0 for t in request.goal_tolerance):
        return "goal_tolerance supports positions only, not velocities or accelerations"
    return ""


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
        # joint.trajectory success: when the goal sets no tolerance, each goal joint must end
        # within the larger of these of its final target (trajectory.goal_tolerances), within
        # trajectory_goal_settle_s unless the goal's goal_time_tolerance says otherwise. Coarse
        # on purpose: it catches a joint that stopped short, not a pose held against gravity.
        self._goal_tolerance_rad = float(config.get("trajectory_goal_tolerance_rad", 0.2))
        self._goal_tolerance_ratio = float(config.get("trajectory_goal_tolerance_ratio", 0.5))
        self._goal_settle_s = float(config.get("trajectory_goal_settle_s", 2.0))
        # ... and each goal joint commanded at least this far must itself have followed
        # (trajectory_follow_min_ratio of its travel): a stuck-joint check, not a precision one.
        self._follow_joint_min_rad = float(config.get("trajectory_follow_joint_min_rad", 0.1))
        self._rate_hz = float((vendor.get("command") or {}).get("rate_hz", 50.0))
        profile_dir = Path(str(node.get_parameter("profile").value)).parent
        max_speed = float(config.get("trajectory_max_speed_radps", 3.0))
        self._trajectories = {
            str(name): traj.load_trajectory(profile_dir / str(path), max_speed_radps=max_speed)
            for name, path in (config.get("trajectories") or {}).items()
        }
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

    # --- goal handling ------------------------------------------------------

    def _accept(self, goal) -> GoalResponse:
        name = str(goal.name)
        if name not in self._presets and name not in self._postures and name not in self._trajectories:
            return GoalResponse.REJECT
        # Early answer only: two goals can both pass here before either has
        # started executing. _execute makes the check that holds.
        with self._lock:
            if self._executing and not bool(goal.interrupt):
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
        with self._lock:
            busy = self._executing > 0 and not bool(request.interrupt)
            if not busy:
                self._executing += 1
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
        request = goal_handle.request
        refusal = _unsupported_trajectory_request(request)
        if refusal:
            return self._fail_trajectory(goal_handle, result, refusal)
        start = self._node.measured_arm_positions()
        if start is None:
            return self._fail_trajectory(goal_handle, result, "no fresh arm measurement to start the motion from")
        # Start from what the platform is driving: see arm_takeover_start.
        origin = self._node.arm_takeover_start(start)
        goal = request.trajectory
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
                measured=start,
            )
            final = plan.positions[-1]
            tolerances = traj.goal_tolerances(
                list(goal.joint_names),
                {str(t.name): float(t.position) for t in request.goal_tolerance},
                origin,
                final,
                default_rad=self._goal_tolerance_rad,
                default_ratio=self._goal_tolerance_ratio,
            )
        except traj.TrajectoryError as exc:
            return self._fail_trajectory(goal_handle, result, str(exc))
        settle = request.goal_time_tolerance.sec + request.goal_time_tolerance.nanosec * 1e-9
        settle_s = settle if settle > 0.0 else self._goal_settle_s
        with self._lock:
            self._generation += 1
            generation = self._generation
        epoch = self._node.runtime_state.stop_epoch

        def on_progress(_progress: float, point: list[float]) -> None:
            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = list(projection.VENDOR_ARM_ORDER)
            feedback.desired.positions = [float(v) for v in point]
            goal_handle.publish_feedback(feedback)

        arrival = {"tolerances": tolerances, "settle_s": settle_s}
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
                arrival=arrival,
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
        errors = arrival["errors"]
        report = ", ".join(f"{joint} {error:.3f} (allowed {tolerances[joint]:.3f})" for joint, error in errors.items())
        message = f"{message}; final error: {report or 'not checked'}"
        if not arrival["arrived"]:
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = f"not at the final target within {settle_s:.1f}s; {message}"
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
        if target and "area" in entry and CONTROL_AREAS.get(target, target) not in (area, str(area)):
            raise CommandRejected("INVALID_TARGET", f"{name} requires area {area}")
        if "targets" in entry and area not in {CONTROL_AREAS[t] for t in entry["targets"]}:
            raise CommandRejected("INVALID_TARGET", f"{name} does not support area {area}")
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
            # Accepted, but no task exists to execute or observe. This must not
            # become a completion: GetMcPresetMotionState distinguishes only
            # "executing" from "completed", so polling a task that was never
            # created answers SUCCESS immediately and the goal would report a
            # motion that never happened.
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"platform accepted preset motion {name} but created no task (task_id 0); nothing was dispatched",
            )
        self._take_over(name, generation)
        return self._await_preset(name, task_id, goal_handle, generation, epoch)

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
        arrival: dict | None = None,
    ) -> tuple[bool, int, str]:
        """Replay a trajectory on the arm channel, easing in from the measured pose.

        ``recorded`` defaults to the profile's recording for ``name``; a
        ``joint.trajectory`` goal passes its own validated plan instead, and
        ``arrival`` (``tolerances`` and ``settle_s``): the arm is kept until
        those joints settle at the final target, and ``arrived`` and ``errors``
        are filled in.
        """
        planned = recorded is not None
        recorded = recorded or self._trajectories.get(name)
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
        offset, resume_note = self._resume_offset(resume_token, name, mode, epoch)
        if goal_handle.is_cancel_requested:
            raise self._cancel_replay(name, mode, recorded, offset, epoch)
        first = recorded.sample(offset)
        # Where the command eases from: the measured pose unless the caller knows the platform's target.
        ease_from = start if origin is None else origin
        if planned and max(abs(a - b) for a, b in zip(ease_from, first, strict=True)) == 0.0:
            # A planned trajectory starts where the command does: nothing to ease.
            ease = 0.0
        else:
            ease = traj.blend_duration(ease_from, first, speed_radps=self._blend_speed, minimum_s=self._blend_min_s)
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
                    self._hold_on_cancel(sent, ease_from, generation, epoch, period)
                    raise self._cancel_replay(name, mode, recorded, reached, epoch)
                point = traj.blend(ease_from, first, t / ease) if t < ease else recorded.sample(reached)
                sent.append((t, point))
                while len(sent) > 1 and sent[1][0] <= t - self._hold_lag_s:
                    sent.popleft()
                if not self._node.command_arm(point, owner=generation, epoch=epoch):
                    self._still_current(generation, epoch)
                    return (
                        False,
                        ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                        f"runtime stopped admitting arm commands during {name}",
                    )
                measured = self._node.measured_arm_positions() or start
                # Commanded travel from where the command started, measured
                # travel from where the arm was: under load the two differ by
                # the sag, which neither the command nor the arm travels.
                for k, (eased, measured_start, target, actual) in enumerate(
                    zip(ease_from, start, point, measured, strict=True)
                ):
                    commanded_peak[k] = max(commanded_peak[k], abs(target - eased))
                    measured_peak[k] = max(measured_peak[k], abs(actual - measured_start))
                if t >= total:
                    break
                if tick % int(self._rate_hz) == 0:
                    on_progress(min(reached / recorded.duration, 1.0), point)
                tick += 1
                time.sleep(max(0.0, started + tick * period - time.monotonic()))
            plan_ended = time.monotonic()
            confirmed = self._confirm_final_target(
                point,
                generation,
                epoch,
                period,
                cancelled=lambda: goal_handle.is_cancel_requested,
                deadline=None if arrival is None else plan_ended + arrival["settle_s"],
            )
            if confirmed is None:  # cancelled while the final target was being confirmed
                self._hold_on_cancel(sent, ease_from, generation, epoch, period)
                raise self._cancel_replay(name, mode, recorded, reached, epoch)
            if arrival is not None:
                # Settle with the arm still owned: released, the public arm
                # stream could override the final target, and a cancel could
                # not hold. The confirmation above spends the same budget.
                deadline = plan_ended + arrival["settle_s"]
                final = point
                while True:
                    measured = self._node.measured_arm_positions()
                    if measured is not None:
                        for k, (measured_start, actual) in enumerate(zip(start, measured, strict=True)):
                            measured_peak[k] = max(measured_peak[k], abs(actual - measured_start))
                        arrival["errors"] = {
                            joint: abs(measured[k] - final[k])
                            for joint in arrival["tolerances"]
                            for k in (projection.VENDOR_ARM_ORDER.index(joint),)
                        }
                    errors = arrival.setdefault("errors", {})
                    arrival["arrived"] = measured is not None and all(
                        errors[joint] <= tolerance for joint, tolerance in arrival["tolerances"].items()
                    )
                    if arrival["arrived"] or time.monotonic() >= deadline:
                        break
                    if goal_handle.is_cancel_requested:
                        self._hold_on_cancel(sent, ease_from, generation, epoch, period)
                        raise self._cancel_replay(name, mode, recorded, reached, epoch)
                    self._still_current(generation, epoch)
                    if self._node.current_mode() != mode:
                        raise _Superseded(ExecuteNamedMotion.Result.CANCELLED, f"runtime left {mode!r} during {name}")
                    self._node.command_arm(final, owner=generation, epoch=epoch)
                    time.sleep(period)
        finally:
            self._node.release_arm(generation)
        # The platform must have followed: the most-moving joint, and for a
        # planned goal also every checked joint commanded far enough to tell a
        # stuck joint from noise. Measured travel is from where the arm was, so
        # a pose held against gravity (a steady offset) does not count against it.
        joints = [max(range(len(start)), key=commanded_peak.__getitem__)]
        if arrival is not None:
            joints += [
                projection.VENDOR_ARM_ORDER.index(name_)
                for name_ in arrival["tolerances"]
                if commanded_peak[projection.VENDOR_ARM_ORDER.index(name_)] >= self._follow_joint_min_rad
            ]
        for joint in joints:
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
        # Without an echo there is nothing to confirm with; with one, say so when it never came.
        echoed = getattr(self._node, "has_arm_command_echo", False)
        unconfirmed = "" if confirmed or not echoed else "; the platform did not confirm the final target"
        return (
            True,
            ExecuteNamedMotion.Result.NONE,
            f"replayed {name} in {mode} ({total:.1f}s){resume_note}{unconfirmed}",
        )

    def _hold_on_cancel(self, sent, ease_from, generation: int, epoch: int, period: float) -> None:
        """Hold where the arm is, and make sure the hold arrives before the release.

        Not at the last target: the arm trails it and would carry on. Not by
        commanding the measured pose either: the platform holds a pose only
        with an error against gravity, so the arm would sag by it. The target
        sent one tracking lag ago is the one being realized; before the first
        tick, the target the command starts from. A stop or a later goal that
        owns the arm by now needs no hold.
        """
        hold = sent[0][1] if sent else ease_from
        self._node.command_arm(hold, owner=generation, epoch=epoch)
        with contextlib.suppress(_Superseded):
            self._confirm_final_target(hold, generation, epoch, period)

    def _confirm_final_target(
        self, point: list[float], generation: int, epoch: int, period: float, *, cancelled=None, deadline=None
    ) -> bool | None:
        """Keep sending the final target until the data plane echoes it, then let the arm go.

        The data plane takes an arm target only while this motion owns the arm,
        and the release travels on another topic: released first, the final
        target would be dropped and the platform would hold the one before it.
        Waits at most ``FINAL_TARGET_CONFIRM_S``; a profile without the echo
        gets the target once more. True when the echo confirmed it; None as
        soon as ``cancelled()`` says the goal was cancelled meanwhile. An
        earlier ``deadline`` (a goal's arrival budget) shortens the wait.
        """
        echoed = getattr(self._node, "has_arm_command_echo", False)
        limit = time.monotonic() + FINAL_TARGET_CONFIRM_S
        deadline = limit if deadline is None else min(limit, deadline)
        while True:
            if cancelled is not None and cancelled():
                return None
            echo = self._node.commanded_arm() if echoed else None
            if (
                echo is not None
                and max(abs(a - b) for a, b in zip(echo, point, strict=True)) <= FINAL_TARGET_TOLERANCE_RAD
            ):
                return True
            self._still_current(generation, epoch)
            if not self._node.command_arm(point, owner=generation, epoch=epoch):
                return False
            if not echoed or time.monotonic() >= deadline:
                return False
            time.sleep(period)

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
