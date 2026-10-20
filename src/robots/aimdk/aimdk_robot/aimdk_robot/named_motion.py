"""Named motions and postures exposed as one neutral action server.

Three mechanisms sit behind ``/motion/execute_named``:

* preset motions (``SetMcPresetMotion``) — wave, handshake, clap, …
* posture MC actions (``SetMcAction``) — sit, squat, lie down, stand up, stairs
* recorded arm trajectories — the preset gestures replayed on the upper-body
  stream, for the modes in which the platform will not play a preset

Which one runs follows the runtime mode: a trajectory mode replays, every other
admitted mode uses the vendor mechanisms.

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
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

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


class _Superseded(Exception):
    """Raised inside a wait loop when a later goal or a stop took over."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
        self._blend_speed = float(config.get("trajectory_blend_speed_radps", 0.5))
        self._blend_min_s = float(config.get("trajectory_blend_min_s", 0.5))
        # A replay only succeeds if the measured arm followed it: the platform
        # drops upper-body commands it cannot use without saying so.
        self._follow_min_ratio = float(config.get("trajectory_follow_min_ratio", 0.3))
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
        self._server = ActionServer(
            node,
            ExecuteNamedMotion,
            NAMED_MOTION_ACTION,
            execute_callback=self._execute,
            goal_callback=self._accept,
            cancel_callback=self._refuse_cancel,
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

    def _refuse_cancel(self, _goal_handle) -> CancelResponse:
        """Named motions are not cancellable; ``/runtime/stop`` interrupts them.

        A posture transition is not interruptible (stopping half-way through
        sitting down is a fall), and the SDK exposes no "stop preset motion"
        call. Accepting a cancel and then ignoring it would be a lie, so it is
        refused and ``/runtime/stop`` named instead, which does interrupt
        everything. A replayed trajectory could stop between ticks, but it is
        the same named motion to the caller and answers cancel the same way.
        """
        self._node.get_logger().warning(
            "named motion cancel refused: the platform cannot cancel a running motion; use /runtime/stop"
        )
        return CancelResponse.REJECT

    def _admission_failure(self) -> tuple[int, str] | None:
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
                outcome = self._run_trajectory(name, mode, str(request.target), goal_handle, generation, epoch)
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

    def _run_trajectory(
        self, name: str, mode: str, target: str, goal_handle, generation: int, epoch: int
    ) -> tuple[bool, int, str]:
        """Replay a recorded trajectory on the arm channel, easing in from the measured pose."""
        recorded = self._trajectories.get(name)
        if recorded is None:
            return (
                False,
                ExecuteNamedMotion.Result.MODE_NOT_ALLOWED,
                f"mode {mode!r} replays only recorded motions {sorted(self._trajectories)}; {name!r} has none",
            )
        # A recording drives exactly the side it was recorded on (the loader
        # refuses one that does not say which), so a request for any other side
        # is refused rather than answered with that side's motion.
        if target and target != recorded.target:
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
        first = list(recorded.positions[0])
        ease = traj.blend_duration(start, first, speed_radps=self._blend_speed, minimum_s=self._blend_min_s)
        total = ease + recorded.duration
        period = 1.0 / self._rate_hz
        self._take_over(name, generation)
        if not self._node.claim_arm(generation):
            return (
                False,
                ExecuteNamedMotion.Result.BUSY,
                f"another source is streaming on the arm channel; {name} does not preempt it",
            )
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = "running"
        commanded_peak = [0.0] * len(start)
        measured_peak = [0.0] * len(start)
        try:
            started = time.monotonic()
            tick = 0
            while True:
                t = tick * period
                self._still_current(generation, epoch)
                if self._node.current_mode() != mode:
                    raise _Superseded(ExecuteNamedMotion.Result.CANCELLED, f"runtime left {mode!r} during {name}")
                point = traj.blend(start, first, t / ease) if t < ease else recorded.sample(t - ease)
                if not self._node.command_arm(point, owner=generation, epoch=epoch):
                    self._still_current(generation, epoch)
                    return (
                        False,
                        ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE,
                        f"runtime stopped admitting arm commands during {name}",
                    )
                measured = self._node.measured_arm_positions() or start
                for k, (origin, target, actual) in enumerate(zip(start, point, measured, strict=True)):
                    commanded_peak[k] = max(commanded_peak[k], abs(target - origin))
                    measured_peak[k] = max(measured_peak[k], abs(actual - origin))
                if t >= total:
                    break
                if tick % int(self._rate_hz) == 0:
                    feedback.progress = min(t / total, 1.0)
                    goal_handle.publish_feedback(feedback)
                tick += 1
                time.sleep(max(0.0, started + tick * period - time.monotonic()))
        finally:
            self._node.release_arm(generation)
        joint = max(range(len(start)), key=commanded_peak.__getitem__)
        if measured_peak[joint] < self._follow_min_ratio * commanded_peak[joint]:
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"platform did not follow {name}: {projection.VENDOR_ARM_ORDER[joint]} was commanded "
                f"{commanded_peak[joint]:.2f} rad and moved {measured_peak[joint]:.2f} rad",
            )
        return True, ExecuteNamedMotion.Result.NONE, f"replayed {name} in {mode} ({total:.1f}s)"

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
