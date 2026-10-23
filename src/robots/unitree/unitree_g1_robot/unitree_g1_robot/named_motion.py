"""Named motions and postures for the Unitree G1, served as one neutral action.

Two vendor mechanisms sit behind the profile's ``motion.named`` endpoint:

* preset arm actions — ``ExecuteAction`` on the ``arm`` RPC service, with the
  action ids the profile's ``vendor.arm_actions`` table carries. The table is
  compared against ``GetActionList`` at startup; a motion the vendor does not
  report is neither advertised nor accepted (design D7).
* whole-body postures — ``SetFsmId`` on the ``loco`` service, with the FSM ids
  the profile's ``vendor.postures`` table carries.

Completion is the platform's, not the dispatch acknowledgement's: a posture
goal stays open until ``GetFsmId`` answers with the posture's FSM id. For the
arm actions the vendor's own RPC answer is the completion signal available to
this change — the action-state topic (``rt/arm/action/state``) is not
published by the in-repo mock, so wiring it is left until there is a source
to observe. unverified: whether that topic reports progress the bridge must
wait for instead is a hardware question (gate 6.3); until it is answered, the
ExecuteAction response is the completion signal.

The platform has no way to cancel either mechanism: a cancel request is
refused instead of accepted and ignored, and the caller is pointed at
``/runtime/stop``, which does interrupt everything through the stop sequence.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rclpy.action import ActionServer, CancelResponse, GoalResponse

from ibrobot_msgs.action import ExecuteNamedMotion
from robot_runtime.contract import LIFECYCLE_ACTIVE
from unitree_g1_robot import projection

#: Contract-side action endpoint; the profile's interfaces declare the same one.
NAMED_MOTION_ACTION = "/motion/execute_named"


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
        self._actions = {
            str(name): int(entry["action_id"]) for name, entry in (vendor.get("arm_actions") or {}).items()
        }
        self._postures = {str(name): int(entry["fsm_id"]) for name, entry in (vendor.get("postures") or {}).items()}
        config = vendor.get("named_motion") or {}
        # Preset arm actions fight the streamed frame for the arms, and a
        # posture change while walking is a fall: both are idle-mode motions.
        self._allowed_modes = [str(mode) for mode in config.get("allowed_modes", ["idle"])]
        self._posture_timeout_s = float(config.get("posture_timeout_s", 5.0))
        self._poll_s = float(config.get("poll_s", 0.05))
        self._lock = threading.Lock()
        # (name, generation) of the goal currently executing; a later goal with
        # interrupt=true replaces it and the earlier wait loop notices.
        self._running: tuple[str, int] | None = None
        self._generation = 0
        interface = (node.profile.get("interfaces") or {}).get("motion.named") or {}
        self._server = ActionServer(
            node,
            ExecuteNamedMotion,
            str(interface.get("endpoint", NAMED_MOTION_ACTION)),
            execute_callback=self._execute,
            goal_callback=self._accept,
            cancel_callback=self._refuse_cancel,
            callback_group=node.action_callback_group,
        )

    # --- goal handling ------------------------------------------------------

    def _accept(self, goal) -> GoalResponse:
        name = str(goal.name)
        if name not in self._node.served_motion_names():
            return GoalResponse.REJECT
        with self._lock:
            if self._running is not None and not bool(goal.interrupt):
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _refuse_cancel(self, _goal_handle) -> CancelResponse:
        """The platform offers no way to cancel either mechanism.

        SetFsmId is not interruptible (stopping half-way through sitting down
        is a fall), and the arm action service has no stop call. Accepting a
        cancel and then ignoring it would be a lie; the honest answer is to
        refuse it and point at ``/runtime/stop``, which does interrupt
        everything through the stop sequence.
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
        mode = str(self._node.current_mode())
        if mode not in self._allowed_modes:
            return (
                ExecuteNamedMotion.Result.MODE_NOT_ALLOWED,
                f"mode {mode!r} does not admit named motions; allowed: {self._allowed_modes}",
            )
        return None

    def _execute(self, goal_handle):
        request = goal_handle.request
        result = ExecuteNamedMotion.Result()
        name = str(request.name)

        failure = self._admission_failure()
        if failure is not None:
            return self._fail(goal_handle, result, *failure)
        with self._lock:
            self._generation += 1
            generation = self._generation
        epoch = self._node.runtime_state.stop_epoch
        try:
            feedback = ExecuteNamedMotion.Feedback()
            feedback.state = "accepted"
            feedback.progress = 0.0
            goal_handle.publish_feedback(feedback)

            # Re-check right before dispatch: admission can change between
            # goal acceptance and this thread getting scheduled.
            failure = self._admission_failure()
            if failure is not None:
                return self._fail(goal_handle, result, *failure)

            if name in self._postures:
                ok, code, message = self._run_posture(name, goal_handle, generation, epoch)
            else:
                ok, code, message = self._run_action(name, goal_handle, generation, epoch)
        except _Superseded as superseded:
            return self._fail(goal_handle, result, superseded.code, superseded.message)
        finally:
            with self._lock:
                if self._running == (name, generation):
                    self._running = None

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
        """Transfer ownership to this goal; the vendor has accepted its request."""
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

    def _run_action(self, name: str, goal_handle, generation: int, epoch: int) -> tuple[bool, int, str]:
        """Dispatch one preset arm action through the arm RPC service."""
        action_id = self._actions[name]
        result = self._node.rpc_call(
            "arm",
            projection.ARM_ACTION_API_IDS["execute_action"],
            projection.action_id_parameter(action_id),
            motion=True,
        )
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return (
                False,
                ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM,
                f"ExecuteAction for {name!r} (action_id {action_id}) {detail}",
            )
        self._take_over(name, generation)
        if self._node.runtime_state.stop_epoch != epoch:
            # The action was dispatched, but a stop engaged in the same
            # window; the goal ends with the stop's code, not a success the
            # stop has already superseded.
            raise _Superseded(ExecuteNamedMotion.Result.STOP_LATCHED, "stop engaged while the motion was running")
        # The vendor answers ExecuteAction synchronously; the action-state
        # topic that would report progress is not published by this change's
        # mock, so the RPC answer is the completion signal available here.
        return True, ExecuteNamedMotion.Result.NONE, f"arm action {name!r} executed (action_id {action_id})"

    def _run_posture(self, name: str, goal_handle, generation: int, epoch: int) -> tuple[bool, int, str]:
        """Switch the whole-body FSM and wait for the platform's own report."""
        fsm_id = self._postures[name]
        result = self._node.rpc_call(
            "loco",
            projection.LOCO_API_IDS["set_fsm_id"],
            projection.fsm_id_parameter(fsm_id),
            motion=True,
        )
        if not result.ok:
            detail = "timed out; not re-sent" if result.timed_out else f"failed with vendor code {result.code}"
            return False, ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, f"SetFsmId({fsm_id}) for {name!r} {detail}"
        self._take_over(name, generation)
        feedback = ExecuteNamedMotion.Feedback()
        feedback.state = "running"
        feedback.progress = 0.5
        goal_handle.publish_feedback(feedback)
        deadline = time.monotonic() + self._posture_timeout_s
        while time.monotonic() < deadline:
            self._still_current(generation, epoch)
            answer = self._node.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {}, timeout_s=0.5)
            if answer.ok and isinstance(answer.data, dict) and answer.data.get("data") == fsm_id:
                return True, ExecuteNamedMotion.Result.NONE, f"posture {name!r} reached (FSM {fsm_id})"
            time.sleep(self._poll_s)
        return (
            False,
            ExecuteNamedMotion.Result.TIMEOUT,
            f"posture {name!r} not confirmed by the vendor within {self._posture_timeout_s:.0f}s",
        )
