"""Serve ``PickObject`` through the robot-agnostic grasp pipeline.

The robot's ``grasp_execution`` configuration names a backend factory
(``backend: "package.module:callable"``); the factory is called as
``factory(node=..., grasp_execution=..., description=...)`` and returns the
robot's ``GraspBackends``. This node owns only what every robot shares:
delegated-executor admission (``pick_goal``), one goal at a time, the goal's
budget, cancellation, feedback and the result. Robots without a backend keep
``pick_executor_node``.

``grasp_execution.pipeline`` may set ``max_attempts``, ``max_moved_replans``
and ``home_tolerance_rad`` (see ``grasp_pipeline.pipeline.PipelineConfig``).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Mapping

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

from embodied_common.dispatch_binding import (
    delegated_executor_identity,
    fill_delegated_executor_identity,
    load_delegated_model_identity,
)
from embodied_common.execution_detail import ExecutionDetail, fill_execution_detail, short_message
from ibrobot_msgs.action import PickObject
from manipulation_execution.grasp_pipeline.backends import backend_spec, load_backends
from manipulation_execution.grasp_pipeline.outcome_message import outcome_message
from manipulation_execution.grasp_pipeline.pipeline import GraspPipeline, PickMode, PickOutcome, PipelineConfig
from manipulation_execution.pick_goal import pick_goal_rejection, remaining_budget_s

EXECUTOR_NAME = "grasp_pipeline"
MODES = {
    PickObject.Goal.MODE_EXECUTE: PickMode.EXECUTE,
    PickObject.Goal.MODE_PLAN_ONLY: PickMode.PLAN_ONLY,
    PickObject.Goal.MODE_OBSERVE_ONLY: PickMode.OBSERVE_ONLY,
}
#: Feedback progress at the start of the pipeline's own phases; plan steps keep the last value.
PROGRESS = {"observe": 0.1, "perceive": 0.2, "plan": 0.3}
#: Between attempts of a backend warmup that has not succeeded yet.
PREPARE_RETRY_S = 1.0
# A fresh capture, scene and home plan after a retained object drops. A cuRobo
# rebuild (the hand off its open preset) cannot be preempted and overruns it.
DROP_HOME_BUDGET_S = 20.0
# After a drop whose way home failed, the arm counts as home again within this
# of the pick's start on every joint: the task-pose arrival tolerance.
HOME_SETTLED_RAD = 0.1


def home_settled(backends, held):
    """Read-only check for the custody: the hand open and empty, the arm back at the pick's start.

    None when the end effector cannot tell an open, empty hand: such a custody
    keeps waiting for a manual check.
    """
    released = getattr(backends.effector, "released", None)
    if released is None:
        return None
    home = np.asarray(held.home_positions, dtype=float)

    def settled():
        measured = np.asarray(backends.executor.measured(held.joint_names), dtype=float)
        return bool(released()) and float(np.max(np.abs(measured - home))) <= HOME_SETTLED_RAD

    return settled


def pipeline_config(grasp_execution: Mapping) -> PipelineConfig:
    options = grasp_execution.get("pipeline") or {}
    if not isinstance(options, Mapping):
        raise ValueError("grasp_execution.pipeline must be a mapping")
    known = set(PipelineConfig.__dataclass_fields__)
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"grasp_execution.pipeline has unknown keys {sorted(unknown)}; known: {sorted(known)}")
    return PipelineConfig(**dict(options))


def verification_status(outcome: PickOutcome) -> int:
    """The close's answer; NOT_RUN when no close finished."""
    if (
        outcome.held is None
        and "close" not in outcome.completed_phases
        and outcome.execution_detail.reason_code != "GRASP_UNCERTAIN"
    ):
        return PickObject.Result.VERIFICATION_NOT_RUN
    return {
        True: PickObject.Result.VERIFICATION_SUCCESS,
        False: PickObject.Result.VERIFICATION_FAILED,
        None: PickObject.Result.VERIFICATION_UNCERTAIN,
    }[outcome.held]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


#: What each attempt's target watch summary keeps in the terminal log (its per-frame records never leave the head).
WATCH_LOG_KEYS = ("mode", "state", "imu", "frames", "votes", "skipped", "moved", "halted", "fault", "error", "host")
#: The end effector phases whose outcome explains a grip: the close, and where it was lost.
GRIP_PHASES = ("close", "lift", "hold_lifted")
GRIP_LOG_KEYS = ("state", "reason", "held", "gripped", "contacts", "measured", "command", "touch_after")


#: The first arm motion of a pick: everything before it is the pick's wait.
MOTION_PHASE = "approach"


class PreMotionTimer:
    """Seconds from the goal's start to each phase start/end, up to the first arm motion.

    ``to_motion_s`` is when the first ``approach`` started (the arm's first command); None when the
    arm never moved. Phases on the way (re-observations included) are kept in order.
    """

    def __init__(self, clock=time.monotonic, *, motion_phase=MOTION_PHASE):
        self._clock, self._started = clock, clock()
        self._motion_phase = motion_phase
        self.to_motion_s: float | None = None
        self.phases: list[tuple[str, str, float]] = []

    def phase(self, name: str, event: str) -> None:
        if self.to_motion_s is not None:
            return
        at = round(self._clock() - self._started, 3)
        self.phases.append((name, event, at))
        if name == self._motion_phase and event == "start":
            self.to_motion_s = at

    def record(self) -> dict:
        return {"to_motion_s": self.to_motion_s, "phases": [list(row) for row in self.phases]}


def thinking_report(report: Mapping) -> dict:
    """The accepted attempt's stage timings: perception spans (side-by-side stages), the planning host's scene
    stages and the planner's own timings. Earlier failed attempts appear only in ``phases``."""
    perceive = report.get("perceive", {}) if isinstance(report.get("perceive"), Mapping) else {}
    timings = perceive.get("perception_timings_s", {}) if isinstance(perceive, Mapping) else {}
    plan = report.get("plan", {}) if isinstance(report.get("plan"), Mapping) else {}
    return {
        "perception_spans_s": timings.get("spans_s", []) if isinstance(timings, Mapping) else [],
        "scene_host_s": perceive.get("scene_host", {}),
        "plan_timings_s": plan.get("request_timings_s", {}),
    }


def terminal_record(outcome: PickOutcome, pre_motion: PreMotionTimer | None = None) -> dict:
    """The pick's one-line terminal log: the chosen arm, phases, the wait before the arm first moved, and per
    attempt the watch summary and the grip."""
    report = outcome.report
    return {
        "event": "grasp_pipeline_terminal",
        "success": outcome.success,
        "arm_side": report.get("plan", {}).get("arm_side", ""),
        "completed_phases": outcome.completed_phases,
        "held": outcome.held,
        "released_after_success": outcome.released_after_success,
        "at_rest": outcome.at_rest,
        "error_code": outcome.error_code,
        "reason_code": outcome.execution_detail.reason_code,
        "error_detail": outcome.message,
        "attempts": outcome.attempts,
        "retries": outcome.retries,
        "first_execution_failure": report.get("first_execution_failure"),
        "execution_failures": report.get("execution_failures", []),
        "handover_s": report.get("handover_s"),
        "pre_motion": None if pre_motion is None else {**pre_motion.record(), **thinking_report(report)},
        "watch": [
            {key: watch[key] for key in WATCH_LOG_KEYS if key in watch}
            for watch in report.get("watch", [])
            if isinstance(watch, Mapping)
        ],
        "grip": [
            [
                {"name": row["name"], **{key: row[key] for key in GRIP_LOG_KEYS if key in row}}
                for row in execution.get("phases", [])
                if isinstance(row, Mapping) and row.get("name") in GRIP_PHASES
            ]
            for execution in report.get("executions", [])
            if isinstance(execution, Mapping)
        ],
    }


def pick_result(outcome: PickOutcome, identity: dict[str, str]) -> PickObject.Result:
    result = PickObject.Result()
    result.success = bool(outcome.success)
    result.error_code = outcome.error_code
    # Only the bounded summary is prose; facts travel independently.
    result.message = outcome_message(outcome)
    fill_execution_detail(result.execution_detail, outcome.execution_detail)
    result.attempts = int(outcome.attempts)
    result.verification_status = verification_status(outcome)
    result.verification_confidence = float(outcome.held_confidence)
    result.completed_phases = list(outcome.completed_phases)
    result.candidate_index = int(outcome.candidate_index)
    result.released_after_success = bool(outcome.released_after_success)
    report = {**outcome.report, "retries": outcome.retries, "error_detail": outcome.message, "mode": outcome.mode}
    result.pipeline_timings_json = json.dumps(report, sort_keys=True, default=_json_value)
    fill_delegated_executor_identity(result.actual_executor, identity)
    return result


class GraspPipelineNode(Node):
    def __init__(self, **node_options) -> None:
        super().__init__("grasp_pipeline_node", **node_options)
        self.declare_parameter("action_name", "/manipulation/execute_pick")
        self.declare_parameter("grasp_execution_json", "{}")
        self.declare_parameter("interface_description_json", "{}")
        self._action_name = str(self.get_parameter("action_name").value)
        config = json.loads(self.get_parameter("grasp_execution_json").value)
        description = json.loads(self.get_parameter("interface_description_json").value)
        if not isinstance(config, dict) or not isinstance(description, dict):
            raise ValueError("grasp_execution_json and interface_description_json must be JSON objects")
        spec = backend_spec(config)
        if spec is None:
            raise ValueError("grasp_execution.backend is required; robots without one use pick_executor_node")
        self._identity = delegated_executor_identity(
            name=EXECUTOR_NAME,
            endpoint_name=self._action_name,
            configuration=config,
            **load_delegated_model_identity(config),
        )
        self._backends = load_backends(spec, node=self, grasp_execution=config, description=description)
        self._pipeline = GraspPipeline(self._backends, pipeline_config(config))
        self._custody = None
        if self._backends.held_object is not None:
            from manipulation_execution.held_custody import HeldCustodyServer

            self._custody = HeldCustodyServer(self, lease_s=float(config.get("held_lease_s", 30.0)))
        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._closing = threading.Event()
        self._ready = threading.Event()
        if self._backends.prepare is None:
            self._ready.set()
        else:
            threading.Thread(target=self._prepare, daemon=True, name="grasp_backend_prepare").start()
        self._action_server = ActionServer(
            self,
            PickObject,
            self._action_name,
            execute_callback=self._execute,
            goal_callback=self._handle_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
            feedback_pub_qos_profile=QoSProfile(depth=256),
        )
        self.get_logger().info(f"grasp pipeline ready: action={self._action_name} backend={spec}")

    def _prepare(self) -> None:
        """Warm the backends up off the executor; goals are refused until it succeeds."""
        while not self._closing.is_set():
            try:
                self._backends.prepare()
            except Exception as exc:  # noqa: BLE001 - retried until the robot and its models are up
                self.get_logger().warning(f"grasp backends not ready: {exc}", throttle_duration_sec=5.0)
                self._closing.wait(PREPARE_RETRY_S)
                continue
            self._ready.set()
            self.get_logger().info(f"grasp backends prepared; accepting {self._action_name} goals")
            return

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000

    def _handle_goal(self, goal) -> GoalResponse:
        reason = pick_goal_rejection(goal, self._identity, self._now_s())
        if reason:
            self.get_logger().warning(f"PickObject goal rejected: {reason}")
            return GoalResponse.REJECT
        if not self._ready.is_set():
            self.get_logger().warning("PickObject goal rejected: the grasp backends are still warming up")
            return GoalResponse.REJECT
        if goal.retain_for_place and (self._custody is None or goal.release_after_success):
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active or (self._custody is not None and self._custody.custody.occupied):
                self.get_logger().warning("PickObject goal rejected: another goal is running")
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        goal_handle.phase_sequence = 0
        goal_handle.phase_history = deque(maxlen=256)
        timer = PreMotionTimer()
        try:
            remaining = remaining_budget_s(goal, self._now_s())
            if remaining <= 0.0:
                outcome = PickOutcome(
                    error_code="TASK_TIMEOUT",
                    message="task budget expired before the pick started",
                    execution_detail=ExecutionDetail(
                        1, "dispatch", "FAILED", "dispatch", "TASK_TIMEOUT", "NOT_NEEDED", 0, "WAIT_SERVICE"
                    ),
                )
            else:
                outcome = self._run(goal_handle, time.monotonic() + remaining, timer)
            result = pick_result(outcome, self._identity)
            result.last_feedback_sequence = goal_handle.phase_sequence
            result.execution_history = list(goal_handle.phase_history)
            handover = outcome.report.get("handover_s")
            if isinstance(handover, dict):
                handover.update(getattr(self, "_retain_timings", {}))
                self._retain_timings = {}
                done = handover.pop("_sequence_done", None)
                if done is not None:
                    handover["to_terminal"] = round(time.monotonic() - done, 4)
            self.get_logger().info(json.dumps(terminal_record(outcome, timer), default=_json_value))
            if not outcome.at_rest:
                self.get_logger().error(f"arm stopped in place: {outcome.error_code}: {outcome.message}")
            if outcome.success:
                goal_handle.succeed()
            elif goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False

    def _run(self, goal_handle, deadline: float, timer: PreMotionTimer | None = None) -> PickOutcome:
        goal = goal_handle.request
        progress = {"value": 0.0, "attempt": 0}

        def on_phase(name: str, event: str, detail: dict) -> None:
            if timer is not None:
                timer.phase(name, event)
            if event == "start":
                if name == "observe":
                    progress["attempt"] += 1
                progress["value"] = PROGRESS.get(name, max(progress["value"], 0.4))
                message = ""
            elif detail.get("state", "succeeded") != "succeeded":
                message = f"{detail.get('state')}: {detail.get('reason', '')}"
            else:
                message = "succeeded"
            feedback = PickObject.Feedback(
                phase=name,
                progress=progress["value"],
                attempt=max(0, progress["attempt"] - 1),
                detail=short_message(message),
            )
            fill_execution_detail(feedback.execution_detail, detail.get("execution_detail"))
            goal_handle.phase_sequence += 1
            feedback.sequence = goal_handle.phase_sequence
            goal_handle.phase_history.append(feedback.execution_detail)
            goal_handle.publish_feedback(feedback)

        return self._pipeline.run(
            str(goal.target_query).strip(),
            mode=MODES[int(goal.mode)],
            deadline=deadline,
            cancelled=lambda: goal_handle.is_cancel_requested,
            on_phase=on_phase,
            release_after_success=self._releases(goal),
            on_held=(lambda observation, proposals, plan: self._retain(goal, observation, proposals, plan))
            if goal.retain_for_place
            else None,
            # Describe the held object while the arm lifts (held_builder.prepare).
            on_closed=getattr(self._backends.held_object, "prepare", None) if goal.retain_for_place else None,
        )

    def _retain(self, goal, observation, proposals, plan):
        from manipulation_execution.grasp_pipeline.sequence import GraspSequence

        backends = self._backends
        began = time.monotonic()
        held = backends.held_object(str(goal.target_query).strip(), observation, proposals, plan)
        self._retain_timings = {"held_object": round(time.monotonic() - began, 4)}
        accepting = time.monotonic()

        def recover():
            return GraspSequence(backends).take_back(plan, cancelled=self._closing.is_set)

        def release():
            backends.executor.finish()
            backends.effector.suspend()

        def drop():
            deadline = time.monotonic() + DROP_HOME_BUDGET_S

            def go_home():
                return backends.plan_home(held, deadline=deadline, cancelled=self._closing.is_set)

            return GraspSequence(backends).home_after_drop(go_home, cancelled=self._closing.is_set)

        self._custody.custody.accept(
            held,
            recover=recover,
            release=release,
            holding=backends.effector.holding,
            root_task_id=goal.dispatch_binding.root_task_id,
            drop=drop if backends.plan_home is not None else None,
            settled=home_settled(backends, held),
        )
        self._retain_timings["custody_accept"] = round(time.monotonic() - accepting, 4)

    def _releases(self, goal) -> bool:
        """Whether this pick puts the object back: the goal asks, or the robot always does."""
        return (
            False
            if goal.retain_for_place
            else bool(goal.release_after_success) or self._pipeline.config.release_after_success
        )

    def destroy_node(self) -> None:
        closing = getattr(self, "_closing", None)
        if closing is not None:
            closing.set()
        try:
            backends = getattr(self, "_backends", None)
            if backends is not None:
                backends.close()
        finally:
            action = getattr(self, "_action_server", None)
            if action is not None:
                action.destroy()
                self._action_server = None
            super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspPipelineNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
