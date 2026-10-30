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
from collections.abc import Mapping

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from embodied_common.dispatch_binding import (
    delegated_executor_identity,
    fill_delegated_executor_identity,
    load_delegated_model_identity,
)
from ibrobot_msgs.action import PickObject
from manipulation_execution.grasp_pipeline.backends import backend_spec, load_backends
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
    if "close" not in outcome.completed_phases:
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


def pick_result(outcome: PickOutcome, identity: dict[str, str]) -> PickObject.Result:
    result = PickObject.Result()
    result.success = bool(outcome.success)
    result.error_code = outcome.error_code
    result.message = outcome.message
    result.attempts = int(outcome.attempts)
    result.verification_status = verification_status(outcome)
    result.verification_confidence = float(outcome.held_confidence)
    result.completed_phases = list(outcome.completed_phases)
    result.candidate_index = int(outcome.candidate_index)
    result.released_after_success = bool(outcome.released_after_success)
    result.pipeline_timings_json = json.dumps(outcome.report, sort_keys=True, default=_json_value)
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
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning("PickObject goal rejected: another goal is running")
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        try:
            remaining = remaining_budget_s(goal, self._now_s())
            if remaining <= 0.0:
                outcome = PickOutcome(error_code="TASK_TIMEOUT", message="task budget expired before the pick started")
            else:
                outcome = self._run(goal_handle, time.monotonic() + remaining)
            result = pick_result(outcome, self._identity)
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

    def _run(self, goal_handle, deadline: float) -> PickOutcome:
        goal = goal_handle.request
        progress = {"value": 0.0, "attempt": 0}

        def on_phase(name: str, event: str, detail: dict) -> None:
            if event == "start":
                if name == "observe":
                    progress["attempt"] += 1
                progress["value"] = PROGRESS.get(name, max(progress["value"], 0.4))
                message = ""
            elif detail.get("state", "succeeded") != "succeeded":
                message = f"{detail.get('state')}: {detail.get('reason', '')}"
            else:
                return
            goal_handle.publish_feedback(
                PickObject.Feedback(
                    phase=name, progress=progress["value"], attempt=max(0, progress["attempt"] - 1), detail=message
                )
            )

        return self._pipeline.run(
            str(goal.target_query).strip(),
            mode=MODES[int(goal.mode)],
            deadline=deadline,
            cancelled=lambda: goal_handle.is_cancel_requested,
            on_phase=on_phase,
            release_after_success=self._releases(goal),
        )

    def _releases(self, goal) -> bool:
        """Whether this pick puts the object back: the goal asks, or the robot always does."""
        return bool(goal.release_after_success) or self._pipeline.config.release_after_success

    def destroy_node(self) -> None:
        closing = getattr(self, "_closing", None)
        if closing is not None:
            closing.set()
        try:
            backends = getattr(self, "_backends", None)
            if backends is not None:
                backends.close()
        finally:
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
