"""Serve backend-driven PlaceObject goals under the Capability Gateway."""

import importlib
import json
import threading
import time
from collections import deque

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from embodied_common.dispatch_binding import delegated_executor_identity, fill_delegated_executor_identity
from embodied_common.execution_detail import ExecutionDetail, fill_execution_detail, short_message
from ibrobot_msgs.action import PickObject, PlaceObject

from .grasp_pipeline.contracts import GraspFailure, PlaceBackends
from .grasp_pipeline.place_pipeline import PlaceOutcome, PlacePipeline
from .grasp_pipeline_node import PreMotionTimer, _json_value
from .held_object_client import HeldObjectClient
from .pick_goal import pick_goal_rejection, remaining_budget_s


class PlacePipelineNode(Node):
    def __init__(self, **options):
        super().__init__("place_pipeline_node", **options)
        self.declare_parameter("action_name", "/manipulation/execute_place")
        self.declare_parameter("placement_execution_json", "{}")
        self.declare_parameter("interface_description_json", "{}")
        self.declare_parameter("grasp_execution_json", "{}")
        config = json.loads(self.get_parameter("placement_execution_json").value)
        description = json.loads(self.get_parameter("interface_description_json").value)
        action = str(self.get_parameter("action_name").value)
        module, name = config["backend"].split(":")
        grasp = json.loads(self.get_parameter("grasp_execution_json").value)
        self.backends = getattr(importlib.import_module(module), name)(
            node=self, placement_execution=config, grasp_execution=grasp, description=description
        )
        if not isinstance(self.backends, PlaceBackends):
            raise ValueError("placement backend must return PlaceBackends")
        self.identity = delegated_executor_identity(
            name="placement_pipeline", endpoint_name=action, configuration=config
        )
        self.pipeline = PlacePipeline(self.backends, **config.get("pipeline", {}))
        self.custody = HeldObjectClient(self)
        self.lock, self.active = threading.Lock(), False
        self.server = ActionServer(
            self,
            PlaceObject,
            action,
            execute_callback=self.execute,
            goal_callback=self.admit,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def admit(self, goal):
        # Share the dispatch and budget validation, using pick's execution mode.
        proxy = PickObject.Goal(
            target_query=goal.target_query,
            timeout_sec=goal.timeout_sec,
            dispatch_binding=goal.dispatch_binding,
            expected_executor=goal.expected_executor,
            mode=PickObject.Goal.MODE_EXECUTE,
        )
        reason = pick_goal_rejection(proxy, self.identity, self.get_clock().now().nanoseconds * 1e-9)
        if reason or not goal.container_query.strip():
            self.get_logger().warning(f"place rejected: {reason or 'empty container query'}")
            return GoalResponse.REJECT
        with self.lock:
            if self.active:
                return GoalResponse.REJECT
            self.active = True
        return GoalResponse.ACCEPT

    def execute(self, handle):
        try:
            return self._execute(handle)
        finally:
            with self.lock:
                self.active = False

    def _execute(self, handle):
        timer = PreMotionTimer(motion_phase="transfer")
        history, sequence = deque(maxlen=256), 0
        held = None
        first_motion_ns = 0

        def phase(name, event, detail):
            nonlocal sequence, first_motion_ns
            timer.phase(name, event)
            if name == "transfer" and event == "start" and not first_motion_ns:
                first_motion_ns = self.get_clock().now().nanoseconds
            feedback = PlaceObject.Feedback(
                phase=name, detail=short_message(event), progress=0.5 if name == "transfer" else 0.1
            )
            fill_execution_detail(feedback.execution_detail, detail.get("execution_detail"))
            sequence += 1
            feedback.sequence = sequence
            history.append(feedback.execution_detail)
            handle.publish_feedback(feedback)

        try:
            goal = handle.request
            # Claims last at most 120 s. Leave three seconds for returning
            # custody after stopping at the execution deadline.
            deadline = time.monotonic() + min(
                117.0, max(0.0, remaining_budget_s(goal, self.get_clock().now().nanoseconds * 1e-9))
            )
            phase("claim", "start", {})
            held = self.custody.claim(
                goal.target_query,
                goal.dispatch_binding,
                deadline=deadline,
                cancelled=lambda: handle.is_cancel_requested,
            )
            phase("claim", "end", {})
            outcome = self.pipeline.run(
                held,
                goal.container_query.strip(),
                deadline=deadline,
                cancelled=lambda: handle.is_cancel_requested,
                on_phase=phase,
            )
        except GraspFailure as failure:
            outcome = PlaceOutcome(
                reason_code=failure.code,
                message=str(failure),
                execution_detail=ExecutionDetail(
                    schema_version=1,
                    phase="claim",
                    phase_state="CANCELED" if failure.code == "CANCELED" else "FAILED",
                    failed_phase="claim",
                    reason_code=failure.code,
                    recovery_state="NOT_NEEDED",
                ),
            )
        except Exception as exc:
            outcome = PlaceOutcome(reason_code="INTERNAL_ERROR", message=str(exc), custody_outcome="uncertain")
        if held is not None:
            try:
                self.custody.returned(outcome.custody_outcome, outcome.reason_code)
            except GraspFailure as failure:
                outcome.success, outcome.reason_code = False, failure.code
                outcome.execution_detail = ExecutionDetail(
                    schema_version=1,
                    phase="return_custody",
                    phase_state="FAILED",
                    failed_phase="return_custody",
                    reason_code=failure.code,
                    # All arm motion has ended. Unknown custody delivery is
                    # not an unconfirmed controller stop and must not keep
                    # the Gateway execution lease or quarantine the Program.
                    recovery_state=outcome.execution_detail.recovery_state
                    if outcome.execution_detail.recovery_state == "UNKNOWN"
                    else "FAILED",
                    suggested_action="MANUAL_CHECK",
                )
        report = {
            **outcome.report,
            "pre_motion": timer.record(),
            "lift_to_transfer_s": None
            if held is None or not first_motion_ns
            else (first_motion_ns - held.lifted_at_ns) * 1e-9,
        }
        self.get_logger().info(
            json.dumps(
                {
                    "event": "place_pipeline_terminal",
                    "success": outcome.success,
                    "reason_code": outcome.reason_code,
                    "message": outcome.message,
                    **report,
                },
                default=_json_value,
            )
        )
        result = PlaceObject.Result(
            success=outcome.success,
            place_succeeded=outcome.success,
            error_code="" if outcome.success else outcome.reason_code,
            message=short_message(outcome.message),
            completed_phases=outcome.completed_phases,
            pipeline_timings_json=json.dumps(report, default=_json_value),
            last_feedback_sequence=sequence,
            execution_history=list(history),
        )
        result.release_status = {
            "not_released": result.RELEASE_NOT_RELEASED,
            "released": result.RELEASE_RELEASED,
            "unknown": result.RELEASE_UNKNOWN,
        }[outcome.release_status]
        result.verification_status = {
            "not_run": result.VERIFICATION_NOT_RUN,
            "success": result.VERIFICATION_SUCCESS,
            "failed": result.VERIFICATION_FAILED,
            "uncertain": result.VERIFICATION_UNCERTAIN,
        }[outcome.verification]
        fill_execution_detail(result.execution_detail, outcome.execution_detail)
        fill_delegated_executor_identity(result.actual_executor, self.identity)
        if outcome.success:
            handle.succeed()
        elif handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.abort()
        return result

    def destroy_node(self):
        self.backends.close()
        self.server.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlacePipelineNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
