"""ROS Action server for the internal imitate_human_motion executor."""

from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from embodied_common.dispatch_binding import (
    copy_binding,
    delegated_executor_identity,
    delegated_executor_identity_matches,
    fill_delegated_executor_identity,
)
from ibrobot_msgs.action import ImitateHumanMotion, PrimitiveCommand
from manipulation_execution.imitate_human_motion_executor import (
    CANCEL_CLEANUP_TIMEOUT,
    AnimationPlan,
    MockExecutor,
    MockGoal,
    MockResult,
    PrimitiveStateUnknown,
)

_PREPARE_DURATION_SEC = 2.0


class _GuardedPrimitivePlayer:
    """Translate the HRI lifecycle into the existing guarded primitives."""

    def __init__(
        self,
        client,
        goal_handle,
        *,
        joint_names: tuple[str, ...],
        reset_positions: dict[str, float],
        rpc_timeout_sec: float,
        deadline: float,
        ros_now_sec,
    ) -> None:
        self._client = client
        self._goal_handle = goal_handle
        self._binding = copy_binding(goal_handle.request.dispatch_binding)
        self._joint_names = joint_names
        self._reset_positions = reset_positions
        self._rpc_timeout = rpc_timeout_sec
        self._ros_now_sec = ros_now_sec
        budget = self._binding.task_budget
        self._task_deadline_ros = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        self._deadline = self._bounded_deadline(deadline)
        raw_uuid = getattr(getattr(goal_handle, "goal_id", None), "uuid", None)
        self._execution_token = bytes(raw_uuid).hex() if raw_uuid is not None else ""

    def _bounded_deadline(self, requested_deadline: float) -> float:
        remaining_budget = max(0.0, self._task_deadline_ros - self._ros_now_sec())
        return min(requested_deadline, time.monotonic() + remaining_budget)

    def _wait(self, future, deadline: float, *, honor_cancel: bool) -> bool:
        while not future.done():
            if honor_cancel and self._goal_handle.is_cancel_requested:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _cancel(self, handle, result_future=None) -> bool:
        try:
            result_future = result_future or handle.get_result_async()
            cancel_future = handle.cancel_goal_async()
        except Exception:
            return False
        cleanup_deadline = self._bounded_deadline(time.monotonic() + self._rpc_timeout)
        if not self._wait(cancel_future, cleanup_deadline, honor_cancel=False):
            return False
        try:
            response = cancel_future.result()
            if response is None or not bool(response.goals_canceling):
                return False
            if not self._wait(result_future, cleanup_deadline, honor_cancel=False):
                return False
            wrapped = result_future.result()
        except Exception:
            return False
        result = getattr(wrapped, "result", None)
        return result is not None and str(getattr(result, "error_code", "")) != CANCEL_CLEANUP_TIMEOUT

    def _cancel_pending(self, send_future) -> bool:
        cleanup_deadline = self._bounded_deadline(time.monotonic() + self._rpc_timeout)
        if not self._wait(send_future, cleanup_deadline, honor_cancel=False):
            send_future.add_done_callback(self._cancel_when_ready)
            return False
        try:
            handle = send_future.result()
        except Exception:
            return False
        return handle is None or not handle.accepted or self._cancel(handle)

    def _cancel_when_ready(self, send_future) -> None:
        try:
            handle = send_future.result()
        except Exception:
            return
        if handle is not None and handle.accepted:
            self._cancel(handle)

    def _run(
        self,
        *,
        primitive_name: str,
        deadline: float,
        duration_sec: float = 0.0,
        joint_positions: tuple[float, ...] = (),
        pose_name: str = "",
        honor_cancel: bool = True,
    ) -> str:
        deadline = self._bounded_deadline(deadline)
        wait_sec = min(self._rpc_timeout, max(0.0, deadline - time.monotonic()))
        if wait_sec <= 0.0 or not self._client.wait_for_server(timeout_sec=wait_sec):
            return "TIMEOUT" if time.monotonic() >= deadline else "FAILED"
        goal = PrimitiveCommand.Goal()
        goal.schema_version = 1
        goal.dispatch_binding = copy_binding(self._binding)
        goal.execution_token = self._execution_token
        goal.primitive_name = primitive_name
        goal.pose_name = pose_name
        goal.joint_names = list(self._joint_names) if joint_positions else []
        goal.joint_positions = list(joint_positions)
        goal.primitive_duration_sec = float(duration_sec)
        goal.timeout_sec = deadline - time.monotonic()
        try:
            send_future = self._client.send_goal_async(goal)
        except Exception:
            return "UNKNOWN"
        if not self._wait(send_future, deadline, honor_cancel=honor_cancel):
            requested_cancel = honor_cancel and self._goal_handle.is_cancel_requested
            if not self._cancel_pending(send_future):
                return "UNKNOWN"
            return "CANCELED" if requested_cancel else "TIMEOUT"
        try:
            handle = send_future.result()
        except Exception:
            return "UNKNOWN"
        if handle is None or not handle.accepted:
            return "FAILED"
        try:
            result_future = handle.get_result_async()
        except Exception:
            return "UNKNOWN"
        if not self._wait(result_future, deadline, honor_cancel=honor_cancel):
            requested_cancel = honor_cancel and self._goal_handle.is_cancel_requested
            if not self._cancel(handle, result_future):
                return "UNKNOWN"
            return "CANCELED" if requested_cancel else "TIMEOUT"
        try:
            wrapped = result_future.result()
        except Exception:
            return "UNKNOWN"
        result = getattr(wrapped, "result", None)
        if result is None or str(getattr(result, "error_code", "")) == CANCEL_CLEANUP_TIMEOUT:
            return "UNKNOWN"
        return "COMPLETED" if bool(result.success) else "FAILED"

    def prepare(self) -> bool:
        outcome = self._run(
            primitive_name="move_to_joint_positions",
            joint_positions=tuple(self._reset_positions[name] for name in self._joint_names),
            duration_sec=_PREPARE_DURATION_SEC,
            deadline=self._deadline,
        )
        if outcome == "UNKNOWN":
            raise PrimitiveStateUnknown("prepare primitive execution state is unknown")
        return outcome == "COMPLETED"

    def play(self, plan: AnimationPlan, duration_sec: float, *, feedback, is_cancel_requested, deadline) -> str:
        segment_duration = plan.duration_sec / (len(plan.waypoints) - 1)
        remaining = duration_sec
        for start, end in zip(plan.waypoints, plan.waypoints[1:], strict=True):
            if remaining <= 0.0:
                break
            active_duration = min(segment_duration, remaining)
            ratio = active_duration / segment_duration
            target = tuple(a + (b - a) * ratio for a, b in zip(start, end, strict=True))
            outcome = self._run(
                primitive_name="move_to_joint_positions",
                joint_positions=target,
                duration_sec=active_duration,
                deadline=deadline,
            )
            if outcome != "COMPLETED":
                return outcome
            remaining -= active_duration
            progress = min(1.0, (duration_sec - remaining) / duration_sec)
            feedback("mock_playback", progress, f"Executing {plan.animation_id}")
        return "COMPLETED"

    def reset(self) -> bool:
        outcome = self._run(
            primitive_name="move_to_named_pose",
            pose_name="home",
            deadline=self._bounded_deadline(time.monotonic() + max(30.0, self._rpc_timeout)),
            honor_cancel=False,
        )
        if outcome == "UNKNOWN":
            raise PrimitiveStateUnknown("reset primitive execution state is unknown")
        return outcome == "COMPLETED"


class ImitateHumanMotionExecutorNode(Node):
    """Serve the internal delegated HRI Action as a launch-managed runtime."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("imitate_human_motion_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("action_name", "/hri/imitate_human_motion")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("startup_warmup", True)
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("reset_positions_json", "{}")
        self.declare_parameter("joint_limits_json", "{}")
        self._action_name = str(self.get_parameter("action_name").value)
        primitive_action_name = str(self.get_parameter("primitive_action_name").value)
        self._rpc_timeout = float(self.get_parameter("rpc_timeout_sec").value)
        self._startup_warmup = bool(self.get_parameter("startup_warmup").value)
        self._joint_names = tuple(str(name) for name in json.loads(self.get_parameter("arm_joint_names_json").value))
        self._reset_positions = {
            str(name): float(value)
            for name, value in json.loads(self.get_parameter("reset_positions_json").value).items()
        }
        self._joint_limits = json.loads(self.get_parameter("joint_limits_json").value)
        self._executor_identity = delegated_executor_identity(
            name="imitate_human_motion",
            endpoint_name=self._action_name,
            configuration={"implementation": "mock_v1"},
        )
        self._mock = MockExecutor(
            joint_names=self._joint_names,
            reset_positions=self._reset_positions,
            joint_limits=self._joint_limits,
            warmup_ready=False,
        )
        self._startup_warmup_attempted = False
        self._goal_lock = threading.Lock()
        self._goal_active = False
        callback_group = ReentrantCallbackGroup()
        self._primitive_client = ActionClient(
            self,
            PrimitiveCommand,
            primitive_action_name,
            callback_group=callback_group,
        )
        self._action_server = ActionServer(
            self,
            ImitateHumanMotion,
            self._action_name,
            execute_callback=self._execute,
            goal_callback=self._handle_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self.get_logger().info(f"imitate_human_motion executor ready: action={self._action_name}")
        if self._startup_warmup:
            self._run_startup_warmup()

    def _run_startup_warmup(self) -> bool:
        """Perform the single launch-time warmup; tasks never repeat it."""
        if self._startup_warmup_attempted:
            return self._mock.status.warmup_ready
        self._startup_warmup_attempted = True
        ready = self._mock.warmup()
        if ready:
            self.get_logger().info("imitate_human_motion warmup READY")
        else:
            self.get_logger().error("imitate_human_motion warmup FAILED")
        return ready

    def _binding_is_valid(self, request) -> bool:
        binding = request.dispatch_binding
        budget = binding.task_budget
        started = budget.started_at.sec + budget.started_at.nanosec / 1_000_000_000
        deadline = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        timeout_sec = float(request.timeout_sec)
        now = self.get_clock().now().nanoseconds / 1_000_000_000
        return bool(
            binding.schema_version == 1
            and str(binding.task_id).strip()
            and str(binding.root_task_id).strip()
            and str(binding.expected_registry_epoch).strip()
            and int(binding.expected_registry_generation) > 0
            and str(binding.expected_registry_digest).strip()
            and str(binding.dispatch_nonce).strip()
            and budget.schema_version == 1
            and budget.started_at.sec >= 0
            and budget.deadline.sec >= 0
            and 0 <= budget.started_at.nanosec < 1_000_000_000
            and 0 <= budget.deadline.nanosec < 1_000_000_000
            and math.isfinite(started)
            and math.isfinite(deadline)
            and deadline > started
            and deadline > now
            and math.isfinite(timeout_sec)
            and timeout_sec > 0.0
            and timeout_sec <= deadline - now
        )

    def _handle_goal(self, request):
        goal = MockGoal(
            arm_side=str(request.arm_side).strip().lower(),
            imitation_duration_sec=float(request.imitation_duration_sec),
            timeout_sec=float(request.timeout_sec),
        )
        if not self._binding_is_valid(request):
            return GoalResponse.REJECT
        if not delegated_executor_identity_matches(request.expected_executor, self._executor_identity):
            return GoalResponse.REJECT
        accepted, reason = self._mock.can_accept(goal)
        if not accepted:
            if "warmup" in reason:
                self.get_logger().info("imitate_human_motion warmup is in progress")
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _handle_cancel(_goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        goal = MockGoal(
            arm_side=str(request.arm_side).strip().lower(),
            imitation_duration_sec=float(request.imitation_duration_sec),
            timeout_sec=float(request.timeout_sec),
        )

        last_logged_phase = ""

        def publish_feedback(phase: str, progress: float, detail: str) -> None:
            nonlocal last_logged_phase
            feedback = ImitateHumanMotion.Feedback()
            feedback.phase = phase
            feedback.progress = float(progress)
            feedback.detail = detail
            goal_handle.publish_feedback(feedback)
            if phase != last_logged_phase:
                self.get_logger().info(f"imitate_human_motion phase={phase}: {detail}")
                last_logged_phase = phase

        runner = _GuardedPrimitivePlayer(
            self._primitive_client,
            goal_handle,
            joint_names=self._joint_names,
            reset_positions=self._reset_positions,
            rpc_timeout_sec=self._rpc_timeout,
            deadline=time.monotonic() + goal.timeout_sec,
            ros_now_sec=lambda: self.get_clock().now().nanoseconds / 1_000_000_000,
        )
        try:
            result_value = self._mock.execute(
                goal,
                feedback=publish_feedback,
                is_cancel_requested=lambda: bool(goal_handle.is_cancel_requested),
                player=runner,
                prepare=runner.prepare,
                recover_safe_pose=runner.reset,
            )
        except PrimitiveStateUnknown as exc:
            result_value = MockResult(
                success=False,
                error_code=CANCEL_CLEANUP_TIMEOUT,
                message=str(exc),
                animation_id="",
                requested_duration_sec=goal.imitation_duration_sec,
                actual_duration_sec=0.0,
                completed_phases=(),
            )
        except Exception as exc:
            self.get_logger().error(f"imitate_human_motion execution failed: {exc}")
            result_value = MockResult(
                success=False,
                error_code="MOCK_PLAYBACK_FAILED",
                message=str(exc),
                animation_id="",
                requested_duration_sec=goal.imitation_duration_sec,
                actual_duration_sec=0.0,
                completed_phases=(),
            )
        finally:
            with self._goal_lock:
                self._goal_active = False
        result = ImitateHumanMotion.Result()
        result.success = result_value.success
        result.error_code = result_value.error_code
        result.message = result_value.message
        result.animation_id = result_value.animation_id
        result.requested_duration_sec = result_value.requested_duration_sec
        result.actual_duration_sec = result_value.actual_duration_sec
        result.completed_phases = list(result_value.completed_phases)
        fill_delegated_executor_identity(result.actual_executor, self._executor_identity)
        if result_value.success:
            goal_handle.succeed()
        elif result_value.error_code == "CANCELED":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitateHumanMotionExecutorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
