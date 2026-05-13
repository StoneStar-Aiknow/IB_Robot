"""Task executor node for the embodied minimal closure."""

import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient

from embodied_agent.base_node import BaseTaskNode
from embodied_agent.task_context import ensure_timeout_context, remaining_task_budget_sec
from embodied_common.base_node import BaseTaskNode
from ibrobot_msgs.action import SkillCommand
from ibrobot_msgs.msg import TaskCommand, TaskStatus


class TaskExecutorNode(BaseTaskNode):
    """Execute a planned task by calling the skill action server in sequence."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("task_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/embodied/planned_task")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("default_task_timeout_sec", 180.0)
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("debug_tracing", False)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._skill_action_name = self.get_parameter("skill_action_name").get_parameter_value().string_value
        self._default_timeout = self.get_parameter("default_task_timeout_sec").get_parameter_value().double_value
        self._rpc_timeout = self.get_parameter("rpc_timeout_sec").get_parameter_value().double_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._active_task_lock = threading.Lock()
        self._active_task_id = ""

        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self._skill_client = ActionClient(self, SkillCommand, self._skill_action_name)
        self.create_subscription(TaskCommand, self._input_topic, self._handle_planned_task, 10)

        self.get_logger().info(
            f"[embodied-debug] task_executor ready: input_topic={self._input_topic}, "
            f"skill_action={self._skill_action_name}"
        )

    def _handle_planned_task(self, msg: TaskCommand) -> None:
        if not self._active_task_lock.acquire(blocking=False):
            self._publish_status(
                task_id=msg.task_id,
                state="rejected",
                success=False,
                message="executor busy",
                error_code="EXECUTOR_BUSY",
                recoverable=True,
                replan_requested=True,
            )
            self.get_logger().warning(f"[embodied-debug] task_executor busy, rejecting task_id={msg.task_id}")
            return

        self._active_task_id = msg.task_id
        worker = threading.Thread(target=self._execute_task, args=(msg,), daemon=True)
        worker.start()

    def _wait_for_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        return future.done()

    def _resolve_task_context(self, msg: TaskCommand) -> dict[str, Any]:
        return ensure_timeout_context(msg.context_json, msg.timeout_sec or self._default_timeout)

    def _remaining_budget_sec(self, context: dict[str, Any]) -> float | None:
        return remaining_task_budget_sec(context)

    def _cancel_goal(self, goal_handle) -> None:
        if goal_handle is None:
            return
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            return
        self._wait_for_future(cancel_future, timeout_sec=self._rpc_timeout)

    def _execute_task(self, msg: TaskCommand) -> None:
        completed_skills: list[str] = []
        try:
            try:
                plan_context = self._resolve_task_context(msg)
            except ValueError:
                plan_context = ensure_timeout_context("{}", msg.timeout_sec or self._default_timeout)

            skill_sequence = plan_context.get("skill_sequence", [])
            if not skill_sequence:
                self._publish_status(
                    task_id=msg.task_id,
                    state="failed",
                    success=False,
                    message="planned task missing skill sequence",
                    error_code="EMPTY_PLAN",
                    replan_requested=True,
                )
                return

            initial_budget = self._remaining_budget_sec(plan_context)
            if initial_budget is not None and initial_budget <= 0.0:
                self._publish_status(
                    task_id=msg.task_id,
                    state="failed",
                    success=False,
                    message="task deadline exceeded before execution started",
                    error_code="TASK_TIMEOUT",
                    recoverable=True,
                    replan_requested=True,
                )
                return

            wait_timeout = self._rpc_timeout if initial_budget is None else min(self._rpc_timeout, initial_budget)
            if not self._skill_client.wait_for_server(timeout_sec=max(0.1, wait_timeout)):
                self._publish_status(
                    task_id=msg.task_id,
                    state="failed",
                    success=False,
                    message="skill action server not available",
                    error_code="SKILL_SERVER_UNAVAILABLE",
                    recoverable=True,
                    replan_requested=True,
                )
                return

            for skill_name in skill_sequence:
                remaining_budget = self._remaining_budget_sec(plan_context)
                if remaining_budget is not None and remaining_budget <= 0.0:
                    self._publish_status(
                        task_id=msg.task_id,
                        state="failed",
                        success=False,
                        message=f"task deadline exceeded before {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="TASK_TIMEOUT",
                        recoverable=True,
                        replan_requested=True,
                    )
                    return

                self._publish_status(
                    task_id=msg.task_id,
                    state="executing",
                    success=True,
                    message=f"executing skill {skill_name}",
                    current_skill=skill_name,
                    completed_skills=completed_skills,
                )
                if self._debug:
                    self.get_logger().info(
                        f"[embodied-debug] task_executor dispatch skill task_id={msg.task_id} skill={skill_name}"
                    )

                goal = SkillCommand.Goal()
                goal.task_id = msg.task_id
                goal.skill_name = skill_name
                goal.target_name = msg.target_name
                goal.place_name = msg.place_name
                goal.motion_direction = msg.motion_direction
                goal.motion_distance = msg.motion_distance
                goal.timeout_sec = (
                    float(remaining_budget)
                    if remaining_budget is not None
                    else float(msg.timeout_sec or self._default_timeout)
                )

                send_goal_future = self._skill_client.send_goal_async(goal)
                send_timeout = min(self._rpc_timeout, goal.timeout_sec)
                if not self._wait_for_future(send_goal_future, timeout_sec=max(0.1, send_timeout)):
                    self._publish_status(
                        task_id=msg.task_id,
                        state="failed",
                        success=False,
                        message=f"timeout while sending goal for {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="GOAL_TIMEOUT",
                        recoverable=True,
                        replan_requested=True,
                    )
                    return

                goal_handle = send_goal_future.result()
                if goal_handle is None or not goal_handle.accepted:
                    self._publish_status(
                        task_id=msg.task_id,
                        state="failed",
                        success=False,
                        message=f"skill goal rejected: {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="GOAL_REJECTED",
                        recoverable=True,
                        replan_requested=True,
                    )
                    return

                result_future = goal_handle.get_result_async()
                result_timeout = self._remaining_budget_sec(plan_context)
                if result_timeout is None:
                    result_timeout = goal.timeout_sec
                if not self._wait_for_future(result_future, timeout_sec=max(0.1, result_timeout)):
                    self._cancel_goal(goal_handle)
                    self._publish_status(
                        task_id=msg.task_id,
                        state="failed",
                        success=False,
                        message=f"skill execution timeout: {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="TASK_TIMEOUT",
                        recoverable=True,
                        replan_requested=True,
                    )
                    return

                action_result = result_future.result()
                result = action_result.result if action_result is not None else None
                if result is None or not result.success:
                    self._publish_status(
                        task_id=msg.task_id,
                        state="failed",
                        success=False,
                        message=result.message if result is not None else "missing result",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code=result.error_code if result is not None else "MISSING_RESULT",
                        recoverable=True,
                        replan_requested=True,
                    )
                    return

                completed_skills.append(skill_name)

            self._publish_status(
                task_id=msg.task_id,
                state="completed",
                success=True,
                message="task completed",
                completed_skills=completed_skills,
            )
            if self._debug:
                self.get_logger().info(
                    f"[embodied-debug] task_executor completed task_id={msg.task_id} skills={completed_skills}"
                )
        except Exception as exc:
            self.get_logger().error(f"[embodied-debug] task_executor unexpected error task_id={msg.task_id}: {exc}")
            self._publish_status(
                task_id=msg.task_id,
                state="failed",
                success=False,
                message=f"unexpected error: {exc}",
                error_code="INTERNAL_ERROR",
            )
        finally:
            self._active_task_id = ""
            self._active_task_lock.release()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
