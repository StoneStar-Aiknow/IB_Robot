"""Task executor node for the embodied minimal closure."""

import threading
import time
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient

from embodied_agent.base_node import BaseTaskNode
from embodied_agent.task_context import TIMEOUT_CONTEXT_KEY, load_task_context, remaining_task_budget_sec
from embodied_common.dispatch_binding import binding_task_id, copy_binding
from embodied_common.skill_request import derive_skill_task_id
from embodied_common.workflow_contracts import compute_workflow_digest
from embodied_common.workflow_lifecycle import WorkflowLifecycleClient, WorkflowLifecycleError
from ibrobot_msgs.action import SkillCommand
from ibrobot_msgs.msg import TaskCommand, TaskStatus
from ibrobot_msgs.srv import BeginWorkflowExecution, FinalizeWorkflowExecution, GetSkillGatewayStatus


class _WorkflowStateUnknown(RuntimeError):
    """A workflow control request may have committed remotely."""


__all__ = ["TaskExecutorNode", "derive_skill_task_id"]


class TaskExecutorNode(BaseTaskNode):
    """Execute a planned task by calling the skill action server in sequence."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("task_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/embodied/planned_task")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("begin_workflow_service", "/embodied/begin_workflow_execution")
        self.declare_parameter("finalize_workflow_service", "/embodied/finalize_workflow_execution")
        self.declare_parameter("default_task_timeout_sec", 180.0)
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("debug_tracing", False)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._skill_action_name = self.get_parameter("skill_action_name").get_parameter_value().string_value
        self._gateway_status_service = self.get_parameter("gateway_status_service").get_parameter_value().string_value
        self._begin_workflow_service = self.get_parameter("begin_workflow_service").get_parameter_value().string_value
        self._finalize_workflow_service = (
            self.get_parameter("finalize_workflow_service").get_parameter_value().string_value
        )
        self._default_timeout = self.get_parameter("default_task_timeout_sec").get_parameter_value().double_value
        self._rpc_timeout = self.get_parameter("rpc_timeout_sec").get_parameter_value().double_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._active_task_lock = threading.Lock()
        self._active_task_id = ""

        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self._skill_client = ActionClient(self, SkillCommand, self._skill_action_name)
        self._gateway_status_client = self.create_client(GetSkillGatewayStatus, self._gateway_status_service)
        self._begin_workflow_client = self.create_client(BeginWorkflowExecution, self._begin_workflow_service)
        self._finalize_workflow_client = self.create_client(FinalizeWorkflowExecution, self._finalize_workflow_service)
        self._workflow_lifecycle = WorkflowLifecycleClient(
            self._call_service,
            self._begin_workflow_client,
            self._begin_workflow_service,
            self._finalize_workflow_client,
            self._finalize_workflow_service,
        )
        self.create_subscription(TaskCommand, self._input_topic, self._handle_planned_task, 10)

        self.get_logger().info(
            f"[embodied-debug] task_executor ready: input_topic={self._input_topic}, "
            f"skill_action={self._skill_action_name}"
        )

    def _handle_planned_task(self, msg: TaskCommand) -> None:
        if not self._active_task_lock.acquire(blocking=False):
            self._publish_status(
                task_id=binding_task_id(msg),
                state="rejected",
                success=False,
                message="executor busy",
                error_code="EXECUTOR_BUSY",
                recoverable=True,
                replan_requested=True,
            )
            self.get_logger().warning(f"[embodied-debug] task_executor busy, rejecting task_id={binding_task_id(msg)}")
            return

        self._active_task_id = binding_task_id(msg)
        worker = threading.Thread(target=self._execute_task, args=(msg,), daemon=True)
        worker.start()

    def _wait_for_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        return future.done()

    def _resolve_task_context(self, msg: TaskCommand) -> dict[str, Any]:
        context = load_task_context(msg.context_json)
        budget = msg.dispatch_binding.task_budget
        started = budget.started_at.sec + budget.started_at.nanosec / 1_000_000_000
        deadline = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        if (
            budget.schema_version != 1
            or budget.started_at.sec < 0
            or budget.deadline.sec < 0
            or not 0 <= budget.started_at.nanosec < 1_000_000_000
            or not 0 <= budget.deadline.nanosec < 1_000_000_000
            or deadline <= started
        ):
            raise ValueError("planned task carries an invalid TaskBudget")
        context[TIMEOUT_CONTEXT_KEY] = {
            "task_timeout_sec": deadline - started,
            "created_at_unix_sec": started,
            "deadline_unix_sec": deadline,
        }
        return context

    def _remaining_budget_sec(self, context: dict[str, Any]) -> float | None:
        return remaining_task_budget_sec(context)

    def _cancel_goal(self, goal_handle, result_future) -> bool:
        if goal_handle is None:
            return False
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            return False
        if not self._wait_for_future(cancel_future, timeout_sec=self._rpc_timeout):
            return False
        response = cancel_future.result()
        if response is None or not getattr(response, "goals_canceling", []):
            return False
        if not self._wait_for_future(result_future, timeout_sec=self._rpc_timeout):
            return False
        action_result = result_future.result()
        return action_result is not None and getattr(action_result, "status", None) == GoalStatus.STATUS_CANCELED

    def _call_service(self, client, request, service_name: str):
        if not client.wait_for_service(timeout_sec=self._rpc_timeout):
            raise RuntimeError(f"service unavailable: {service_name}")
        future = client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=self._rpc_timeout):
            raise RuntimeError(f"service timed out: {service_name}")
        response = future.result()
        if response is None:
            raise RuntimeError(f"service returned no response: {service_name}")
        return response

    @staticmethod
    def _set_time(message, value: float) -> None:
        seconds = int(value)
        nanoseconds = int(round((value - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        message.sec = seconds
        message.nanosec = nanoseconds

    def _begin_workflow(self, msg: TaskCommand, workflow_steps, plan_context):
        status_request = GetSkillGatewayStatus.Request()
        status_request.schema_version = 1
        status = self._call_service(self._gateway_status_client, status_request, self._gateway_status_service)
        if not status.registry_epoch or status.registry_generation <= 0 or not status.registry_digest:
            raise RuntimeError("Gateway registry is not ready")
        planned_identity = (
            msg.dispatch_binding.expected_registry_epoch,
            int(msg.dispatch_binding.expected_registry_generation),
            msg.dispatch_binding.expected_registry_digest,
        )
        if not all(planned_identity):
            raise RuntimeError("planned task is missing exact registry identity")
        if planned_identity != (status.registry_epoch, int(status.registry_generation), status.registry_digest):
            raise RuntimeError("SKILL_REGISTRY_VERSION_MISMATCH")
        timeout_context = plan_context.get(TIMEOUT_CONTEXT_KEY, {})
        if not isinstance(timeout_context, dict):
            raise RuntimeError("task timeout context is invalid")
        supplied_budget = msg.dispatch_binding.task_budget
        if supplied_budget.schema_version == 1:
            started = supplied_budget.started_at.sec + supplied_budget.started_at.nanosec / 1_000_000_000
            deadline = supplied_budget.deadline.sec + supplied_budget.deadline.nanosec / 1_000_000_000
        else:
            started = float(timeout_context.get("created_at_unix_sec", time.time()))
            deadline = float(timeout_context.get("deadline_unix_sec", started + float(msg.timeout_sec)))
        if deadline <= time.time():
            raise RuntimeError("task deadline exceeded before workflow admission")

        binding = copy_binding(msg.dispatch_binding)
        root_task_id = binding_task_id(msg)
        binding.schema_version = 1
        binding.task_id = root_task_id
        binding.root_task_id = root_task_id
        binding.workflow_step_index = 0
        binding.root_lease_nonce = ""
        binding.dispatch_nonce = ""
        if binding.task_budget.schema_version != 1:
            binding.task_budget.schema_version = 1
            self._set_time(binding.task_budget.started_at, started)
            self._set_time(binding.task_budget.deadline, deadline)
        expected_workflow_digest = compute_workflow_digest(
            root_task_id=root_task_id,
            task_budget=binding.task_budget,
            expected_registry_epoch=planned_identity[0],
            expected_registry_generation=planned_identity[1],
            expected_registry_digest=planned_identity[2],
            workflow_steps=workflow_steps,
        )
        if binding.workflow_digest and binding.workflow_digest != expected_workflow_digest:
            raise RuntimeError("SKILL_WORKFLOW_DIGEST_MISMATCH")
        binding.workflow_digest = expected_workflow_digest
        try:
            lifecycle = getattr(self, "_workflow_lifecycle", None) or WorkflowLifecycleClient(
                self._call_service,
                self._begin_workflow_client,
                self._begin_workflow_service,
                self._finalize_workflow_client,
                self._finalize_workflow_service,
            )
            return lifecycle.begin(binding, workflow_steps)
        except WorkflowLifecycleError as exc:
            if exc.code == "SKILL_CANCEL_TIMEOUT":
                raise _WorkflowStateUnknown(str(exc)) from exc
            raise RuntimeError(f"{exc.code}: {exc}") from exc

    def _finalize_workflow(self, binding, terminal_state: int, completed_step_count: int) -> bool:
        try:
            lifecycle = getattr(self, "_workflow_lifecycle", None) or WorkflowLifecycleClient(
                self._call_service,
                self._begin_workflow_client,
                self._begin_workflow_service,
                self._finalize_workflow_client,
                self._finalize_workflow_service,
            )
            response = lifecycle.finalize(binding, terminal_state, completed_step_count)
            if not response.success and response.error_code != "GATEWAY_FINALIZATION_FAILED":
                raise RuntimeError(f"{response.error_code}: {response.message}")
            return bool(response.success)
        except WorkflowLifecycleError as exc:
            if exc.code == "SKILL_CANCEL_TIMEOUT":
                raise _WorkflowStateUnknown(str(exc)) from exc
            raise RuntimeError(f"{exc.code}: {exc}") from exc

    def _execute_task(self, msg: TaskCommand) -> None:
        completed_skills: list[str] = []
        workflow_binding = None
        child_state_unknown = False
        failure_terminal_state = FinalizeWorkflowExecution.Request.FAILED
        try:
            try:
                plan_context = self._resolve_task_context(msg)
            except (ValueError, TypeError):
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="failed",
                    success=False,
                    message="planned task budget is invalid",
                    error_code="SKILL_SCHEMA_INVALID",
                )
                return

            workflow_steps = tuple(msg.workflow_steps)
            if not workflow_steps:
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="failed",
                    success=False,
                    message="planned task missing skill sequence",
                    error_code="EMPTY_PLAN",
                    replan_requested=True,
                )
                return

            try:
                derive_skill_task_id(binding_task_id(msg), 0)
            except (TypeError, ValueError):
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="failed",
                    success=False,
                    message="invalid task ID",
                    error_code="INVALID_TASK_ID",
                )
                return

            initial_budget = self._remaining_budget_sec(plan_context)
            if initial_budget is not None and initial_budget <= 0.0:
                self._publish_status(
                    task_id=binding_task_id(msg),
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
                    task_id=binding_task_id(msg),
                    state="failed",
                    success=False,
                    message="skill action server not available",
                    error_code="CAPABILITY_NOT_READY",
                    recoverable=True,
                    replan_requested=True,
                )
                return
            try:
                workflow_binding = self._begin_workflow(msg, workflow_steps, plan_context)
            except _WorkflowStateUnknown as exc:
                child_state_unknown = True
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="unknown",
                    success=False,
                    message=str(exc),
                    error_code="CHILD_STATE_UNKNOWN",
                )
                return
            except (RuntimeError, TypeError, ValueError) as exc:
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="failed",
                    success=False,
                    message=str(exc),
                    error_code="WORKFLOW_ADMISSION_FAILED",
                    recoverable=True,
                    replan_requested=True,
                )
                return

            for skill_index, step in enumerate(workflow_steps):
                skill_name = step.skill_name
                child_task_id = derive_skill_task_id(binding_task_id(msg), skill_index)
                remaining_budget = self._remaining_budget_sec(plan_context)
                if remaining_budget is not None and remaining_budget <= 0.0:
                    self._publish_status(
                        task_id=binding_task_id(msg),
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
                    task_id=binding_task_id(msg),
                    state="executing",
                    success=True,
                    message=f"executing skill {skill_name}",
                    current_skill=skill_name,
                    completed_skills=completed_skills,
                )
                if self._debug:
                    self.get_logger().info(
                        f"[embodied-debug] task_executor dispatch parent_task_id={binding_task_id(msg)} "
                        f"child_task_id={child_task_id} skill={skill_name}"
                    )

                goal = SkillCommand.Goal()
                goal.dispatch_binding = copy_binding(workflow_binding)
                goal.dispatch_binding.task_id = child_task_id
                goal.dispatch_binding.root_task_id = binding_task_id(msg)
                goal.dispatch_binding.workflow_step_index = skill_index
                goal.skill_name = skill_name
                goal.target_name = step.target_name
                goal.container_name = step.container_name
                goal.place_name = step.place_name
                goal.motion_direction = step.motion_direction
                goal.motion_distance = step.motion_distance
                goal.timeout_sec = float(step.timeout_sec or msg.timeout_sec or self._default_timeout)

                send_goal_future = self._skill_client.send_goal_async(goal)
                send_timeout = min(self._rpc_timeout, goal.timeout_sec)
                if not self._wait_for_future(send_goal_future, timeout_sec=max(0.1, send_timeout)):
                    child_state_unknown = True
                    self._publish_status(
                        task_id=binding_task_id(msg),
                        state="unknown",
                        success=False,
                        message=f"skill goal acceptance state unknown: {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="CHILD_STATE_UNKNOWN",
                    )
                    return

                goal_handle = send_goal_future.result()
                if goal_handle is None:
                    child_state_unknown = True
                    self._publish_status(
                        task_id=binding_task_id(msg),
                        state="unknown",
                        success=False,
                        message=f"skill goal acceptance state unknown: {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="CHILD_STATE_UNKNOWN",
                    )
                    return
                if not goal_handle.accepted:
                    self._publish_status(
                        task_id=binding_task_id(msg),
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
                    if not self._cancel_goal(goal_handle, result_future):
                        child_state_unknown = True
                        self._publish_status(
                            task_id=binding_task_id(msg),
                            state="unknown",
                            success=False,
                            message=f"skill cancellation state unknown: {skill_name}",
                            current_skill=skill_name,
                            completed_skills=completed_skills,
                            error_code="CHILD_STATE_UNKNOWN",
                        )
                        return
                    failure_terminal_state = FinalizeWorkflowExecution.Request.CANCELED
                    self._publish_status(
                        task_id=binding_task_id(msg),
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
                terminal_status = getattr(action_result, "status", GoalStatus.STATUS_UNKNOWN)
                if result is None or terminal_status not in {
                    GoalStatus.STATUS_SUCCEEDED,
                    GoalStatus.STATUS_CANCELED,
                    GoalStatus.STATUS_ABORTED,
                }:
                    child_state_unknown = True
                    self._publish_status(
                        task_id=binding_task_id(msg),
                        state="unknown",
                        success=False,
                        message=f"skill terminal result unknown: {skill_name}",
                        current_skill=skill_name,
                        completed_skills=completed_skills,
                        error_code="CHILD_STATE_UNKNOWN",
                    )
                    return
                if not result.success:
                    if terminal_status == GoalStatus.STATUS_CANCELED:
                        failure_terminal_state = FinalizeWorkflowExecution.Request.CANCELED
                    self._publish_status(
                        task_id=binding_task_id(msg),
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

            try:
                finalized = self._finalize_workflow(
                    workflow_binding,
                    FinalizeWorkflowExecution.Request.SUCCEEDED,
                    len(completed_skills),
                )
            except _WorkflowStateUnknown as exc:
                child_state_unknown = True
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="unknown",
                    success=False,
                    message=str(exc),
                    completed_skills=completed_skills,
                    error_code="CHILD_STATE_UNKNOWN",
                )
                return
            if not finalized:
                try:
                    finalized = self._finalize_workflow(
                        workflow_binding,
                        FinalizeWorkflowExecution.Request.SUCCEEDED,
                        len(completed_skills),
                    )
                except _WorkflowStateUnknown:
                    finalized = False
            if not finalized:
                child_state_unknown = True
                self._publish_status(
                    task_id=binding_task_id(msg),
                    state="unknown",
                    success=False,
                    message="workflow finalization state is unknown",
                    completed_skills=completed_skills,
                    error_code="CHILD_STATE_UNKNOWN",
                )
                return
            workflow_binding = None

            self._publish_status(
                task_id=binding_task_id(msg),
                state="completed",
                success=True,
                message="task completed",
                completed_skills=completed_skills,
            )
            if self._debug:
                self.get_logger().info(
                    f"[embodied-debug] task_executor completed task_id={binding_task_id(msg)} skills={completed_skills}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"[embodied-debug] task_executor unexpected error task_id={binding_task_id(msg)}: {exc}"
            )
            self._publish_status(
                task_id=binding_task_id(msg),
                state="failed",
                success=False,
                message=f"unexpected error: {exc}",
                error_code="INTERNAL_ERROR",
            )
        finally:
            if workflow_binding is not None and not child_state_unknown:
                try:
                    finalized = self._finalize_workflow(
                        workflow_binding,
                        failure_terminal_state,
                        len(completed_skills),
                    )
                except _WorkflowStateUnknown:
                    finalized = False
                if not finalized:
                    try:
                        finalized = self._finalize_workflow(
                            workflow_binding,
                            failure_terminal_state,
                            len(completed_skills),
                        )
                    except _WorkflowStateUnknown:
                        finalized = False
                if not finalized:
                    self.get_logger().error("workflow failure finalization did not converge")
                    self._publish_status(
                        task_id=binding_task_id(msg),
                        state="unknown",
                        success=False,
                        message="workflow finalization state is unknown",
                        completed_skills=completed_skills,
                        error_code="GATEWAY_FINALIZATION_FAILED",
                        recoverable=False,
                        replan_requested=False,
                    )
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
