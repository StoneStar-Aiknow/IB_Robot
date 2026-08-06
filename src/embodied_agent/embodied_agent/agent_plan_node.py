"""ROS boundary for short-lived Hermes Agent plans."""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Sequence
from contextlib import suppress
from threading import RLock

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from embodied_agent.agent_plan_store import AgentPlan, AgentPlanError, AgentPlanStore
from embodied_agent.command_parser import parse_text_workflow
from embodied_common.command_parser import load_skill_aliases
from embodied_common.dispatch_binding import new_binding, workflow_step
from embodied_common.skill_request import derive_skill_task_id
from embodied_common.workflow_contracts import CanonicalWorkflowStep, compute_workflow_digest
from ibrobot_msgs.action import ExecuteAgentPlan, SkillCommand
from ibrobot_msgs.msg import AgentPlan as AgentPlanMsg
from ibrobot_msgs.msg import SkillDiagnostic
from ibrobot_msgs.srv import (
    BeginWorkflowExecution,
    ConfirmAgentPlan,
    FinalizeWorkflowExecution,
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    PlanAgentCommand,
    ValidateAgentPlan,
    ValidateSkill,
)
from skill_catalog.consumer import CatalogConsumerError, CatalogIdentity, verify_snapshot_response


class _ChildStateUnknown(AgentPlanError):
    """A child may still own or execute the robot; do not close its root scope."""


class _WorkflowStateUnknown(AgentPlanError):
    """A Begin/Finalize request may have committed remotely; preserve local state."""


class AgentPlanNode(Node):
    """Own the Agent plan lifecycle and delegate execution to the Gateway."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("agent_plan_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot")
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("begin_workflow_service", "/embodied/begin_workflow_execution")
        self.declare_parameter("finalize_workflow_service", "/embodied/finalize_workflow_execution")
        self.declare_parameter("plan_service", "/embodied/plan_agent_command")
        self.declare_parameter("validate_plan_service", "/embodied/validate_agent_plan")
        self.declare_parameter("confirm_plan_service", "/embodied/confirm_agent_plan")
        self.declare_parameter("execute_plan_action", "/embodied/execute_agent_plan")
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("plan_ttl_sec", 300.0)
        self.declare_parameter("plan_store_max_records", 1024)
        self.declare_parameter("default_target_name", "demo_object")
        self.declare_parameter("default_place_name", "tray_right")
        self.declare_parameter("default_relative_motion_step_m", 0.03)
        self.declare_parameter("skill_aliases_json", "")

        self._status_service = self._string_parameter("gateway_status_service")
        self._validate_skill_service = self._string_parameter("validate_skill_service")
        self._snapshot_service = self._string_parameter("skill_catalog_snapshot_service")
        self._skill_action_name = self._string_parameter("skill_action_name")
        self._begin_workflow_service = self._string_parameter("begin_workflow_service")
        self._finalize_workflow_service = self._string_parameter("finalize_workflow_service")
        self._rpc_timeout = self.get_parameter("rpc_timeout_sec").get_parameter_value().double_value
        if not math.isfinite(self._rpc_timeout) or self._rpc_timeout <= 0.0:
            raise ValueError("rpc_timeout_sec must be finite and positive")
        self._default_target = self._string_parameter("default_target_name")
        self._default_place = self._string_parameter("default_place_name")
        self._default_relative_motion_step = (
            self.get_parameter("default_relative_motion_step_m").get_parameter_value().double_value
        )
        self._skill_aliases = load_skill_aliases(self._string_parameter("skill_aliases_json"))
        self._store = AgentPlanStore(
            ttl_sec=self.get_parameter("plan_ttl_sec").get_parameter_value().double_value,
            max_records=self.get_parameter("plan_store_max_records").get_parameter_value().integer_value,
        )
        self._store_lock = RLock()

        callback_group = ReentrantCallbackGroup()
        self._status_client = self.create_client(
            GetSkillGatewayStatus, self._status_service, callback_group=callback_group
        )
        self._validate_skill_client = self.create_client(
            ValidateSkill, self._validate_skill_service, callback_group=callback_group
        )
        self._snapshot_client = self.create_client(
            GetSkillSnapshot, self._snapshot_service, callback_group=callback_group
        )
        self._begin_workflow_client = self.create_client(
            BeginWorkflowExecution, self._begin_workflow_service, callback_group=callback_group
        )
        self._finalize_workflow_client = self.create_client(
            FinalizeWorkflowExecution, self._finalize_workflow_service, callback_group=callback_group
        )
        self._skill_client = ActionClient(self, SkillCommand, self._skill_action_name, callback_group=callback_group)
        self._plan_server = self.create_service(
            PlanAgentCommand,
            self._string_parameter("plan_service"),
            self._plan_command,
            callback_group=callback_group,
        )
        self._validate_plan_server = self.create_service(
            ValidateAgentPlan,
            self._string_parameter("validate_plan_service"),
            self._validate_plan,
            callback_group=callback_group,
        )
        self._confirm_plan_server = self.create_service(
            ConfirmAgentPlan,
            self._string_parameter("confirm_plan_service"),
            self._confirm_plan,
            callback_group=callback_group,
        )
        self._execute_plan_server = ActionServer(
            self,
            ExecuteAgentPlan,
            self._string_parameter("execute_plan_action"),
            execute_callback=self._execute_plan,
            goal_callback=self._handle_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )

    def _string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    @staticmethod
    def _handle_goal(goal) -> GoalResponse:
        if (
            goal.schema_version != 1
            or not goal.plan_token.strip()
            or not goal.confirmation_token.strip()
            or not goal.task_id.strip()
            or not math.isfinite(goal.timeout_sec)
            or goal.timeout_sec <= 0.0
        ):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _handle_cancel(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _wait_future(future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done()

    def _call_service(self, client, request, service_name: str):
        if not client.wait_for_service(timeout_sec=self._rpc_timeout):
            raise AgentPlanError("SKILL_SERVER_UNAVAILABLE", f"service unavailable: {service_name}")
        future = client.call_async(request)
        if not self._wait_future(future, self._rpc_timeout):
            raise AgentPlanError("SKILL_SERVER_UNAVAILABLE", f"service timed out: {service_name}")
        try:
            return future.result()
        except Exception as exc:
            raise AgentPlanError("SKILL_SERVER_UNAVAILABLE", f"service failed: {service_name}") from exc

    def _gateway_status(self):
        request = GetSkillGatewayStatus.Request()
        request.schema_version = 1
        response = self._call_service(self._status_client, request, self._status_service)
        if not response.registry_epoch or response.registry_generation <= 0 or not response.registry_digest:
            raise AgentPlanError("SKILL_REGISTRY_NOT_READY")
        return response

    @staticmethod
    def _identity(status) -> tuple[str, int, str]:
        return status.registry_epoch, int(status.registry_generation), status.registry_digest

    def _catalog_view(self, status):
        identity = CatalogIdentity(*self._identity(status))
        request = GetSkillSnapshot.Request()
        request.schema_version = 1
        request.registry_epoch = identity.registry_epoch
        request.generation = identity.generation
        snapshot = self._call_service(self._snapshot_client, request, self._snapshot_service)
        try:
            return verify_snapshot_response(snapshot, identity)
        except CatalogConsumerError as exc:
            raise AgentPlanError(exc.code, str(exc)) from exc

    @staticmethod
    def _diagnostic(code: str, message: str, *, field_path: str = "") -> SkillDiagnostic:
        diagnostic = SkillDiagnostic()
        diagnostic.schema_version = 1
        diagnostic.severity = SkillDiagnostic.ERROR
        diagnostic.error_code = code
        diagnostic.field_path = field_path
        diagnostic.message = message
        return diagnostic

    @staticmethod
    def _copy_diagnostics(diagnostics: Sequence[SkillDiagnostic]) -> list[SkillDiagnostic]:
        return list(diagnostics)

    @staticmethod
    def _float32(value: float) -> float:
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]

    @staticmethod
    def _to_plan_message(plan: AgentPlan) -> AgentPlanMsg:
        message = AgentPlanMsg()
        message.schema_version = 1
        message.plan_id = plan.plan_id
        message.plan_token = plan.plan_token
        message.plan_kind = plan.plan_kind
        message.raw_command = plan.raw_command
        message.workflow_steps = [AgentPlanNode._to_workflow_step_message(step) for step in plan.workflow_steps]
        message.plan_digest = plan.plan_digest
        message.registry_epoch = plan.registry_epoch
        message.registry_generation = plan.registry_generation
        message.registry_digest = plan.registry_digest
        message.expires_at.sec, message.expires_at.nanosec = plan.expires_at
        return message

    @staticmethod
    def _to_workflow_step_message(step: CanonicalWorkflowStep):
        return workflow_step(
            skill_name=step.skill_name,
            target_name=step.target_name,
            place_name=step.place_name,
            motion_direction=step.motion_direction,
            motion_distance=step.motion_distance,
            timeout_sec=step.timeout_sec,
        )

    def _parse_steps(self, raw_command: str, status, catalog) -> tuple[CanonicalWorkflowStep, ...]:
        parsed_steps, message = parse_text_workflow(
            raw_command,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
            skill_aliases={name: list(values) for name, values in catalog.aliases.items()}
            or self._skill_aliases
            or None,
        )
        if not parsed_steps:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", message or "command cannot be planned")
        for parsed in parsed_steps:
            skill_name = parsed.skill_sequence[0]
            capability = catalog.capability_view.get(skill_name)
            if (
                skill_name not in catalog.planner_visible_names
                or capability is None
                or capability.get("semantic_level") not in {"atomic_operator", "skill"}
            ):
                raise AgentPlanError("SKILL_UNKNOWN", f"skill is not planner-visible: {skill_name}")
        return tuple(
            CanonicalWorkflowStep(
                schema_version=1,
                skill_name=parsed.skill_sequence[0],
                target_name=parsed.target_name,
                place_name=parsed.place_name,
                motion_direction=parsed.motion_direction,
                motion_distance=self._float32(parsed.motion_distance),
                timeout_sec=self._float32(status.default_skill_timeout_sec),
            )
            for parsed in parsed_steps
        )

    def _plan_command(self, request, response):
        try:
            if request.schema_version != 1:
                raise AgentPlanError("SKILL_SCHEMA_INVALID", "schema_version must be 1")
            status = self._gateway_status()
            catalog = self._catalog_view(status)
            steps = self._parse_steps(request.raw_command, status, catalog)
            epoch, generation, digest = self._identity(status)
            with self._store_lock:
                plan = self._store.create_plan(
                    request_id=request.request_id,
                    raw_command=request.raw_command,
                    workflow_steps=steps,
                    registry_epoch=epoch,
                    registry_generation=generation,
                    registry_digest=digest,
                )
            response.success = True
            response.plan = self._to_plan_message(plan)
        except AgentPlanError as exc:
            response.success = False
            response.error_code = exc.code
            response.message = str(exc)
            response.diagnostics = [self._diagnostic(exc.code, str(exc))]
        return response

    def _validate_step(self, plan: AgentPlan, step: CanonicalWorkflowStep):
        request = ValidateSkill.Request()
        request.dispatch_binding = new_binding(task_id=plan.plan_id)
        request.dispatch_binding.expected_registry_epoch = plan.registry_epoch
        request.dispatch_binding.expected_registry_generation = plan.registry_generation
        request.dispatch_binding.expected_registry_digest = plan.registry_digest
        request.skill_name = step.skill_name
        request.target_name = step.target_name
        request.place_name = step.place_name
        request.motion_direction = step.motion_direction
        request.motion_distance = step.motion_distance
        return self._call_service(self._validate_skill_client, request, self._validate_skill_service)

    def _validate_plan(self, request, response):
        try:
            if request.schema_version != 1:
                raise AgentPlanError("SKILL_SCHEMA_INVALID", "schema_version must be 1")
            status = self._gateway_status()
            with self._store_lock:
                plan = self._store.validate(
                    plan_token=request.plan_token,
                    registry_epoch=status.registry_epoch,
                    registry_generation=status.registry_generation,
                    registry_digest=status.registry_digest,
                )
            response.plan_id = plan.plan_id
            response.plan_digest = plan.plan_digest
            first_error_code = ""
            first_error_message = ""
            for step in plan.workflow_steps:
                validation = self._validate_step(plan, step)
                response.diagnostics.extend(self._copy_diagnostics(validation.diagnostics))
                actual_identity = (
                    validation.actual_registry_epoch,
                    int(validation.actual_registry_generation),
                    validation.actual_registry_digest,
                )
                expected_identity = (plan.registry_epoch, plan.registry_generation, plan.registry_digest)
                if actual_identity != expected_identity and not first_error_code:
                    first_error_code = "SKILL_REGISTRY_VERSION_MISMATCH"
                    first_error_message = "validation used a different registry identity"
                elif not validation.allowed and not first_error_code:
                    first_error_code = validation.error_code or "SKILL_REJECTED"
                    first_error_message = validation.reason or first_error_code
            if first_error_code:
                response.allowed = False
                response.error_code = first_error_code
                response.message = first_error_message
                return response
            with self._store_lock:
                self._store.mark_validated(
                    plan_token=request.plan_token,
                    registry_epoch=status.registry_epoch,
                    registry_generation=status.registry_generation,
                    registry_digest=status.registry_digest,
                )
            response.allowed = True
            response.message = "allowed"
        except AgentPlanError as exc:
            response.allowed = False
            response.error_code = exc.code
            response.message = str(exc)
            response.diagnostics = [self._diagnostic(exc.code, str(exc))]
        return response

    def _confirm_plan(self, request, response):
        try:
            if request.schema_version != 1:
                raise AgentPlanError("SKILL_SCHEMA_INVALID", "schema_version must be 1")
            status = self._gateway_status()
            if self._identity(status) != (
                request.registry_epoch,
                int(request.registry_generation),
                request.registry_digest,
            ):
                raise AgentPlanError("SKILL_REGISTRY_VERSION_MISMATCH")
            task_budget_sec = self._float32(request.task_budget_sec)
            if task_budget_sec > float(status.task_budget_sec):
                raise AgentPlanError("TIMEOUT_EXCEEDS_POLICY")
            with self._store_lock:
                confirmation = self._store.confirm(
                    plan_token=request.plan_token,
                    plan_digest=request.plan_digest,
                    task_id=request.task_id,
                    registry_epoch=request.registry_epoch,
                    registry_generation=request.registry_generation,
                    registry_digest=request.registry_digest,
                    task_budget_sec=task_budget_sec,
                )
            response.confirmed = confirmation.confirmed
            response.confirmation_token = confirmation.confirmation_token
            response.confirmed_task_budget_sec = confirmation.task_budget_sec
            response.task_budget_started_at.sec, response.task_budget_started_at.nanosec = (
                confirmation.task_budget_started_at
            )
            response.task_budget_deadline.sec, response.task_budget_deadline.nanosec = confirmation.task_budget_deadline
        except AgentPlanError as exc:
            response.confirmed = False
            response.error_code = exc.code
            response.message = str(exc)
            response.diagnostics = [self._diagnostic(exc.code, str(exc))]
        return response

    def _root_binding(self, plan: AgentPlan, task_id: str, execution):
        binding = new_binding(task_id=task_id)
        binding.expected_registry_epoch = plan.registry_epoch
        binding.expected_registry_generation = plan.registry_generation
        binding.expected_registry_digest = plan.registry_digest
        binding.task_budget.schema_version = 1
        binding.task_budget.started_at.sec, binding.task_budget.started_at.nanosec = execution.task_budget_started_at
        binding.task_budget.deadline.sec, binding.task_budget.deadline.nanosec = execution.task_budget_deadline
        return binding

    def _publish_feedback(self, goal_handle, state: str, detail: str, skill_name: str, index: int) -> None:
        feedback = ExecuteAgentPlan.Feedback()
        feedback.state = state
        feedback.detail = detail
        feedback.current_skill = skill_name
        feedback.workflow_step_index = index
        goal_handle.publish_feedback(feedback)

    def _send_skill(self, goal_handle, binding, step: CanonicalWorkflowStep, execution_deadline: float):
        step_timeout = step.timeout_sec if step.timeout_sec > 0.0 else execution_deadline - time.monotonic()
        child_deadline = min(execution_deadline, time.monotonic() + step_timeout)
        wait_timeout = min(self._rpc_timeout, max(0.0, child_deadline - time.monotonic()))
        if wait_timeout <= 0.0 or not self._skill_client.wait_for_server(timeout_sec=wait_timeout):
            raise AgentPlanError("SKILL_SERVER_UNAVAILABLE")
        goal = SkillCommand.Goal()
        goal.dispatch_binding = binding
        goal.skill_name = step.skill_name
        goal.target_name = step.target_name
        goal.place_name = step.place_name
        goal.motion_direction = step.motion_direction
        goal.motion_distance = step.motion_distance
        goal.timeout_sec = step.timeout_sec if step.timeout_sec > 0.0 else step_timeout
        send_future = self._skill_client.send_goal_async(goal)
        send_timeout = min(self._rpc_timeout, max(0.0, child_deadline - time.monotonic()))
        if send_timeout <= 0.0 or not self._wait_future(send_future, send_timeout):
            raise _ChildStateUnknown("SKILL_EXECUTION_STATE_UNKNOWN", "skill goal acceptance state is unknown")
        child_handle = send_future.result()
        if child_handle is None or not child_handle.accepted:
            raise AgentPlanError("SKILL_REJECTED", "skill goal rejected")
        result_future = child_handle.get_result_async()
        deadline = child_deadline

        def cancel_child(error_code: str) -> None:
            try:
                cancel_future = child_handle.cancel_goal_async()
            except Exception as exc:
                raise _ChildStateUnknown("SKILL_CANCEL_TIMEOUT", "child cancellation state is unknown") from exc
            if not self._wait_future(cancel_future, self._rpc_timeout):
                raise _ChildStateUnknown("SKILL_CANCEL_TIMEOUT", "child cancel response is unknown")
            cancel_response = cancel_future.result()
            if cancel_response is None or not getattr(cancel_response, "goals_canceling", []):
                raise _ChildStateUnknown("SKILL_CANCEL_TIMEOUT", "child cancellation was not accepted")
            if not self._wait_future(result_future, self._rpc_timeout):
                raise _ChildStateUnknown("SKILL_CANCEL_TIMEOUT", "child terminal result is unknown")
            action_result = result_future.result()
            if action_result is None or action_result.status != GoalStatus.STATUS_CANCELED:
                raise _ChildStateUnknown("SKILL_CANCEL_TIMEOUT", "child did not confirm a canceled terminal state")
            raise AgentPlanError(error_code)

        while not result_future.done() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                cancel_child("SKILL_CANCELLED")
            time.sleep(0.02)
        if not result_future.done():
            cancel_child("SKILL_TIMEOUT")
        action_result = result_future.result()
        result = action_result.result if action_result is not None else None
        terminal_status = getattr(action_result, "status", GoalStatus.STATUS_UNKNOWN)
        if result is None or terminal_status not in {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        }:
            raise _ChildStateUnknown("SKILL_EXECUTION_STATE_UNKNOWN", "skill terminal result is unknown")
        if not result.success:
            raise AgentPlanError(
                result.error_code if result is not None and result.error_code else "SKILL_EXECUTION_FAILED",
                result.message if result is not None else "skill result missing",
            )
        return result

    def _begin_workflow(self, plan: AgentPlan, binding, steps):
        binding.workflow_digest = compute_workflow_digest(
            root_task_id=binding.root_task_id,
            task_budget=binding.task_budget,
            expected_registry_epoch=plan.registry_epoch,
            expected_registry_generation=plan.registry_generation,
            expected_registry_digest=plan.registry_digest,
            workflow_steps=steps,
        )
        request = BeginWorkflowExecution.Request()
        request.dispatch_binding = binding
        request.workflow_steps = [self._to_workflow_step_message(step) for step in steps]
        try:
            response = self._call_service(self._begin_workflow_client, request, self._begin_workflow_service)
        except AgentPlanError as exc:
            raise _WorkflowStateUnknown("SKILL_EXECUTION_STATE_UNKNOWN", "workflow admission state is unknown") from exc
        if not response.success:
            raise AgentPlanError(response.error_code or "SKILL_REJECTED", response.message)
        if response.workflow_digest != binding.workflow_digest or not response.root_lease_nonce:
            raise AgentPlanError("SKILL_WORKFLOW_DIGEST_MISMATCH")
        return response.root_lease_nonce

    def _finalize_workflow(self, binding, terminal_state: int, completed_step_count: int):
        request = FinalizeWorkflowExecution.Request()
        request.dispatch_binding = binding
        request.terminal_state = terminal_state
        request.completed_step_count = completed_step_count
        try:
            return self._call_service(self._finalize_workflow_client, request, self._finalize_workflow_service)
        except Exception as exc:
            raise _WorkflowStateUnknown("SKILL_CANCEL_TIMEOUT", "workflow finalization state is unknown") from exc

    def _execute_plan(self, goal_handle):
        request = goal_handle.request
        result = ExecuteAgentPlan.Result()
        completed = 0
        plan = None
        binding = None
        workflow_started = False
        try:
            status = self._gateway_status()
            with self._store_lock:
                execution = self._store.accept_execution(
                    plan_token=request.plan_token,
                    confirmation_token=request.confirmation_token,
                    task_id=request.task_id,
                    registry_epoch=status.registry_epoch,
                    registry_generation=status.registry_generation,
                    registry_digest=status.registry_digest,
                    task_budget_sec=float(request.timeout_sec),
                )
            plan = execution.plan
            if execution.state == "TERMINAL":
                result.success = not execution.terminal_code
                result.error_code = execution.terminal_code
                result.message = execution.terminal_message
                result.plan_id = plan.plan_id
                result.plan_digest = plan.plan_digest
                result.workflow_digest = execution.workflow_digest
                result.completed_step_count = execution.completed_step_count
                result.actual_registry_epoch = plan.registry_epoch
                result.actual_registry_generation = plan.registry_generation
                result.actual_registry_digest = plan.registry_digest
                if result.success:
                    goal_handle.succeed()
                else:
                    goal_handle.abort()
                return result
            if not execution.newly_accepted:
                raise AgentPlanError("SKILL_EXECUTION_BUSY", "agent plan execution is already active")
            steps = plan.workflow_steps
            deadline_unix = execution.task_budget_deadline[0] + execution.task_budget_deadline[1] / 1_000_000_000
            remaining_timeout = deadline_unix - time.time()
            if remaining_timeout <= 0.0:
                raise AgentPlanError("SKILL_TIMEOUT", "confirmed task budget expired before execution")
            execution_deadline = time.monotonic() + remaining_timeout
            binding = self._root_binding(plan, request.task_id, execution)
            if len(steps) > 1:
                binding.root_lease_nonce = self._begin_workflow(plan, binding, steps)
                workflow_started = True
            for index, step in enumerate(steps):
                if goal_handle.is_cancel_requested:
                    raise AgentPlanError("SKILL_CANCELLED")
                child_binding = binding
                if workflow_started:
                    child_binding = new_binding(
                        task_id=derive_skill_task_id(request.task_id, index), root_task_id=request.task_id
                    )
                    child_binding.task_budget = binding.task_budget
                    child_binding.expected_registry_epoch = plan.registry_epoch
                    child_binding.expected_registry_generation = plan.registry_generation
                    child_binding.expected_registry_digest = plan.registry_digest
                    child_binding.workflow_digest = binding.workflow_digest
                    child_binding.workflow_step_index = index
                    child_binding.root_lease_nonce = binding.root_lease_nonce
                self._publish_feedback(goal_handle, "executing", f"executing {step.skill_name}", step.skill_name, index)
                self._send_skill(goal_handle, child_binding, step, execution_deadline)
                completed += 1
            if workflow_started:
                finalized = self._finalize_workflow(binding, FinalizeWorkflowExecution.Request.SUCCEEDED, completed)
                if not finalized.success:
                    finalized = self._finalize_workflow(binding, FinalizeWorkflowExecution.Request.SUCCEEDED, completed)
                if not finalized.success:
                    raise _WorkflowStateUnknown(
                        finalized.error_code or "SKILL_FINALIZATION_FAILED",
                        finalized.message or "workflow finalization state is unknown",
                    )
                workflow_started = False
            with self._store_lock:
                self._store.mark_terminal(
                    plan_token=plan.plan_token,
                    task_id=request.task_id,
                    terminal_message="agent plan completed",
                    workflow_digest=binding.workflow_digest,
                    completed_step_count=completed,
                )
            result.success = True
            result.message = "agent plan completed"
            goal_handle.succeed()
        except (_ChildStateUnknown, _WorkflowStateUnknown) as exc:
            # The child may still execute or retain a lease. Keep the plan ACCEPTED and
            # workflow open; finalizing or caching a terminal result would be unsafe.
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.abort()
        except AgentPlanError as exc:
            finalization_converged = not workflow_started
            if workflow_started and binding is not None:
                terminal = (
                    FinalizeWorkflowExecution.Request.CANCELED
                    if exc.code == "SKILL_CANCELLED"
                    else FinalizeWorkflowExecution.Request.FAILED
                )
                try:
                    finalized = self._finalize_workflow(binding, terminal, completed)
                    if not finalized.success:
                        finalized = self._finalize_workflow(binding, terminal, completed)
                    finalization_converged = bool(finalized.success)
                except AgentPlanError:
                    finalization_converged = False
            if not finalization_converged:
                exc = AgentPlanError("SKILL_CANCEL_TIMEOUT", "workflow finalization state is unknown")
            if plan is not None and finalization_converged:
                with self._store_lock, suppress(AgentPlanError):
                    self._store.mark_terminal(
                        plan_token=plan.plan_token,
                        task_id=request.task_id,
                        terminal_code=exc.code,
                        terminal_message=str(exc),
                        workflow_digest=binding.workflow_digest if binding is not None else "",
                        completed_step_count=completed,
                    )
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            if exc.code == "SKILL_CANCELLED":
                goal_handle.canceled()
            else:
                goal_handle.abort()
        if plan is not None:
            result.plan_id = plan.plan_id
            result.plan_digest = plan.plan_digest
            result.actual_registry_epoch = plan.registry_epoch
            result.actual_registry_generation = plan.registry_generation
            result.actual_registry_digest = plan.registry_digest
        result.workflow_digest = binding.workflow_digest if binding is not None else ""
        result.completed_step_count = completed
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentPlanNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
