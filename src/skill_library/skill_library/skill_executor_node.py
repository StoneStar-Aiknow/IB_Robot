"""Skill and primitive execution node for the embodied minimal closure."""

import copy
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unique_identifier_msgs.msg import UUID

from embodied_common.capability_view import build_capability_view
from embodied_common.skill_templates import get_skill_templates
from ibrobot_msgs.action import ExecuteTaskPlan, PickObject, PrimitiveCommand, SkillCommand
from ibrobot_msgs.msg import SkillCapabilityStatus, TaskStep
from ibrobot_msgs.srv import GetSkillGatewayStatus, MoveToConfiguration, ValidatePrimitive, ValidateSkill
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from skill_library.gateway_policy import (
    GATEWAY_FINALIZATION_FAILED,
    BoundedRequestLedger,
    ExecutionOwner,
    GatewayPolicy,
    GatewayRequest,
    RootExecutionLease,
    RuntimeSnapshot,
    build_skill_parameter_schemas,
    build_skill_requirements,
)
from skill_library.resolver import PrimitiveSpec, load_json_mapping, resolve_skill_primitives

EE_POSITION_TOLERANCE_M = 0.02
SKILL_CANCEL_TIMEOUT = "SKILL_CANCEL_TIMEOUT"
PRIMITIVE_CANCEL_CLEANUP_TIMEOUT = "CANCEL_CLEANUP_TIMEOUT"


@dataclass
class _RetainedAdmissionCleanup:
    admission: object
    audit_context: dict[str, str]
    duration_sec: float = 0.0
    step_count: int = 0
    retained: bool = False
    parent_terminal: bool = False
    pending_goal_key: bytes | None = None
    confirmed_goal_key: bytes | None = None
    finalizing: bool = False


@dataclass(frozen=True)
class _EePoseSnapshot:
    pose: PoseStamped
    received_monotonic: float


class _EePoseSnapshotExpired(Exception):
    pass


class _LateCleanupConfirmation:
    """Confirm one downstream cleanup terminal state and notify listeners once."""

    def __init__(self) -> None:
        self._confirmed = False
        self._callbacks: list[Callable[[], None]] = []
        self._lock = RLock()

    def add_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if not self._confirmed:
                self._callbacks.append(callback)
                return
        callback()

    def confirm(self) -> bool:
        with self._lock:
            if self._confirmed:
                return False
            self._confirmed = True
            callbacks = self._callbacks
            self._callbacks = []
        for callback in callbacks:
            callback()
        return True

    def watch_result_future(self, result_future) -> None:
        def confirm_terminal(future) -> None:
            try:
                wrapped_result = future.result()
            except Exception:
                return
            result = getattr(wrapped_result, "result", wrapped_result)
            if getattr(result, "error_code", "") != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                self.confirm()

        result_future.add_done_callback(confirm_terminal)

    def watch_goal_future(self, goal_future, cancel_goal: Callable[[object], None]) -> None:
        def confirm_late_goal(future) -> None:
            try:
                goal_handle = future.result()
            except Exception:
                return
            if goal_handle is None or not goal_handle.accepted:
                self.confirm()
                return
            cancel_goal(goal_handle)
            try:
                self.watch_result_future(goal_handle.get_result_async())
            except Exception:
                return

        goal_future.add_done_callback(confirm_late_goal)


class _DeferredTerminalGoalHandle:
    """Proxy child terminal requests until Gateway bookkeeping is complete."""

    def __init__(self, goal_handle) -> None:
        self._goal_handle = goal_handle
        self._terminal_intent = ""
        self._committed = False
        self._lock = RLock()
        self.late_cleanup_confirmation: _LateCleanupConfirmation | None = None

    @property
    def terminal_intent(self) -> str:
        return self._terminal_intent

    def _record_terminal_intent(self, intent: str) -> None:
        with self._lock:
            if not self._terminal_intent:
                self._terminal_intent = intent

    def succeed(self) -> None:
        self._record_terminal_intent("succeeded")

    def abort(self) -> None:
        self._record_terminal_intent("aborted")

    def canceled(self) -> None:
        self._record_terminal_intent("canceled")

    def commit(self) -> bool:
        """Commit the recorded child terminal state exactly once."""
        with self._lock:
            if self._committed:
                return False
            self._committed = True
            if self._terminal_intent == "succeeded":
                self._goal_handle.succeed()
            elif self._terminal_intent == "canceled":
                self._goal_handle.canceled()
            else:
                self._goal_handle.abort()
            return True

    def force_abort(self) -> bool:
        """Abort the real action exactly once when bookkeeping cannot complete."""
        with self._lock:
            if self._committed:
                return False
            self._committed = True
            self._goal_handle.abort()
            return True

    def __getattr__(self, name):
        return getattr(self._goal_handle, name)


def _duration_from_seconds(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _build_joint_trajectory_goal(
    joint_names: list[str],
    joint_waypoints: list[list[float]],
    waypoint_duration_sec: float,
) -> FollowJointTrajectory.Goal:
    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = list(joint_names)

    for index, waypoint in enumerate(joint_waypoints, start=1):
        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in waypoint]
        point.time_from_start = _duration_from_seconds(float(waypoint_duration_sec) * index)
        goal_msg.trajectory.points.append(point)
    return goal_msg


def _load_json_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    parsed = load_json_mapping(f'{{"items": {raw_value}}}').get("items", [])
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in parsed]


class SkillExecutorNode(Node):
    """Expose skill and primitive actions with explicit safety validation."""

    def __init__(self, parameter_overrides=None, node_name: str | None = None) -> None:
        super().__init__(node_name or "skill_executor_node", parameter_overrides=parameter_overrides)
        startup_descriptor = ParameterDescriptor(read_only=True)
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("validate_primitive_service", "/embodied/validate_primitive")
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("skill_templates_json", "")
        self.declare_parameter("relative_motion_reference_frame", "base")
        self.declare_parameter("relative_motion_direction_mapping_json", "{}")
        self.declare_parameter("rpc_timeout_sec", 5.0, descriptor=startup_descriptor)
        self.declare_parameter("gripper_settle_sec", 1.5, descriptor=startup_descriptor)
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("arm_trajectory_action_name", "/arm_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("task_executor_action_name", "/task_executor/execute_task_plan")
        self.declare_parameter("pick_action_name", "/manipulation/execute_pick")
        self.declare_parameter("move_configuration_service", "/moveit_gateway/move_to_configuration")
        self.declare_parameter("ee_pose_topic", "/robot_status/ee_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cmd_pose_topic", "/cmd_pose")
        self.declare_parameter("debug_tracing", False)
        self.declare_parameter("motion_authorized", False, descriptor=startup_descriptor)
        self.declare_parameter("active_control_mode", "", descriptor=startup_descriptor)
        self.declare_parameter("skill_required_control_mode", "", descriptor=startup_descriptor)
        self.declare_parameter(
            "skill_gateway_status_service",
            "/embodied/get_skill_gateway_status",
            descriptor=startup_descriptor,
        )
        self.declare_parameter("robot_name", "unknown", descriptor=startup_descriptor)
        self.declare_parameter("default_skill_timeout_sec", 30.0, descriptor=startup_descriptor)
        self.declare_parameter("task_budget_sec", 180.0, descriptor=startup_descriptor)
        self.declare_parameter("robot_state_freshness_sec", 0.5, descriptor=startup_descriptor)
        self.declare_parameter("scene_freshness_sec", 0.5, descriptor=startup_descriptor)
        self.declare_parameter("model_idle_timeout_sec", 120.0, descriptor=startup_descriptor)
        self.declare_parameter("config_digest", "", descriptor=startup_descriptor)
        self.declare_parameter("ledger_terminal_capacity", 100, descriptor=startup_descriptor)

        self._skill_action_name = self.get_parameter("skill_action_name").get_parameter_value().string_value
        self._primitive_action_name = self.get_parameter("primitive_action_name").get_parameter_value().string_value
        self._validate_skill_service = self.get_parameter("validate_skill_service").get_parameter_value().string_value
        self._validate_primitive_service = (
            self.get_parameter("validate_primitive_service").get_parameter_value().string_value
        )
        self._named_poses = load_json_mapping(self.get_parameter("named_poses_json").get_parameter_value().string_value)
        self._named_targets = load_json_mapping(
            self.get_parameter("named_targets_json").get_parameter_value().string_value
        )
        raw_skill_templates_json = self.get_parameter("skill_templates_json").get_parameter_value().string_value
        raw_skill_templates = load_json_mapping(raw_skill_templates_json) if raw_skill_templates_json.strip() else {}
        self._skill_templates = get_skill_templates(raw_skill_templates)
        self._relative_motion_reference_frame = (
            self.get_parameter("relative_motion_reference_frame").get_parameter_value().string_value
        )
        self._relative_motion_direction_mapping = load_json_mapping(
            self.get_parameter("relative_motion_direction_mapping_json").get_parameter_value().string_value
        )
        self._rpc_timeout = self.get_parameter("rpc_timeout_sec").get_parameter_value().double_value
        self._gripper_settle_sec = self.get_parameter("gripper_settle_sec").get_parameter_value().double_value
        self._gripper_open = self.get_parameter("gripper_open_position").get_parameter_value().double_value
        self._gripper_closed = self.get_parameter("gripper_closed_position").get_parameter_value().double_value
        self._arm_joint_names = _load_json_list(
            self.get_parameter("arm_joint_names_json").get_parameter_value().string_value
        )
        self._joint_limits = load_json_mapping(
            self.get_parameter("joint_limits_json").get_parameter_value().string_value
        )
        self._arm_trajectory_action_name = (
            self.get_parameter("arm_trajectory_action_name").get_parameter_value().string_value
        )
        self._task_executor_action_name = (
            self.get_parameter("task_executor_action_name").get_parameter_value().string_value
        )
        self._pick_action_name = self.get_parameter("pick_action_name").get_parameter_value().string_value
        self._move_configuration_service = (
            self.get_parameter("move_configuration_service").get_parameter_value().string_value
        )
        self._ee_pose_topic = self.get_parameter("ee_pose_topic").get_parameter_value().string_value
        self._joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self._cmd_pose_topic = self.get_parameter("cmd_pose_topic").get_parameter_value().string_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._motion_authorized = self.get_parameter("motion_authorized").get_parameter_value().bool_value
        self._active_control_mode = self.get_parameter("active_control_mode").get_parameter_value().string_value
        self._skill_required_control_mode = (
            self.get_parameter("skill_required_control_mode").get_parameter_value().string_value
        )
        self._skill_gateway_status_service = (
            self.get_parameter("skill_gateway_status_service").get_parameter_value().string_value
        )
        self._robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self._default_skill_timeout_sec = (
            self.get_parameter("default_skill_timeout_sec").get_parameter_value().double_value
        )
        self._task_budget_sec = self.get_parameter("task_budget_sec").get_parameter_value().double_value
        self._robot_state_freshness_sec = (
            self.get_parameter("robot_state_freshness_sec").get_parameter_value().double_value
        )
        self._scene_freshness_sec = self.get_parameter("scene_freshness_sec").get_parameter_value().double_value
        self._model_idle_timeout_sec = self.get_parameter("model_idle_timeout_sec").get_parameter_value().double_value
        self._ledger_terminal_capacity = (
            self.get_parameter("ledger_terminal_capacity").get_parameter_value().integer_value
        )
        self._skill_requirements = build_skill_requirements(self._skill_templates)
        self._skill_parameter_schemas = build_skill_parameter_schemas(self._skill_templates)
        self._normalized_gateway_config = {
            "name": self._robot_name,
            "embodied": {
                "named_poses": self._named_poses,
                "named_targets": self._named_targets,
                "skill_templates": self._skill_templates,
                "timeouts": {
                    "default_skill_timeout_sec": self._default_skill_timeout_sec,
                    "task_budget_sec": self._task_budget_sec,
                    "robot_state_freshness_sec": self._robot_state_freshness_sec,
                    "scene_freshness_sec": self._scene_freshness_sec,
                    "model_idle_timeout_sec": self._model_idle_timeout_sec,
                    "rpc_timeout_sec": self._rpc_timeout,
                    "gripper_settle_sec": self._gripper_settle_sec,
                },
            },
        }
        self._gateway_timeout_policy = resolve_embodied_timeout_policy(self._normalized_gateway_config["embodied"])
        self._default_skill_timeout_sec = self._gateway_timeout_policy["default_skill_timeout_sec"]
        self._task_budget_sec = self._gateway_timeout_policy["task_budget_sec"]
        self._robot_state_freshness_sec = self._gateway_timeout_policy["robot_state_freshness_sec"]
        self._scene_freshness_sec = self._gateway_timeout_policy["scene_freshness_sec"]
        self._model_idle_timeout_sec = self._gateway_timeout_policy["model_idle_timeout_sec"]
        self._rpc_timeout = self._gateway_timeout_policy["rpc_timeout_sec"]
        self._gripper_settle_sec = self._gateway_timeout_policy["gripper_settle_sec"]
        self._normalized_gateway_config["embodied"]["timeouts"] = dict(self._gateway_timeout_policy)
        self._gateway_ledger = BoundedRequestLedger(self._ledger_terminal_capacity)
        self._gateway_lease = RootExecutionLease()
        self._gateway_policy = GatewayPolicy(
            self._gateway_timeout_policy,
            self._skill_requirements,
            parameter_schemas=self._skill_parameter_schemas,
            ledger=self._gateway_ledger,
            lease=self._gateway_lease,
        )
        self._state_lock = RLock()
        self._active_skill_admission = None
        self._active_skill_owner = None
        self._active_audit_context = None
        self._pending_internal_primitive_goals = {}
        self._active_internal_primitive_goals = {}
        self._internal_pick_handoffs = {}
        self._retained_admission_cleanup = {}
        self._capability_view = build_capability_view(
            self._normalized_gateway_config,
            timeout_policy=self._gateway_timeout_policy,
        )
        computed_digest = self._capability_view["capability_digest"]
        configured_digest = self.get_parameter("config_digest").get_parameter_value().string_value
        if configured_digest and configured_digest != computed_digest:
            raise ValueError("config_digest must match the capability view digest")
        self._config_digest = computed_digest
        self._latest_ee_pose = None
        self._latest_ee_pose_monotonic = None
        self._latest_joint_state = None
        self._skill_goal_lock = threading.Lock()
        self._skill_goal_active = False

        callback_group = ReentrantCallbackGroup()
        self._pose_publisher = self.create_publisher(Pose, self._cmd_pose_topic, 10)
        self._task_executor_client = ActionClient(
            self, ExecuteTaskPlan, self._task_executor_action_name, callback_group=callback_group
        )
        self._arm_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._arm_trajectory_action_name,
            callback_group=callback_group,
        )
        self.create_subscription(
            PoseStamped,
            self._ee_pose_topic,
            self._handle_ee_pose,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._handle_joint_state,
            10,
            callback_group=callback_group,
        )
        self._validate_skill_client = self.create_client(
            ValidateSkill, self._validate_skill_service, callback_group=callback_group
        )
        self._validate_primitive_client = self.create_client(
            ValidatePrimitive, self._validate_primitive_service, callback_group=callback_group
        )
        self._primitive_client = ActionClient(
            self, PrimitiveCommand, self._primitive_action_name, callback_group=callback_group
        )
        self._pick_client = ActionClient(self, PickObject, self._pick_action_name, callback_group=callback_group)
        self._move_configuration_client = self.create_client(
            MoveToConfiguration,
            self._move_configuration_service,
            callback_group=callback_group,
        )
        self._gateway_status_server = self.create_service(
            GetSkillGatewayStatus,
            self._skill_gateway_status_service,
            self._get_gateway_status,
            callback_group=callback_group,
        )
        self._skill_server = ActionServer(
            self,
            SkillCommand,
            self._skill_action_name,
            execute_callback=self._execute_skill,
            goal_callback=self._handle_skill_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self._primitive_server = ActionServer(
            self,
            PrimitiveCommand,
            self._primitive_action_name,
            execute_callback=self._execute_primitive,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )

        self.get_logger().info(
            "[embodied-debug] skill_executor ready: "
            f"skill_action={self._skill_action_name}, primitive_action={self._primitive_action_name}, "
            f"relative_frame={self._relative_motion_reference_frame}, "
            f"direction_mapping={self._relative_motion_direction_mapping or 'default'}, "
            f"ee_pose_topic={self._ee_pose_topic}, joint_state_topic={self._joint_state_topic}"
        )

    def _handle_ee_pose(self, msg: PoseStamped) -> None:
        with self._state_guard():
            self._latest_ee_pose = msg
            self._latest_ee_pose_monotonic = time.monotonic()

    def _handle_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _runtime_snapshot(self) -> RuntimeSnapshot:
        owner = self._gateway_lease.owner
        return RuntimeSnapshot(
            motion_authorized=self._motion_authorized,
            active_control_mode=self._active_control_mode,
            required_control_mode=self._skill_required_control_mode,
            busy=owner is not None,
            active_task_id=owner.root_task_id if owner is not None else "",
            validate_ready=self._validate_skill_client.service_is_ready(),
            task_executor_ready=self._task_executor_client.server_is_ready(),
            arm_trajectory_ready=self._arm_trajectory_client.server_is_ready(),
            ee_pose_fresh=self._ee_pose_is_fresh(),
        )

    def _ee_pose_is_fresh(self) -> bool:
        with self._state_guard():
            return self._ee_pose_is_fresh_unlocked()

    def _ee_pose_is_fresh_unlocked(self) -> bool:
        return (
            self._latest_ee_pose is not None
            and self._latest_ee_pose_monotonic is not None
            and self._ee_pose_receipt_is_fresh(self._latest_ee_pose_monotonic)
        )

    def _ee_pose_receipt_is_fresh(self, received_monotonic: float) -> bool:
        return time.monotonic() - received_monotonic <= self._robot_state_freshness_sec

    def _fresh_ee_pose_snapshot(self) -> _EePoseSnapshot | None:
        with self._state_guard():
            if not self._ee_pose_is_fresh_unlocked():
                return None
            return _EePoseSnapshot(
                pose=copy.deepcopy(self._latest_ee_pose),
                received_monotonic=self._latest_ee_pose_monotonic,
            )

    def _get_gateway_status(self, request, response):
        snapshot = self._runtime_snapshot()
        query = self._gateway_ledger.query(request.task_id, request.payload_hash)
        response.schema_version = 1
        response.robot_name = self._robot_name
        response.motion_authorized = snapshot.motion_authorized
        response.active_control_mode = snapshot.active_control_mode
        response.busy = snapshot.is_busy
        response.active_task_id = snapshot.active_task_id
        response.default_skill_timeout_sec = self._default_skill_timeout_sec
        response.task_budget_sec = self._task_budget_sec
        response.rpc_timeout_sec = self._rpc_timeout
        response.config_digest = self._config_digest
        response.request_state = query.state
        response.request_error_code = query.error_code
        response.capabilities = [
            self._capability_status(skill_name, snapshot) for skill_name in sorted(self._skill_requirements)
        ]
        return response

    def _capability_status(self, skill_name: str, snapshot: RuntimeSnapshot) -> SkillCapabilityStatus:
        decision = self._gateway_policy.evaluate(
            GatewayRequest(task_id=f"status-{skill_name}", skill_name=skill_name),
            snapshot,
            validate_parameters=False,
        )
        capability = SkillCapabilityStatus()
        capability.name = skill_name
        capability.ready = decision.admitted
        capability.reason = f"{decision.error_code}: {decision.message}" if decision.error_code else ""
        capability.required_control_mode = self._skill_required_control_mode
        return capability

    def _audit(
        self,
        event: str,
        *,
        task_id: str = "",
        payload_hash: str = "",
        skill: str = "",
        error_code: str = "",
        duration_sec: float | None = None,
        step_count: int | None = None,
    ) -> None:
        payload = {"event": event}
        for field_name, value in (
            ("task_id", task_id),
            ("payload_hash", payload_hash),
            ("skill", skill),
            ("error_code", error_code),
            ("duration_sec", duration_sec),
            ("step_count", step_count),
        ):
            if value not in ("", None):
                payload[field_name] = value
        self.get_logger().info(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _public_audit_context_copy(self) -> dict[str, str]:
        with self._state_guard():
            context = getattr(self, "_active_audit_context", None)
            if context is None:
                return {}
            return {
                field_name: str(context[field_name])
                for field_name in ("task_id", "payload_hash", "skill")
                if field_name in context and context[field_name] not in ("", None)
            }

    def _audit_cancel_propagated(self) -> None:
        context = self._public_audit_context_copy()
        if context:
            self._audit("cancel_propagated", **context)

    def _handle_cancel(self, _cancel_request):
        self._audit("cancel_requested", **self._public_audit_context_copy())
        return CancelResponse.ACCEPT

    def _handle_skill_goal(self, _goal_request):
        # GatewayPolicy owns admission decisions and must see concurrent roots so
        # it can return the documented busy reason to callers.
        if hasattr(self, "_gateway_policy"):
            return GoalResponse.ACCEPT
        with self._skill_goal_lock:
            if self._skill_goal_active:
                return GoalResponse.REJECT
            self._skill_goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _wait_for_future(
        future,
        timeout_sec: float,
        cancel_requested: Callable[[], bool] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if cancel_requested is not None and cancel_requested():
                if cancel_callback is not None:
                    cancel_callback()
                return False
            time.sleep(0.05)
        return future.done()

    @staticmethod
    def _abort_skill(result, goal_handle, executed_primitives, error_code: str, message: str):
        """Set failure fields on *result*, abort the goal, and return result."""
        result.success = False
        result.error_code = error_code
        result.message = message
        result.executed_primitives = executed_primitives
        goal_handle.abort()
        return result

    @staticmethod
    def _cancel_skill(result, goal_handle, executed_primitives, skill_name: str):
        result.success = False
        result.error_code = "SKILL_CANCELLED"
        result.message = f"skill cancelled: {skill_name}"
        result.executed_primitives = executed_primitives
        goal_handle.canceled()
        return result

    @staticmethod
    def _finish_primitive_failure(result, goal_handle, error_code: str, message: str, pose_name: str):
        result.success = False
        result.error_code = (
            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT if message.startswith("cancel cleanup timed out") else error_code
        )
        result.message = message
        result.pose_name = pose_name
        if result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
            goal_handle.abort()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    @staticmethod
    def _late_cleanup_confirmation(goal_handle) -> _LateCleanupConfirmation | None:
        return getattr(goal_handle, "late_cleanup_confirmation", None)

    def _best_effort_cancel_goal(self, goal_handle, *, audit: bool = True) -> None:
        if audit:
            self._audit_cancel_propagated()
        with suppress(Exception):
            goal_handle.cancel_goal_async()

    def _cancel_goal(self, goal_handle, result_future=None, late_cleanup_confirmation=None) -> bool:
        if goal_handle is None:
            return True
        self._audit_cancel_propagated()
        if result_future is None:
            try:
                result_future = goal_handle.get_result_async()
            except Exception:
                self._best_effort_cancel_goal(goal_handle, audit=False)
                return False
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if not self._wait_for_future(cancel_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        try:
            cancel_response = cancel_future.result()
        except Exception:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if hasattr(cancel_response, "goals_canceling") and not cancel_response.goals_canceling:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if not self._wait_for_future(result_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        try:
            wrapped_result = result_future.result()
        except Exception:
            return False
        child_result = getattr(wrapped_result, "result", None)
        return getattr(child_result, "error_code", "") != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT

    def _cancel_goal_when_ready(self, goal_future, late_cleanup_confirmation=None) -> None:
        def cancel_accepted_goal(completed_future) -> None:
            try:
                goal_handle = completed_future.result()
            except Exception:
                return
            if goal_handle is not None and goal_handle.accepted:
                self._cancel_goal(goal_handle, late_cleanup_confirmation=late_cleanup_confirmation)

        goal_future.add_done_callback(cancel_accepted_goal)

    def _cancel_goal_future(self, goal_future, late_cleanup_confirmation=None) -> bool:
        if not self._wait_for_future(goal_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_goal_future(goal_future, self._best_effort_cancel_goal)
            else:
                self._cancel_goal_when_ready(goal_future)
            return False
        try:
            goal_handle = goal_future.result()
        except Exception:
            return False
        if goal_handle is None or not goal_handle.accepted:
            return True
        return self._cancel_goal(goal_handle, late_cleanup_confirmation=late_cleanup_confirmation)

    def _sleep_with_cancel(self, goal_handle, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(0.05)
        return True

    def _wait_for_pose_target(self, goal_handle, target_pose: Pose, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            ee_pose_snapshot = self._fresh_ee_pose_snapshot()
            if ee_pose_snapshot is not None:
                ee_pose = ee_pose_snapshot.pose
                dx = float(ee_pose.pose.position.x - target_pose.position.x)
                dy = float(ee_pose.pose.position.y - target_pose.position.y)
                dz = float(ee_pose.pose.position.z - target_pose.position.z)
                if (dx * dx + dy * dy + dz * dz) ** 0.5 <= EE_POSITION_TOLERANCE_M:
                    return True
            time.sleep(0.05)
        return False

    def _validate_skill(
        self,
        skill_name: str,
        target_name: str,
        place_name: str,
        motion_direction: str = "",
        motion_distance: float = 0.0,
    ) -> tuple[bool, str]:
        if not self._validate_skill_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, "validate_skill service unavailable"

        request = ValidateSkill.Request()
        request.skill_name = skill_name
        request.target_name = target_name
        request.place_name = place_name
        request.motion_direction = motion_direction
        request.motion_distance = float(motion_distance)
        future = self._validate_skill_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=self._rpc_timeout):
            return False, "validate_skill timeout"
        response = future.result()
        if response is None:
            return False, "validate_skill returned no response"
        return response.allowed, response.reason

    def _validate_primitive(
        self,
        primitive_name: str,
        pose_name: str,
        gripper_position: float,
        relative_dx: float = 0.0,
        relative_dy: float = 0.0,
        relative_dz: float = 0.0,
        target_pose: Pose | None = None,
        velocity_scaling: float = 0.0,
        joint_names: list[str] | None = None,
        joint_positions: list[float] | None = None,
        joint_waypoints: list[float] | None = None,
        joint_waypoint_count: int = 0,
        primitive_duration_sec: float = 0.0,
        waypoint_duration_sec: float = 0.0,
    ) -> tuple[bool, str]:
        if not self._validate_primitive_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, "validate_primitive service unavailable"

        request = ValidatePrimitive.Request()
        request.primitive_name = primitive_name
        request.pose_name = pose_name
        request.relative_dx = float(relative_dx)
        request.relative_dy = float(relative_dy)
        request.relative_dz = float(relative_dz)
        if target_pose is not None:
            request.target_x = float(target_pose.position.x)
            request.target_y = float(target_pose.position.y)
            request.target_z = float(target_pose.position.z)
            request.target_qx = float(target_pose.orientation.x)
            request.target_qy = float(target_pose.orientation.y)
            request.target_qz = float(target_pose.orientation.z)
            request.target_qw = float(target_pose.orientation.w)
        else:
            request.target_x = 0.0
            request.target_y = 0.0
            request.target_z = 0.0
            request.target_qx = 0.0
            request.target_qy = 0.0
            request.target_qz = 0.0
            request.target_qw = 0.0
        request.velocity_scaling = float(velocity_scaling)
        request.gripper_position = float(gripper_position)
        request.joint_names = list(joint_names or [])
        request.joint_positions = [float(position) for position in (joint_positions or [])]
        request.joint_waypoints = [float(position) for position in (joint_waypoints or [])]
        request.joint_waypoint_count = int(joint_waypoint_count)
        request.primitive_duration_sec = float(primitive_duration_sec)
        request.waypoint_duration_sec = float(waypoint_duration_sec)
        future = self._validate_primitive_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=self._rpc_timeout):
            return False, "validate_primitive timeout"
        response = future.result()
        if response is None:
            return False, "validate_primitive returned no response"
        return response.allowed, response.reason

    def _current_joint_positions(self) -> dict[str, float]:
        if self._latest_joint_state is None:
            return {}
        return {
            str(name): float(position)
            for name, position in zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=False)
        }

    def _pose_from_name(self, pose_name: str) -> Pose:
        pose_cfg = self._named_poses[pose_name]
        position = pose_cfg.get("position", {})
        orientation = pose_cfg.get("orientation", {})
        pose = Pose()
        pose.position.x = float(position.get("x", 0.0))
        pose.position.y = float(position.get("y", 0.0))
        pose.position.z = float(position.get("z", 0.0))
        pose.orientation.x = float(orientation.get("x", 0.0))
        pose.orientation.y = float(orientation.get("y", 0.0))
        pose.orientation.z = float(orientation.get("z", 0.0))
        pose.orientation.w = float(orientation.get("w", 1.0))
        return pose

    @staticmethod
    def _pose_from_relative_offset(ee_pose: PoseStamped, dx: float, dy: float, dz: float) -> Pose:
        pose = Pose()
        pose.position.x = float(ee_pose.pose.position.x + dx)
        pose.position.y = float(ee_pose.pose.position.y + dy)
        pose.position.z = float(ee_pose.pose.position.z + dz)
        # Keep current EE orientation unchanged during position-only moves
        pose.orientation.x = float(ee_pose.pose.orientation.x)
        pose.orientation.y = float(ee_pose.pose.orientation.y)
        pose.orientation.z = float(ee_pose.pose.orientation.z)
        pose.orientation.w = float(ee_pose.pose.orientation.w)
        return pose

    @staticmethod
    def _pose_from_gripper_rotation(ee_pose: PoseStamped, primitive_name: str, angle_deg: float) -> Pose:
        angle_rad = math.radians(abs(angle_deg))
        half = angle_rad / 2.0
        sign = -1.0 if primitive_name == "rotate_gripper_cw" else 1.0
        cur = ee_pose.pose.orientation
        qc = (cur.x, cur.y, cur.z, cur.w)
        qd = (0.0, 0.0, sign * math.sin(half), math.cos(half))

        # Right-multiply: q_target = q_current * q_delta (local-frame rotation)
        qx = qc[3] * qd[0] + qc[0] * qd[3] + qc[1] * qd[2] - qc[2] * qd[1]
        qy = qc[3] * qd[1] - qc[0] * qd[2] + qc[1] * qd[3] + qc[2] * qd[0]
        qz = qc[3] * qd[2] + qc[0] * qd[1] - qc[1] * qd[0] + qc[2] * qd[3]
        qw = qc[3] * qd[3] - qc[0] * qd[0] - qc[1] * qd[1] - qc[2] * qd[2]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 1e-9:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        target_pose = Pose()
        target_pose.position.x = float(ee_pose.pose.position.x)
        target_pose.position.y = float(ee_pose.pose.position.y)
        target_pose.position.z = float(ee_pose.pose.position.z)
        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw
        return target_pose

    @staticmethod
    def _goal_id_key(goal_id: UUID) -> bytes:
        return bytes(goal_id.uuid)

    def _state_guard(self):
        # _state_lock is initialized in __init__ before any callback can fire;
        # the getattr fallback only protects against test fixtures that skip __init__.
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = RLock()
            object.__setattr__(self, "_state_lock", lock)
        return lock

    def _register_internal_primitive_goal(self, goal_id: UUID, admission, task_id: str) -> bytes:
        goal_key = self._goal_id_key(goal_id)
        with self._state_guard():
            self._pending_internal_primitive_goals[goal_key] = (admission, task_id)
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context={}),
            )
            if cleanup.admission is admission:
                cleanup.pending_goal_key = goal_key
                cleanup.confirmed_goal_key = None
        return goal_key

    def _register_internal_pick_handoff(self, goal_id: UUID, admission, task_id: str) -> tuple[str, bytes]:
        raw_goal_key = self._goal_id_key(goal_id)
        execution_token = raw_goal_key.hex()
        cleanup_key = b"pick\x00" + raw_goal_key
        with self._state_guard():
            handoffs = getattr(self, "_internal_pick_handoffs", None)
            if handoffs is None:
                handoffs = {}
                self._internal_pick_handoffs = handoffs
            handoffs[execution_token] = (admission, task_id)
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context={}),
            )
            if cleanup.admission is admission:
                cleanup.pending_goal_key = cleanup_key
                cleanup.confirmed_goal_key = None
        return execution_token, cleanup_key

    def _internal_pick_admission(self, goal):
        execution_token = str(getattr(goal, "execution_token", "")).strip()
        if not execution_token:
            return None
        with self._state_guard():
            handoff = getattr(self, "_internal_pick_handoffs", {}).get(execution_token)
            if handoff is None or handoff[1] != str(goal.task_id):
                return None
            return handoff[0]

    def _forget_internal_pick_handoff(self, execution_token: str) -> None:
        with self._state_guard():
            getattr(self, "_internal_pick_handoffs", {}).pop(execution_token, None)

    def _forget_internal_pick_handoffs_for_admission(self, admission) -> None:
        with self._state_guard():
            handoffs = getattr(self, "_internal_pick_handoffs", {})
            stale_tokens = [token for token, handoff in handoffs.items() if handoff[0] is admission]
            for token in stale_tokens:
                del handoffs[token]

    def _confirm_internal_pick_cleanup(self, admission, execution_token: str, cleanup_key: bytes) -> None:
        self._forget_internal_pick_handoff(execution_token)
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.get(id(admission))
            if cleanup is not None and cleanup.admission is admission and cleanup.pending_goal_key == cleanup_key:
                cleanup.confirmed_goal_key = cleanup_key
        self._converge_retained_admission(admission)

    def _activate_internal_primitive_goal(self, goal_handle):
        goal_id = getattr(goal_handle, "goal_id", None)
        if goal_id is None:
            return b"", None
        goal_key = self._goal_id_key(goal_id)
        with self._state_guard():
            pending = getattr(self, "_pending_internal_primitive_goals", {}).pop(goal_key, None)
            if pending is None or pending[1] != str(goal_handle.request.task_id):
                return goal_key, None
            self._active_internal_primitive_goals[goal_key] = pending
            return goal_key, pending[0]

    def _forget_internal_primitive_goal(self, goal_key: bytes) -> None:
        with self._state_guard():
            getattr(self, "_pending_internal_primitive_goals", {}).pop(goal_key, None)
            getattr(self, "_active_internal_primitive_goals", {}).pop(goal_key, None)

    def _retain_admission_cleanup(
        self,
        admission,
        audit_context: dict[str, str],
        duration_sec: float,
        step_count: int,
    ) -> None:
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context=dict(audit_context)),
            )
            cleanup.audit_context = dict(audit_context)
            cleanup.duration_sec = duration_sec
            cleanup.step_count = step_count
            cleanup.retained = True
            cleanup.parent_terminal = True
        self._converge_retained_admission(admission)

    def _confirm_late_internal_cleanup(self, admission, goal_key: bytes) -> None:
        self._forget_internal_primitive_goal(goal_key)
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.get(id(admission))
            if cleanup is not None and cleanup.admission is admission and cleanup.pending_goal_key == goal_key:
                cleanup.confirmed_goal_key = goal_key
        self._converge_retained_admission(admission)

    def _converge_retained_admission(self, admission) -> None:
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.get(id(admission))
            if (
                cleanup is None
                or cleanup.admission is not admission
                or not cleanup.retained
                or not cleanup.parent_terminal
                or cleanup.pending_goal_key is None
                or cleanup.confirmed_goal_key != cleanup.pending_goal_key
                or cleanup.finalizing
                or self._active_skill_admission is not admission
            ):
                return
            cleanup.finalizing = True
            try:
                self._gateway_policy.finalize(
                    admission,
                    error_code=SKILL_CANCEL_TIMEOUT,
                    terminal_metadata={"duration_sec": cleanup.duration_sec, "step_count": cleanup.step_count},
                )
            except Exception:
                cleanup.finalizing = False
                return
            self._active_skill_admission = None
            self._active_skill_owner = None
            self._active_audit_context = None
            self._forget_internal_pick_handoffs_for_admission(admission)
            del self._retained_admission_cleanup[id(admission)]

    def _release_late_external_primitive(self, token) -> None:
        with suppress(Exception):
            self._gateway_policy.release_external_primitive(token)

    def _schedule_late_internal_cleanup(self, goal_future, admission, goal_key: bytes) -> None:
        confirmation = _LateCleanupConfirmation()
        confirmation.add_callback(lambda: self._confirm_late_internal_cleanup(admission, goal_key))
        confirmation.watch_goal_future(goal_future, self._best_effort_cancel_goal)

    def _admit_primitive(self, goal, internal_admission) -> tuple[str, object | None]:
        snapshot = self._runtime_snapshot()
        if (
            internal_admission is not None
            and self._gateway_policy.borrow_internal(
                internal_admission,
                str(goal.task_id),
                str(goal.primitive_name),
            )
            is not None
        ):
            return "", None
        task_id = str(goal.task_id).strip() or f"external-primitive-{uuid.uuid4()}"
        return self._gateway_policy.admit_external_primitive(task_id, snapshot)

    def _execute_primitive(self, goal_handle):
        # See _execute_skill for the rationale of this test-fixture fallback.
        if not hasattr(self, "_gateway_policy"):
            return self._execute_primitive_unchecked(goal_handle)

        goal_key, registered_goal_admission = self._activate_internal_primitive_goal(goal_handle)
        internal_admission = registered_goal_admission or self._internal_pick_admission(goal_handle.request)
        error_code, token = self._admit_primitive(goal_handle.request, internal_admission)
        if error_code:
            if registered_goal_admission is not None:
                self._forget_internal_primitive_goal(goal_key)
            result = PrimitiveCommand.Result()
            result.success = False
            result.error_code = error_code
            result.message = error_code
            result.pose_name = goal_handle.request.pose_name
            goal_handle.abort()
            return result
        deferred_goal_handle = _DeferredTerminalGoalHandle(goal_handle)
        late_cleanup_confirmation = _LateCleanupConfirmation()
        if registered_goal_admission is not None:
            late_cleanup_confirmation.add_callback(
                lambda: self._confirm_late_internal_cleanup(internal_admission, goal_key)
            )
        elif token is not None:
            late_cleanup_confirmation.add_callback(lambda: self._release_late_external_primitive(token))
        deferred_goal_handle.late_cleanup_confirmation = late_cleanup_confirmation
        result = None
        try:
            result = self._execute_primitive_unchecked(deferred_goal_handle)
        except Exception:
            result = PrimitiveCommand.Result()
            result.success = False
            result.error_code = PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
            result.message = "cancel cleanup timed out: primitive execution state is unknown"
            result.pose_name = goal_handle.request.pose_name
            deferred_goal_handle.force_abort()
            return result
        finally:
            if registered_goal_admission is not None:
                self._forget_internal_primitive_goal(goal_key)
        if token is not None and result.error_code != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
            try:
                released = self._gateway_policy.release_external_primitive(token)
            except Exception:
                released = False
            if not released:
                result.success = False
                result.error_code = GATEWAY_FINALIZATION_FAILED
                result.message = "external primitive lease finalization failed"
                deferred_goal_handle.force_abort()
                return result
        committed = deferred_goal_handle.commit()
        if (
            committed
            and registered_goal_admission is not None
            and result.error_code != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
        ):
            late_cleanup_confirmation.confirm()
        return result

    def _execute_primitive_unchecked(self, goal_handle):
        goal = goal_handle.request
        result = PrimitiveCommand.Result()

        target_pose = None
        ee_pose_snapshot = None
        if goal.primitive_name in {"move_relative_ee", "rotate_gripper_cw", "rotate_gripper_ccw"}:
            ee_pose_snapshot = self._fresh_ee_pose_snapshot()
            if ee_pose_snapshot is None:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
        if goal.primitive_name == "move_to_pose":
            target_pose = goal.target_pose
        elif goal.primitive_name == "move_relative_ee":
            target_pose = self._pose_from_relative_offset(
                ee_pose_snapshot.pose,
                goal.relative_dx,
                goal.relative_dy,
                goal.relative_dz,
            )
        elif goal.primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            target_pose = self._pose_from_gripper_rotation(
                ee_pose_snapshot.pose,
                goal.primitive_name,
                goal.relative_dz,
            )

        allowed, reason = self._validate_primitive(
            goal.primitive_name,
            goal.pose_name,
            goal.gripper_position,
            relative_dx=goal.relative_dx,
            relative_dy=goal.relative_dy,
            relative_dz=goal.relative_dz,
            target_pose=target_pose,
            velocity_scaling=goal.velocity_scaling,
            joint_names=list(goal.joint_names),
            joint_positions=list(goal.joint_positions),
            joint_waypoints=list(goal.joint_waypoints),
            joint_waypoint_count=goal.joint_waypoint_count,
            primitive_duration_sec=goal.primitive_duration_sec,
            waypoint_duration_sec=goal.waypoint_duration_sec,
        )
        if not allowed:
            result.success = False
            result.error_code = "SAFETY_REJECTED"
            result.message = reason
            result.pose_name = goal.pose_name
            goal_handle.abort()
            return result

        feedback = PrimitiveCommand.Feedback()
        feedback.state = "dispatching"
        feedback.detail = f"primitive={goal.primitive_name}"
        goal_handle.publish_feedback(feedback)

        if goal.primitive_name == "move_to_configuration":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            ok, err_msg = self._exec_move_configuration(
                goal_handle,
                list(goal.joint_names),
                list(goal.joint_positions),
                float(goal.velocity_scaling),
                move_timeout,
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name == "move_to_joint_positions":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            ok, err_msg = self._exec_arm_joint_trajectory(
                goal_handle,
                list(goal.joint_names),
                list(goal.joint_positions),
                goal.task_id,
                move_timeout,
                float(goal.primitive_duration_sec),
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name == "move_through_joint_positions":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            joint_count = len(goal.joint_names)
            waypoints = [
                list(goal.joint_waypoints[index : index + joint_count])
                for index in range(0, len(goal.joint_waypoints), joint_count)
            ]
            ok, err_msg = self._exec_arm_joint_waypoint_trajectory(
                goal_handle,
                list(goal.joint_names),
                waypoints,
                goal.task_id,
                move_timeout,
                float(goal.waypoint_duration_sec),
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name in {"move_to_named_pose", "move_to_pose", "move_relative_ee"}:
            if goal.primitive_name == "move_to_named_pose":
                try:
                    pose = self._pose_from_name(goal.pose_name)
                except KeyError:
                    result.success = False
                    result.error_code = "UNKNOWN_POSE"
                    result.message = f"unknown named pose: {goal.pose_name!r}"
                    result.pose_name = goal.pose_name
                    goal_handle.abort()
                    return result
            elif goal.primitive_name == "move_to_pose":
                pose = goal.target_pose
            else:
                pose = target_pose
            if self._debug:
                self.get_logger().info(
                    f"[embodied-debug] primitive {goal.primitive_name} via task_dispatch "
                    f"task_id={goal.task_id} pose={goal.pose_name or '-'} "
                    f"delta=({goal.relative_dx:.3f}, {goal.relative_dy:.3f}, {goal.relative_dz:.3f}) "
                    f"xyz=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
                )
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            try:
                ok, err_msg = self._exec_arm_via_task_dispatch(
                    goal_handle,
                    goal.primitive_name,
                    pose,
                    goal.task_id,
                    move_timeout,
                    velocity_scaling=float(goal.velocity_scaling),
                    ee_pose_receipt_monotonic=(
                        ee_pose_snapshot.received_monotonic if ee_pose_snapshot is not None else None
                    ),
                )
            except _EePoseSnapshotExpired:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
            if not ok:
                return self._finish_primitive_failure(
                    result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, goal.pose_name
                )
        elif goal.primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            try:
                ok, err_msg = self._exec_rotate_gripper_via_task_dispatch(
                    goal_handle,
                    goal.primitive_name,
                    target_pose,
                    goal.task_id,
                    move_timeout,
                    ee_pose_snapshot.received_monotonic,
                )
            except _EePoseSnapshotExpired:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        else:
            # Delegate gripper control to task_dispatch via ExecuteTaskPlan action
            ok, err_msg = self._exec_gripper_via_task_dispatch(
                goal_handle, goal.primitive_name, goal.gripper_position, goal.task_id
            )
            if not ok:
                return self._finish_primitive_failure(
                    result, goal_handle, "PRIMITIVE_GRIPPER_FAILED", err_msg, goal.pose_name
                )
        result.success = True
        result.error_code = ""
        result.message = f"primitive completed: {goal.primitive_name}"
        result.pose_name = goal.pose_name
        goal_handle.succeed()
        return result

    def _exec_rotate_gripper_via_task_dispatch(
        self,
        goal_handle,
        primitive_name: str,
        target_pose: Pose,
        task_id: str,
        timeout_sec: float,
        ee_pose_receipt_monotonic: float,
    ) -> tuple[bool, str]:
        """Dispatch a wrist rotation target derived from a fresh EE pose snapshot."""

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] primitive {primitive_name} via task_dispatch "
                f"task_id={task_id} target_quat=({target_pose.orientation.x:.3f},"
                f"{target_pose.orientation.y:.3f},{target_pose.orientation.z:.3f},"
                f"{target_pose.orientation.w:.3f})"
            )

        return self._exec_arm_via_task_dispatch(
            goal_handle,
            primitive_name,
            target_pose,
            task_id,
            timeout_sec,
            ee_pose_receipt_monotonic=ee_pose_receipt_monotonic,
        )

    def _exec_arm_via_task_dispatch(
        self,
        goal_handle,
        primitive_name: str,
        target_pose: Pose,
        task_id: str,
        timeout_sec: float,
        *,
        ee_pose_receipt_monotonic: float | None = None,
        velocity_scaling: float = 0.0,
    ) -> tuple[bool, str]:
        """Send a MOVE_TO_POSE TaskStep to task_dispatch ExecuteTaskPlan action server."""
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._task_executor_client.wait_for_server(timeout_sec=2.0):
            msg = f"task_executor action server not available: {self._task_executor_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        if ee_pose_receipt_monotonic is not None and not self._ee_pose_receipt_is_fresh(ee_pose_receipt_monotonic):
            raise _EePoseSnapshotExpired

        step = TaskStep()
        step.type = TaskStep.MOVE_TO_POSE
        step.label = primitive_name
        step.target_pose = target_pose
        step.velocity_scaling = float(velocity_scaling)

        goal_msg = ExecuteTaskPlan.Goal()
        goal_msg.steps = [step]
        goal_msg.task_id = task_id or str(uuid.uuid4())
        goal_msg.task_description = primitive_name

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] primitive arm command via task_dispatch "
                f"task_id={task_id} primitive={primitive_name} "
                f"action={self._task_executor_action_name}"
            )

        if ee_pose_receipt_monotonic is not None and not self._ee_pose_receipt_is_fresh(ee_pose_receipt_monotonic):
            raise _EePoseSnapshotExpired
        send_future = self._task_executor_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm goal rejected by task_executor"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm motion"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return False, "cancelled during arm motion" if cleaned else "cancel cleanup timed out during arm motion"
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm motion" if cleaned else "cancel cleanup timed out during arm motion",
                )
            time.sleep(0.05)

        result = result_future.result().result
        if not result.success:
            return False, f"arm motion failed: {result.message}"
        return True, ""

    def _exec_move_configuration(
        self,
        goal_handle,
        joint_names: list[str],
        joint_positions: list[float],
        velocity_scaling: float,
        timeout_sec: float,
    ) -> tuple[bool, str]:
        if not self._move_configuration_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, f"move configuration service unavailable: {self._move_configuration_service}"

        request = MoveToConfiguration.Request()
        request.target_joint_state = JointState()
        request.target_joint_state.name = list(joint_names)
        request.target_joint_state.position = [float(position) for position in joint_positions]
        request.velocity_scaling = float(velocity_scaling)
        future = self._move_configuration_client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if goal_handle.is_cancel_requested:
                future.cancel()
                return False, "cancelled during MoveIt configuration motion"
            if time.monotonic() >= deadline:
                future.cancel()
                return False, "timeout waiting for MoveIt configuration motion"
            time.sleep(0.05)

        response = future.result()
        if response is None:
            return False, "move configuration service returned no response"
        if not response.success:
            return False, response.message
        return True, ""

    def _exec_arm_joint_trajectory(
        self,
        goal_handle,
        joint_names: list[str],
        joint_positions: list[float],
        task_id: str,
        timeout_sec: float,
        duration_sec: float,
    ) -> tuple[bool, str]:
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._arm_trajectory_client.wait_for_server(timeout_sec=2.0):
            msg = f"arm trajectory action server not available: {self._arm_trajectory_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        goal_msg = _build_joint_trajectory_goal(
            list(joint_names),
            [[float(position) for position in joint_positions]],
            max(0.1, float(duration_sec)),
        )

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive arm joint trajectory "
                f"task_id={task_id} joints={joint_names} positions={joint_positions}"
            )

        send_future = self._arm_trajectory_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm trajectory goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm trajectory goal rejected"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm joint trajectory execution"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during arm joint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm joint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint trajectory execution",
                )
            time.sleep(0.05)

        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            error_code = result.result.error_code if result is not None else "unknown"
            return False, f"arm trajectory execution failed: {error_code}"
        return True, ""

    def _exec_arm_joint_waypoint_trajectory(
        self,
        goal_handle,
        joint_names: list[str],
        joint_waypoints: list[list[float]],
        task_id: str,
        timeout_sec: float,
        waypoint_duration_sec: float,
    ) -> tuple[bool, str]:
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._arm_trajectory_client.wait_for_server(timeout_sec=2.0):
            msg = f"arm trajectory action server not available: {self._arm_trajectory_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        goal_msg = _build_joint_trajectory_goal(joint_names, joint_waypoints, waypoint_duration_sec)

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive arm joint waypoint trajectory "
                f"task_id={task_id} joints={joint_names} waypoints={len(joint_waypoints)} "
                f"waypoint_duration={waypoint_duration_sec:.3f}"
            )

        send_future = self._arm_trajectory_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm trajectory goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm trajectory goal rejected"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm joint waypoint trajectory execution"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during arm joint waypoint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint waypoint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm joint waypoint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint waypoint trajectory execution",
                )
            time.sleep(0.05)

        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            error_code = result.result.error_code if result is not None else "unknown"
            return False, f"arm trajectory execution failed: {error_code}"
        return True, ""

    def _exec_gripper_via_task_dispatch(
        self, goal_handle, primitive_name: str, gripper_position: float, task_id: str
    ) -> tuple[bool, str]:
        """Send a GRIPPER TaskStep to task_dispatch ExecuteTaskPlan action server."""
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive gripper command via task_dispatch "
                f"task_id={task_id} primitive={primitive_name} value={gripper_position:.3f} "
                f"action={self._task_executor_action_name}"
            )
        if not self._task_executor_client.wait_for_server(timeout_sec=2.0):
            msg = f"task_executor action server not available: {self._task_executor_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        step = TaskStep()
        step.type = TaskStep.GRIPPER
        step.label = primitive_name
        step.gripper_position = float(gripper_position)

        goal_msg = ExecuteTaskPlan.Goal()
        goal_msg.steps = [step]
        goal_msg.task_id = task_id or str(uuid.uuid4())
        goal_msg.task_description = primitive_name

        send_future = self._task_executor_client.send_goal_async(goal_msg)
        accept_timeout = self._gripper_settle_sec + 3.0
        exec_timeout = max(accept_timeout, 15.0)
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending gripper goal"
                    if cleaned
                    else "cancel cleanup timed out while sending gripper goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for gripper goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending gripper goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "gripper goal rejected by task_executor"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during gripper execution"
        deadline = time.monotonic() + exec_timeout
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during gripper execution"
                    if cleaned
                    else "cancel cleanup timed out during gripper execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for gripper execution"
                    if cleaned
                    else "cancel cleanup timed out during gripper execution",
                )
            time.sleep(0.05)

        result = result_future.result().result
        if not result.success:
            return False, f"gripper execution failed: {result.message}"
        return True, ""

    def _finalize_active_admission(
        self,
        admission,
        *,
        error_code: str,
        duration_sec: float,
        step_count: int,
    ) -> bool:
        with self._state_guard():
            if self._active_skill_admission is not admission:
                return False
            try:
                self._gateway_policy.finalize(
                    admission,
                    error_code=error_code,
                    terminal_metadata={"duration_sec": duration_sec, "step_count": step_count},
                )
            except Exception:
                return False
            self._active_skill_admission = None
            self._active_skill_owner = None
            self._active_audit_context = None
            self._forget_internal_pick_handoffs_for_admission(admission)
            self._retained_admission_cleanup.pop(id(admission), None)
            return True

    def _execute_pick_skill(
        self,
        goal_handle,
        template: dict,
        *,
        canonical_task_id: str | None = None,
    ) -> SkillCommand.Result:
        goal = goal_handle.request
        result = SkillCommand.Result()
        timeout_sec = float(goal.timeout_sec or template.get("timeout_sec", 180.0))

        if not self._pick_client.wait_for_server(timeout_sec=self._rpc_timeout):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "PICK_SERVER_UNAVAILABLE",
                f"pick action server unavailable: {self._pick_action_name}",
            )

        completed_phases: list[str] = []

        def _feedback_cb(feedback_msg) -> None:
            pick_feedback = feedback_msg.feedback
            phase = str(pick_feedback.phase)
            if phase and (not completed_phases or completed_phases[-1] != phase):
                completed_phases.append(phase)
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"pick:{phase}:{pick_feedback.detail}"
            goal_handle.publish_feedback(feedback)

        pick_goal = PickObject.Goal()
        pick_goal.task_id = canonical_task_id if canonical_task_id is not None else str(goal.task_id).strip()
        pick_goal.target_query = goal.target_name
        pick_goal.timeout_sec = timeout_sec

        pick_goal_id = None
        execution_token = ""
        cleanup_key = b""
        late_cleanup_confirmation = None
        with self._state_guard():
            admission = getattr(self, "_active_skill_admission", None)
        if admission is not None:
            pick_goal_id = UUID(uuid=list(uuid.uuid4().bytes))
            execution_token, cleanup_key = self._register_internal_pick_handoff(
                pick_goal_id,
                admission,
                str(pick_goal.task_id),
            )
            late_cleanup_confirmation = _LateCleanupConfirmation()
            late_cleanup_confirmation.add_callback(
                lambda: self._confirm_internal_pick_cleanup(
                    admission,
                    execution_token,
                    cleanup_key,
                )
            )

        try:
            if pick_goal_id is None:
                send_future = self._pick_client.send_goal_async(pick_goal, feedback_callback=_feedback_cb)
            else:
                send_future = self._pick_client.send_goal_async(
                    pick_goal,
                    feedback_callback=_feedback_cb,
                    goal_uuid=pick_goal_id,
                )
        except Exception:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                SKILL_CANCEL_TIMEOUT,
                "pick goal send state is unknown",
            )
        if not self._wait_for_future(send_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_goal_future(send_future, self._best_effort_cancel_goal)
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                SKILL_CANCEL_TIMEOUT if late_cleanup_confirmation is not None else "PICK_GOAL_TIMEOUT",
                "pick goal cleanup state is unknown"
                if late_cleanup_confirmation is not None
                else "timed out sending pick goal",
            )

        try:
            pick_handle = send_future.result()
        except Exception:
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                SKILL_CANCEL_TIMEOUT if late_cleanup_confirmation is not None else "PICK_GOAL_REJECTED",
                "pick goal response state is unknown",
            )
        if pick_handle is None or not pick_handle.accepted:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.confirm()
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                "PICK_GOAL_REJECTED",
                "pick executor rejected the goal",
            )

        try:
            result_future = pick_handle.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(pick_handle)
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                SKILL_CANCEL_TIMEOUT,
                "pick result state is unknown",
            )
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not result_future.done():
            if goal_handle.is_cancel_requested:
                if self._cancel_goal(pick_handle, result_future, late_cleanup_confirmation):
                    if late_cleanup_confirmation is not None:
                        late_cleanup_confirmation.confirm()
                    return self._cancel_skill(
                        result,
                        goal_handle,
                        [f"grasp_pipeline:{phase}" for phase in completed_phases],
                        goal.skill_name,
                    )
                return self._abort_skill(
                    result,
                    goal_handle,
                    [f"grasp_pipeline:{phase}" for phase in completed_phases],
                    SKILL_CANCEL_TIMEOUT,
                    "pick cancel cleanup timed out",
                )
            if time.monotonic() >= deadline:
                if not self._cancel_goal(pick_handle, result_future, late_cleanup_confirmation):
                    return self._abort_skill(
                        result,
                        goal_handle,
                        [f"grasp_pipeline:{phase}" for phase in completed_phases],
                        SKILL_CANCEL_TIMEOUT,
                        "pick timeout cleanup timed out",
                    )
                if late_cleanup_confirmation is not None:
                    late_cleanup_confirmation.confirm()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [f"grasp_pipeline:{phase}" for phase in completed_phases],
                    "SKILL_TIMEOUT",
                    "pick skill deadline exceeded",
                )
            time.sleep(0.05)

        if not result_future.done():
            if not self._cancel_goal(pick_handle, result_future, late_cleanup_confirmation):
                return self._abort_skill(
                    result,
                    goal_handle,
                    [f"grasp_pipeline:{phase}" for phase in completed_phases],
                    SKILL_CANCEL_TIMEOUT,
                    "pick executor shutdown cleanup state is unknown",
                )
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.confirm()
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                "PICK_ABORTED",
                "pick executor stopped before returning a result",
            )

        try:
            action_result = result_future.result()
        except Exception:
            self._best_effort_cancel_goal(pick_handle)
            return self._abort_skill(
                result,
                goal_handle,
                [f"grasp_pipeline:{phase}" for phase in completed_phases],
                SKILL_CANCEL_TIMEOUT,
                "pick result state is unknown",
            )
        pick_result = action_result.result if action_result is not None else None
        result.executed_primitives = [f"grasp_pipeline:{phase}" for phase in completed_phases]
        if pick_result is None:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.confirm()
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "MISSING_PICK_RESULT",
                "pick executor returned no result",
            )
        if pick_result.error_code in {PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SKILL_CANCEL_TIMEOUT}:
            if execution_token:
                self._forget_internal_pick_handoff(execution_token)
        elif late_cleanup_confirmation is not None:
            late_cleanup_confirmation.confirm()
        if not pick_result.success:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                pick_result.error_code or "PICK_FAILED",
                pick_result.message,
            )

        result.success = True
        result.error_code = ""
        result.message = pick_result.message
        goal_handle.succeed()
        return result

    def _execute_skill(self, goal_handle):
        try:
            return self._execute_skill_gateway(goal_handle)
        finally:
            with self._skill_goal_lock:
                self._skill_goal_active = False

    def _execute_skill_gateway(self, goal_handle):
        # _gateway_policy is always assigned in __init__ on the production path;
        # this branch only triggers for unit-test fixtures built via object.__new__
        # that bypass __init__. A node with a half-initialized __init__ would have
        # raised before reaching the action server, so this fallback does not create
        # a production bypass; it keeps test fixtures that exercise _execute_skill_child
        # working without re-stubbing the full Gateway.
        if not hasattr(self, "_gateway_policy"):
            return self._execute_skill_child(goal_handle)

        goal = goal_handle.request
        request = GatewayRequest(
            task_id=goal.task_id,
            skill_name=goal.skill_name,
            target_name=goal.target_name,
            place_name=goal.place_name,
            motion_direction=goal.motion_direction,
            motion_distance=goal.motion_distance,
            timeout_sec=None if goal.timeout_sec == 0.0 else goal.timeout_sec,
        )
        owner = ExecutionOwner.skill_command(str(goal.task_id).strip())
        with self._state_guard():
            admission = self._gateway_policy.admit(request, self._runtime_snapshot(), owner)
            prepared = admission.prepared_request
            audit_context = {
                "task_id": prepared.identity.task_id if prepared is not None else str(goal.task_id).strip(),
                "payload_hash": prepared.identity.payload_hash if prepared is not None else "",
                "skill": str(goal.skill_name).strip(),
            }
            if admission.admitted:
                self._active_skill_admission = admission
                self._active_skill_owner = owner
                self._active_audit_context = audit_context
        self._audit("requested", **audit_context)
        if not admission.admitted:
            self._audit("precondition_rejected", error_code=admission.error_code, **audit_context)
            result = SkillCommand.Result()
            return self._abort_skill(
                result,
                goal_handle,
                [],
                admission.error_code,
                admission.message or admission.error_code,
            )

        started = time.monotonic()
        result = None
        deferred_goal_handle = _DeferredTerminalGoalHandle(goal_handle)
        try:
            allowed, reason = self._validate_skill(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                motion_direction=goal.motion_direction,
                motion_distance=goal.motion_distance,
            )
            self._audit(
                "safety_validated",
                error_code="" if allowed else "SKILL_REJECTED",
                **audit_context,
            )
            if not allowed:
                result = SkillCommand.Result()
                result = self._abort_skill(result, deferred_goal_handle, [], "SKILL_REJECTED", reason)
            else:
                result = self._execute_skill_child(
                    deferred_goal_handle,
                    validation_done=True,
                    effective_timeout_sec=admission.effective_timeout_sec,
                    canonical_task_id=prepared.identity.task_id,
                )
        except Exception as exc:
            self.get_logger().error(f"[skill_executor] gateway execution failed: {exc}")
            self._audit("gateway_exception", error_code="SKILL_REJECTED", **audit_context)
            result = SkillCommand.Result()
            result = self._abort_skill(
                result,
                deferred_goal_handle,
                [],
                "SKILL_REJECTED",
                f"gateway execution failed: {exc}",
            )
        finally:
            duration_sec = time.monotonic() - started
            error_code = result.error_code if result is not None else "SKILL_REJECTED"
            step_count = len(result.executed_primitives) if result is not None else 0
            cleanup_unknown = error_code in {PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SKILL_CANCEL_TIMEOUT}
            if cleanup_unknown:
                result.success = False
                result.error_code = SKILL_CANCEL_TIMEOUT
                result.message = "cancel cleanup timed out"
                self._audit(
                    "terminal",
                    error_code=result.error_code,
                    duration_sec=duration_sec,
                    step_count=step_count,
                    **audit_context,
                )
                deferred_goal_handle.force_abort()
                self._retain_admission_cleanup(admission, audit_context, duration_sec, step_count)
            else:
                if not self._finalize_active_admission(
                    admission,
                    error_code=error_code,
                    duration_sec=duration_sec,
                    step_count=step_count,
                ):
                    self.get_logger().error(GATEWAY_FINALIZATION_FAILED)
                    result.success = False
                    result.error_code = GATEWAY_FINALIZATION_FAILED
                    result.message = "gateway finalization failed"
                    self._audit(
                        "terminal",
                        error_code=result.error_code,
                        duration_sec=duration_sec,
                        step_count=step_count,
                        **audit_context,
                    )
                    deferred_goal_handle.force_abort()
                else:
                    self._audit(
                        "terminal",
                        error_code=error_code,
                        duration_sec=duration_sec,
                        step_count=step_count,
                        **audit_context,
                    )
                    deferred_goal_handle.commit()
        return result

    def _execute_skill_child(
        self,
        goal_handle,
        *,
        validation_done: bool = False,
        effective_timeout_sec: float | None = None,
        canonical_task_id: str | None = None,
    ):
        goal = goal_handle.request
        result = SkillCommand.Result()

        if not validation_done:
            allowed, reason = self._validate_skill(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                motion_direction=goal.motion_direction,
                motion_distance=goal.motion_distance,
            )
            if not allowed:
                return self._abort_skill(result, goal_handle, [], "SKILL_REJECTED", reason)

        template = self._skill_templates.get(goal.skill_name, {})
        if str(template.get("executor", "")).strip() == "grasp_pipeline":
            return self._execute_pick_skill(
                goal_handle,
                template,
                canonical_task_id=canonical_task_id,
            )

        try:
            primitives: list[PrimitiveSpec] = resolve_skill_primitives(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                goal.motion_direction,
                goal.motion_distance,
                self._named_targets,
                self._gripper_open,
                self._gripper_closed,
                self._skill_templates,
                self._relative_motion_direction_mapping,
                current_joint_positions=self._current_joint_positions(),
                arm_joint_names=self._arm_joint_names,
            )
        except Exception as exc:
            return self._abort_skill(result, goal_handle, [], "SKILL_RESOLUTION_FAILED", str(exc))

        with self._state_guard():
            audit_context = getattr(self, "_active_audit_context", None)
        if audit_context is not None:
            self._audit("accepted", **audit_context)

        skill_deadline = None
        if effective_timeout_sec is not None:
            skill_deadline = time.monotonic() + effective_timeout_sec
        elif goal.timeout_sec > 0.0:
            skill_deadline = time.monotonic() + float(goal.timeout_sec)

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] skill_executor start "
                f"task_id={goal.task_id} skill={goal.skill_name} primitives={primitives}"
            )

        if not self._primitive_client.wait_for_server(timeout_sec=self._rpc_timeout):
            result.success = False
            result.error_code = "PRIMITIVE_SERVER_UNAVAILABLE"
            result.message = "primitive action server unavailable"
            result.executed_primitives = []
            goal_handle.abort()
            return result

        executed_primitives: list[str] = []
        total_steps = len(primitives)
        for step_count, primitive in enumerate(primitives, start=1):
            if goal_handle.is_cancel_requested:
                self._audit_cancel_propagated()
                return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)

            remaining_timeout = None
            if skill_deadline is not None:
                remaining_timeout = skill_deadline - time.monotonic()
                if remaining_timeout <= 0.0:
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_TIMEOUT",
                        f"skill deadline exceeded before primitive {primitive.primitive_name}",
                    )

            primitive_name = primitive.primitive_name
            pose_name = primitive.pose_name
            gripper_position = primitive.gripper_position
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"step {step_count} of {total_steps}"
            goal_handle.publish_feedback(feedback)

            primitive_goal = PrimitiveCommand.Goal()
            primitive_goal.task_id = canonical_task_id if canonical_task_id is not None else goal.task_id
            primitive_goal.primitive_name = primitive_name
            primitive_goal.pose_name = pose_name
            primitive_goal.velocity_scaling = 0.0
            primitive_goal.relative_dx = float(primitive.relative_dx)
            primitive_goal.relative_dy = float(primitive.relative_dy)
            primitive_goal.relative_dz = float(primitive.relative_dz)
            primitive_goal.gripper_position = float(gripper_position)
            primitive_goal.joint_names = list(primitive.joint_names)
            primitive_goal.joint_positions = [float(position) for position in primitive.joint_positions]
            primitive_goal.joint_waypoints = [
                float(position) for waypoint in primitive.joint_waypoints for position in waypoint
            ]
            primitive_goal.joint_waypoint_count = len(primitive.joint_waypoints)
            primitive_goal.primitive_duration_sec = float(primitive.duration_sec)
            primitive_goal.waypoint_duration_sec = float(primitive.waypoint_duration_sec)
            primitive_goal.timeout_sec = remaining_timeout if remaining_timeout is not None else 0.0

            if audit_context is not None:
                self._audit("primitive_started", step_count=step_count, **audit_context)

            internal_goal_key = None
            internal_late_cleanup_confirmation = None
            with self._state_guard():
                admission = getattr(self, "_active_skill_admission", None)
            if admission is not None:
                internal_goal_id = UUID(uuid=list(uuid.uuid4().bytes))
                internal_goal_key = self._register_internal_primitive_goal(
                    internal_goal_id,
                    admission,
                    str(primitive_goal.task_id),
                )
                internal_late_cleanup_confirmation = _LateCleanupConfirmation()
                internal_late_cleanup_confirmation.add_callback(
                    lambda admission=admission, goal_key=internal_goal_key: self._confirm_late_internal_cleanup(
                        admission, goal_key
                    )
                )
                try:
                    send_goal_future = self._primitive_client.send_goal_async(
                        primitive_goal, goal_uuid=internal_goal_id
                    )
                except Exception:
                    self._forget_internal_primitive_goal(internal_goal_key)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"primitive send state is unknown: {primitive_name}",
                    )
            else:
                try:
                    send_goal_future = self._primitive_client.send_goal_async(primitive_goal)
                except Exception:
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"primitive send state is unknown: {primitive_name}",
                    )
            send_goal_timeout = (
                self._rpc_timeout if remaining_timeout is None else min(self._rpc_timeout, remaining_timeout)
            )

            if not self._wait_for_future(
                send_goal_future,
                timeout_sec=max(0.1, send_goal_timeout),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                if internal_goal_key is not None:
                    if goal_handle.is_cancel_requested:
                        if self._cancel_goal_future(send_goal_future, internal_late_cleanup_confirmation):
                            self._forget_internal_primitive_goal(internal_goal_key)
                            return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                        return self._abort_skill(
                            result,
                            goal_handle,
                            executed_primitives,
                            SKILL_CANCEL_TIMEOUT,
                            f"cancel cleanup timed out for primitive {primitive_name}",
                        )
                    internal_late_cleanup_confirmation.watch_goal_future(
                        send_goal_future,
                        self._best_effort_cancel_goal,
                    )
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if goal_handle.is_cancel_requested:
                    if self._cancel_goal_future(send_goal_future):
                        if internal_goal_key is not None:
                            self._forget_internal_primitive_goal(internal_goal_key)
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if not self._cancel_goal_future(send_goal_future):
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_TIMEOUT",
                    f"timed out sending primitive {primitive_name}",
                )

            try:
                primitive_handle = send_goal_future.result()
            except Exception:
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive send state is unknown: {primitive_name}",
                )
            if primitive_handle is None or not primitive_handle.accepted:
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_REJECTED",
                    f"primitive goal rejected: {primitive_name}",
                )

            try:
                result_future = primitive_handle.get_result_async()
            except Exception:
                self._best_effort_cancel_goal(primitive_handle)
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive result state is unknown: {primitive_name}",
                )
            remaining_timeout = None
            if skill_deadline is not None:
                remaining_timeout = skill_deadline - time.monotonic()
                if remaining_timeout <= 0.0:
                    if not self._cancel_goal(primitive_handle, result_future):
                        if internal_late_cleanup_confirmation is not None:
                            internal_late_cleanup_confirmation.watch_result_future(result_future)
                        return self._abort_skill(
                            result,
                            goal_handle,
                            executed_primitives,
                            SKILL_CANCEL_TIMEOUT,
                            f"cancel cleanup timed out for primitive {primitive_name}",
                        )
                    if internal_goal_key is not None:
                        self._forget_internal_primitive_goal(internal_goal_key)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_TIMEOUT",
                        f"primitive timed out: {primitive_name}",
                    )
            wait_timeout = remaining_timeout if remaining_timeout is not None else 30.0
            if not self._wait_for_future(
                result_future,
                timeout_sec=max(0.1, wait_timeout),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                if goal_handle.is_cancel_requested:
                    if self._cancel_goal(primitive_handle, result_future):
                        if internal_goal_key is not None:
                            self._forget_internal_primitive_goal(internal_goal_key)
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    if internal_late_cleanup_confirmation is not None:
                        internal_late_cleanup_confirmation.watch_result_future(result_future)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if not self._cancel_goal(primitive_handle, result_future):
                    if internal_late_cleanup_confirmation is not None:
                        internal_late_cleanup_confirmation.watch_result_future(result_future)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result, goal_handle, executed_primitives, "SKILL_TIMEOUT", f"primitive timed out: {primitive_name}"
                )

            try:
                action_result = result_future.result()
            except Exception:
                self._best_effort_cancel_goal(primitive_handle)
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive result state is unknown: {primitive_name}",
                )
            if internal_goal_key is not None:
                self._forget_internal_primitive_goal(internal_goal_key)
            primitive_result = action_result.result if action_result is not None else None
            if primitive_result is None or not primitive_result.success:
                error_code = primitive_result.error_code if primitive_result is not None else "MISSING_PRIMITIVE_RESULT"
                message = primitive_result.message if primitive_result is not None else "missing primitive result"
                if error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                    error_code = SKILL_CANCEL_TIMEOUT
                return self._abort_skill(result, goal_handle, executed_primitives, error_code, message)

            primitive_label = primitive_name if not pose_name else f"{primitive_name}:{pose_name}"
            if primitive_name == "move_relative_ee":
                primitive_label = (
                    f"{primitive_name}:{primitive.relative_dx:.3f},"
                    f"{primitive.relative_dy:.3f},{primitive.relative_dz:.3f}"
                )
            if primitive_name == "move_to_joint_positions":
                primitive_label = f"{primitive_name}:" + ",".join(
                    f"{joint_name}={joint_position:.3f}"
                    for joint_name, joint_position in zip(
                        primitive.joint_names, primitive.joint_positions, strict=False
                    )
                )
            if primitive_name == "move_through_joint_positions":
                primitive_label = f"{primitive_name}:{len(primitive.joint_waypoints)} waypoints"
            executed_primitives.append(primitive_label)

        result.success = True
        result.error_code = ""
        result.message = f"skill completed: {goal.skill_name}"
        result.executed_primitives = executed_primitives
        goal_handle.succeed()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkillExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
