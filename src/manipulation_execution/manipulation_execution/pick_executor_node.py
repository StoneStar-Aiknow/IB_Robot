"""ROS lifecycle for the closed-loop PickObject action server."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from embodied_common.dispatch_binding import (
    copy_binding,
    delegated_executor_identity,
    delegated_executor_identity_matches,
    fill_delegated_executor_identity,
    load_delegated_model_identity,
)
from ibrobot_msgs.action import PickObject, PrimitiveCommand
from ibrobot_msgs.srv import MoveToConfiguration, PlanGrasp, VerifyGrasp
from manipulation_execution.phases.execution import ExecutionPhase
from manipulation_execution.phases.flow import PickFlowPhase
from manipulation_execution.phases.planning import PlanningPhase
from manipulation_execution.phases.preparation import PreparationPhase
from manipulation_execution.pick_executor_helpers import PickExecutorHelpers
from manipulation_execution.pick_executor_models import (
    BaseSceneGeometry,
    FlowState,
    IKPayload,
    PickCancelled,
    PickFlowError,
    PlannerSceneGeometry,
    PreparedCandidate,
    RankedCandidate,
)

__all__ = [
    "BaseSceneGeometry",
    "FlowState",
    "IKPayload",
    "PickCancelled",
    "PickExecutorNode",
    "PickFlowError",
    "PlannerSceneGeometry",
    "PreparedCandidate",
    "RankedCandidate",
    "main",
]


class PickExecutorNode(
    PickFlowPhase,
    ExecutionPhase,
    PreparationPhase,
    PlanningPhase,
    PickExecutorHelpers,
    Node,
):
    """Own ROS resources and delegate pick behavior to focused phase mixins."""

    _PHASE_PROGRESS = {
        "preflight": 0.02,
        "observe": 0.08,
        "planning": 0.18,
        "selecting": 0.32,
        "open": 0.38,
        "approach": 0.48,
        "descend": 0.60,
        "close": 0.68,
        "verify_close": 0.74,
        "probe_lift": 0.80,
        "verify_probe": 0.84,
        "lift": 0.92,
        "verify_lift": 0.97,
        "release": 0.99,
        "completed": 1.0,
    }

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("pick_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("action_name", "/manipulation/execute_pick")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("grasp_execution_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("home_joint_positions_json", "{}")
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("rpc_timeout_sec", 5.0)

        self._action_name = self.get_parameter("action_name").value
        self._primitive_action_name = self.get_parameter("primitive_action_name").value
        self._dispatch_nonce = ""
        self._dispatch_binding = None
        self._config = self._load_json_object(self.get_parameter("grasp_execution_json").value)
        self._executor_identity = delegated_executor_identity(
            name="grasp_pipeline",
            endpoint_name=self._action_name,
            configuration=self._config,
            **load_delegated_model_identity(self._config),
        )
        self._workspace = self._load_json_object(self.get_parameter("workspace_json").value)
        self._home_joint_positions = self._load_home_joint_positions(
            self.get_parameter("home_joint_positions_json").value
        )
        self._arm_joint_names = self._load_json_list(self.get_parameter("arm_joint_names_json").value)
        self._gripper_open = float(self.get_parameter("gripper_open_position").value)
        self._gripper_closed = float(self.get_parameter("gripper_closed_position").value)
        self._rpc_timeout = float(self.get_parameter("rpc_timeout_sec").value)
        self._ready_timeout = float(self._config.get("ready_timeout_sec", 30.0))
        self._joint_state_topic = str(self._config.get("joint_state_topic", "/joint_states"))
        ik_config = self._config.get("ik", {})
        self._ik_worker_count = int(ik_config.get("worker_count", 0))
        if not 0 <= self._ik_worker_count <= 8:
            raise ValueError("ik.worker_count must be between 0 and 8")
        self._ik_worker_prefix = str(ik_config.get("worker_namespace_prefix", "/ik_worker")).rstrip("/")
        if self._ik_worker_count > 0 and not self._ik_worker_prefix:
            raise ValueError("ik.worker_namespace_prefix must not be empty when worker_count is positive")

        self._planner_service = str(self._config.get("planner_service", "/grasp_planner/plan_grasp"))
        self._verifier_service = str(self._config.get("verifier_service", "/grasp_verifier/verify_grasp"))
        self._move_configuration_service = str(
            self._config.get("move_configuration_service", "/moveit_gateway/move_to_configuration")
        )
        self._ik_service = str(self._config.get("ik_service", "/compute_ik"))
        self._fk_service = str(self._config.get("fk_service", "/compute_fk"))
        self._base_frame = str(self._config.get("base_frame", "base"))
        self._ee_frame = str(self._config.get("ee_frame", "gripper"))
        self._verification_policy = str(self._config.get("verification", "required")).lower()
        self._target_geometry = self._config.get("target_geometry", {})
        self._validate_home_joint_config()
        self._mesh_directory: Path | None = None
        if bool(self._target_geometry.get("tabletop_filter", False)):
            mesh_package = str(self._target_geometry.get("mesh_package", "robot_description"))
            mesh_subdirectory = str(self._target_geometry.get("mesh_directory", "meshes/lerobot/so101"))
            try:
                self._mesh_directory = Path(get_package_share_directory(mesh_package)) / mesh_subdirectory
            except Exception as exc:
                self.get_logger().error(f"Cannot resolve SO101 target-gripper meshes: {exc}")

        callback_group = ReentrantCallbackGroup()
        self._joint_state_lock = threading.Lock()
        self._latest_joint_state: JointState | None = None
        self._ik_worker_verification: tuple[tuple[object, ...], float] | None = None
        self._kinematics_health_lock = threading.Lock()
        self._kinematics_unhealthy_workers: set[int] = set()
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._handle_joint_state,
            10,
            callback_group=callback_group,
        )
        self._planner_client = self.create_client(PlanGrasp, self._planner_service, callback_group=callback_group)
        self._verifier_client = self.create_client(
            VerifyGrasp,
            self._verifier_service,
            callback_group=callback_group,
        )
        self._move_configuration_client = self.create_client(
            MoveToConfiguration,
            self._move_configuration_service,
            callback_group=callback_group,
        )
        self._ik_client = self.create_client(GetPositionIK, self._ik_service, callback_group=callback_group)
        self._fk_client = self.create_client(GetPositionFK, self._fk_service, callback_group=callback_group)
        self._ik_worker_clients = [
            self.create_client(
                GetPositionIK,
                f"{self._ik_worker_prefix}_{index}/compute_ik",
                callback_group=callback_group,
            )
            for index in range(self._ik_worker_count)
        ]
        self._fk_worker_clients = [
            self.create_client(
                GetPositionFK,
                f"{self._ik_worker_prefix}_{index}/compute_fk",
                callback_group=callback_group,
            )
            for index in range(self._ik_worker_count)
        ]
        self._primitive_client = ActionClient(
            self,
            PrimitiveCommand,
            self._primitive_action_name,
            callback_group=callback_group,
        )
        planner_timeout = float(self._config.get("planner", {}).get("timeout_sec", 120.0))
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=max(10.0, planner_timeout + 10.0)))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._action_server = ActionServer(
            self,
            PickObject,
            self._action_name,
            execute_callback=self._execute_pick,
            goal_callback=self._handle_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self.get_logger().info(
            f"PickExecutor ready: action={self._action_name} planner={self._planner_service} "
            f"verifier={self._verifier_service} primitive={self._primitive_action_name} "
            f"ik_workers={self._ik_worker_count}"
        )

    def _handle_goal(self, goal_request):
        if not str(goal_request.target_query).strip():
            return GoalResponse.REJECT
        if not delegated_executor_identity_matches(goal_request.expected_executor, self._executor_identity):
            return GoalResponse.REJECT
        dispatch_nonce = str(goal_request.dispatch_binding.dispatch_nonce).strip()
        if bool(goal_request.supervised_direct) and dispatch_nonce:
            return GoalResponse.REJECT
        if not bool(goal_request.supervised_direct) and not dispatch_nonce:
            return GoalResponse.REJECT
        binding = goal_request.dispatch_binding
        budget = binding.task_budget
        if (
            binding.schema_version != 1
            or not str(binding.task_id).strip()
            or not str(binding.root_task_id).strip()
            or not binding.expected_registry_epoch
            or binding.expected_registry_generation <= 0
            or not binding.expected_registry_digest
            or budget.schema_version != 1
        ):
            return GoalResponse.REJECT
        started = budget.started_at.sec + budget.started_at.nanosec / 1_000_000_000
        deadline = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        timeout_sec = float(goal_request.timeout_sec)
        now = self.get_clock().now().nanoseconds / 1_000_000_000
        if (
            budget.started_at.sec < 0
            or budget.deadline.sec < 0
            or not 0 <= budget.started_at.nanosec < 1_000_000_000
            or not 0 <= budget.deadline.nanosec < 1_000_000_000
            or not math.isfinite(started)
            or not math.isfinite(deadline)
            or deadline <= started
            or deadline <= now
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0.0
        ):
            return GoalResponse.REJECT
        if int(goal_request.mode) not in {
            PickObject.Goal.MODE_EXECUTE,
            PickObject.Goal.MODE_PLAN_ONLY,
            PickObject.Goal.MODE_OBSERVE_ONLY,
        }:
            return GoalResponse.REJECT
        if not math.isfinite(float(goal_request.release_drop_height_m)):
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
            self._dispatch_nonce = dispatch_nonce
            self._supervised_direct = bool(goal_request.supervised_direct)
            self._direct_primitive_index = 0
            self._dispatch_binding = copy_binding(goal_request.dispatch_binding)
        return GoalResponse.ACCEPT

    def _result_from_state(self, state: FlowState) -> PickObject.Result:
        result = PickExecutorHelpers._result_from_state(state)
        fill_delegated_executor_identity(result.actual_executor, self._executor_identity)
        return result

    def _handle_joint_state(self, message: JointState) -> None:
        with self._joint_state_lock:
            self._latest_joint_state = message

    def _snapshot_joint_state(self) -> JointState | None:
        with self._joint_state_lock:
            if self._latest_joint_state is None:
                return None
            return self._copy_joint_state(self._latest_joint_state)

    def _kinematics_unhealthy_snapshot(self) -> set[int]:
        """Return a stable copy of the unhealthy IK/FK worker indices."""
        with self._kinematics_health_lock:
            return set(self._kinematics_unhealthy_workers)

    @staticmethod
    def _handle_cancel(_goal_handle):
        return CancelResponse.ACCEPT


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickExecutorNode()
    executor = MultiThreadedExecutor(num_threads=max(4, node._ik_worker_count * 2 + 2))
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
