"""Closed-loop PickObject action server."""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState

from ibrobot_msgs.action import PickObject, PrimitiveCommand
from ibrobot_msgs.msg import GraspCandidate
from ibrobot_msgs.srv import DetectSegment, MoveToConfiguration, PlanGrasp, VerifyGrasp
from manipulation_execution.contact_compensation import ContactPrediction, compensate_contact_xy
from manipulation_execution.grasp_geometry import (
    CandidatePlan,
    FixedFingerBaseSide,
    FixedFingerEnvelope,
    build_candidate_plan,
    canonicalize_joint5,
    contact_distance_score,
    fixed_finger_base_side_alignment,
    fixed_finger_envelope_score,
    fixed_finger_robust_gap,
    grasp_axis_errors,
    joint5_closing_axis_correction,
    prepared_candidate_soft_score,
    quaternion_from_matrix,
    quaternion_matrix,
    source_contact_camera,
    transform_matrix,
    xyz_within_workspace,
)
from manipulation_execution.so101_geometry import (
    TablePlane,
    axis_error_deg,
    gripper_geometry_metrics_batch,
    gripper_mesh_min_z,
    quaternion_error_deg,
    tabletop_clearance,
    transform_point,
    transform_table_plane,
)

JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG = 20.0


@dataclass
class FlowState:
    completed_phases: list[str]
    pose_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    frame_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    verification_records: list[dict[str, Any]] = field(default_factory=list)
    attempt: int = 0
    verification_status: int = 0
    verification_confidence: float = 0.0
    debug_output_dir: str = ""


@dataclass(frozen=True)
class RankedCandidate:
    index: int
    candidate: GraspCandidate
    plan: CandidatePlan
    score: float
    contact_distance_m: float | None = None
    fixed_finger_base_side: FixedFingerBaseSide | None = None


@dataclass(frozen=True)
class IKPayload:
    joint_state: JointState
    ee_xyz: tuple[float, float, float]
    ee_quaternion: tuple[float, float, float, float]
    joint5_retry_applied: bool = False
    original_joint5: float | None = None
    approach_axis_error_deg: float | None = None
    closing_axis_error_deg: float | None = None


@dataclass(frozen=True)
class PreparedCandidate:
    ranked: RankedCandidate
    plan: CandidatePlan
    final_joint_state: JointState
    actual_ee_xyz: tuple[float, float, float]
    actual_ee_quaternion: tuple[float, float, float, float]
    contact_residual_xy_m: float
    contact_z_error_m: float
    approach_axis_error_deg: float | None
    closing_axis_error_deg: float
    tabletop_clearance_m: float | None
    mesh_min_z: float | None
    fixed_finger_envelope: FixedFingerEnvelope | None
    fk_fixed_finger_base_side: FixedFingerBaseSide | None
    selection_score: float


@dataclass(frozen=True)
class PlannerSceneGeometry:
    object_centroid_camera: tuple[float, float, float] | None = None
    table_normal_camera: tuple[float, float, float] | None = None
    table_offset_camera: float = 0.0
    table_inlier_ratio: float = 0.0
    object_top_camera: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class BaseSceneGeometry:
    table_plane: TablePlane | None = None
    object_top_base: tuple[float, float, float] | None = None


class PickFlowError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class PickCancelled(PickFlowError):
    def __init__(self) -> None:
        super().__init__("PICK_CANCELLED", "pick execution cancelled")


class PickExecutorNode(Node):
    """Plan, execute, and verify one object grasp."""

    _PHASE_PROGRESS = {
        "preflight": 0.02,
        "observe": 0.08,
        "planning": 0.18,
        "selecting": 0.32,
        "open": 0.38,
        "approach": 0.48,
        "pregrasp": 0.54,
        "descend": 0.60,
        "close": 0.68,
        "verify_close": 0.74,
        "probe_lift": 0.80,
        "verify_probe": 0.84,
        "lift": 0.92,
        "verify_lift": 0.97,
        "completed": 1.0,
    }

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("pick_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("action_name", "/manipulation/execute_pick")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("grasp_execution_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("rpc_timeout_sec", 5.0)

        self._action_name = self.get_parameter("action_name").value
        self._primitive_action_name = self.get_parameter("primitive_action_name").value
        self._config = self._load_json_object(self.get_parameter("grasp_execution_json").value)
        self._workspace = self._load_json_object(self.get_parameter("workspace_json").value)
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
        self._detect_service = str(self._config.get("detect_service", "/grounded_sam2/detect_and_segment"))
        self._move_configuration_service = str(
            self._config.get("move_configuration_service", "/moveit_gateway/move_to_configuration")
        )
        self._ik_service = str(self._config.get("ik_service", "/compute_ik"))
        self._fk_service = str(self._config.get("fk_service", "/compute_fk"))
        self._base_frame = str(self._config.get("base_frame", "base"))
        self._ee_frame = str(self._config.get("ee_frame", "gripper"))
        self._verification_policy = str(self._config.get("verification", "required")).lower()
        self._target_geometry = self._config.get("target_geometry", {})
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
        self._detect_client = self.create_client(DetectSegment, self._detect_service, callback_group=callback_group)
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

    @staticmethod
    def _load_json_object(raw_value: str) -> dict[str, Any]:
        parsed = json.loads(raw_value or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    @staticmethod
    def _load_json_list(raw_value: str) -> list[str]:
        parsed = json.loads(raw_value or "[]")
        if not isinstance(parsed, list):
            raise ValueError("expected JSON list")
        return [str(item) for item in parsed]

    def _handle_goal(self, goal_request):
        if not str(goal_request.target_query).strip():
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def _handle_joint_state(self, message: JointState) -> None:
        with self._joint_state_lock:
            self._latest_joint_state = message

    @staticmethod
    def _copy_joint_state(message: JointState) -> JointState:
        copied = JointState()
        copied.header = message.header
        copied.name = list(message.name)
        copied.position = [float(value) for value in message.position]
        copied.velocity = [float(value) for value in message.velocity]
        copied.effort = [float(value) for value in message.effort]
        return copied

    def _snapshot_joint_state(self) -> JointState | None:
        with self._joint_state_lock:
            if self._latest_joint_state is None:
                return None
            return self._copy_joint_state(self._latest_joint_state)

    @staticmethod
    def _handle_cancel(_goal_handle):
        return CancelResponse.ACCEPT

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    @staticmethod
    def _pose(xyz: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(value) for value in xyz)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            float(value) for value in quaternion
        )
        return pose

    @staticmethod
    def _pose_components(pose: Pose) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return (
            (float(pose.position.x), float(pose.position.y), float(pose.position.z)),
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )

    def _check_cancel(self, goal_handle) -> None:
        if goal_handle.is_cancel_requested:
            raise PickCancelled()

    def _publish_feedback(
        self,
        goal_handle,
        state: FlowState,
        phase: str,
        detail: str,
    ) -> None:
        self._check_cancel(goal_handle)
        if not state.completed_phases or state.completed_phases[-1] != phase:
            state.completed_phases.append(phase)
        feedback = PickObject.Feedback()
        feedback.phase = phase
        feedback.progress = float(self._PHASE_PROGRESS.get(phase, 0.0))
        feedback.attempt = int(state.attempt)
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _wait_future(self, future, goal_handle, deadline: float, timeout_sec: float, label: str):
        local_deadline = min(deadline, time.monotonic() + max(0.1, timeout_sec))
        while rclpy.ok() and not future.done():
            self._check_cancel(goal_handle)
            if time.monotonic() >= local_deadline:
                future.cancel()
                raise PickFlowError("RPC_TIMEOUT", f"{label} timed out")
            time.sleep(0.05)
        response = future.result()
        if response is None:
            raise PickFlowError("RPC_FAILED", f"{label} returned no response")
        return response

    def _wait_for_service(self, client, deadline: float, service_name: str, *, required: bool = True) -> bool:
        timeout = min(self._ready_timeout, self._remaining(deadline))
        ready = client.service_is_ready() or client.wait_for_service(timeout_sec=max(0.1, timeout))
        if not ready and required:
            raise PickFlowError("SERVICE_UNAVAILABLE", f"required service unavailable: {service_name}")
        return ready

    def _preflight(self, goal_handle, deadline: float, state: FlowState) -> None:
        self._publish_feedback(goal_handle, state, "preflight", "checking grasp services and safe primitive server")
        if not self._primitive_client.wait_for_server(timeout_sec=min(self._ready_timeout, self._remaining(deadline))):
            raise PickFlowError("PRIMITIVE_SERVER_UNAVAILABLE", self._primitive_action_name)
        self._wait_for_service(self._planner_client, deadline, self._planner_service)
        self._wait_for_service(self._detect_client, deadline, self._detect_service)
        self._wait_for_service(
            self._move_configuration_client,
            deadline,
            self._move_configuration_service,
        )
        verification_required = self._verification_policy == "required"
        self._wait_for_service(
            self._verifier_client,
            deadline,
            self._verifier_service,
            required=verification_required,
        )
        self._wait_for_service(self._ik_client, deadline, self._ik_service)
        self._wait_for_service(self._fk_client, deadline, self._fk_service)
        for index, (ik_client, fk_client) in enumerate(
            zip(self._ik_worker_clients, self._fk_worker_clients, strict=True)
        ):
            namespace = f"{self._ik_worker_prefix}_{index}"
            self._wait_for_service(ik_client, deadline, f"{namespace}/compute_ik")
            self._wait_for_service(fk_client, deadline, f"{namespace}/compute_fk")
        joint_state_deadline = min(deadline, time.monotonic() + self._ready_timeout)
        while self._snapshot_joint_state() is None:
            self._check_cancel(goal_handle)
            if time.monotonic() >= joint_state_deadline:
                raise PickFlowError(
                    "JOINT_STATE_UNAVAILABLE",
                    f"no current joint state received from {self._joint_state_topic}",
                )
            time.sleep(0.05)
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._lookup_base_to_camera(camera_frame)

    def _run_primitive(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        primitive_name: str,
        *,
        pose: Pose | None = None,
        pose_name: str = "",
        velocity_scaling: float = 0.0,
        gripper_position: float = 0.0,
        joint_state: JointState | None = None,
        duration_sec: float = 0.0,
    ) -> None:
        primitive_goal = PrimitiveCommand.Goal()
        primitive_goal.task_id = task_id
        primitive_goal.primitive_name = primitive_name
        primitive_goal.pose_name = pose_name
        if pose is not None:
            primitive_goal.target_pose = pose
        primitive_goal.velocity_scaling = float(velocity_scaling)
        primitive_goal.gripper_position = float(gripper_position)
        if joint_state is not None:
            position_by_name = dict(zip(joint_state.name, joint_state.position, strict=False))
            missing = [name for name in self._arm_joint_names if name not in position_by_name]
            if missing:
                raise PickFlowError(
                    "IK_JOINTS_MISSING", f"IK result missing arm joints: {', '.join(missing)}", retryable=True
                )
            primitive_goal.joint_names = list(self._arm_joint_names)
            primitive_goal.joint_positions = [float(position_by_name[name]) for name in self._arm_joint_names]
            primitive_goal.primitive_duration_sec = float(duration_sec)
        primitive_goal.timeout_sec = float(self._remaining(deadline))

        send_future = self._primitive_client.send_goal_async(primitive_goal)
        primitive_handle = self._wait_future(send_future, goal_handle, deadline, self._rpc_timeout, primitive_name)
        if not primitive_handle.accepted:
            raise PickFlowError("PRIMITIVE_REJECTED", f"primitive rejected: {primitive_name}", retryable=True)

        result_future = primitive_handle.get_result_async()
        try:
            action_result = self._wait_future(
                result_future,
                goal_handle,
                deadline,
                self._remaining(deadline),
                primitive_name,
            )
        except PickCancelled:
            primitive_handle.cancel_goal_async()
            raise
        except PickFlowError:
            primitive_handle.cancel_goal_async()
            raise
        result = action_result.result
        if not result.success:
            raise PickFlowError(
                result.error_code or "PRIMITIVE_FAILED",
                result.message or f"primitive failed: {primitive_name}",
                retryable=primitive_name in {"move_to_named_pose", "move_to_pose", "move_to_configuration"},
            )

    def _move_to_observe(self, goal_handle, deadline: float, state: FlowState, task_id: str) -> None:
        observe_pose = str(self._config.get("observe_pose", "observe_table"))
        if not observe_pose:
            return
        self._publish_feedback(goal_handle, state, "observe", f"moving to {observe_pose}")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_named_pose",
            pose_name=observe_pose,
            velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("observe_settle_sec", 0.6)))
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._record_frame_diagnostic(state, "observe_settled_latest", camera_frame)

    def _sleep_with_cancel(self, goal_handle, deadline: float, duration_sec: float) -> None:
        end = min(deadline, time.monotonic() + max(0.0, duration_sec))
        while time.monotonic() < end:
            self._check_cancel(goal_handle)
            time.sleep(0.05)

    @staticmethod
    def _table_geometry_from_response(response) -> tuple[tuple[float, float, float] | None, float, float]:
        if bool(response.execution_table_plane_found):
            normal_msg = response.execution_table_plane_normal
            offset = float(response.execution_table_plane_offset)
            inlier_ratio = float(response.execution_table_plane_inlier_ratio)
        elif bool(response.table_plane_found):
            normal_msg = response.table_plane_normal
            offset = float(response.table_plane_offset)
            inlier_ratio = float(response.table_plane_inlier_ratio)
        else:
            return None, 0.0, 0.0

        normal = (float(normal_msg.x), float(normal_msg.y), float(normal_msg.z))
        if not all(np.isfinite(value) for value in (*normal, offset, inlier_ratio)):
            return None, 0.0, 0.0
        if np.linalg.norm(normal) <= 1e-9:
            return None, 0.0, 0.0
        return normal, offset, inlier_ratio

    def _request_grasps(self, goal_handle, deadline: float, state: FlowState, target_query: str):
        self._publish_feedback(goal_handle, state, "planning", f"planning grasps for {target_query!r}")
        planner = self._config.get("planner", {})
        request = PlanGrasp.Request()
        request.text_prompt = target_query
        request.confidence_threshold = float(planner.get("confidence_threshold", 0.10))
        request.grasp_threshold = float(planner.get("grasp_threshold", 0.50))
        request.debug_output_mode = str(planner.get("debug_output_mode", "diagnostic"))
        future = self._planner_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            deadline,
            min(float(planner.get("timeout_sec", 120.0)), self._remaining(deadline)),
            "grasp planning",
        )
        state.debug_output_dir = str(response.debug_output_dir)
        if not response.success:
            raise PickFlowError("GRASP_PLANNING_FAILED", response.message)
        candidates = list(response.grasps.grasps)
        if not candidates:
            raise PickFlowError("NO_GRASP_CANDIDATES", "GraspGen returned no candidates")
        scoring = self._config.get("execution_scoring", {})
        centroid_source = str(scoring.get("centroid_source", "volume")).strip().lower()
        use_volume = centroid_source == "volume" and float(response.object_volume_m3) > 0.0
        centroid_msg = response.object_volume_centroid_xyz if use_volume else response.object_centroid_xyz
        object_centroid = None
        if int(response.object_point_count) > 0:
            values = (float(centroid_msg.x), float(centroid_msg.y), float(centroid_msg.z))
            if all(np.isfinite(value) for value in values):
                object_centroid = values
        if object_centroid is None and float(scoring.get("contact_distance_weight", 0.0)) > 0.0:
            object_centroid = self._request_fallback_detection_centroid(
                goal_handle,
                deadline,
                target_query,
                centroid_source,
                float(planner.get("confidence_threshold", 0.10)),
            )
        table_normal, table_offset, table_inlier_ratio = self._table_geometry_from_response(response)
        object_top = None
        top_values = (
            float(response.object_top_xyz.x),
            float(response.object_top_xyz.y),
            float(response.object_top_xyz.z),
        )
        if table_normal is not None and all(np.isfinite(value) for value in top_values):
            object_top = top_values
        scene = PlannerSceneGeometry(
            object_centroid_camera=object_centroid,
            table_normal_camera=table_normal,
            table_offset_camera=table_offset,
            table_inlier_ratio=table_inlier_ratio,
            object_top_camera=object_top,
        )
        return response.grasps.header, candidates, scene

    def _request_fallback_detection_centroid(
        self,
        goal_handle,
        deadline: float,
        target_query: str,
        centroid_source: str,
        confidence_threshold: float,
    ) -> tuple[float, float, float] | None:
        request = DetectSegment.Request()
        request.text_prompt = target_query
        request.confidence_threshold = confidence_threshold
        try:
            response = self._wait_future(
                self._detect_client.call_async(request),
                goal_handle,
                deadline,
                min(float(self._config.get("planner", {}).get("timeout_sec", 120.0)), self._remaining(deadline)),
                "fallback detection",
            )
        except PickFlowError as exc:
            self.get_logger().warning(f"fallback detection skipped: {exc}")
            return None
        if not response.success:
            self.get_logger().warning(f"fallback detection failed: {response.message}")
            return None
        minimum_points = int(self._config.get("candidate_selection", {}).get("min_point_count", 100))
        matching = [
            detection
            for detection in response.detections.detections
            if target_query.lower() in str(detection.label).lower() and int(detection.point_count) >= minimum_points
        ]
        if not matching:
            return None
        detection = max(matching, key=lambda item: (float(item.confidence), int(item.point_count)))
        use_volume = centroid_source == "volume" and float(detection.volume_m3) > 0.0
        centroid = detection.volume_centroid_xyz if use_volume else detection.centroid_xyz
        values = (float(centroid.x), float(centroid.y), float(centroid.z))
        if not all(np.isfinite(value) for value in values):
            return None
        self.get_logger().info(
            f"fallback detection centroid source={'volume' if use_volume else 'surface'} xyz={values}"
        )
        return values

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        if stamp is None:
            return 0
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_matrix(
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    def _lookup_base_transform(self, frame_id: str, stamp=None):
        if not frame_id:
            raise PickFlowError("GRASP_FRAME_MISSING", "grasp candidates have no frame_id")
        stamp_ns = self._stamp_to_ns(stamp)
        lookup_time = Time() if stamp_ns == 0 else Time.from_msg(stamp)
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame,
                frame_id,
                lookup_time,
                timeout=Duration(seconds=self._rpc_timeout),
            )
        except Exception as exc:
            lookup_label = "latest" if stamp_ns == 0 else f"stamp_ns={stamp_ns}"
            raise PickFlowError(
                "TF_UNAVAILABLE",
                f"cannot transform {frame_id} to {self._base_frame} at {lookup_label}: {exc}",
            ) from exc

    def _lookup_base_to_camera(self, frame_id: str, stamp=None) -> np.ndarray:
        return self._transform_to_matrix(self._lookup_base_transform(frame_id, stamp))

    @staticmethod
    def _matrix_diagnostic(matrix: np.ndarray) -> dict[str, Any]:
        return {
            "translation_xyz": [float(value) for value in matrix[:3, 3]],
            "quaternion_xyzw": [float(value) for value in quaternion_from_matrix(matrix)],
            "matrix_rowmajor": [float(value) for value in matrix.reshape(-1)],
        }

    def _record_frame_diagnostic(
        self,
        state: FlowState,
        label: str,
        camera_frame: str,
        stamp=None,
        *,
        camera_transform=None,
    ) -> None:
        config = self._config.get("frame_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return
        try:
            if camera_transform is None:
                camera_transform = self._lookup_base_transform(camera_frame, stamp)
            ee_transform = self._lookup_base_transform(self._ee_frame, stamp)
        except PickFlowError as exc:
            self.get_logger().warning(f"frame diagnostic skipped for {label}: {exc}")
            return

        base_to_camera = self._transform_to_matrix(camera_transform)
        base_to_ee = self._transform_to_matrix(ee_transform)
        ee_to_camera = np.linalg.inv(base_to_ee) @ base_to_camera
        requested_stamp_ns = self._stamp_to_ns(stamp)
        record = {
            "label": label,
            "base_frame": self._base_frame,
            "ee_frame": self._ee_frame,
            "camera_frame": camera_frame,
            "lookup_mode": "latest" if requested_stamp_ns == 0 else "capture_stamp",
            "requested_stamp_ns": requested_stamp_ns,
            "camera_transform_stamp_ns": self._stamp_to_ns(camera_transform.header.stamp),
            "ee_transform_stamp_ns": self._stamp_to_ns(ee_transform.header.stamp),
            "base_to_ee": self._matrix_diagnostic(base_to_ee),
            "base_to_camera": self._matrix_diagnostic(base_to_camera),
            "ee_to_camera": self._matrix_diagnostic(ee_to_camera),
        }
        state.frame_diagnostics.append(record)
        self.get_logger().info(
            f"frame diagnostic label={label} lookup={record['lookup_mode']} stamp_ns={requested_stamp_ns} "
            f"base_to_ee_xyz={record['base_to_ee']['translation_xyz']} "
            f"base_to_ee_q={record['base_to_ee']['quaternion_xyzw']} "
            f"base_to_camera_xyz={record['base_to_camera']['translation_xyz']}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "pick_frame_diagnostics.json"
            try:
                output_path.write_text(json.dumps(state.frame_diagnostics, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")

    @staticmethod
    def _scene_geometry_base(base_to_camera: np.ndarray, scene: PlannerSceneGeometry) -> BaseSceneGeometry:
        table_plane = None
        if scene.table_normal_camera is not None:
            table_plane = transform_table_plane(
                base_to_camera,
                scene.table_normal_camera,
                scene.table_offset_camera,
                inlier_ratio=scene.table_inlier_ratio,
            )
        object_top_base = None
        if scene.object_top_camera is not None:
            object_top_base = transform_point(base_to_camera, scene.object_top_camera)
        return BaseSceneGeometry(table_plane=table_plane, object_top_base=object_top_base)

    def _rank_candidates(
        self,
        default_frame: str,
        default_stamp,
        default_transform: np.ndarray,
        candidates: list[GraspCandidate],
        scene: PlannerSceneGeometry,
        scene_base: BaseSceneGeometry,
    ) -> list[RankedCandidate]:
        selection = self._config.get("candidate_selection", {})
        scoring = self._config.get("execution_scoring", {})
        min_confidence = float(selection.get("min_confidence", 0.0))
        require_collision_free = bool(selection.get("require_collision_free", False))
        min_contact_z = float(selection.get("min_contact_z", 0.0))
        min_approach_z = float(selection.get("min_approach_z", 0.04))
        min_topdown_score = float(selection.get("min_topdown_score", 0.0))
        topdown_min_z = float(selection.get("topdown_min_z", -0.25))
        min_topdown_dot = max(0.0, min(0.999, -topdown_min_z))
        confidence_weight = float(scoring.get("confidence_weight", selection.get("confidence_weight", 1.0)))
        topdown_weight = float(scoring.get("topdown_weight", selection.get("topdown_weight", 0.35)))
        contact_weight = max(0.0, float(scoring.get("contact_distance_weight", 0.0)))
        contact_scale = max(1e-6, float(scoring.get("contact_distance_scale_m", 0.06)))
        source_contact = self._config.get("source_contact_point", [0.0, 0.0, 0.195])
        target_gripper = self._config.get("target_gripper", {})
        base_side_config = target_gripper.get("fixed_finger_base_side", {})
        check_fixed_finger_base_side = (
            bool(base_side_config.get("enabled", False))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
        )
        minimum_base_side_alignment = max(
            -1.0,
            min(1.0, float(base_side_config.get("min_alignment_cos", 0.0))),
        )
        max_candidates = int(selection.get("max_candidates", 80))
        default_stamp_ns = self._stamp_to_ns(default_stamp)
        transforms: dict[tuple[str, int], np.ndarray] = {(default_frame, default_stamp_ns): default_transform}
        ranked: list[RankedCandidate] = []

        for index, candidate in enumerate(candidates):
            confidence = float(candidate.confidence)
            if confidence < min_confidence:
                continue
            if require_collision_free and not bool(candidate.collision_free):
                continue
            frame_id = candidate.header.frame_id or default_frame
            candidate_stamp = candidate.header.stamp
            if self._stamp_to_ns(candidate_stamp) == 0:
                candidate_stamp = default_stamp
            transform_key = (frame_id, self._stamp_to_ns(candidate_stamp))
            if transform_key not in transforms:
                transforms[transform_key] = self._lookup_base_to_camera(frame_id, candidate_stamp)
            try:
                plan = build_candidate_plan(
                    candidate.pose_matrix,
                    transforms[transform_key],
                    float(candidate.target_width_m),
                    float(candidate.target_width_quality),
                    self._config,
                    width_axis_camera=candidate.width_axis_camera,
                    target_width_min_offset_m=float(candidate.target_width_min_offset_m),
                    target_width_max_offset_m=float(candidate.target_width_max_offset_m),
                )
            except (ValueError, ArithmeticError):
                continue
            normalized_topdown = max(
                0.0,
                min(1.0, (plan.topdown_score - min_topdown_dot) / (1.0 - min_topdown_dot)),
            )
            plan = replace(plan, topdown_score=normalized_topdown)
            if (
                plan.target_contact_base[2] < min_contact_z
                or plan.approach[2] < min_approach_z
                or plan.topdown_score < min_topdown_score
            ):
                continue
            if any(not xyz_within_workspace(xyz, self._workspace)[0] for xyz in (plan.approach, plan.grasp, plan.lift)):
                continue
            fixed_finger_base_side = None
            if check_fixed_finger_base_side:
                if plan.target_width_min_base is None or plan.target_width_max_base is None:
                    continue
                try:
                    fixed_finger_base_side = fixed_finger_base_side_alignment(
                        plan.grasp,
                        plan.quaternion,
                        target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                        plan.target_width_min_base,
                        plan.target_width_max_base,
                        base_side_config.get("reference_point_base", [0.0, 0.0, 0.0]),
                    )
                except ValueError:
                    continue
                if fixed_finger_base_side.alignment_cos < minimum_base_side_alignment:
                    continue
            contact_distance = None
            contact_score = 0.0
            if scene.object_centroid_camera is not None:
                contact = source_contact_camera(candidate.pose_matrix, source_contact)
                contact_distance, contact_score = contact_distance_score(
                    contact,
                    scene.object_centroid_camera,
                    contact_scale,
                )
            score = (
                confidence_weight * confidence + topdown_weight * plan.topdown_score + contact_weight * contact_score
            )
            ranked.append(
                RankedCandidate(
                    index=index,
                    candidate=candidate,
                    plan=plan,
                    score=score,
                    contact_distance_m=contact_distance,
                    fixed_finger_base_side=fixed_finger_base_side,
                )
            )

        if bool(self._target_geometry.get("tabletop_filter", False)) and ranked:
            if self._mesh_directory is None or scene_base.table_plane is None:
                raise PickFlowError(
                    "TARGET_TABLETOP_UNAVAILABLE",
                    "SO101 tabletop filter requires target meshes and a fitted table plane",
                )
            minimum_clearance = float(self._target_geometry.get("tabletop_clearance_m", 0.0))
            try:
                geometry_metrics = gripper_geometry_metrics_batch(
                    self._mesh_directory,
                    [
                        (
                            item.plan.approach,
                            item.plan.grasp,
                            item.plan.quaternion,
                            float(item.candidate.target_width_m),
                        )
                        for item in ranked
                    ],
                    scene_base.table_plane,
                    clearance_threshold_m=minimum_clearance,
                )
            except Exception as exc:
                raise PickFlowError("TARGET_GEOMETRY_FAILED", str(exc)) from exc
            ranked = [
                item
                for item, (_, clearance) in zip(ranked, geometry_metrics, strict=True)
                if clearance is not None and clearance >= minimum_clearance
            ]

        ranked.sort(
            key=lambda item: (
                -item.score,
                float("inf") if item.contact_distance_m is None else item.contact_distance_m,
                -float(item.candidate.confidence),
                item.index,
            )
        )
        if max_candidates > 0:
            ranked = ranked[:max_candidates]
        if not ranked:
            raise PickFlowError("NO_SAFE_GRASP_CANDIDATES", "no candidate passed geometry and workspace checks")
        return ranked

    def _solve_ik(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        client=None,
    ) -> JointState:
        client = self._ik_client if client is None else client
        ik_config = self._config.get("ik", {})
        request = GetPositionIK.Request()
        request.ik_request.group_name = str(ik_config.get("group_name", "arm"))
        request.ik_request.ik_link_name = self._ee_frame
        request.ik_request.pose_stamped.header.frame_id = self._base_frame
        request.ik_request.pose_stamped.pose = pose
        request.ik_request.avoid_collisions = bool(ik_config.get("avoid_collisions", False))
        if seed is not None:
            request.ik_request.robot_state.joint_state = seed
        ik_timeout = float(ik_config.get("timeout_sec", 2.0))
        request.ik_request.timeout = Duration(seconds=ik_timeout).to_msg()
        future = client.call_async(request)
        response = self._wait_future(future, goal_handle, deadline, ik_timeout + 1.0, "IK")
        if int(response.error_code.val) != 1:
            raise PickFlowError("IK_FAILED", f"IK failed with code {response.error_code.val}", retryable=True)
        return response.solution.joint_state

    def _compute_fk(self, joint_state: JointState, goal_handle, deadline: float, *, client=None) -> Pose:
        client = self._fk_client if client is None else client
        request = GetPositionFK.Request()
        request.header.frame_id = self._base_frame
        request.fk_link_names = [self._ee_frame]
        request.robot_state.joint_state = joint_state
        future = client.call_async(request)
        response = self._wait_future(future, goal_handle, deadline, self._rpc_timeout, "FK")
        if int(response.error_code.val) != 1 or not response.pose_stamped:
            raise PickFlowError("FK_FAILED", f"FK failed with code {response.error_code.val}", retryable=True)
        return response.pose_stamped[0].pose

    @staticmethod
    def _joint_position(joint_state: JointState, joint_name: str) -> float | None:
        positions = dict(zip(joint_state.name, joint_state.position, strict=False))
        value = positions.get(joint_name)
        return None if value is None else float(value)

    @staticmethod
    def _joint_state_with_joint5(joint_state: JointState, joint5: float) -> JointState:
        seed = PickExecutorNode._copy_joint_state(joint_state)
        seed.position = [
            float(joint5) if str(name) == "5" else float(position)
            for name, position in zip(joint_state.name, joint_state.position, strict=False)
        ]
        return seed

    def _orientation_guard(self) -> dict[str, Any]:
        target_gripper = self._config.get("target_gripper", {})
        guard = target_gripper.get("ik_orientation_guard", {})
        return guard if isinstance(guard, dict) else {}

    def _joint5_abs_max(self) -> float | None:
        value = self._orientation_guard().get("joint5_abs_max")
        if value is None:
            return None
        limit = float(value)
        if not math.isfinite(limit) or limit <= 0.0:
            raise PickFlowError("INVALID_GRASP_CONFIG", "ik_orientation_guard.joint5_abs_max must be positive")
        return limit

    def _validate_joint5(self, joint_state: JointState) -> float | None:
        limit = self._joint5_abs_max()
        if limit is None:
            return None
        value = self._joint_position(joint_state, "5")
        if value is None:
            raise PickFlowError("IK_JOINT5_MISSING", "IK result has no joint 5", retryable=True)
        if abs(value) > limit:
            raise PickFlowError(
                "IK_JOINT5_LIMIT",
                f"joint 5 absolute value {abs(value):.4f} exceeds {limit:.4f}",
                retryable=True,
            )
        return value

    def _apply_joint5_retry_if_needed(
        self,
        pose: Pose,
        solution: JointState,
        goal_handle,
        deadline: float,
        *,
        ik_client=None,
    ) -> tuple[JointState, float | None]:
        limit = self._joint5_abs_max()
        if limit is None:
            return solution, None
        original_joint5 = self._joint_position(solution, "5")
        if original_joint5 is None:
            return solution, None
        if abs(original_joint5) <= math.pi / 2.0:
            return solution, None

        bounded_joint5 = canonicalize_joint5(original_joint5)
        retry_seed = self._joint_state_with_joint5(solution, bounded_joint5)
        retry_solution = self._solve_ik(pose, goal_handle, deadline, retry_seed, client=ik_client)
        retry_joint5 = self._joint_position(retry_solution, "5")
        if retry_joint5 is None or abs(retry_joint5) > limit:
            raise PickFlowError(
                "IK_JOINT5_RETRY_FAILED",
                f"joint 5 retry did not enter limit {limit:.4f}: {retry_joint5}",
                retryable=True,
            )
        return retry_solution, original_joint5

    def _grasp_orientation_errors(
        self,
        target_quaternion: tuple[float, float, float, float],
        actual_quaternion: tuple[float, float, float, float],
    ):
        guard = self._orientation_guard()
        if not bool(guard.get("enabled", False)):
            return None
        target_gripper = self._config.get("target_gripper", {})
        return grasp_axis_errors(
            target_quaternion,
            actual_quaternion,
            guard.get("approach_axis_ee", [0.0, 0.0, 1.0]),
            target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
            closing_axis_180_symmetric=bool(guard.get("closing_axis_180_symmetric", False)),
        )

    def _orientation_limits(self) -> tuple[float, float]:
        guard = self._orientation_guard()
        max_approach = float(guard.get("max_approach_error_deg", 25.0))
        max_closing = min(
            float(guard.get("max_closing_error_deg", JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG)),
            JOINT5_ORIENTATION_MAX_CLOSING_ERROR_DEG,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 180.0 for value in (max_approach, max_closing)):
            raise PickFlowError(
                "INVALID_GRASP_CONFIG",
                "ik_orientation_guard axis error limits must be within [0, 180] degrees",
            )
        return max_approach, max_closing

    def _solve_grasp_ik_fk(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        validate_orientation: bool = True,
        ik_client=None,
        fk_client=None,
    ) -> IKPayload:
        joint_state = self._solve_ik(pose, goal_handle, deadline, seed, client=ik_client)
        joint_state, original_joint5 = self._apply_joint5_retry_if_needed(
            pose,
            joint_state,
            goal_handle,
            deadline,
            ik_client=ik_client,
        )
        self._validate_joint5(joint_state)
        fk_pose = self._compute_fk(joint_state, goal_handle, deadline, client=fk_client)
        ee_xyz, ee_quaternion = self._pose_components(fk_pose)
        _, target_quaternion = self._pose_components(pose)
        errors = self._grasp_orientation_errors(target_quaternion, ee_quaternion)
        approach_error = None if errors is None else errors.approach_deg
        closing_error = None if errors is None else errors.closing_deg
        if validate_orientation and errors is not None:
            max_approach, max_closing = self._orientation_limits()
            if errors.approach_deg > max_approach or errors.closing_deg > max_closing:
                raise PickFlowError(
                    "IK_ORIENTATION_REJECTED",
                    f"FK orientation error exceeds limits: approach={errors.approach_deg:.3f}/{max_approach:.3f} "
                    f"closing={errors.closing_deg:.3f}/{max_closing:.3f}",
                    retryable=True,
                )
        return IKPayload(
            joint_state=joint_state,
            ee_xyz=ee_xyz,
            ee_quaternion=ee_quaternion,
            joint5_retry_applied=original_joint5 is not None,
            original_joint5=original_joint5,
            approach_axis_error_deg=approach_error,
            closing_axis_error_deg=closing_error,
        )

    def _solve_orientation_consistent_grasp_ik_fk(
        self,
        pose: Pose,
        goal_handle,
        deadline: float,
        seed: JointState | None = None,
        *,
        ik_client=None,
        fk_client=None,
    ) -> IKPayload:
        guard = self._orientation_guard()
        if not bool(guard.get("enabled", False)):
            return self._solve_grasp_ik_fk(
                pose,
                goal_handle,
                deadline,
                seed,
                ik_client=ik_client,
                fk_client=fk_client,
            )

        _, target_quaternion = self._pose_components(pose)
        target_gripper = self._config.get("target_gripper", {})
        max_approach, max_closing = self._orientation_limits()
        current_seed = seed
        seen_joint5: set[float] = set()
        last_reason = "orientation correction was not attempted"
        for attempt in range(3):
            payload = self._solve_grasp_ik_fk(
                pose,
                goal_handle,
                deadline,
                current_seed,
                validate_orientation=False,
                ik_client=ik_client,
                fk_client=fk_client,
            )
            joint5 = self._joint_position(payload.joint_state, "5")
            approach_error = payload.approach_axis_error_deg
            closing_error = payload.closing_axis_error_deg
            if joint5 is None or approach_error is None or closing_error is None:
                last_reason = "orientation correction requires joint 5 and FK axis errors"
                break
            if approach_error <= max_approach and closing_error <= max_closing:
                return payload
            last_reason = (
                f"attempt={attempt} approach={approach_error:.3f}/{max_approach:.3f} "
                f"closing={closing_error:.3f}/{max_closing:.3f}"
            )
            try:
                correction = joint5_closing_axis_correction(
                    target_quaternion,
                    payload.ee_quaternion,
                    guard.get("approach_axis_ee", [0.0, 0.0, 1.0]),
                    target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                    closing_axis_180_symmetric=bool(guard.get("closing_axis_180_symmetric", False)),
                )
            except ValueError as exc:
                last_reason = str(exc)
                break
            corrected_joint5 = canonicalize_joint5(joint5 + correction)
            correction_key = round(corrected_joint5, 9)
            if correction_key in seen_joint5 or abs(corrected_joint5 - joint5) <= 1e-6:
                break
            seen_joint5.add(round(joint5, 9))
            current_seed = self._joint_state_with_joint5(payload.joint_state, corrected_joint5)
        raise PickFlowError(
            "IK_ORIENTATION_REJECTED",
            f"no orientation-consistent joint 5 branch: {last_reason}",
            retryable=True,
        )

    def _validate_joint5_branch_continuity(self, seed: JointState | None, solution: JointState) -> None:
        if seed is None or self._joint5_abs_max() is None:
            return
        seed_joint5 = self._joint_position(seed, "5")
        solution_joint5 = self._joint_position(solution, "5")
        if seed_joint5 is None or solution_joint5 is None:
            return
        delta = abs(solution_joint5 - seed_joint5)
        if delta > math.pi / 2.0:
            raise PickFlowError(
                "IK_JOINT5_BRANCH_CHANGED",
                f"joint 5 branch changed by {delta:.4f} rad ({seed_joint5:.4f} -> {solution_joint5:.4f})",
                retryable=True,
            )

    @staticmethod
    def _contact_for_pose(pose: Pose, contact_ee: tuple[float, float, float]) -> tuple[float, float, float]:
        xyz, quaternion = PickExecutorNode._pose_components(pose)
        rotation = quaternion_matrix(quaternion)
        contact = np.asarray(xyz, dtype=np.float64) + rotation @ np.asarray(contact_ee, dtype=np.float64)
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    def _validate_fk_fixed_finger_base_side(
        self,
        candidate_index: int,
        plan: CandidatePlan,
        payload: IKPayload,
    ) -> FixedFingerBaseSide | None:
        target_gripper = self._config.get("target_gripper", {})
        base_side_config = target_gripper.get("fixed_finger_base_side", {})
        if not (
            bool(base_side_config.get("enabled", False))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
        ):
            return None
        if plan.target_width_min_base is None or plan.target_width_max_base is None:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
                f"candidate {candidate_index}: target width extent is unavailable for final FK fixed-finger check",
                retryable=True,
            )
        try:
            alignment = fixed_finger_base_side_alignment(
                payload.ee_xyz,
                payload.ee_quaternion,
                target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                plan.target_width_min_base,
                plan.target_width_max_base,
                base_side_config.get("reference_point_base", [0.0, 0.0, 0.0]),
            )
        except ValueError as exc:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
                f"candidate {candidate_index}: final FK fixed-finger check failed: {exc}",
                retryable=True,
            ) from exc
        minimum_alignment = max(-1.0, min(1.0, float(base_side_config.get("min_alignment_cos", 0.0))))
        if alignment.alignment_cos < minimum_alignment:
            raise PickFlowError(
                "FK_FIXED_FINGER_BASE_SIDE_REJECTED",
                f"candidate {candidate_index}: final FK fixed finger is on the outer side "
                f"(alignment={alignment.alignment_cos:.3f} < {minimum_alignment:.3f}, "
                f"inward_offset={alignment.inward_offset_m:.4f}m)",
                retryable=True,
            )
        return alignment

    def _prepare_candidate(
        self,
        ranked: RankedCandidate,
        scene_base: BaseSceneGeometry,
        goal_handle,
        deadline: float,
        *,
        apply_compensation: bool = False,
        initial_seed: JointState | None = None,
        ik_client=None,
        fk_client=None,
    ) -> PreparedCandidate:
        plan = ranked.plan
        compensation = self._config.get("contact_compensation", {})
        payload: IKPayload
        contact_residual_xy = 0.0
        z_error = 0.0
        compensation_enabled = bool(compensation.get("enabled", True))
        if compensation_enabled and apply_compensation:

            def _predict(command_xyz, previous_payload):
                seed = previous_payload.joint_state if previous_payload is not None else initial_seed
                command_pose = self._pose(command_xyz, plan.quaternion)
                predicted = self._solve_grasp_ik_fk(
                    command_pose,
                    goal_handle,
                    deadline,
                    seed,
                    ik_client=ik_client,
                    fk_client=fk_client,
                )
                return ContactPrediction(
                    contact_base=self._contact_for_pose(
                        self._pose(predicted.ee_xyz, predicted.ee_quaternion),
                        plan.target_contact_ee,
                    ),
                    payload=predicted,
                )

            result = compensate_contact_xy(
                plan.grasp,
                plan.target_contact_base,
                _predict,
                tolerance_m=float(compensation.get("xy_tolerance_m", 0.003)),
                max_iterations=int(compensation.get("max_iterations", 6)),
                max_correction_m=float(compensation.get("max_correction_m", 0.03)),
            )
            if not result.converged:
                raise PickFlowError(
                    "CONTACT_COMPENSATION_FAILED",
                    f"candidate {ranked.index}: {result.reason}",
                    retryable=True,
                )
            z_error = float(plan.target_contact_base[2] - result.prediction.contact_base[2])
            selection_min_contact_z = float(self._config.get("candidate_selection", {}).get("min_contact_z", 0.0))
            if result.prediction.contact_base[2] < selection_min_contact_z:
                raise PickFlowError(
                    "IK_FK_PREDICTED_CONTACT_Z",
                    f"candidate {ranked.index}: predicted contact z {result.prediction.contact_base[2]:.4f} "
                    f"< min_contact_z {selection_min_contact_z:.4f}",
                    retryable=True,
                )
            correction_x = result.correction_x
            correction_y = result.correction_y
            plan = replace(
                plan,
                grasp=result.command_xyz,
                lift=(plan.lift[0] + correction_x, plan.lift[1] + correction_y, plan.lift[2]),
            )
            payload = result.prediction.payload
            contact_residual_xy = math.hypot(float(result.residual_x), float(result.residual_y))
        else:
            payload = self._solve_orientation_consistent_grasp_ik_fk(
                self._pose(plan.grasp, plan.quaternion),
                goal_handle,
                deadline,
                initial_seed,
                ik_client=ik_client,
                fk_client=fk_client,
            )
            predicted_contact = self._contact_for_pose(
                self._pose(payload.ee_xyz, payload.ee_quaternion),
                plan.target_contact_ee,
            )
            contact_residual_xy = math.hypot(
                plan.target_contact_base[0] - predicted_contact[0],
                plan.target_contact_base[1] - predicted_contact[1],
            )
            z_error = float(plan.target_contact_base[2] - predicted_contact[2])

            if compensation_enabled:
                max_correction = max(0.0, float(compensation.get("max_correction_m", 0.03)))
                residual_x = float(plan.target_contact_base[0] - predicted_contact[0])
                residual_y = float(plan.target_contact_base[1] - predicted_contact[1])
                if abs(residual_x) > max_correction or abs(residual_y) > max_correction:
                    raise PickFlowError(
                        "CONTACT_COMPENSATION_FAILED",
                        f"candidate {ranked.index}: contact residual x={residual_x:.4f} y={residual_y:.4f} "
                        f"exceeds {max_correction:.4f}",
                        retryable=True,
                    )

        if compensation_enabled and abs(z_error) > float(compensation.get("max_z_error_m", 0.015)):
            raise PickFlowError(
                "CONTACT_Z_ERROR",
                f"candidate {ranked.index}: contact z error {z_error:.4f}",
                retryable=True,
            )

        fk_fixed_finger_base_side = self._validate_fk_fixed_finger_base_side(ranked.index, plan, payload)

        check_orientation = bool(self._config.get("ik", {}).get("check_orientation", False))
        position_only_quaternion = (0.0, 0.0, 0.0, 1.0)
        for label, xyz in (("approach", plan.approach), ("lift", plan.lift)):
            allowed, reason = xyz_within_workspace(xyz, self._workspace)
            if not allowed:
                raise PickFlowError("WORKSPACE_REJECTED", f"candidate {ranked.index} {label}: {reason}", retryable=True)
            ik_quaternion = plan.quaternion if check_orientation else position_only_quaternion
            self._solve_ik(
                self._pose(xyz, ik_quaternion),
                goal_handle,
                deadline,
                initial_seed,
                client=ik_client,
            )

        closing_axis = self._config.get("target_gripper", {}).get("closing_axis_ee", [1.0, 0.0, 0.0])
        closing_error = payload.closing_axis_error_deg
        if closing_error is None:
            closing_error = axis_error_deg(plan.quaternion, payload.ee_quaternion, closing_axis)

        mesh_min_z = None
        actual_tabletop_clearance = None
        if self._mesh_directory is not None:
            try:
                mesh_min_z = gripper_mesh_min_z(
                    self._mesh_directory,
                    payload.ee_xyz,
                    payload.ee_quaternion,
                    float(ranked.candidate.target_width_m),
                )
                if bool(self._target_geometry.get("tabletop_filter", False)):
                    if scene_base.table_plane is None:
                        raise PickFlowError("TARGET_TABLETOP_UNAVAILABLE", "fitted table plane is unavailable")
                    actual_tabletop_clearance = tabletop_clearance(
                        self._mesh_directory,
                        payload.ee_xyz,
                        payload.ee_xyz,
                        payload.ee_quaternion,
                        float(ranked.candidate.target_width_m),
                        scene_base.table_plane,
                        sweep_steps=1,
                    )
            except PickFlowError:
                raise
            except Exception as exc:
                raise PickFlowError("TARGET_GEOMETRY_FAILED", str(exc), retryable=True) from exc
            minimum_clearance = float(self._target_geometry.get("tabletop_clearance_m", 0.0))
            if actual_tabletop_clearance is not None and actual_tabletop_clearance < minimum_clearance:
                raise PickFlowError(
                    "TARGET_TABLETOP_COLLISION",
                    f"candidate {ranked.index}: FK tabletop clearance {actual_tabletop_clearance:.4f}m "
                    f"< {minimum_clearance:.4f}m",
                    retryable=True,
                )

        prepared_scoring = self._config.get("prepared_candidate_scoring", {})
        envelope = None
        target_gripper = self._config.get("target_gripper", {})
        robust_gap_config = target_gripper.get("fixed_finger_robust_gap", {})
        if (
            (bool(prepared_scoring.get("enabled", False)) or bool(robust_gap_config.get("enabled", False)))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
            and plan.target_width_min_base is not None
            and plan.target_width_max_base is not None
        ):
            envelope = fixed_finger_envelope_score(
                plan.grasp,
                plan.quaternion,
                target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                plan.target_width_min_base,
                plan.target_width_max_base,
                plan.fixed_finger_target_gap_m,
                gap_sigma_m=float(prepared_scoring.get("fixed_finger_gap_sigma_m", 0.006)),
                reliable_max_opening_m=float(prepared_scoring.get("reliable_max_opening_m", 0.072)),
                moving_min_clearance_m=float(prepared_scoring.get("moving_finger_min_clearance_m", 0.003)),
                fixed_score_weight=float(prepared_scoring.get("fixed_finger_score_weight", 0.80)),
            )
        selection_score = prepared_candidate_soft_score(
            prepared_scoring,
            fixed_finger_envelope=None if envelope is None else envelope.score,
            contact_residual_xy_m=contact_residual_xy,
            contact_z_error_m=abs(z_error),
            confidence=float(ranked.candidate.confidence),
            centroid_distance_m=ranked.contact_distance_m,
        )

        return PreparedCandidate(
            ranked=ranked,
            plan=plan,
            final_joint_state=payload.joint_state,
            actual_ee_xyz=payload.ee_xyz,
            actual_ee_quaternion=payload.ee_quaternion,
            contact_residual_xy_m=contact_residual_xy,
            contact_z_error_m=abs(z_error),
            approach_axis_error_deg=payload.approach_axis_error_deg,
            closing_axis_error_deg=closing_error,
            tabletop_clearance_m=actual_tabletop_clearance,
            mesh_min_z=mesh_min_z,
            fixed_finger_envelope=envelope,
            fk_fixed_finger_base_side=fk_fixed_finger_base_side,
            selection_score=selection_score,
        )

    def _verify_ik_worker_pool(
        self,
        candidate: RankedCandidate,
        joint_seed: JointState,
        goal_handle,
        deadline: float,
    ) -> None:
        if not self._ik_worker_clients:
            return
        check_orientation = bool(self._config.get("ik", {}).get("check_orientation", False))
        quaternion = candidate.plan.quaternion if check_orientation else (0.0, 0.0, 0.0, 1.0)
        pose = self._pose(candidate.plan.approach, quaternion)
        primary = self._solve_ik(pose, goal_handle, deadline, joint_seed)
        worker = self._solve_ik(
            pose,
            goal_handle,
            deadline,
            joint_seed,
            client=self._ik_worker_clients[0],
        )
        primary_positions = dict(zip(primary.name, primary.position, strict=False))
        worker_positions = dict(zip(worker.name, worker.position, strict=False))
        common_names = sorted(primary_positions.keys() & worker_positions.keys())
        if not common_names:
            raise PickFlowError("IK_WORKER_MISMATCH", "primary and worker IK returned no common joints")
        max_delta = max(abs(float(primary_positions[name]) - float(worker_positions[name])) for name in common_names)
        if max_delta > 1e-8:
            raise PickFlowError(
                "IK_WORKER_MISMATCH",
                f"primary and worker IK differ by {max_delta:.12f} rad",
            )
        self.get_logger().info(
            f"IK worker verification passed: workers={self._ik_worker_count} max_joint_delta={max_delta:.12f}"
        )

    def _prepare_ranked_candidates(
        self,
        ranked: list[RankedCandidate],
        scene_base: BaseSceneGeometry,
        joint_seed: JointState,
        goal_handle,
        deadline: float,
    ) -> tuple[list[PreparedCandidate], PickFlowError | None]:
        started = time.monotonic()
        if not self._ik_worker_clients:
            results: list[PreparedCandidate | PickFlowError] = []
            for candidate in ranked:
                try:
                    results.append(
                        self._prepare_candidate(
                            candidate,
                            scene_base,
                            goal_handle,
                            deadline,
                            initial_seed=joint_seed,
                        )
                    )
                except PickFlowError as exc:
                    results.append(exc)
        else:
            self._verify_ik_worker_pool(ranked[0], joint_seed, goal_handle, deadline)
            worker_count = min(len(self._ik_worker_clients), len(ranked))
            partitions: list[list[tuple[int, RankedCandidate]]] = [[] for _ in range(worker_count)]
            for position, candidate in enumerate(ranked):
                partitions[position % worker_count].append((position, candidate))

            ordered_results: list[PreparedCandidate | PickFlowError | None] = [None] * len(ranked)

            def prepare_partition(worker_index: int, partition: list[tuple[int, RankedCandidate]]):
                partition_results = []
                for position, candidate in partition:
                    try:
                        result = self._prepare_candidate(
                            candidate,
                            scene_base,
                            goal_handle,
                            deadline,
                            initial_seed=joint_seed,
                            ik_client=self._ik_worker_clients[worker_index],
                            fk_client=self._fk_worker_clients[worker_index],
                        )
                    except PickFlowError as exc:
                        result = exc
                    partition_results.append((position, result))
                return partition_results

            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pick-candidate-ik") as pool:
                jobs = [pool.submit(prepare_partition, index, partition) for index, partition in enumerate(partitions)]
                for job in jobs:
                    for position, result in job.result():
                        ordered_results[position] = result
            if any(result is None for result in ordered_results):
                raise PickFlowError("IK_WORKER_INCOMPLETE", "parallel IK worker pool returned incomplete results")
            results = [result for result in ordered_results if result is not None]
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_ik_fk duration_s={time.monotonic() - started:.3f} "
                f"workers={worker_count} candidates={len(ranked)}"
            )

        if not self._ik_worker_clients:
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_ik_fk duration_s={time.monotonic() - started:.3f} "
                f"workers=0 candidates={len(ranked)}"
            )

        prepared_candidates: list[PreparedCandidate] = []
        last_error: PickFlowError | None = None
        for candidate, result in zip(ranked, results, strict=True):
            if isinstance(result, PickCancelled):
                raise result
            if isinstance(result, PickFlowError):
                last_error = result
                self.get_logger().warning(
                    f"pick candidate preparation failed: candidate={candidate.index} "
                    f"code={result.code} retryable={result.retryable} message={result}"
                )
                if not result.retryable:
                    raise result
                continue
            prepared_candidates.append(result)
        return prepared_candidates, last_error

    def _record_prepared_ranking(self, state: FlowState, candidates: list[PreparedCandidate]) -> None:
        if not state.debug_output_dir:
            return
        records = []
        for rank, item in enumerate(candidates, start=1):
            envelope = item.fixed_finger_envelope
            records.append(
                {
                    "rank": rank,
                    "candidate_index": item.ranked.index,
                    "selection_score": item.selection_score,
                    "fixed_finger_envelope_score": None if envelope is None else envelope.score,
                    "fixed_finger_gap_score": None if envelope is None else envelope.fixed_score,
                    "fixed_finger_gap_m": None if envelope is None else envelope.fixed_gap_m,
                    "fixed_finger_target_gap_m": None if envelope is None else envelope.target_gap_m,
                    "moving_finger_gap_m": None if envelope is None else envelope.moving_gap_m,
                    "moving_finger_gap_score": None if envelope is None else envelope.moving_score,
                    "contact_residual_xy_m": item.contact_residual_xy_m,
                    "contact_z_error_m": item.contact_z_error_m,
                    "ik_fk_approach_axis_error_deg": item.approach_axis_error_deg,
                    "ik_fk_closing_axis_error_deg": item.closing_axis_error_deg,
                    "ik_grasp_joint5": self._joint_position(item.final_joint_state, "5"),
                    "centroid_distance_m": item.ranked.contact_distance_m,
                    "confidence": float(item.ranked.candidate.confidence),
                    "source_rank_score": item.ranked.score,
                    "target_width_m": float(item.ranked.candidate.target_width_m),
                    "target_width_quality": float(item.ranked.candidate.target_width_quality),
                    "target_width_min_offset_m": float(item.ranked.candidate.target_width_min_offset_m),
                    "target_width_max_offset_m": float(item.ranked.candidate.target_width_max_offset_m),
                    "fixed_finger_base_side_alignment_cos": (
                        None
                        if item.ranked.fixed_finger_base_side is None
                        else item.ranked.fixed_finger_base_side.alignment_cos
                    ),
                    "fixed_finger_inward_offset_m": (
                        None
                        if item.ranked.fixed_finger_base_side is None
                        else item.ranked.fixed_finger_base_side.inward_offset_m
                    ),
                    "fk_fixed_finger_base_side_alignment_cos": (
                        None if item.fk_fixed_finger_base_side is None else item.fk_fixed_finger_base_side.alignment_cos
                    ),
                    "fk_fixed_finger_inward_offset_m": (
                        None
                        if item.fk_fixed_finger_base_side is None
                        else item.fk_fixed_finger_base_side.inward_offset_m
                    ),
                }
            )
        output_path = Path(state.debug_output_dir) / "prepared_candidate_ranking.json"
        try:
            output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.get_logger().warning(f"failed to write {output_path}: {exc}")

    def _record_verification(
        self,
        state: FlowState,
        phase: str,
        candidate_index: int,
        candidate: GraspCandidate,
        response: VerifyGrasp.Response,
    ) -> None:
        status = int(response.status)
        record = {
            "label": {
                "verify_close": "close",
                "verify_probe": "probe_lift",
                "verify_lift": "lift",
            }.get(phase, phase.removeprefix("verify_")),
            "success": bool(response.success),
            "status": status,
            "status_name": {
                VerifyGrasp.Response.STATUS_FAILED: "failed",
                VerifyGrasp.Response.STATUS_SUCCESS: "success",
                VerifyGrasp.Response.STATUS_UNCERTAIN: "uncertain",
            }.get(status, f"unknown_{status}"),
            "confidence": float(response.confidence),
            "message": response.message,
            "expected_target_width_m": float(candidate.target_width_m),
            "evidence": list(response.evidence),
            "candidate_index": candidate_index,
        }
        state.verification_records.append(record)
        self.get_logger().info(
            f"grasp verification phase={phase} candidate={candidate_index} evidence={record['evidence']}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "grasp_verification.json"
            payload = {
                "service": self._verifier_service,
                "policy": self._verification_policy,
                "records": state.verification_records,
            }
            try:
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")

    def _verify(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        phase: str,
        task_id: str,
        target_query: str,
        candidate_index: int,
        candidate: GraspCandidate,
    ) -> None:
        if self._verification_policy == "disabled":
            return
        if not self._verifier_client.service_is_ready():
            if self._verification_policy == "optional":
                return
            raise PickFlowError("VERIFIER_UNAVAILABLE", self._verifier_service)
        self._publish_feedback(goal_handle, state, phase, "sampling grasp retention evidence")
        request = VerifyGrasp.Request()
        request.task_id = task_id
        request.text_prompt = target_query
        request.grasp = candidate
        request.expected_target_width_m = float(candidate.target_width_m)
        request.post_grasp_wait_s = float(self._config.get("verification_wait_sec", 0.1))
        future = self._verifier_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            deadline,
            min(float(self._config.get("verification_timeout_sec", 5.0)), self._remaining(deadline)),
            phase,
        )
        state.verification_confidence = float(response.confidence)
        self._record_verification(state, phase, candidate_index, candidate, response)
        if response.status == VerifyGrasp.Response.STATUS_SUCCESS and response.success:
            state.verification_status = PickObject.Result.VERIFICATION_SUCCESS
            return
        if response.status == VerifyGrasp.Response.STATUS_UNCERTAIN:
            state.verification_status = PickObject.Result.VERIFICATION_UNCERTAIN
            raise PickFlowError("GRASP_UNCERTAIN", response.message)
        state.verification_status = PickObject.Result.VERIFICATION_FAILED
        raise PickFlowError("GRASP_VERIFICATION_FAILED", response.message)

    def _recover_after_close_failure(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        prepared: PreparedCandidate,
        pregrasp_xyz: tuple[float, float, float],
    ) -> None:
        retreat = (
            prepared.plan.grasp[0],
            prepared.plan.grasp[1],
            max(prepared.plan.grasp[2], pregrasp_xyz[2]),
        )
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_pose",
            pose=self._pose(retreat, prepared.plan.quaternion),
            velocity_scaling=float(self._config.get("lift_velocity_scaling", 0.05)),
        )
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "open_gripper",
            gripper_position=self._gripper_open,
        )
        observe_pose = str(self._config.get("observe_pose", "observe_table"))
        if observe_pose:
            self._run_primitive(
                goal_handle,
                deadline,
                task_id,
                "move_to_named_pose",
                pose_name=observe_pose,
                velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
            )

    def _recover_after_retention_failure(self, goal_handle, deadline: float, task_id: str) -> None:
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "open_gripper",
            gripper_position=self._gripper_open,
        )
        observe_pose = str(self._config.get("observe_pose", "observe_table"))
        if observe_pose:
            self._run_primitive(
                goal_handle,
                deadline,
                task_id,
                "move_to_named_pose",
                pose_name=observe_pose,
                velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
            )

    def _current_ee_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._ee_frame,
                Time(),
                timeout=Duration(seconds=self._rpc_timeout),
            )
        except Exception as exc:
            raise PickFlowError(
                "TF_UNAVAILABLE",
                f"cannot read current {self._ee_frame} pose in {self._base_frame}: {exc}",
                retryable=True,
            ) from exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    def _record_pose_diagnostic(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        label: str,
        commanded_xyz: tuple[float, float, float],
        commanded_quaternion: tuple[float, float, float, float],
        contact_ee: tuple[float, float, float],
        *,
        planned_contact: tuple[float, float, float] | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        config = self._config.get("pose_diagnostics", {})
        robust_gap_config = self._config.get("target_gripper", {}).get("fixed_finger_robust_gap", {})
        if not bool(config.get("enabled", False)) and not (
            label == "grasp" and bool(robust_gap_config.get("enabled", False))
        ):
            return None
        self._sleep_with_cancel(goal_handle, deadline, float(config.get("settle_sec", 0.25)))
        try:
            actual_xyz, actual_quaternion = self._current_ee_pose()
        except PickFlowError as exc:
            self.get_logger().warning(f"pose diagnostic skipped for {label}: {exc}")
            return None

        position_delta = (
            actual_xyz[0] - commanded_xyz[0],
            actual_xyz[1] - commanded_xyz[1],
            actual_xyz[2] - commanded_xyz[2],
        )
        actual_contact = self._contact_for_pose(self._pose(actual_xyz, actual_quaternion), contact_ee)
        target_contact = planned_contact or self._contact_for_pose(
            self._pose(commanded_xyz, commanded_quaternion), contact_ee
        )
        contact_residual = (
            target_contact[0] - actual_contact[0],
            target_contact[1] - actual_contact[1],
            target_contact[2] - actual_contact[2],
        )
        contact_xy_error = math.hypot(contact_residual[0], contact_residual[1])
        action = "continue"
        if label == "grasp":
            abort_threshold = max(0.0, float(config.get("grasp_abort_log_threshold_m", 0.03)))
            realign_threshold = max(0.0, float(config.get("grasp_realign_log_threshold_m", 0.01)))
            warn_threshold = max(0.0, float(config.get("grasp_warn_threshold_m", 0.008)))
            if abort_threshold > 0.0 and contact_xy_error > abort_threshold:
                action = "log_only_abort_threshold_exceeded"
            elif realign_threshold > 0.0 and contact_xy_error > realign_threshold:
                action = "log_only_realign_threshold_exceeded"
            elif warn_threshold > 0.0 and contact_xy_error > warn_threshold:
                action = "warn_continue"
            else:
                action = "continue_without_low_height_realign"

        record = {
            "label": label,
            "commanded_xyz": list(commanded_xyz),
            "commanded_quaternion_xyzw": list(commanded_quaternion),
            "actual_xyz": list(actual_xyz),
            "actual_quaternion_xyzw": list(actual_quaternion),
            "actual_minus_command_xyz": list(position_delta),
            "actual_minus_command_norm_m": math.sqrt(sum(value * value for value in position_delta)),
            "rotation_error_deg": quaternion_error_deg(commanded_quaternion, actual_quaternion),
            "closing_axis_error_deg": axis_error_deg(
                commanded_quaternion,
                actual_quaternion,
                self._config.get("target_gripper", {}).get("closing_axis_ee", [1.0, 0.0, 0.0]),
            ),
            "planned_contact_base": list(target_contact),
            "actual_contact_base": list(actual_contact),
            "contact_residual_xyz": list(contact_residual),
            "contact_xy_error_m": contact_xy_error,
            "action": action,
        }
        state.pose_diagnostics.append(record)
        self.get_logger().info(
            f"pose diagnostic label={label} position_delta={position_delta} "
            f"rotation_error_deg={record['rotation_error_deg']:.2f} "
            f"closing_axis_error_deg={record['closing_axis_error_deg']:.2f} "
            f"contact_residual={contact_residual} contact_xy_error_m={contact_xy_error:.4f} action={action}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "pick_pose_diagnostics.json"
            try:
                output_path.write_text(json.dumps(state.pose_diagnostics, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")
        return contact_residual, actual_quaternion

    def _move_branch_locked_pose(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        xyz: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
        velocity_scaling: float,
        seed: JointState | None,
        *,
        validate_orientation: bool = True,
    ) -> IKPayload:
        pose = self._pose(xyz, quaternion)
        payload = self._solve_grasp_ik_fk(
            pose,
            goal_handle,
            deadline,
            seed,
            validate_orientation=validate_orientation,
        )
        self._validate_joint5_branch_continuity(seed, payload.joint_state)
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_configuration",
            joint_state=payload.joint_state,
            velocity_scaling=velocity_scaling,
        )
        return payload

    def _realign_contact(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        phase: str,
        command_xyz: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
        contact_ee: tuple[float, float, float],
        velocity_scaling: float,
        joint_seed: JointState | None,
    ) -> tuple[tuple[float, float, float], JointState | None]:
        config = self._config.get("contact_realign", {})
        if not bool(config.get("enabled", True)):
            return command_xyz, joint_seed
        target_contact = self._contact_for_pose(self._pose(command_xyz, quaternion), contact_ee)
        current_command = command_xyz
        current_seed = joint_seed
        tolerance = max(0.0, float(config.get("tolerance_m", 0.008)))
        maximum_correction = max(0.0, float(config.get("max_correction_m", 0.03)))
        for iteration in range(max(0, int(config.get("max_iterations", 4)))):
            self._sleep_with_cancel(goal_handle, deadline, float(config.get("settle_sec", 0.25)))
            actual_xyz, actual_quaternion = self._current_ee_pose()
            actual_contact = self._contact_for_pose(self._pose(actual_xyz, actual_quaternion), contact_ee)
            correction = (
                target_contact[0] - actual_contact[0],
                target_contact[1] - actual_contact[1],
                target_contact[2] - actual_contact[2],
            )
            error_norm = math.sqrt(sum(value * value for value in correction))
            self.get_logger().info(
                f"contact realign phase={phase} iteration={iteration} error="
                f"({correction[0]:.4f},{correction[1]:.4f},{correction[2]:.4f}) norm={error_norm:.4f}"
            )
            if error_norm <= tolerance:
                return current_command, current_seed
            next_command = (
                current_command[0] + correction[0],
                current_command[1] + correction[1],
                current_command[2] + correction[2],
            )
            cumulative = math.sqrt(sum((next_command[index] - command_xyz[index]) ** 2 for index in range(3)))
            if cumulative > maximum_correction:
                raise PickFlowError(
                    "CONTACT_REALIGN_FAILED",
                    f"{phase} contact correction {cumulative:.4f}m exceeds {maximum_correction:.4f}m",
                    retryable=True,
                )
            allowed, reason = xyz_within_workspace(next_command, self._workspace)
            if not allowed:
                raise PickFlowError("WORKSPACE_REJECTED", f"{phase} realign: {reason}", retryable=True)
            payload = self._move_branch_locked_pose(
                goal_handle,
                deadline,
                task_id,
                next_command,
                quaternion,
                velocity_scaling,
                current_seed,
            )
            current_seed = payload.joint_state
            current_command = next_command
        return current_command, current_seed

    def _pregrasp_pose(
        self,
        prepared: PreparedCandidate,
        scene_base: BaseSceneGeometry,
    ) -> tuple[float, float, float]:
        config = self._config.get("contact_realign", {})
        if not bool(config.get("pregrasp_enabled", False)) or prepared.mesh_min_z is None:
            return prepared.plan.approach
        clearance = max(0.0, float(config.get("pregrasp_clearance_m", 0.02)))
        target_reference_z = prepared.mesh_min_z + clearance
        if scene_base.object_top_base is not None:
            target_reference_z = scene_base.object_top_base[2] + clearance
        dz = target_reference_z - prepared.mesh_min_z
        return (
            prepared.plan.grasp[0],
            prepared.plan.grasp[1],
            prepared.plan.grasp[2] + dz,
        )

    def _execute_candidate(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        task_id: str,
        target_query: str,
        prepared: PreparedCandidate,
        scene_base: BaseSceneGeometry,
    ) -> None:
        prepared = self._prepare_candidate(
            prepared.ranked,
            scene_base,
            goal_handle,
            deadline,
            initial_seed=prepared.final_joint_state,
        )
        plan = prepared.plan
        candidate = prepared.ranked.candidate
        active_joint_state = prepared.final_joint_state
        self._publish_feedback(goal_handle, state, "open", "opening gripper")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "open_gripper",
            gripper_position=self._gripper_open,
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("open_settle_sec", 0.3)))

        self._publish_feedback(goal_handle, state, "approach", f"approaching candidate {prepared.ranked.index}")
        approach_payload = self._move_branch_locked_pose(
            goal_handle,
            deadline,
            task_id,
            plan.approach,
            plan.quaternion,
            float(self._config.get("approach_velocity_scaling", 0.05)),
            active_joint_state,
        )
        active_joint_state = approach_payload.joint_state

        aligned_approach, active_joint_state = self._realign_contact(
            goal_handle,
            deadline,
            task_id,
            "approach",
            plan.approach,
            plan.quaternion,
            plan.target_contact_ee,
            float(self._config.get("approach_velocity_scaling", 0.05)),
            active_joint_state,
        )
        self._record_pose_diagnostic(
            goal_handle,
            deadline,
            state,
            "approach",
            aligned_approach,
            plan.quaternion,
            plan.target_contact_ee,
        )

        pregrasp = self._pregrasp_pose(prepared, scene_base)
        self._publish_feedback(goal_handle, state, "pregrasp", "moving to safe target-relative pregrasp")
        pregrasp_payload = self._move_branch_locked_pose(
            goal_handle,
            deadline,
            task_id,
            pregrasp,
            plan.quaternion,
            float(self._config.get("descend_velocity_scaling", 0.03)),
            active_joint_state,
        )
        active_joint_state = pregrasp_payload.joint_state
        aligned_pregrasp, active_joint_state = self._realign_contact(
            goal_handle,
            deadline,
            task_id,
            "pregrasp",
            pregrasp,
            plan.quaternion,
            plan.target_contact_ee,
            float(self._config.get("descend_velocity_scaling", 0.03)),
            active_joint_state,
        )
        self._record_pose_diagnostic(
            goal_handle,
            deadline,
            state,
            "pregrasp",
            aligned_pregrasp,
            plan.quaternion,
            plan.target_contact_ee,
        )
        realign_delta_x = aligned_pregrasp[0] - pregrasp[0]
        realign_delta_y = aligned_pregrasp[1] - pregrasp[1]
        adjusted_plan = replace(
            plan,
            approach=aligned_pregrasp,
            grasp=(plan.grasp[0] + realign_delta_x, plan.grasp[1] + realign_delta_y, plan.grasp[2]),
            lift=(plan.lift[0] + realign_delta_x, plan.lift[1] + realign_delta_y, plan.lift[2]),
        )
        prepared = self._prepare_candidate(
            replace(prepared.ranked, plan=adjusted_plan),
            scene_base,
            goal_handle,
            deadline,
            apply_compensation=True,
            initial_seed=active_joint_state,
        )
        plan = prepared.plan
        active_joint_state = prepared.final_joint_state

        self._publish_feedback(goal_handle, state, "descend", "descending to compensated grasp configuration")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_configuration",
            joint_state=prepared.final_joint_state,
            velocity_scaling=float(self._config.get("descend_velocity_scaling", 0.03)),
            duration_sec=float(self._config.get("descend_duration_sec", 2.0)),
        )
        self.get_logger().info(
            f"grasp prediction candidate={prepared.ranked.index} commanded_xyz={plan.grasp} "
            f"predicted_xyz={prepared.actual_ee_xyz} "
            f"predicted_closing_axis_error_deg={prepared.closing_axis_error_deg:.2f} "
            f"contact_xy_residual_m={prepared.contact_residual_xy_m:.4f} "
            f"contact_z_error_m={prepared.contact_z_error_m:.4f}"
        )
        grasp_measurement = self._record_pose_diagnostic(
            goal_handle,
            deadline,
            state,
            "grasp",
            prepared.actual_ee_xyz,
            prepared.actual_ee_quaternion,
            plan.target_contact_ee,
            planned_contact=plan.target_contact_base,
        )
        target_gripper = self._config.get("target_gripper", {})
        robust_gap_config = target_gripper.get("fixed_finger_robust_gap", {})
        if bool(robust_gap_config.get("enabled", False)):
            robust_gap = None
            if prepared.fixed_finger_envelope is not None and grasp_measurement is not None:
                contact_residual, actual_quaternion = grasp_measurement
                robust_gap = fixed_finger_robust_gap(
                    prepared.fixed_finger_envelope.fixed_gap_m,
                    prepared.fixed_finger_envelope.target_gap_m,
                    contact_residual,
                    actual_quaternion,
                    target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                    max_target_gap_deficit_m=float(robust_gap_config.get("max_target_gap_deficit_m", 0.003)),
                )
            if robust_gap is not None and state.pose_diagnostics:
                state.pose_diagnostics[-1].update(
                    {
                        "fixed_finger_contact_error_m": robust_gap.contact_error_along_closing_axis_m,
                        "fixed_finger_effective_gap_m": robust_gap.effective_gap_m,
                        "fixed_finger_required_gap_m": robust_gap.required_gap_m,
                        "fixed_finger_robust_gap_passed": robust_gap.passed,
                        "action": (
                            "continue_fixed_finger_robust_gap_passed"
                            if robust_gap.passed
                            else "retreat_fixed_finger_robust_gap_rejected"
                        ),
                    }
                )
                if state.debug_output_dir:
                    output_path = Path(state.debug_output_dir) / "pick_pose_diagnostics.json"
                    try:
                        output_path.write_text(json.dumps(state.pose_diagnostics, indent=2) + "\n", encoding="utf-8")
                    except OSError as exc:
                        self.get_logger().warning(f"failed to write {output_path}: {exc}")
            if robust_gap is None or not robust_gap.passed:
                self._run_primitive(
                    goal_handle,
                    deadline,
                    task_id,
                    "move_to_pose",
                    pose=self._pose(aligned_pregrasp, plan.quaternion),
                    velocity_scaling=float(self._config.get("descend_velocity_scaling", 0.03)),
                )
                detail = (
                    "measurement unavailable"
                    if robust_gap is None
                    else (
                        f"effective_gap={robust_gap.effective_gap_m:.4f}m "
                        f"required_gap={robust_gap.required_gap_m:.4f}m "
                        f"closing_axis_error={robust_gap.contact_error_along_closing_axis_m:.4f}m"
                    )
                )
                raise PickFlowError(
                    "FIXED_FINGER_ROBUST_GAP_REJECTED",
                    f"candidate {prepared.ranked.index}: {detail}",
                    retryable=True,
                )

        self._publish_feedback(goal_handle, state, "close", "closing gripper on target")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "close_gripper",
            gripper_position=self._gripper_closed,
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("hold_sec", 0.8)))

        try:
            self._verify(
                goal_handle,
                deadline,
                state,
                "verify_close",
                task_id,
                target_query,
                prepared.ranked.index,
                candidate,
            )
            self._record_pose_diagnostic(
                goal_handle,
                deadline,
                state,
                "close",
                prepared.actual_ee_xyz,
                prepared.actual_ee_quaternion,
                plan.target_contact_ee,
                planned_contact=plan.target_contact_base,
            )
        except PickFlowError:
            if bool(self._config.get("recover_after_close_failure", True)):
                self._recover_after_close_failure(goal_handle, deadline, task_id, prepared, aligned_pregrasp)
            raise

        probe_height = max(0.0, float(self._config.get("probe_lift_height_m", 0.03)))
        if probe_height > 0.0 and plan.lift[2] > plan.grasp[2] + 1e-6:
            probe_xyz = (plan.lift[0], plan.lift[1], min(plan.lift[2], plan.grasp[2] + probe_height))
            if probe_xyz[2] < plan.lift[2] - 1e-6:
                self._publish_feedback(goal_handle, state, "probe_lift", "performing slow retention-check lift")
                probe_payload = self._move_branch_locked_pose(
                    goal_handle,
                    deadline,
                    task_id,
                    probe_xyz,
                    plan.quaternion,
                    float(self._config.get("probe_lift_velocity_scaling", 0.02)),
                    active_joint_state,
                )
                active_joint_state = probe_payload.joint_state
                try:
                    self._verify(
                        goal_handle,
                        deadline,
                        state,
                        "verify_probe",
                        task_id,
                        target_query,
                        prepared.ranked.index,
                        candidate,
                    )
                except PickFlowError:
                    if bool(self._config.get("recover_after_retention_failure", True)):
                        self._recover_after_retention_failure(goal_handle, deadline, task_id)
                    raise
                self._record_pose_diagnostic(
                    goal_handle,
                    deadline,
                    state,
                    "probe_lift",
                    probe_xyz,
                    plan.quaternion,
                    plan.target_contact_ee,
                )

        self._publish_feedback(goal_handle, state, "lift", "lifting verified target")
        self._move_branch_locked_pose(
            goal_handle,
            deadline,
            task_id,
            plan.lift,
            plan.quaternion,
            float(self._config.get("lift_velocity_scaling", 0.05)),
            active_joint_state,
            validate_orientation=False,
        )
        try:
            self._verify(
                goal_handle,
                deadline,
                state,
                "verify_lift",
                task_id,
                target_query,
                prepared.ranked.index,
                candidate,
            )
        except PickFlowError:
            if bool(self._config.get("recover_after_retention_failure", True)):
                self._recover_after_retention_failure(goal_handle, deadline, task_id)
            raise
        self._record_pose_diagnostic(
            goal_handle,
            deadline,
            state,
            "lift",
            plan.lift,
            plan.quaternion,
            plan.target_contact_ee,
        )

    @staticmethod
    def _result_from_state(state: FlowState) -> PickObject.Result:
        result = PickObject.Result()
        result.attempts = int(state.attempt)
        result.verification_status = int(state.verification_status)
        result.verification_confidence = float(state.verification_confidence)
        result.debug_output_dir = state.debug_output_dir
        result.completed_phases = list(state.completed_phases)
        return result

    def _execute_pick(self, goal_handle):
        goal = goal_handle.request
        state = FlowState(completed_phases=[])
        timeout_sec = float(goal.timeout_sec or self._config.get("timeout_sec", 180.0))
        deadline = time.monotonic() + timeout_sec
        target_query = str(goal.target_query).strip()
        task_id = str(goal.task_id).strip() or f"pick-{int(time.time() * 1000)}"
        try:
            if len(target_query) > 200:
                raise PickFlowError("INVALID_TARGET", "target_query is too long")
            self._preflight(goal_handle, deadline, state)
            self._move_to_observe(goal_handle, deadline, state, task_id)
            stage_started = time.monotonic()
            grasp_header, candidates, scene = self._request_grasps(
                goal_handle,
                deadline,
                state,
                target_query,
            )
            self.get_logger().info(
                f"PIPELINE_TIMING stage=graspgen_request duration_s={time.monotonic() - stage_started:.3f}"
            )
            default_frame = str(grasp_header.frame_id)
            capture_transform = self._lookup_base_transform(default_frame, grasp_header.stamp)
            base_to_camera = self._transform_to_matrix(capture_transform)
            self._record_frame_diagnostic(
                state,
                "grasp_capture_stamp",
                default_frame,
                grasp_header.stamp,
                camera_transform=capture_transform,
            )
            self._record_frame_diagnostic(state, "post_plan_latest", default_frame)
            scene_base = self._scene_geometry_base(base_to_camera, scene)
            self._publish_feedback(goal_handle, state, "selecting", f"evaluating {len(candidates)} candidates")
            stage_started = time.monotonic()
            ranked = self._rank_candidates(
                default_frame,
                grasp_header.stamp,
                base_to_camera,
                candidates,
                scene,
                scene_base,
            )
            self.get_logger().info(
                f"PIPELINE_TIMING stage=candidate_geometry_ranking "
                f"duration_s={time.monotonic() - stage_started:.3f} candidates={len(ranked)}"
            )
            candidate_seed = self._snapshot_joint_state()
            if candidate_seed is None:
                raise PickFlowError(
                    "JOINT_STATE_UNAVAILABLE",
                    f"no current joint state received from {self._joint_state_topic}",
                )
            max_attempts = int(self._config.get("max_execution_attempts", 1))
            prepared_candidates, last_error = self._prepare_ranked_candidates(
                ranked,
                scene_base,
                candidate_seed,
                goal_handle,
                deadline,
            )

            if not prepared_candidates:
                if last_error is None:
                    raise PickFlowError("NO_EXECUTABLE_CANDIDATE", "no candidate could be prepared")
                raise PickFlowError(
                    "ALL_CANDIDATES_FAILED",
                    f"all {len(ranked)} ranked candidates failed preparation; last={last_error.code}: {last_error}",
                )

            prepared_scoring = self._config.get("prepared_candidate_scoring", {})
            if bool(prepared_scoring.get("enabled", False)):
                prepared_candidates.sort(
                    key=lambda item: (
                        -item.selection_score,
                        item.contact_z_error_m,
                        item.contact_residual_xy_m,
                        -item.ranked.score,
                        item.ranked.index,
                    )
                )
            else:
                prepared_candidates.sort(
                    key=lambda item: (
                        item.contact_z_error_m,
                        item.contact_residual_xy_m,
                    )
                )
            self._record_prepared_ranking(state, prepared_candidates)
            summary_parts = []
            for item in prepared_candidates[:10]:
                envelope = item.fixed_finger_envelope
                fixed_text = "n/a" if envelope is None else f"{envelope.fixed_gap_m:.4f}/{envelope.score:.3f}"
                base_side_text = (
                    "n/a"
                    if item.ranked.fixed_finger_base_side is None
                    else f"{item.ranked.fixed_finger_base_side.alignment_cos:.3f}"
                )
                fk_base_side_text = (
                    "n/a"
                    if item.fk_fixed_finger_base_side is None
                    else f"{item.fk_fixed_finger_base_side.alignment_cos:.3f}"
                )
                summary_parts.append(
                    f"{item.ranked.index}:score={item.selection_score:.3f}/fixed={fixed_text}/"
                    f"base_side={base_side_text}/fk_base_side={fk_base_side_text}/"
                    f"z={item.contact_z_error_m:.4f}/xy={item.contact_residual_xy_m:.4f}/"
                    f"approach_axis={item.approach_axis_error_deg}/closing_axis={item.closing_axis_error_deg:.1f}"
                )
            summary = ", ".join(summary_parts)
            self.get_logger().info(f"prepared candidate rank: {summary}")

            attempt_candidates = (
                prepared_candidates if max_attempts <= 0 else prepared_candidates[: max(1, max_attempts)]
            )
            for attempt, prepared in enumerate(attempt_candidates, start=1):
                state.attempt = attempt
                candidate = prepared.ranked
                try:
                    self._execute_candidate(
                        goal_handle,
                        deadline,
                        state,
                        task_id,
                        target_query,
                        prepared,
                        scene_base,
                    )
                    self._publish_feedback(goal_handle, state, "completed", "object grasped and verified")
                    result = self._result_from_state(state)
                    result.success = True
                    result.error_code = ""
                    result.message = (
                        f"picked {target_query!r}; candidate={candidate.index}; attempts={attempt}; "
                        f"verification_confidence={state.verification_confidence:.3f}"
                    )
                    goal_handle.succeed()
                    return result
                except PickCancelled:
                    raise
                except PickFlowError as exc:
                    last_error = exc
                    self.get_logger().warning(
                        f"pick candidate failed: attempt={attempt} candidate={candidate.index} "
                        f"code={exc.code} retryable={exc.retryable} message={exc}"
                    )
                    if not exc.retryable:
                        raise
            if last_error is None:
                raise PickFlowError("NO_EXECUTION_RESULT", "prepared candidates produced no execution result")
            raise PickFlowError(
                "ALL_CANDIDATES_FAILED",
                f"all {state.attempt} attempted candidates failed; last={last_error.code}: {last_error}",
            )
        except PickCancelled as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.canceled()
            return result
        except PickFlowError as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.abort()
            return result
        except Exception as exc:
            self.get_logger().exception("unexpected pick execution failure")
            result = self._result_from_state(state)
            result.success = False
            result.error_code = "INTERNAL_ERROR"
            result.message = str(exc)
            goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False


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
