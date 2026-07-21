"""Skill and primitive execution node for the embodied minimal closure."""

import math
import time
import uuid
from collections.abc import Callable

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ibrobot_msgs.action import ExecuteTaskPlan, PrimitiveCommand, SkillCommand
from ibrobot_msgs.msg import TaskStep
from ibrobot_msgs.srv import ValidatePrimitive, ValidateSkill
from skill_library.resolver import PrimitiveSpec, load_json_mapping, resolve_skill_primitives

EE_POSITION_TOLERANCE_M = 0.02


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

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("skill_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("validate_primitive_service", "/embodied/validate_primitive")
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("skill_templates_json", "{}")
        self.declare_parameter("relative_motion_reference_frame", "base")
        self.declare_parameter("relative_motion_direction_mapping_json", "{}")
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("gripper_settle_sec", 1.5)
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("arm_trajectory_action_name", "/arm_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("task_executor_action_name", "/task_executor/execute_task_plan")
        self.declare_parameter("ee_pose_topic", "/robot_status/ee_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("debug_tracing", False)

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
        self._skill_templates = load_json_mapping(
            self.get_parameter("skill_templates_json").get_parameter_value().string_value
        )
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
        self._ee_pose_topic = self.get_parameter("ee_pose_topic").get_parameter_value().string_value
        self._joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._latest_ee_pose = None
        self._latest_joint_state = None

        callback_group = ReentrantCallbackGroup()
        self._pose_publisher = self.create_publisher(Pose, "/cmd_pose", 10)
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
        self._skill_server = ActionServer(
            self,
            SkillCommand,
            self._skill_action_name,
            execute_callback=self._execute_skill,
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
        self._latest_ee_pose = msg

    def _handle_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    @staticmethod
    def _handle_cancel(_cancel_request):
        return CancelResponse.ACCEPT

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
        result.error_code = "CANCEL_CLEANUP_TIMEOUT" if message.startswith("cancel cleanup timed out") else error_code
        result.message = message
        result.pose_name = pose_name
        if result.error_code == "CANCEL_CLEANUP_TIMEOUT":
            goal_handle.abort()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _cancel_goal(self, goal_handle, result_future=None) -> bool:
        if goal_handle is None:
            return True
        if result_future is None:
            try:
                result_future = goal_handle.get_result_async()
            except Exception:
                return False
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            return False
        if not self._wait_for_future(cancel_future, timeout_sec=self._rpc_timeout):
            return False
        if not self._wait_for_future(result_future, timeout_sec=self._rpc_timeout):
            return False
        try:
            wrapped_result = result_future.result()
        except Exception:
            return False
        child_result = getattr(wrapped_result, "result", None)
        return getattr(child_result, "error_code", "") != "CANCEL_CLEANUP_TIMEOUT"

    def _cancel_goal_when_ready(self, goal_future) -> None:
        def cancel_accepted_goal(completed_future) -> None:
            try:
                goal_handle = completed_future.result()
            except Exception:
                return
            if goal_handle is not None and goal_handle.accepted:
                self._cancel_goal(goal_handle)

        goal_future.add_done_callback(cancel_accepted_goal)

    def _cancel_goal_future(self, goal_future) -> bool:
        if not self._wait_for_future(goal_future, timeout_sec=self._rpc_timeout):
            self._cancel_goal_when_ready(goal_future)
            return False
        try:
            goal_handle = goal_future.result()
        except Exception:
            return False
        if goal_handle is None or not goal_handle.accepted:
            return True
        return self._cancel_goal(goal_handle)

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
            if self._latest_ee_pose is not None:
                dx = float(self._latest_ee_pose.pose.position.x - target_pose.position.x)
                dy = float(self._latest_ee_pose.pose.position.y - target_pose.position.y)
                dz = float(self._latest_ee_pose.pose.position.z - target_pose.position.z)
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
        else:
            request.target_x = 0.0
            request.target_y = 0.0
            request.target_z = 0.0
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

    def _pose_from_relative_offset(self, dx: float, dy: float, dz: float) -> Pose | None:
        if self._latest_ee_pose is None:
            return None
        pose = Pose()
        pose.position.x = float(self._latest_ee_pose.pose.position.x + dx)
        pose.position.y = float(self._latest_ee_pose.pose.position.y + dy)
        pose.position.z = float(self._latest_ee_pose.pose.position.z + dz)
        # Keep current EE orientation unchanged during position-only moves
        pose.orientation.x = float(self._latest_ee_pose.pose.orientation.x)
        pose.orientation.y = float(self._latest_ee_pose.pose.orientation.y)
        pose.orientation.z = float(self._latest_ee_pose.pose.orientation.z)
        pose.orientation.w = float(self._latest_ee_pose.pose.orientation.w)
        return pose

    def _execute_primitive(self, goal_handle):
        goal = goal_handle.request
        result = PrimitiveCommand.Result()

        target_pose = None
        if goal.primitive_name == "move_relative_ee":
            target_pose = self._pose_from_relative_offset(
                goal.relative_dx,
                goal.relative_dy,
                goal.relative_dz,
            )
            if target_pose is None:
                result.success = False
                result.error_code = "CURRENT_EE_POSE_UNAVAILABLE"
                result.message = "current ee pose is unavailable for relative motion"
                result.pose_name = ""
                goal_handle.abort()
                return result

        allowed, reason = self._validate_primitive(
            goal.primitive_name,
            goal.pose_name,
            goal.gripper_position,
            relative_dx=goal.relative_dx,
            relative_dy=goal.relative_dy,
            relative_dz=goal.relative_dz,
            target_pose=target_pose,
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

        if goal.primitive_name == "move_to_joint_positions":
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
        elif goal.primitive_name in {"move_to_named_pose", "move_relative_ee"}:
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
            ok, err_msg = self._exec_arm_via_task_dispatch(
                goal_handle, goal.primitive_name, pose, goal.task_id, move_timeout
            )
            if not ok:
                return self._finish_primitive_failure(
                    result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, goal.pose_name
                )
        elif goal.primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            if self._latest_ee_pose is None:
                result.success = False
                result.error_code = "CURRENT_EE_POSE_UNAVAILABLE"
                result.message = "current ee pose is unavailable for gripper rotation"
                result.pose_name = ""
                goal_handle.abort()
                return result
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            ok, err_msg = self._exec_rotate_gripper_via_task_dispatch(
                goal_handle, goal.primitive_name, goal.relative_dz, goal.task_id, move_timeout
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
        angle_deg: float,
        task_id: str,
        timeout_sec: float,
    ) -> tuple[bool, str]:
        """Rotate gripper around its local Z-axis by angle_deg from current orientation.

        Reads the current EE orientation and right-multiplies a local-Z delta
        rotation so the result is always relative to wherever the gripper is now.
        """
        if self._latest_ee_pose is None:
            return False, "no EE pose available for rotate_gripper"

        angle_rad = math.radians(abs(angle_deg))
        half = angle_rad / 2.0
        sign = -1.0 if primitive_name == "rotate_gripper_cw" else 1.0

        # Current EE orientation (x, y, z, w)
        cur = self._latest_ee_pose.pose.orientation
        qc = (cur.x, cur.y, cur.z, cur.w)

        # Delta: rotate around local EE Z-axis by angle_deg
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
        target_pose.position.x = float(self._latest_ee_pose.pose.position.x)
        target_pose.position.y = float(self._latest_ee_pose.pose.position.y)
        target_pose.position.z = float(self._latest_ee_pose.pose.position.z)
        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] primitive {primitive_name} via task_dispatch "
                f"task_id={task_id} angle={angle_deg:.1f}deg "
                f"cur=({qc[0]:.3f},{qc[1]:.3f},{qc[2]:.3f},{qc[3]:.3f}) "
                f"-> quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})"
            )

        return self._exec_arm_via_task_dispatch(goal_handle, primitive_name, target_pose, task_id, timeout_sec)

    def _exec_arm_via_task_dispatch(
        self, goal_handle, primitive_name: str, target_pose: Pose, task_id: str, timeout_sec: float
    ) -> tuple[bool, str]:
        """Send a MOVE_TO_POSE TaskStep to task_dispatch ExecuteTaskPlan action server."""
        if not self._task_executor_client.wait_for_server(timeout_sec=2.0):
            msg = f"task_executor action server not available: {self._task_executor_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        step = TaskStep()
        step.type = TaskStep.MOVE_TO_POSE
        step.label = primitive_name
        step.target_pose = target_pose
        step.velocity_scaling = 0.0  # use default

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

        send_future = self._task_executor_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future)
                return (
                    False,
                    "cancelled while sending arm goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future)
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

        result_future = gh.get_result_async()
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future)
                return False, "cancelled during arm motion" if cleaned else "cancel cleanup timed out during arm motion"
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future)
                return (
                    False,
                    "timeout waiting for arm motion" if cleaned else "cancel cleanup timed out during arm motion",
                )
            time.sleep(0.05)

        result = result_future.result().result
        if not result.success:
            return False, f"arm motion failed: {result.message}"
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
                cleaned = self._cancel_goal_future(send_future)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future)
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

        result_future = gh.get_result_async()
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future)
                return (
                    False,
                    "cancelled during arm joint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future)
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
                cleaned = self._cancel_goal_future(send_future)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future)
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

        result_future = gh.get_result_async()
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future)
                return (
                    False,
                    "cancelled during arm joint waypoint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint waypoint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future)
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
                cleaned = self._cancel_goal_future(send_future)
                return (
                    False,
                    "cancelled while sending gripper goal"
                    if cleaned
                    else "cancel cleanup timed out while sending gripper goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future)
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

        result_future = gh.get_result_async()
        deadline = time.monotonic() + exec_timeout
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future)
                return (
                    False,
                    "cancelled during gripper execution"
                    if cleaned
                    else "cancel cleanup timed out during gripper execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future)
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

    def _execute_skill(self, goal_handle):
        goal = goal_handle.request
        result = SkillCommand.Result()

        allowed, reason = self._validate_skill(
            goal.skill_name,
            goal.target_name,
            goal.place_name,
            motion_direction=goal.motion_direction,
            motion_distance=goal.motion_distance,
        )
        if not allowed:
            return self._abort_skill(result, goal_handle, [], "SKILL_REJECTED", reason)

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

        skill_deadline = None
        if goal.timeout_sec > 0.0:
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
        for primitive in primitives:
            if goal_handle.is_cancel_requested:
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
            feedback.detail = f"{primitive_name}:{pose_name or gripper_position}"
            if primitive_name in {"move_to_joint_positions", "move_through_joint_positions"}:
                feedback.detail = f"{primitive_name}:{','.join(primitive.joint_names)}"
            goal_handle.publish_feedback(feedback)

            primitive_goal = PrimitiveCommand.Goal()
            primitive_goal.task_id = goal.task_id
            primitive_goal.primitive_name = primitive_name
            primitive_goal.pose_name = pose_name
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

            send_goal_future = self._primitive_client.send_goal_async(primitive_goal)
            send_goal_timeout = (
                self._rpc_timeout if remaining_timeout is None else min(self._rpc_timeout, remaining_timeout)
            )

            if not self._wait_for_future(
                send_goal_future,
                timeout_sec=max(0.1, send_goal_timeout),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                if goal_handle.is_cancel_requested:
                    if self._cancel_goal_future(send_goal_future):
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_CANCEL_CLEANUP_TIMEOUT",
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                self._cancel_goal_future(send_goal_future)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_TIMEOUT",
                    f"timed out sending primitive {primitive_name}",
                )

            primitive_handle = send_goal_future.result()
            if primitive_handle is None or not primitive_handle.accepted:
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_REJECTED",
                    f"primitive goal rejected: {primitive_name}",
                )

            result_future = primitive_handle.get_result_async()
            remaining_timeout = None
            if skill_deadline is not None:
                remaining_timeout = skill_deadline - time.monotonic()
                if remaining_timeout <= 0.0:
                    if not self._cancel_goal(primitive_handle, result_future):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            executed_primitives,
                            "SKILL_CANCEL_CLEANUP_TIMEOUT",
                            f"cancel cleanup timed out for primitive {primitive_name}",
                        )
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
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_CANCEL_CLEANUP_TIMEOUT",
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if not self._cancel_goal(primitive_handle, result_future):
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_CANCEL_CLEANUP_TIMEOUT",
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                return self._abort_skill(
                    result, goal_handle, executed_primitives, "SKILL_TIMEOUT", f"primitive timed out: {primitive_name}"
                )

            action_result = result_future.result()
            primitive_result = action_result.result if action_result is not None else None
            if primitive_result is None or not primitive_result.success:
                error_code = primitive_result.error_code if primitive_result is not None else "MISSING_PRIMITIVE_RESULT"
                message = primitive_result.message if primitive_result is not None else "missing primitive result"
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
