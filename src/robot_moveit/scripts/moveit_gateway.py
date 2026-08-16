#!/usr/bin/env python3
"""
MoveIt 2 Gateway Node for IB-Robot.

ROS Interfaces:
    Subscriptions:
        /cmd_pose (geometry_msgs/Pose) — fire-and-forget Pose commands
        configured arm-only joint state topic (sensor_msgs/JointState)
        /motion_mode/base_navigation_enabled (std_msgs/Bool)
        configured hardware feedback heartbeat (ibrobot_msgs/JointCurrent)
    Publishers:
        /robot_status/ee_pose (geometry_msgs/PoseStamped) — 10 Hz
        /moveit_gateway/motion_status (std_msgs/String) — "idle" | "executing" | "succeeded" | "failed"
        /motion_mode/navigation_enabled (std_msgs/Bool, transient local)
    Services:
        /moveit_gateway/move_to_pose (ibrobot_msgs/MoveToPose) — synchronous move (blocks until done)
        /moveit_gateway/move_to_configuration (ibrobot_msgs/MoveToConfiguration) — execute a validated IK solution
        /motion_mode/set_navigation_enabled (std_srvs/SetBool) — switch arm/base command authorization
"""

import math
import threading
import time

import numpy as np
import rclpy
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import Constraints, OrientationConstraint
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header, String
from std_srvs.srv import SetBool

try:
    from ibrobot_msgs.msg import JointCurrent
    from ibrobot_msgs.srv import MoveToConfiguration, MoveToPose

    _HAS_MOVE_SERVICES = True
except ImportError:
    _HAS_MOVE_SERVICES = False

# TF2 and MoveIt 2 imports
import tf2_ros

from pymoveit2 import MoveIt2, MoveIt2State


class MoveItGateway(Node):
    def __init__(self):
        super().__init__("moveit_gateway")

        # 1. Callback Group
        self.callback_group = ReentrantCallbackGroup()

        # 2. Parameters (no defaults - fail-fast if not provided via launch file)
        self.declare_parameter("arm_group_name")
        self.declare_parameter("base_link")
        self.declare_parameter("ee_link")
        self.declare_parameter("joint_names")
        self.declare_parameter("shoulder_link")
        self.declare_parameter("motion_start_timeout_s", 5.0)
        self.declare_parameter("motion_execution_timeout_s", 30.0)
        self.declare_parameter("motion_cancel_timeout_s", 5.0)
        self.declare_parameter("motion_status_hold_s", 0.0)
        self.declare_parameter("motion_feedback_timeout_s", 0.3)
        self.declare_parameter("motion_feedback_tolerance_rad", 0.12)
        self.declare_parameter("motion_require_tf_sync", True)
        self.declare_parameter("motion_hardware_feedback_topic", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("motion_mode_enabled", False)
        self.declare_parameter("navigation_enabled_on_startup", False)
        self.declare_parameter("navigation_enabled_topic", "motion_mode/navigation_enabled")
        self.declare_parameter("navigation_mode_ack_topic", "motion_mode/base_navigation_enabled")
        self.declare_parameter("set_navigation_enabled_service", "motion_mode/set_navigation_enabled")
        self.declare_parameter("controller_switch_service", "controller_manager/switch_controller")
        self.declare_parameter(
            "motion_mode_manipulation_controllers",
            ["arm_trajectory_controller", "gripper_trajectory_controller"],
        )
        self.declare_parameter("motion_mode_navigation_controllers", ["base_velocity_controller"])
        self.declare_parameter("motion_mode_transition_timeout_s", 2.0)
        self.declare_parameter("motion_mode_bridge_heartbeat_timeout_s", 1.0)

        self.group_name = self.get_parameter("arm_group_name").value
        self.base_link = self.get_parameter("base_link").value
        self.ee_link = self.get_parameter("ee_link").value
        self.joint_names = self.get_parameter("joint_names").value
        self.shoulder_link = self.get_parameter("shoulder_link").value
        self._motion_start_timeout_s = max(float(self.get_parameter("motion_start_timeout_s").value), 0.0)
        self._motion_execution_timeout_s = max(float(self.get_parameter("motion_execution_timeout_s").value), 0.0)
        self._motion_cancel_timeout_s = max(float(self.get_parameter("motion_cancel_timeout_s").value), 0.0)
        self._motion_status_hold_s = max(float(self.get_parameter("motion_status_hold_s").value), 0.0)
        self._motion_feedback_timeout_s = max(float(self.get_parameter("motion_feedback_timeout_s").value), 0.0)
        self._motion_feedback_tolerance_rad = max(float(self.get_parameter("motion_feedback_tolerance_rad").value), 0.0)
        self._motion_require_tf_sync = bool(self.get_parameter("motion_require_tf_sync").value)
        self._motion_hardware_feedback_topic = str(self.get_parameter("motion_hardware_feedback_topic").value).strip()
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value).strip()
        self._motion_mode_enabled = bool(self.get_parameter("motion_mode_enabled").value)
        navigation_enabled_on_startup = bool(self.get_parameter("navigation_enabled_on_startup").value)
        self._navigation_enabled_topic = str(self.get_parameter("navigation_enabled_topic").value).strip()
        self._navigation_mode_ack_topic = str(self.get_parameter("navigation_mode_ack_topic").value).strip()
        self._set_navigation_enabled_service = str(self.get_parameter("set_navigation_enabled_service").value).strip()
        self._controller_switch_service = str(self.get_parameter("controller_switch_service").value).strip()
        self._motion_mode_manipulation_controllers = list(
            self.get_parameter("motion_mode_manipulation_controllers").value
        )
        self._motion_mode_navigation_controllers = list(self.get_parameter("motion_mode_navigation_controllers").value)
        self._motion_mode_transition_timeout_s = max(
            float(self.get_parameter("motion_mode_transition_timeout_s").value), 0.0
        )
        self._motion_mode_bridge_heartbeat_timeout_s = max(
            float(self.get_parameter("motion_mode_bridge_heartbeat_timeout_s").value), 0.0
        )
        self._initialize_motion_coordinator()
        self._navigation_enabled = navigation_enabled_on_startup if self._motion_mode_enabled else False

        self._joint_state_lock = threading.Lock()
        self._joint_state_sequence = 0
        self._hardware_feedback_sequence = 0
        self._latest_hardware_feedback_stamp_ns = 0
        self.latest_joint_state = None
        self.get_logger().info("Initializing MoveIt Gateway for SO101...")

        # 3. TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 4. Initialize MoveIt 2 with the callback group
        try:
            self.moveit2 = MoveIt2(
                node=self,
                group_name=self.group_name,
                joint_names=self.joint_names,
                base_link_name=self.base_link,
                end_effector_name=self.ee_link,
                use_move_group_action=True,
                ignore_new_calls_while_executing=True,
                callback_group=self.callback_group,
            )
            self.get_logger().info("MoveIt2 interface connected")
        except Exception as e:
            self.get_logger().error(f"MoveIt2 connect failed: {e}")
            self.moveit2 = None

        # 5. Publishers and Subscribers (using the reentrant group)
        self.joint_state_sub = self.create_subscription(
            JointState,
            self._joint_state_topic,
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )
        self.hardware_feedback_sub = None
        if self._motion_hardware_feedback_topic:
            self.hardware_feedback_sub = self.create_subscription(
                JointCurrent,
                self._motion_hardware_feedback_topic,
                self.hardware_feedback_callback,
                10,
                callback_group=self.callback_group,
            )
            self.get_logger().info(f"Post-motion hardware feedback barrier: {self._motion_hardware_feedback_topic}")

        self.ee_pose_pub = self.create_publisher(PoseStamped, "/robot_status/ee_pose", 10)

        self.cmd_pose_sub = self.create_subscription(
            Pose,
            "/cmd_pose",
            self.cmd_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        # 6. Motion status publisher (for task_dispatch and external monitors)
        self.motion_status_pub = self.create_publisher(String, "/moveit_gateway/motion_status", 10)
        self._motion_status = "idle"

        self.navigation_mode_pub = None
        self.navigation_mode_ack_sub = None
        self.set_navigation_enabled_srv = None
        self.controller_switch_client = None
        if self._motion_mode_enabled:
            mode_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            self.navigation_mode_pub = self.create_publisher(Bool, self._navigation_enabled_topic, mode_qos)
            self.navigation_mode_ack_sub = self.create_subscription(
                Bool,
                self._navigation_mode_ack_topic,
                self.navigation_mode_ack_callback,
                10,
                callback_group=self.callback_group,
            )
            self.set_navigation_enabled_srv = self.create_service(
                SetBool,
                self._set_navigation_enabled_service,
                self._set_navigation_enabled_service_cb,
                callback_group=self.callback_group,
            )
            self.controller_switch_client = self.create_client(
                SwitchController,
                self._controller_switch_service,
                callback_group=self.callback_group,
            )
            self._publish_navigation_mode(self._navigation_enabled)
            self.get_logger().info(
                f"Persistent motion mode enabled: navigation_enabled={self._navigation_enabled}, "
                f"service={self._set_navigation_enabled_service}"
            )

        # 7. MoveToPose service (synchronous move, used by task_dispatch)
        if _HAS_MOVE_SERVICES:
            self.move_to_pose_srv = self.create_service(
                MoveToPose,
                "/moveit_gateway/move_to_pose",
                self._move_to_pose_service_cb,
                callback_group=self.callback_group,
            )
            self.get_logger().info("MoveToPose service registered")
            self.move_to_configuration_srv = self.create_service(
                MoveToConfiguration,
                "/moveit_gateway/move_to_configuration",
                self._move_to_configuration_service_cb,
                callback_group=self.callback_group,
            )
            self.get_logger().info("MoveToConfiguration service registered")
        else:
            self.get_logger().warn(
                "ibrobot_msgs motion services not available — services disabled (rebuild ibrobot_msgs to enable)"
            )

        self.timer = self.create_timer(0.1, self.publish_ee_pose, callback_group=self.callback_group)
        self.motion_watchdog_timer = self.create_timer(
            0.1,
            self._motion_watchdog_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info("MoveIt Gateway fully initialized")

    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        四元数乘法: q = q1 * q2 (使用 scipy)
        四元数格式: [x, y, z, w]
        """
        r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])
        r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
        result = r1 * r2
        return tuple(result.as_quat().tolist())

    @staticmethod
    def quaternion_conjugate(q):
        """
        四元数共轭: q* = [ -x, -y, -z, w ] (使用 scipy)
        """
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        return tuple(r.inv().as_quat().tolist())

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        四元数转旋转矩阵 (使用 scipy)
        q: [x, y, z, w]
        返回: 3x3旋转矩阵 (numpy array)
        """
        return R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()

    @staticmethod
    def rotation_matrix_to_quaternion(R_mat):
        """
        旋转矩阵转四元数 (使用 scipy)
        R_mat: 3x3旋转矩阵 (numpy array or nested list)
        返回: [x, y, z, w]
        """
        return tuple(R.from_matrix(R_mat).as_quat().tolist())

    def constrain_to_z_axis_only(self, quat):
        """
        只约束末端执行器的Z轴方向，放松绕Z轴的旋转 (numpy简化版)。
        这适用于5自由度机械臂，因为5个关节无法满足完整的6DOF约束。

        原理：
        - 保持Z轴方向不变（这约束了2个自由度：pitch和yaw）
        - 放松绕Z轴的旋转（释放1个自由度：roll）
        - 使用"最小旋转"原则，保持与原姿态接近

        Args:
            quat: 原始四元数 (x, y, z, w)

        Returns:
            tuple: 约束后的四元数 (x', y', z', w')
        """
        # 1. 转换为旋转矩阵
        R = self.quaternion_to_rotation_matrix(quat)

        # 2. 提取并归一化Z轴（第3列）
        z_axis = R[:, 2]
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        # 3. 构造新的X轴（最小旋转原则）
        orig_x_axis = R[:, 0]
        # 将原X轴投影到垂直于Z轴的平面: proj = x - (x·z) * z
        x_axis = orig_x_axis - np.dot(orig_x_axis, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)

        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            # X轴退化，使用替代策略
            if abs(z_axis[2]) < 0.9:
                # Z轴非垂直，使用水平方向
                z_xy_norm = np.linalg.norm(z_axis[:2])
                x_axis = np.array([-z_axis[1], z_axis[0], 0.0]) / z_xy_norm
            else:
                # Z轴垂直，使用world X方向
                x_axis = np.array([1.0, 0.0, 0.0])

        # 4. Y轴 = Z × X (叉积)
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm

        # 5. 重建旋转矩阵（列存储：X、Y、Z轴）
        R_constrained = np.column_stack([x_axis, y_axis, z_axis])

        # 6. 转换回四元数
        q_constrained = self.rotation_matrix_to_quaternion(R_constrained)

        return q_constrained

    def project_orientation_to_shoulder_xz_plane(self, quat):
        """
        将方向四元数投影到shoulder坐标系的XZ平面 (numpy简化版)。

        流程：
        1. 获取base到shoulder的变换
        2. 将方向从base坐标系转换到shoulder坐标系
        3. 在shoulder坐标系中，将旋转矩阵的Y轴分量约束到XZ平面
        4. 转换回四元数并转回base坐标系

        Args:
            quat: base坐标系中的四元数 (x, y, z, w)

        Returns:
            tuple: 投影后的四元数 (x', y', z', w')
        """
        try:
            # 获取base到shoulder的静态变换
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                self.shoulder_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )

            # 提取变换的四元数
            base_to_shoulder_q = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )

            # shoulder到base的变换（共轭）
            shoulder_to_base_q = self.quaternion_conjugate(base_to_shoulder_q)

        except Exception as e:
            self.get_logger().warning(f"Failed to get base->shoulder transform: {e}, using identity")
            # 如果获取失败，假设base和shoulder对齐
            base_to_shoulder_q = (0.0, 0.0, 0.0, 1.0)
            shoulder_to_base_q = (0.0, 0.0, 0.0, 1.0)

        # 1. 将方向从base转换到shoulder坐标系: q_shoulder = q_base_to_shoulder * q_base
        q_shoulder = self.quaternion_multiply(base_to_shoulder_q, quat)

        self.get_logger().debug(
            f"  Base to shoulder quaternion: ({base_to_shoulder_q[0]:.3f}, {base_to_shoulder_q[1]:.3f}, {base_to_shoulder_q[2]:.3f}, {base_to_shoulder_q[3]:.3f})"
        )
        self.get_logger().debug(
            f"  Orientation in shoulder frame: ({q_shoulder[0]:.3f}, {q_shoulder[1]:.3f}, {q_shoulder[2]:.3f}, {q_shoulder[3]:.3f})"
        )

        # 2. 转换为旋转矩阵
        R_shoulder = self.quaternion_to_rotation_matrix(q_shoulder)

        # 3. 在shoulder坐标系中，约束到XZ平面（Y分量为0）
        # 提取三个轴向量
        x_axis = R_shoulder[:, 0]
        y_axis = R_shoulder[:, 1]
        z_axis = R_shoulder[:, 2]

        # 约束X轴和Z轴到XZ平面（将Y分量设为0）
        x_axis_constrained = np.array([x_axis[0], 0.0, x_axis[2]])
        z_axis_constrained = np.array([z_axis[0], 0.0, z_axis[2]])

        # 归一化X轴
        x_norm = np.linalg.norm(x_axis_constrained)
        if x_norm > 1e-6:
            x_axis = x_axis_constrained / x_norm
        else:
            x_axis = np.array([1.0, 0.0, 0.0])

        # 归一化Z轴
        z_norm = np.linalg.norm(z_axis_constrained)
        if z_norm > 1e-6:
            z_axis = z_axis_constrained / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        # 重建Y轴 = Z × X (叉积)
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm

        # 4. 重建旋转矩阵（列存储）
        R_constrained = np.column_stack([x_axis, y_axis, z_axis])

        # 5. 转换回四元数
        q_shoulder_constrained = self.rotation_matrix_to_quaternion(R_constrained)

        # 6. 转换回base坐标系: q_base = q_shoulder_to_base * q_shoulder_constrained
        q_base_constrained = self.quaternion_multiply(shoulder_to_base_q, q_shoulder_constrained)

        return q_base_constrained

    def create_orientation_constraint(self, target_quat, link_name, frame_id, tolerances=(0.3, 0.3, 0.05)):
        """
        创建带有容差的姿态约束，用于5DOF机械臂的IK求解。

        Args:
            target_quat: 目标四元数
            link_name: 约束的link（如"gripper"）
            frame_id: 参考坐标系（如"base"）
            tolerances: (x_tol, y_tol, z_tol) 容差元组（弧度）

        Returns:
            OrientationConstraint: 姿态约束对象
        """
        constraint = OrientationConstraint()
        constraint.header = Header()
        constraint.header.frame_id = frame_id
        constraint.link_name = link_name

        # 设置目标姿态
        constraint.orientation.x = target_quat[0]
        constraint.orientation.y = target_quat[1]
        constraint.orientation.z = target_quat[2]
        constraint.orientation.w = target_quat[3]

        # 设置容差（弧度）
        # X/Y轴容差较大（放松绕Z轴旋转），Z轴容差较小（保持方向）
        constraint.absolute_x_axis_tolerance = tolerances[0]
        constraint.absolute_y_axis_tolerance = tolerances[1]
        constraint.absolute_z_axis_tolerance = tolerances[2]

        # 约束权重（1.0表示严格约束）
        constraint.weight = 1.0

        return constraint

    def joint_state_callback(self, msg):
        with self._joint_state_lock:
            self.latest_joint_state = msg
            self._joint_state_sequence += 1
        # 调试：打印关节状态
        if msg is not None and hasattr(msg, "name") and hasattr(msg, "position"):
            self.get_logger().debug(f"Joint state updated: {list(msg.name)} = {[f'{p:.3f}' for p in msg.position]}")

    @staticmethod
    def _message_stamp_ns(message) -> int:
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return 0
        return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))

    def hardware_feedback_callback(self, msg):
        """Record feedback emitted only after a successful hardware read cycle."""
        with self._joint_state_lock:
            self._hardware_feedback_sequence += 1
            self._latest_hardware_feedback_stamp_ns = self._message_stamp_ns(msg)

    def _tf_has_caught_up(self, joint_state_stamp_ns: int) -> tuple[bool, int]:
        if not self._motion_require_tf_sync:
            return True, 0
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0),
            )
        except Exception:
            return False, 0
        transform_stamp_ns = self._message_stamp_ns(transform)
        if joint_state_stamp_ns <= 0:
            return True, transform_stamp_ns
        return transform_stamp_ns >= joint_state_stamp_ns, transform_stamp_ns

    def _wait_for_post_motion_feedback(self, target_positions: dict[str, float] | None = None) -> bool:
        """Wait for post-terminal hardware, joint, and corresponding TF feedback."""
        timeout_s = self._motion_feedback_timeout_s
        if timeout_s <= 0.0:
            return True

        started = time.monotonic()
        deadline = started + timeout_s
        with self._joint_state_lock:
            initial_joint_sequence = self._joint_state_sequence
            initial_hardware_sequence = self._hardware_feedback_sequence

        last_missing: list[str] = []
        last_max_error = math.inf
        last_joint_stamp_ns = 0
        last_hardware_stamp_ns = 0
        last_tf_stamp_ns = 0
        accepted_joint_sequence = 0
        hardware_synchronized = not self._motion_hardware_feedback_topic
        tf_synchronized = not self._motion_require_tf_sync
        while time.monotonic() < deadline:
            with self._joint_state_lock:
                joint_sequence = self._joint_state_sequence
                hardware_sequence = self._hardware_feedback_sequence
                message = self.latest_joint_state
                last_hardware_stamp_ns = self._latest_hardware_feedback_stamp_ns

            hardware_sequence_advanced = hardware_sequence > initial_hardware_sequence
            if accepted_joint_sequence == 0 and joint_sequence > initial_joint_sequence and message is not None:
                last_joint_stamp_ns = self._message_stamp_ns(message)
                positions_converged = target_positions is None
                if target_positions is not None:
                    positions = {
                        str(name): float(position)
                        for name, position in zip(message.name, message.position, strict=False)
                        if math.isfinite(float(position))
                    }
                    last_missing = [name for name in target_positions if name not in positions]
                    if not last_missing:
                        last_max_error = max(abs(positions[name] - target) for name, target in target_positions.items())
                        positions_converged = last_max_error <= self._motion_feedback_tolerance_rad
                if positions_converged:
                    accepted_joint_sequence = joint_sequence

            if accepted_joint_sequence > 0:
                hardware_synchronized = not self._motion_hardware_feedback_topic or (
                    hardware_sequence_advanced
                    and (last_joint_stamp_ns <= 0 or last_hardware_stamp_ns >= last_joint_stamp_ns)
                )
                tf_synchronized, last_tf_stamp_ns = self._tf_has_caught_up(last_joint_stamp_ns)
                if hardware_synchronized and tf_synchronized:
                    error_text = "n/a" if target_positions is None else f"{last_max_error:.4f} rad"
                    self.get_logger().info(
                        f"Post-motion feedback synchronized in {time.monotonic() - started:.3f}s "
                        f"(max_joint_error={error_text}, hardware={hardware_synchronized}, "
                        f"joint_stamp_ns={last_joint_stamp_ns}, tf_stamp_ns={last_tf_stamp_ns})"
                    )
                    return True
            time.sleep(0.01)

        missing_text = ",".join(last_missing) if last_missing else "none"
        error_text = "unavailable" if not math.isfinite(last_max_error) else f"{last_max_error:.4f}"
        self.get_logger().warning(
            f"Post-motion feedback barrier did not converge within {timeout_s:.3f}s "
            f"(missing={missing_text}, max_joint_error={error_text} rad, "
            f"hardware={hardware_synchronized}, tf={tf_synchronized}, "
            f"joint_stamp_ns={last_joint_stamp_ns}, hardware_stamp_ns={last_hardware_stamp_ns}, "
            f"tf_stamp_ns={last_tf_stamp_ns})"
        )
        return False

    def cmd_pose_callback(self, msg):
        token = self._claim_motion("cmd_pose")
        if token is None:
            self.get_logger().warning(f"{self._motion_claim_rejection_reason()}; dropping /cmd_pose command")
            return

        self._prepare_motion(token)
        self.get_logger().info(f"Target Pose: x={msg.position.x:.3f}, y={msg.position.y:.3f}, z={msg.position.z:.3f}")
        # 计算并输出目标位置在shoulder坐标系中的Z轴坐标
        try:
            trans = self.tf_buffer.lookup_transform(
                self.shoulder_link,
                self.base_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            # 获取平移偏移（shoulder原点在base坐标系中的位置）
            trans_x = trans.transform.translation.x
            trans_y = trans.transform.translation.y
            trans_z = trans.transform.translation.z

            # 获取旋转变换
            t_x = trans.transform.rotation.x
            t_y = trans.transform.rotation.y
            t_z = trans.transform.rotation.z
            t_w = trans.transform.rotation.w

            # 目标位置在base坐标系中
            p_base = (msg.position.x, msg.position.y, msg.position.z)

            # 计算目标点相对于shoulder原点的位置向量
            # p_relative = p_base - translation
            p_relative = (p_base[0] - trans_x, p_base[1] - trans_y, p_base[2] - trans_z)

            # 应用旋转变换到相对位置向量
            R = self.quaternion_to_rotation_matrix((t_x, t_y, t_z, t_w))

            p_shoulder = (
                R[0][0] * p_relative[0] + R[0][1] * p_relative[1] + R[0][2] * p_relative[2],
                R[1][0] * p_relative[0] + R[1][1] * p_relative[1] + R[1][2] * p_relative[2],
                R[2][0] * p_relative[0] + R[2][1] * p_relative[1] + R[2][2] * p_relative[2],
            )

            # 计算距离shoulder原点的距离
            dist_shoulder = math.sqrt(p_shoulder[0] ** 2 + p_shoulder[1] ** 2 + p_shoulder[2] ** 2)

            # 计算距离base原点的距离
            dist_base = math.sqrt(p_base[0] ** 2 + p_base[1] ** 2 + p_base[2] ** 2)

            self.get_logger().info(
                f"  Target in shoulder frame: x={p_shoulder[0]:.3f}, y={p_shoulder[1]:.3f}, z={p_shoulder[2]:.3f}"
            )
            self.get_logger().info(f"  Distance from base origin: {dist_base:.3f} m")
            self.get_logger().info(f"  Distance from shoulder origin: {dist_shoulder:.3f} m")
        except Exception as e:
            self.get_logger().warning(f"Failed to transform to shoulder frame: {e}")

        try:
            if self._move_with_strategies(msg.position, msg.orientation):
                self._defer_motion_completion(token, forced_result=None)
            elif self._ensure_motion_stopped(token, "cmd_pose"):
                self._finalize_motion(token, success=False)
        except Exception as e:
            self.get_logger().error(f"/cmd_pose motion failed: {e}")
            if self._ensure_motion_stopped(token, "cmd_pose"):
                self._finalize_motion(token, success=False)

    def _move_with_strategies(self, position, orientation_msg) -> bool:
        """尝试多种 5-DOF 姿态策略 + 分层容差，直到 IK 成功。

        被 cmd_pose_callback 和 _move_to_pose_service_cb 共同调用，
        保证两条路径的 5-DOF 适配行为完全一致。

        Returns:
            True 表示某个策略成功，False 表示全部策略均失败。
        """
        orig_quat = (
            orientation_msg.x,
            orientation_msg.y,
            orientation_msg.z,
            orientation_msg.w,
        )

        # 零四元数保护
        if (
            abs(orig_quat[0]) < 1e-9
            and abs(orig_quat[1]) < 1e-9
            and abs(orig_quat[2]) < 1e-9
            and abs(orig_quat[3]) < 1e-9
        ):
            self.get_logger().warning("Received zero quaternion, using default orientation (0, 0, 0, 1)")
            orig_quat = (0.0, 0.0, 0.0, 1.0)

        # 5-DOF 姿态约束策略（从严格到宽松）
        strategies = [
            ("Gripper Z-axis constraint", self.constrain_to_z_axis_only(orig_quat)),
            (
                "Shoulder XZ plane projection",
                self.project_orientation_to_shoulder_xz_plane(orig_quat),
            ),
        ]

        # Fallback: 当前末端姿态（只改位置，保持现有姿态）
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            current_quat = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            )
            strategies.append(("Current orientation (position only)", current_quat))
        except Exception as e:
            self.get_logger().warning(f"Failed to get current orientation: {e}")
            strategies.append(("Default orientation (no rotation)", (0.0, 0.0, 0.0, 1.0)))

        # 分层容差策略（从严格到宽松）
        tolerance_strategies = [
            ("Strict tolerance", (0.1, 0.1, 0.05)),  # X/Y ~5.7°, Z ~2.8°
            ("Medium tolerance", (0.3, 0.3, 0.1)),  # X/Y ~17.2°, Z ~5.7°
            ("Relaxed tolerance", (0.5, 0.5, 0.15)),  # X/Y ~28.6°, Z ~8.6°
            ("Z-axis only", (1.0, 1.0, 0.2)),  # X/Y ~57.3°, Z ~11.5°
            ("No constraints", None),
        ]

        for strategy_name, quat in strategies:
            adjusted_pose = Pose()
            adjusted_pose.position = position
            adjusted_pose.orientation.x = quat[0]
            adjusted_pose.orientation.y = quat[1]
            adjusted_pose.orientation.z = quat[2]
            adjusted_pose.orientation.w = quat[3]

            self.get_logger().info(
                f"Trying {strategy_name}: "
                f"({orig_quat[0]:.3f}, {orig_quat[1]:.3f}, {orig_quat[2]:.3f}, {orig_quat[3]:.3f}) -> "
                f"({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
            )

            for tol_name, tolerances in tolerance_strategies:
                if self.solve_and_move(adjusted_pose, orientation_tolerance=tolerances):
                    self.get_logger().info(f"IK succeeded with {strategy_name} + {tol_name}")
                    return True
                else:
                    self.get_logger().debug(f"  Failed with {tol_name}, trying next...")

        self.get_logger().error("IK failed with all strategies!")
        return False

    def publish_ee_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01),
            )
            msg = PoseStamped()
            msg.header = trans.header
            msg.pose.position.x = trans.transform.translation.x
            msg.pose.position.y = trans.transform.translation.y
            msg.pose.position.z = trans.transform.translation.z
            msg.pose.orientation = trans.transform.rotation
            self.ee_pose_pub.publish(msg)
        except Exception:
            pass

    def _initialize_motion_coordinator(self):
        """Initialize gateway-level ownership for the shared MoveIt2 instance."""
        self._motion_lock = threading.Lock()
        self._motion_mode_transition_lock = threading.Lock()
        self._motion_token_counter = 0
        self._active_motion_token = None
        self._active_motion_owner = None
        self._active_motion_previous_velocity = None
        self._active_motion_deferred = False
        self._active_motion_forced_result = None
        self._active_motion_finalizing = False
        self._navigation_enabled = False
        self._navigation_mode_ack_condition = threading.Condition()
        self._base_navigation_enabled_ack = None
        self._last_bridge_heartbeat_monotonic = time.monotonic()
        self._motion_mode_fault = ""

    def _claim_motion(self, owner: str) -> int | None:
        """Atomically claim the shared MoveIt2 execution state."""
        with self._motion_lock:
            if self._navigation_enabled or self._active_motion_token is not None:
                return None

            self._motion_token_counter += 1
            token = self._motion_token_counter
            self._active_motion_token = token
            self._active_motion_owner = owner
            self._active_motion_previous_velocity = None
            self._active_motion_deferred = False
            self._active_motion_forced_result = None
            self._active_motion_finalizing = False
            return token

    def _motion_claim_rejection_reason(self) -> str:
        with self._motion_lock:
            if self._navigation_enabled:
                return "Arm motion is disabled while navigation mode is active"
            return "MoveIt gateway is busy"

    def navigation_mode_ack_callback(self, msg: Bool) -> None:
        """Record that the base bridge applied a mode and emitted the zero boundary."""
        with self._navigation_mode_ack_condition:
            self._base_navigation_enabled_ack = bool(msg.data)
            self._last_bridge_heartbeat_monotonic = time.monotonic()
            self._navigation_mode_ack_condition.notify_all()

    def _publish_navigation_mode(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.navigation_mode_pub.publish(msg)

    def _wait_for_base_mode_ack(self, enabled: bool) -> bool:
        deadline = time.monotonic() + self._motion_mode_transition_timeout_s
        with self._navigation_mode_ack_condition:
            while self._base_navigation_enabled_ack != enabled:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._navigation_mode_ack_condition.wait(timeout=remaining)
        return True

    def _request_base_mode(self, enabled: bool) -> bool:
        with self._navigation_mode_ack_condition:
            self._base_navigation_enabled_ack = None
        self._publish_navigation_mode(enabled)
        return self._wait_for_base_mode_ack(enabled)

    def _switch_controllers(self, *, activate: list[str], deactivate: list[str]) -> bool:
        if not activate and not deactivate:
            return True
        client = self.controller_switch_client
        if client is None or not client.wait_for_service(timeout_sec=self._motion_mode_transition_timeout_s):
            self.get_logger().error(f"Controller switch service unavailable: {self._controller_switch_service}")
            return False

        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=self._motion_mode_transition_timeout_s).to_msg()
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout=self._motion_mode_transition_timeout_s + 0.1):
            self.get_logger().error("Controller switch request timed out")
            return False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Controller switch failed: {exc}")
            return False
        if result is None or not bool(result.ok):
            self.get_logger().error("Controller manager rejected the motion-mode switch")
            return False
        return True

    def _set_navigation_enabled_serialized(self, requested: bool, response):
        with self._motion_lock:
            current = self._navigation_enabled
            if requested and not current:
                if self._active_motion_token is not None:
                    response.success = False
                    response.message = "Cannot enable navigation while an arm motion is active"
                    return response
                self._navigation_enabled = True

        if requested:
            if current:
                response.success = not bool(self._motion_mode_fault)
                response.message = self._motion_mode_fault or "Navigation already enabled; arm commands are blocked"
                return response

            if not self._switch_controllers(
                activate=[],
                deactivate=self._motion_mode_manipulation_controllers,
            ):
                with self._motion_lock:
                    self._navigation_enabled = False
                response.success = False
                response.message = "Failed to deactivate manipulation controllers"
                return response
            if not self._request_base_mode(True):
                self._switch_controllers(
                    activate=self._motion_mode_manipulation_controllers,
                    deactivate=[],
                )
                with self._motion_lock:
                    self._navigation_enabled = False
                response.success = False
                response.message = "Base bridge did not acknowledge navigation mode"
                return response
            if not self._switch_controllers(
                activate=self._motion_mode_navigation_controllers,
                deactivate=[],
            ):
                self._publish_navigation_mode(False)
                self._switch_controllers(
                    activate=self._motion_mode_manipulation_controllers,
                    deactivate=[],
                )
                with self._motion_lock:
                    self._navigation_enabled = False
                response.success = False
                response.message = "Failed to activate navigation controllers"
                return response
            self._motion_mode_fault = ""
            response.success = True
            response.message = "Navigation enabled; manipulation controllers are inactive"
            return response

        if not current:
            response.success = True
            response.message = "Manipulation already enabled; navigation controllers are inactive"
            return response

        if not self._request_base_mode(False):
            response.success = False
            response.message = "Arm commands remain blocked because the base bridge did not acknowledge stop"
            return response
        if not self._switch_controllers(
            activate=self._motion_mode_manipulation_controllers,
            deactivate=self._motion_mode_navigation_controllers,
        ):
            response.success = False
            response.message = "Arm commands remain blocked because controller switching failed"
            return response
        with self._motion_lock:
            self._navigation_enabled = False
        self._motion_mode_fault = ""
        response.success = True
        response.message = "Manipulation enabled; navigation controllers are inactive"
        return response

    def _set_navigation_enabled_service_cb(self, request, response):
        """Serialize controller ownership transitions across arm and base."""
        if not self._motion_mode_enabled:
            response.success = False
            response.message = "Persistent motion mode is disabled"
            return response

        if not self._motion_mode_transition_lock.acquire(blocking=False):
            response.success = False
            response.message = "Another motion-mode transition is in progress"
            return response
        try:
            return self._set_navigation_enabled_serialized(bool(request.data), response)
        finally:
            self._motion_mode_transition_lock.release()

    def _owns_motion(self, token: int) -> bool:
        with self._motion_lock:
            return self._active_motion_token == token

    def _prepare_motion(self, token: int) -> bool:
        """Reset per-motion MoveIt state after ownership has been acquired."""
        if not self._owns_motion(token):
            return False
        if self.moveit2 is not None:
            self.moveit2.motion_suceeded = False
        self._publish_motion_status("executing")
        return True

    def _set_motion_velocity(self, token: int, velocity_scaling: float) -> bool:
        """Apply a temporary velocity while retaining it for token-safe cleanup."""
        with self._motion_lock:
            if self._active_motion_token != token or self.moveit2 is None:
                return False
            self._active_motion_previous_velocity = self.moveit2.max_velocity
            self.moveit2.max_velocity = max(velocity_scaling, 0.001)
            return True

    def _query_moveit_state(self) -> MoveIt2State | None:
        if self.moveit2 is None:
            return MoveIt2State.IDLE
        try:
            return self.moveit2.query_state()
        except Exception as e:
            self.get_logger().error(f"Failed to query MoveIt2 state: {e}")
            return None

    def _motion_execution_succeeded(self) -> bool:
        if self.moveit2 is None or not self.moveit2.motion_suceeded:
            return False
        error_code = self.moveit2.get_last_execution_error_code()
        return error_code is None or int(error_code.val) == 1

    def _defer_motion_completion(self, token: int, forced_result: bool | None) -> bool:
        """Keep the token busy until the watchdog observes a terminal MoveIt state."""
        with self._motion_lock:
            if self._active_motion_token != token or self._active_motion_finalizing:
                return False
            self._active_motion_deferred = True
            self._active_motion_forced_result = forced_result
            return True

    def _finalize_motion(self, token: int, success: bool) -> bool:
        """Restore shared state and release ownership only after MoveIt is idle."""
        if self._query_moveit_state() != MoveIt2State.IDLE:
            self._defer_motion_completion(token, forced_result=success)
            return False

        with self._motion_lock:
            if self._active_motion_token != token or self._active_motion_finalizing:
                return False
            self._active_motion_finalizing = True
            previous_velocity = self._active_motion_previous_velocity

        if self.moveit2 is not None and previous_velocity is not None:
            self.moveit2.max_velocity = previous_velocity

        self._publish_motion_status("succeeded" if success else "failed")
        if self._motion_status_hold_s > 0.0:
            time.sleep(self._motion_status_hold_s)

        with self._motion_lock:
            if self._active_motion_token != token:
                return False
            # Publish idle before releasing the token so a new owner cannot have
            # its executing status overwritten by this completion path.
            self._publish_motion_status("idle")
            self._active_motion_token = None
            self._active_motion_owner = None
            self._active_motion_previous_velocity = None
            self._active_motion_deferred = False
            self._active_motion_forced_result = None
            self._active_motion_finalizing = False
        return True

    def _ensure_motion_stopped(self, token: int, context: str) -> bool:
        """Cancel an active request and wait briefly for MoveIt to become idle."""
        deadline = time.monotonic() + self._motion_cancel_timeout_s
        cancel_sent = False

        while self._owns_motion(token):
            state = self._query_moveit_state()
            if state == MoveIt2State.IDLE:
                return True
            if state == MoveIt2State.EXECUTING and not cancel_sent:
                self.moveit2.cancel_execution()
                cancel_sent = True
            if state is None or time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        self.get_logger().error(
            f"{context} did not reach MoveIt2 IDLE within {self._motion_cancel_timeout_s:.1f}s; gateway remains busy"
        )
        self._defer_motion_completion(token, forced_result=False)
        return False

    def _motion_watchdog_callback(self):
        """Finalize asynchronous or timed-out motion after MoveIt becomes idle."""
        if self._motion_mode_enabled:
            with self._motion_lock:
                navigation_enabled = self._navigation_enabled
            heartbeat_age = time.monotonic() - self._last_bridge_heartbeat_monotonic
            if (
                navigation_enabled
                and self._motion_mode_bridge_heartbeat_timeout_s > 0.0
                and heartbeat_age > self._motion_mode_bridge_heartbeat_timeout_s
                and not self._motion_mode_fault
                and self._motion_mode_transition_lock.acquire(blocking=False)
            ):
                try:
                    if self._switch_controllers(
                        activate=[],
                        deactivate=self._motion_mode_navigation_controllers,
                    ):
                        self._motion_mode_fault = "Base bridge heartbeat lost; navigation controllers were stopped"
                        self.get_logger().error(self._motion_mode_fault)
                finally:
                    self._motion_mode_transition_lock.release()

        with self._motion_lock:
            if self._active_motion_token is None or not self._active_motion_deferred or self._active_motion_finalizing:
                return
            token = self._active_motion_token
            forced_result = self._active_motion_forced_result

        if self._query_moveit_state() != MoveIt2State.IDLE:
            return
        success = self._motion_execution_succeeded() if forced_result is None else forced_result
        self._finalize_motion(token, success=success)

    def _publish_motion_status(self, status: str):
        """Publish motion status for external observers (task_dispatch, etc.)."""
        self._motion_status = status
        msg = String()
        msg.data = status
        self.motion_status_pub.publish(msg)

    def _move_to_pose_service_cb(self, request, response):
        """Synchronous move-to-pose service handler for task_dispatch integration.

        Performs the full IK + plan + execute pipeline and blocks until
        motion completes (or fails/times out).
        """
        t0 = time.monotonic()
        token = self._claim_motion("MoveToPose")
        if token is None:
            response.success = False
            response.message = self._motion_claim_rejection_reason()
            response.execution_time_s = time.monotonic() - t0
            return response

        target = request.target_pose
        self.get_logger().info(
            f"[Service] MoveToPose request: ({target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f})"
        )
        self._prepare_motion(token)
        terminal_confirmed = True

        try:
            # Apply the same 5-DOF orientation strategies as cmd_pose_callback
            success = self._move_with_strategies(target.position, target.orientation)
            if success:
                response.success, response.message, terminal_confirmed = self._wait_for_motion_completion(
                    token, "MoveToPose"
                )
                if response.success and not self._wait_for_post_motion_feedback():
                    response.success = False
                    response.message = "Motion completed but fresh joint feedback was not observed"
            else:
                response.success = False
                response.message = "IK/planning failed"
                terminal_confirmed = self._ensure_motion_stopped(token, "MoveToPose planning failure")
        except Exception as e:
            response.success = False
            response.message = f"Exception: {e}"
            self.get_logger().error(f"[Service] MoveToPose exception: {e}")
            terminal_confirmed = self._ensure_motion_stopped(token, "MoveToPose exception")

        response.execution_time_s = time.monotonic() - t0
        if terminal_confirmed:
            self._finalize_motion(token, success=response.success)
        self.get_logger().info(
            f"[Service] MoveToPose result: success={response.success}, time={response.execution_time_s:.1f}s"
        )
        return response

    def _move_to_configuration_service_cb(self, request, response):
        """Plan and execute a caller-provided IK solution without re-solving IK."""
        t0 = time.monotonic()
        token = self._claim_motion("MoveToConfiguration")
        if token is None:
            response.success = False
            response.message = self._motion_claim_rejection_reason()
            response.execution_time_s = time.monotonic() - t0
            return response

        self._prepare_motion(token)
        terminal_confirmed = True

        try:
            target = request.target_joint_state
            if len(target.name) != len(target.position):
                raise ValueError(
                    f"target_joint_state name/position length mismatch: {len(target.name)} != {len(target.position)}"
                )
            if len(set(target.name)) != len(target.name):
                raise ValueError("target_joint_state contains duplicate joint names")
            positions_by_name = {
                str(name): float(position) for name, position in zip(target.name, target.position, strict=False)
            }
            missing = [name for name in self.joint_names if name not in positions_by_name]
            if missing:
                raise ValueError(f"target_joint_state is missing arm joints: {missing}")

            joint_positions = [positions_by_name[name] for name in self.joint_names]
            if not all(math.isfinite(position) for position in joint_positions):
                raise ValueError("target_joint_state contains non-finite positions")

            velocity_scaling = float(request.velocity_scaling)
            if not math.isfinite(velocity_scaling) or not 0.0 <= velocity_scaling <= 1.0:
                raise ValueError("velocity_scaling must be finite and within [0.0, 1.0]")
            if self.moveit2 is not None and velocity_scaling > 0.0:
                self._set_motion_velocity(token, velocity_scaling)

            self.get_logger().info(
                f"[Service] MoveToConfiguration request: joints={self.joint_names} positions={joint_positions}"
            )
            success = self.move_to_joint(joint_positions)
            if success:
                response.success, response.message, terminal_confirmed = self._wait_for_motion_completion(
                    token, "MoveToConfiguration"
                )
                arm_feedback_targets = {name: positions_by_name[name] for name in self.joint_names}
                if response.success and not self._wait_for_post_motion_feedback(arm_feedback_targets):
                    response.success = False
                    response.message = "Motion completed but joint feedback did not converge to the target"
            else:
                response.success = False
                response.message = "Joint planning failed"
                terminal_confirmed = self._ensure_motion_stopped(token, "MoveToConfiguration planning failure")
        except Exception as e:
            response.success = False
            response.message = f"Exception: {e}"
            self.get_logger().error(f"[Service] MoveToConfiguration exception: {e}")
            terminal_confirmed = self._ensure_motion_stopped(token, "MoveToConfiguration exception")

        response.execution_time_s = time.monotonic() - t0
        if terminal_confirmed:
            self._finalize_motion(token, success=response.success)
        self.get_logger().info(
            f"[Service] MoveToConfiguration result: success={response.success}, time={response.execution_time_s:.1f}s"
        )
        return response

    def _wait_for_motion_completion(self, token: int, service_name: str) -> tuple[bool, str, bool]:
        """Wait for the asynchronous MoveIt action started by a service callback."""
        moveit2 = self.moveit2
        if moveit2 is None:
            return False, "MoveIt2 engine not ready", True

        start_deadline = time.monotonic() + self._motion_start_timeout_s
        state = self._query_moveit_state()
        while state == MoveIt2State.REQUESTING and time.monotonic() < start_deadline:
            time.sleep(0.05)
            state = self._query_moveit_state()

        if state == MoveIt2State.REQUESTING:
            self._defer_motion_completion(token, forced_result=False)
            return (
                False,
                f"Motion request did not start within {self._motion_start_timeout_s:.1f}s; gateway remains busy",
                False,
            )
        if state is None:
            self._defer_motion_completion(token, forced_result=False)
            return False, "Unable to confirm MoveIt2 state; gateway remains busy", False
        if state == MoveIt2State.IDLE:
            if self._motion_execution_succeeded():
                return True, "Motion completed", True
            return False, "Motion execution failed (MoveIt reported unsuccessful)", True

        exec_started = time.monotonic()
        exec_deadline = exec_started + self._motion_execution_timeout_s
        while state == MoveIt2State.EXECUTING and time.monotonic() < exec_deadline:
            time.sleep(0.1)
            state = self._query_moveit_state()

        execution_time = time.monotonic() - exec_started
        if state == MoveIt2State.EXECUTING:
            self.get_logger().warn(f"[Service] {service_name} execution timed out after {execution_time:.1f}s")
            cancellation_confirmed = self._ensure_motion_stopped(token, f"{service_name} cancellation")
            if cancellation_confirmed:
                return False, f"Execution timed out after {execution_time:.1f}s and was cancelled", True
            return (
                False,
                f"Execution timed out after {execution_time:.1f}s; cancellation is still pending",
                False,
            )
        if state == MoveIt2State.REQUESTING or state is None:
            self._defer_motion_completion(token, forced_result=False)
            return False, "MoveIt2 did not reach a terminal state; gateway remains busy", False
        if self._motion_execution_succeeded():
            return True, "Motion completed", True

        error_code = moveit2.get_last_execution_error_code()
        if error_code is not None and int(error_code.val) != 1:
            return False, f"Motion execution failed with MoveIt error code {int(error_code.val)}", True
        return False, "Motion execution failed (MoveIt reported unsuccessful)", True

    def solve_and_move(self, target_pose, orientation_tolerance=None):
        """
        尝试IK求解并移动到目标位姿。

        Args:
            target_pose: 目标位姿
            orientation_tolerance: 姿态容差 or None（无constraints）

        Returns:
            True表示成功，False表示失败
        """
        if not self.moveit2:
            self.get_logger().error("MoveIt2 engine not ready")
            return False

        # 打印目标位置（用于调试可达性）
        target_pos = target_pose.position
        self.get_logger().info(f"  Target position: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")

        # 简单的可达性检查：距离原点的距离
        dist_from_origin = math.sqrt(target_pos.x**2 + target_pos.y**2 + target_pos.z**2)
        self.get_logger().info(f"  Distance from origin: {dist_from_origin:.3f} m")

        # 打印当前关节状态（如果有）
        if self.latest_joint_state is not None and hasattr(self.latest_joint_state, "position"):
            self.get_logger().debug(f"  Current joints: {[f'{p:.2f}' for p in self.latest_joint_state.position]}")

        try:
            # 检查关节状态是否有效
            start_state = None
            if self.latest_joint_state is not None:
                # 验证关节状态是否包含所需的关节数量
                if hasattr(self.latest_joint_state, "position") and len(self.latest_joint_state.position) >= len(
                    self.joint_names
                ):
                    start_state = self.latest_joint_state
                else:
                    self.get_logger().warning(
                        f"Invalid joint state: has {len(self.latest_joint_state.position) if hasattr(self.latest_joint_state, 'position') else 0} joints, "
                        f"need {len(self.joint_names)}. Using solver's internal state."
                    )
            else:
                self.get_logger().warning("No joint state available, using solver's internal state")

            # 创建Constraints（如果指定了容差）
            constraints = None
            if orientation_tolerance is not None:
                constraints = Constraints()
                target_quat = (
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                )
                constraints.orientation_constraints.append(
                    self.create_orientation_constraint(
                        target_quat=target_quat,
                        link_name=self.ee_link,
                        frame_id=self.base_link,
                        tolerances=orientation_tolerance,
                    )
                )
                self.get_logger().info(f"Using orientation tolerance: {orientation_tolerance}")

            # 1. Use async IK call to avoid internal spin_once calls
            # 只在有有效状态时才传递start_joint_state参数
            if start_state is not None:
                if constraints is not None:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        start_joint_state=start_state,
                        constraints=constraints,
                    )
                else:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        start_joint_state=start_state,
                    )
            else:
                # 不传递start_joint_state，让求解器使用内部状态
                if constraints is not None:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        constraints=constraints,
                    )
                else:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position, quat_xyzw=target_pose.orientation
                    )

            # 2. Wait for the future safely in a MultiThreadedExecutor environment
            # Since the executor is running in parallel, it will fulfill the future.
            start_wait = time.monotonic()
            while not future.done():
                time.sleep(0.01)
                if time.monotonic() - start_wait > 5.0:
                    self.get_logger().error("IK Service Timeout")
                    return False

            ik_solution = self.moveit2.get_compute_ik_result(future)

            if ik_solution is not None:
                joint_positions = []
                for name in self.joint_names:
                    if name in ik_solution.name:
                        idx = ik_solution.name.index(name)
                        joint_positions.append(float(ik_solution.position[idx]))

                self.get_logger().info(f"IK Success: {joint_positions}")
                return self.move_to_joint(joint_positions)
            else:
                self.get_logger().warning("IK Solver failed: No valid solution")
                # 检查IK求解器是否支持Constraints
                if orientation_tolerance is not None:
                    self.get_logger().warning(
                        "  Note: Constraints may not be supported by LMA solver. Try position_only_ik: True in kinematics.yaml"
                    )
                return False

        except Exception as e:
            self.get_logger().error(f"IK Workflow failed: {e}")
            return False

    def move_to_joint(self, joint_positions):
        if not self.moveit2:
            return False
        self.get_logger().info("Moving to joints...")
        try:
            self.moveit2.clear_goal_constraints()
            self.moveit2.move_to_configuration(joint_positions)
            state = self._query_moveit_state()
            return state in (MoveIt2State.REQUESTING, MoveIt2State.EXECUTING) or self._motion_execution_succeeded()
        except Exception as e:
            self.get_logger().error(f"Move error: {e}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = MoveItGateway()

    # Use MultiThreadedExecutor to handle concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
