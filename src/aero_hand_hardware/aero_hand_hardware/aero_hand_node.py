"""ROS 2 command and state bridge for an Aero Hand."""

from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from .aero_hand_driver import JOINT_COUNT, AeroHandDriver

DEFAULT_RIGHT_JOINT_NAMES = [
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp_ip",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]


def _advance_deadline(deadline: float, now: float, period: float) -> float:
    while deadline <= now:
        deadline += period
    return deadline


def _get_float_array_parameter(node: Node, name: str) -> list[float]:
    fallback = Parameter(name, Parameter.Type.DOUBLE_ARRAY, [])
    return [float(value) for value in node.get_parameter_or(name, fallback).value]


class AeroHandNode(Node):
    """Cache radians commands and perform all SDK I/O from one timer."""

    def __init__(self, driver: AeroHandDriver | None = None):
        super().__init__("aero_hand")
        self.declare_parameter("port", "")
        self.declare_parameter("baudrate", 921600)
        self.declare_parameter("mock", False)
        self.declare_parameter("joint_names", DEFAULT_RIGHT_JOINT_NAMES)
        self.declare_parameter("command_topic", "/aero_hand_right/commands")
        self.declare_parameter("joint_state_topic", "/aero_hand_right/joint_states")
        self.declare_parameter("command_frequency", 50.0)
        self.declare_parameter("state_frequency", 20.0)
        self.declare_parameter("command_timeout", 0.25)
        self.declare_parameter("command_lower_limits", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("command_upper_limits", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("estop_topic", "/emergency_stop")
        self.declare_parameter("estop_behavior", "hold")
        self.declare_parameter("safe_pose", Parameter.Type.DOUBLE_ARRAY)

        port = str(self.get_parameter("port").value).strip()
        baudrate = int(self.get_parameter("baudrate").value)
        mock = bool(self.get_parameter("mock").value)
        self.joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        command_topic = str(self.get_parameter("command_topic").value)
        joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.command_frequency = float(self.get_parameter("command_frequency").value)
        self.state_frequency = float(self.get_parameter("state_frequency").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.command_lower_limits = _get_float_array_parameter(self, "command_lower_limits")
        self.command_upper_limits = _get_float_array_parameter(self, "command_upper_limits")
        self.estop_topic = str(self.get_parameter("estop_topic").value).strip()
        self.estop_behavior = str(self.get_parameter("estop_behavior").value).strip().lower()
        self.safe_pose_rad = _get_float_array_parameter(self, "safe_pose")

        self._validate_parameters(command_topic, joint_state_topic)
        self.command_limits = (
            list(zip(self.command_lower_limits, self.command_upper_limits, strict=True))
            if self.command_lower_limits
            else [(-math.inf, math.inf)] * JOINT_COUNT
        )
        self.driver = driver or AeroHandDriver(
            port=port or None,
            baudrate=baudrate,
            mock=mock,
            command_frequency=self.command_frequency,
            state_frequency=self.state_frequency,
            logger=self.get_logger(),
        )
        self.driver.connect()

        self._command_lock = threading.Lock()
        self._latest_command_rad: list[float] | None = None
        self._latest_command_time = 0.0
        self._estop_active = False
        self._safe_pose_pending = False
        self._next_state_read = time.monotonic()
        self._last_warning: dict[str, float] = {}

        self.command_sub = self.create_subscription(
            Float64MultiArray,
            command_topic,
            self.command_callback,
            10,
        )
        self.estop_sub = self.create_subscription(Bool, self.estop_topic, self.estop_callback, 10)
        self.joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.timer = self.create_timer(1.0 / self.command_frequency, self.control_callback)
        self.get_logger().info(f"AeroHandNode ready: mock={mock}, command={command_topic}, state={joint_state_topic}")

    def _validate_parameters(self, command_topic: str, joint_state_topic: str) -> None:
        if len(self.joint_names) != JOINT_COUNT or len(set(self.joint_names)) != JOINT_COUNT:
            raise ValueError("joint_names must contain seven unique names")
        if not command_topic.strip() or not joint_state_topic.strip():
            raise ValueError("command_topic and joint_state_topic must be non-empty")
        if self.command_frequency <= 0.0 or self.state_frequency <= 0.0:
            raise ValueError("command_frequency and state_frequency must be positive")
        if self.command_timeout <= 0.0:
            raise ValueError("command_timeout must be positive")
        if bool(self.command_lower_limits) != bool(self.command_upper_limits):
            raise ValueError("command_lower_limits and command_upper_limits must be provided together")
        if self.command_lower_limits:
            if len(self.command_lower_limits) != JOINT_COUNT or len(self.command_upper_limits) != JOINT_COUNT:
                raise ValueError("command limits must contain exactly seven lower and upper values")
            if any(
                not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower
                for lower, upper in zip(self.command_lower_limits, self.command_upper_limits, strict=True)
            ):
                raise ValueError("command limits must contain finite increasing ranges")
        if not self.estop_topic:
            raise ValueError("estop_topic must be non-empty")
        if self.estop_behavior not in ("hold", "safe_pose"):
            raise ValueError("estop_behavior must be 'hold' or 'safe_pose'")
        if self.estop_behavior == "safe_pose" and (
            len(self.safe_pose_rad) != JOINT_COUNT or not all(math.isfinite(value) for value in self.safe_pose_rad)
        ):
            raise ValueError("safe_pose must contain seven finite radian positions when estop_behavior is safe_pose")
        if (
            self.estop_behavior == "safe_pose"
            and self.command_lower_limits
            and any(
                value < lower or value > upper
                for value, lower, upper in zip(
                    self.safe_pose_rad,
                    self.command_lower_limits,
                    self.command_upper_limits,
                    strict=True,
                )
            )
        ):
            raise ValueError("safe_pose must stay within the configured command limits")

    def command_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != JOINT_COUNT or not all(math.isfinite(value) for value in msg.data):
            self._warn_rate_limited("invalid_command", "Rejected invalid Aero Hand command")
            return
        bounded = [
            max(lower, min(upper, float(value)))
            for value, (lower, upper) in zip(msg.data, self.command_limits, strict=True)
        ]
        if any(
            not math.isclose(float(value), limited, abs_tol=1e-9)
            for value, limited in zip(msg.data, bounded, strict=True)
        ):
            self._warn_rate_limited("limited_command", "Clamped Aero Hand command to configured joint limits")
        with self._command_lock:
            if self._estop_active:
                return
            self._latest_command_rad = bounded
            self._latest_command_time = time.monotonic()

    def estop_callback(self, msg: Bool) -> None:
        """Latch the node and driver so no queued command can cross E-stop."""
        active = bool(msg.data)
        if active:
            with self._command_lock:
                self._estop_active = True
                self._latest_command_rad = None
                self._latest_command_time = 0.0
                self._safe_pose_pending = self.estop_behavior == "safe_pose"
            self.driver.set_emergency_stop(True)
            return

        self.driver.set_emergency_stop(False)
        with self._command_lock:
            self._estop_active = False
            self._latest_command_rad = None
            self._latest_command_time = 0.0
            self._safe_pose_pending = False

    def control_callback(self) -> None:
        now = time.monotonic()
        with self._command_lock:
            estop_active = self._estop_active
            if estop_active:
                safe_pose_pending = self._safe_pose_pending
                if safe_pose_pending:
                    command_rad = list(self.safe_pose_rad)
                else:
                    command_rad = None
                command_time = now
                command_age = 0.0
            else:
                safe_pose_pending = False
                command_rad = list(self._latest_command_rad) if self._latest_command_rad is not None else None
                command_time = self._latest_command_time
                command_age = now - command_time

        if estop_active:
            if safe_pose_pending:
                try:
                    # Blocking so a failed safe-pose write is retried next cycle
                    # instead of being silently dropped into the command queue.
                    self.driver.set_joint_positions(
                        [math.degrees(value) for value in command_rad],
                        blocking=True,
                        allow_during_estop=True,
                    )
                except Exception as exc:
                    self._warn_rate_limited("safe_pose_io", f"Aero Hand E-stop safe pose failed: {exc}")
                else:
                    with self._command_lock:
                        if self._estop_active:
                            self._safe_pose_pending = False
        elif command_rad is not None:
            if command_age <= self.command_timeout:
                try:
                    self.driver.set_joint_positions([math.degrees(value) for value in command_rad])
                except Exception as exc:
                    self._warn_rate_limited("command_io", f"Aero Hand command failed: {exc}")
            else:
                self._warn_rate_limited(
                    "stale_command",
                    f"Aero Hand command is stale ({command_age:.3f}s); holding the last hardware position",
                )
                with self._command_lock:
                    if self._latest_command_time == command_time:
                        self._latest_command_rad = None

        if now < self._next_state_read:
            return
        self._next_state_read = _advance_deadline(self._next_state_read, now, 1.0 / self.state_frequency)
        try:
            positions_rad = [math.radians(value) for value in self.driver.get_joint_positions()]
        except Exception as exc:
            self._warn_rate_limited("state_io", f"Aero Hand state read failed: {exc}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = positions_rad
        self.joint_state_pub.publish(msg)

    def _warn_rate_limited(self, key: str, message: str, period: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_warning.get(key, 0.0) < period:
            return
        self._last_warning[key] = now
        self.get_logger().warning(message)

    def destroy_node(self):
        try:
            self.driver.disconnect()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AeroHandNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
