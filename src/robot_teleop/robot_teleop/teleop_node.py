"""
TeleopNode - Main ROS 2 node for teleoperation control

This node bridges teleoperation devices to robot controllers,
providing zero-latency control with safety filtering.
"""

import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .base_teleop import BaseTeleopDevice
from .device_factory import device_factory
from .safety_filter import SafetyFilter


class TeleopNode(Node):
    """
    Main teleoperation control node.

    This node:
    1. Loads teleoperation device from configuration
    2. Reads joint targets from device at high frequency
    3. Applies safety filtering (joint limits)
    4. Publishes commands to robot controllers

    Publishers:
        - arm_command_topic (Float64MultiArray, default /arm_position_controller/commands)
        - gripper_command_topic (Float64MultiArray, default /gripper_position_controller/commands)
        - /diagnostics (DiagnosticArray)

    Subscribers:
        - /emergency_stop (Bool) - Emergency stop signal

    Parameters:
        - control_frequency (double): Control loop frequency in Hz (default: 50.0)
        - device_config (dict): Teleoperation device configuration
        - joint_limits (dict): Joint limits for safety filter
        - arm_command_topic (string): Arm controller command topic
        - gripper_command_topic (string): Gripper controller command topic
    """

    def __init__(self):
        """Initialize teleop node."""
        super().__init__("robot_teleop_node")

        # Declare parameters
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("device_config", "")
        self.declare_parameter("joint_limits", "")
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.declare_parameter("gripper_joint_names", ["6"])

        self.declare_parameter("arm_command_topic", "/arm_position_controller/commands")
        self.declare_parameter("gripper_command_topic", "/gripper_position_controller/commands")

        # Get parameters
        self.control_frequency = self.get_parameter("control_frequency").value
        device_config_str = self.get_parameter("device_config").value
        joint_limits_str = self.get_parameter("joint_limits").value
        self.arm_joint_names = self.get_parameter("arm_joint_names").value
        self.gripper_joint_names = self.get_parameter("gripper_joint_names").value
        self.arm_command_topic = self.get_parameter("arm_command_topic").value
        self.gripper_command_topic = self.get_parameter("gripper_command_topic").value

        # Parse JSON parameters if provided as strings
        import json

        device_config = (
            json.loads(device_config_str) if isinstance(device_config_str, str) and device_config_str else {}
        )
        joint_limits = json.loads(joint_limits_str) if isinstance(joint_limits_str, str) and joint_limits_str else {}

        # Initialize device
        self.device: BaseTeleopDevice | None = None
        self._device_lock = threading.Lock()

        try:
            self.device = device_factory(device_config, node=self)
            self.get_logger().info(f"Created device: {device_config.get('type', 'unknown')}")

            # Connect to device
            if self.device.connect():
                self.get_logger().info("Device connected successfully")
            else:
                self.get_logger().error("Device connection failed")
        except Exception as e:
            self.get_logger().error(f"Failed to create/connect device: {e}")
            raise

        # Initialize safety filter
        self.safety_filter = SafetyFilter(joint_limits)

        # Publishers
        self.arm_cmd_pub = self.create_publisher(Float64MultiArray, self.arm_command_topic, 10)

        self.gripper_cmd_pub = self.create_publisher(Float64MultiArray, self.gripper_command_topic, 10)

        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # Emergency stop
        self.estop_active = False
        self.estop_sub = self.create_subscription(
            JointState,  # Using JointState as placeholder for Bool
            "/emergency_stop",
            self.estop_callback,
            10,
        )

        # Control loop timer
        timer_period = 1.0 / self.control_frequency  # seconds
        self.timer = self.create_timer(
            timer_period, self.control_loop_callback, callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Diagnostics
        self.loop_count = 0
        self.last_loop_time = time.time()
        self.avg_loop_time = 0.0
        self.max_loop_time = 0.0

        self.get_logger().info(f"TeleopNode initialized at {self.control_frequency} Hz")

    def control_loop_callback(self):
        """
        Main control loop - called at control_frequency.

        Reads device, applies safety, publishes commands.
        """
        loop_start = time.time()

        # Skip if emergency stop active
        if self.estop_active:
            return

        # Read from device
        with self._device_lock:
            if self.device is None or not self.device.is_connected:
                return

            try:
                joint_targets = self.device.get_joint_targets()
            except Exception as e:
                self.get_logger().error(f"Device read failed: {e}")
                return

        # Apply safety filter
        safe_targets = self.safety_filter.apply_limits(joint_targets)

        if not safe_targets:
            return

        # Only publish arm commands when all arm keys are present (avoids sending
        # zeros when PhoneDevice omits arm keys during Servo Cartesian control)
        if all(name in safe_targets for name in self.arm_joint_names):
            arm_msg = Float64MultiArray()
            arm_msg.data = [safe_targets[name] for name in self.arm_joint_names]
            self.arm_cmd_pub.publish(arm_msg)

        if self.gripper_joint_names and all(name in safe_targets for name in self.gripper_joint_names):
            gripper_msg = Float64MultiArray()
            gripper_msg.data = [safe_targets[name] for name in self.gripper_joint_names]
            self.gripper_cmd_pub.publish(gripper_msg)

        # Update diagnostics
        loop_time = time.time() - loop_start
        self._update_diagnostics(loop_time)

    def estop_callback(self, msg):
        """Handle emergency stop signal."""
        # For now, treat any message as E-stop trigger
        # In production, would check Bool message
        self.estop_active = True
        self.get_logger().warn("Emergency stop activated!")

    def _update_diagnostics(self, loop_time: float):
        """Update diagnostic statistics."""
        self.loop_count += 1

        # Update timing stats
        if self.loop_count == 1:
            self.avg_loop_time = loop_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_loop_time = alpha * loop_time + (1 - alpha) * self.avg_loop_time

        self.max_loop_time = max(self.max_loop_time, loop_time)

        # Publish diagnostics every 50 cycles
        if self.loop_count % 50 == 0:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            status = DiagnosticStatus()
            status.name = "robot_teleop"
            status.level = DiagnosticStatus.OK if self.avg_loop_time < 0.005 else DiagnosticStatus.WARN
            status.message = f"Loop time: avg={self.avg_loop_time * 1000:.2f}ms, max={self.max_loop_time * 1000:.2f}ms"

            diag_msg.status.append(status)
            self.diag_pub.publish(diag_msg)

            # Log warning if latency high
            if self.avg_loop_time > 0.005:  # 5ms threshold
                self.get_logger().warn(f"High latency detected: {self.avg_loop_time * 1000:.2f}ms > 5ms")

    def destroy_node(self):
        """Clean up resources on node shutdown."""
        self.get_logger().info("Shutting down TeleopNode...")

        with self._device_lock:
            if self.device is not None:
                try:
                    self.device.disconnect()
                    self.get_logger().info("Device disconnected")
                except Exception as e:
                    self.get_logger().error(f"Error disconnecting device: {e}")

        super().destroy_node()


def main(args=None):
    """Entry point for teleop_node."""
    rclpy.init(args=args)

    try:
        node = TeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"TeleopNode failed: {e}")
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()
