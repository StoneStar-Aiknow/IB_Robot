"""LeKiwi base node: cmd_vel projection, wheel odometry, and the navigation gating interface.

Relocated from the generic ``robot_navigation`` cmd_vel bridge into the LeKiwi
runtime (design D5/D2). Kinematics come from ``lekiwi_sdk`` (pure, bus-free
functions), so the ros2_control path and the SDK share one implementation.

- ``/cmd_vel`` (Twist) -> ``base_velocity_controller/commands`` (wheel rad/s),
  accepted only while navigation is enabled, the runtime mode allows base
  commands, and the runtime stop latch is clear; zeroed when commands go
  stale (``base.cmd_vel.staleness_s``).
- ``/joint_states`` wheel feedback -> ``/odom`` + TF (odom -> base_link).
- Navigation gating (``base.navigation_gate``): ``SetBool`` service that
  clears pending commands and stops the base before acknowledging, plus an
  acknowledgment heartbeat. Names are the ones navigation stacks already use.
"""

from __future__ import annotations

import sys
import threading
import time

import lekiwi_sdk_py as lk
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster

from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime import contract as C


class LeKiwiBaseNode(Node):
    def __init__(self):
        super().__init__("lekiwi_base")
        self.declare_parameter("wheel_joints", ["7", "8", "9"])
        self.declare_parameter("wheel_radius", 0.05)
        self.declare_parameter("base_radius", 0.125)
        self.declare_parameter("max_wheel_radps", 4.602)
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("staleness_s", 0.5)
        self.declare_parameter("cmd_vel_topic", C.CMD_VEL_TOPIC)
        self.declare_parameter("wheel_command_topic", "/base_velocity_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("odom_topic", C.ODOM_TOPIC)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("runtime_status_topic", C.STATUS_TOPIC)
        self.declare_parameter("base_modes", ["base_navigation"])
        self.declare_parameter("navigation_enable_service", C.NAVIGATION_ENABLE_SERVICE)
        self.declare_parameter("navigation_ack_topic", C.NAVIGATION_ACK_TOPIC)
        self.declare_parameter("navigation_enabled_on_startup", False)
        self.declare_parameter("ack_frequency", 10.0)

        self._wheel_joints = [str(j) for j in self.get_parameter("wheel_joints").value]
        self._geometry = lk.BaseGeometry()
        self._geometry.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self._geometry.base_radius = float(self.get_parameter("base_radius").value)
        self._geometry.max_wheel_radps = float(self.get_parameter("max_wheel_radps").value)
        rate = float(self.get_parameter("control_frequency").value)
        self._staleness = float(self.get_parameter("staleness_s").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._base_modes = {str(m) for m in self.get_parameter("base_modes").value}

        self._lock = threading.Lock()
        self._target = (0.0, 0.0, 0.0)
        self._last_cmd = 0.0
        self._nav_enabled = bool(self.get_parameter("navigation_enabled_on_startup").value)
        self._stop_latched = False
        self._active_mode = ""
        self._status_seen = False
        self._pose = lk.BasePose()
        self._last_wheel_positions: list[float] | None = None
        self._last_wheel_velocities = [0.0, 0.0, 0.0]
        self._last_published_zero = False

        cb = ReentrantCallbackGroup()
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10)
        self._wheel_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("wheel_command_topic").value), reliable
        )
        self._odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), reliable)
        self._ack_pub = self.create_publisher(Bool, str(self.get_parameter("navigation_ack_topic").value), reliable)
        self._tf = TransformBroadcaster(self) if self._publish_tf else None
        self.create_subscription(
            Twist, str(self.get_parameter("cmd_vel_topic").value), self._on_cmd_vel, 10, callback_group=cb
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_states,
            50,
            callback_group=cb,
        )
        self.create_subscription(
            RuntimeStatus,
            str(self.get_parameter("runtime_status_topic").value),
            self._on_runtime_status,
            10,
            callback_group=cb,
        )
        self.create_service(
            SetBool,
            str(self.get_parameter("navigation_enable_service").value),
            self._on_set_navigation,
            callback_group=cb,
        )
        self.create_timer(1.0 / rate, self._control_loop, callback_group=cb)
        ack_hz = max(float(self.get_parameter("ack_frequency").value), 0.1)
        self.create_timer(1.0 / ack_hz, self._publish_ack, callback_group=cb)
        self.get_logger().info(
            f"LeKiwi base: wheels {self._wheel_joints}, navigation_enabled={self._nav_enabled}, "
            f"base modes {sorted(self._base_modes)}"
        )

    # --- gating -------------------------------------------------------------------

    def _commands_permitted(self) -> bool:
        if self._stop_latched or not self._nav_enabled:
            return False
        # Fail closed until the runtime has spoken. Before the first
        # RuntimeStatus arrives the mode and the stop latch are both unknown,
        # and "unknown" must not mean "allowed": a profile may enable
        # navigation on startup, which would otherwise let /cmd_vel drive the
        # base during the window before the runtime reconciles.
        return self._status_seen and self._active_mode in self._base_modes

    def _on_runtime_status(self, msg: RuntimeStatus) -> None:
        with self._lock:
            self._stop_latched = bool(msg.stop_latched)
            self._active_mode = str(msg.active_mode)
            self._status_seen = True
            if self._stop_latched or self._active_mode not in self._base_modes:
                self._target = (0.0, 0.0, 0.0)

    def _on_set_navigation(self, request, response):
        # Clear-and-stop before acknowledging. The zero boundary is published
        # while the lock is held, so a control-loop callback cannot interleave
        # a stale non-zero command between this stop and the acknowledgment.
        with self._lock:
            self._target = (0.0, 0.0, 0.0)
            self._last_cmd = 0.0
            self._nav_enabled = bool(request.data)
            self._publish_wheels([0.0, 0.0, 0.0])
            self._last_published_zero = True
        self._publish_ack()
        response.success = True
        response.message = "navigation enabled" if self._nav_enabled else "navigation disabled"
        return response

    def _publish_ack(self) -> None:
        self._ack_pub.publish(Bool(data=self._nav_enabled))

    # --- command path ---------------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._lock:
            if not self._commands_permitted():
                return
            self._target = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self._last_cmd = time.monotonic()

    def _publish_wheels(self, radps: list[float]) -> None:
        self._wheel_pub.publish(Float64MultiArray(data=[float(v) for v in radps]))

    def _control_loop(self) -> None:
        # The gate check, the target read and the publish are one critical
        # section. Publishing outside it let the navigation-disable service
        # complete in the gap, so a stale non-zero wheel command could reach
        # the controller *after* the stop and the acknowledgment.
        with self._lock:
            stale = self._last_cmd == 0.0 or (time.monotonic() - self._last_cmd) > self._staleness
            active = self._commands_permitted() and not stale
            vx, vy, wz = self._target if active else (0.0, 0.0, 0.0)
            if active:
                self._publish_wheels(lk.body_to_wheel_velocities(vx, vy, wz, self._geometry))
                self._last_published_zero = False
            elif not self._last_published_zero:
                self._publish_wheels([0.0, 0.0, 0.0])
                self._last_published_zero = True

    # --- odometry -------------------------------------------------------------------

    def _on_joint_states(self, msg: JointState) -> None:
        index = {str(n): i for i, n in enumerate(msg.name)}
        if any(j not in index for j in self._wheel_joints):
            return
        positions = [float(msg.position[index[j]]) for j in self._wheel_joints]
        if len(msg.velocity) == len(msg.name):
            self._last_wheel_velocities = [float(msg.velocity[index[j]]) for j in self._wheel_joints]
        if self._last_wheel_positions is None:
            self._last_wheel_positions = positions
            return
        deltas = [p - q for p, q in zip(positions, self._last_wheel_positions, strict=True)]
        self._last_wheel_positions = positions
        body = lk.wheel_deltas_to_body(deltas, self._geometry)
        if body is None:
            return
        lk.integrate_pose(self._pose, *body)
        self._publish_odom(msg.header.stamp)

    def _publish_odom(self, stamp) -> None:
        import math

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._pose.x
        odom.pose.pose.position.y = self._pose.y
        odom.pose.pose.orientation.z = math.sin(self._pose.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self._pose.theta / 2.0)
        body_vel = lk.wheel_deltas_to_body(self._last_wheel_velocities, self._geometry)
        if body_vel is not None:
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = body_vel
        self._odom_pub.publish(odom)
        if self._tf is not None:
            tf = TransformStamped()
            tf.header = odom.header
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = self._pose.x
            tf.transform.translation.y = self._pose.y
            tf.transform.rotation = odom.pose.pose.orientation
            self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = LeKiwiBaseNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
