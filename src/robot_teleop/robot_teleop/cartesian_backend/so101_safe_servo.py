"""SO101 Safe Servo client backend.

A thin device-side adapter for the ``safe_servo`` solver mode. From the
caller's perspective, this object behaves *exactly* like
:class:`VelocityServoBackend`::

    backend.servo(linear=(vx, vy, vz), angular=(wx, wy, wz))

Internally the backend splits the unified twist into two **private** topics
so the upstream device code in :mod:`robot_teleop.devices.xbox_controller` and
:mod:`robot_teleop.phone.phone_device` does not know that the Safe Servo node
expects separate base-linear and tool-angular streams:

* ``linear_cmd_base``  — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = base_link``, carries ``linear`` (already base-frame per
  convention).
* ``angular_cmd_tool`` — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = tool_frame`` (typically ``gripper``), carries ``angular``
  *unchanged* (no :class:`ToolAngularAdapter` conversion!). The node
  integrates this directly into joint 4 (pitch) and joint 5 (roll).

The encapsulation rule is *mandatory*: the angular vector reaches the
backend in tool-frame semantic form (xBox writes raw stick → tool; Phone
adapts upstream). Any base-frame
conversion would silently reinterpret the operator's pitch/roll intent.
"""

from __future__ import annotations

from geometry_msgs.msg import Vector3Stamped
from rclpy.task import Future
from std_srvs.srv import Trigger

from .base import CartesianBackend

Vec3 = tuple[float, float, float]


_DEFAULT_LINEAR_TOPIC = "/so101_safe_servo_node/linear_cmd_base"
_DEFAULT_ANGULAR_TOPIC = "/so101_safe_servo_node/angular_cmd_tool"
_DEFAULT_START_SRV = "/so101_safe_servo_node/start"
_DEFAULT_STOP_SRV = "/so101_safe_servo_node/stop"

# Physical max speeds that map to a user scale value of 1.0.
# Must stay in sync with max_wrist_angular_speed in so101_safe_servo.yaml.
_MAX_LINEAR_SPEED_MPS: float = 1.0  # 1.0 m/s at scale=1.0
_MAX_ANGULAR_SPEED_RPS: float = 6.0  # 6.0 rad/s at scale=1.0 (≈ ST3215 no-load ceiling)


class SO101SafeServoBackend(CartesianBackend):
    """Publish split linear/angular commands to ``so101_safe_servo_node``."""

    def __init__(
        self,
        node,
        tf_buffer,  # noqa: ARG002 — accepted for signature parity with siblings
        base_link: str,
        tool_frame: str,
        linear_speed: float = 0.3,
        angular_speed: float = 0.67,
        linear_topic: str = _DEFAULT_LINEAR_TOPIC,
        angular_topic: str = _DEFAULT_ANGULAR_TOPIC,
        start_srv: str = _DEFAULT_START_SRV,
        stop_srv: str = _DEFAULT_STOP_SRV,
        **_unused,  # tolerate extra kwargs (stale_threshold_s, etc.)
    ) -> None:
        self._node = node
        self._base = base_link
        self._tool = tool_frame
        # linear_speed / angular_speed are 0.0~1.0 fractions of the physical
        # maximums defined by _MAX_LINEAR_SPEED_MPS / _MAX_ANGULAR_SPEED_RPS.
        # Users configure a unitless 0~1 value in robot config; this class
        # converts it to actual m/s or rad/s before publishing.
        self._linear_speed = float(linear_speed) * _MAX_LINEAR_SPEED_MPS
        self._angular_speed = float(angular_speed) * _MAX_ANGULAR_SPEED_RPS

        self._linear_pub = node.create_publisher(Vector3Stamped, linear_topic, 10)
        self._angular_pub = node.create_publisher(Vector3Stamped, angular_topic, 10)
        self._start_cli = node.create_client(Trigger, start_srv)
        self._stop_cli = node.create_client(Trigger, stop_srv)
        self._requested_enabled = False
        self._active_enabled = False
        self._start_retry_timer = None

    # ------------------------------------------------------------------ enable
    def enable(self) -> bool:
        self._requested_enabled = True
        if not self._start_cli.service_is_ready():
            self._node.get_logger().warn(
                "so101_safe_servo_node start service not ready; will retry every 0.5 s",
            )
            self._schedule_start_retry()
            return False
        self._send_start_request()
        return self._active_enabled

    def _schedule_start_retry(self) -> None:
        """Create a 0.5-s repeating timer that calls /start once the service is up."""
        if self._start_retry_timer is not None:
            return  # already scheduled
        self._start_retry_timer = self._node.create_timer(0.5, self._retry_start)

    def _retry_start(self) -> None:
        if not self._requested_enabled:
            self._cancel_start_retry()
            return
        if self._start_cli.service_is_ready():
            self._node.get_logger().info("so101_safe_servo_node start service now ready; requesting start")
            self._send_start_request()

    def _send_start_request(self) -> None:
        future = self._start_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_start_response)

    def _on_start_response(self, future: Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._active_enabled = False
            self._node.get_logger().error(f"so101_safe_servo_node start request failed: {exc}")
            if self._requested_enabled:
                self._schedule_start_retry()
            return

        if response.success:
            self._active_enabled = True
            self._cancel_start_retry()
            self._node.get_logger().info(response.message or "so101_safe_servo_node enabled")
            return

        self._active_enabled = False
        self._node.get_logger().error(response.message or "so101_safe_servo_node rejected start request")
        if self._requested_enabled:
            self._schedule_start_retry()

    def _cancel_start_retry(self) -> None:
        if self._start_retry_timer is None:
            return
        self._start_retry_timer.cancel()
        self._start_retry_timer.destroy()
        self._start_retry_timer = None

    def disable(self) -> bool:
        self._requested_enabled = False
        self._cancel_start_retry()
        if self._stop_cli.service_is_ready():
            self._stop_cli.call_async(Trigger.Request())
        self._active_enabled = False
        return True

    # ------------------------------------------------------------------ servo
    def servo(self, linear: Vec3, angular: Vec3) -> None:
        if not self._active_enabled:
            return
        stamp = self._node.get_clock().now().to_msg()

        lin_msg = Vector3Stamped()
        lin_msg.header.stamp = stamp
        lin_msg.header.frame_id = self._base
        lin_msg.vector.x = float(linear[0]) * self._linear_speed
        lin_msg.vector.y = float(linear[1]) * self._linear_speed
        lin_msg.vector.z = float(linear[2]) * self._linear_speed
        self._linear_pub.publish(lin_msg)

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = stamp
        ang_msg.header.frame_id = self._tool  # semantic tool-frame, no TF conversion
        ang_msg.vector.x = float(angular[0]) * self._angular_speed
        ang_msg.vector.y = float(angular[1]) * self._angular_speed
        ang_msg.vector.z = float(angular[2]) * self._angular_speed
        self._angular_pub.publish(ang_msg)

    @property
    def is_enabled(self) -> bool:
        return self._active_enabled
