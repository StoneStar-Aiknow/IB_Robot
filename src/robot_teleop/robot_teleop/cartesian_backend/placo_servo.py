"""SO101 Placo Servo client backend.

A thin device-side adapter for the ``placo_servo`` solver mode. From the
caller's perspective this behaves *exactly* like the sibling backends::

    backend.servo(linear=(vx, vy, vz), angular=(wx, wy, wz))

so the upstream device code in :mod:`robot_teleop.devices.xbox_controller`
and :mod:`robot_teleop.phone.phone_device` stays solver-agnostic.

Internally the backend publishes the twist to two **private** topics consumed
by ``so101_placo_servo_node``:

* ``linear_cmd_base``  — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = base_link``, carries ``linear`` (already base-frame per the
  device convention).
* ``angular_cmd_base`` — :class:`geometry_msgs.msg.Vector3Stamped`,
  ``frame_id = base_link``, carries ``angular`` **converted tool→base** via
  :class:`ToolAngularAdapter`.

``placo_servo`` controls a true Cartesian orientation: the QP orientation task
expects a base-frame angular velocity, so the raw tool-frame stick angular must
be rotated into the base frame before publishing.
"""

from __future__ import annotations

from geometry_msgs.msg import Vector3Stamped
from rclpy.task import Future
from std_srvs.srv import Trigger

from .base import CartesianBackend
from .frame_adapter import ToolAngularAdapter

Vec3 = tuple[float, float, float]


_DEFAULT_LINEAR_TOPIC = "/so101_placo_servo_node/linear_cmd_base"
_DEFAULT_ANGULAR_TOPIC = "/so101_placo_servo_node/angular_cmd_base"
_DEFAULT_START_SRV = "/so101_placo_servo_node/start"
_DEFAULT_STOP_SRV = "/so101_placo_servo_node/stop"

# Physical max speeds that map to a user scale value of 1.0.
_MAX_LINEAR_SPEED_MPS: float = 1.0  # 1.0 m/s at scale=1.0
_MAX_ANGULAR_SPEED_RPS: float = 6.0  # 6.0 rad/s at scale=1.0


class PlacoServoBackend(CartesianBackend):
    """Publish base-frame linear/angular commands to ``so101_placo_servo_node``."""

    def __init__(
        self,
        node,
        tf_buffer,
        base_link: str,
        tool_frame: str,
        linear_speed: float = 0.3,
        angular_speed: float = 0.67,
        linear_topic: str = _DEFAULT_LINEAR_TOPIC,
        angular_topic: str = _DEFAULT_ANGULAR_TOPIC,
        start_srv: str = _DEFAULT_START_SRV,
        stop_srv: str = _DEFAULT_STOP_SRV,
        stale_threshold_s: float = 0.2,
        **_unused,  # tolerate extra kwargs for signature parity with siblings
    ) -> None:
        self._node = node
        self._base = base_link
        # tool→base converter for the angular channel (true Cartesian posture).
        self._adapter = ToolAngularAdapter(
            node=node,
            tf_buffer=tf_buffer,
            base_link=base_link,
            tool_frame=tool_frame,
            stale_threshold_s=stale_threshold_s,
        )
        # linear_speed / angular_speed are 0.0~1.0 fractions of the physical
        # maximums; convert to actual m/s, rad/s before publishing.
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
                "so101_placo_servo_node start service not ready; will retry every 0.5 s",
            )
            self._schedule_start_retry()
            return self._requested_enabled
        self._send_start_request()
        return self._requested_enabled

    def _schedule_start_retry(self) -> None:
        if self._start_retry_timer is not None:
            return
        self._start_retry_timer = self._node.create_timer(0.5, self._retry_start)

    def _retry_start(self) -> None:
        if not self._requested_enabled:
            self._cancel_start_retry()
            return
        if self._start_cli.service_is_ready():
            self._node.get_logger().info("so101_placo_servo_node start service now ready; requesting start")
            self._send_start_request()

    def _send_start_request(self) -> None:
        future = self._start_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_start_response)

    def _on_start_response(self, future: Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._active_enabled = False
            self._node.get_logger().error(f"so101_placo_servo_node start request failed: {exc}")
            if self._requested_enabled:
                self._schedule_start_retry()
            return

        if response.success:
            self._active_enabled = True
            self._cancel_start_retry()
            self._node.get_logger().info(response.message or "so101_placo_servo_node enabled")
            return

        self._active_enabled = False
        self._node.get_logger().error(response.message or "so101_placo_servo_node rejected start request")
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
        # Convert tool-frame angular → base frame (true Cartesian posture).
        v_base, w_base = self._adapter.convert(linear, angular)
        stamp = self._node.get_clock().now().to_msg()

        lin_msg = Vector3Stamped()
        lin_msg.header.stamp = stamp
        lin_msg.header.frame_id = self._base
        lin_msg.vector.x = float(v_base[0]) * self._linear_speed
        lin_msg.vector.y = float(v_base[1]) * self._linear_speed
        lin_msg.vector.z = float(v_base[2]) * self._linear_speed
        self._linear_pub.publish(lin_msg)

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = stamp
        ang_msg.header.frame_id = self._base  # base-frame angular (converted)
        ang_msg.vector.x = float(w_base[0]) * self._angular_speed
        ang_msg.vector.y = float(w_base[1]) * self._angular_speed
        ang_msg.vector.z = float(w_base[2]) * self._angular_speed
        self._angular_pub.publish(ang_msg)

    @property
    def is_enabled(self) -> bool:
        return self._active_enabled
