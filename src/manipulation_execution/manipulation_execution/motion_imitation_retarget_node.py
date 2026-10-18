"""Live retargeting for the X2 mirror: PEAR frames in, a rate-capped arm preview out.

This replaces ``motion_imitation_mock_node`` and keeps its contract: same node
name, same ``~/play`` service, same preview and status topics.

    PEAR frame stream  ->  RetargetStream  ->  limit clamp + rate cap  ->  preview topic
                           (worker thread)     (x2.safety.RateLimiter)     (Float64MultiArray)

**This node never commands the robot.** It has no publisher on the runtime's
arm or hand channels; it publishes only the preview, and ``mirror_session_node``
is the only outlet that takes it through the session's output gate.

**It feeds the retargeter only person detections.** Frames whose ``source`` is
not ``yolox`` (``center_fallback``: nobody in the picture) are counted and
dropped, and do not count as a sign of life.

**It publishes nothing when the input is stale or when paused.** If no
``yolox`` frame arrived within ``input_staleness_sec`` the tick is held (the
session gate keeps its last command); after such a gap the retargeter's
per-person state is reset before the next frame. ``~/play``
(``std_srvs/SetBool``) true starts a round from the neutral pose
(``reset_pose_json``); false pauses. To watch it alone:
``ros2 service call /motion_imitation_mock/play std_srvs/srv/SetBool "{data: true}"``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import PearFrame

from .imitation_retargeting.live_player import LivePlayer
from .imitation_retargeting.streaming import create_stream
from .imitation_retargeting.x2.safety import effective_limits_rad

ARM_JOINT_COUNT = 14
LIMIT_MARGIN_DEG = 1.0


def _validated_positive(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _validated_nonnegative(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and not negative")
    return number


def _validated_topic(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    if not text.startswith("/"):
        raise ValueError(f"{name} must be an absolute ROS name")
    return text


class MotionImitationRetargetNode(Node):
    """Retarget live PEAR frames through the rate cap onto the preview topic."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("motion_imitation_mock", parameter_overrides=parameter_overrides)
        self.declare_parameter("pear_frame_topic", "/motion_imitation_stream/pear_frame")
        self.declare_parameter("preview_topic", "/motion_imitation_mock/arm_command_preview")
        self.declare_parameter("status_topic", "/motion_imitation_mock/status")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("input_staleness_sec", 0.5)
        self.declare_parameter("status_period_sec", 5.0)
        self.declare_parameter("joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("velocity_limits_json", "{}")
        self.declare_parameter("reset_pose_json", "[]")
        self.declare_parameter("retargeter", "x2_followme")
        self.declare_parameter("vel_scale", 1.0)
        self.declare_parameter("latency_s", 0.4)
        self.declare_parameter("lookahead_s", 0.25)
        self.declare_parameter("mirror", True)

        self._pear_topic = _validated_topic(self.get_parameter("pear_frame_topic").value, "pear_frame_topic")
        preview_topic = _validated_topic(self.get_parameter("preview_topic").value, "preview_topic")
        status_topic = _validated_topic(self.get_parameter("status_topic").value, "status_topic")
        rate_hz = _validated_positive(self.get_parameter("publish_rate_hz").value, "publish_rate_hz")
        staleness_sec = _validated_positive(self.get_parameter("input_staleness_sec").value, "input_staleness_sec")
        status_period = _validated_positive(self.get_parameter("status_period_sec").value, "status_period_sec")
        vel_scale = _validated_positive(self.get_parameter("vel_scale").value, "vel_scale")
        latency_s = _validated_nonnegative(self.get_parameter("latency_s").value, "latency_s")
        lookahead_s = _validated_nonnegative(self.get_parameter("lookahead_s").value, "lookahead_s")
        if type(self.get_parameter("mirror").value) is not bool:
            raise ValueError("mirror must be a boolean")
        mirror = self.get_parameter("mirror").value
        retargeter = str(self.get_parameter("retargeter").value).strip()

        self._joint_names = tuple(json.loads(str(self.get_parameter("joint_names_json").value)))
        if len(self._joint_names) != ARM_JOINT_COUNT:
            raise ValueError(f"joint_names_json must name exactly {ARM_JOINT_COUNT} joints")
        lower, upper = self._limit_arrays(json.loads(str(self.get_parameter("joint_limits_json").value)))
        rated = self._velocity_array(json.loads(str(self.get_parameter("velocity_limits_json").value)))
        reset_pose = self._reset_pose(json.loads(str(self.get_parameter("reset_pose_json").value)))

        stream = create_stream(retargeter, base_pose=reset_pose, latency_s=latency_s, lookahead_s=lookahead_s,
                               mirror=mirror)
        if self._joint_names != tuple(stream.joint_names):
            raise ValueError(
                f"joint_names_json {list(self._joint_names)} does not match the {retargeter} stream's joint "
                f"order {list(stream.joint_names)}"
            )
        low_rad, high_rad = effective_limits_rad(lower, upper, LIMIT_MARGIN_DEG)
        self._player = LivePlayer(
            stream, low_rad=low_rad, high_rad=high_rad, rated_rad_s=rated, vel_scale=vel_scale, rate_hz=rate_hz,
            staleness_s=staleness_sec, base_pose=reset_pose,
        )

        self._preview_publisher = self.create_publisher(Float64MultiArray, preview_topic, 10)
        self._status_publisher = self.create_publisher(String, status_topic, 10)
        self.create_subscription(PearFrame, self._pear_topic, self._on_pear_frame, qos_profile_sensor_data)
        self.create_service(SetBool, "~/play", self._on_play)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.create_timer(status_period, self._publish_status)

        self._player.start()
        self.get_logger().info(
            f"motion imitation retarget ready: retargeter={retargeter}, {rate_hz:g} Hz, vel_scale={vel_scale:g}, "
            f"mirror={mirror}; paused until ~/play; computed commands go to {preview_topic} only"
        )

    def _limit_arrays(self, limits: object) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(limits, dict) or set(limits) != set(self._joint_names):
            raise ValueError("joint_limits_json must cover exactly the named joints")
        lower, upper = [], []
        for name in self._joint_names:
            band = limits[name]
            if not isinstance(band, list | tuple) or len(band) != 2:
                raise ValueError(f"joint limit for {name} must be [lower, upper]")
            low, high = float(band[0]), float(band[1])
            if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
                raise ValueError(f"joint limit for {name} must be finite and ordered")
            lower.append(low)
            upper.append(high)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def _velocity_array(self, limits: object) -> np.ndarray:
        if not isinstance(limits, dict) or set(limits) != set(self._joint_names):
            raise ValueError("velocity_limits_json must cover exactly the named joints")
        rates = []
        for name in self._joint_names:
            rate = float(limits[name])
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"velocity limit for {name} must be finite and positive")
            rates.append(rate)
        return np.asarray(rates, dtype=np.float64)

    @staticmethod
    def _reset_pose(values: object) -> np.ndarray:
        if not isinstance(values, list) or len(values) != ARM_JOINT_COUNT:
            raise ValueError(f"reset_pose_json must list exactly {ARM_JOINT_COUNT} joint angles")
        pose = np.asarray([float(v) for v in values], dtype=np.float64)
        if not np.all(np.isfinite(pose)):
            raise ValueError("reset_pose_json must be finite")
        return pose

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_play(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """True: start a round from the neutral pose. False: pause; nothing is published."""
        self._player.play(bool(request.data))
        response.success = True
        response.message = "playing from the neutral pose" if request.data else "paused"
        self.get_logger().info(f"motion imitation retarget {response.message}")
        return response

    def _on_pear_frame(self, message: PearFrame) -> None:
        stamp = message.header.stamp
        self._player.on_frame(
            self._now(), stamp.sec + stamp.nanosec * 1e-9, message.sequence, message.source,
            np.asarray(message.smplx_pose_raw, dtype=np.float64),
        )

    def _tick(self) -> None:
        command = self._player.tick(self._now())
        if command is None:
            # Publishing nothing is the hold: the session gate keeps sending its last command.
            return
        message = Float64MultiArray()
        message.data = command
        self._preview_publisher.publish(message)

    def _publish_status(self) -> None:
        status = self._player.status(self._now())
        status["input_topic"] = self._pear_topic
        message = String()
        message.data = json.dumps(status, ensure_ascii=False)
        self._status_publisher.publish(message)
        if status["input_frames"] == 0:
            self.get_logger().info(
                f"motion imitation retarget idle: no frames on {self._pear_topic}; nothing is being computed"
            )
        else:
            self.get_logger().info(f"motion imitation retarget status {message.data}")

    def destroy_node(self) -> None:
        self._player.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MotionImitationRetargetNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
