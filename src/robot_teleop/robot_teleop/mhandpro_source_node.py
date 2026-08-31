"""ROS 2 source node for normalized hand state and optional raw mHandPro frames."""

from __future__ import annotations

import statistics
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import HumanHandState, MHandProFrame

from .devices.mhandpro_sdk import CS_SUCCEEDED
from .devices.mhandpro_source import ReplayGloveSource, SharedRealMHandProSource
from .hand_state import HUMAN_HAND_SCHEMA, extract_human_hand_geometry

FAILURE_POLICIES = {"require_all", "allow_available"}


class MHandProSourceNode(Node):
    """Keep vendor acquisition independent from target-hand retargeting."""

    def __init__(self):
        super().__init__("mhandpro_source")
        self.declare_parameter("source_name", "mhandpro")
        self.declare_parameter("lib_path", "")
        self.declare_parameter("sides", ["right"])
        self.declare_parameter("mock", False)
        self.declare_parameter("require_p_pose", True)
        self.declare_parameter("calibrate_p_pose_on_startup", False)
        self.declare_parameter("publish_frequency", 50.0)
        self.declare_parameter("publish_raw_frame", False)
        self.declare_parameter("stale_timeout", 0.2)
        self.declare_parameter("startup_timeout", 30.0)
        self.declare_parameter("calibration_timeout", 30.0)
        self.declare_parameter("p_pose_quality_frames", 5)
        self.declare_parameter("p_pose_max_openness", 0.7)
        self.declare_parameter("calibration_service", "/hand_sources/mhandpro/calibrate_p_pose")
        self.declare_parameter("topic_prefix", "/hand_sources/mhandpro")
        self.declare_parameter("replay_rate_hz", 50.0)
        self.declare_parameter("replay_segment_seconds", 0.7)
        self.declare_parameter("failure_policy", "require_all")
        self.declare_parameter("auto_reconnect", True)
        self.declare_parameter("reconnect_initial_delay", 1.0)
        self.declare_parameter("reconnect_max_delay", 10.0)
        self.declare_parameter("reconnect_max_attempts", 0)

        self.source_name = str(self.get_parameter("source_name").value).strip()
        self.sides = tuple(dict.fromkeys(str(side) for side in self.get_parameter("sides").value))
        self.mock = bool(self.get_parameter("mock").value)
        self.require_p_pose = bool(self.get_parameter("require_p_pose").value) and not self.mock
        self.calibrate_p_pose_on_startup = (
            bool(self.get_parameter("calibrate_p_pose_on_startup").value) and self.require_p_pose
        )
        self.stale_timeout = float(self.get_parameter("stale_timeout").value)
        self.calibration_timeout = float(self.get_parameter("calibration_timeout").value)
        self.p_pose_quality_frames = int(self.get_parameter("p_pose_quality_frames").value)
        self.p_pose_max_openness = float(self.get_parameter("p_pose_max_openness").value)
        self.publish_raw_frame = bool(self.get_parameter("publish_raw_frame").value)
        self.failure_policy = str(self.get_parameter("failure_policy").value).strip().lower()
        self.auto_reconnect = bool(self.get_parameter("auto_reconnect").value) and not self.mock
        self.reconnect_initial_delay = float(self.get_parameter("reconnect_initial_delay").value)
        self.reconnect_max_delay = float(self.get_parameter("reconnect_max_delay").value)
        self.reconnect_max_attempts = int(self.get_parameter("reconnect_max_attempts").value)
        publish_frequency = float(self.get_parameter("publish_frequency").value)
        topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        if not self.source_name or not self.sides or not set(self.sides).issubset({"left", "right"}):
            raise ValueError("mHandPro source_name and sides must be valid")
        if min(self.stale_timeout, self.calibration_timeout, publish_frequency) <= 0.0:
            raise ValueError("mHandPro timing parameters must be positive")
        if self.p_pose_quality_frames <= 0:
            raise ValueError("mHandPro p_pose_quality_frames must be positive")
        if self.p_pose_max_openness <= 0.0:
            raise ValueError("mHandPro p_pose_max_openness must be positive")
        if self.failure_policy not in FAILURE_POLICIES:
            raise ValueError(f"mHandPro failure_policy must be one of {sorted(FAILURE_POLICIES)}")
        if self.reconnect_initial_delay <= 0.0 or self.reconnect_max_delay < self.reconnect_initial_delay:
            raise ValueError("mHandPro reconnect delays must be positive and ordered")
        if self.reconnect_max_attempts < 0:
            raise ValueError("mHandPro reconnect_max_attempts must be non-negative")

        self._sources = {}
        startup_error = None
        if self.mock:
            for side in self.sides:
                self._sources[side] = ReplayGloveSource(
                    side,
                    rate_hz=float(self.get_parameter("replay_rate_hz").value),
                    segment_seconds=float(self.get_parameter("replay_segment_seconds").value),
                )
                self._sources[side].connect()
            self._shared_source = None
        else:
            lib_path = str(self.get_parameter("lib_path").value).strip()
            if not lib_path:
                raise ValueError("mHandPro lib_path is required when mock is false")
            self._shared_source = SharedRealMHandProSource(
                lib_path,
                self.sides,
                startup_timeout=float(self.get_parameter("startup_timeout").value),
                failure_policy=self.failure_policy,
            )
            try:
                self._shared_source.connect()
            except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
                if not self.auto_reconnect or self.calibrate_p_pose_on_startup:
                    raise
                startup_error = str(exc)

        self._ready = not self.require_p_pose
        self._ready_sides = set(self.sides) if self._ready else set()
        self._last_sequences = {side: -1 for side in self.sides}
        self._reconnect_attempts = 0
        self._next_reconnect_at = time.monotonic() + self.reconnect_initial_delay
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()
        self._reconnect_thread = None
        self._shutdown = threading.Event()
        self._last_reconnect_log = 0.0
        if startup_error is None and self.calibrate_p_pose_on_startup:
            quality = self._run_p_pose_calibration()
            self.get_logger().info(f"Startup P-pose succeeded; hand source is ready ({quality})")
        self._raw_publishers = (
            {
                side: self.create_publisher(MHandProFrame, f"{topic_prefix}/{side}/frame", qos_profile_sensor_data)
                for side in self.sides
            }
            if self.publish_raw_frame
            else {}
        )
        self._state_publishers = {
            side: self.create_publisher(HumanHandState, f"{topic_prefix}/{side}/state", qos_profile_sensor_data)
            for side in self.sides
        }
        self._health_publishers = {
            side: self.create_publisher(String, f"{topic_prefix}/{side}/health", 10) for side in self.sides
        }
        service_name = str(self.get_parameter("calibration_service").value).strip()
        self._calibration_service = self.create_service(Trigger, service_name, self._calibrate_p_pose)
        self._timer = self.create_timer(1.0 / publish_frequency, self._publish_latest)
        if startup_error is not None:
            self.get_logger().warning(
                f"mHandPro source {self.source_name!r} started disconnected for {self.sides}; "
                f"automatic reconnect is active: {startup_error}"
            )
        else:
            state = "ready" if self._ready else "waiting for runtime P-pose"
            self.get_logger().info(f"mHandPro source {self.source_name!r} connected for {self.sides}: {state}")

    def _source_for(self, side):
        return self._sources[side] if self.mock else self._shared_source

    def _calibrate_p_pose(self, _request, response):
        try:
            quality = self._run_p_pose_calibration()
            response.success = True
            response.message = f"mHandPro P-pose succeeded; hand source is ready ({quality})"
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
            self._ready = False
            self._ready_sides.clear()
            response.success = False
            response.message = str(exc)
        return response

    def _run_p_pose_calibration(self) -> str:
        """Run SDK alignment and unlock only after post-calibration quality checks."""
        source = self._source_for(self.sides[0])
        self._ready = False
        self._ready_sides.clear()
        state, progress = source.calibrate_p_pose(self.calibration_timeout)
        if state != CS_SUCCEEDED:
            raise RuntimeError(f"mHandPro P-pose failed (state={state}, progress={progress:.1f})")
        ready_sides = {side for side in self.sides if source.is_side_connected(side)}
        if self.failure_policy == "require_all" and ready_sides != set(self.sides):
            raise RuntimeError("All configured mHandPro gloves must remain connected during P-pose validation")
        if not ready_sides:
            raise RuntimeError("No configured mHandPro glove is connected for P-pose validation")
        quality = self._validate_post_calibration_frames(ready_sides)
        self._ready_sides = ready_sides
        self._ready = bool(ready_sides)
        return quality

    def _validate_post_calibration_frames(self, sides) -> str:
        """Reject an SDK success result unless fresh, extended complete skeletons follow it."""
        frames_by_side = {side: [] for side in sides}
        seen = {
            side: (frame.sequence if (frame := self._source_for(side).latest_frame(side)) is not None else -1)
            for side in sides
        }
        deadline = time.monotonic() + min(3.0, self.calibration_timeout)
        while time.monotonic() < deadline:
            for side in sides:
                source = self._source_for(side)
                frame = source.latest_frame(side)
                if frame is None or frame.sequence == seen[side]:
                    continue
                seen[side] = frame.sequence
                if not source.is_side_connected(side) or time.monotonic() - frame.timestamp > self.stale_timeout:
                    continue
                try:
                    geometry = extract_human_hand_geometry(frame.positions, frame.virtual_positions, side)
                except (TypeError, ValueError, ArithmeticError):
                    continue
                frames_by_side[side].append(geometry.openness_score)
            if all(len(values) >= self.p_pose_quality_frames for values in frames_by_side.values()):
                break
            time.sleep(0.005)
        results = []
        for side, values in frames_by_side.items():
            if len(values) < self.p_pose_quality_frames:
                raise RuntimeError(
                    f"P-pose quality check received only {len(values)} fresh {side} frames; "
                    f"expected {self.p_pose_quality_frames}"
                )
            median_openness = statistics.median(values)
            if median_openness > self.p_pose_max_openness:
                raise RuntimeError(
                    f"P-pose quality check rejected the {side} hand: flexion score {median_openness:.3f} "
                    f"exceeds {self.p_pose_max_openness:.3f}"
                )
            results.append(f"{side} flexion={median_openness:.3f}")
        return ", ".join(results)

    def _publish_latest(self):
        self._maybe_reconnect()
        shared_source = self._source_for(self.sides[0])
        all_connected = all(shared_source.is_side_connected(side) for side in self.sides)
        if self.failure_policy == "require_all" and not all_connected:
            self._ready = False
            self._ready_sides.clear()

        for side in self.sides:
            source = self._source_for(side)
            frame = source.latest_frame(side)
            if frame is None:
                connected = source.is_side_connected(side)
                self._publish_health(side, self._status(connected, float("inf")))
                continue
            is_new_frame = frame.sequence != self._last_sequences[side]
            if is_new_frame:
                self._last_sequences[side] = frame.sequence
            age = time.monotonic() - frame.timestamp
            connected = source.is_side_connected(side)
            if not connected:
                self._ready_sides.discard(side)
            valid = (
                connected
                and side in self._ready_sides
                and (self.failure_policy == "allow_available" or all_connected)
                and age <= self.stale_timeout
            )
            status = "ready" if valid else self._status(connected, age)
            self._publish_health(side, status)
            if is_new_frame and self.publish_raw_frame:
                self._raw_publishers[side].publish(self._raw_message(frame, valid, status))
            self._state_publishers[side].publish(self._state_message(frame, valid, status))

    def _publish_health(self, side: str, status: str) -> None:
        message = String()
        message.data = status
        self._health_publishers[side].publish(message)

    def _maybe_reconnect(self):
        if not self.auto_reconnect or self._shutdown.is_set():
            return
        source = self._source_for(self.sides[0])
        connected_sides = {side for side in self.sides if source.is_side_connected(side)}
        connection_is_usable = (
            connected_sides == set(self.sides) if self.failure_policy == "require_all" else bool(connected_sides)
        )
        if connection_is_usable:
            self._reconnect_attempts = 0
            self._next_reconnect_at = time.monotonic() + self.reconnect_initial_delay
            return
        now = time.monotonic()
        if now < self._next_reconnect_at:
            return
        if self.reconnect_max_attempts and self._reconnect_attempts >= self.reconnect_max_attempts:
            if now - self._last_reconnect_log >= max(10.0, self.reconnect_max_delay):
                self._last_reconnect_log = now
                self.get_logger().error("mHandPro reconnect attempts exhausted; restart the source node")
            return

        with self._reconnect_lock:
            if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
                return
            self._reconnecting = True
            self._ready = False
            self._ready_sides.clear()
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_once,
                args=(source,),
                name=f"mhandpro-reconnect-{self.source_name}",
                daemon=True,
            )
            self._reconnect_thread.start()

    def _reconnect_once(self, source) -> None:
        """Reconnect outside the ROS executor so health and stale-state publishing continue."""
        try:
            source.disconnect()
            if self._shutdown.is_set():
                return
            source.connect()
            self._reconnect_attempts = 0
            self._next_reconnect_at = time.monotonic() + self.reconnect_initial_delay
            if not self.require_p_pose:
                ready_sides = {side for side in self.sides if source.is_side_connected(side)}
                self._ready_sides = ready_sides
                self._ready = bool(ready_sides)
            if self._shutdown.is_set():
                return
            if self.require_p_pose:
                message = "mHandPro source reconnected; P-pose is required before output unlocks"
            else:
                message = "mHandPro source reconnected; available sides are ready"
            self.get_logger().warning(message)
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
            if self._shutdown.is_set():
                return
            self._reconnect_attempts += 1
            delay = min(
                self.reconnect_max_delay,
                self.reconnect_initial_delay * (2 ** min(self._reconnect_attempts - 1, 10)),
            )
            self._next_reconnect_at = time.monotonic() + delay
            self._last_reconnect_log = time.monotonic()
            self.get_logger().warning(
                f"mHandPro reconnect attempt {self._reconnect_attempts} failed; retry in {delay:.1f}s: {exc}"
            )
        finally:
            self._reconnecting = False

    def _status(self, connected: bool, age: float) -> str:
        reconnect_available = not self.reconnect_max_attempts or self._reconnect_attempts < self.reconnect_max_attempts
        if not connected and self.failure_policy == "allow_available":
            source = self._source_for(self.sides[0])
            if any(source.is_side_connected(side) for side in self.sides):
                return "disconnected"
        if not connected and self.auto_reconnect and reconnect_available:
            return "reconnecting"
        if not connected:
            return "disconnected"
        if not self._ready:
            return "waiting_p_pose"
        if age > self.stale_timeout:
            return "stale"
        return "invalid"

    def _raw_message(self, frame, valid: bool, status: str) -> MHandProFrame:
        message = MHandProFrame()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"{self.source_name}_{frame.side}_hand"
        message.source = self.source_name
        message.schema = "mhandpro_full_v1"
        message.side = frame.side
        message.sequence = frame.sequence
        message.sdk_frame_index = frame.sdk_frame_index
        message.valid = valid
        message.status = status
        message.device_power = frame.device_power
        message.source_frequency_hz = max(0, frame.frequency)
        message.positions = [_point(value) for value in frame.positions]
        message.orientations = [_quaternion_wxyz(value) for value in frame.quaternions or []]
        message.virtual_positions = [_point(value) for value in frame.virtual_positions or []]
        message.angular_velocity = [_vector(value) for value in frame.gyroscope or []]
        message.linear_acceleration = [_vector(value) for value in frame.accelerations or []]
        message.linear_velocity = [_vector(value) for value in frame.velocities or []]
        message.sensor_states = list(frame.sensor_states or [])
        return message

    def _state_message(self, frame, valid: bool, status: str) -> HumanHandState:
        message = HumanHandState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"{self.source_name}_{frame.side}_hand"
        message.source = self.source_name
        message.schema = HUMAN_HAND_SCHEMA
        message.side = frame.side
        message.sequence = frame.sequence
        message.status = status
        message.landmarks = [_point(value) for value in frame.positions]
        message.orientations = [_quaternion_wxyz(value) for value in frame.quaternions or []]
        message.virtual_tips = [_point(value) for value in frame.virtual_positions or []]
        if valid:
            try:
                geometry = extract_human_hand_geometry(frame.positions, frame.virtual_positions, frame.side)
                message.feature_names = list(geometry.feature_names)
                message.features = list(geometry.features)
                message.confidence = [1.0] * len(geometry.features)
                message.openness_score = geometry.openness_score
                message.valid = True
                return message
            except (TypeError, ValueError, ArithmeticError) as exc:
                message.status = f"invalid_geometry:{exc}"
        message.valid = False
        return message

    def destroy_node(self):
        self._shutdown.set()
        if self._shared_source is not None:
            self._shared_source.disconnect()
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=1.0)
        for source in self._sources.values():
            source.disconnect()
        return super().destroy_node()


def _point(values) -> Point:
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _vector(values) -> Vector3:
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _quaternion_wxyz(values) -> Quaternion:
    return Quaternion(x=float(values[1]), y=float(values[2]), z=float(values[3]), w=float(values[0]))


def main(args=None):
    rclpy.init(args=args)
    node = MHandProSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
