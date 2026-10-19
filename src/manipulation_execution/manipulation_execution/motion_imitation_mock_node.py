"""Mock retargeting and mock safety for the X2 mirror, ahead of the session gate.

This closes the motion-imitation data flow without a retargeter. Its stages
are separate so that each can be replaced independently:

    PEAR frame stream  ->  mock retarget  ->  mock safety  ->  preview topic
                           (plays a recorded    (clamps and      (Float64MultiArray)
                            animation)          rate-limits)

The two replaceable stages are the ``_retarget`` and ``_apply_safety`` methods. A
retargeter replaces the first and keeps the output contract (14 joint angles in
radians, in ``joint_names_json`` order); a safety policy replaces the second.

**This node never commands the robot.** It has no publisher on the runtime's
arm or hand channels. ``mirror_session_node`` is the only outlet: while it
mirrors, its mock window pipeline (``mirror_session.mock_playback``) takes each
preview through the session's output gate.

**It plays only when asked.** ``~/play`` (``std_srvs/SetBool``) true restarts
the clip at frame 0 and plays; false pauses, and a paused node publishes
nothing. The session plays it once per round. To watch it alone:
``ros2 service call /motion_imitation_mock/play std_srvs/srv/SetBool "{data: true}"``.

The PEAR stream is a liveness gate, not a data source: the mock ignores the
pose content but only plays while frames are arriving, so losing the camera
stops the motion instead of leaving the animation running on its own.

Known limits of the mock safety:

* The first command after ``~/play`` true, and after the PEAR stream returns
  from a gap longer than ``input_staleness_sec``, is clamped but not
  rate-limited: the clip restarts at frame 0 and the output steps there from
  whatever pose was held. In the shipped clip a frame is at most 1.41 rad from
  frame 0.
* The rate limit uses the configured joint velocity limits and the time
  actually elapsed since the last command, so a late timer tick is allowed a
  proportionally larger step.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import PearFrame

ARM_JOINT_COUNT = 14
_ANIMATION_FORMAT = "aimdk-upper-body-animation"
_ANIMATION_VERSION = 1


def _validated_positive(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _validated_topic(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    if not text.startswith("/"):
        raise ValueError(f"{name} must be an absolute ROS name")
    return text


class Animation:
    """A sampled upper-body animation: uniform grid, radians, arm stream joint order."""

    def __init__(self, arm_pos: np.ndarray, rate_hz: float) -> None:
        self.arm_pos = arm_pos
        self.rate_hz = rate_hz

    @property
    def frames(self) -> int:
        return int(self.arm_pos.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.frames / self.rate_hz

    def sample(self, elapsed_sec: float, *, loop: bool) -> np.ndarray | None:
        """The row for ``elapsed_sec``, or None once a non-looping clip is done."""
        index = int(round(elapsed_sec * self.rate_hz))
        if index >= self.frames:
            if not loop:
                return None
            index %= self.frames
        return self.arm_pos[max(0, index)]


def load_animation(manifest_path: Path) -> Animation:
    """Load the manifest/npz pair, refusing anything that is not the expected contract."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != _ANIMATION_FORMAT:
        raise ValueError(f"animation format must be {_ANIMATION_FORMAT}: {manifest_path}")
    if int(manifest.get("version", 0)) != _ANIMATION_VERSION:
        raise ValueError(f"animation version must be {_ANIMATION_VERSION}: {manifest_path}")
    rate_hz = _validated_positive(manifest.get("rate_hz"), "animation rate_hz")
    data_file = str(manifest.get("data_file", "")).strip()
    if not data_file:
        raise ValueError(f"animation manifest names no data_file: {manifest_path}")
    with np.load(manifest_path.parent / data_file) as payload:
        arm_pos = np.asarray(payload["arm_pos"], dtype=np.float64)
    if arm_pos.ndim != 2 or arm_pos.shape[1] != ARM_JOINT_COUNT:
        raise ValueError(f"animation arm_pos must be (N, {ARM_JOINT_COUNT}), got {arm_pos.shape}")
    if arm_pos.shape[0] < 2 or not np.all(np.isfinite(arm_pos)):
        raise ValueError("animation arm_pos must have at least two finite frames")
    return Animation(arm_pos, rate_hz)


class MotionImitationMockNode(Node):
    """Play a recorded animation through the mock safety stage onto the preview topic."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("motion_imitation_mock", parameter_overrides=parameter_overrides)
        self.declare_parameter("animation_manifest", "")
        self.declare_parameter("pear_frame_topic", "/motion_imitation_stream/pear_frame")
        self.declare_parameter("preview_topic", "/motion_imitation_mock/arm_command_preview")
        self.declare_parameter("status_topic", "/motion_imitation_mock/status")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("input_staleness_sec", 0.5)
        self.declare_parameter("status_period_sec", 5.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("velocity_limits_json", "{}")

        manifest = str(self.get_parameter("animation_manifest").value).strip()
        if not manifest:
            raise ValueError("animation_manifest is required")
        self._animation = load_animation(Path(manifest))
        self._pear_topic = _validated_topic(self.get_parameter("pear_frame_topic").value, "pear_frame_topic")
        preview_topic = _validated_topic(self.get_parameter("preview_topic").value, "preview_topic")
        status_topic = _validated_topic(self.get_parameter("status_topic").value, "status_topic")
        self._rate_hz = _validated_positive(self.get_parameter("publish_rate_hz").value, "publish_rate_hz")
        self._staleness_sec = _validated_positive(
            self.get_parameter("input_staleness_sec").value, "input_staleness_sec"
        )
        status_period = _validated_positive(self.get_parameter("status_period_sec").value, "status_period_sec")
        if type(self.get_parameter("loop").value) is not bool:
            raise ValueError("loop must be a boolean")
        self._loop = self.get_parameter("loop").value

        self._joint_names = tuple(json.loads(str(self.get_parameter("joint_names_json").value)))
        if len(self._joint_names) != ARM_JOINT_COUNT:
            raise ValueError(f"joint_names_json must name exactly {ARM_JOINT_COUNT} joints")
        self._lower, self._upper = self._limit_arrays(json.loads(str(self.get_parameter("joint_limits_json").value)))
        self._max_rate = self._velocity_array(json.loads(str(self.get_parameter("velocity_limits_json").value)))

        self._lock = threading.Lock()
        self._last_input_at = 0.0
        self._input_frames = 0
        self._ticks = 0
        self._published = 0
        self._held_stale = 0
        self._playing = False
        self._play_requests = 0
        self._finished = False
        self._clamped_joints: set[str] = set()
        self._rate_limited_joints: set[str] = set()
        self._playback_started_at = 0.0
        self._last_command: np.ndarray | None = None
        self._last_command_at = 0.0

        self._preview_publisher = self.create_publisher(Float64MultiArray, preview_topic, 10)
        self._status_publisher = self.create_publisher(String, status_topic, 10)
        self.create_subscription(PearFrame, self._pear_topic, self._on_pear_frame, qos_profile_sensor_data)
        self.create_service(SetBool, "~/play", self._on_play)
        self.create_timer(1.0 / self._rate_hz, self._tick)
        self.create_timer(status_period, self._publish_status)

        self.get_logger().info(
            f"motion imitation mock ready: animation={self._animation.frames} frames "
            f"@ {self._animation.rate_hz:g} Hz ({self._animation.duration_sec:.2f}s), loop={self._loop}; "
            f"paused until ~/play; computed commands go to {preview_topic} only"
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

    def _restart_locked(self, now: float) -> None:
        self._playback_started_at = now
        self._last_command = None
        self._last_command_at = 0.0
        self._finished = False

    def _on_play(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """True: restart the clip at frame 0 and play. False: pause; nothing is published."""
        with self._lock:
            self._play_requests += 1
            self._playing = bool(request.data)
            if self._playing:
                self._restart_locked(time.monotonic())
        response.success = True
        response.message = "playing from frame 0" if request.data else "paused"
        self.get_logger().info(f"motion imitation mock {response.message}")
        return response

    def _on_pear_frame(self, _message: PearFrame) -> None:
        """Liveness only: the mock deliberately ignores the pose it is handed."""
        with self._lock:
            now = time.monotonic()
            if self._last_input_at == 0.0 or now - self._last_input_at > self._staleness_sec:
                # First frame, or the stream came back after a gap: restart the
                # clip rather than jumping into the middle of it.
                self._restart_locked(now)
            self._last_input_at = now
            self._input_frames += 1

    def _retarget(self, elapsed_sec: float) -> np.ndarray | None:
        """Sample the clip. A retargeter replaces this; the contract is 14 radians in joint_names_json order."""
        return self._animation.sample(elapsed_sec, loop=self._loop)

    def _apply_safety(self, target: np.ndarray, dt_sec: float) -> np.ndarray:
        """Clamp to the joint limits and rate-limit against the last command.

        The velocity budget is ``limit * dt_sec`` using the time actually
        elapsed since the last command, not a nominal tick. The playback index
        advances on the wall clock, so a late tick legitimately covers more of
        the clip; budgeting per nominal tick instead makes the limiter fire on
        almost every joint whenever the timer runs slow and reshapes a
        trajectory that was already velocity-fitted for this robot.
        """
        clamped = np.clip(target, self._lower, self._upper)
        outside = np.flatnonzero(clamped != target)
        if self._last_command is None or dt_sec <= 0.0:
            limited = clamped
            slewed: np.ndarray = np.empty(0, dtype=np.int64)
        else:
            budget = self._max_rate * dt_sec
            raw_delta = clamped - self._last_command
            # Whether a joint was slewed is decided on the delta against the
            # budget, never by comparing a reconstructed value back to the
            # target: ``last + (target - last)`` differs from ``target`` in the
            # last bits, so that comparison reports every joint as limited on
            # every tick even when nothing was touched.
            slewed = np.flatnonzero(np.abs(raw_delta) > budget)
            limited = clamped
            if slewed.size:
                limited = clamped.copy()
                limited[slewed] = self._last_command[slewed] + np.clip(
                    raw_delta[slewed], -budget[slewed], budget[slewed]
                )
        if outside.size or slewed.size:
            # Named once per joint: a clip produced for this robot should need
            # neither, so the first occurrence is what gets logged.
            new_clamped = {self._joint_names[i] for i in outside} - self._clamped_joints
            new_slewed = {self._joint_names[i] for i in slewed} - self._rate_limited_joints
            self._clamped_joints |= new_clamped
            self._rate_limited_joints |= new_slewed
            if new_clamped:
                self.get_logger().warning(f"mock safety clamped to limits: {sorted(new_clamped)}")
            if new_slewed:
                self.get_logger().warning(f"mock safety rate-limited: {sorted(new_slewed)}")
        return limited

    def _tick(self) -> None:
        with self._lock:
            self._ticks += 1
            now = time.monotonic()
            fresh = self._last_input_at > 0.0 and now - self._last_input_at <= self._staleness_sec
            started_at = self._playback_started_at
            finished = self._finished
            playing = self._playing
        if not playing:
            return
        if not fresh or finished:
            # Publishing nothing is the hold: the session gate keeps sending
            # its last command.
            if not fresh:
                with self._lock:
                    self._held_stale += 1
            return
        target = self._retarget(now - started_at)
        if target is None:
            with self._lock:
                self._finished = True
            self.get_logger().info("motion imitation mock animation complete (loop=false)")
            return
        with self._lock:
            dt_sec = now - self._last_command_at if self._last_command_at else 0.0
        command = self._apply_safety(np.asarray(target, dtype=np.float64), dt_sec)
        message = Float64MultiArray()
        message.data = [float(value) for value in command]
        with self._lock:
            if not self._playing or self._playback_started_at != started_at:
                # Paused or restarted while this tick computed: drop it, so
                # nothing from before a restart follows the play response.
                return
            self._preview_publisher.publish(message)
            self._last_command = command
            self._last_command_at = now
            self._published += 1

    def _publish_status(self) -> None:
        with self._lock:
            now = time.monotonic()
            status = {
                "playing": self._playing,
                "play_requests": self._play_requests,
                "animation_frames": self._animation.frames,
                "animation_rate_hz": self._animation.rate_hz,
                "loop": self._loop,
                "input_topic": self._pear_topic,
                "input_frames": self._input_frames,
                "input_age_sec": round(now - self._last_input_at, 2) if self._last_input_at else None,
                "ticks": self._ticks,
                "published": self._published,
                "held_stale": self._held_stale,
                "finished": self._finished,
                "clamped_joints": sorted(self._clamped_joints),
                "rate_limited_joints": sorted(self._rate_limited_joints),
                "last_command": [round(float(v), 4) for v in self._last_command]
                if self._last_command is not None
                else None,
            }
        message = String()
        message.data = json.dumps(status, ensure_ascii=False)
        self._status_publisher.publish(message)
        if status["input_frames"] == 0:
            self.get_logger().info(
                f"motion imitation mock idle: no frames on {self._pear_topic}; nothing is being computed"
            )
        else:
            self.get_logger().info(f"motion imitation mock status {message.data}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MotionImitationMockNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
