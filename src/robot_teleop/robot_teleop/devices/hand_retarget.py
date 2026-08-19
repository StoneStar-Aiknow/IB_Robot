"""BaseTeleopDevice adapter for target-independent hand-state topics."""

from __future__ import annotations

import math
import threading
import time

from rclpy.qos import qos_profile_sensor_data

from ibrobot_msgs.msg import HumanHandState

from ..base_teleop import BaseTeleopDevice
from ..hand_retargeting import HandObservation, create_retargeter


class HandRetargetDevice(BaseTeleopDevice):
    def __init__(self, config, node=None):
        super().__init__(config)
        if node is None:
            raise ValueError("hand_retarget requires a ROS node for its HumanHandState subscription")
        self._node = node
        self.source_topic = str(config.get("source_topic", "")).strip()
        self.expected_source = str(config.get("source_name", "")).strip()
        self.side = str(config.get("side", "")).strip()
        self.stale_timeout = float(config.get("stale_timeout", 0.2))
        self.joint_names = tuple(str(name) for name in config.get("joint_names", ()))
        if not self.source_topic or self.side not in ("left", "right") or self.stale_timeout <= 0.0:
            raise ValueError("hand_retarget source_topic, side, and stale_timeout must be valid")
        retargeter_config = dict(config.get("retargeter", {}) or {})
        retargeter_config.setdefault("side", config.get("side", "right"))
        retargeter_config.setdefault("joint_names", list(self.joint_names))
        retargeter_config.setdefault("joint_limits", config.get("joint_limits", {}) or {})
        retargeter_config.setdefault("calib_file", config.get("calib_file", ""))
        self._retargeter = create_retargeter(retargeter_config)
        if tuple(self._retargeter.output_channels) != self.joint_names:
            raise ValueError("hand_retarget output channels must exactly match configured joint_names")
        self._lock = threading.Lock()
        self._latest: HandObservation | None = None
        self._received_at = -math.inf
        self._estop_latched = False
        self._last_log_times = {}
        self._subscription = node.create_subscription(
            HumanHandState,
            self.source_topic,
            self._state_callback,
            qos_profile_sensor_data,
        )

    def connect(self) -> bool:
        with self._lock:
            self._is_connected = True
            self._estop_latched = False
            self._latest = None
            self._received_at = -math.inf
        return True

    def disconnect(self) -> None:
        with self._lock:
            self._is_connected = False
            self._estop_latched = True
            self._latest = None
            self._received_at = -math.inf
        self._retargeter.reset()

    def emergency_stop(self) -> None:
        """Latch output off and discard the last hand state during E-stop."""
        with self._lock:
            self._estop_latched = True
            self._latest = None
            self._received_at = -math.inf
        self._retargeter.reset()

    def emergency_stop_released(self) -> None:
        """Re-arm only after a new post-E-stop HumanHandState frame arrives."""
        with self._lock:
            self._estop_latched = False
            self._latest = None
            self._received_at = -math.inf
        self._retargeter.reset()

    def get_joint_targets(self) -> dict[str, float]:
        with self._lock:
            if not self._is_connected or self._estop_latched:
                return {}
            observation = self._latest
            age = time.monotonic() - self._received_at
        if observation is None:
            self._rate_limited_log("missing", f"No hand state received from {self.source_topic}")
            return {}
        if age > self.stale_timeout:
            self._rate_limited_log("stale", f"Ignoring stale hand state ({age:.3f}s old)")
            return {}
        if not observation.valid:
            self._rate_limited_log("invalid", f"Hand source is not ready: {observation.status}")
            return {}
        try:
            targets = self._retargeter.retarget(observation)
            if set(targets) != set(self.joint_names) or not all(math.isfinite(value) for value in targets.values()):
                raise ValueError("Hand retargeter returned an invalid target set")
            return targets
        except (TypeError, ValueError, ArithmeticError) as exc:
            self._rate_limited_log("retarget", f"Ignoring invalid hand observation: {exc}")
            return {}

    def _state_callback(self, message) -> None:
        with self._lock:
            if self._estop_latched or not self._is_connected:
                return
        try:
            observation = HandObservation.from_message(message)
        except (AttributeError, TypeError, ValueError) as exc:
            self._rate_limited_log("decode", f"Invalid HumanHandState message: {exc}")
            return
        if observation.side != self.side:
            self._rate_limited_log(
                "side",
                f"Ignoring {observation.side} hand state on the {self.side} retarget device",
            )
            return
        if self.expected_source and observation.source != self.expected_source:
            self._rate_limited_log(
                "source",
                f"Ignoring hand state from {observation.source!r}; expected {self.expected_source!r}",
            )
            return
        with self._lock:
            if self._estop_latched or not self._is_connected:
                return
            self._latest = observation
            self._received_at = time.monotonic()

    def _rate_limited_log(self, key: str, message: str, interval: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_log_times.get(key, -math.inf) < interval:
            return
        self._last_log_times[key] = now
        self._node.get_logger().warning(message)
