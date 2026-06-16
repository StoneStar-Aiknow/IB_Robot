"""``tool_frame`` → ``base`` angular-velocity converter.

Converts a tool-frame angular velocity into the base frame using the
current TF rotation between the two frames:

.. math:: \\omega^B = R_{B \\leftarrow T} \\cdot \\omega^T

with three safety layers:

1. ``tool_frame == base_link`` → pass-through, no TF lookup
2. TF unavailable yet (cold-start) → caches ``last_R``; hold until first success
3. **TF staleness watchdog**: if cached ``last_R`` is older than
   ``stale_threshold_s`` (default 0.2 s) the adapter forces
   ``angular = (0,0,0)`` — preventing "death twitching" when TF drops mid-motion.
"""

from __future__ import annotations

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as SciR

try:
    from tf2_ros import (
        ExtrapolationException,
        LookupException,
        TransformException,
    )
except ImportError:  # pragma: no cover - tf2_ros always available in workspace
    LookupException = Exception  # type: ignore[assignment,misc]
    ExtrapolationException = Exception  # type: ignore[assignment,misc]
    TransformException = Exception  # type: ignore[assignment,misc]


Vec3 = tuple[float, float, float]


class ToolAngularAdapter:
    """Convert per-frame angular velocity from ``tool_frame`` to ``base``."""

    def __init__(
        self,
        node,
        tf_buffer,
        base_link: str,
        tool_frame: str,
        lookup_timeout_s: float = 0.01,
        stale_threshold_s: float = 0.2,
    ):
        self._node = node
        self._tf = tf_buffer
        self._base = base_link
        self._tool = tool_frame
        self._timeout = Duration(seconds=lookup_timeout_s)
        self._stale_threshold = Duration(seconds=stale_threshold_s)
        self._last_R: SciR | None = None
        self._last_R_stamp: Time | None = None

    @property
    def is_pass_through(self) -> bool:
        return self._tool == self._base

    def convert(self, linear: Vec3, angular_tool: Vec3) -> tuple[Vec3, Vec3]:
        """Return ``(linear, angular_base)`` ready for the unified solver."""
        if self.is_pass_through:
            return linear, angular_tool

        now = self._node.get_clock().now()
        try:
            t = self._tf.lookup_transform(self._base, self._tool, Time(), self._timeout)
            q = t.transform.rotation
            R_B_T = SciR.from_quat([q.x, q.y, q.z, q.w])
            self._last_R = R_B_T
            self._last_R_stamp = now
        except (LookupException, ExtrapolationException, TransformException) as exc:
            if self._last_R is None or self._last_R_stamp is None:
                self._warn_throttled(f"TF {self._base} <- {self._tool} not ready: {exc}")
                return linear, (0.0, 0.0, 0.0)
            # Stale TF cache: force zero angular to prevent death twitching.
            if (now - self._last_R_stamp) > self._stale_threshold:
                self._warn_throttled(
                    f"TF {self._base} <- {self._tool} stale "
                    f"> {self._stale_threshold.nanoseconds * 1e-9:.2f}s, "
                    f"forcing angular=(0,0,0)"
                )
                return linear, (0.0, 0.0, 0.0)
            R_B_T = self._last_R

        omega_base = R_B_T.apply(np.asarray(angular_tool, dtype=np.float64))
        return linear, (float(omega_base[0]), float(omega_base[1]), float(omega_base[2]))

    def _warn_throttled(self, msg: str) -> None:
        logger = self._node.get_logger()
        # rclpy logger exposes warn() always; throttle attr varies by ROS version.
        warn_throttled = getattr(logger, "warn_throttle", None)
        if callable(warn_throttled):
            warn_throttled(1.0, msg)
        else:
            logger.warn(msg)
