"""Joint-limit clamp and per-tick rate cap, ported from the runtime player.

Source: ~/X2-show/ref/video2x2_runtime (commit 0717710),
aimdk_user_ws/src/upper_body_animation/upper_body_animation/
joint_limits.py (LIMITS_DEG, effective_limits_rad) and
followme_player.py (_step_limit, _advance; the non-approaching branch only).
"""

from __future__ import annotations

import numpy as np

# (low, high) in degrees, in arm_pos order, measured on the real robot.
# Copied verbatim from joint_limits.py (video2x2_runtime commit 0717710).
LIMITS_DEG = (
    (-176.5, 116.9), (-3.0, 179.0), (-145.0, 145.0), (-135.0, 0.0),
    (-146.0, 146.0), (-30.0, 30.0), (-40.0, 40.0),
    (-176.5, 116.9), (-179.0, 3.0), (-145.0, 145.0), (-135.0, 0.0),
    (-146.0, 146.0), (-30.0, 30.0), (-40.0, 40.0),
)


def effective_limits_rad(model_low_rad, model_high_rad, margin_deg=1.0) -> tuple[np.ndarray, np.ndarray]:
    """(model ∩ measured) pulled margin_deg inside. 14 channels, arm_pos order."""

    low = np.maximum(np.degrees(np.asarray(model_low_rad, dtype=np.float64)),
                     np.array([lo for lo, _ in LIMITS_DEG], dtype=np.float64))
    high = np.minimum(np.degrees(np.asarray(model_high_rad, dtype=np.float64)),
                      np.array([hi for _, hi in LIMITS_DEG], dtype=np.float64))
    margin = max(0.0, float(margin_deg))
    inner_low = low + margin
    inner_high = high - margin
    inverted = inner_low > inner_high
    if inverted.any():
        midpoint = (low[inverted] + high[inverted]) / 2.0
        inner_low[inverted] = midpoint
        inner_high[inverted] = midpoint
    return np.radians(inner_low), np.radians(inner_high)


class RateLimiter:
    """One rate-capped, limit-clamped step per tick from the last commanded pose."""

    def __init__(self, low_rad, high_rad, rated_rad_s, vel_scale, dt):
        self.low_rad = np.asarray(low_rad, dtype=np.float64)
        self.high_rad = np.asarray(high_rad, dtype=np.float64)
        self.rated_rad_s = np.asarray(rated_rad_s, dtype=np.float64)
        self.vel_scale = float(vel_scale)
        self.dt = float(dt)
        n = self.low_rad.shape[0]
        self.cap_trips = np.zeros(n, dtype=np.int64)
        self.cap_worst_rad = np.zeros(n)
        self.limit_trips = np.zeros(n, dtype=np.int64)
        self.limit_worst_rad = np.zeros(n)
        self._prev = None

    def reset(self, start=None) -> None:
        """Next step starts from ``start``; None = pass the next goal through untouched."""

        self._prev = None if start is None else np.array(start, dtype=np.float64)

    def step(self, goal) -> np.ndarray:
        goal = np.asarray(goal, dtype=np.float64)
        if self._prev is None:
            self._prev = goal.copy()
            return self._prev
        step = self.rated_rad_s * self.vel_scale * self.dt
        delta = goal - self._prev
        excess = np.abs(delta) - step
        hit = excess > 0
        if hit.any():
            self.cap_trips += hit
            self.cap_worst_rad = np.maximum(self.cap_worst_rad,
                                            np.where(hit, excess, 0.0))
        pose = self._prev + np.clip(delta, -step, step)

        clamped = np.clip(pose, self.low_rad, self.high_rad)
        over = np.abs(pose - clamped)
        touched = over > 0
        if touched.any():
            self.limit_trips += touched
            self.limit_worst_rad = np.maximum(self.limit_worst_rad, over)
        self._prev = clamped
        return clamped
