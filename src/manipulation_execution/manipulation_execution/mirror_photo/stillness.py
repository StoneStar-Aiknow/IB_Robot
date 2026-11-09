"""How fast the person moves, from the PEAR body rotations, and when they hold still.

``SpeedMeter`` turns consecutive ``PearFrame.smplx_pose_raw`` into one number:
the mean rotation angle of the 21 body joints between two frames over the
camera time between them (rad/s), smoothed. The root (whole-body turn) and the
hands are left out. ``StillTrigger`` watches that number (``StillRule``).

The 6D layout is ``imitation_retargeting.rotations.decode_rotation_6d``'s; it
is repeated here rather than imported, because importing that package pulls in
the whole retargeting stack. The angle between two rotations reads the same
whichever way the 6D pairs are laid out (``trace(A^T B)`` is the elementwise
product sum), so only the Gram-Schmidt order matters.
"""

from __future__ import annotations

import math

import numpy as np

from .config import StillRule

BODY = slice(1, 22)
POSE_VALUES = 52 * 6
# A gap in the PEAR stream longer than this starts the speed over.
MAX_GAP_SEC = 0.5
# Smoothing time constant of the speed (s).
SMOOTHING_SEC = 0.2
_EPS = 1e-12


def body_rotations(raw) -> np.ndarray:
    """The 21 body joints' rotation matrices, ``(21, 3, 3)``; ``ValueError`` for a malformed pose."""
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if x.shape != (POSE_VALUES,) or not np.isfinite(x).all():
        raise ValueError(f"smplx_pose_raw must be {POSE_VALUES} finite values")
    pairs = x.reshape(52, 2, 3)[BODY]
    a1, a2 = pairs[:, 0], pairs[:, 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), _EPS)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)


def mean_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Mean rotation angle (rad) between two ``(K, 3, 3)`` stacks, joint by joint."""
    cosine = (np.sum(first * second, axis=(1, 2)) - 1.0) / 2.0
    return float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))))


class SpeedMeter:
    """Smoothed body speed (rad/s) per PEAR frame; None until two frames close enough in time."""

    def __init__(self):
        self._last: tuple[float, np.ndarray] | None = None
        self._speed: float | None = None

    def update(self, stamp: float, raw) -> float | None:
        rotations = body_rotations(raw)
        last, self._last = self._last, (stamp, rotations)
        if last is None:
            return None
        dt = stamp - last[0]
        if dt <= 0.0:
            # Out of order or repeated: the newer frame stays the reference.
            self._last = last
            return None
        if dt > MAX_GAP_SEC:
            self._speed = None
            return None
        speed = mean_angle(last[1], rotations) / dt
        if self._speed is None:
            self._speed = speed
        else:
            alpha = 1.0 - math.exp(-dt / SMOOTHING_SEC)
            self._speed += alpha * (speed - self._speed)
        return self._speed

    def reset(self) -> None:
        self._last = None
        self._speed = None


class StillTrigger:
    """``StillRule`` over the speed: returns the stamp of the slowest frame of a held stretch, once per pose."""

    def __init__(self, rule: StillRule):
        self.rule = rule
        self.reset()

    def reset(self) -> None:
        self._below_since: float | None = None
        self._slowest: tuple[float, float] | None = None  # (speed, stamp)
        self._armed = True
        self._moved = False
        self._fired_at: float | None = None

    def update(self, stamp: float, speed: float | None) -> float | None:
        if speed is None:
            self._below_since = None
            self._slowest = None
            return None
        rule = self.rule
        if not self._armed:
            self._moved = self._moved or speed > rule.v_move
            if self._moved and stamp - self._fired_at >= rule.min_gap_sec:
                self._armed = True
                self._moved = False
            else:
                return None
        if speed >= rule.v_still:
            self._below_since = None
            self._slowest = None
            return None
        if self._below_since is None:
            self._below_since = stamp
        if self._slowest is None or speed < self._slowest[0]:
            self._slowest = (speed, stamp)
        if stamp - self._below_since < rule.hold_sec:
            return None
        fired = self._slowest[1]
        self._armed = False
        self._moved = False
        self._fired_at = stamp
        self._below_since = None
        self._slowest = None
        return fired
