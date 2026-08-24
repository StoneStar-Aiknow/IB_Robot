"""Ego-motion-compensated target motion classification in odom."""

import math
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class MotionState(IntEnum):
    UNKNOWN = 0
    STATIONARY = 1
    MOVING = 2


@dataclass(frozen=True)
class MotionEstimate:
    state: MotionState
    displacement_m: float
    speed_mps: float
    threshold_m: float
    ego_motion_active: bool
    reason: str


@dataclass(frozen=True)
class _Sample:
    stamp_s: float
    position: np.ndarray
    covariance: np.ndarray


class EgoCompensatedMotionClassifier:
    """Classify independent target motion from timestamped odom positions."""

    def __init__(
        self,
        *,
        window_samples: int = 3,
        min_displacement_m: float = 0.08,
        min_speed_mps: float = 0.05,
        sigma_multiplier: float = 3.0,
        moving_confirmation_windows: int = 2,
        stationary_confirmation_windows: int = 2,
        max_position_variance_m2: float = 0.25,
        max_sample_gap_s: float = 3.0,
        robot_linear_motion_threshold_mps: float = 0.02,
        robot_angular_motion_threshold_rps: float = 0.05,
    ) -> None:
        if window_samples < 2:
            raise ValueError("window_samples must be at least 2")
        if moving_confirmation_windows < 1 or stationary_confirmation_windows < 1:
            raise ValueError("confirmation windows must be positive")
        self._samples: deque[_Sample] = deque(maxlen=window_samples)
        self._min_displacement = float(min_displacement_m)
        self._min_speed = float(min_speed_mps)
        self._sigma_multiplier = float(sigma_multiplier)
        self._moving_confirmation = int(moving_confirmation_windows)
        self._stationary_confirmation = int(stationary_confirmation_windows)
        self._max_position_variance = float(max_position_variance_m2)
        self._max_sample_gap = float(max_sample_gap_s)
        self._robot_linear_threshold = float(robot_linear_motion_threshold_mps)
        self._robot_angular_threshold = float(robot_angular_motion_threshold_rps)
        self._state = MotionState.UNKNOWN
        self._moving_windows = 0
        self._stationary_windows = 0

    @property
    def state(self) -> MotionState:
        return self._state

    def update(
        self,
        *,
        stamp_s: float,
        position_odom: tuple[float, float],
        position_covariance: np.ndarray,
        robot_linear_speed_mps: float = 0.0,
        robot_angular_speed_rps: float = 0.0,
    ) -> MotionEstimate:
        position = np.asarray(position_odom, dtype=np.float64)
        covariance = np.asarray(position_covariance, dtype=np.float64)
        if position.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("position and covariance must have shapes (2,) and (2, 2)")
        if not math.isfinite(stamp_s) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(covariance)):
            raise ValueError("motion inputs must be finite")
        ego_motion_active = (
            abs(robot_linear_speed_mps) >= self._robot_linear_threshold
            or abs(robot_angular_speed_rps) >= self._robot_angular_threshold
        )
        if float(np.max(np.linalg.eigvalsh(covariance))) > self._max_position_variance:
            return self._unknown(ego_motion_active, "target position covariance exceeds motion limit")
        if self._samples:
            gap = stamp_s - self._samples[-1].stamp_s
            if gap <= 0.0 or gap > self._max_sample_gap:
                self.reset()
                self._samples.append(_Sample(stamp_s, position, covariance))
                return self._estimate(ego_motion_active, "motion window reset after invalid timestamp gap")
        self._samples.append(_Sample(stamp_s, position, covariance))
        if len(self._samples) < self._samples.maxlen:
            return self._estimate(ego_motion_active, "collecting motion window")

        first = self._samples[0]
        latest = self._samples[-1]
        displacement = float(np.linalg.norm(latest.position - first.position))
        elapsed = latest.stamp_s - first.stamp_s
        speed = displacement / elapsed
        combined_covariance = first.covariance + latest.covariance
        uncertainty = self._sigma_multiplier * math.sqrt(
            max(float(np.max(np.linalg.eigvalsh(combined_covariance))), 0.0)
        )
        threshold = max(self._min_displacement, uncertainty)
        moving = displacement > threshold and speed > self._min_speed
        if moving:
            self._moving_windows += 1
            self._stationary_windows = 0
            if self._moving_windows >= self._moving_confirmation:
                self._state = MotionState.MOVING
            reason = "target motion exceeds odom displacement, speed, and covariance gates"
        else:
            self._moving_windows = 0
            self._stationary_windows += 1
            if self._stationary_windows >= self._stationary_confirmation:
                self._state = MotionState.STATIONARY
            reason = "target remains stationary after ego-motion compensation"
        return MotionEstimate(self._state, displacement, speed, threshold, ego_motion_active, reason)

    def reset(self) -> None:
        self._samples.clear()
        self._state = MotionState.UNKNOWN
        self._moving_windows = 0
        self._stationary_windows = 0

    def _unknown(self, ego_motion_active: bool, reason: str) -> MotionEstimate:
        self.reset()
        return self._estimate(ego_motion_active, reason)

    def _estimate(self, ego_motion_active: bool, reason: str) -> MotionEstimate:
        return MotionEstimate(self._state, 0.0, 0.0, self._min_displacement, ego_motion_active, reason)
