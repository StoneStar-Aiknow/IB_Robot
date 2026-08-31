"""Continuous task-space retargeting from a hand skeleton to Aero compact joints."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .mocap_retarget import extract_features
from .retarget_utils import percentile

MIN_THUMB_DIRECTION_RANGE_RAD = math.radians(5.0)
MIN_THUMB_QUATERNION_SPAN_RAD = math.radians(5.0)
MIN_THUMB_POSITION_CURVE_RANGE = 0.02
MIN_FINGER_CURVE_RANGE = 0.05


def _sub(first, second):
    return np.asarray(first, dtype=float) - np.asarray(second, dtype=float)


def _length(vector) -> float:
    return float(np.linalg.norm(vector))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _curve_ratio(points) -> float:
    """Return scale-independent chain shortening: zero straight, larger when curled."""
    segments = [_length(_sub(child, parent)) for parent, child in zip(points[:-1], points[1:], strict=True)]
    chain_length = sum(segments)
    if chain_length < 1e-9 or any(length < 1e-9 for length in segments):
        raise ValueError("Hand skeleton contains overlapping chain nodes")
    chord = _length(_sub(points[-1], points[0]))
    return _clamp(1.0 - chord / chain_length)


@dataclass(frozen=True, slots=True)
class TaskSpaceRetargetConfig:
    """Dimensionless shape thresholds and solver weights."""

    thumb_adduction_range_rad: float = 0.7
    thumb_opposition_range_rad: float = 0.62
    thumb_curve_range: float = 0.08
    thumb_quaternion_deadband_rad: float | None = None
    thumb_quaternion_range_rad: float | None = None
    finger_curve_range: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    response_gamma: float = 0.65
    thumb_curve_deadband_fraction: float = 0.35
    finger_curve_deadband_fraction: float = 0.1
    neutral_frames: int = 25
    cmc_axis_weight: float = 5.0
    thumb_curve_weight: float = 3.0
    finger_curve_weight: float = 4.0
    smoothness_weight: float = 0.15
    max_normalized_step: float = 0.12

    @classmethod
    def from_dict(cls, values: dict | None) -> TaskSpaceRetargetConfig:
        values = values or {}
        unknown = sorted(set(values).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown task_space retarget settings: " + ", ".join(unknown))
        converted = {}
        for key, value in values.items():
            if key == "finger_curve_range":
                if isinstance(value, int | float):
                    value = [value] * 4
                if not isinstance(value, list | tuple) or len(value) != 4:
                    raise ValueError(f"{key} must contain four values")
                converted[key] = tuple(float(item) for item in value)
            elif key == "neutral_frames":
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        values = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            values.extend(value if isinstance(value, tuple) else [value])
        if not all(math.isfinite(value) for value in values if value is not None):
            raise ValueError("Task-space retarget settings must be finite")
        if (
            min(
                self.thumb_adduction_range_rad,
                self.thumb_opposition_range_rad,
                self.thumb_curve_range,
                *self.finger_curve_range,
            )
            <= 0.0
        ):
            raise ValueError("Task-space motion ranges must be positive")
        if not 0.0 < self.response_gamma <= 1.0:
            raise ValueError("response_gamma must be in (0, 1]")
        if not 0.0 <= self.thumb_curve_deadband_fraction < 1.0:
            raise ValueError("thumb_curve_deadband_fraction must be in [0, 1)")
        quaternion_thresholds = (self.thumb_quaternion_deadband_rad, self.thumb_quaternion_range_rad)
        if (quaternion_thresholds[0] is None) != (quaternion_thresholds[1] is None):
            raise ValueError("Thumb quaternion deadband and range must be configured together")
        if quaternion_thresholds[0] is not None and not (0.0 <= quaternion_thresholds[0] < quaternion_thresholds[1]):
            raise ValueError("Thumb quaternion range must be greater than its non-negative deadband")
        if not 0.0 <= self.finger_curve_deadband_fraction < 1.0:
            raise ValueError("finger_curve_deadband_fraction must be in [0, 1)")
        if self.neutral_frames < 1:
            raise ValueError("neutral_frames must be positive")
        if (
            min(
                self.cmc_axis_weight,
                self.thumb_curve_weight,
                self.finger_curve_weight,
            )
            <= 0.0
        ):
            raise ValueError("Task-space shape weights must be positive")
        if self.smoothness_weight < 0.0 or not 0.0 < self.max_normalized_step <= 1.0:
            raise ValueError("Task-space smoothing settings are invalid")


@dataclass(frozen=True, slots=True)
class HandShapeTarget:
    cmc_abduction: float
    cmc_flexion: float
    thumb_curve: float
    finger_curves: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class HandShapeMetrics:
    thumb_adduction: float
    thumb_opposition: float
    thumb_curve: float
    finger_curves: tuple[float, float, float, float]


def extract_hand_shape_metrics(positions, side: str = "right") -> HandShapeMetrics:
    """Measure scale-independent hand geometry before user-range normalization."""
    if len(positions) < 20:
        raise ValueError("Twenty hand positions are required")
    points = [np.asarray(position, dtype=float) for position in positions[:20]]
    if any(point.shape != (3,) or not np.all(np.isfinite(point)) for point in points):
        raise ValueError("Hand positions must be finite xyz triples")

    features = extract_features(positions, side)
    return HandShapeMetrics(
        thumb_adduction=features[0],
        thumb_opposition=features[1],
        thumb_curve=_curve_ratio([points[index] for index in (1, 2, 3)]),
        finger_curves=tuple(
            _curve_ratio([points[index] for index in chain])
            for chain in ((4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))
        ),
    )


def _relative_curve(value: float, neutral: float, span: float, deadband_fraction: float) -> float:
    return _clamp(max(0.0, value - neutral - span * deadband_fraction) / (span * (1.0 - deadband_fraction)))


def shape_target_from_metrics(
    metrics: HandShapeMetrics,
    neutral: HandShapeMetrics,
    config: TaskSpaceRetargetConfig,
    *,
    thumb_curve_override: float | None = None,
) -> HandShapeTarget:
    """Normalize motion relative to the current SDK connection's open neutral."""
    adduction = _clamp(abs(metrics.thumb_adduction - neutral.thumb_adduction) / config.thumb_adduction_range_rad)
    out_of_plane = _clamp(abs(metrics.thumb_opposition - neutral.thumb_opposition) / config.thumb_opposition_range_rad)
    return HandShapeTarget(
        cmc_abduction=adduction**config.response_gamma,
        cmc_flexion=out_of_plane**config.response_gamma,
        thumb_curve=(
            _relative_curve(
                metrics.thumb_curve,
                neutral.thumb_curve,
                config.thumb_curve_range,
                config.thumb_curve_deadband_fraction,
            )
            ** config.response_gamma
            if thumb_curve_override is None
            else _clamp(thumb_curve_override)
        ),
        finger_curves=tuple(
            _relative_curve(value, baseline, span, config.finger_curve_deadband_fraction) ** config.response_gamma
            for value, baseline, span in zip(
                metrics.finger_curves,
                neutral.finger_curves,
                config.finger_curve_range,
                strict=True,
            )
        ),
    )


def fit_task_space_thresholds(open_frames, sweep_frames, side: str = "right") -> dict[str, float | list[float]]:
    """Fit generic task-space ranges from an open reference and unconstrained sweep."""
    if not open_frames or not sweep_frames:
        raise ValueError("Task-space fitting requires open-reference and sweep frames")
    open_metrics = [extract_hand_shape_metrics(frame, side) for frame in open_frames]
    sweep_metrics = [extract_hand_shape_metrics(frame, side) for frame in sweep_frames]
    neutral = HandShapeMetrics(
        thumb_adduction=statistics.median(item.thumb_adduction for item in open_metrics),
        thumb_opposition=statistics.median(item.thumb_opposition for item in open_metrics),
        thumb_curve=statistics.median(item.thumb_curve for item in open_metrics),
        finger_curves=tuple(
            statistics.median(item.finger_curves[index] for item in open_metrics) for index in range(4)
        ),
    )
    thresholds = {
        "thumb_adduction_range_rad": percentile(
            [abs(item.thumb_adduction - neutral.thumb_adduction) for item in sweep_metrics], 0.98
        ),
        "thumb_opposition_range_rad": percentile(
            [abs(item.thumb_opposition - neutral.thumb_opposition) for item in sweep_metrics], 0.98
        ),
        "thumb_curve_range": percentile(
            [max(0.0, item.thumb_curve - neutral.thumb_curve) for item in sweep_metrics], 0.98
        ),
        "finger_curve_range": [
            percentile(
                [max(0.0, item.finger_curves[index] - neutral.finger_curves[index]) for item in sweep_metrics],
                0.98,
            )
            for index in range(4)
        ],
    }
    TaskSpaceRetargetConfig.from_dict(thresholds)
    return thresholds


def _rotation_from_wxyz(quaternion) -> Rotation:
    if len(quaternion) != 4 or not all(math.isfinite(float(value)) for value in quaternion):
        raise ValueError("mHandPro quaternions must be finite wxyz quadruples")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        raise ValueError("mHandPro quaternion norm is zero")
    return Rotation.from_quat([x / norm, y / norm, z / norm, w / norm])


def _thumb_relative_rotation(quaternions) -> Rotation:
    if quaternions is None or len(quaternions) < 4:
        raise ValueError("Twenty mHandPro node quaternions are required")
    return _rotation_from_wxyz(quaternions[2]).inv() * _rotation_from_wxyz(quaternions[3])


def fit_thumb_quaternion_thresholds(open_frames, sweep_frames) -> dict[str, float]:
    """Fit a CMC-invariant thumb tendon range from relative distal-node rotation."""
    if not open_frames or not sweep_frames:
        raise ValueError("Thumb quaternion fitting requires open-reference and sweep frames")
    open_rotations = [_thumb_relative_rotation(frame.quaternions) for frame in open_frames]
    sweep_rotations = [_thumb_relative_rotation(frame.quaternions) for frame in sweep_frames]
    neutral = Rotation.concatenate(open_rotations).mean()
    open_angles = [float(np.linalg.norm((neutral.inv() * rotation).as_rotvec())) for rotation in open_rotations]
    sweep_angles = [float(np.linalg.norm((neutral.inv() * rotation).as_rotvec())) for rotation in sweep_rotations]
    return {
        "thumb_quaternion_deadband_rad": percentile(open_angles, 0.98),
        "thumb_quaternion_range_rad": percentile(sweep_angles, 0.98),
    }


def validate_fitted_task_space(thresholds: dict) -> None:
    """Reject captures whose useful motion is too close to noise."""
    config = TaskSpaceRetargetConfig.from_dict(thresholds)
    if min(config.thumb_adduction_range_rad, config.thumb_opposition_range_rad) < MIN_THUMB_DIRECTION_RANGE_RAD:
        raise ValueError("Task-space thumb direction coverage is below 5 degrees")
    if min(config.finger_curve_range) < MIN_FINGER_CURVE_RANGE:
        raise ValueError("Task-space finger curve coverage is too small")
    if config.thumb_quaternion_range_rad is not None:
        useful_span = config.thumb_quaternion_range_rad - config.thumb_quaternion_deadband_rad
        if useful_span < MIN_THUMB_QUATERNION_SPAN_RAD:
            raise ValueError("Task-space thumb quaternion coverage is below 5 degrees above open noise")
    elif config.thumb_curve_range < MIN_THUMB_POSITION_CURVE_RANGE:
        raise ValueError("Task-space thumb curve coverage is too small")


class AeroHandRetargeter:
    """Fit seven normalized Aero targets to continuous spatial hand-shape objectives."""

    def __init__(self, config: TaskSpaceRetargetConfig | None = None, *, side: str = "right"):
        self.config = config or TaskSpaceRetargetConfig()
        self.config.validate()
        if side not in ("left", "right"):
            raise ValueError("Aero Hand retarget side must be left or right")
        self.side = side
        self._previous = np.zeros(7, dtype=float)
        self._neutral_samples: list[HandShapeMetrics] = []
        self._neutral: HandShapeMetrics | None = None
        self._thumb_rotation_samples: list[Rotation] = []
        self._neutral_thumb_rotation: Rotation | None = None

    def reset(self) -> None:
        self._previous = np.zeros(7, dtype=float)
        self._neutral_samples = []
        self._neutral = None
        self._thumb_rotation_samples = []
        self._neutral_thumb_rotation = None

    def retarget(self, positions, quaternions=None) -> list[float]:
        metrics = extract_hand_shape_metrics(positions, self.side)
        quaternion_mode = self.config.thumb_quaternion_range_rad is not None
        thumb_rotation = _thumb_relative_rotation(quaternions) if quaternion_mode else None
        if self._neutral is None:
            self._neutral_samples.append(metrics)
            if thumb_rotation is not None:
                self._thumb_rotation_samples.append(thumb_rotation)
            if len(self._neutral_samples) < self.config.neutral_frames:
                return [0.0] * 7
            self._neutral = HandShapeMetrics(
                thumb_adduction=statistics.median(item.thumb_adduction for item in self._neutral_samples),
                thumb_opposition=statistics.median(item.thumb_opposition for item in self._neutral_samples),
                thumb_curve=statistics.median(item.thumb_curve for item in self._neutral_samples),
                finger_curves=tuple(
                    statistics.median(item.finger_curves[index] for item in self._neutral_samples) for index in range(4)
                ),
            )
            if self._thumb_rotation_samples:
                self._neutral_thumb_rotation = Rotation.concatenate(self._thumb_rotation_samples).mean()
            return [0.0] * 7
        thumb_curve_override = None
        if thumb_rotation is not None:
            angle = float(np.linalg.norm((self._neutral_thumb_rotation.inv() * thumb_rotation).as_rotvec()))
            useful_range = self.config.thumb_quaternion_range_rad - self.config.thumb_quaternion_deadband_rad
            thumb_curve_override = (
                _clamp((angle - self.config.thumb_quaternion_deadband_rad) / useful_range) ** self.config.response_gamma
            )
        target = shape_target_from_metrics(
            metrics,
            self._neutral,
            self.config,
            thumb_curve_override=thumb_curve_override,
        )
        initial = self._initial_guess(target)
        previous = self._previous

        def residual(values):
            result = [
                self.config.cmc_axis_weight * (values[0] - target.cmc_abduction),
                self.config.cmc_axis_weight * (values[1] - target.cmc_flexion),
                self.config.thumb_curve_weight * (values[2] - target.thumb_curve),
            ]
            result.extend(
                self.config.finger_curve_weight * (value - desired)
                for value, desired in zip(values[3:], target.finger_curves, strict=True)
            )
            if self.config.smoothness_weight:
                result.extend(self.config.smoothness_weight * (values - previous))
            return result

        solved = least_squares(
            residual,
            initial,
            bounds=(np.zeros(7), np.ones(7)),
            max_nfev=24,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        ).x
        delta = np.clip(
            solved - previous,
            -self.config.max_normalized_step,
            self.config.max_normalized_step,
        )
        solved = previous + delta
        self._previous = solved
        return [float(value) for value in solved]

    @staticmethod
    def _initial_guess(target: HandShapeTarget) -> np.ndarray:
        return np.asarray(
            [target.cmc_abduction, target.cmc_flexion, target.thumb_curve, *target.finger_curves],
            dtype=float,
        )
