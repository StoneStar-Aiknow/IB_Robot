"""Target-independent hand observations and pluggable mechanical-hand retargeters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .devices.aero_compact_retarget import AeroFingerModelConfig, AeroThumbModelConfig, aero_compact_to_normalized
from .devices.glove_calibration import load_calibration
from .devices.mocap_retarget import CHANNEL_NAMES, FEATURE_SCHEMA_AERO_COMPACT
from .devices.retarget_utils import validate_joint_limits
from .hand_state import HUMAN_HAND_SCHEMA

HAND_LANDMARK_COUNT = 20
HAND_ORIENTATION_COUNT = 20
HAND_VIRTUAL_TIP_COUNT = 5


@dataclass(frozen=True, slots=True)
class HandObservation:
    source: str
    schema: str
    side: str
    sequence: int
    timestamp_ns: int
    valid: bool
    status: str
    positions: list[list[float]]
    quaternions_wxyz: list[list[float]]
    virtual_positions: list[list[float]]
    features: dict[str, float]

    @classmethod
    def from_message(cls, message) -> HandObservation:
        landmarks = list(message.landmarks)
        orientations = list(message.orientations)
        virtual_tips = list(message.virtual_tips)
        feature_names = [str(name) for name in message.feature_names]
        feature_values = [float(value) for value in message.features]
        source = str(message.source).strip()
        schema = str(message.schema).strip()
        side = str(message.side).strip()
        if not source:
            raise ValueError("HumanHandState source must be non-empty")
        if schema != HUMAN_HAND_SCHEMA:
            raise ValueError(f"HumanHandState schema must be {HUMAN_HAND_SCHEMA!r}, got {schema!r}")
        if side not in ("left", "right"):
            raise ValueError("HumanHandState side must be 'left' or 'right'")
        if len(landmarks) != HAND_LANDMARK_COUNT:
            raise ValueError(f"HumanHandState landmarks must contain exactly {HAND_LANDMARK_COUNT} points")
        if len(orientations) != HAND_ORIENTATION_COUNT:
            raise ValueError(f"HumanHandState orientations must contain exactly {HAND_ORIENTATION_COUNT} quaternions")
        if len(virtual_tips) != HAND_VIRTUAL_TIP_COUNT:
            raise ValueError(f"HumanHandState virtual_tips must contain exactly {HAND_VIRTUAL_TIP_COUNT} points")
        if len(feature_names) != len(feature_values):
            raise ValueError("HumanHandState feature_names and features must have equal lengths")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("HumanHandState feature_names must be unique")
        positions = [[float(point.x), float(point.y), float(point.z)] for point in landmarks]
        quaternions = [
            [float(orientation.w), float(orientation.x), float(orientation.y), float(orientation.z)]
            for orientation in orientations
        ]
        virtual_positions = [[float(point.x), float(point.y), float(point.z)] for point in virtual_tips]
        numeric_values = [
            *(value for point in positions for value in point),
            *(value for quaternion in quaternions for value in quaternion),
            *(value for point in virtual_positions for value in point),
            *feature_values,
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("HumanHandState geometry and features must be finite")
        return cls(
            source=source,
            schema=schema,
            side=side,
            sequence=int(message.sequence),
            timestamp_ns=int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec),
            valid=bool(message.valid),
            status=str(message.status),
            positions=positions,
            quaternions_wxyz=quaternions,
            virtual_positions=virtual_positions,
            features=dict(zip(feature_names, feature_values, strict=True)),
        )


class HandRetargeter(Protocol):
    @property
    def output_channels(self) -> tuple[str, ...]: ...

    def retarget(self, observation: HandObservation) -> dict[str, float]: ...

    def reset(self) -> None: ...


_RETARGETERS = {}


def register_retargeter(name: str, factory) -> None:
    normalized = str(name).strip()
    if not normalized or normalized in _RETARGETERS:
        raise ValueError(f"Hand retargeter name must be non-empty and unique: {name!r}")
    _RETARGETERS[normalized] = factory


def create_retargeter(config: dict) -> HandRetargeter:
    retargeter_type = str(config.get("type", "")).strip()
    try:
        factory = _RETARGETERS[retargeter_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported hand retargeter type: {retargeter_type!r}") from exc
    return factory(config)


class AeroCompactRetargeter:
    """Preserve the verified seven-channel Aero mapping behind a generic plugin boundary."""

    def __init__(self, config: dict):
        self.side = str(config.get("side", "right"))
        self._output_channels = tuple(str(name) for name in config.get("joint_names", ()))
        if len(self._output_channels) != 7 or len(set(self._output_channels)) != 7:
            raise ValueError("aero_compact retargeter requires seven unique output channels")
        calibration_file = str(config.get("calib_file", "")).strip()
        if not calibration_file or not Path(calibration_file).expanduser().is_file():
            raise ValueError(f"aero_compact calibration file not found: {calibration_file!r}")
        self.calibration = load_calibration(calibration_file, self.side, require_persistence=False)
        if self.calibration.get("feature_schema") != FEATURE_SCHEMA_AERO_COMPACT:
            raise ValueError(f"aero_compact requires calibration schema {FEATURE_SCHEMA_AERO_COMPACT!r}")
        self.model = AeroThumbModelConfig.from_dict(config.get("aero_thumb_model"))
        self.finger_model = AeroFingerModelConfig.from_dict(config.get("aero_finger_model"))
        self.joint_limits = validate_joint_limits(self._output_channels, config.get("joint_limits"))
        deadbands = config.get("normalized_deadbands", {}) or {}
        unknown = set(deadbands) - set(CHANNEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown Aero normalized deadbands: {sorted(unknown)}")
        self.deadbands = {str(name): float(value) for name, value in deadbands.items()}
        self._last_thumb_targets = [self.joint_limits[channel][0] for channel in self._output_channels[:3]]

    @property
    def output_channels(self) -> tuple[str, ...]:
        return self._output_channels

    def retarget(self, observation: HandObservation) -> dict[str, float]:
        if observation.side != self.side:
            raise ValueError(f"Expected {self.side} hand state, got {observation.side}")
        normalized = aero_compact_to_normalized(
            observation.positions,
            observation.virtual_positions,
            self.calibration,
            self.side,
            self.model,
            self.finger_model,
        )
        normalized = [
            _apply_deadband(value, self.deadbands.get(channel, 0.0))
            for channel, value in zip(CHANNEL_NAMES, normalized, strict=True)
        ]
        targets = {}
        for channel, value in zip(self.output_channels, normalized, strict=True):
            lower, upper = self.joint_limits[channel]
            targets[channel] = lower + value * (upper - lower)
        for index, channel in enumerate(self.output_channels[:3]):
            previous = self._last_thumb_targets[index]
            maximum_step = self.model.max_thumb_step_rad[index]
            targets[channel] = previous + max(-maximum_step, min(maximum_step, targets[channel] - previous))
            self._last_thumb_targets[index] = targets[channel]
        return targets

    def reset(self) -> None:
        self._last_thumb_targets[:] = [self.joint_limits[channel][0] for channel in self._output_channels[:3]]


class SynergyMatrixRetargeter:
    """Declarative feature-to-actuator mapping for rapid mechanical-hand adaptation."""

    def __init__(self, config: dict):
        self.input_features = tuple(str(name) for name in config.get("input_features", ()))
        self._output_channels = tuple(str(name) for name in config.get("joint_names", ()))
        matrix = config.get("matrix", ())
        offsets = config.get("offsets", [0.0] * len(self._output_channels))
        if (
            not self.input_features
            or len(set(self.input_features)) != len(self.input_features)
            or not self._output_channels
            or len(set(self._output_channels)) != len(self._output_channels)
        ):
            raise ValueError("synergy_matrix requires input_features and joint_names")
        if len(matrix) != len(self._output_channels) or any(len(row) != len(self.input_features) for row in matrix):
            raise ValueError("synergy_matrix dimensions must be output_channels x input_features")
        if len(offsets) != len(self._output_channels):
            raise ValueError("synergy_matrix offsets must match output channels")
        self.matrix = tuple(tuple(float(value) for value in row) for row in matrix)
        self.offsets = tuple(float(value) for value in offsets)
        if not all(math.isfinite(value) for row in self.matrix for value in row) or not all(
            math.isfinite(value) for value in self.offsets
        ):
            raise ValueError("synergy_matrix weights and offsets must be finite")
        self.joint_limits = validate_joint_limits(self._output_channels, config.get("joint_limits"))

    @property
    def output_channels(self) -> tuple[str, ...]:
        return self._output_channels

    def retarget(self, observation: HandObservation) -> dict[str, float]:
        try:
            inputs = [observation.features[name] for name in self.input_features]
        except KeyError as exc:
            raise ValueError(f"Hand state is missing feature {exc.args[0]!r}") from exc
        targets = {}
        for channel, row, offset in zip(self.output_channels, self.matrix, self.offsets, strict=True):
            value = offset + sum(weight * feature for weight, feature in zip(row, inputs, strict=True))
            lower, upper = self.joint_limits[channel]
            value = max(lower, min(upper, value))
            if not math.isfinite(value):
                raise ValueError(f"synergy_matrix produced a non-finite value for {channel!r}")
            targets[channel] = value
        return targets

    def reset(self) -> None:
        return None


def _apply_deadband(value: float, deadband: float) -> float:
    if not math.isfinite(deadband) or not 0.0 <= deadband < 1.0:
        raise ValueError("Normalized deadband must be in [0, 1)")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Normalized hand target must be finite")
    return max(0.0, min(1.0, (value - deadband) / (1.0 - deadband)))


register_retargeter("aero_compact", AeroCompactRetargeter)
register_retargeter("synergy_matrix", SynergyMatrixRetargeter)
