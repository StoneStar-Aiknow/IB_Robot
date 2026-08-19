"""Model-based retargeting from mHandPro geometry to Aero compact joints."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .mocap_retarget import (
    FEATURE_SCHEMA_AERO_COMPACT,
    MIN_CALIBRATION_SPAN,
    extract_finger_flexions,
    extract_sdk_skeleton_features,
    extract_thumb_kinematics,
)
from .retarget_utils import percentile

THUMB_FEATURE_NAMES = ("root_yaw_rad", "root_pitch_rad", "mcp_flex_rad", "ip_flex_rad")
THUMB_ENDPOINT_NAMES = (*THUMB_FEATURE_NAMES, "mcp_ip_flex_rad")
FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_ACTIVE_TRIM_FRACTION = 0.08
AERO_THUMB_MCP_IP_WEIGHTS = (9.4372, 12.5)
# Thumb root motion is often sampled at a comfortable pose boundary rather than
# a repeatable mechanical end stop.  Use an interior sweep envelope so one
# accidental extreme pose cannot dilute the runtime mapping sensitivity.
THUMB_ROOT_SWEEP_LOWER_QUANTILE = 0.10
THUMB_ROOT_SWEEP_UPPER_QUANTILE = 0.90
THUMB_FLEX_SWEEP_LOWER_QUANTILE = 0.02
THUMB_FLEX_SWEEP_UPPER_QUANTILE = 0.98


@dataclass(frozen=True, slots=True)
class AeroFingerModelConfig:
    """Map anatomical PIP/DIP flexion to each Aero finger tendon."""

    pip_weight: float = 0.55
    dip_weight: float = 0.45
    open_threshold_rad: float = math.radians(15.0)
    closed_threshold_rad: float = math.radians(50.0)
    active_trim_fraction: float = FINGER_ACTIVE_TRIM_FRACTION

    @classmethod
    def from_dict(cls, values: dict | None) -> AeroFingerModelConfig:
        values = values or {}
        unknown = sorted(set(values).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown Aero finger model settings: " + ", ".join(unknown))
        converted = {key: float(value) for key, value in values.items()}
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        values = (
            self.pip_weight,
            self.dip_weight,
            self.open_threshold_rad,
            self.closed_threshold_rad,
            self.active_trim_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Aero finger model settings must be finite")
        if self.pip_weight < 0.0 or self.dip_weight < 0.0 or self.pip_weight + self.dip_weight <= 0.0:
            raise ValueError("Aero finger PIP/DIP weights must be non-negative and non-zero")
        if self.open_threshold_rad < 0.0 or self.closed_threshold_rad <= self.open_threshold_rad:
            raise ValueError("Aero finger thresholds must satisfy 0 <= open < closed")
        if not 0.0 <= self.active_trim_fraction < 1.0:
            raise ValueError("Aero finger active_trim_fraction must be in [0, 1)")


def _smoothstep(value: float) -> float:
    clipped = max(0.0, min(1.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _finger_contractions(positions, config: AeroFingerModelConfig) -> list[float]:
    flexions = extract_finger_flexions(positions)
    weight_sum = config.pip_weight + config.dip_weight
    contractions = []
    for name in FINGER_NAMES:
        _, pip_flex, dip_flex = flexions[name]
        contractions.append((config.pip_weight * pip_flex + config.dip_weight * dip_flex) / weight_sum)
    return contractions


def _finger_shape_targets(positions, config: AeroFingerModelConfig, endpoints=None) -> list[float]:
    contractions = _finger_contractions(positions, config)
    targets = []
    for name, contraction in zip(FINGER_NAMES, contractions, strict=True):
        endpoint = endpoints.get(name) if isinstance(endpoints, dict) else None
        if endpoint is None:
            neutral = 0.0
            active = config.closed_threshold_rad
        else:
            neutral = float(endpoint["neutral"])
            active = float(endpoint["active"])
            active -= (active - neutral) * config.active_trim_fraction
        usable_span = active - neutral - config.open_threshold_rad
        if not all(math.isfinite(value) for value in (neutral, active)) or usable_span <= 0.0:
            raise ValueError(f"Aero {name} calibrated finger span is too small after trimming")
        progress = (contraction - neutral - config.open_threshold_rad) / usable_span
        target = _smoothstep(progress)
        if not math.isfinite(target):
            raise ValueError(f"Aero {name} shape target is non-finite")
        targets.append(target)
    return targets


@dataclass(frozen=True, slots=True)
class AeroThumbModelConfig:
    """Map anatomical thumb axes to the three Aero compact thumb joints."""

    root_output_scales: tuple[float, float] = (0.95, 0.94)
    root_neutral_trims: tuple[float, float] = (0.0, 0.0)
    root_active_trims: tuple[float, float] = (0.0, 0.0)
    root_deadband_rad: float = math.radians(1.0)
    mcp_ip_weights: tuple[float, float] = AERO_THUMB_MCP_IP_WEIGHTS
    tendon_deadband_rad: float = math.radians(3.0)
    tendon_output_scale: float = 0.60
    max_thumb_step_rad: tuple[float, float, float] = (math.inf, math.inf, math.inf)

    @classmethod
    def from_dict(cls, values: dict | None) -> AeroThumbModelConfig:
        values = values or {}
        unknown = sorted(set(values).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown Aero thumb model settings: " + ", ".join(unknown))
        converted = dict(values)
        for key in ("root_output_scales", "mcp_ip_weights"):
            if key not in converted:
                continue
            scales = converted[key]
            if not isinstance(scales, list | tuple) or len(scales) != 2:
                raise ValueError(f"Aero thumb {key} must contain two values")
            converted[key] = tuple(float(item) for item in scales)
        for key in ("root_neutral_trims", "root_active_trims"):
            if key in converted:
                trims = converted[key]
                if not isinstance(trims, list | tuple) or len(trims) != 2:
                    raise ValueError(f"Aero thumb {key} must contain two values")
                converted[key] = tuple(float(item) for item in trims)
        if "max_thumb_step_rad" in converted:
            steps = converted["max_thumb_step_rad"]
            if not isinstance(steps, list | tuple) or len(steps) != 3:
                raise ValueError("Aero thumb max_thumb_step_rad must contain three values")
            converted["max_thumb_step_rad"] = tuple(float(item) for item in steps)
        for key in ("root_deadband_rad", "tendon_deadband_rad", "tendon_output_scale"):
            if key in converted:
                converted[key] = float(converted[key])
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        values = [
            *self.root_output_scales,
            *self.root_neutral_trims,
            *self.root_active_trims,
            *self.mcp_ip_weights,
            self.root_deadband_rad,
            self.tendon_deadband_rad,
            self.tendon_output_scale,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Aero thumb model settings must be finite")
        if any(value < 0.0 for value in self.mcp_ip_weights) or sum(self.mcp_ip_weights) <= 0.0:
            raise ValueError("Aero thumb MCP/IP weights must be non-negative and non-zero")
        if not math.isclose(
            self.mcp_ip_weights[0] * AERO_THUMB_MCP_IP_WEIGHTS[1],
            self.mcp_ip_weights[1] * AERO_THUMB_MCP_IP_WEIGHTS[0],
            rel_tol=1e-6,
        ):
            raise ValueError("Aero thumb MCP/IP weights must preserve the SDK tendon coefficient ratio")
        output_scales = (*self.root_output_scales, self.tendon_output_scale)
        if any(not 0.0 < value <= 1.0 for value in output_scales):
            raise ValueError("Aero thumb output scales must be in (0, 1]")
        if any(not 0.0 <= value < 1.0 for value in (*self.root_neutral_trims, *self.root_active_trims)):
            raise ValueError("Aero thumb root trims must be in [0, 1)")
        if any(
            neutral + active >= 1.0
            for neutral, active in zip(self.root_neutral_trims, self.root_active_trims, strict=True)
        ):
            raise ValueError("Aero thumb neutral and active trims must leave a non-empty input range")
        if self.root_deadband_rad < 0.0 or self.tendon_deadband_rad < 0.0:
            raise ValueError("Aero thumb deadbands must be non-negative")
        if any(math.isnan(value) or value <= 0.0 for value in self.max_thumb_step_rad):
            raise ValueError("Aero thumb max_thumb_step_rad values must be positive")


def _directional_normalize(
    value: float,
    endpoint: dict,
    deadband: float,
    label: str,
    *,
    neutral_trim: float = 0.0,
    active_trim: float = 0.0,
) -> float:
    if not isinstance(endpoint, dict) or set(endpoint) != {"neutral", "active"}:
        raise ValueError(f"Aero compact {label} endpoint must contain neutral and active")
    neutral = float(endpoint["neutral"])
    active = float(endpoint["active"])
    if not math.isfinite(value) or not math.isfinite(neutral) or not math.isfinite(active):
        raise ValueError(f"Aero compact {label} endpoint values must be finite")
    span = active - neutral
    span_magnitude = abs(span)
    if (
        span_magnitude < MIN_CALIBRATION_SPAN
        or not 0.0 <= neutral_trim < 1.0
        or not 0.0 <= active_trim < 1.0
        or neutral_trim + active_trim >= 1.0
    ):
        raise ValueError(f"Aero compact {label} endpoint span is too small")
    neutral += span * neutral_trim
    active -= span * active_trim
    span = active - neutral
    span_magnitude = abs(span)
    if deadband >= span_magnitude:
        raise ValueError(f"Aero compact {label} endpoint span is too small after trimming")
    direction = math.copysign(1.0, span)
    progress = (value - neutral) * direction
    return max(0.0, min(1.0, (progress - deadband) / (span_magnitude - deadband)))


def _fit_endpoint(
    values,
    neutral: float,
    label: str,
    *,
    lower_quantile: float = THUMB_FLEX_SWEEP_LOWER_QUANTILE,
    upper_quantile: float = THUMB_FLEX_SWEEP_UPPER_QUANTILE,
) -> dict[str, float]:
    lower = percentile(values, lower_quantile)
    upper = percentile(values, upper_quantile)
    active = max((lower, upper), key=lambda candidate: abs(candidate - neutral))
    if abs(active - neutral) < MIN_CALIBRATION_SPAN:
        raise ValueError(f"Aero compact {label} sweep coverage is too small")
    return {"neutral": float(neutral), "active": float(active)}


def _mcp_ip_flex(mcp_flex: float, ip_flex: float, weights: tuple[float, float]) -> float:
    """Project two human joints onto Aero's single MCP/IP tendon coordinate."""
    weight_sum = sum(weights)
    return (weights[0] * mcp_flex + weights[1] * ip_flex) / weight_sum


def build_aero_compact_calibration(
    open_frames,
    sweep_frames,
    side,
    mcp_ip_weights: tuple[float, float] = AERO_THUMB_MCP_IP_WEIGHTS,
) -> dict:
    """Fit human thumb endpoints without learning four-finger end-range poses."""
    if not open_frames or not sweep_frames:
        raise ValueError(f"{side} Aero compact calibration requires open-reference and motion-sweep frames")
    open_features = [
        extract_sdk_skeleton_features(frame.positions, frame.virtual_positions, side) for frame in open_frames
    ]
    low = [statistics.median(row[index] for row in open_features) for index in range(7)]
    open_rows = [extract_thumb_kinematics(frame.positions, frame.virtual_positions, side) for frame in open_frames]
    sweep_rows = [extract_thumb_kinematics(frame.positions, frame.virtual_positions, side) for frame in sweep_frames]
    neutral = (
        statistics.median(row.root_yaw for row in open_rows),
        statistics.median(row.root_pitch for row in open_rows),
        statistics.median(row.mcp_flex for row in open_rows),
        statistics.median(row.ip_flex for row in open_rows),
    )
    columns = (
        [row.root_yaw for row in sweep_rows],
        [row.root_pitch for row in sweep_rows],
        [row.mcp_flex for row in sweep_rows],
        [row.ip_flex for row in sweep_rows],
    )
    open_finger_rows = [_finger_contractions(frame.positions, AeroFingerModelConfig()) for frame in open_frames]
    sweep_finger_rows = [_finger_contractions(frame.positions, AeroFingerModelConfig()) for frame in sweep_frames]
    thumb_endpoints = {}
    for index, (name, values, start) in enumerate(zip(THUMB_FEATURE_NAMES, columns, neutral, strict=True)):
        quantiles = (
            (THUMB_ROOT_SWEEP_LOWER_QUANTILE, THUMB_ROOT_SWEEP_UPPER_QUANTILE)
            if index < 2
            else (THUMB_FLEX_SWEEP_LOWER_QUANTILE, THUMB_FLEX_SWEEP_UPPER_QUANTILE)
        )
        thumb_endpoints[name] = _fit_endpoint(
            values,
            start,
            name,
            lower_quantile=quantiles[0],
            upper_quantile=quantiles[1],
        )
    combined_neutral = _mcp_ip_flex(neutral[2], neutral[3], mcp_ip_weights)
    combined_values = [_mcp_ip_flex(row.mcp_flex, row.ip_flex, mcp_ip_weights) for row in sweep_rows]
    thumb_endpoints["mcp_ip_flex_rad"] = _fit_endpoint(
        combined_values,
        combined_neutral,
        "mcp_ip_flex_rad",
    )
    finger_endpoints = {}
    for index, name in enumerate(FINGER_NAMES):
        finger_endpoints[name] = _fit_endpoint(
            [row[index] for row in sweep_finger_rows],
            statistics.median(row[index] for row in open_finger_rows),
            name,
        )
    low[2] = combined_neutral
    low[3:] = [finger_endpoints[name]["neutral"] for name in FINGER_NAMES]
    high = [
        thumb_endpoints["root_yaw_rad"]["active"],
        thumb_endpoints["root_pitch_rad"]["active"],
        thumb_endpoints["mcp_ip_flex_rad"]["active"],
        *(finger_endpoints[name]["active"] for name in FINGER_NAMES),
    ]
    return {
        "low": low,
        "high": high,
        "thumb_endpoints": thumb_endpoints,
        "finger_endpoints": finger_endpoints,
        "feature_schema": FEATURE_SCHEMA_AERO_COMPACT,
    }


def aero_compact_to_normalized(
    positions,
    virtual_positions,
    calibration,
    side,
    config: AeroThumbModelConfig,
    finger_config: AeroFingerModelConfig | None = None,
):
    """Return seven normalized Aero compact targets from complete hand geometry."""
    if calibration.get("feature_schema") != FEATURE_SCHEMA_AERO_COMPACT:
        raise ValueError(
            f"{side} calibration feature_schema must be {FEATURE_SCHEMA_AERO_COMPACT!r} for aero_compact mode"
        )
    endpoints = calibration.get("thumb_endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != set(THUMB_ENDPOINT_NAMES):
        raise ValueError(f"{side} aero_compact calibration requires five thumb_endpoints")
    finger_endpoints = calibration.get("finger_endpoints")
    if not isinstance(finger_endpoints, dict) or set(finger_endpoints) != set(FINGER_NAMES):
        raise ValueError(f"{side} aero_compact calibration requires four finger_endpoints")

    thumb = extract_thumb_kinematics(positions, virtual_positions, side)
    root_features = tuple(
        _directional_normalize(
            value,
            endpoints[name],
            config.root_deadband_rad,
            name,
            neutral_trim=config.root_neutral_trims[index],
            active_trim=config.root_active_trims[index],
        )
        for index, (name, value) in enumerate(
            zip(("root_yaw_rad", "root_pitch_rad"), (thumb.root_yaw, thumb.root_pitch), strict=True)
        )
    )
    # The mHandPro palm-local pitch/yaw planes correspond to Aero's physical
    # CMC abduction/flexion axes in the opposite order.
    root_targets = (root_features[1], root_features[0])

    mcp_ip_flex = _mcp_ip_flex(thumb.mcp_flex, thumb.ip_flex, config.mcp_ip_weights)
    tendon_target = _directional_normalize(
        mcp_ip_flex,
        endpoints["mcp_ip_flex_rad"],
        config.tendon_deadband_rad,
        "mcp_ip_flex_rad",
    )

    fingers = _finger_shape_targets(
        positions,
        finger_config or AeroFingerModelConfig(),
        finger_endpoints,
    )

    return [
        max(0.0, min(1.0, root_targets[0])) * config.root_output_scales[0],
        max(0.0, min(1.0, root_targets[1])) * config.root_output_scales[1],
        tendon_target * config.tendon_output_scale,
        *fingers,
    ]
