"""Shared numerical and safety helpers for hand retargeting models."""

from __future__ import annotations

import math


def percentile(values, quantile: float) -> float:
    """Return a linearly interpolated percentile from finite numeric samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= quantile <= 1.0:
        raise ValueError("Percentile requires samples and a quantile in [0, 1]")
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Percentile samples must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def validate_joint_limits(output_channels, joint_limits) -> dict[str, tuple[float, float]]:
    """Require a finite, increasing radian range for every retarget output."""
    if not isinstance(joint_limits, dict):
        raise ValueError("Retargeter joint_limits must be a mapping")
    validated = {}
    for channel in output_channels:
        limits = joint_limits.get(channel)
        if not isinstance(limits, dict) or "min" not in limits or "max" not in limits:
            raise ValueError(f"Missing joint limits for retarget channel {channel!r}")
        lower = float(limits["min"])
        upper = float(limits["max"])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(f"Invalid joint limits for retarget channel {channel!r}")
        validated[str(channel)] = (lower, upper)
    return validated
