"""Outlier rejection, low-pass, clamping and rate limiting on (samples, channels) arrays."""

from __future__ import annotations

import numpy as np

# A sample further than this from its neighbourhood median is a tracking glitch,
# not motion: at 15 Hz it would mean more than 600 deg/s on a single joint.
OUTLIER_THRESHOLD_RAD = float(np.radians(40.0))
OUTLIER_WINDOW = 5
LOWPASS_WIDTH = 3


def reject_outliers(
    values: np.ndarray, *, window: int = OUTLIER_WINDOW, threshold: float = OUTLIER_THRESHOLD_RAD
) -> tuple[np.ndarray, int]:
    """Hampel filter: replace samples far from their windowed median by that median."""
    count = values.shape[0]
    if count < 3:
        return values.copy(), 0
    half = window // 2
    index = np.clip(np.arange(count)[:, None] + np.arange(-half, half + 1)[None, :], 0, count - 1)
    median = np.median(values[index], axis=1)
    outlier = np.abs(values - median) > threshold
    return np.where(outlier, median, values), int(np.count_nonzero(outlier))


def low_pass(values: np.ndarray, *, width: int = LOWPASS_WIDTH) -> np.ndarray:
    """Centred moving average with edge padding; keeps the sample count."""
    if width <= 1 or values.shape[0] == 0:
        return values.copy()
    half = width // 2
    padded = np.pad(values, ((half, width - 1 - half), (0, 0)), mode="edge")
    kernel = np.full(width, 1.0 / width)
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])], axis=1)


def clamp(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, int]:
    clamped = np.clip(values, lower, upper)
    return clamped, int(np.count_nonzero(clamped != values))


def rate_limit(values: np.ndarray, max_step: float, *, start: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Limit every step between consecutive samples to ``max_step`` per channel.

    With ``start`` the first sample is also limited relative to it, so a
    trajectory played from ``start`` never jumps. Output stays between the
    previous output and the input sample, so it never leaves a range both of
    them are inside.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    previous = None if start is None else np.asarray(start, dtype=np.float64)
    limited = 0
    for k in range(out.shape[0]):
        if previous is not None:
            step = out[k] - previous
            bounded = np.clip(step, -max_step, max_step)
            limited += int(np.count_nonzero(bounded != step))
            out[k] = previous + bounded
        previous = out[k]
    return out, limited
