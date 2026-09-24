"""Data gates, resampling onto the fixed playback grid, and the entry blend.

The downstream trajectory carries one ``waypoint_duration_sec`` for the whole
batch, so the plan must be a uniform grid. Waypoint ``k`` is reached at
``(k + 1) * period``; with ``M`` waypoints the playback lasts ``M * period``,
which is made exactly the requested duration.
"""

from __future__ import annotations

import numpy as np

GRID_PERIOD_SEC = 0.05
# Rate PEAR was measured to deliver on the robot; only used to judge density.
NOMINAL_SOURCE_HZ = 15.0
MIN_DENSITY = 0.1
MAX_GAP_FRACTION = 0.5
# Stamps spanning more than this many windows cannot come from one capture.
MAX_SPAN_FRACTION = 2.0
MIN_ENTRY_SEC = 0.5
MAX_ENTRY_SEC = 1.5


def grid(duration_sec: float) -> tuple[int, float]:
    """Waypoint count and period so that ``count * period == duration_sec``."""
    count = max(1, int(round(duration_sec / GRID_PERIOD_SEC)))
    return count, duration_sec / count


def gate(t: np.ndarray, duration_sec: float) -> str | None:
    """Why the capture timeline cannot be played back, or ``None`` if it can."""
    count = int(t.shape[0])
    if count < 2:
        return f"only {count} usable frame(s)"
    span = float(t[-1] - t[0])
    if not span > 0.0:
        return "frame stamps span no time"
    if span > MAX_SPAN_FRACTION * duration_sec:
        return f"frame stamps span {span:.2f}s for a {duration_sec:.2f}s capture"
    density = count / (NOMINAL_SOURCE_HZ * duration_sec)
    if density < MIN_DENSITY:
        return f"frame density {density:.3f} below {MIN_DENSITY}"
    gap = float(np.max(np.diff(t)))
    if gap > MAX_GAP_FRACTION * duration_sec:
        return f"frame gap {gap:.2f}s exceeds {MAX_GAP_FRACTION * duration_sec:.2f}s"
    return None


def resample(t: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Stretch the capture span onto ``count`` grid points and interpolate linearly."""
    position = (t - t[0]) / (t[-1] - t[0]) * (count - 1) if count > 1 else np.zeros_like(t)
    target = np.arange(count, dtype=np.float64)
    return np.stack([np.interp(target, position, values[:, c]) for c in range(values.shape[1])], axis=1)


def entry_blend(values: np.ndarray, start: np.ndarray, period: float, max_speed: float) -> np.ndarray:
    """Ease in from ``start`` over the first samples without adding any.

    The first sample equals ``start`` exactly, so playback begins where the
    arm already is. The blend window grows with the distance to cover so its peak speed,
    ``1.5 * distance / window`` for a smoothstep, stays near ``max_speed``.
    """
    distance = float(np.max(np.abs(values[0] - start))) if values.shape[0] else 0.0
    window = min(max(1.5 * distance / max_speed, MIN_ENTRY_SEC), MAX_ENTRY_SEC)
    x = np.clip(np.arange(values.shape[0]) * period / window, 0.0, 1.0)
    weight = (x * x * (3.0 - 2.0 * x))[:, None]
    return (1.0 - weight) * start + weight * values
