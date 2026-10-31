"""Idle sway around the prepare pose, played whenever retargeting cannot be."""

from __future__ import annotations

import math

import numpy as np

# Two incommensurate slow frequencies, so the sway never visibly repeats, with
# constant per-joint phases so joints do not move in lockstep and the output is
# reproducible.
_FREQUENCIES_HZ = (0.17, 0.29)
_WEIGHTS = (0.6, 0.4)
_PHASES = ((0.0, 1.3), (2.1, 0.4), (4.2, 2.9), (1.0, 5.1), (3.3, 2.2))


def idle_sway(
    count: int,
    period: float,
    center: np.ndarray,
    amplitude: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """``(count, C)`` sway of at most ``amplitude`` around ``center`` on the grid.

    The centre is first pulled inside the range by the amplitude, so a joint
    whose prepare angle sits on a range edge sways to one side of it instead
    of leaving the range.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    swing_lower = np.minimum(lower + amplitude, (lower + upper) / 2.0)
    swing_upper = np.maximum(upper - amplitude, (lower + upper) / 2.0)
    middle = np.clip(np.asarray(center, dtype=np.float64), swing_lower, swing_upper)
    t = np.arange(count, dtype=np.float64)[:, None] * period
    channels = middle.shape[0]
    phases = np.array([_PHASES[c % len(_PHASES)] for c in range(channels)])
    wave = sum(
        weight * np.sin(2.0 * math.pi * frequency * t + phases[None, :, i])
        for i, (frequency, weight) in enumerate(zip(_FREQUENCIES_HZ, _WEIGHTS, strict=True))
    )
    return middle + amplitude * wave
