"""How much a grid trajectory moves, normalised per channel range."""

from __future__ import annotations

import numpy as np


def travel(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Sum over channels of path length divided by that channel's range.

    Measured after clamping, so it reflects how much the robot itself would
    move; the normalisation keeps a wide joint from dominating a narrow one.
    """
    if values.shape[0] < 2:
        return 0.0
    span = np.maximum(np.asarray(upper) - np.asarray(lower), 1e-9)
    return float(np.sum(np.sum(np.abs(np.diff(values, axis=0)), axis=0) / span))


def pick_side(scores: dict[str, float], arm_side: str) -> str:
    """``left``/``right`` are taken as asked; ``auto`` takes the side that moves more."""
    if arm_side in scores:
        return arm_side
    if arm_side != "auto":
        raise ValueError(f"unknown arm side {arm_side!r}")
    return max(sorted(scores), key=lambda side: scores[side])
