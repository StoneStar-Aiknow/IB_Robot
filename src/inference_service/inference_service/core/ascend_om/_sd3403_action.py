"""Shared SD3403 ACT action decoding helpers."""

from __future__ import annotations

import numpy as np

DEFAULT_ACTION_DIM = 6


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def decode_sd3403_action_array(
    data: np.ndarray,
    action_dim: int = DEFAULT_ACTION_DIM,
) -> np.ndarray:
    """Decode the SD3403 worker action output into shape ``(1, -1, action_dim)``.

    The worker returns the action directly as ``(1, chunk, action_dim)`` (e.g.
    ``(1, 100, 6)``); there is no stride padding to strip on the Python side.
    """
    action_dim = _validate_positive_int("action_dim", action_dim)
    action = np.asarray(data, dtype=np.float32)
    if action.ndim >= 2 and action.shape[-1] == action_dim:
        return np.array(action.reshape(1, -1, action_dim), dtype=np.float32, copy=True, order="C")
    raise RuntimeError(
        f"unexpected action tensor shape={tuple(action.shape)}, expected last dim == action_dim({action_dim})"
    )
