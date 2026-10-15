"""Unit tests for benchmark_robotwin.action_codec."""

from __future__ import annotations

import numpy as np
import pytest
from benchmark_robotwin.action_codec import validate_robotwin_action_dict

from benchmark_runtime.adapter import PreNativeValidationError


def _action(values: list[float]) -> dict[str, np.ndarray]:
    return {"action": np.asarray(values, dtype=np.float32)}


def test_default_arm_dim_14_ok() -> None:
    canonical = validate_robotwin_action_dict(_action(list(np.zeros(14))))
    assert canonical.shape == (14,)
    assert canonical.dtype == np.float32


def test_arm_dim_7_gives_16() -> None:
    canonical = validate_robotwin_action_dict(_action(list(np.zeros(16))), arm_dim=7)
    assert canonical.shape == (16,)


def test_wrong_dim_rejected() -> None:
    with pytest.raises(PreNativeValidationError, match="action dim must be 14"):
        validate_robotwin_action_dict(_action(list(np.zeros(16))))


def test_arm_dim_7_mismatch_rejected() -> None:
    with pytest.raises(PreNativeValidationError, match="arm_dim=7"):
        validate_robotwin_action_dict(_action(list(np.zeros(14))), arm_dim=7)


def test_nan_rejected() -> None:
    values = list(np.zeros(14))
    values[3] = float("nan")
    with pytest.raises(PreNativeValidationError, match="NaN"):
        validate_robotwin_action_dict(_action(values))


def test_clipping() -> None:
    # Arm qpos channels are absolute joint targets in radians and must pass
    # through unclipped; only the gripper channels are normalized [0, 1].
    values = [2.0] * 6 + [-0.5] + [-2.0] * 6 + [3.0]
    canonical = validate_robotwin_action_dict(_action(values))
    assert np.all(canonical[:6] == 2.0)
    assert canonical[6] == 0.0
    assert np.all(canonical[7:13] == -2.0)
    assert canonical[13] == 1.0


def test_extra_keys_rejected() -> None:
    payload = {"action": np.zeros(14, dtype=np.float32), "extra": np.zeros(1)}
    with pytest.raises(PreNativeValidationError, match="exactly one key"):
        validate_robotwin_action_dict(payload)


def test_non_dict_rejected() -> None:
    with pytest.raises(PreNativeValidationError, match="must be a dict"):
        validate_robotwin_action_dict([0.0] * 14)  # type: ignore[arg-type]
