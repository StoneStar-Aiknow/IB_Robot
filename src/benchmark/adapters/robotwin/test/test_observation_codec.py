"""Unit tests for benchmark_robotwin.observation_codec."""

from __future__ import annotations

import numpy as np
import pytest
from benchmark_robotwin.observation_codec import ObservationCodecError, convert_observation


def _raw_obs(arm_dim: int = 6, height: int = 480, width: int = 640) -> dict:
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((height, width, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((height, width, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((height, width, 3), dtype=np.uint8)},
        },
        "joint_action": {
            "left_arm": np.linspace(-1, 1, arm_dim, dtype=np.float32),
            "left_gripper": 0.5,
            "right_arm": np.linspace(-1, 1, arm_dim, dtype=np.float32),
            "right_gripper": 0.25,
        },
    }


def test_default_layout() -> None:
    converted = convert_observation(_raw_obs())
    assert set(converted) == {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.state",
    }
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        assert converted[key].shape == (480, 640, 3)
        assert converted[key].dtype == np.uint8
    state = converted["observation.state"]
    assert state.shape == (14,)
    assert state.dtype == np.float32
    assert state[6] == pytest.approx(0.5)
    assert state[13] == pytest.approx(0.25)


def test_arm_dim_7_state_is_16() -> None:
    converted = convert_observation(_raw_obs(arm_dim=7), arm_dim=7)
    assert converted["observation.state"].shape == (16,)


def test_arm_dim_mismatch_rejected() -> None:
    with pytest.raises(ObservationCodecError, match="left_arm must have 6"):
        convert_observation(_raw_obs(arm_dim=7), arm_dim=6)


def test_resize_nearest_neighbor() -> None:
    raw = _raw_obs(height=240, width=320)
    converted = convert_observation(raw)
    assert converted["observation.images.cam_high"].shape == (480, 640, 3)


def test_missing_camera_rejected() -> None:
    raw = _raw_obs()
    del raw["observation"]["left_camera"]
    with pytest.raises(ObservationCodecError, match="left_camera"):
        convert_observation(raw)


def test_nan_state_rejected() -> None:
    raw = _raw_obs()
    raw["joint_action"]["left_arm"] = raw["joint_action"]["left_arm"].copy()
    raw["joint_action"]["left_arm"][0] = float("nan")
    with pytest.raises(ObservationCodecError, match="NaN"):
        convert_observation(raw)


def test_non_mapping_rejected() -> None:
    with pytest.raises(ObservationCodecError, match="must be a Mapping"):
        convert_observation([1, 2, 3])  # type: ignore[arg-type]
