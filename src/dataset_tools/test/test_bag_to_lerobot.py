"""Tests for bag_to_lerobot helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.bag_to_lerobot import (  # noqa: E402
    _build_feature_conversion_table,
    _clean_float_array,
    _dataset_feature_names_for_spec,
    _estimate_stream_rate_hz,
    _resolve_video_codec,
    _selected_indices_for_ticks,
)
from robot_config.utils import resolve_calibration_source_specs_from_config  # noqa: E402


def test_resolve_video_codec_prefers_h264_in_auto_mode(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        return DummyCodec(is_encoder=name == "h264")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "h264"


def test_resolve_video_codec_falls_back_to_av1_when_h264_missing(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        if name == "h264":
            raise ValueError("missing")
        return DummyCodec(is_encoder=name == "libsvtav1")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "libsvtav1"


def test_estimate_stream_rate_hz_uses_timestamp_span():
    ts = [0, 33_333_333, 66_666_666, 100_000_000]

    assert abs(_estimate_stream_rate_hz(ts) - 30.0) < 0.05


def test_selected_indices_for_ticks_exposes_hold_duplicates_from_phase_offset():
    ts = np.array([0, 34, 68], dtype=np.int64)
    ticks = np.array([0, 33, 66], dtype=np.int64)

    selected = _selected_indices_for_ticks(
        policy="hold",
        ts_ns=ts,
        ticks_ns=ticks,
        step_ns=33,
        tol_ns=0,
    )

    assert selected.tolist() == [0, 0, 1]


def test_dataset_feature_names_for_current_preserve_contract_names():
    class Spec:
        key = "observation.current"
        names = ["current.1", "current.2"]

    assert _dataset_feature_names_for_spec(Spec()) == ["current.1", "current.2"]


def test_clean_float_array_replaces_non_finite_values():
    arr = _clean_float_array([1.0, np.nan, np.inf, -np.inf], np.float32)

    assert arr.dtype == np.float32
    assert arr.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_clean_float_array_warns_for_non_current_features(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.state")
    assert any("observation.state" in rec.message for rec in caplog.records if rec.levelno >= 30)


def test_clean_float_array_silent_for_current(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.current")
    assert not any("non-finite" in rec.message for rec in caplog.records if rec.levelno >= 30)


def test_fallback_conversion_table_prefers_explicit_source_specs_over_legacy_pathsep(tmp_path):
    front = tmp_path / "front.json"
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    legacy = tmp_path / "legacy.json"
    for index, path in enumerate((front, left, right, legacy), start=1):
        path.write_text(json.dumps({"1": {"range_min": 1000 + index, "range_max": 3000 + index}}))

    specs = resolve_calibration_source_specs_from_config(
        {
            "ros2_control": {
                "xacro_args": {
                    "calib_file_front": str(front),
                    "calib_file_left": str(left),
                    "calib_file_right": str(right),
                }
            }
        }
    )

    table = _build_feature_conversion_table(
        feature_names=["joint1_front", "joint1_left", "joint1_right"],
        conversion_meta={},
        fallback_config={
            "norm_mode": "range_m100_100",
            "gripper_joints": [],
            "calibration_source_specs": specs,
            "calibration_file": str(legacy),
        },
    )

    assert len(table) == 3
    front_rad_min, front_rad_max, *_ = table[0]
    left_rad_min, left_rad_max, *_ = table[1]
    right_rad_min, right_rad_max, *_ = table[2]

    ticks_per_rad = 4096.0 / (2.0 * np.pi)
    assert front_rad_min == (1001 - 2048.0) / ticks_per_rad
    assert front_rad_max == (3001 - 2048.0) / ticks_per_rad
    assert left_rad_min == (1002 - 2048.0) / ticks_per_rad
    assert left_rad_max == (3002 - 2048.0) / ticks_per_rad
    assert right_rad_min == (1003 - 2048.0) / ticks_per_rad
    assert right_rad_max == (3003 - 2048.0) / ticks_per_rad
