import numpy as np
import pytest

from semantic_mapping.frame_processor import MaskCandidate, filter_masks, prepare_frame


def _frame(**overrides):
    values = {
        "image_bgr": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.full((4, 5), 1000, dtype=np.uint16),
        "intrinsics": np.asarray([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]),
        "depth_scale": 1000.0,
        "rgb_stamp_ns": 1_000_000_000,
        "depth_stamp_ns": 1_010_000_000,
        "info_stamp_ns": 0,
        "camera_frame": "d435_color_optical_frame",
        "translation": np.zeros(3),
        "rotation": np.eye(3),
        "max_stamp_skew_ns": 20_000_000,
        "depth_trunc_m": 4.0,
        "min_valid_depth_ratio": 0.5,
    }
    values.update(overrides)
    return prepare_frame(**values)


def test_prepare_frame_accepts_aligned_rgbd_and_unstamped_camera_info():
    frame = _frame()

    assert frame.camera_frame == "d435_color_optical_frame"
    assert frame.valid_depth_ratio == 1.0


def test_prepare_frame_rejects_timestamp_skew_and_invalid_transform():
    with pytest.raises(ValueError, match="timestamps exceed"):
        _frame(depth_stamp_ns=1_100_000_000)
    rotation = np.eye(3)
    rotation[0, 0] = 2.0
    with pytest.raises(ValueError, match="valid rotation"):
        _frame(rotation=rotation)


def test_prepare_frame_rejects_low_depth_quality():
    depth = np.zeros((4, 5), dtype=np.uint16)
    depth[0] = 1000

    with pytest.raises(ValueError, match="valid ratio"):
        _frame(depth=depth)


def test_filter_masks_records_deterministic_rejections():
    frame = _frame()
    large = np.zeros((4, 5), dtype=np.uint8)
    large[:3, :3] = 1
    overlap = large.copy()
    small = np.zeros((4, 5), dtype=np.uint8)
    small[3, 4] = 1
    depth_invalid = np.zeros((4, 5), dtype=np.uint8)
    depth_invalid[:, 3:] = 1
    frame.depth[:, 3:] = 0
    candidates = [
        MaskCandidate(overlap, 0.8),
        MaskCandidate(small, 0.9),
        MaskCandidate(large, 0.95),
        MaskCandidate(depth_invalid, 0.7),
        MaskCandidate(np.ones((2, 2), dtype=np.uint8), 1.0),
    ]

    accepted, diagnostics = filter_masks(
        frame,
        candidates,
        max_masks=2,
        min_mask_pixels=4,
        min_mask_area_ratio=0.1,
        min_valid_depth_ratio=0.5,
        max_overlap_ratio=0.5,
        depth_trunc_m=4.0,
    )

    assert accepted == [2]
    assert diagnostics.accepted_count == 1
    assert diagnostics.rejected_invalid == 1
    assert diagnostics.rejected_too_small == 1
    assert diagnostics.rejected_depth == 1
    assert diagnostics.rejected_overlap == 1


def test_filter_masks_uses_input_index_as_final_tie_breaker():
    frame = _frame()
    first = np.zeros((4, 5), dtype=np.uint8)
    first[:, :2] = 1
    second = np.zeros((4, 5), dtype=np.uint8)
    second[:, 3:] = 1

    accepted, _ = filter_masks(
        frame,
        [MaskCandidate(first, 0.5), MaskCandidate(second, 0.5)],
        max_masks=1,
        min_mask_pixels=1,
        min_mask_area_ratio=0.0,
        min_valid_depth_ratio=0.0,
        max_overlap_ratio=1.0,
        depth_trunc_m=4.0,
    )

    assert accepted == [0]
