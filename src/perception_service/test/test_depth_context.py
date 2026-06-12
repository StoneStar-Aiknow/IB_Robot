import numpy as np

from perception_service.depth_context import summarize_depth_frame


def test_summarize_depth_frame_reports_stats_and_notes():
    depth = np.array(
        [
            [0.0, 0.25, 0.30],
            [0.18, 0.22, 0.45],
            [0.20, 0.28, 0.35],
        ],
        dtype=np.float32,
    )

    summary = summarize_depth_frame(depth, camera_info=None, max_valid_depth_m=2.0)

    assert summary["available"] is True
    assert summary["depth_valid_ratio"] > 0.8
    assert summary["scene_depth_range_m"] == [0.18, 0.45]
    assert summary["center_depth_m"] is not None
    assert any("near obstacles" in note for note in summary["notes"])


def test_summarize_depth_frame_handles_empty_valid_pixels():
    depth = np.zeros((4, 4), dtype=np.float32)

    summary = summarize_depth_frame(depth)

    assert summary["available"] is False
    assert "no valid pixels" in summary["notes"][0]
