import numpy as np
import pytest

from object_tracker.geometry import back_project_pixel, robust_box_depth


def test_robust_box_depth_rejects_background_outlier():
    depth = np.full((10, 10), 2000, dtype=np.uint16)
    depth[4, 4] = 7000
    depth[5, 5] = 0

    measurement = robust_box_depth(depth, (1, 1, 9, 9), central_fraction=1.0)

    assert measurement is not None
    assert measurement.depth_m == pytest.approx(2.0)
    assert measurement.valid_ratio == pytest.approx(63 / 64)
    assert measurement.sample_count == 62


def test_robust_box_depth_rejects_insufficient_valid_samples():
    depth = np.zeros((8, 8), dtype=np.uint16)
    depth[3, 3] = 1000

    assert robust_box_depth(depth, (0, 0, 8, 8), min_valid_ratio=0.2) is None


def test_back_project_pixel_uses_color_intrinsics():
    point = back_project_pixel(420.0, 290.0, 2.0, 500.0, 500.0, 320.0, 240.0)

    np.testing.assert_allclose(point, [0.4, 0.2, 2.0])
