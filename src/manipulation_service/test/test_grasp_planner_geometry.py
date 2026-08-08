import numpy as np
import pytest

from manipulation_service.grasp_planner_node import _compute_detection_geometry, _volume_centroid_hull


def test_volume_centroid_hull_matches_unit_cube():
    points = np.asarray(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float64,
    )

    centroid, volume = _volume_centroid_hull(points)

    np.testing.assert_allclose(centroid, [0.5, 0.5, 0.5])
    assert volume == pytest.approx(1.0)


def test_detection_geometry_uses_mask_depth_and_rejects_depth_outlier():
    mask = np.ones((4, 4), dtype=np.uint8)
    depth = np.full((4, 4), 1000, dtype=np.uint16)
    depth[0, 0] = 5000
    intrinsics = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    surface, volume_centroid, volume, point_count = _compute_detection_geometry(
        mask,
        depth,
        intrinsics,
        depth_scale=1000.0,
    )

    assert point_count == 15
    np.testing.assert_allclose(surface, [1.6, 1.6, 1.0])
    np.testing.assert_allclose(volume_centroid, surface)
    assert volume == 0.0
