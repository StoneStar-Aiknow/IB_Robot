import numpy as np

from semantic_mapping.geometry import project_masked_depth, transform_geometry


def test_project_masked_depth_backprojects_into_optical_frame():
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1:, 1:] = 1
    intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])

    geometry = project_masked_depth(mask, depth, intrinsics, 1000.0, 3.0, min_points=4)

    assert geometry is not None
    assert geometry.points.shape == (4, 3)
    assert np.allclose(geometry.centroid, [0.005, 0.005, 1.0])


def test_transform_geometry_moves_points_and_recomputes_extent():
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    mask = np.ones((2, 2), dtype=np.uint8)
    intrinsics = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    geometry = project_masked_depth(mask, depth, intrinsics, 1000.0, 3.0, min_points=4)

    world = transform_geometry(geometry, np.array([1.0, 2.0, 3.0]), np.eye(3))

    assert np.allclose(world.centroid, geometry.centroid + [1.0, 2.0, 3.0])
    assert np.all(world.size >= 0.0)


def test_project_masked_depth_rejects_unsynchronized_dimensions():
    with np.testing.assert_raises_regex(ValueError, "identical dimensions"):
        project_masked_depth(
            np.ones((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint16),
            np.eye(3),
            1000.0,
            3.0,
            1,
        )
