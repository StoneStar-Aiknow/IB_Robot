import numpy as np

from semantic_mapping.geometry import ObjectGeometry, is_ground_object, project_masked_depth, transform_geometry


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


def _geometry(points):
    values = np.asarray(points, dtype=np.float32)
    return ObjectGeometry(values, np.median(values, axis=0), np.ptp(values, axis=0))


def test_ground_object_filter_keeps_supported_bounded_geometry():
    geometry = _geometry([[0.0, 0.0, 0.01], [0.2, 0.2, 0.3], [0.1, 0.1, 0.15]])

    assert is_ground_object(
        geometry,
        0.0,
        max_bottom_clearance_m=0.15,
        max_object_height_m=0.75,
        max_footprint_m=1.2,
        max_object_extent_m=0.65,
    )


def test_ground_object_filter_rejects_elevated_and_floor_spanning_geometry():
    elevated = _geometry([[0.0, 0.0, 0.5], [0.2, 0.2, 0.7], [0.1, 0.1, 0.6]])
    floor = _geometry([[-2.0, -2.0, 0.0], [2.0, 2.0, 0.03], [0.0, 0.0, 0.01]])
    options = {
        "max_bottom_clearance_m": 0.15,
        "max_object_height_m": 0.75,
        "max_footprint_m": 1.2,
        "max_object_extent_m": 0.65,
    }

    assert not is_ground_object(elevated, 0.0, **options)
    assert not is_ground_object(floor, 0.0, **options)


def test_ground_object_filter_rejects_geometry_far_from_reference_frame():
    nearby = _geometry([[1.0, 1.0, 0.01], [1.2, 1.2, 0.3], [1.1, 1.1, 0.15]])
    options = {
        "max_bottom_clearance_m": 0.15,
        "max_object_height_m": 0.75,
        "max_footprint_m": 1.2,
        "max_object_extent_m": 0.65,
        "reference_position_xy": np.array([0.0, 0.0]),
        "max_horizontal_distance_m": 1.0,
    }

    assert not is_ground_object(nearby, 0.0, **options)
    options["max_horizontal_distance_m"] = 2.0
    assert is_ground_object(nearby, 0.0, **options)


def test_ground_object_filter_rejects_scene_scale_extent():
    oversized = _geometry([[0.0, 0.0, 0.01], [0.7, 0.2, 0.3], [0.35, 0.1, 0.15]])

    assert not is_ground_object(
        oversized,
        0.0,
        max_bottom_clearance_m=0.15,
        max_object_height_m=0.75,
        max_footprint_m=1.2,
        max_object_extent_m=0.65,
    )
