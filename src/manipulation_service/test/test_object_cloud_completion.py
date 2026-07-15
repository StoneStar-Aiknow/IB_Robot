import numpy as np

from manipulation_service.graspgen_wrapper import (
    TablePlane,
    complete_object_cloud_prismatic_extrude,
    complete_scene_cloud_table_holes,
)


def test_prismatic_extrude_reaches_tilted_table_without_filling_interior():
    rng = np.random.default_rng(7)
    normal = np.array([-0.02, 0.35, -0.94], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    d = 0.20

    table_x = np.linspace(-0.22, 0.22, 60)
    table_y = np.linspace(-0.16, 0.16, 50)
    xx, yy = np.meshgrid(table_x, table_y)
    zz = -(normal[0] * xx + normal[1] * yy + d) / normal[2]
    scene = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    center = np.array([0.02, -0.03], dtype=np.float64)
    theta = rng.uniform(0.0, 2.0 * np.pi, 2400)
    radius = np.sqrt(rng.uniform(0.0, 1.0, len(theta)))
    u = center[0] + 0.075 * radius * np.cos(theta)
    v = center[1] + 0.024 * radius * np.sin(theta)
    table_z = -(normal[0] * u + normal[1] * v + d) / normal[2]
    height = 0.018 + 0.018 * (1.0 - radius**2)
    obj = np.column_stack([u, v, table_z]) + height[:, None] * normal

    completed = complete_object_cloud_prismatic_extrude(
        obj,
        scene,
        max_added_points=5000,
        num_layers=5,
        ransac_distance_threshold=0.002,
        ransac_min_inlier_ratio=0.5,
        seed=11,
    )

    assert len(completed) > 1000
    signed = completed @ normal + d
    assert signed.min() < 0.0015
    assert np.quantile(signed, 0.10) < 0.006
    assert signed.max() > 0.02

    tangent_dist = np.linalg.norm(completed[:, :2] - center, axis=1)
    assert np.quantile(tangent_dist, 0.05) > 0.018


def test_table_hole_fill_is_limited_to_object_footprint():
    xs = np.linspace(-0.20, 0.20, 81)
    ys = np.linspace(-0.16, 0.16, 65)
    xx, yy = np.meshgrid(xs, ys)
    scene = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    center_hole = np.linalg.norm(scene[:, :2], axis=1) < 0.035
    far_hole = np.linalg.norm(scene[:, :2] - np.array([0.15, 0.10]), axis=1) < 0.035
    scene = scene[~(center_hole | far_hole)]

    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    footprint = np.column_stack(
        [
            0.025 * np.cos(theta),
            0.018 * np.sin(theta),
            np.full_like(theta, 0.025),
        ]
    )

    completed = complete_scene_cloud_table_holes(
        scene,
        footprint_points=footprint,
        table_plane=TablePlane(normal=np.array([0.0, 0.0, 1.0]), d=0.0, inlier_ratio=1.0),
        grid_size=0.005,
        footprint_dilation_cells=3,
        max_added_points=2000,
    )

    assert len(completed) > 100
    assert np.linalg.norm(completed[:, :2], axis=1).max() < 0.065
    assert np.linalg.norm(completed[:, :2] - np.array([0.15, 0.10]), axis=1).min() > 0.08
