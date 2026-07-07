import numpy as np

from manipulation_service.graspgen_wrapper import complete_object_cloud_prismatic_extrude


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
