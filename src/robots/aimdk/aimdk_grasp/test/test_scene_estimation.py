"""Geometric regression tests for support selection and depth-aware self filtering."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.scene_estimation import (
    LEG_OBSTACLE,
    NoSupportError,
    SceneEstimationConfig,
    _sphere_membership,
    _sphere_membership_cpu,
    _voxel_surface,
    estimate_scene,
    select_support_plane,
    self_filter_mask,
)
from aimdk_grasp.types import CameraIntrinsics
from scipy.spatial.transform import Rotation


@pytest.fixture(autouse=True)
def unbounded_left_arm_reach(monkeypatch):
    """Synthetic scenes place their camera anywhere in torso_link, far from the
    real X2 left shoulder, so the reach crop keeps everything here."""
    from aimdk_grasp import scene_estimation

    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.zeros(3), np.inf))


def horizontal_grid(width, height, z, count):
    x, y = np.meshgrid(np.linspace(-width / 2, width / 2, count), np.linspace(-height / 2, height / 2, count))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def test_support_selection_prefers_target_table_to_larger_floor():
    table = horizontal_grid(0.44, 0.74, 0.8, 25)
    floor = horizontal_grid(2.0, 2.0, 0.0, 55)
    target = horizontal_grid(0.04, 0.04, 0.85, 5)
    rotation = Rotation.from_euler("xyz", [15, -10, 30], degrees=True).as_matrix()
    translation = np.array([1.2, -0.7, 0.4])
    points = np.vstack((table, floor)) @ rotation.T + translation
    target = target @ rotation.T + translation
    plane = select_support_plane(points, target, rotation[:, 2])
    np.testing.assert_allclose(plane.normal, rotation[:, 2], atol=1e-8)
    np.testing.assert_allclose((plane.point - translation) @ rotation[:, 2], 0.8, atol=1e-8)
    assert plane.target_heights_m["p05"] == pytest.approx(0.05)
    cuboid = plane.cuboid(SceneEstimationConfig())
    basis = Rotation.from_quat(np.array(cuboid["pose"][3:])[[1, 2, 3, 0]]).as_matrix()
    local = (plane.inliers - cuboid["pose"][:3]) @ basis
    assert np.all(np.abs(local) <= np.array(cuboid["dims"]) / 2 + 1e-8)
    top = np.array(cuboid["pose"][:3]) + basis[:, 2] * cuboid["dims"][2] / 2
    assert abs((top - plane.point) @ plane.normal) < 1e-8
    assert np.linalg.det(basis) == pytest.approx(1.0)


@pytest.mark.parametrize("target_offset", [[0, 0, -0.1], [0, 0, 0.3], [1.0, 0, 0.05]])
def test_support_rejects_target_below_too_far_or_outside_footprint(target_offset):
    table = horizontal_grid(0.5, 0.5, 0.0, 25)
    target = horizontal_grid(0.02, 0.02, 0.0, 5) + target_offset
    with pytest.raises(NoSupportError, match="No near-field support"):
        select_support_plane(table, target, [0, 0, 1])


def test_support_without_mask_uses_nearest_horizontal_plane():
    near = horizontal_grid(0.4, 0.5, 0.5, 25)
    far = horizontal_grid(2.0, 2.0, 1.5, 55)
    plane = select_support_plane(np.vstack((far, near)), np.empty((0, 3)), [0, 0, 1])
    assert plane.point[2] == pytest.approx(0.5)


@pytest.mark.parametrize("targeted", [False, True])
def test_disconnected_coplanar_surfaces_do_not_stretch_support(targeted):
    table = horizontal_grid(0.4, 0.5, 0.7, 25)
    other = horizontal_grid(0.5, 0.5, 0.7, 25) + [0.0, 1.2, 0.0]
    # A small remote cluster must not enlarge the table either.
    stray = horizontal_grid(0.02, 0.02, 0.7, 3) + [1.1, 0.0, 0.0]
    target = horizontal_grid(0.02, 0.02, 0.75, 5) if targeted else np.empty((0, 3))
    plane = select_support_plane(np.vstack((table, other, stray)), target, [0, 0, 1])
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1e-8)
    assert np.abs(plane.inliers[:, 0]).max() <= 0.2 + 1e-8
    assert np.abs(plane.inliers[:, 1]).max() <= 0.25 + 1e-8
    assert sorted(plane.cuboid(SceneEstimationConfig())["dims"]) == pytest.approx([0.04, 0.44, 0.54])


def test_target_over_gap_between_coplanar_supports_is_rejected():
    left = horizontal_grid(0.3, 0.4, 0.7, 20) + [-0.4, 0.0, 0.0]
    right = horizontal_grid(0.3, 0.4, 0.7, 20) + [0.4, 0.0, 0.0]
    target = horizontal_grid(0.02, 0.02, 0.75, 5)
    with pytest.raises(NoSupportError, match="No near-field support"):
        select_support_plane(np.vstack((left, right)), target, [0, 0, 1])


def test_sparse_disconnected_samples_cannot_jointly_pass_support_minimum():
    points = np.array([[x, y, 0.7] for x in np.arange(10) * 0.1 for y in np.arange(10) * 0.1])
    with pytest.raises(NoSupportError, match="No near-field support"):
        select_support_plane(points, np.empty((0, 3)), [0, 0, 1])


@pytest.mark.parametrize("cpu_only", [False, True])
def test_self_filter_preserves_same_ray_obstacles_and_uses_full_transform(cpu_only):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [25, 40, -90], degrees=True).as_matrix()
    transform[:3, 3] = [1.0, -2.0, 0.3]
    center_camera = np.array([0.0, 0.0, 1.0])
    center_torso = transform[:3, :3] @ center_camera + transform[:3, 3]
    spheres = np.array([[*center_torso, 0.1], [*center_torso, -1.0]])
    points = np.array([[0, 0, 1.0], [0, 0, 0.8], [0, 0, 1.2], [0.105, 0, 1.0], [0.12, 0, 1.0]])
    np.testing.assert_array_equal(
        self_filter_mask(points, spheres, transform, cpu_only=cpu_only), [False, True, True, False, True]
    )
    np.testing.assert_array_equal(
        self_filter_mask(points, spheres[1:], transform, cpu_only=cpu_only), np.ones(5, dtype=bool)
    )
    np.testing.assert_array_equal(
        self_filter_mask(points, np.empty((0, 4)), transform, cpu_only=cpu_only), np.ones(5, dtype=bool)
    )


def test_self_filter_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="Nx4"):
        self_filter_mask([[0, 0, 1]], [[0, 0, np.nan, 0.1]], np.eye(4))
    with pytest.raises(ValueError, match="rotation"):
        self_filter_mask([[0, 0, 1]], np.empty((0, 4)), np.diag([2, 1, 1, 1]))


@pytest.mark.parametrize("pitch,margin", [(0.005, 0.01), (0.02, 0.035), (0.03, 0.052)])
def test_voxel_reconstruction_does_not_refill_filtered_sphere(pitch, margin):
    import trimesh

    rng = np.random.default_rng(4)
    points = rng.normal(size=(10000, 3))
    points *= (0.101 + margin) / np.linalg.norm(points, axis=1)[:, None]
    sphere = np.array([[0.0, 0.0, 0.0, 0.1]])
    points = points[self_filter_mask(points, sphere, np.eye(4), margin)]
    vertices, faces = _voxel_surface(points, pitch)
    closest = trimesh.triangles.closest_point(vertices[faces], np.zeros((len(faces), 3)))
    assert np.linalg.norm(closest, axis=1).min() > 0.1


@pytest.mark.parametrize(
    "change",
    [
        {"self_filter_margin_m": 0.01},
        {"mesh_pitch_m": 0.04},
        {"validation_mesh_pitch_m": 0.01},
        {"target_mesh_pitch_m": 0.04},
    ],
)
def test_scene_config_rejects_voxel_refilling_robot(change):
    with pytest.raises(ValueError, match="Voxel diagonal"):
        SceneEstimationConfig(**change)


def test_sparse_voxel_mesh_matches_dense_boundary_order():
    from aimdk_grasp.scene_estimation import _voxel_cells

    rng = np.random.default_rng(8)
    points = np.concatenate([rng.uniform(-0.15, 0.15, (900, 3)), [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    pitch = 0.025
    origin, cells = _voxel_cells(points, pitch)
    dense = np.zeros(cells.max(axis=0) + 3, dtype=bool)
    dense[tuple(cells.T)] = True
    templates = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
        ],
        dtype=np.float64,
    )
    expected, triangles, base = [], [], 0
    for axis, direction, face in ((0, 1, 1), (0, -1, 0), (1, 1, 3), (1, -1, 2), (2, 1, 5), (2, -1, 4)):
        coords = np.argwhere(dense & ~np.roll(dense, -direction, axis=axis))
        expected.append((coords[:, None] + templates[face]).reshape(-1, 3) * pitch + origin)
        ids = np.arange(len(coords)) * 4 + base
        triangles.append(np.column_stack((ids, ids + 1, ids + 2, ids, ids + 2, ids + 3)).reshape(-1, 3))
        base += 4 * len(coords)
    vertices, faces = _voxel_surface(points, pitch)
    np.testing.assert_array_equal(vertices, np.concatenate(expected))
    np.testing.assert_array_equal(faces, np.concatenate(triangles))


def test_voxel_volume_and_winding_match_independent_closed_mesh_with_cavity():
    import trimesh
    from aimdk_grasp.scene_estimation import _surface_mesh

    cells = np.indices((5, 5, 5)).reshape(3, -1).T
    # Hollow shell, duplicates, and a disconnected component exercise both
    # internal and external boundaries and unique-cell volume counting.
    shell = cells[(cells == 0).any(axis=1) | (cells == 4).any(axis=1)]
    cloud = np.vstack((shell, shell, [[10, 0, 0]])) * 0.03125 + [0.5, -0.75, 1.25]
    raw_vertices, raw_faces = _voxel_surface(cloud, 0.03125)
    mesh, counts = _surface_mesh(cloud, 0.03125, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(mesh["vertices"], raw_vertices)
    np.testing.assert_array_equal(mesh["faces"].reshape(-1, 3), raw_faces[:, ::-1])
    geometry = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"].reshape(-1, 3))
    assert geometry.is_watertight and geometry.is_winding_consistent and geometry.volume > 0
    assert counts["occupied_volume_m3"] == pytest.approx(geometry.volume, rel=1e-10)


def test_large_cuda_voxel_boundary_exactly_matches_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from aimdk_grasp.scene_estimation import _voxel_surface_data

    rng = np.random.default_rng(4)
    points = rng.uniform(-0.3, 0.3, (65000, 3))
    points = np.vstack((points, points[:10000], [[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]))
    gpu = _voxel_surface_data(points, 0.005)
    assert gpu[2] >= 50000
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu = _voxel_surface_data(points, 0.005)
    assert gpu[2] == cpu[2]
    np.testing.assert_array_equal(gpu[0], cpu[0])
    np.testing.assert_array_equal(gpu[1], cpu[1])


def test_self_filter_backends_match_bruteforce():
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, (5000, 3)) + [0.0, 0.0, 1.5]
    centers = rng.uniform(-1.0, 1.0, (25, 3)) + [0.0, 0.0, 1.5]
    radii = rng.uniform(0.01, 0.3, 25)
    brute = ~(((points[:, None] - centers[None]) ** 2).sum(-1) <= radii[None, :] ** 2).any(axis=1)
    np.testing.assert_array_equal(_sphere_membership_cpu(points, centers, radii), brute)
    np.testing.assert_array_equal(_sphere_membership(points, centers, radii), brute)
    assert brute.any() and not brute.all()


@pytest.mark.parametrize("cpu", [False, True])
def test_shared_distance_filter_matches_separate_margins_and_preserves_targets(monkeypatch, cpu):
    from aimdk_grasp.scene_estimation import _self_filter_masks

    if cpu:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(12)
    points = rng.uniform(-0.4, 0.4, (4096, 3)) + [0.0, 0.0, 1.0]
    centers = rng.uniform(-0.25, 0.25, (40, 3)) + [0.0, 0.0, 1.0]
    radii = rng.uniform(0.03, 0.12, len(centers))
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [18, -35, 70], degrees=True).as_matrix()
    matrix[:3, 3] = [0.3, -0.8, 1.2]
    spheres = np.column_stack((centers @ matrix[:3, :3].T + matrix[:3, 3], radii))
    # Disabled spheres must not erase anything, even with a positive margin.
    spheres = np.vstack((spheres, [*matrix[:3, 3], -1.0]))
    target = np.arange(len(points)) % 7 == 0
    fine, coarse = _self_filter_masks(points, spheres, matrix, [0.01, 0.052])
    expected_fine = self_filter_mask(points, spheres, matrix, 0.01)
    expected_coarse = self_filter_mask(points[expected_fine & ~target], spheres, matrix, 0.052)
    np.testing.assert_array_equal(fine, expected_fine)
    np.testing.assert_array_equal(coarse[fine & ~target], expected_coarse)
    assert fine.any() and not fine.all() and (fine & ~coarse).any()
    for index, margin in enumerate((0.01, 0.052)):
        brute = ~(((points[:, None] - centers[None]) ** 2).sum(-1) <= (radii[None] + margin) ** 2).any(axis=1)
        np.testing.assert_array_equal((fine, coarse)[index], brute)


# These synthetic scenes are nothing but a table under the target. The tests
# below inspect the coarse environment mesh, so they keep the table in it
# (the default leaves it to the cuboid; see tests/test_reach_crop.py).
TABLE_MESHED = SceneEstimationConfig(leave_table_to_cuboid=False)


def scene_inputs() -> dict[str, Any]:
    depth = np.ones((50, 50), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[23:27, 23:27] = True
    depth[mask] = 0.9
    matrix = np.diag([1.0, -1.0, -1.0, 1.0])
    matrix[:3, 3] = [1.0, 2.0, 2.0]
    transform = ImageTransform(123, "optical", "torso_link", matrix)
    return dict(
        depth=depth,
        intrinsics=CameraIntrinsics(50, 50, 25, 25),
        transform=transform,
        stamp_ns=123,
        camera_frame="optical",
        received_at=7.0,
        robot_spheres_torso=np.empty((0, 4)),
        target_mask=mask,
    )


def test_scene_reconstruction_separates_target_and_preserves_frames_and_age():
    import trimesh

    inputs = scene_inputs()
    # A robot sphere removes a corner of the table without erasing the target.
    inputs["robot_spheres_torso"] = np.array([[0.6, 2.4, 1.0, 0.05]])
    estimate = estimate_scene(**inputs, config=TABLE_MESHED)
    assert estimate.received_at == 7.0 and estimate.stamp_ns == 123
    assert estimate.frame_id == "torso_link" and not estimate.calibration_verified
    assert estimate.diagnostics["self_filtered_points"] > 0
    assert set(estimate.scene["mesh"]) == {"environment", "target"}
    counts = estimate.diagnostics["meshes"]
    assert counts["target"]["points"] == 16
    assert sum(x["points"] for x in counts.values()) + estimate.diagnostics["self_filtered_points"] == 2500
    assert estimate.support.target_heights_m["median"] == pytest.approx(0.1)
    for name, mesh in estimate.scene["mesh"].items():
        geometry = trimesh.Trimesh(vertices=mesh["vertices"], faces=np.asarray(mesh["faces"]).reshape(-1, 3))
        assert geometry.is_watertight and geometry.is_winding_consistent
        assert geometry.volume > 0
        pose = mesh["pose"]
        rot = Rotation.from_quat(np.array(pose[3:])[[1, 2, 3, 0]]).as_matrix()
        vertices = np.asarray(mesh["vertices"]) @ rot.T + pose[:3]
        expected_z = 1.1 if name == "target" else 1.0
        assert np.max(np.abs(vertices[:, 2] - expected_z)) <= 0.031
        assert counts[name]["faces"] > 0


def test_coarse_scene_keeps_fine_target_and_restores_filtered_obstacles_in_validation():
    inputs = scene_inputs()
    inputs["robot_spheres_torso"] = np.array([[0.6, 2.4, 1.0, 0.05]])
    estimate = estimate_scene(**inputs, config=TABLE_MESHED)
    fine = estimate_scene(**inputs, config=replace(TABLE_MESHED, mesh_pitch_m=0.005, self_filter_margin_m=0.01))
    assert estimate.diagnostics["planning_extra_filtered_points"] > 0
    assert estimate.diagnostics["meshes"]["environment"]["faces"] < fine.diagnostics["meshes"]["environment"]["faces"]
    assert estimate.diagnostics["validation_self_filtered_points"] == fine.diagnostics["self_filtered_points"]
    assert estimate.scene["cuboid"] == fine.scene["cuboid"]
    # The coarse scene adds the leg boxes; the fine one keeps only the table cuboid.
    assert estimate.validation_scene["cuboid"] == {"table": estimate.scene["cuboid"]["table"]}
    assert estimate.scene["mesh"]["target"] is estimate.validation_scene["mesh"]["target"]
    for name in ("target", "environment"):
        for field in ("vertices", "faces", "pose"):
            np.testing.assert_array_equal(
                estimate.validation_scene["mesh"][name][field], fine.scene["mesh"][name][field]
            )


def test_coarse_planning_target_keeps_fine_validation_target():
    inputs = scene_inputs()
    fine = estimate_scene(**inputs, config=TABLE_MESHED)
    coarse = estimate_scene(**inputs, config=replace(TABLE_MESHED, target_mesh_pitch_m=0.015))
    assert coarse.diagnostics["meshes"]["target"]["pitch_m"] == 0.015
    assert coarse.scene["mesh"]["target"] is not coarse.validation_scene["mesh"]["target"]
    for field in ("vertices", "faces", "pose"):
        np.testing.assert_array_equal(
            coarse.validation_scene["mesh"]["target"][field], fine.validation_scene["mesh"]["target"][field]
        )


def test_coarse_planning_target_uses_planning_self_filter_margin():
    inputs = scene_inputs()
    # Target points lie at z=1.1 around (1, 2); this sphere clears them by the
    # validation margin but not all of them by the planning margin.
    inputs["robot_spheres_torso"] = np.array([[1.0, 2.0, 1.15, 0.005]])
    fine = estimate_scene(**inputs, config=TABLE_MESHED)
    coarse = estimate_scene(**inputs, config=replace(TABLE_MESHED, target_mesh_pitch_m=0.015))
    assert fine.diagnostics["meshes"]["target"]["points"] == 16
    assert 0 < coarse.diagnostics["meshes"]["target"]["points"] < 16
    np.testing.assert_array_equal(
        coarse.validation_scene["mesh"]["target"]["faces"], fine.validation_scene["mesh"]["target"]["faces"]
    )


def test_scene_invalid_depth_keeps_target_pixel_correspondence():
    inputs = scene_inputs()
    inputs["depth"][0, :4] = [np.nan, np.inf, 0.0, 5.0]
    inputs["depth"][23, 23:25] = [0.0, np.nan]
    estimate = estimate_scene(**inputs, config=TABLE_MESHED)
    assert estimate.diagnostics["input_points"] == 2494
    assert estimate.diagnostics["meshes"]["target"]["points"] == 14
    assert estimate.diagnostics["meshes"]["environment"]["points"] == 2480


def test_every_fine_scene_carries_the_standing_leg_and_hip_in_torso_coordinates():
    import trimesh

    estimate = estimate_scene(**scene_inputs(), config=TABLE_MESHED)
    leg = estimate.validation_scene["mesh"][LEG_OBSTACLE]
    assert LEG_OBSTACLE not in estimate.scene["mesh"] and leg["pose"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    hulls = trimesh.Trimesh(vertices=leg["vertices"], faces=leg["faces"].reshape(-1, 3)).split(only_watertight=True)
    assert len(hulls) == 5

    def inside(point):
        return any(hull.contains([point])[0] for hull in hulls)

    # move_A1 (2026-10-15): the palm reached (0.05, 0.20, -0.21) behind the hip front;
    # the hanging hand at the start is outside, beside the hip.
    assert inside([0.03, 0.15, -0.20]) and not inside([0.165, 0.274, -0.175])
    unlegged = estimate_scene(**scene_inputs(), config=replace(TABLE_MESHED, leg_obstacle=False))
    assert LEG_OBSTACLE not in unlegged.validation_scene["mesh"]


def test_the_leg_as_shrunk_boxes_goes_to_the_coarse_scene_only():
    from aimdk_grasp.scene_estimation import leg_boxes

    def outside(boxes, point):
        """Signed distance (m, + outside) from the nearest box."""
        distances = []
        for box in boxes.values():
            w, x, y, z = box["pose"][3:]
            local = np.abs(Rotation.from_quat([x, y, z, w]).as_matrix().T @ (np.asarray(point) - box["pose"][:3]))
            local -= np.asarray(box["dims"]) / 2
            distances.append(np.linalg.norm(np.maximum(local, 0)) + min(local.max(), 0))
        return min(distances)

    boxes = leg_boxes(SceneEstimationConfig().coarse_leg_shrink_m)
    assert len(boxes) == 5 and all(min(box["dims"]) > 0 for box in boxes.values())
    # Live 2026-10-26: the thumb went 37-42 mm into the pelvis hull reaching mid-table; the
    # boxes hold those points, and the hanging hand at the start stays well clear of them.
    assert outside(boxes, [0.047, 0.035, -0.134]) < -0.02 and outside(boxes, [0.039, 0.014, -0.125]) < -0.02
    assert outside(boxes, [0.165, 0.274, -0.175]) > 0.1
    assert outside(leg_boxes(0.0), [0.047, 0.035, -0.134]) < outside(boxes, [0.047, 0.035, -0.134])
    estimate = estimate_scene(**scene_inputs(), config=TABLE_MESHED)
    assert set(estimate.scene["cuboid"]) == {"table", *boxes}
    # A box corner may stand out of its hull: the fine check keeps the exact hulls.
    assert set(estimate.validation_scene["cuboid"]) == {"table"} and LEG_OBSTACLE in estimate.validation_scene["mesh"]
    plain = estimate_scene(**scene_inputs(), config=replace(TABLE_MESHED, coarse_leg_boxes=False))
    assert set(plain.scene["cuboid"]) == {"table"}


def test_scene_without_target_mask_keeps_all_obstacles():
    inputs = scene_inputs()
    inputs["target_mask"] = None
    estimate = estimate_scene(**inputs, config=TABLE_MESHED)
    assert set(estimate.scene["mesh"]) == {"environment"}
    assert estimate.diagnostics["meshes"]["environment"]["points"] == 2500
    assert estimate.support.target_heights_m == {}


def test_support_component_selection_preserves_disconnected_environment_points():
    inputs = scene_inputs()
    # A missing strip separates the same-depth surface into two components.
    # The object is supported by the large component; the small one remains an
    # observed environment obstacle at both reconstruction resolutions.
    inputs["depth"][:, 7:13] = 0.0
    estimate = estimate_scene(**inputs, config=TABLE_MESHED, lazy_validation=False)
    for scene in (estimate.scene, estimate.validation_scene):
        environment = scene["mesh"]["environment"]
        vertices = np.asarray(environment["vertices"])
        assert vertices[:, 0].min() < 0.53  # remote component's torso X coordinate
    assert estimate.diagnostics["meshes"]["environment"]["points"] == 2184
    assert estimate.support.inliers[:, 0].min() >= 0.75


@pytest.mark.parametrize(
    "change",
    [
        {"stamp_ns": 124},
        {"camera_frame": "depth_optical"},
        {"received_at": np.nan},
        {"depth_scale": np.inf},
        {"intrinsics": CameraIntrinsics(np.nan, 50, 25, 25)},
        {"target_mask": np.zeros((2, 2), bool)},
    ],
)
def test_scene_rejects_mismatched_or_invalid_inputs(change):
    inputs = scene_inputs()
    inputs.update(change)
    with pytest.raises(ValueError):
        estimate_scene(**inputs, config=TABLE_MESHED)


def test_scene_rejects_empty_target_and_wrong_base():
    inputs = scene_inputs()
    inputs["target_mask"][:] = False
    with pytest.raises(RuntimeError, match="insufficient valid depth"):
        estimate_scene(**inputs, config=TABLE_MESHED)
    inputs["transform"] = replace(inputs["transform"], base_frame="map")
    with pytest.raises(ValueError, match="torso_link"):
        estimate_scene(**inputs, config=TABLE_MESHED)


def test_lazy_fine_scene_owns_inputs_and_builds_once_for_concurrent_readers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from aimdk_grasp import scene_estimation as module

    inputs = scene_inputs()
    eager = estimate_scene(**inputs, config=TABLE_MESHED, lazy_validation=False)
    original = module._surface_mesh
    calls = []

    def mesh(cloud, pitch, pose):
        calls.append(pitch)
        return original(cloud, pitch, pose)

    monkeypatch.setattr(module, "_surface_mesh", mesh)
    lazy = estimate_scene(**inputs, config=TABLE_MESHED)
    assert isinstance(lazy.validation_scene, module.LazyScene)
    assert not lazy.validation_scene.built
    assert calls == [0.03, 0.005]  # coarse environment and fine target only
    inputs["depth"][:] = 0.0
    inputs["transform"].base_from_camera[:] = 0.0
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: lazy.validation_scene.materialize(), range(4)))
    assert all(value is results[0] for value in results)
    assert calls == [0.03, 0.005, 0.005]
    assert lazy.received_at == 7.0 and lazy.stamp_ns == 123
    for name in ("environment", "target"):
        for field in ("vertices", "faces", "pose"):
            np.testing.assert_array_equal(results[0]["mesh"][name][field], eager.validation_scene["mesh"][name][field])


def test_lazy_fine_build_failure_cannot_return_partial_scene():
    from aimdk_grasp.scene_estimation import LazyScene

    def fail():
        raise ValueError("voxel build failed")

    scene = LazyScene(fail)
    with pytest.raises(ValueError, match="voxel"):
        scene.materialize()
    with pytest.raises(RuntimeError, match="construction failed"):
        scene["mesh"]
    assert not scene.built


def test_target_edge_grows_over_the_surface_but_not_the_table():
    from types import SimpleNamespace

    from aimdk_grasp.scene_estimation import SceneEstimationConfig, grow_target_edge

    plane = SimpleNamespace(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    column = np.array([[x, 0.0, z] for x in np.arange(0.0, 0.024, 0.003) for z in np.arange(0.0, 0.1, 0.003)])
    table = np.array([[x, y, 0.0] for x in np.arange(-0.03, 0.06, 0.003) for y in (-0.003, 0.003)])
    far = np.array([[0.3, 0.0, 0.05]])
    points = np.vstack([column, table, far])
    target = np.zeros(len(points), bool)
    target[: len(column)] = column[:, 0] < 0.012  # the mask stops short of the column's edge
    grown, moved = grow_target_edge(points, target, plane, SceneEstimationConfig())
    assert grown[: len(column)][column[:, 2] > 0.01].all()  # the rest of the surface joins
    assert not grown[len(column) :].any()  # the table and a far object stay
    assert moved == int((grown & ~target).sum())


def test_largest_cluster_mask_drops_a_disconnected_second_object_only():
    from aimdk_grasp.rgbd_to_cloud import largest_cluster_mask
    from aimdk_grasp.types import CameraIntrinsics

    intrinsics = CameraIntrinsics(500.0, 500.0, 40.0, 40.0)
    depth = np.full((80, 80), 2.0, dtype=np.float32)  # background
    mask = np.zeros((80, 80), bool)
    depth[20:50, 30:50] = 0.5  # target: 30 x 20 px at 0.5 m
    mask[20:50, 30:50] = True
    depth[35, 30:50] = 0.0  # a depth hole row inside the target
    depth[50:56, 30:50] = 1.2  # touching in the image, 0.7 m behind
    mask[50:56, 30:50] = True
    cleaned, removed = largest_cluster_mask(mask, depth, intrinsics)
    assert removed == 6 * 20
    assert cleaned[20:50, 30:50].all() and not cleaned[50:56].any()
    same, none = largest_cluster_mask(cleaned, depth, intrinsics)
    assert none == 0 and np.array_equal(same, cleaned)
