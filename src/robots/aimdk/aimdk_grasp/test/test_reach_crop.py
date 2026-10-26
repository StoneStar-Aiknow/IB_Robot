"""Left-arm reach bound and the scene-estimation crop it drives."""

import numpy as np
import pytest
from aimdk_grasp import scene_estimation
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.hand import hand_links
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import planning_model
from aimdk_grasp.scene_estimation import SceneEstimationConfig, estimate_scene
from aimdk_grasp.types import CameraIntrinsics

#: The gripper's open preset (native order): fingers extended, so the sampled
#: spheres come close to the reach bound.
OPEN = np.array([-0.129, 1.54, -0.255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(autouse=True)
def unbounded_left_arm_reach(monkeypatch):
    """Synthetic scenes place their camera anywhere in torso_link, far from the
    real X2 left shoulder; tests of the crop patch left_arm_reach themselves."""
    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.zeros(3), np.inf))


def test_reach_is_an_upper_bound_on_every_sampled_left_arm_sphere():
    from aimdk_grasp.robot_spheres import RobotSphereModel

    model = RobotSphereModel()
    center, radius = model.left_arm_reach()
    assert np.allclose(center, [0.003, 0.1427, 0.2416], atol=1e-3) and 0.6 < radius < 0.8
    feedback = Feedback(0.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), OPEN)
    limits = np.array([joint["bounds"] for name in LEFT_ARM for joint in model.joints if joint["name"] == name])
    q = np.random.default_rng(0).uniform(limits[:, 0], limits[:, 1], (2000, len(LEFT_ARM)))
    spheres = np.asarray(model.spheres(feedback, q)[0])
    moving = spheres[:, :, :3].std(axis=0).max(axis=1) > 1e-6
    reached = np.linalg.norm(spheres[:, moving, :3] - center, axis=2) + spheres[:, moving, 3]
    assert moving.sum() > 100 and reached.max() <= radius
    assert reached.max() > radius - 0.05  # the bound is tight, not just large


def test_palm_reach_is_an_upper_bound_on_every_sampled_palm_origin():
    from aimdk_grasp.robot_spheres import RobotSphereModel

    model = RobotSphereModel()
    center, radius = model.left_palm_reach()
    np.testing.assert_allclose(center, model.left_arm_reach()[0])
    assert 0.45 < radius < model.left_arm_reach()[1]
    feedback = Feedback(0.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), OPEN)
    limits = np.array([joint["bounds"] for name in LEFT_ARM for joint in model.joints if joint["name"] == name])
    q = np.random.default_rng(0).uniform(limits[:, 0], limits[:, 1], (2000, len(LEFT_ARM)))
    model.spheres(feedback, q)
    palm = model.last_link_frames[:, model.links.index("R_palm"), :, 3]
    reached = np.linalg.norm(palm - center, axis=1)
    assert reached.max() <= radius and reached.max() > radius - 0.1


def inputs():
    # Camera 1 m above torso origin looking down (+z optical = -z torso). Left
    # half: table 1 m below (near field) with a 4x4 target; right half: a floor
    # 1.5 m below (beyond support_max_depth_m).
    depth = np.ones((60, 60), dtype=np.float32)
    depth[:, 45:] = 1.5
    mask = np.zeros_like(depth, dtype=bool)
    mask[28:32, 20:24] = True
    depth[mask] = 0.9
    matrix = np.diag([1.0, -1.0, -1.0, 1.0])
    matrix[:3, 3] = [0.0, 0.0, 1.0]
    return dict(
        depth=depth,
        intrinsics=CameraIntrinsics(30, 30, 30, 30),
        transform=ImageTransform(5, "optical", "torso_link", matrix),
        stamp_ns=5,
        camera_frame="optical",
        received_at=0.0,
        robot_spheres_torso=np.empty((0, 4)),
        target_mask=mask,
    )


def mesh_xyz(scene, name):
    from scipy.spatial.transform import Rotation

    mesh = scene["mesh"][name]
    rotation = Rotation.from_quat(np.asarray(mesh["pose"][3:])[[1, 2, 3, 0]]).as_matrix()
    return np.asarray(mesh["vertices"]) @ rotation.T + mesh["pose"][:3]


def mesh_xy(scene, name):
    return mesh_xyz(scene, name)[:, :2]


def test_crop_limits_environment_mesh_but_not_support_fit_or_target(monkeypatch):
    # Reach sphere (0.3 m + 0.1 m margin) around a table point left of the target.
    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.array([-0.2, 0.0, 0.0]), 0.3))
    cropped = estimate_scene(**inputs())
    full = estimate_scene(**inputs(), config=SceneEstimationConfig(crop_to_left_arm_reach=False))
    d = cropped.diagnostics
    # Far floor points are dropped before the self filter; the near-field table is
    # still filtered and fitted but only its reachable part is meshed.
    assert d["depth_points"] == full.diagnostics["input_points"] == 3600
    assert d["reach_cropped_points"] == 15 * 60 and d["unreachable_near_points"] > 0
    assert cropped.support.point == pytest.approx(full.support.point)
    assert cropped.support.normal == pytest.approx(full.support.normal)
    assert len(cropped.support.inliers) == len(full.support.inliers)
    assert d["target_edge_grown_points"] == full.diagnostics["target_edge_grown_points"]
    assert d["meshes"]["target"] == full.diagnostics["meshes"]["target"]
    # The table itself lives in the fine mesh (the coarse one leaves it to the
    # cuboid). Points lie within reach + margin; voxel corners may extend one
    # voxel diagonal beyond them.
    voxel_diagonal = np.sqrt(3) * SceneEstimationConfig().mesh_pitch_m
    fine, fine_full = cropped.validation_scene.materialize(), full.validation_scene.materialize()
    assert np.linalg.norm(mesh_xy(fine, "environment") - [-0.2, 0.0], axis=1).max() <= 0.4 + voxel_diagonal
    assert np.linalg.norm(mesh_xy(fine_full, "environment") - [-0.2, 0.0], axis=1).max() > 0.6


def test_table_is_left_to_the_cuboid_in_the_planning_mesh_only():
    # A 6x6-pixel box standing 5 cm on the table away from the target stays an obstacle.
    values = inputs()
    values["depth"][10:16, 10:16] = 0.95
    estimate = estimate_scene(**values)
    d = estimate.diagnostics
    assert d["planning_table_points"] > 2000 and d["self_filtered_points"] == 0
    coarse = mesh_xyz(estimate.scene, "environment")
    coarse = coarse[coarse[:, 2] > -0.2][:, :2]  # table level (floor is at -0.5 m)
    box_center = np.array([(12.5 - 30) / 30, -(12.5 - 30) / 30]) * 0.95  # pixel centre -> torso xy at 0.95 m
    distance = np.linalg.norm(coarse - box_center, axis=1)
    assert distance.min() < 0.1  # the box is still meshed ...
    assert distance.max() < 0.2  # ... and nothing else at table level: the surface is left to the cuboid
    fine = estimate.validation_scene.materialize()
    fine_xyz = mesh_xyz(fine, "environment")
    assert (fine_xyz[:, 2] > -0.2).sum() > 10 * len(coarse)  # the fine check keeps the table
    # Every point left to the cuboid really is inside it.
    table = estimate.support
    assert table.contains(np.array([table.point]), SceneEstimationConfig())
    assert not table.contains(np.array([table.point + 0.2 * table.normal]), SceneEstimationConfig())


def test_target_outside_reach_is_still_kept(monkeypatch):
    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.array([5.0, 5.0, 5.0]), 0.1))
    estimate = estimate_scene(**inputs())
    full = estimate_scene(**inputs(), config=SceneEstimationConfig(crop_to_left_arm_reach=False))
    # The crop never deletes the object being grasped; nothing else is reachable.
    assert estimate.diagnostics["meshes"]["target"] == full.diagnostics["meshes"]["target"]
    assert "environment" not in estimate.scene["mesh"]
    assert len(estimate.validation_scene.materialize()["mesh"].get("environment", {}).get("vertices", [])) == 0


def test_config_rejects_negative_margin():
    with pytest.raises(ValueError):
        SceneEstimationConfig(reach_margin_m=-0.1)


def test_voxel_lattice_does_not_depend_on_far_points():
    from aimdk_grasp.scene_estimation import _voxel_cells

    near = np.random.default_rng(2).uniform([0.4, -0.1, 0.1], [0.6, 0.1, 0.3], (500, 3))
    far = np.array([[-1.234, 2.718, -0.577]])
    for pitch in (0.03, 0.005):
        origin, cells = _voxel_cells(near, pitch)
        origin_far, cells_far = _voxel_cells(np.vstack((near, far)), pitch)
        # Same world-space voxel for every near point whatever else the cloud holds.
        assert np.allclose(origin + cells * pitch, origin_far + cells_far[:-1] * pitch)
        assert np.allclose(np.round(origin / pitch), origin / pitch)


def test_merged_optimizer_hand_contains_the_original_cover():
    from aimdk_grasp.robot_model import collision_robot_config

    hand = set(hand_links(planning_model()))
    original = collision_robot_config()["robot_cfg"]["kinematics"]["collision_spheres"]
    merged = collision_robot_config(hand_merge_radius_m=0.02)["robot_cfg"]["kinematics"]["collision_spheres"]
    assert (
        sum(len(merged[link]) for link in hand & set(merged))
        < sum(len(original[link]) for link in hand & set(original)) / 4
    )
    for link, spheres in original.items():
        if link not in hand:
            assert merged[link] == spheres
            continue
        balls = np.array([[*s["center"], s["radius"]] for s in merged[link]])
        for sphere in spheres:
            # Some merged ball contains this original ball entirely.
            reach = np.linalg.norm(balls[:, :3] - sphere["center"], axis=1) + sphere["radius"]
            assert np.any(reach <= balls[:, 3] + 1e-9), link


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_self_filter_broad_phase_matches_the_exact_test(seed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU path")
    from aimdk_grasp.scene_estimation import _sphere_membership_cpu, _sphere_membership_many

    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.4, 0.4, (300, 3))
    radii = np.stack([rng.uniform(0.005, 0.05, 300), rng.uniform(0.005, 0.05, 300) + 0.04])
    # Points far away, near and exactly on sphere surfaces (both margin sets).
    surface = centers[rng.integers(0, 300, 4000)]
    direction = rng.normal(size=(4000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    on = surface + direction * radii[rng.integers(0, 2, 4000), rng.integers(0, 300, 4000)][:, None]
    points = np.concatenate([rng.uniform(-1.5, 1.5, (20000, 3)), on, rng.uniform(-0.45, 0.45, (8000, 3))])
    result = _sphere_membership_many(points, centers, radii)
    # The previous GPU path: float64 cdist of every point to every sphere.
    # (It and the CPU path can differ in the last bit for points exactly on a
    # surface, so the reference is the GPU computation the broad phase skips.)
    device = torch.device("cuda")
    distances = torch.cdist(
        torch.as_tensor(points, dtype=torch.float64, device=device),
        torch.as_tensor(centers, dtype=torch.float64, device=device),
    )
    for index in range(2):
        brute = ~(distances <= torch.as_tensor(radii[index], dtype=torch.float64, device=device)).any(dim=1)
        np.testing.assert_array_equal(result[index], brute.cpu().numpy())
    cpu = _sphere_membership_cpu(points, centers, radii[1])
    assert (result[1] != cpu).sum() <= 10 and result.any() and not result.all()


def _split_scene(values, config):
    prepared = scene_estimation.prepare_scene(
        values["depth"],
        values["intrinsics"],
        values["transform"],
        stamp_ns=values["stamp_ns"],
        camera_frame=values["camera_frame"],
        received_at=values["received_at"],
        robot_spheres_torso=values["robot_spheres_torso"],
        config=config,
    )
    return scene_estimation.complete_scene(prepared, values["target_mask"])


def _assert_same_estimate(actual, expected):
    assert actual.received_at == expected.received_at
    assert (actual.stamp_ns, actual.camera_frame, actual.calibration_verified) == (
        expected.stamp_ns,
        expected.camera_frame,
        expected.calibration_verified,
    )
    assert actual.support.point == pytest.approx(expected.support.point)
    assert actual.support.normal == pytest.approx(expected.support.normal)
    np.testing.assert_allclose(actual.support.inliers, expected.support.inliers)
    assert actual.support.target_heights_m == pytest.approx(expected.support.target_heights_m)
    for key, value in expected.diagnostics.items():
        if key != "timings_s":
            assert actual.diagnostics[key] == value, key
    assert set(actual.diagnostics["timings_s"]) == set(expected.diagnostics["timings_s"])
    assert actual.scene["cuboid"] == expected.scene["cuboid"]
    assert set(actual.scene["mesh"]) == set(expected.scene["mesh"])
    for name, mesh in expected.scene["mesh"].items():
        assert actual.scene["mesh"][name]["pose"] == mesh["pose"]
        np.testing.assert_array_equal(actual.scene["mesh"][name]["vertices"], mesh["vertices"])
        np.testing.assert_array_equal(actual.scene["mesh"][name]["faces"], mesh["faces"])
    np.testing.assert_array_equal(
        actual.validation_scene.materialize()["mesh"]["environment"]["vertices"],
        expected.validation_scene.materialize()["mesh"]["environment"]["vertices"],
    )


def test_prepare_complete_is_estimate_scene(monkeypatch):
    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.array([-0.2, 0.0, 0.0]), 0.3))
    values = inputs()
    # One sphere clips the target and the table (self-filter runs on both
    # stages); one sits far away. Camera coordinates: x right, y down, depth z.
    values["robot_spheres_torso"] = np.array([[-0.15, -0.15, 0.25, 0.2], [0.9, 0.9, 0.9, 0.1]])
    _assert_same_estimate(_split_scene(values, SceneEstimationConfig()), estimate_scene(**values))
    _assert_same_estimate(
        _split_scene(values, SceneEstimationConfig(crop_to_left_arm_reach=False)),
        estimate_scene(**values, config=SceneEstimationConfig(crop_to_left_arm_reach=False)),
    )


def test_prepare_complete_recovers_target_pixels_the_crop_dropped(monkeypatch):
    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.array([-0.2, 0.0, 0.0]), 0.3))
    values = inputs()
    values["robot_spheres_torso"] = np.array([[-0.15, -0.15, 0.25, 0.2]])
    # Target pixels on the far floor are beyond the reach sphere and the near
    # field: prepare_scene drops them, complete_scene must bring them back in
    # the original pixel order (estimate_scene kept them via `| target`).
    mask = values["target_mask"].copy()
    mask[40:44, 50:54] = True
    values["target_mask"] = mask

    def attempt(callable):
        try:
            return ("ok", callable())
        except Exception as exc:
            return ("error", type(exc).__name__, str(exc))

    assert attempt(lambda: _split_scene(values, SceneEstimationConfig())) == attempt(lambda: estimate_scene(**values))
    prepared = scene_estimation.prepare_scene(
        values["depth"],
        values["intrinsics"],
        values["transform"],
        stamp_ns=values["stamp_ns"],
        camera_frame=values["camera_frame"],
        received_at=values["received_at"],
        robot_spheres_torso=values["robot_spheres_torso"],
    )
    assert len(prepared.dropped_pixel) and values["target_mask"].reshape(-1)[prepared.dropped_pixel].any()
