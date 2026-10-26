"""Geometric coverage, missing observations and foreground ordering of depth registration."""

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from aimdk_grasp import registration  # noqa: E402
from aimdk_grasp.registration import CameraCalibration, register_depth_footprints, register_rgbd  # noqa: E402

REGISTER = registration.register_depth_footprints


def require_cuda():
    warp.init()
    if not warp.get_cuda_device_count():
        pytest.skip("no CUDA device")


@pytest.fixture(autouse=True, params=["cpu", "cuda"])
def device(request, monkeypatch):
    """Every case runs the same kernel on Warp's CPU device and, if present, the GPU."""
    if request.param == "cuda":
        require_cuda()
    monkeypatch.setattr(
        registration,
        "register_depth_footprints",
        lambda *args, **options: REGISTER(*args, **{**options, "device": request.param}),
    )
    return request.param


def calibration(width, height, focal, frame):
    return CameraCalibration(
        width,
        height,
        np.array([[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        frame,
    )


def register(depth, rc, dc, matrix=None, **options):
    return register_rgbd(
        np.zeros((rc.height, rc.width, 3), dtype=np.uint8),
        depth,
        rc,
        dc,
        np.eye(4) if matrix is None else matrix,
        **options,
    ).depth


def test_magnified_plane_has_full_coverage_and_metric_depth():
    dc, rc = calibration(20, 12, 10.0, "depth"), calibration(40, 24, 20.0, "rgb")
    depth = np.full((12, 20), 1500, dtype=np.uint16)
    np.testing.assert_array_equal(register(depth, rc, dc, depth_scale=0.001), 1.5)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_missing_pixels_stay_missing_and_depth_steps_are_not_averaged(invalid):
    dc, rc = calibration(7, 5, 10.0, "depth"), calibration(14, 10, 20.0, "rgb")
    depth = np.full((5, 7), 2.0)
    depth[:, :3] = 0.5
    depth[1:4, 3:5] = invalid
    expected = depth.copy()
    expected[1:4, 3:5] = 0.0
    expected = np.repeat(np.repeat(expected, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(register(depth, rc, dc), expected)


def ray_oracle(depth, rc, dc, matrix):
    """Intersect each RGB ray with every source pixel's independent Z patch.

    Small analytic control, without projecting/rasterizing quads. Calibration is
    undistorted here. Avoid pixel-boundary ties in the fixtures.
    """
    v, u = np.indices((rc.height, rc.width))
    rays = np.stack(((u - rc.k[0, 2]) / rc.k[0, 0], (v - rc.k[1, 2]) / rc.k[1, 1], np.ones_like(u)), axis=-1)
    local_rays = rays @ matrix[:3, :3]
    local_translation = matrix[:3, :3].T @ matrix[:3, 3]
    nearest = np.full((rc.height, rc.width), np.inf)
    for y, x in zip(*np.nonzero(np.isfinite(depth) & (depth > 0)), strict=True):
        z = (depth[y, x] + local_translation[2]) / local_rays[:, :, 2]
        points = local_rays * z[:, :, None] - local_translation
        sx = points[:, :, 0] / points[:, :, 2] * dc.k[0, 0] + dc.k[0, 2]
        sy = points[:, :, 1] / points[:, :, 2] * dc.k[1, 1] + dc.k[1, 2]
        visible = (z > 0) & (np.abs(sx - x) <= 0.5) & (np.abs(sy - y) <= 0.5)
        nearest[visible] = np.minimum(nearest[visible], z[visible])
    return np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)


@pytest.mark.parametrize("angle", [0.0, 0.18, -0.21])
def test_rotation_translation_occlusion_and_depth_match_ray_intersections(angle):
    dc, rc = calibration(7, 5, 12.0, "depth"), calibration(24, 18, 36.0, "rgb")
    depth = np.full((5, 7), 2.0)
    depth[:, 2:4] = 0.5
    depth[2, 5] = 0.0
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    matrix[:3, 3] = [-0.035, 0.011, 0.023]
    expected = ray_oracle(depth, rc, dc, matrix)
    result = register(depth, rc, dc, matrix)
    np.testing.assert_allclose(result, expected, atol=2e-7, rtol=0.0)
    assert np.count_nonzero(result) > 0


def test_rotated_footprints_do_not_fill_their_axis_aligned_bounding_box():
    dc, rc = calibration(3, 3, 5.0, "depth"), calibration(15, 15, 25.0, "rgb")
    depth = np.zeros((3, 3))
    depth[1, 1] = 1.0
    matrix = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    result = register(depth, rc, dc, matrix)
    np.testing.assert_allclose(result, ray_oracle(depth, rc, dc, matrix), atol=1e-7)
    y, x = np.nonzero(result)
    assert (x.max() - x.min() + 1) * (y.max() - y.min() + 1) > len(x)


def test_subpixel_near_obstacle_is_retained_and_large_out_of_view_projection_is_ignored():
    dc, rc = calibration(4, 4, 100.0, "depth"), calibration(1, 1, 0.1, "rgb")
    depth = np.full((4, 4), 2.0)
    depth[0, 0] = 0.4
    assert register(depth, rc, dc)[0, 0] == pytest.approx(0.4)
    matrix = np.eye(4)
    matrix[0, 3] = 1e8
    assert not register(depth, rc, dc, matrix).any()


def test_distorted_source_coverage_uses_undistorted_pixel_corners():
    import cv2

    dc = CameraCalibration(
        20,
        16,
        np.array([[10.0, 0.0, 9.5], [0.0, 10.0, 7.5], [0.0, 0.0, 1.0]]),
        np.array([0.1, -0.02, 0.0, 0.0, 0.0]),
        "depth",
    )
    rc = calibration(30, 24, 25.0, "rgb")
    result = register(np.ones((16, 20)), rc, dc)
    np.testing.assert_array_equal(result, 1.0)
    # Independently verify that this RGB field of view lies within the raw sensor.
    v, u = np.indices(result.shape)
    points = np.stack(((u - rc.k[0, 2]) / rc.k[0, 0], (v - rc.k[1, 2]) / rc.k[1, 1], np.ones_like(u)), axis=-1)
    projected, _ = cv2.projectPoints(points.reshape(-1, 3), np.zeros(3), np.zeros(3), dc.k, dc.distortion)
    assert (projected.reshape(-1, 2) >= -0.5).all()
    assert (projected.reshape(-1, 2) <= [dc.width - 0.5, dc.height - 0.5]).all()


def test_calibration_cache_does_not_reuse_another_extrinsic_or_input_depth():
    dc, rc = calibration(5, 4, 10.0, "depth"), calibration(12, 10, 24.0, "rgb")
    depth = np.ones((4, 5))
    first = register(depth, rc, dc)
    angle = 0.15
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    changed = register(depth, rc, dc, matrix)
    np.testing.assert_allclose(changed, ray_oracle(depth, rc, dc, matrix), atol=2e-7)
    assert not np.array_equal(first, changed)
    depth[:] = 2.0
    np.testing.assert_array_equal(register(depth, rc, dc), 2.0)
    np.testing.assert_array_equal(first, 1.0)


def test_empty_and_behind_camera_samples_do_not_create_coverage():
    dc, rc = calibration(5, 4, 10.0, "depth"), calibration(12, 10, 24.0, "rgb")
    assert not register(np.zeros((4, 5)), rc, dc).any()
    matrix = np.eye(4)
    matrix[2, 3] = -2.0
    assert not register(np.ones((4, 5)), rc, dc, matrix).any()


def test_cuda_matches_cpu_on_a_rotated_noisy_scene(device):
    if device != "cuda":
        pytest.skip("compared once, from the CUDA parametrization")
    require_cuda()
    rng = np.random.default_rng(3)
    dc, rc = calibration(64, 48, 50.0, "depth"), calibration(80, 60, 62.0, "rgb")
    depth = rng.integers(400, 2500, size=(48, 64)).astype(np.uint16)
    depth[rng.random(depth.shape) < 0.1] = 0
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[np.cos(0.2), 0.0, np.sin(0.2)], [0.0, 1.0, 0.0], [-np.sin(0.2), 0.0, np.cos(0.2)]])
    matrix[:3, 3] = [0.03, -0.01, 0.005]
    cpu = register_depth_footprints(depth, rc, dc, matrix, 0.001, device="cpu")
    cuda = register_depth_footprints(depth, rc, dc, matrix, 0.001, device="cuda")
    assert (cpu > 0).sum() > 1000
    np.testing.assert_array_equal(cuda, cpu)
