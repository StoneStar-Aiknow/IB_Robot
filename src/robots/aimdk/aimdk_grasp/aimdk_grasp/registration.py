"""Register head RGB-D frames: rectify RGB and rasterize depth into it on the GPU.

Each valid depth sample represents its own half-pixel footprint at constant
source Z. No surface joins different depth samples. A nearest-surface z-buffer
resolves overlaps; original sensor holes remain unobserved. Subpixel footprints
retain a nearest-center sample so minification cannot silently discard thin
obstacles. The kernel runs on NVIDIA Warp: on the planning host's GPU in
production (``device="cuda"``), or on Warp's CPU device where no GPU exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import warp as wp

from aimdk_grasp.frames import rigid
from aimdk_grasp.types import CameraIntrinsics, RGBDFrame

REGISTRATION_METHOD = "depth_pixel_footprint_v1"


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    k: np.ndarray
    distortion: np.ndarray
    frame_id: str
    distortion_model: str = "plumb_bob"

    def __post_init__(self):
        k = np.array(self.k, dtype=np.float64, copy=True).reshape(3, 3)
        d = np.array(self.distortion, dtype=np.float64, copy=True).reshape(-1)
        if (
            min(self.width, self.height) <= 0
            or not self.frame_id
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0.0, 0.0, 1.0])
            or not np.isfinite(d).all()
        ):
            raise ValueError("Invalid camera calibration")
        if self.distortion_model not in ("", "plumb_bob") or len(d) not in (0, 4, 5, 8, 12, 14):
            raise ValueError("Only OpenCV plumb_bob calibration is supported")
        k.setflags(write=False)
        d.setflags(write=False)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "distortion", d)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(self.k[0, 0], self.k[1, 1], self.k[0, 2], self.k[1, 2])


def calibration_signature(rgb: CameraCalibration, depth: CameraCalibration, rgb_from_depth) -> str:
    digest = hashlib.sha256()
    digest.update(REGISTRATION_METHOD.encode())
    for calibration in (rgb, depth):
        digest.update(
            repr((calibration.width, calibration.height, calibration.frame_id, calibration.distortion_model)).encode()
        )
        digest.update(calibration.k.tobytes())
        digest.update(calibration.distortion.tobytes())
    digest.update(rigid(rgb_from_depth).astype(np.float64).tobytes())
    return digest.hexdigest()


def register_rgbd(
    rgb, depth, rgb_calibration, depth_calibration, rgb_from_depth, *, depth_scale=1.0, device="cuda"
) -> RGBDFrame:
    """Undistort RGB and rasterize the depth pixel footprints into it.

    Already rectified/aligned inputs use zero distortion and the identity
    transform. Different RGB and depth resolutions are supported.
    """
    import cv2

    rc, dc = rgb_calibration, depth_calibration
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.shape != (rc.height, rc.width, 3) or rgb.dtype != np.uint8:
        raise ValueError("uint8 RGB dimensions must match its calibration")
    if depth.shape != (dc.height, dc.width) or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("Depth dimensions and positive scale must match its calibration")
    matrix = rigid(rgb_from_depth)
    rectified = None
    if rc.distortion.size and np.any(rc.distortion):
        maps = _undistort_maps(rc.width, rc.height, rc.k.tobytes(), rc.distortion.astype(np.float64).tobytes())
        # Same maps and interpolation as cv2.undistort (identical output, tested),
        # which rebuilt them on every frame: 60-100 ms per capture on the Orin.
        # cv2 releases the GIL: the RGB remap (13 ms on the Orin CPU) runs
        # while the depth is registered (CUDA kernel, 30-40 ms).
        rectified = _remap_pool().submit(cv2.remap, rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    registered = register_depth_footprints(depth, rc, dc, matrix, depth_scale, device=device)
    rectified = rgb.copy() if rectified is None else rectified.result()
    return RGBDFrame(rectified, registered, rc.intrinsics, rc.frame_id)


@lru_cache(maxsize=1)
def _remap_pool():
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(1, thread_name_prefix="rgb_remap")


@lru_cache(maxsize=4)
def _undistort_maps(width, height, k_bytes, distortion_bytes):
    import cv2

    k = np.frombuffer(k_bytes, dtype=np.float64).reshape(3, 3)
    distortion = np.frombuffer(distortion_bytes, dtype=np.float64)
    return cv2.initUndistortRectifyMap(k, distortion, None, k, (width, height), cv2.CV_16SC2)


def register_depth_footprints(depth, rgb_calibration, depth_calibration, matrix, depth_scale, *, device="cuda"):
    """Depth in metres on the RGB pixel grid; 0 where nothing was observed."""
    rc, dc = rgb_calibration, depth_calibration
    scaled = depth.astype(np.float64) * depth_scale
    if (
        depth.shape == (rc.height, rc.width)
        and np.array_equal(rc.k, dc.k)
        and not np.any(dc.distortion)
        and np.array_equal(matrix, np.eye(4))
    ):
        valid = np.isfinite(scaled) & (scaled > 0) & (scaled <= np.finfo(np.float32).max)
        return np.where(valid, scaled, 0.0).astype(np.float32)
    centers, corners = _rays(
        dc.width, dc.height, tuple(dc.k.flat), tuple(dc.distortion), tuple(matrix[:3, :3].flat), device
    )
    registered = wp.full((rc.height, rc.width), np.inf, dtype=wp.float32, device=device)
    wp.launch(
        _rasterize,
        dim=scaled.shape,
        inputs=[
            wp.array(np.ascontiguousarray(scaled), dtype=wp.float64, device=device),
            centers,
            corners,
            wp.vec3d(*matrix[:3, 3]),
            wp.vec3d(*matrix[:3, 2]),
            wp.vec4d(rc.k[0, 0], rc.k[1, 1], rc.k[0, 2], rc.k[1, 2]),
            rc.width,
            rc.height,
            registered,
        ],
        device=device,
    )
    result = registered.numpy()
    result[~np.isfinite(result)] = 0.0
    return result


@lru_cache(maxsize=2)
def _rotated_rays(width, height, intrinsic, distortion, rotation):
    import cv2

    k = np.asarray(intrinsic).reshape(3, 3)
    d = np.asarray(distortion)
    matrix = np.asarray(rotation).reshape(3, 3)
    result = []
    for corners in (False, True):
        v, u = np.indices((height + corners, width + corners), dtype=np.float64)
        pixels = np.stack((u, v), axis=-1) - (0.5 if corners else 0.0)
        if d.size and np.any(d):
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, d).reshape(pixels.shape)
        else:
            xy = (pixels - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
        rays = np.concatenate((xy, np.ones((*xy.shape[:2], 1))), axis=-1)
        rays = np.einsum("...j,ij->...i", rays, matrix)
        rays.setflags(write=False)
        result.append(rays)
    return tuple(result)


@lru_cache(maxsize=4)
def _rays(width, height, intrinsic, distortion, rotation, device):
    return tuple(
        wp.array(np.ascontiguousarray(rays), dtype=wp.float64, device=device)
        for rays in _rotated_rays(width, height, intrinsic, distortion, rotation)
    )


Quad = wp.types.matrix(shape=(4, 2), dtype=wp.float64)


@wp.kernel
def _rasterize(
    depth: wp.array2d(dtype=wp.float64),
    centers: wp.array3d(dtype=wp.float64),
    corners: wp.array3d(dtype=wp.float64),
    translation: wp.vec3d,
    normal: wp.vec3d,
    intrinsic: wp.vec4d,
    width: int,
    height: int,
    output: wp.array2d(dtype=wp.float32),
):
    y, x = wp.tid()
    z = depth[y, x]
    if not wp.isfinite(z) or z <= wp.float64(0.0) or z > wp.float64(3.4028234663852886e38):
        return
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    vertices = Quad()
    # bool(...) declares mutable kernel locals; a bare literal is a Warp constant.
    front = bool(True)  # noqa: UP018
    min_x, min_y = wp.float64(1e300), wp.float64(1e300)
    max_x, max_y = wp.float64(-1e300), wp.float64(-1e300)
    offsets_x, offsets_y = wp.vec4i(0, 1, 1, 0), wp.vec4i(0, 0, 1, 1)
    for i in range(4):
        sy, sx = y + offsets_y[i], x + offsets_x[i]
        p = wp.vec3d(corners[sy, sx, 0], corners[sy, sx, 1], corners[sy, sx, 2]) * z + translation
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]) or p[2] <= wp.float64(1e-9):
            front = False
        else:
            projected_u, projected_v = p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy
            vertices[i, 0] = projected_u
            vertices[i, 1] = projected_v
            min_x, min_y = wp.min(min_x, projected_u), wp.min(min_y, projected_v)
            max_x, max_y = wp.max(max_x, projected_u), wp.max(max_y, projected_v)
    covered = bool(False)  # noqa: UP018
    if front:
        # Clamp before converting to integers, including near-plane projections.
        x0 = int(wp.ceil(wp.clamp(min_x - wp.float64(1e-9), wp.float64(0.0), wp.float64(width))))
        y0 = int(wp.ceil(wp.clamp(min_y - wp.float64(1e-9), wp.float64(0.0), wp.float64(height))))
        x1 = int(wp.floor(wp.clamp(max_x + wp.float64(1e-9), wp.float64(-1.0), wp.float64(width - 1))))
        y1 = int(wp.floor(wp.clamp(max_y + wp.float64(1e-9), wp.float64(-1.0), wp.float64(height - 1))))
        # The transformed source-Z plane is n.p = source_z + n.translation.
        nx, ny = normal[0] / fx, normal[1] / fy
        nz = normal[2] - nx * cx - ny * cy
        offset = wp.dot(normal, translation)
        for v in range(y0, y1 + 1):
            for u in range(x0, x1 + 1):
                positive, negative = bool(True), bool(True)  # noqa: UP018
                for i in range(4):
                    j = (i + 1) % 4
                    ex, ey = vertices[j, 0] - vertices[i, 0], vertices[j, 1] - vertices[i, 1]
                    cross = ex * (wp.float64(v) - vertices[i, 1]) - ey * (wp.float64(u) - vertices[i, 0])
                    positive = positive and cross >= wp.float64(-1e-9)
                    negative = negative and cross <= wp.float64(1e-9)
                denominator = nx * wp.float64(u) + ny * wp.float64(v) + nz
                if (positive or negative) and wp.abs(denominator) > wp.float64(1e-12):
                    value = (z + offset) / denominator
                    if wp.isfinite(value) and value > wp.float64(0.0) and value <= wp.float64(3.4028234663852886e38):
                        wp.atomic_min(output, v, u, wp.float32(value))
                        covered = True
    if not covered:
        # A footprint crossing the camera plane has no bounded projection;
        # retain only its measured center instead of expanding it to infinity.
        p = wp.vec3d(centers[y, x, 0], centers[y, x, 1], centers[y, x, 2]) * z + translation
        if wp.isfinite(p[2]) and p[2] > wp.float64(0.0) and p[2] <= wp.float64(3.4028234663852886e38):
            fallback_u, fallback_v = wp.rint(p[0] / p[2] * fx + cx), wp.rint(p[1] / p[2] * fy + cy)
            if (
                fallback_u >= wp.float64(0.0)
                and fallback_v >= wp.float64(0.0)
                and fallback_u < wp.float64(width)
                and fallback_v < wp.float64(height)
            ):
                wp.atomic_min(output, int(fallback_v), int(fallback_u), wp.float32(p[2]))
