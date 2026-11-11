"""Calibrated overlay for a fixed camera: table grid in world coordinates and the gripper tips.

The Agent reads object positions off the head camera. A bare 640x480 image gives it no
scale, so it guesses offsets in metres from pixels. With the camera's calibration (K and its
camera-to-world pose, which RoboDojo reports like any calibrated robot camera) this module
draws the table plane's world grid (labelled x / y lines) and each gripper tip with its
vertical drop onto the table. That is robot self-knowledge only: calibration plus
proprioception, never an object pose.

Pure numpy + PIL so it is unit-testable without ROS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Camera axes in its own frame. USD/Isaac cameras look along -Z with +Y up.
_AXES = {
    "usd": np.diag([1.0, -1.0, -1.0]),  # usd camera frame -> optical (x right, y down, z forward)
    "ros": np.eye(3),
}

ARM_COLOURS = {"left": (40, 120, 255), "right": (255, 60, 40)}


class CameraModel:
    def __init__(
        self,
        K: Sequence[Sequence[float]],
        cam_to_world: Sequence[Sequence[float]],
        resolution: Sequence[int],
        image_size: Sequence[int],
        axes: str = "usd",
    ):
        k = np.asarray(K, dtype=np.float64).copy()
        sx = float(image_size[0]) / float(resolution[0])
        sy = float(image_size[1]) / float(resolution[1])
        k[0] *= sx
        k[1] *= sy
        self.K = k
        t = np.asarray(cam_to_world, dtype=np.float64)
        self.R_wc = t[:3, :3]  # camera -> world rotation
        self.t_wc = t[:3, 3]
        self.axes = _AXES[axes]
        self.size = (int(image_size[0]), int(image_size[1]))

    @classmethod
    def from_calibration(cls, calib: Mapping, image_size: Sequence[int], axes: str = "usd") -> CameraModel:
        return cls(calib["K"], calib["cam_to_world"], calib["resolution"], image_size, axes)

    def project(self, points_world) -> np.ndarray:
        """``(N, 3)`` world points -> ``(N, 3)`` of ``u, v, depth``; depth <= 0 means behind the camera."""
        p = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        cam = (p - self.t_wc) @ self.R_wc  # world -> camera frame (R^T (p - t)), row vectors
        opt = cam @ self.axes.T
        z = opt[:, 2]
        safe = np.where(z > 1e-6, z, 1e-6)
        u = self.K[0, 0] * opt[:, 0] / safe + self.K[0, 2]
        v = self.K[1, 1] * opt[:, 1] / safe + self.K[1, 2]
        return np.stack([u, v, z], axis=1)


def _fmt(v: float) -> str:
    return f"{v:+.1f}".replace("+0.0", "0").replace("-0.0", "0")


def draw_overlay(
    rgb: np.ndarray,
    camera: CameraModel,
    *,
    table_z: float,
    x_range: Sequence[float] = (-0.6, 0.6),
    y_range: Sequence[float] = (-0.4, 0.4),
    step: float = 0.1,
    grippers: Mapping[str, Sequence[float]] | None = None,
) -> np.ndarray:
    """Return a copy of ``rgb`` (HxWx3 uint8) with the table grid and gripper markers drawn."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3]).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    xs = np.round(np.arange(x_range[0], x_range[1] + 1e-9, step), 3)
    ys = np.round(np.arange(y_range[0], y_range[1] + 1e-9, step), 3)
    w, h = camera.size

    def line(a, b, fill, width=1):
        uv = camera.project([a, b])
        if (uv[:, 2] > 0).all():
            draw.line([(uv[0, 0], uv[0, 1]), (uv[1, 0], uv[1, 1])], fill=fill, width=width)
        return uv

    grid = (255, 255, 0, 110)
    for x in xs:
        line((x, ys[0], table_z), (x, ys[-1], table_z), grid, 2 if abs(x) < 1e-9 else 1)
    for y in ys:
        line((xs[0], y, table_z), (xs[-1], y, table_z), grid, 2 if abs(y) < 1e-9 else 1)
    # Label the intersections themselves ("x,y" in metres) so a position can be read next to the
    # object instead of being traced back to the table's edge through the perspective.
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (i + j) % 2:
                continue
            u, v, depth = camera.project([(x, y, table_z)])[0]
            if depth <= 0 or not (0 <= u < w - 20 and 0 <= v < h - 8):
                continue
            draw.rectangle([u - 1, v - 1, u + 1, v + 1], fill=(255, 255, 0, 230))
            label = f"{_fmt(x)},{_fmt(y)}"
            draw.text((u + 2, v + 1), label, fill=(0, 0, 0, 200))
            draw.text((u + 1, v), label, fill=(255, 255, 0, 235))
    for side, pos in (grippers or {}).items():
        colour = ARM_COLOURS.get(side, (255, 255, 255))
        tip = (float(pos[0]), float(pos[1]), float(pos[2]))
        foot = (tip[0], tip[1], table_z)
        uv = camera.project([tip, foot])
        if not (uv[:, 2] > 0).all():
            continue
        draw.line([(uv[0, 0], uv[0, 1]), (uv[1, 0], uv[1, 1])], fill=(*colour, 200), width=1)
        r = 4
        draw.ellipse([uv[1, 0] - r, uv[1, 1] - r, uv[1, 0] + r, uv[1, 1] + r], outline=(*colour, 255), width=2)
        draw.ellipse([uv[0, 0] - 2, uv[0, 1] - 2, uv[0, 0] + 2, uv[0, 1] + 2], fill=(*colour, 255))
        draw.text((uv[1, 0] + 6, uv[1, 1] - 6), side[0].upper(), fill=(*colour, 255))
    return np.asarray(image)
