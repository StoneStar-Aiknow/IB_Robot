"""Depth/mask to point-cloud conversion, alignment and outlier removal (ROS-free)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .grasp_types import GraspDiagnostic

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def depth_and_segmentation_to_point_clouds(
    *,
    depth_image: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_object_id: int = 1,
    remove_object_from_scene: bool = False,
) -> tuple[np.ndarray, np.ndarray, None, None]:
    depth = np.asarray(depth_image)
    segmentation = np.asarray(segmentation_mask)
    if depth.ndim != 2 or segmentation.shape != depth.shape:
        raise ValueError(
            "depth_image and segmentation_mask must be matching 2D arrays, "
            f"got depth={depth.shape} segmentation={segmentation.shape}"
        )
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    object_pixels = segmentation == target_object_id
    if not object_pixels.any():
        raise ValueError(f"No points found for target object ID {target_object_id}")
    other_ids = np.unique(segmentation[(segmentation != 0) & ~object_pixels])
    if len(other_ids):
        raise ValueError(f"Multiple objects detected in segmentation mask: {other_ids}")

    z = depth.astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z > 0.0)
    ys_v, xs_v = np.where(valid)
    zv = z[ys_v, xs_v]
    xv = (xs_v - np.float32(cx)) * zv / np.float32(fx)
    yv = (ys_v - np.float32(cy)) * zv / np.float32(fy)
    xyz_v = np.stack((xv, yv, zv), axis=-1)
    object_at_valid = object_pixels[ys_v, xs_v]
    object_pc = xyz_v[object_at_valid]
    if len(object_pc) == 0:
        raise ValueError(f"No valid depth points found for target object ID {target_object_id}")
    if remove_object_from_scene:
        scene_pc = xyz_v[~object_at_valid]
    else:
        scene_pc = xyz_v
    return scene_pc, object_pc, None, None


def point_cloud_outlier_removal(
    object_pc: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.014,
    neighbor_count: int = 20,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    points = torch.as_tensor(object_pc, dtype=torch.float32).reshape(-1, 3)
    if len(points) <= 1:
        return points, points[:0]

    neighbors = min(max(1, int(neighbor_count)), len(points) - 1)
    keep_chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        distances = torch.cdist(points[start:stop], points, p=1)
        row_indices = torch.arange(stop - start)
        column_indices = torch.arange(start, stop)
        distances[row_indices, column_indices] = float("inf")
        nearest = torch.topk(distances, neighbors, dim=1, largest=False).values
        keep_chunks.append(nearest.mean(dim=1) < float(threshold))
    keep = torch.cat(keep_chunks)
    return points[keep], points[~keep]


def align_mask_to_depth(segmentation_mask: np.ndarray, depth_m: np.ndarray, diag: GraspDiagnostic) -> np.ndarray:
    mask = np.asarray(segmentation_mask)
    diag.mask_shape = _shape_string(mask)
    diag.depth_shape = _shape_string(depth_m)

    if depth_m.ndim != 2:
        raise ValueError(f"depth image must be 2-D, got shape {depth_m.shape}")
    if mask.ndim != 2:
        raise ValueError(f"segmentation mask must be 2-D, got shape {mask.shape}")

    if mask.shape == depth_m.shape:
        return mask

    import cv2

    diag.mask_resized = True
    return cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)


def populate_depth_mask_diagnostics(mask: np.ndarray, depth_m: np.ndarray, diag: GraspDiagnostic) -> None:
    mask_bool = mask > 0
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    valid_depth_in_mask = valid_depth & mask_bool

    diag.mask_pixel_count = int(mask_bool.sum())
    diag.valid_depth_pixel_count = int(valid_depth.sum())
    diag.valid_depth_in_mask_count = int(valid_depth_in_mask.sum())
    if diag.mask_pixel_count > 0:
        diag.valid_depth_ratio_in_mask = diag.valid_depth_in_mask_count / diag.mask_pixel_count


def downsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[np.sort(idx)]


def points_from_pixels(
    depth_m: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    zs = depth_m[ys, xs]
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    return np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)


def dilate_bool_grid(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        src = out.copy()
        expanded = src.copy()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if du == 0 and dv == 0:
                    continue
                src_u = slice(max(0, -du), src.shape[0] - max(0, du))
                dst_u = slice(max(0, du), src.shape[0] - max(0, -du))
                src_v = slice(max(0, -dv), src.shape[1] - max(0, dv))
                dst_v = slice(max(0, dv), src.shape[1] - max(0, -dv))
                expanded[dst_u, dst_v] |= src[src_u, src_v]
        out = expanded
    return out


def _shape_string(array: np.ndarray) -> str:
    return "x".join(str(dim) for dim in array.shape)
