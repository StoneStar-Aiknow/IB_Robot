"""Pure mask-containment evaluation for placement verification (design D3).

Moved unchanged in body from ``place/node.py``; the names used by the node,
the offline replay or the tests lost their leading underscore. This module is
ROS-node-free, not strictly pure: ``load_exclusion_mask`` reads files and may
use its lazy ``ament_index_python`` lookup with the source-tree fallback. It
must not import ``rclpy`` or any ROS message package; ``place/replay.py``
imports only this module, never the node.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MaskContainment:
    inside: bool
    target_pixel_count: int
    inside_pixel_count: int
    inside_fraction: float
    center_inside: bool


def finite_positive(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def mask_to_bool(message) -> np.ndarray:
    """Decode a mono8 ROS image without depending on OpenCV."""
    encoding = str(message.encoding).lower()
    if encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"unsupported segmentation encoding: {message.encoding}")
    height = int(message.height)
    width = int(message.width)
    row_items = int(message.step)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if height <= 0 or width <= 0 or row_items < width or raw.size < height * row_items:
        raise ValueError("segmentation mask payload is inconsistent with its dimensions")
    return raw[: height * row_items].reshape((height, row_items))[:, :width] > 0


def _filled_container_region(mask: np.ndarray, inset_ratio: float) -> np.ndarray:
    """Fill the segmented container silhouette row-by-row and inset its edges."""
    source = np.asarray(mask, dtype=bool)
    region = np.zeros_like(source)
    rows = np.flatnonzero(np.any(source, axis=1))
    if rows.size == 0:
        return region
    y_min = int(rows[0])
    y_max = int(rows[-1])
    y_inset = int(round((y_max - y_min + 1) * float(inset_ratio)))
    for y in rows:
        if int(y) < y_min + y_inset or int(y) > y_max - y_inset:
            continue
        columns = np.flatnonzero(source[int(y)])
        if columns.size == 0:
            continue
        x_min = int(columns[0])
        x_max = int(columns[-1])
        x_inset = int(round((x_max - x_min + 1) * float(inset_ratio)))
        start = x_min + x_inset
        stop = x_max - x_inset + 1
        if start < stop:
            region[int(y), start:stop] = True
    return region


def normalized_polygon_mask(shape: tuple[int, int], polygons: object) -> np.ndarray:
    """Rasterize normalized ``[x, y]`` polygons into a boolean image mask.

    The placement camera can publish a different resolution on different robots.  Keeping
    the exclusion geometry normalized makes the same SSOT configuration valid for both
    640x360 PC frames and the 1280x720 310P frames.
    """
    height, width = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (0, 0)
    if height <= 0 or width <= 0:
        raise ValueError("exclusion mask shape must contain positive height and width")
    if not isinstance(polygons, list):
        raise ValueError("exclusion mask polygons must be a list")
    result = np.zeros((height, width), dtype=bool)
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid += 0.5
    y_grid += 0.5
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("each exclusion mask polygon must contain at least three points")
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("exclusion mask polygon points must be finite [x, y] pairs")
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("exclusion mask polygon points must be normalized to [0, 1]")
        points[:, 0] *= width
        points[:, 1] *= height
        inside = np.zeros((height, width), dtype=bool)
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            if y1 == y2:
                continue
            crosses = (y1 > y_grid) != (y2 > y_grid)
            x_intersection = (x2 - x1) * (y_grid - y1) / (y2 - y1) + x1
            inside ^= crosses & (x_grid < x_intersection)
        result |= inside
    return result


def load_exclusion_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    """Load a mono mask file and resize it to the current RGB frame."""
    configured_path = str(path).strip()
    configured_path = re.sub(r"\$\(env\s+(\w+)\)", lambda match: os.environ.get(match.group(1), ""), configured_path)
    if "$(find robot_config)" in configured_path:
        try:
            from ament_index_python.packages import get_package_share_directory

            configured_path = configured_path.replace(
                "$(find robot_config)", get_package_share_directory("robot_config")
            )
        except (ImportError, KeyError, RuntimeError):
            # Source-tree fallback: this module lives at
            # src/manipulation_execution/manipulation_execution/place/containment.py,
            # so the workspace root is three levels up (design D3 of the
            # package-layout change; was parents[2] in the flat layout).
            configured_path = configured_path.replace(
                "$(find robot_config)", str(Path(__file__).resolve().parents[3] / "robot_config")
            )
    mask_path = Path(configured_path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"target exclusion mask not found: {mask_path}")
    try:
        from PIL import Image as PILImage

        source = np.asarray(PILImage.open(mask_path).convert("L"), dtype=np.uint8)
    except ImportError:
        import cv2

        source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError(f"unable to read target exclusion mask: {mask_path}") from None
    if source.ndim != 2 or source.size == 0:
        raise ValueError(f"target exclusion mask must be a non-empty grayscale image: {mask_path}")
    height, width = int(shape[0]), int(shape[1])
    if source.shape != (height, width):
        try:
            from PIL import Image as PILImage

            source = np.asarray(
                PILImage.fromarray(source).resize((width, height), PILImage.Resampling.NEAREST), dtype=np.uint8
            )
        except ImportError:
            import cv2

            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
    return source > 0


def bbox_exclusion_overlap(bbox: object, exclusion_mask: np.ndarray) -> float:
    """Return the fraction of a detection bbox covered by an exclusion mask."""
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if values.size != 4:
        return 0.0
    height, width = exclusion_mask.shape
    x1, y1, x2, y2 = values
    left = max(0, min(width, int(math.floor(float(min(x1, x2))))))
    right = max(0, min(width, int(math.ceil(float(max(x1, x2))))))
    top = max(0, min(height, int(math.floor(float(min(y1, y2))))))
    bottom = max(0, min(height, int(math.ceil(float(max(y1, y2))))))
    if right <= left or bottom <= top:
        return 0.0
    area = (right - left) * (bottom - top)
    return float(np.count_nonzero(exclusion_mask[top:bottom, left:right])) / float(area)


def evaluate_mask_containment(
    container_mask,
    target_mask,
    *,
    min_target_pixels: int,
    min_inside_fraction: float,
    container_inset_ratio: float,
) -> MaskContainment:
    """Evaluate a target mask against the filled 2-D container region."""
    container = mask_to_bool(container_mask)
    target = mask_to_bool(target_mask)
    return evaluate_mask_arrays(
        container,
        target,
        min_target_pixels=min_target_pixels,
        min_inside_fraction=min_inside_fraction,
        container_inset_ratio=container_inset_ratio,
    )


def evaluate_mask_arrays(
    container_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    min_target_pixels: int,
    min_inside_fraction: float,
    container_inset_ratio: float,
) -> MaskContainment:
    """Evaluate replayable boolean masks without requiring live ROS messages."""
    container = np.asarray(container_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if container.shape != target.shape:
        raise ValueError("container and target masks have different dimensions")
    if container.ndim != 2 or target.ndim != 2:
        raise ValueError("container and target masks must be 2-D")
    target_pixel_count = int(np.count_nonzero(target))
    if target_pixel_count < int(min_target_pixels):
        return MaskContainment(False, target_pixel_count, 0, 0.0, False)
    region = _filled_container_region(container, container_inset_ratio)
    target_y, target_x = np.nonzero(target)
    center_y = int(round(float(np.mean(target_y))))
    center_x = int(round(float(np.mean(target_x))))
    center_y = min(max(center_y, 0), region.shape[0] - 1)
    center_x = min(max(center_x, 0), region.shape[1] - 1)
    center_inside = bool(region[center_y, center_x])
    inside_pixel_count = int(np.count_nonzero(target & region))
    inside_fraction = inside_pixel_count / target_pixel_count
    return MaskContainment(
        inside=center_inside and inside_fraction >= float(min_inside_fraction),
        target_pixel_count=target_pixel_count,
        inside_pixel_count=inside_pixel_count,
        inside_fraction=inside_fraction,
        center_inside=center_inside,
    )
