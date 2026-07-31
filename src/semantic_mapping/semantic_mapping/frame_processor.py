"""Shared validation and deterministic filtering for RGB-D semantic frames."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PreparedFrame:
    image_bgr: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    depth_scale: float
    stamp_ns: int
    camera_frame: str
    translation: np.ndarray
    rotation: np.ndarray
    valid_depth_ratio: float


@dataclass(frozen=True)
class MaskCandidate:
    mask: np.ndarray
    score: float


@dataclass
class MaskFilterDiagnostics:
    input_count: int = 0
    accepted_count: int = 0
    rejected_invalid: int = 0
    rejected_too_small: int = 0
    rejected_depth: int = 0
    rejected_overlap: int = 0
    rejected_limit: int = 0
    accepted_indices: list[int] = field(default_factory=list)


def prepare_frame(
    *,
    image_bgr: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    rgb_stamp_ns: int,
    depth_stamp_ns: int,
    info_stamp_ns: int,
    camera_frame: str,
    translation: np.ndarray,
    rotation: np.ndarray,
    max_stamp_skew_ns: int,
    depth_trunc_m: float,
    min_valid_depth_ratio: float = 0.0,
) -> PreparedFrame:
    """Validate an aligned RGB-D frame and its timestamped camera transform."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("RGB image must be a bgr8 uint8 HxWx3 array")
    if depth.ndim != 2 or image_bgr.shape[:2] != depth.shape:
        raise ValueError("RGB and aligned depth dimensions differ")
    if not camera_frame:
        raise ValueError("synchronized RGB-D frame has no camera frame_id")
    if rgb_stamp_ns <= 0 or depth_stamp_ns <= 0:
        raise ValueError("RGB and depth timestamps must be positive")
    if max_stamp_skew_ns < 0:
        raise ValueError("maximum timestamp skew must be non-negative")
    timestamps = [rgb_stamp_ns, depth_stamp_ns]
    if info_stamp_ns > 0:
        timestamps.append(info_stamp_ns)
    if max(timestamps) - min(timestamps) > max_stamp_skew_ns:
        raise ValueError("RGB, aligned depth, and CameraInfo timestamps exceed the configured skew")

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("camera intrinsics must be a finite 3x3 matrix")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    translation = np.asarray(translation, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("timestamped camera transform has invalid dimensions")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise ValueError("timestamped camera transform contains non-finite values")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("timestamped camera rotation is not a valid rotation matrix")
    if depth_scale <= 0.0 or depth_trunc_m <= 0.0:
        raise ValueError("depth scale and truncation must be positive")
    if not 0.0 <= min_valid_depth_ratio <= 1.0:
        raise ValueError("minimum valid depth ratio must be between zero and one")

    depth_m = depth.astype(np.float64) / depth_scale
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_trunc_m)
    valid_depth_ratio = float(np.count_nonzero(valid) / valid.size)
    if valid_depth_ratio < min_valid_depth_ratio:
        raise ValueError(f"aligned depth valid ratio {valid_depth_ratio:.4f} is below {min_valid_depth_ratio:.4f}")
    return PreparedFrame(
        image_bgr=image_bgr,
        depth=depth,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        stamp_ns=rgb_stamp_ns,
        camera_frame=camera_frame,
        translation=translation,
        rotation=rotation,
        valid_depth_ratio=valid_depth_ratio,
    )


def filter_masks(
    frame: PreparedFrame,
    candidates: list[MaskCandidate],
    *,
    max_masks: int,
    min_mask_pixels: int,
    min_mask_area_ratio: float,
    min_valid_depth_ratio: float,
    max_overlap_ratio: float,
    depth_trunc_m: float,
) -> tuple[list[int], MaskFilterDiagnostics]:
    """Return accepted candidate indices using deterministic score/area ordering."""
    if max_masks <= 0 or min_mask_pixels < 0:
        raise ValueError("max_masks must be positive and min_mask_pixels must be non-negative")
    for value, name in (
        (min_mask_area_ratio, "min_mask_area_ratio"),
        (min_valid_depth_ratio, "min_valid_depth_ratio"),
        (max_overlap_ratio, "max_overlap_ratio"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")

    diagnostics = MaskFilterDiagnostics(input_count=len(candidates))
    ranked = []
    depth_m = frame.depth.astype(np.float64) / frame.depth_scale
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_trunc_m)
    image_area = frame.depth.size
    for index, candidate in enumerate(candidates):
        mask = np.asarray(candidate.mask)
        if mask.shape != frame.depth.shape or mask.ndim != 2 or not np.isfinite(mask).all():
            diagnostics.rejected_invalid += 1
            continue
        binary = mask > 0
        area = int(np.count_nonzero(binary))
        if area == 0:
            diagnostics.rejected_invalid += 1
            continue
        if area < min_mask_pixels or area / image_area < min_mask_area_ratio:
            diagnostics.rejected_too_small += 1
            continue
        if np.count_nonzero(binary & valid_depth) / area < min_valid_depth_ratio:
            diagnostics.rejected_depth += 1
            continue
        ranked.append((index, candidate, binary, area))

    ranked.sort(key=lambda item: (-float(item[1].score), -item[3], item[0]))
    accepted = []
    for index, _, binary, area in ranked:
        if any(np.count_nonzero(binary & previous) / area > max_overlap_ratio for previous in accepted):
            diagnostics.rejected_overlap += 1
            continue
        if len(accepted) >= max_masks:
            diagnostics.rejected_limit += 1
            continue
        accepted.append(binary)
        diagnostics.accepted_indices.append(index)
    diagnostics.accepted_count = len(diagnostics.accepted_indices)
    return diagnostics.accepted_indices, diagnostics
