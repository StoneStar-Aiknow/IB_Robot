"""Stage-specific observable conformance metrics for perception backends."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ConformanceThresholds:
    minimum_mask_iou: float = 0.9
    maximum_mask_count_delta: int = 0
    minimum_embedding_cosine: float = 0.99
    maximum_label_score_delta: float = 0.05
    maximum_centroid_error_m: float = 0.01
    maximum_extent_error_m: float = 0.02
    maximum_grasp_count_delta: int = 0
    maximum_grasp_translation_error_m: float = 0.005
    maximum_grasp_rotation_error_deg: float = 5.0
    maximum_grasp_confidence_delta: float = 0.05


@dataclass(frozen=True)
class ConformanceReport:
    passed: bool
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    failures: tuple[str, ...] = ()


def mask_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("conformance masks must have identical dimensions")
    reference = reference > 0
    candidate = candidate > 0
    union = np.count_nonzero(reference | candidate)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(reference & candidate) / union)


def embedding_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("conformance embeddings must have identical dimensions")
    norm = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if norm <= 1e-12:
        raise ValueError("conformance embeddings must have non-zero norm")
    return float(np.dot(reference, candidate) / norm)


def grasp_pose_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Return the translation error in metres and the rotation error in degrees.

    Both arguments are 4x4 camera-frame grasp transforms. The rotation error is the
    geodesic angle of ``R_ref.T @ R_cand``, which is the only comparison that stays
    meaningful across the axis-angle round trip the denoiser performs.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(4, 4)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(4, 4)
    translation = float(np.linalg.norm(reference[:3, 3] - candidate[:3, 3]))
    trace = float(np.trace(reference[:3, :3].T @ candidate[:3, :3]))
    return translation, float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def evaluate_grasp_conformance(
    *,
    reference_poses: list[np.ndarray],
    candidate_poses: list[np.ndarray],
    reference_confidences: list[float],
    candidate_confidences: list[float],
    thresholds: ConformanceThresholds = ConformanceThresholds(),
) -> ConformanceReport:
    """Compare two backends' grasp output rank by rank.

    Grasp lists are confidence-sorted, so the comparison is positional: the reference's
    best grasp must match the candidate's best grasp. A pose that merely appears
    somewhere in both lists is not equivalent behaviour - the executor only ever tries
    the first few.
    """
    failures = []
    count_delta = abs(len(reference_poses) - len(candidate_poses))
    pairs = [
        grasp_pose_error(reference, candidate)
        for reference, candidate in zip(reference_poses, candidate_poses, strict=False)
    ]
    translation_error = max((error for error, _ in pairs), default=0.0)
    rotation_error = max((error for _, error in pairs), default=0.0)
    confidence_delta = max(
        (
            abs(float(reference) - float(candidate))
            for reference, candidate in zip(reference_confidences, candidate_confidences, strict=False)
        ),
        default=0.0,
    )

    if count_delta > thresholds.maximum_grasp_count_delta:
        failures.append("grasp count delta exceeds threshold")
    if translation_error > thresholds.maximum_grasp_translation_error_m:
        failures.append("grasp translation error exceeds threshold")
    if rotation_error > thresholds.maximum_grasp_rotation_error_deg:
        failures.append("grasp rotation error exceeds threshold")
    if confidence_delta > thresholds.maximum_grasp_confidence_delta:
        failures.append("grasp confidence delta exceeds threshold")

    return ConformanceReport(
        passed=not failures,
        metrics={
            "grasp_count_delta": count_delta,
            "grasp_translation_error_m": translation_error,
            "grasp_rotation_error_deg": rotation_error,
            "grasp_confidence_delta": confidence_delta,
        },
        failures=tuple(failures),
    )


def evaluate_backend_conformance(
    *,
    reference_masks: list[np.ndarray],
    candidate_masks: list[np.ndarray],
    reference_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    reference_label: str,
    candidate_label: str,
    reference_label_score: float,
    candidate_label_score: float,
    reference_centroid: np.ndarray,
    candidate_centroid: np.ndarray,
    reference_extent: np.ndarray,
    candidate_extent: np.ndarray,
    reference_failure: tuple[bool, str] = (False, ""),
    candidate_failure: tuple[bool, str] = (False, ""),
    thresholds: ConformanceThresholds = ConformanceThresholds(),
) -> ConformanceReport:
    """Compare equivalent backend stages without conflating label and mask metrics."""
    failures = []
    count_delta = abs(len(reference_masks) - len(candidate_masks))
    paired_ious = [
        mask_iou(reference, candidate) for reference, candidate in zip(reference_masks, candidate_masks, strict=False)
    ]
    minimum_iou = min(paired_ious, default=1.0 if not reference_masks and not candidate_masks else 0.0)
    cosine = embedding_cosine(reference_embedding, candidate_embedding)
    label_top1_match = reference_label == candidate_label
    label_score_delta = abs(float(reference_label_score) - float(candidate_label_score))
    centroid_error = float(
        np.linalg.norm(
            np.asarray(reference_centroid, dtype=np.float64) - np.asarray(candidate_centroid, dtype=np.float64)
        )
    )
    extent_error = float(
        np.max(np.abs(np.asarray(reference_extent, dtype=np.float64) - np.asarray(candidate_extent, dtype=np.float64)))
    )
    failure_semantics_match = reference_failure == candidate_failure

    if count_delta > thresholds.maximum_mask_count_delta:
        failures.append("mask count delta exceeds threshold")
    if minimum_iou < thresholds.minimum_mask_iou:
        failures.append("mask IoU is below threshold")
    if cosine < thresholds.minimum_embedding_cosine:
        failures.append("embedding cosine is below threshold")
    if not label_top1_match:
        failures.append("label top-1 differs")
    if label_score_delta > thresholds.maximum_label_score_delta:
        failures.append("label score delta exceeds threshold")
    if centroid_error > thresholds.maximum_centroid_error_m:
        failures.append("centroid error exceeds threshold")
    if extent_error > thresholds.maximum_extent_error_m:
        failures.append("extent error exceeds threshold")
    if not failure_semantics_match:
        failures.append("failure semantics differ")

    return ConformanceReport(
        passed=not failures,
        metrics={
            "mask_count_delta": count_delta,
            "minimum_mask_iou": minimum_iou,
            "embedding_cosine": cosine,
            "label_top1_match": label_top1_match,
            "label_score_delta": label_score_delta,
            "centroid_error_m": centroid_error,
            "extent_error_m": extent_error,
            "failure_semantics_match": failure_semantics_match,
        },
        failures=tuple(failures),
    )
