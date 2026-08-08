"""Pure GraspGen-to-robot geometry conversion helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class CandidatePlan:
    approach: Vector3
    grasp: Vector3
    lift: Vector3
    quaternion: Quaternion
    approach_axis: Vector3
    target_contact_ee: Vector3
    target_contact_base: Vector3
    target_width_m: float
    width_reason: str
    fixed_finger_target_gap_m: float
    target_width_min_base: Vector3 | None
    target_width_max_base: Vector3 | None
    topdown_score: float


@dataclass(frozen=True)
class FixedFingerEnvelope:
    fixed_gap_m: float
    moving_gap_m: float | None
    target_gap_m: float
    fixed_score: float
    moving_score: float | None
    score: float


@dataclass(frozen=True)
class FixedFingerBaseSide:
    fixed_contact_base: Vector3
    target_center_base: Vector3
    reference_point_base: Vector3
    inward_offset_m: float
    alignment_cos: float


@dataclass(frozen=True)
class FixedFingerRobustGap:
    contact_error_along_closing_axis_m: float
    effective_gap_m: float
    required_gap_m: float
    gap_deficit_m: float
    max_target_gap_deficit_m: float
    measurement_tolerance_m: float
    passed: bool


@dataclass(frozen=True)
class GraspAxisErrors:
    approach_deg: float
    closing_deg: float


def target_extent_along_closing_axis(
    ee_quaternion: Quaternion,
    closing_axis_ee: Iterable[float],
    target_width_min_base: Vector3,
    target_width_max_base: Vector3,
) -> float:
    """Project the measured target extent onto the actual FK closing axis."""

    closing_axis_base = quaternion_matrix(ee_quaternion) @ _normalized(closing_axis_ee)
    target_extent = np.asarray(target_width_max_base, dtype=np.float64) - np.asarray(
        target_width_min_base,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(target_extent)):
        raise ValueError("target width extent must be finite")
    return abs(float(np.dot(target_extent, closing_axis_base)))


def _triplet(value: Any, default: Vector3) -> Vector3:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalized(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(value), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("vector must be non-zero")
    return vector / norm


def euler_xyz_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def quaternion_matrix(quaternion: Quaternion) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(matrix: np.ndarray) -> Quaternion:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return (
        float(quaternion[0] / norm),
        float(quaternion[1] / norm),
        float(quaternion[2] / norm),
        float(quaternion[3] / norm),
    )


def grasp_axis_errors(
    target_quaternion: Quaternion,
    actual_quaternion: Quaternion,
    approach_axis_ee: Iterable[float],
    closing_axis_ee: Iterable[float],
    *,
    closing_axis_180_symmetric: bool = False,
) -> GraspAxisErrors:
    """Return signed-axis alignment errors for the actual FK gripper pose."""

    target_rotation = quaternion_matrix(target_quaternion)
    actual_rotation = quaternion_matrix(actual_quaternion)

    def error_deg(axis_ee: Iterable[float], *, symmetric: bool = False) -> float:
        axis = _normalized(axis_ee)
        target_axis = target_rotation @ axis
        actual_axis = actual_rotation @ axis
        cosine = float(np.dot(target_axis, actual_axis))
        if symmetric:
            cosine = abs(cosine)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    return GraspAxisErrors(
        approach_deg=error_deg(approach_axis_ee),
        closing_deg=error_deg(closing_axis_ee, symmetric=closing_axis_180_symmetric),
    )


def transform_matrix(translation: Vector3, quaternion: Quaternion) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(quaternion)
    matrix[:3, 3] = translation
    return matrix


def matrix_from_rowmajor(values: Iterable[float]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64).reshape(4, 4)
    matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
    return matrix


def source_contact_camera(
    candidate_pose_matrix: Iterable[float],
    source_contact_point: Iterable[float],
) -> Vector3:
    """Return the GraspGen contact reference in the camera frame."""

    source_contact = _triplet(list(source_contact_point), (0.0, 0.0, 0.195))
    contact = matrix_from_rowmajor(candidate_pose_matrix) @ np.array([*source_contact, 1.0], dtype=np.float64)
    return (float(contact[0]), float(contact[1]), float(contact[2]))


def contact_distance_score(contact: Vector3, centroid: Vector3, scale_m: float) -> tuple[float, float]:
    """Return contact-to-centroid distance and a linearly decaying score."""

    distance = float(np.linalg.norm(np.asarray(contact, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)))
    scale = max(1e-6, float(scale_m))
    return distance, max(0.0, min(1.0, 1.0 - distance / scale))


def _target_contact_geometry(
    grasp_execution: dict[str, Any],
    target_width_m: float,
    target_width_quality: float,
) -> tuple[np.ndarray, float, float, str]:
    target_gripper = grasp_execution.get("target_gripper", {})
    fixed_contact = np.asarray(
        _triplet(target_gripper.get("fixed_finger_contact_ee"), (0.008, 0.0, -0.080)),
        dtype=np.float64,
    )
    closing_axis = _normalized(_triplet(target_gripper.get("closing_axis_ee"), (1.0, 0.0, 0.0)))
    quality_min = float(target_gripper.get("width_quality_min", 0.75))
    min_width = float(target_gripper.get("min_width_m", 0.008))
    max_width = float(target_gripper.get("max_width_m", 0.080))
    width = float(target_width_m)
    if not math.isfinite(width) or width <= 0.0 or float(target_width_quality) < quality_min:
        width = float(target_gripper.get("fallback_width_m", 0.035))
        source = "fallback"
    else:
        source = "measured"

    width = min(max(width, min_width), max_width)
    width_with_clearance = min(max(width + float(target_gripper.get("width_clearance_m", 0.003)), min_width), max_width)
    base_margin = max(0.0, float(target_gripper.get("fixed_finger_margin_m", 0.0)))
    max_margin = max(base_margin, float(target_gripper.get("fixed_finger_margin_max_m", base_margin)))
    width_ref = max(0.0, float(target_gripper.get("fixed_finger_margin_width_ref_m", 0.035)))
    width_gain = max(0.0, float(target_gripper.get("fixed_finger_margin_width_gain", 0.0)))
    margin = min(base_margin + max(0.0, width_ref - width) * width_gain, max_margin)
    center_offset = 0.5 * width_with_clearance + margin
    fixed_gap = center_offset - 0.5 * width
    return (
        fixed_contact + closing_axis * center_offset,
        width,
        fixed_gap,
        f"{source}:width={width:.4f}:margin={margin:.4f}",
    )


def target_contact_for_candidate(
    grasp_execution: dict[str, Any],
    target_width_m: float,
    target_width_quality: float,
) -> tuple[np.ndarray, str]:
    target_contact, _, _, reason = _target_contact_geometry(
        grasp_execution,
        target_width_m,
        target_width_quality,
    )
    return target_contact, reason


def fixed_finger_envelope_score(
    ee_xyz: Vector3,
    ee_quaternion: Quaternion,
    fixed_finger_contact_ee: Iterable[float],
    closing_axis_ee: Iterable[float],
    target_width_min_base: Vector3,
    target_width_max_base: Vector3,
    target_gap_m: float,
    *,
    gap_sigma_m: float,
    reliable_max_opening_m: float,
    moving_min_clearance_m: float,
    fixed_score_weight: float,
) -> FixedFingerEnvelope:
    rotation = quaternion_matrix(ee_quaternion)
    fixed_contact = np.asarray(ee_xyz, dtype=np.float64) + rotation @ np.asarray(
        list(fixed_finger_contact_ee), dtype=np.float64
    )
    closing_axis = rotation @ _normalized(closing_axis_ee)
    projections = [
        float(np.dot(np.asarray(point, dtype=np.float64) - fixed_contact, closing_axis))
        for point in (target_width_min_base, target_width_max_base)
    ]
    fixed_gap = min(projections)
    far_extent = max(projections)
    sigma = max(1e-6, float(gap_sigma_m))
    fixed_score = math.exp(-0.5 * ((fixed_gap - float(target_gap_m)) / sigma) ** 2)

    moving_gap = None
    moving_score = None
    score = fixed_score
    maximum_opening = float(reliable_max_opening_m)
    if maximum_opening > 0.0:
        moving_gap = maximum_opening - far_extent
        clearance = max(1e-6, float(moving_min_clearance_m))
        moving_score = max(0.0, min(1.0, moving_gap / clearance))
        fixed_weight = max(0.0, min(1.0, float(fixed_score_weight)))
        score = fixed_weight * fixed_score + (1.0 - fixed_weight) * moving_score

    return FixedFingerEnvelope(
        fixed_gap_m=fixed_gap,
        moving_gap_m=moving_gap,
        target_gap_m=float(target_gap_m),
        fixed_score=fixed_score,
        moving_score=moving_score,
        score=score,
    )


def fixed_finger_base_side_alignment(
    ee_xyz: Vector3,
    ee_quaternion: Quaternion,
    fixed_finger_contact_ee: Iterable[float],
    target_width_min_base: Vector3,
    target_width_max_base: Vector3,
    reference_point_base: Iterable[float] = (0.0, 0.0, 0.0),
) -> FixedFingerBaseSide:
    """Measure whether the fixed finger is on the target side facing the robot base."""

    rotation = quaternion_matrix(ee_quaternion)
    fixed_contact = np.asarray(ee_xyz, dtype=np.float64) + rotation @ np.asarray(
        list(fixed_finger_contact_ee), dtype=np.float64
    )
    target_center = 0.5 * (
        np.asarray(target_width_min_base, dtype=np.float64) + np.asarray(target_width_max_base, dtype=np.float64)
    )
    reference_point = np.asarray(_triplet(list(reference_point_base), (0.0, 0.0, 0.0)), dtype=np.float64)
    inward_xy = reference_point[:2] - target_center[:2]
    fixed_offset_xy = fixed_contact[:2] - target_center[:2]
    inward_norm = float(np.linalg.norm(inward_xy))
    fixed_offset_norm = float(np.linalg.norm(fixed_offset_xy))
    if inward_norm <= 1e-9:
        raise ValueError("target center must differ from the fixed-finger reference point in XY")
    if fixed_offset_norm <= 1e-9:
        raise ValueError("fixed finger contact must differ from the target center in XY")
    inward_offset = float(np.dot(fixed_offset_xy, inward_xy / inward_norm))
    alignment_cos = max(-1.0, min(1.0, inward_offset / fixed_offset_norm))
    return FixedFingerBaseSide(
        fixed_contact_base=(float(fixed_contact[0]), float(fixed_contact[1]), float(fixed_contact[2])),
        target_center_base=(float(target_center[0]), float(target_center[1]), float(target_center[2])),
        reference_point_base=(float(reference_point[0]), float(reference_point[1]), float(reference_point[2])),
        inward_offset_m=inward_offset,
        alignment_cos=alignment_cos,
    )


def fixed_finger_robust_gap(
    fixed_gap_m: float,
    target_gap_m: float,
    contact_residual_base: Iterable[float],
    ee_quaternion: Quaternion,
    closing_axis_ee: Iterable[float],
    *,
    max_target_gap_deficit_m: float,
    measurement_tolerance_m: float = 0.0,
) -> FixedFingerRobustGap:
    """Apply the measured contact residual to the fixed-finger safety gap."""

    closing_axis_base = quaternion_matrix(ee_quaternion) @ _normalized(closing_axis_ee)
    contact_error = float(np.dot(np.asarray(list(contact_residual_base), dtype=np.float64), closing_axis_base))
    maximum_deficit = max(0.0, float(max_target_gap_deficit_m))
    measurement_tolerance = max(0.0, float(measurement_tolerance_m))
    effective_gap = float(fixed_gap_m) + contact_error
    required_gap = max(0.0, float(target_gap_m) - maximum_deficit)
    gap_deficit = max(0.0, required_gap - effective_gap)
    return FixedFingerRobustGap(
        contact_error_along_closing_axis_m=contact_error,
        effective_gap_m=effective_gap,
        required_gap_m=required_gap,
        gap_deficit_m=gap_deficit,
        max_target_gap_deficit_m=maximum_deficit,
        measurement_tolerance_m=measurement_tolerance,
        passed=gap_deficit <= measurement_tolerance + 1e-9,
    )


def prepared_candidate_soft_score(
    config: dict[str, Any],
    *,
    fixed_finger_envelope: float | None,
    contact_residual_xy_m: float,
    contact_z_error_m: float,
    confidence: float,
    centroid_distance_m: float | None = None,
    robust_gap_headroom_m: float | None = None,
) -> float:
    def quality(error: float, scale: float) -> float:
        return 1.0 - min(max(abs(float(error)) / max(1e-6, float(scale)), 0.0), 1.0)

    weighted_terms = [
        (
            quality(contact_residual_xy_m, float(config.get("contact_xy_scale_m", 0.03))),
            max(0.0, float(config.get("contact_xy_weight", 0.25))),
        ),
        (
            quality(contact_z_error_m, float(config.get("contact_z_scale_m", 0.02))),
            max(0.0, float(config.get("contact_z_weight", 0.15))),
        ),
        (
            min(max(float(confidence), 0.0), 1.0),
            max(0.0, float(config.get("confidence_weight", 0.05))),
        ),
    ]
    envelope_value = (
        float(config.get("missing_fixed_finger_envelope_score", 0.20))
        if fixed_finger_envelope is None
        else fixed_finger_envelope
    )
    weighted_terms.append(
        (
            min(max(float(envelope_value), 0.0), 1.0),
            max(0.0, float(config.get("fixed_finger_envelope_weight", 0.55))),
        )
    )
    if centroid_distance_m is not None and math.isfinite(float(centroid_distance_m)):
        weighted_terms.append(
            (
                quality(float(centroid_distance_m), float(config.get("centroid_distance_scale_m", 0.01))),
                max(0.0, float(config.get("centroid_distance_weight", 0.0))),
            )
        )
    headroom_weight = max(0.0, float(config.get("robust_gap_headroom_weight", 0.0)))
    if headroom_weight > 0.0:
        if robust_gap_headroom_m is None or not math.isfinite(float(robust_gap_headroom_m)):
            headroom_quality = float(config.get("missing_robust_gap_headroom_score", 0.5))
        else:
            scale = max(1e-6, float(config.get("robust_gap_headroom_scale_m", 0.004)))
            headroom_quality = 0.5 + float(robust_gap_headroom_m) / (2.0 * scale)
        weighted_terms.append((min(max(headroom_quality, 0.0), 1.0), headroom_weight))
    total_weight = sum(weight for _, weight in weighted_terms)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in weighted_terms) / total_weight


def target_width_extent_points(
    candidate_pose_matrix: Iterable[float],
    base_to_camera: np.ndarray,
    width_axis_camera: Iterable[float],
    target_width_min_offset_m: float,
    target_width_max_offset_m: float,
    target_width_quality: float,
    width_quality_min: float,
) -> tuple[Vector3, Vector3] | None:
    width_axis = np.asarray(list(width_axis_camera), dtype=np.float64)
    width_axis_norm = float(np.linalg.norm(width_axis))
    quality = float(target_width_quality)
    if (
        not math.isfinite(quality)
        or quality < float(width_quality_min)
        or not np.all(np.isfinite(width_axis))
        or not math.isfinite(width_axis_norm)
        or width_axis_norm <= 1e-9
        or not math.isfinite(float(target_width_min_offset_m))
        or not math.isfinite(float(target_width_max_offset_m))
        or float(target_width_max_offset_m) <= float(target_width_min_offset_m)
    ):
        return None

    width_axis /= width_axis_norm
    candidate_matrix = matrix_from_rowmajor(candidate_pose_matrix)
    candidate_origin = candidate_matrix[:3, 3]
    transform = np.asarray(base_to_camera, dtype=np.float64)
    extent_points = []
    for width_offset in (target_width_min_offset_m, target_width_max_offset_m):
        point_camera = candidate_origin + width_axis * float(width_offset)
        point_base = transform @ np.array([*point_camera, 1.0], dtype=np.float64)
        extent_points.append((float(point_base[0]), float(point_base[1]), float(point_base[2])))
    return extent_points[0], extent_points[1]


def build_candidate_plan(
    candidate_pose_matrix: Iterable[float],
    base_to_camera: np.ndarray,
    target_width_m: float,
    target_width_quality: float,
    grasp_execution: dict[str, Any],
    *,
    width_axis_camera: Iterable[float] | None = None,
    target_width_min_offset_m: float = 0.0,
    target_width_max_offset_m: float = 0.0,
) -> CandidatePlan:
    source_contact = np.asarray(
        _triplet(grasp_execution.get("source_contact_point"), (0.0, 0.0, 0.195)),
        dtype=np.float64,
    )
    adapter = grasp_execution.get("adapter", {})
    adapter_rotation = euler_xyz_matrix(_triplet(adapter.get("source_to_ee_rpy"), (math.pi, 0.0, 0.0)))
    target_contact, target_width_used, fixed_finger_target_gap, width_reason = _target_contact_geometry(
        grasp_execution,
        target_width_m,
        target_width_quality,
    )
    adapter_translation = source_contact - adapter_rotation @ target_contact
    source_to_ee = np.eye(4, dtype=np.float64)
    source_to_ee[:3, :3] = adapter_rotation
    source_to_ee[:3, 3] = adapter_translation

    candidate_matrix = matrix_from_rowmajor(candidate_pose_matrix)
    base_to_camera = np.asarray(base_to_camera, dtype=np.float64)
    base_to_source = base_to_camera @ candidate_matrix
    base_to_ee = base_to_source @ source_to_ee
    target_offset = np.asarray(
        _triplet(grasp_execution.get("candidate_target_offset_base_m"), (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(target_offset)):
        raise ValueError("candidate_target_offset_base_m must be finite")
    base_to_ee[:3, 3] += target_offset
    grasp: Vector3 = (float(base_to_ee[0, 3]), float(base_to_ee[1, 3]), float(base_to_ee[2, 3]))
    approach_axis: Vector3 = (
        float(base_to_source[0, 2]),
        float(base_to_source[1, 2]),
        float(base_to_source[2, 2]),
    )
    approach_distance = float(grasp_execution.get("approach_distance_m", 0.08))
    lift_distance = float(grasp_execution.get("lift_distance_m", 0.165))
    approach: Vector3 = (
        grasp[0] - approach_axis[0] * approach_distance,
        grasp[1] - approach_axis[1] * approach_distance,
        grasp[2] - approach_axis[2] * approach_distance,
    )
    lift = (grasp[0], grasp[1], grasp[2] + lift_distance)
    quaternion = quaternion_from_matrix(base_to_ee)
    contact = base_to_ee @ np.array([*target_contact, 1.0], dtype=np.float64)
    target_width_min_base = None
    target_width_max_base = None
    width_quality_min = float(grasp_execution.get("target_gripper", {}).get("width_quality_min", 0.75))
    width_axis_value = (0.0, 0.0, 0.0) if width_axis_camera is None else width_axis_camera
    extent_points = target_width_extent_points(
        candidate_pose_matrix,
        base_to_camera,
        width_axis_value,
        target_width_min_offset_m,
        target_width_max_offset_m,
        target_width_quality,
        width_quality_min,
    )
    if extent_points is not None:
        target_width_min_base, target_width_max_base = extent_points
    topdown_score = max(0.0, min(1.0, -approach_axis[2]))
    return CandidatePlan(
        approach=approach,
        grasp=grasp,
        lift=lift,
        quaternion=quaternion,
        approach_axis=approach_axis,
        target_contact_ee=(float(target_contact[0]), float(target_contact[1]), float(target_contact[2])),
        target_contact_base=(float(contact[0]), float(contact[1]), float(contact[2])),
        target_width_m=target_width_used,
        width_reason=width_reason,
        fixed_finger_target_gap_m=fixed_finger_target_gap,
        target_width_min_base=target_width_min_base,
        target_width_max_base=target_width_max_base,
        topdown_score=topdown_score,
    )


def xyz_within_workspace(xyz: Vector3, workspace: dict[str, Any]) -> tuple[bool, str]:
    if not all(math.isfinite(value) for value in xyz):
        return False, "non-finite position"
    for index, axis in enumerate(("x", "y", "z")):
        limits = workspace.get(axis)
        if (
            isinstance(limits, list | tuple)
            and len(limits) == 2
            and (xyz[index] < float(limits[0]) or xyz[index] > float(limits[1]))
        ):
            return False, f"{axis} outside workspace: {xyz[index]:.4f}"
    max_radius = workspace.get("max_radius_m")
    if max_radius is not None and math.sqrt(sum(value * value for value in xyz)) > float(max_radius):
        return False, "radius outside workspace"
    return True, ""
