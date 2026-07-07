"""Sensor-fusion helpers for post-grasp verification."""

from __future__ import annotations

from dataclasses import dataclass

STATUS_FAILED = 0
STATUS_SUCCESS = 1
STATUS_UNCERTAIN = 2


@dataclass(frozen=True)
class DepthVisibilityStats:
    """Compact wrist-depth visibility summary."""

    valid_fraction: float
    near_fraction: float
    median_depth_m: float | None
    occluded: bool


@dataclass(frozen=True)
class GraspVerificationInput:
    """Evidence sampled after gripper close or after lift."""

    gripper_position: float | None
    gripper_closed_position: float
    gripper_contact_min_opening: float
    gripper_no_contact_max_opening: float
    gripper_joint: str
    gripper_current_abs_a: float | None
    current_contact_threshold_a: float
    wrist_depth: DepthVisibilityStats | None = None
    expected_target_width_m: float = 0.0


@dataclass(frozen=True)
class GraspVerificationResult:
    """Fused grasp-verification result."""

    status: int
    success: bool
    confidence: float
    message: str
    evidence: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def evaluate_grasp(input_data: GraspVerificationInput) -> GraspVerificationResult:
    """Fuse gripper, current, and wrist visibility evidence.

    Wrist camera occlusion is intentionally not a hard failure. Large objects can
    block a wrist-mounted RealSense precisely when the grasp is good, so occlusion
    is only weak positive/diagnostic evidence.
    """

    success_score = 0.0
    failure_score = 0.0
    evidence: list[str] = []

    if input_data.expected_target_width_m > 0.0:
        evidence.append(f"expected_target_width_m: {input_data.expected_target_width_m:.4f}")

    if input_data.gripper_position is None:
        evidence.append("gripper_position: unavailable")
    else:
        opening_from_closed = abs(input_data.gripper_position - input_data.gripper_closed_position)
        evidence.append(
            f"gripper_position: joint={input_data.gripper_joint or 'unknown'} "
            f"value={input_data.gripper_position:.4f} opening_from_closed={opening_from_closed:.4f}"
        )
        if opening_from_closed >= input_data.gripper_contact_min_opening:
            success_score += 0.55
            evidence.append("gripper_contact: stopped before fully closing")
        elif opening_from_closed <= input_data.gripper_no_contact_max_opening:
            failure_score += 0.45
            evidence.append("gripper_contact: fully closed or near closed")
        else:
            success_score += 0.18
            failure_score += 0.12
            evidence.append("gripper_contact: small residual opening")

    if input_data.gripper_current_abs_a is None:
        evidence.append("gripper_current_abs_a: unavailable")
    else:
        evidence.append(
            f"gripper_current_abs_a: {input_data.gripper_current_abs_a:.4f} "
            f"threshold={input_data.current_contact_threshold_a:.4f}"
        )
        if input_data.gripper_current_abs_a >= input_data.current_contact_threshold_a:
            success_score += 0.35
            evidence.append("current_contact: contact/load current detected")
        else:
            failure_score += 0.20
            evidence.append("current_contact: below contact threshold")

    if input_data.wrist_depth is None:
        evidence.append("wrist_visibility: unavailable")
    else:
        stats = input_data.wrist_depth
        median = "nan" if stats.median_depth_m is None else f"{stats.median_depth_m:.4f}"
        evidence.append(
            "wrist_visibility: "
            f"valid_fraction={stats.valid_fraction:.3f} "
            f"near_fraction={stats.near_fraction:.3f} "
            f"median_depth_m={median} "
            f"occluded={stats.occluded}"
        )
        if stats.occluded:
            success_score += 0.10
            evidence.append("wrist_occlusion: ignored as failure because large grasped objects can block wrist view")

    if success_score == 0.0 and failure_score == 0.0:
        return GraspVerificationResult(
            status=STATUS_UNCERTAIN,
            success=False,
            confidence=0.0,
            message="No usable grasp-verification evidence is available",
            evidence=evidence,
        )

    margin = success_score - failure_score
    if success_score >= 0.65 and margin >= 0.20:
        confidence = _clamp01(success_score)
        return GraspVerificationResult(
            status=STATUS_SUCCESS,
            success=True,
            confidence=confidence,
            message=f"Grasp verified by fused evidence (score={success_score:.2f})",
            evidence=evidence,
        )

    if failure_score >= 0.55 and -margin >= 0.20:
        confidence = _clamp01(failure_score)
        return GraspVerificationResult(
            status=STATUS_FAILED,
            success=False,
            confidence=confidence,
            message=f"Grasp likely failed by fused evidence (score={failure_score:.2f})",
            evidence=evidence,
        )

    confidence = _clamp01(max(success_score, failure_score))
    return GraspVerificationResult(
        status=STATUS_UNCERTAIN,
        success=False,
        confidence=confidence,
        message=f"Grasp verification is uncertain (success={success_score:.2f}, failure={failure_score:.2f})",
        evidence=evidence,
    )
