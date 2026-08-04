"""State and result models shared by pick execution phases."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sensor_msgs.msg import JointState

from ibrobot_msgs.msg import GraspCandidate
from manipulation_execution.grasp_geometry import CandidatePlan, FixedFingerBaseSide, FixedFingerEnvelope
from manipulation_execution.so101_geometry import TablePlane


@dataclass
class FlowState:
    completed_phases: list[str]
    pose_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    frame_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    verification_records: list[dict[str, Any]] = field(default_factory=list)
    attempt: int = 0
    verification_status: int = 0
    verification_confidence: float = 0.0
    debug_output_dir: str = ""
    candidate_index: int = -1
    released_after_success: bool = False
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    active_phase: str = field(default="", repr=False)
    active_phase_started_at: float = field(default=0.0, repr=False)

    def add_timing(self, stage: str, duration_s: float) -> None:
        """Accumulate one non-negative wall-clock timing sample."""
        duration = max(0.0, float(duration_s))
        self.pipeline_timings[stage] = self.pipeline_timings.get(stage, 0.0) + duration

    def enter_phase(self, phase: str, *, now: float | None = None) -> tuple[str, float] | None:
        """Start a feedback phase and return the completed phase timing, if any."""
        entered_at = time.monotonic() if now is None else float(now)
        if phase == self.active_phase:
            return None
        completed = self.finish_active_phase(now=entered_at)
        self.active_phase = str(phase)
        self.active_phase_started_at = entered_at
        return completed

    def finish_active_phase(self, *, now: float | None = None) -> tuple[str, float] | None:
        """Finish the active feedback phase and aggregate its elapsed wall time."""
        if not self.active_phase:
            return None
        finished_at = time.monotonic() if now is None else float(now)
        phase = self.active_phase
        duration = max(0.0, finished_at - self.active_phase_started_at)
        self.active_phase = ""
        self.active_phase_started_at = 0.0
        if phase != "completed":
            self.add_timing(f"phase_{phase}", duration)
        return phase, duration


@dataclass(frozen=True)
class RankedCandidate:
    index: int
    candidate: GraspCandidate
    plan: CandidatePlan
    score: float
    contact_distance_m: float | None = None
    fixed_finger_base_side: FixedFingerBaseSide | None = None


@dataclass(frozen=True)
class IKPayload:
    joint_state: JointState
    ee_xyz: tuple[float, float, float]
    ee_quaternion: tuple[float, float, float, float]
    joint5_retry_applied: bool = False
    original_joint5: float | None = None
    approach_axis_error_deg: float | None = None
    closing_axis_error_deg: float | None = None


@dataclass(frozen=True)
class PreparedCandidate:
    ranked: RankedCandidate
    plan: CandidatePlan
    final_joint_state: JointState
    actual_ee_xyz: tuple[float, float, float]
    actual_ee_quaternion: tuple[float, float, float, float]
    contact_residual_xy_m: float
    contact_z_error_m: float
    approach_axis_error_deg: float | None
    closing_axis_error_deg: float
    tabletop_clearance_m: float | None
    mesh_min_z: float | None
    fixed_finger_envelope: FixedFingerEnvelope | None
    fk_fixed_finger_base_side: FixedFingerBaseSide | None
    predicted_robust_gap_headroom_m: float | None
    selection_score: float


@dataclass(frozen=True)
class PlannerSceneGeometry:
    object_centroid_camera: tuple[float, float, float] | None = None
    table_normal_camera: tuple[float, float, float] | None = None
    table_offset_camera: float = 0.0
    table_inlier_ratio: float = 0.0
    object_top_camera: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class BaseSceneGeometry:
    table_plane: TablePlane | None = None
    object_top_base: tuple[float, float, float] | None = None


class PickFlowError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class PickCancelled(PickFlowError):
    def __init__(self) -> None:
        super().__init__("PICK_CANCELLED", "pick execution cancelled")
