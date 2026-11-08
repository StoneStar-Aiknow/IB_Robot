"""The X2 grasp planner: the request's candidates to a checked, executable grasp plan.

Ported from the planning half of the X2 pipeline's request processor. The
request was admitted at capture (fresh feedback and image, see HeadCamera);
here the hand must be open, candidates out of the planning palm's reach are dropped and
the most confident ``max_planning_candidates`` of the rest are planned with
cuRobo at the admitted snapshot time. The plan must keep the start and the
shared segment endpoints, and the target must be unchanged in a strictly
newer image (the executor checks feedback and robot state, not the scene:
this is the only check after the planned image). Last, the plan must stay in
the motion controller's limits with a gravity sag the raise can carry.

The plan's steps are the X2 sequence's: open the hand, approach, grasp,
settle, close, lift, hold lifted. A put-back retraces them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    Flag,
    GraspFailure,
    GraspPlan,
    Hold,
    Move,
    Observation,
    Proposals,
    SegmentKind,
)

from .arm_gravity import ArmGravity, require_executable
from .grasp_session import check_approach_order
from .hand import OPEN_HAND_TOLERANCE_RAD, THUMB_ABAD_OPEN_TOLERANCE_RAD, HandMapping
from .joints import arm_joints
from .observation import AdmittedState, LiveState
from .planner import PlannerFault, palm_goals_from_camera_grasps
from .revalidation import ReobserveRequired, RevalidationConfig, require_fresh, require_posture, require_stable_patch
from .scene_estimation import SceneEstimate

NO_EXECUTABLE_CANDIDATE = "NO_EXECUTABLE_CANDIDATE"
REOBSERVE_REQUIRED = "REOBSERVE_REQUIRED"
# cuRobo's IK position tolerance (IKSolverCfg default, used by the planner).
IK_POSITION_TOLERANCE_M = 0.005
# The hand closes on a settled arm: 0.3 s after the grasp segment the arm was
# within 1.7 mrad of where it settled after 1-5 s (live 2026-10-09, 17 holds).
GRASP_SETTLE_S = 0.3
LIFT_HOLD_S = 3.0


class _SearchExhausted(Exception):
    """The planning call tried its candidates and none planned (it raised RuntimeError)."""


@dataclass(frozen=True)
class GraspRequestConfig:
    planning_profile: str = "latency"
    approach_seed_policy: str = "clear_path"
    # Seed waypoint above the approach pose along lift_up, not the grasp standoff.
    approach_over_height_m: float = 0.18
    # None retains the historical profile/side-first order. Explicit order wins.
    approach_order_m: tuple[float, ...] | None = None
    screen_check: bool = True
    # budget_s is the hard request deadline from admission (fail, return no
    # plan); it only guards against a hung solver. target_s is the latency
    # goal: reported as within_target, never enforced.
    budget_s: float = 10.0
    target_s: float = 4.0
    max_goalset: int = 100
    # Candidates beyond the left-palm kinematic bound are dropped before
    # screening; the most confident max_planning_candidates of the rest are
    # screened (two 100-problem screen batches).
    reach_prefilter: bool = True
    max_planning_candidates: int = 200
    carry_lift_m: float = 0.0
    revalidation: RevalidationConfig = field(default_factory=RevalidationConfig)

    def __post_init__(self):
        if self.planning_profile not in ("default", "latency"):
            raise ValueError("Unknown planning profile")
        if self.approach_seed_policy not in ("native", "clear_path"):
            raise ValueError("Unknown approach seed policy")
        if (
            isinstance(self.approach_over_height_m, bool)
            or not isinstance(self.approach_over_height_m, int | float)
            or not np.isfinite(self.approach_over_height_m)
            or self.approach_over_height_m <= 0
        ):
            raise ValueError("Finite positive approach_over_height_m required")
        object.__setattr__(self, "approach_order_m", check_approach_order(self.approach_order_m))
        if not np.isfinite(self.budget_s) or self.budget_s <= 0:
            raise ValueError("Finite positive request budget required")
        if not np.isfinite(self.target_s) or self.target_s <= 0:
            raise ValueError("Finite positive latency target required")
        for name in ("max_goalset", "max_planning_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Positive integer {name} required")
        if not isinstance(self.screen_check, bool) or not isinstance(self.reach_prefilter, bool):
            raise ValueError("Boolean screening and prefilter settings required")
        if not np.isfinite(self.carry_lift_m) or not 0 <= self.carry_lift_m <= 0.25:
            raise ValueError("carry_lift_m must be in [0, 0.25]")


def left_palm_reach():
    """Cached torso-frame (center, radius) bound on the R_palm origin (see RobotSphereModel)."""
    from .robot_spheres import left_reach_bounds

    center, _, radius = left_reach_bounds()
    return center, radius


def palm_reach(side: str = "left"):
    """:func:`left_palm_reach` for either arm's planning model."""
    if side == "left":
        return left_palm_reach()
    from .robot_spheres import reach_bounds

    center, _, radius = reach_bounds(side)
    return center, radius


def prefilter_candidates(grasps, confidence, base_from_camera, grasp_to_palm, max_candidates, side: str = "left"):
    """Indices of the most confident candidates whose palm goal is kinematically reachable.

    No reachable palm origin lies outside the bound, and IK accepts goals
    within its position tolerance, so this only drops candidates IK cannot
    solve. On recorded live inputs about half of GraspGen-X's candidates lie
    beyond it.
    """
    center, radius = palm_reach(side)
    goals = palm_goals_from_camera_grasps(grasps, base_from_camera, grasp_to_palm)
    reach = np.linalg.norm(goals[:, :3, 3] - center, axis=1) <= radius + IK_POSITION_TOLERANCE_M
    order = np.flatnonzero(reach)[np.argsort(-np.asarray(confidence)[reach], kind="stable")]
    return order[:max_candidates], dict(
        generated=len(grasps),
        in_reach=int(reach.sum()),
        kept=int(min(len(order), max_candidates)),
        palm_reach_m=float(radius),
    )


def check_initial_hand(mapping: HandMapping, admitted: AdmittedState) -> dict:
    """Both current and image-time feedback must describe an open hand."""
    errors = {}
    live = admitted.live
    for name, feedback in (("current", live.feedback), ("image", live.observation.image_feedback)):
        try:
            errors[name + "_max_error_rad"] = mapping.require_open(feedback.hand(mapping.side))
        except RuntimeError as exc:
            raise RuntimeError(f"{name.capitalize()} feedback: {exc}") from exc
    return dict(
        preset="open",
        confirmed=True,
        tolerance_rad=OPEN_HAND_TOLERANCE_RAD,
        thumb_abad_tolerance_rad=THUMB_ABAD_OPEN_TOLERANCE_RAD,
        **errors,
    )


def segment(name: str, trajectory, side: str = "left") -> ArmSegment:
    return ArmSegment(name, SegmentKind.TRAJECTORY, arm_joints(side), trajectory.positions, trajectory.times)


def grasp_steps(plan, side: str = "left") -> tuple:
    """The X2 sequence. A lift the planner did not plan (it only holds the grasp pose) is refused, as the X2
    pipeline's preflight refused ``lift_not_planned``: closing and holding in place would report a pick that never
    left the table."""
    if np.ptp(plan.lift.positions, axis=0).max() < 1e-6:
        raise GraspFailure("NO_EXECUTABLE_CANDIDATE", "the planner did not plan a lift for this candidate")
    return (
        Effect(EffectorAction.PRESHAPE),
        Move(segment("approach", plan.approach, side)),
        Move(segment("grasp", plan.grasp, side)),
        Hold(GRASP_SETTLE_S, "settle_grasp"),
        Effect(EffectorAction.CLOSE),
        Move(segment("lift", plan.lift, side)),
        Hold(LIFT_HOLD_S, "hold_lifted"),
    )


class X2GraspPlanner:
    """``GraspPlanner`` for one X2 arm (``hand.side``) over that arm's warmed GraspPlanningService.

    ``latest_state`` takes a newer capture for the target recheck
    (HeadCamera.capture_state); ``clock`` is the source clock it is judged on.
    """

    def __init__(
        self,
        service,
        *,
        grasp_to_palm: np.ndarray,
        hand: HandMapping,
        gravity: ArmGravity,
        limits: Mapping[str, tuple[float, float]],
        latest_state: Callable[[], LiveState],
        clock: Callable[[], float],
        config: GraspRequestConfig = GraspRequestConfig(),
    ):
        self.service, self.grasp_to_palm, self.hand = service, np.asarray(grasp_to_palm, dtype=float), hand
        self.gravity, self.limits = gravity, limits
        self.latest_state, self.clock, self.config = latest_state, clock, config

    def plan(self, observation: Observation, proposals: Proposals, *, deadline: float, cancelled: Flag) -> GraspPlan:
        admitted = observation.robot_state
        if not isinstance(admitted, AdmittedState) or not isinstance(proposals.scene, SceneEstimate):
            raise GraspFailure("INTERNAL_ERROR", "the X2 planner plans X2 captures with the X2 proposer's scene")
        # Planning time is perf_counter (GraspPlanningService); the goal deadline monotonic.
        budget = admitted.started + self.config.budget_s - time.monotonic()
        planning_deadline = time.perf_counter() + min(budget, deadline - time.monotonic())

        def check_deadline():
            if cancelled():
                raise GraspFailure("CANCELED", "goal cancelled")
            if time.monotonic() >= deadline:
                raise GraspFailure("SKILL_TIMEOUT", "goal deadline passed")
            if time.perf_counter() >= planning_deadline:
                raise TimeoutError("Grasp request deadline exceeded")

        report: dict = {"arm_side": self.hand.side, "request_config": asdict(self.config), "request_timings_s": {}}

        def measure(name, operation):
            before = time.perf_counter()
            try:
                return operation()
            finally:
                report["request_timings_s"][name] = time.perf_counter() - before

        try:
            return self._plan(admitted, proposals, report, measure, check_deadline, planning_deadline)
        except GraspFailure:
            raise
        except ReobserveRequired as exc:
            raise GraspFailure(REOBSERVE_REQUIRED, str(exc), retryable=True) from exc
        except (_SearchExhausted, TimeoutError) as exc:
            # The search found nothing in this candidate set, or ran out of time:
            # a new observation samples new candidates (live 2026-11-03: 7 of 48
            # picks, all before the arm moved, none retried). Candidates out of
            # reach, a refused plan and a planner fault are not retried.
            cause = exc.__cause__ if isinstance(exc, _SearchExhausted) else exc
            raise GraspFailure(
                NO_EXECUTABLE_CANDIDATE,
                f"{type(cause).__name__}: {cause}",
                retryable=True,
                reason_code="PLAN_TIMEOUT" if isinstance(exc, TimeoutError) else "NO_COMPLETE_PATH",
            ) from exc
        except (RuntimeError, ValueError, PlannerFault) as exc:
            raise GraspFailure(
                NO_EXECUTABLE_CANDIDATE,
                f"{type(exc).__name__}: {exc}",
                reason_code="PLANNER_FAULT" if isinstance(exc, PlannerFault) else "PLAN_REJECTED",
            ) from exc
        finally:
            ready = report["request_timings_s"].get("complete_plan_ready")
            report["within_target"] = ready is not None and ready <= self.config.target_s

    def refine(self, plan: GraspPlan, completed: int, *, deadline: float, cancelled: Flag) -> GraspPlan:
        raise GraspFailure("INTERNAL_ERROR", "X2 grasp plans are not refined")

    # -- the request -----------------------------------------------------------------

    def _plan(self, admitted, proposals, report, measure, check_deadline, planning_deadline) -> GraspPlan:
        live, estimate = admitted.live, proposals.scene
        observation, start = live.observation, live.feedback
        transform = observation.transform
        if estimate.stamp_ns != transform.stamp_ns or estimate.camera_frame != transform.camera_frame:
            raise ValueError("Scene and candidates must belong to the planned image")
        report["initial_hand"] = check_initial_hand(self.hand, admitted)
        report["source"] = dict(
            stamp_ns=transform.stamp_ns,
            camera_frame=transform.camera_frame,
            scene_received_at=observation.received_at,
            feedback_received_at=start.received_at,
            image_feedback_received_at=observation.image_feedback.received_at,
            admitted_at=admitted.admitted_at,
            calibration_verified=transform.calibration_verified,
            calibration_id=observation.calibration_id,
        )
        grasps, confidence = proposals.grasps, proposals.confidence
        report["candidate_count"] = len(grasps)
        kept = np.arange(len(grasps))
        if self.config.reach_prefilter:
            kept, report["candidate_prefilter"] = measure(
                "prefilter",
                lambda: prefilter_candidates(
                    grasps,
                    confidence,
                    transform.base_from_camera,
                    self.grasp_to_palm,
                    self.config.max_planning_candidates,
                    self.hand.side,
                ),
            )
            report["candidate_prefilter"]["kept_indices"] = kept.tolist()
            if not len(kept):
                raise GraspFailure(
                    NO_EXECUTABLE_CANDIDATE,
                    f"No grasp candidate within {self.hand.side}-palm reach",
                    reason_code="OUT_OF_REACH",
                )
        check_deadline()
        try:
            plan, planning = measure(
                "planning_call",
                lambda: self.service.plan(
                    start,
                    estimate,
                    grasps[kept],
                    confidence[kept],
                    transform.base_from_camera,
                    screen_check=self.config.screen_check,
                    deadline=planning_deadline,
                    target_points_camera=proposals.target_points,
                    checkpoint=check_deadline,
                    snapshot_now=admitted.admitted_at,
                ),
            )
        except GraspFailure:
            raise  # a cancel or the goal deadline at a checkpoint (GraspFailure is a RuntimeError)
        except (TimeoutError, PlannerFault):
            raise
        except RuntimeError as exc:
            raise _SearchExhausted() from exc
        report.update(planning)
        report["generated_candidate_index"] = int(kept[report["candidate_index"]])
        error = float(
            max(
                np.abs(plan.approach.positions[0] - start.vector(arm_joints(self.hand.side))).max(),
                np.abs(plan.grasp.positions[0] - plan.approach.positions[-1]).max(),
                np.abs(plan.lift.positions[0] - plan.grasp.positions[-1]).max(),
            )
        )
        report["continuity_max_error_rad"] = error
        if not np.isfinite(error) or error > 1e-6:
            raise RuntimeError("Plan does not preserve the start and shared segment endpoints")
        check_deadline()
        report["target_recheck"] = measure(
            "target_recheck", lambda: self._recheck_target(observation, start, proposals.target_mask)
        )
        check_deadline()
        try:
            report["max_sag_rad"] = require_executable(
                {name: getattr(plan, name).positions for name in ("approach", "grasp", "lift")},
                self.limits,
                self.gravity,
            )
        except ValueError as exc:
            raise ValueError(f"gravity compensation refused the plan: {exc}") from exc
        report["full_plan_valid"] = True
        report["request_timings_s"]["complete_plan_ready"] = time.monotonic() - admitted.started
        steps = grasp_steps(plan, self.hand.side)
        carry_steps = None
        if self.config.carry_lift_m:
            with self.service.session(start, estimate) as planner:
                lift, height = planner.carry_lift(plan.grasp, self.config.carry_lift_m, check_deadline)
            require_executable({"carry": lift.positions}, self.limits, self.gravity)
            carry_steps = (*steps[:5], Move(segment("lift_carry", lift, self.hand.side)))
            report["carry_lift_m"] = height
        return GraspPlan(steps, report["generated_candidate_index"], None, report, carry_steps)

    def _recheck_target(self, planned, start, mask) -> dict:
        limits = self.config.revalidation
        state = self.latest_state()
        stale = None
        try:
            require_fresh(state, self.clock(), limits)
        except ReobserveRequired as exc:
            # A feedback sample a few ms late (live 2026-10-09: 106 ms against
            # the 100 ms limit) would otherwise discard the plan and re-run
            # perception; one more capture costs ~0.1 s. Limits are unchanged.
            stale = str(exc)
            state = self.latest_state()
            require_fresh(state, self.clock(), limits)
        # Strictly newer: the planned image compared with itself proves nothing.
        if state.observation.transform.stamp_ns <= planned.transform.stamp_ns:
            raise ReobserveRequired("Latest observation is not newer than the planned image")
        posture_error = require_posture(start, state.feedback, limits, side=self.hand.side)
        patch = require_stable_patch(planned, state.observation, mask, limits)
        return dict(
            patch,
            posture_error_rad=posture_error,
            stale_first_sample=stale,
            planned_stamp_ns=planned.transform.stamp_ns,
            checked_stamp_ns=state.observation.transform.stamp_ns,
        )
