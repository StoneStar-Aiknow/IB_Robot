"""Choose an X2 arm from target geometry and keep planning and execution on that arm."""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, Move

from .joints import arm_joints, check_side

# Targets within 5 cm of the torso midline can be attempted with either arm.
MIDLINE_M = 0.05


def target_arm_order(points, base_from_camera, forced="auto"):
    """Use the median target point's torso y; explicit arm_side always wins."""
    if forced != "auto":
        return (check_side(forced),)
    points = np.asarray(points, dtype=float)
    transform = np.asarray(base_from_camera, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError("Finite nonempty target points required for automatic arm selection")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Finite torso-from-camera transform required")
    y = float(np.median((points @ transform[:3, :3].T + transform[:3, 3])[:, 1]))
    preferred = "left" if y >= 0 else "right"
    return (preferred,) if abs(y) > MIDLINE_M else (preferred, "right" if preferred == "left" else "left")


@dataclass
class ArmScenes:
    estimates: dict
    selected_side: str | None = None
    errors: dict = field(default_factory=dict)


class AutomaticPlanner:
    """Try the geometry-selected resident planners under the request's shared deadline.

    Near the midline, reserve time for each arm. GPU work is serialized by the
    head's shared lock; return immediately when the first arm has a valid plan.
    """

    def __init__(self, planners, budget_s=10.0):
        self.planners = planners
        self.budget_s = budget_s

    def plan(self, observation, proposals, *, deadline, cancelled):
        scenes = proposals.scene
        if not isinstance(scenes, ArmScenes):
            raise GraspFailure("INTERNAL_ERROR", "Automatic planning requires per-arm collision scenes")
        failures = dict(scenes.errors)
        sides = tuple(scenes.estimates)
        planning_end = min(deadline, observation.robot_state.started + self.budget_s)
        budget_exhausted = False
        for index, side in enumerate(sides):
            if cancelled():
                raise GraspFailure("CANCELED", "goal cancelled before arm selection")
            now = time.monotonic()
            if now >= deadline:
                raise GraspFailure("SKILL_TIMEOUT", "goal deadline passed during arm selection")
            if now >= planning_end:
                budget_exhausted = True
                break
            arm_deadline = now + (planning_end - now) / (len(sides) - index)
            try:
                plan = self.planners[side].plan(
                    observation,
                    replace(proposals, scene=scenes.estimates[side]),
                    deadline=arm_deadline,
                    cancelled=cancelled,
                )
            except GraspFailure as exc:
                if exc.code not in ("NO_EXECUTABLE_CANDIDATE", "SKILL_TIMEOUT"):
                    raise
                failures[side] = dict(
                    code=exc.code, reason_code=exc.reason_code, message=str(exc), retryable=exc.retryable
                )
                continue
            scenes.selected_side = side
            return replace(
                plan,
                report={
                    **plan.report,
                    "arm_side": side,
                    "arm_selection": dict(attempted=list(failures) + [side], failures=failures),
                },
            )
        # A shared reason remains useful to the Agent; differing arm failures
        # keep the generic summary rather than claiming one cause for both.
        reasons = {item.get("reason_code") or item["code"] for item in failures.values()}
        reason = next(iter(reasons)) if len(reasons) == 1 else "NO_EXECUTABLE_CANDIDATE"
        raise GraspFailure(
            "NO_EXECUTABLE_CANDIDATE",
            f"No selected arm could plan: {failures}",
            retryable=budget_exhausted or any(item.get("retryable", False) for item in failures.values()),
            reason_code="PLAN_TIMEOUT" if budget_exhausted else reason,
        )

    def refine(self, plan, completed, **budget):
        return self.planners[check_side(plan.report["arm_side"])].refine(plan, completed, **budget)


class SelectedPart:
    """Delegate execution calls to the arm chosen by the accepted plan."""

    def __init__(self, parts, selection):
        self.parts, self.selection = parts, selection
        selection.parts.append(self)

    def __getattr__(self, name):
        side = self.selection.side
        if side is None:
            raise GraspFailure("INTERNAL_ERROR", "No arm has been selected by a grasp plan")
        return getattr(self.parts[side], name)

    def finish(self):
        # Observe-only and failed planning requests have no selected executor.
        if self.selection.side is not None:
            finish = getattr(self.parts[self.selection.side], "finish", None)
            if finish is not None:
                finish()


class ExecutionSelection:
    def __init__(self):
        self.side = None
        self.parts = []

    def accept(self, plan):
        side = check_side(plan.report.get("arm_side"))
        moves = [step.segment for step in (*plan.steps, *(plan.release or ())) if isinstance(step, Move)]
        if not moves or any(tuple(segment.joint_names) != arm_joints(side) for segment in moves):
            raise GraspFailure("INTERNAL_ERROR", "Selected arm and trajectory joints disagree")
        if self.side is not None and self.side != side:
            # A regrasp may choose the other arm within the same goal. Stop the
            # previous arm/hand keepalives before another publisher takes over.
            for part in self.parts:
                part.finish()
        self.side = side
        return plan


class SelectingPlanner:
    def __init__(self, planner, selection):
        self.planner, self.selection = planner, selection

    def plan(self, *args, **budget):
        return self.selection.accept(self.planner.plan(*args, **budget))

    def refine(self, *args, **budget):
        return self.selection.accept(self.planner.refine(*args, **budget))


class ArmWatch:
    """Keep the watch on the successful plan's arm, including over remote ARM calls."""

    def __init__(self, watches):
        self.watches, self.active = watches, next(iter(watches))

    def arm(self, observation, proposals, plan):
        side = plan.report["arm_side"] if plan is not None else proposals.scene.selected_side
        side = check_side(side)
        if side != self.active:
            previous = self.watches[self.active]
            # Both arms' watches share the chest IMU, and a disarm pauses it. The
            # plan just resumed it for this execution; switching arms must not
            # pause it again (live 2026-11-04: right pick after a left one, no
            # orientation for any frame, a still orange read "not observable").
            imu = getattr(previous, "imu", None)
            running = imu is not None and not imu.paused
            previous.disarm()
            if running:
                imu.resume()
        self.active = side
        return self.watches[side].arm(observation, proposals, plan)

    def __getattr__(self, name):
        return getattr(self.watches[self.active], name)


class ArmServices:
    def __init__(self, services):
        self.services = services

    def prepare(self, feedback, scenes):
        for side, estimate in scenes.estimates.items():
            self.services[side].prepare(feedback, estimate)

    def preload(self, feedback, scenes):
        for side, estimate in scenes.estimates.items():
            self.services[side].preload(feedback, estimate)


class AutomaticHead:
    """Two per-arm heads sharing their camera, IMU and reentrant GPU lock."""

    def __init__(self, arms, robots):
        self.arms, self.robots = arms, robots
        primary = next(iter(arms.values()))
        self.camera, self.gpu, self.imu = primary.camera, primary.gpu, primary.imu
        self.service = ArmServices({side: arm.service for side, arm in arms.items()})
        self.planner = AutomaticPlanner(
            {side: arm.planner for side, arm in arms.items()}, next(iter(robots.values())).planning.budget_s
        )
        self.watch = ArmWatch({side: arm.watch for side, arm in arms.items()}) if primary.watch is not None else None

    def check_hands(self, admitted):
        from .grasp_request import check_initial_hand

        for robot in self.robots.values():
            check_initial_hand(robot.hand, admitted)

    def prepare_scene(self, observation):
        from .proposer import prepare_capture_scene

        with self.gpu:
            return {
                side: prepare_capture_scene(observation, arm.spheres, self.robots[side].perception.scene)
                for side, arm in self.arms.items()
            }

    def complete_scene(self, prepared, mask, observation):
        from .rgbd_to_cloud import mask_depth_to_points
        from .scene_estimation import NoSupportError, complete_scene

        if mask is None:
            sides = tuple(self.arms)
        else:
            registered = observation.robot_state.live.observation
            frame = registered.frame
            settings = next(iter(self.robots.values())).perception
            points = mask_depth_to_points(
                mask,
                frame.depth,
                frame.intrinsics,
                depth_scale=frame.depth_scale,
                min_depth_m=settings.min_depth_m,
                max_depth_m=settings.max_depth_m,
            )
            sides = target_arm_order(points, observation.base_from_camera)
        estimates, errors = {}, {}
        with self.gpu:
            for side in sides:
                try:
                    estimates[side] = complete_scene(prepared[side], mask, lazy_validation=mask is not None)
                except NoSupportError as exc:
                    errors[side] = dict(code="NO_SUPPORT", message=str(exc))
        if not estimates:
            raise NoSupportError(f"No selected arm has a supported scene: {errors}")
        return ArmScenes(estimates, errors=errors)
