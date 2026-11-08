"""Held-hand placement planning; independent of pick's open-hand admission."""

import time

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction, GraspFailure, Move, PlacePlan

from .arm_gravity import require_executable
from .collision_spheres import merge_sphere_cover
from .grasp_request import segment
from .held_object import held_center
from .place_targets import place_targets
from .planner import ATTACH_LINK, ATTACH_SLOTS, PlacementSearchExhausted, PlannerFault
from .proposer import measured_spheres, prepare_capture_scene
from .revalidation import ReobserveRequired, require_fresh, require_posture
from .scene_estimation import NoSupportError, complete_scene
from .trajectory import TimedTrajectory


def _hand_cover(model, feedback, side, hands, slots, *, margin_m=0.01, voxel_m=0.02):
    """Palm-frame spheres covering ``hands`` (finger poses) beyond the open-hand model.

    The planner keeps the open hand it was warmed with; what these poses add
    is voxelized, inflated by ``margin_m`` and merged under the smallest radius
    cap (5 mm steps) whose cover fits ``slots`` spheres.
    """
    hands = np.atleast_2d(hands)
    for pose in hands:
        try:
            model.mapping.check(pose)
        except ValueError as exc:
            raise GraspFailure("PLACE_PLAN_FAILED", f"held hand pose is outside the hand model: {exc}") from exc
    poses = np.vstack([hands, model.mapping.preset("open")[None]])
    # The cover is in the palm frame, so any arm pose gives the same one. The
    # measured arm may sit a hair past a model bound (live 2026-11-06: right
    # wrist pitch 0.526 > 0.5236, refused here as a "hand pose"); FK at the
    # nearest in-bound arm pose instead. Nothing is planned from this pose.
    bounds = {joint["name"]: joint["bounds"] for joint in model.joints if joint["name"] in model.arm}
    lower, upper = (np.array([bounds[name][index] for name in model.arm]) for index in (0, 1))
    q = np.clip(feedback.vector(model.arm), lower, upper)
    spheres, valid = model.spheres(feedback, np.repeat(q[None], len(poses), axis=0), hand=poses)
    if not valid.all():
        raise GraspFailure(
            "PLACE_PLAN_FAILED", "the held hand, the other arm or the head is outside the model joint limits"
        )
    palm = model.last_link_frames[:, model.link_indices["R_palm"]]
    moving = np.array([name.startswith("R_") for name in model.sphere_links])
    local = np.einsum("nsj,njk->nsk", spheres[:, moving, :3] - palm[:, None, :3, 3], palm[:, :3, :3])
    radii = spheres[0, moving, 3]
    reference, swept = local[-1], local[:-1].reshape(-1, 3)
    swept_r = np.tile(radii, len(hands))
    distance = np.linalg.norm(swept[:, None] - reference[None], axis=2)
    outside = ~(distance + swept_r[:, None] <= radii[None] + 1e-4).any(axis=1)
    if not outside.any():
        return np.empty((0, 4))
    points, point_r = swept[outside], swept_r[outside] + margin_m
    _, label = np.unique(np.floor(points / voxel_m).astype(int), axis=0, return_inverse=True)
    balls = []
    for cell in range(label.max() + 1):
        member, member_r = points[label == cell], point_r[label == cell]
        center = member.mean(axis=0)
        balls.append(
            {"center": center.tolist(), "radius": float(np.max(np.linalg.norm(member - center, axis=1) + member_r))}
        )
    low, high = 0.0, 0.15
    cover = merge_sphere_cover(balls, max_radius_m=high)
    if len(cover) > slots:
        raise GraspFailure("PLACE_PLAN_FAILED", "hand cover does not fit the reserved spheres")
    while high - low > 0.005:
        cap = (low + high) / 2
        candidate = merge_sphere_cover(balls, max_radius_m=cap)
        if len(candidate) <= slots:
            high, cover = cap, candidate
        else:
            low = cap
    return np.asarray([[*ball["center"], ball["radius"]] for ball in cover], dtype=float)


def held_return_path(held):
    """Pick's recorded way from the handover back home, or None."""
    if not held.return_times:
        return None
    rows = len(held.return_times)
    return TimedTrajectory(
        np.asarray(held.return_times, float),
        np.asarray(held.return_positions, float).reshape(rows, -1),
        tuple(held.joint_names),
    )


def placement_cover(model, feedback, held, *, slots=ATTACH_SLOTS):
    """Palm-frame Nx4 spheres attached while carrying: the held hand's filler, then the object (last)."""
    filler = _hand_cover(model, feedback, held.arm_side, feedback.hand(held.arm_side), slots - 1)
    cloud = np.asarray(held.points_in_palm).reshape(-1, 3)
    return np.vstack([filler, [[*held_center(cloud), float(held.radius_m)]]])


def opening_cover(model, feedback, held, *, slots=ATTACH_SLOTS, samples=6):
    """Palm-frame spheres swept while the hand opens at the release pose (the object is gone)."""
    hands = np.linspace(feedback.hand(held.arm_side), model.mapping.preset("open"), samples + 1)[:-1]
    return _hand_cover(model, feedback, held.arm_side, hands, slots)


class X2PlacePlanner:
    def __init__(self, head, robots):
        self.head, self.robots = head, robots

    def close(self):
        """Nothing of its own: placement borrows each arm's grasp planner."""

    def plan(self, observation, container, held, *, deadline, cancelled):
        side = held.arm_side
        parts = self.head.arms[side] if hasattr(self.head, "arms") else self.head
        robot = self.robots[side]
        admitted = observation.robot_state
        feedback = admitted.live.feedback

        def check():
            if cancelled() or time.monotonic() >= deadline:
                raise GraspFailure("CANCELED" if cancelled() else "SKILL_TIMEOUT", "place planning interrupted")

        check()
        if tuple(held.joint_names) != tuple(parts.spheres.arm) or not np.allclose(
            feedback.vector(held.joint_names), held.joint_positions, atol=0.02, rtol=0
        ):
            raise GraspFailure("HELD_OBJECT_MISMATCH", "holding arm moved from the handover pose")
        try:
            measured_spheres(parts.spheres, feedback)
            palm = np.eye(4)
            palm[:3] = parts.spheres.last_link_frames[0, parts.spheres.link_indices["R_palm"]]
            targets = place_targets(container, held, palm, robot.place)
            attached = placement_cover(parts.spheres, feedback, held)
            opening = opening_cover(parts.spheres, feedback, held)
            with parts.gpu:
                # The arm's own warmed grasp planner, the held object attached
                # in place: a carry neither rebuilds nor re-warms the planner.
                with parts.service.session(feedback, container.scene, attached=attached) as planner:
                    check()
                    paths = planner.plan_place(
                        feedback,
                        targets,
                        held.home_positions,
                        admitted.admitted_at,
                        check_deadline=check,
                        deadline=time.perf_counter() + max(0, deadline - time.monotonic()),
                        opening=opening,
                        return_path=held_return_path(held),
                    )
                    selected = dict(getattr(planner, "place_report", {}))
                report = {
                    "arm_side": side,
                    "attachment": ATTACH_LINK,
                    "attached_spheres": len(attached),
                    "opening_spheres": len(opening),
                    "candidate_count": len(targets),
                    **selected,
                }
                report["max_sag_rad"] = require_executable(
                    {str(i): path.positions for i, path in enumerate(paths)}, robot.limits, robot.gravity
                )
            # A plan may take seconds: recapture and recheck the locked posture
            # before returning it. Grip is checked by the pipeline throughout.
            current = parts.camera.capture_state()
            require_fresh(current, parts.camera.clock(), robot.planning.revalidation)
            require_posture(feedback, current.feedback, robot.planning.revalidation, side=side)
            check()
            transfer, descent, retreat, home = paths
            return PlacePlan(
                (Move(segment("transfer", transfer, side)), Move(segment("descend", descent, side))),
                (Effect(EffectorAction.RELEASE),),
                (Move(segment("retreat", retreat, side)), Move(segment("home", home, side))),
                container,
                report,
            )
        except GraspFailure:
            raise
        except (PlacementSearchExhausted, TimeoutError, ReobserveRequired) as exc:
            raise GraspFailure("PLACE_PLAN_FAILED", str(exc), retryable=True) from exc
        except (RuntimeError, ValueError, PlannerFault) as exc:
            raise GraspFailure("PLACE_PLAN_FAILED", str(exc)) from exc

    def plan_home(self, held, *, deadline, cancelled):
        """The empty, open hand's way home after a drop, planned on a fresh capture.

        A fresh capture puts the dropped object, wherever it landed, into the
        collision scene. The holding arm's grasp planner models exactly that
        hand: open and empty, as at its warmup.
        """
        side = held.arm_side
        parts = self.head.arms[side] if hasattr(self.head, "arms") else self.head
        robot = self.robots[side]

        def check():
            if cancelled() or time.monotonic() >= deadline:
                raise GraspFailure("CANCELED" if cancelled() else "SKILL_TIMEOUT", "home planning interrupted")

        check()
        try:
            observation = parts.camera.capture(deadline=deadline, cancelled=cancelled)
            imu = getattr(self.head, "imu", None)
            if imu is not None:
                imu.pause_after(observation.stamp_ns)  # as after any head capture
            feedback = observation.robot_state.live.feedback
            check()
            with parts.gpu:
                prepared = prepare_capture_scene(observation, parts.spheres, robot.perception.scene)
                scene = complete_scene(prepared, None, lazy_validation=False)
                check()
                with parts.service.session(feedback, scene) as planner:
                    path = planner.plan_home(
                        feedback,
                        held.home_positions,
                        return_path=held_return_path(held),
                        check_deadline=check,
                        deadline=time.perf_counter() + max(0, deadline - time.monotonic()),
                    )
            require_executable({"home": path.positions}, robot.limits, robot.gravity)
            check()
            return Move(segment("home", path, side))
        except GraspFailure:
            raise
        except (RuntimeError, ValueError, TimeoutError, PlannerFault, NoSupportError) as exc:
            raise GraspFailure("HOME_PLAN_FAILED", str(exc)) from exc
