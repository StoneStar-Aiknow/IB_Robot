"""Held-hand placement planning; independent of pick's open-hand admission."""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction, GraspFailure, Move, PlacePlan

from .arm_gravity import require_executable
from .collision_spheres import merge_sphere_cover_steps, steps_at
from .grasp_request import segment
from .held_object import held_center
from .inputs_record import INPUTS_DIR_ENV, inputs_directory, save_record
from .place_targets import place_targets
from .planner import ATTACH_LINK, ATTACH_SLOTS, PlacementSearchExhausted, PlannerFault
from .proposer import measured_spheres, prepare_capture_scene
from .revalidation import ReobserveRequired, require_fresh, require_posture
from .scene_estimation import NoSupportError, complete_scene
from .trajectory import TimedTrajectory

#: The grasp head's plan-input directory (inputs_record): each place plan's
#: inputs and outcome are pickled there as place_<ns>.pkl.
PLACE_INPUTS_DIR_ENV = INPUTS_DIR_ENV
#: The camera's own profile of the recheck capture (HeadCamera.calls), as logged.
RECHECK_CAPTURE_FIELDS = (
    "rounds",
    "wait_s",
    "side_drain_s",
    "match_s",
    "gpu_wait_s",
    "registration_s",
    "feedback_wait_s",
    "total_s",
    "image_age_s",
)


def save_place_inputs(directory: Path, recorded: dict) -> threading.Thread:
    """One place plan's inputs and outcome as place_<ns>.pkl (inputs_record.save_record).

    Everything recorded is already built and no longer changed by the planner;
    the place scene is never lazy (estimate_container, lazy_validation=False).
    """
    return save_record(directory, "place", recorded)


def _inside_any(points, point_r, balls, ball_r, tolerance, *, groups=8):
    """Whether each point ball lies inside some ball: ``|p - b| + r_p <= r_b + tolerance``.

    The same test and arithmetic as over the full point x ball matrix, evaluated
    only on the pairs within reach of each ball. A KD-tree finds them per radius
    group (the hand's radii span 0.9-46 mm), so a group of small balls is not
    searched at the largest ball's reach. For the opening sweep the full matrix was
    2676 x 446 (39 ms on x86); this checks ~37k pairs (4.5 ms).
    """
    from scipy.spatial import cKDTree

    inside = np.zeros(len(points), dtype=bool)
    edges = np.quantile(ball_r, np.linspace(0.0, 1.0, groups + 1))
    tree = cKDTree(points)
    for group in range(groups):
        upper = ball_r < edges[group + 1] if group < groups - 1 else True
        members = np.nonzero((ball_r >= edges[group]) & upper)[0]
        if not len(members):
            continue
        # Slack far above the trees' own rounding: a missed passing pair would change the cover.
        reach = float(np.max(ball_r[members])) + tolerance + 1e-6
        pairs = tree.sparse_distance_matrix(cKDTree(balls[members]), reach, output_type="ndarray")
        i, j = pairs["i"], members[pairs["j"]]
        diff = points[i] - balls[j]
        distance = np.sqrt(np.add.reduce(diff * diff, axis=1))
        inside[i[distance + point_r[i] <= ball_r[j] + tolerance]] = True
    return inside


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
    outside = ~_inside_any(swept, swept_r, reference, radii, 1e-4)
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
    # One greedy merge at the widest cap; every narrower cap's cover is one of its steps.
    states, radii = merge_sphere_cover_steps(balls, max_radius_m=high)
    cover = steps_at(states, radii, high)
    if len(cover) > slots:
        raise GraspFailure("PLACE_PLAN_FAILED", "hand cover does not fit the reserved spheres")
    while high - low > 0.005:
        cap = (low + high) / 2
        candidate = steps_at(states, radii, cap)
        if len(candidate) <= slots:
            high, cover = cap, candidate
        else:
            low = cap
    return np.array(cover, dtype=float)


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
        # Where a place plan's seconds go, for successes and failures alike:
        # one JSON line on stderr (the head log) and, on success, the report.
        started = time.perf_counter()
        timings: dict = {}
        trace = {"outcome": "error", "planner": {}}
        # Inputs for an offline replay of this plan (place_replay), when the head records plan inputs.
        directory = inputs_directory()
        recorded = (
            dict(
                kind="place",
                arm_side=side,
                feedback=feedback,
                container=container,
                held=held,
                place_config=robot.place,
                admitted_at=admitted.admitted_at,
                budget_s=deadline - time.monotonic(),
            )
            if directory
            else None
        )

        def lap(name, since):
            now = time.perf_counter()
            timings[name] = round(now - since, 4)
            return now

        try:
            tick = time.perf_counter()
            parts_s = timings["targets_parts"] = {}
            clock = [tick]

            def part(name):
                now = time.perf_counter()
                parts_s[name] = round(now - clock[0], 4)
                clock[0] = now

            measured_spheres(parts.spheres, feedback)
            palm = np.eye(4)
            palm[:3] = parts.spheres.last_link_frames[0, parts.spheres.link_indices["R_palm"]]
            if recorded is not None:
                recorded["palm"] = palm.copy()  # place_targets' input, also when it refuses the container
            part("spheres")
            targets = place_targets(container, held, palm, robot.place)
            part("place_targets")
            attached = placement_cover(parts.spheres, feedback, held)
            part("placement_cover")
            opening = opening_cover(parts.spheres, feedback, held)
            part("opening_cover")
            if recorded is not None:
                recorded.update(targets=targets, attached=attached, opening=opening)
            tick = lap("targets", tick)
            with parts.gpu:
                tick = lap("gpu_wait", tick)
                # The arm's own warmed grasp planner, the held object attached
                # in place: a carry neither rebuilds nor re-warms the planner.
                with parts.service.session(feedback, container.scene, attached=attached) as planner:
                    tick = lap("session", tick)
                    check()
                    try:
                        paths = planner.plan_place(
                            feedback,
                            targets,
                            held.home_positions if getattr(robot.place, "plan_return_home", True) else None,
                            admitted.admitted_at,
                            check_deadline=check,
                            deadline=time.perf_counter() + max(0, deadline - time.monotonic()),
                            opening=opening,
                            return_path=held_return_path(held),
                        )
                    finally:
                        tick = lap("plan_place", tick)
                        selected = dict(getattr(planner, "place_report", {}))
                        trace["planner"] = selected
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
                tick = lap("executable", tick)
            # A plan may take seconds: recapture and recheck the locked posture
            # before returning it. Grip is checked by the pipeline throughout.
            recheck = timings["recheck_parts"] = {}
            current = parts.camera.capture_state()
            began = time.perf_counter()
            recheck["capture"] = round(began - tick, 4)
            stamp = current.observation.transform.stamp_ns
            # This capture's own entry: the watch captures on another thread meanwhile.
            own = next((call for call in reversed(parts.camera.calls) if call.get("stamp_ns") == stamp), {})
            recheck["camera"] = {key: round(own[key], 4) for key in RECHECK_CAPTURE_FIELDS if key in own}
            require_fresh(current, parts.camera.clock(), robot.planning.revalidation)
            require_posture(feedback, current.feedback, robot.planning.revalidation, side=side)
            check()
            recheck["checks"] = round(time.perf_counter() - began, 4)
            lap("recheck", tick)
            timings["total"] = round(time.perf_counter() - started, 4)
            report["plan_timings_s"] = dict(timings)
            trace["outcome"] = "ok"
            transfer, descent, retreat, *home = paths
            return PlacePlan(
                (Move(segment("transfer", transfer, side)), Move(segment("descend", descent, side))),
                (Effect(EffectorAction.RELEASE),),
                (Move(segment("retreat", retreat, side)), *(Move(segment("home", path, side)) for path in home)),
                container,
                report,
            )
        except GraspFailure as exc:
            trace["outcome"] = exc.code
            raise
        except (PlacementSearchExhausted, TimeoutError, ReobserveRequired) as exc:
            trace["outcome"] = type(exc).__name__
            raise GraspFailure("PLACE_PLAN_FAILED", str(exc), retryable=True) from exc
        except (RuntimeError, ValueError, PlannerFault) as exc:
            trace["outcome"] = type(exc).__name__
            raise GraspFailure("PLACE_PLAN_FAILED", str(exc)) from exc
        finally:
            timings.setdefault("total", round(time.perf_counter() - started, 4))
            print(
                json.dumps({"place_plan": {"arm_side": side, **trace, "timings_s": timings}}, default=str),
                file=sys.stderr,
                flush=True,
            )
            if recorded is not None:
                # Also a refusal before planning (CONTAINER_TOO_SMALL / OUT_OF_REACH from place_targets).
                recorded["outcome"] = dict(trace, timings_s=timings)
                save_place_inputs(directory, recorded)

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
