"""Persistent, serialized planning service shared by local/ROS request handlers."""

import os
import pickle
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np

from manipulation_execution.grasp_pipeline.candidates import (
    MIN_ENCLOSED_POINTS,
    approach_priority,
    enclosed_points,
    sweep_boxes,
)

from .grasp_session import SIDE_FIRST_LONGEST_APPROACH, check_approach_order, select_plan
from .joints import check_side
from .planner import LOCKED_SPHERE_SHIFT_M, LeftArmPlanner, PlannerFault, device_fault

# A directory: every plan call's inputs and outcome are pickled there (plan_replay replays them).
PLAN_INPUTS_DIR_ENV = "IBROBOT_X2_PLAN_INPUTS_DIR"


def save_plan_inputs(directory: Path, recorded: dict, estimate, validation_scene) -> Path:
    """Write one plan call with its already-materialized fine scene.

    The caller materializes the scene under the shared GPU lock. Recording is
    synchronous after planning, so its scene construction and file write add to
    the request time; the pickle keeps the validation geometry as a plain mapping.
    """
    recorded = dict(recorded, estimate=replace(estimate, validation_scene=dict(validation_scene)))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"plan_{time.time_ns()}.pkl"
    with open(path, "wb") as stream:
        pickle.dump(recorded, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _report_rebuild(planner, feedback) -> None:
    """Say on stderr why the planner is about to be rebuilt, before it costs the time.

    A rebuild is the most expensive thing this service does and the only one
    with no other trace: warmup took 47.6-49.9 s on the Orin (2026-10-26/27),
    `_plan` cannot preempt it inside the request deadline, and the request it
    lands in reports whatever gave up first -- a plan timeout, or an RPC
    timeout on another head call. Live on the robot this one line named the
    cause at once: ``hand=0.4623 <-- over`` (then in radians) after a failed
    release left the hand half closed, with the right arm and the head still.
    """
    if planner is None:
        print("planner rebuild: first build", file=sys.stderr, flush=True)
        return
    try:
        moved = planner.locked_shift_m(feedback)
    except BaseException as exc:  # noqa: BLE001 - a diagnostic must not change what the request does
        print(f"planner rebuild: cause unavailable: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return
    limit_mm = LOCKED_SPHERE_SHIFT_M * 1000.0
    detail = ", ".join(
        f"{group}={value * 1000.0:.1f}mm{'  <-- over' if value > LOCKED_SPHERE_SHIFT_M else ''}"
        for group, value in moved.items()
    )
    print(f"planner rebuild: locked spheres moved (limit {limit_mm:g} mm): {detail}", file=sys.stderr, flush=True)


# Candidates per cuRobo goalset. Screening keeps its full capacity; only the
# final choice is limited to the best-ordered few (see select_plan).
GOALSET_GROUP = 5
# Approach distance (m) of the screened-candidate reachability ranking; every
# live plan on 2026-10-09 used the first, 15 cm approach. None disables it.
APPROACH_REACH_OFFSET_M = -0.15
# Rank screened candidates by LeftArmPlanner.grasp_line_clear: this many, in
# planning order (the ones the first attempts would use). 0 disables it.
GRASP_LINE_CANDIDATES = 10
# Of those, candidates whose open hand ends closer than this (m) to the target
# points at contact go last within their tier, in goalsets of their own (kept,
# never removed). Live 2026-10-14: the two misses of eight still picks were the
# palm pushing the bottle, planned 5.6 and 7.0 mm from it; the real hand reached
# about 10 mm (p90) beyond its spheres in the watch frames. Replay of 134 live
# inputs: 128 planned either way, winners this tight 27 -> 6, +5 ms median
# (one request +0.44 s); as a tier of their own 7, but five requests +0.4-0.9 s.
# None disables it.
TARGET_CLEARANCE_M = 0.01
# Measure that clearance along the whole grasp line, not only at contact (live
# 2026-10-14 clr_3: pinky pushed the bottle halfway, 12.5 mm clear at contact).
# Along the planned segment the 3 pushes of 13 still picks that day were 5.6-7.0
# mm clear, the 10 grips 11.4 mm or more. Replay of 141 live inputs with the
# fallback below: 135/135 planned, median plan time unchanged, attempts 360 -> 322.
TARGET_CLEARANCE_LINE = True
# Whether those later candidates get goalsets of their own (mixing them: replay
# winners this tight 16 -> 60 of 270), and whether a tier that failed so is
# tried once more in mixed goalsets.
TARGET_CLEARANCE_ALONE = True
TARGET_CLEARANCE_MIXED_RETRY = True
# Candidates below this confidence go last within their tier, after the tight
# ones, in goalsets of their own (kept, never removed). Still picks to the lift,
# by the executed candidate's confidence: >= 0.85 31/32 gripped, 0.7-0.85 9/11,
# below 0.7 2/5 (live 2026-10-15 n_mask s3: 0.36, the thumb closed on air).
# Replay of 158 live inputs on the 3090: 302/316 planned either way, winners
# below 0.7 54 -> 8, +12 ms median. As a tier after all others instead: same
# successes, but s3 took 44 attempts (5.3 s) to reach the same 0.36 candidate,
# the only one that plans there. None disables it.
LOW_CONFIDENCE = 0.7
# Candidates whose palm faces the robot's left (OmniHand grasp frame: the
# fingers curl toward +x, so the palm faces +x; base-frame y of that axis above
# this) go after the tight ones of their tier, before the low-confidence ones,
# in goalsets of their own (kept, never removed). Not a success-rate rule: a
# human-like preference. Closed live picks 10-12..10-18 gripped 18/23 palm-left
# vs 34/44 otherwise; removing them lost live st2 (bottle at 0.40, 0.04: only
# palm-left grasps plan there) in a replay of 169 inputs. None disables it.
PALM_LEFT_Y = 0.707
# When every candidate reachable at APPROACH_REACH_OFFSET_M has a blocked grasp
# line, the unreachable ones are checked again at this approach (m) and count as
# reachable there (live 2026-10-14 clr_1: four blocked candidates failed 13
# attempts, 8.9 s on the Orin, before the first unreachable one, which planned at
# -0.10 m; replay 2.6 -> 0.6 s on the 3090, 12.6 s on the Orin). Used by 4 of 141
# replayed inputs. None disables it.
APPROACH_REACH_FALLBACK_M = -0.10


class GraspPlanningService:
    """Own one warmed planner of one arm (``side`` planner option) across requests and RGB-D scene updates.

    Call prepare at startup, then submit fresh SceneEstimates to plan. Locked
    head/other-arm/hand changes rebuild the robot snapshot; planning-arm motion and
    mesh updates reuse the existing solvers. Two arms are two services. All GPU calls are serialized.
    ``gripper`` is the deployed grasp model's gripper description
    (hand.load_gripper_description): its palm mount goes to the planner, its
    closing sweep to the enclosure check.
    """

    def __init__(
        self,
        *,
        gripper,
        max_goalset=100,
        max_batch_size=None,
        clock=time.monotonic,
        planner_factory: Callable = LeftArmPlanner,
        planning_profile="default",
        approach_seed_policy="clear_path",
        approach_over_height_m=0.18,
        approach_order_m=None,
        gpu_lock=None,
        **planner_options,
    ):
        if planning_profile not in ("default", "latency"):
            raise ValueError("Unknown planning profile")
        if approach_seed_policy not in ("native", "clear_path"):
            raise ValueError("Unknown approach seed policy")
        if (
            isinstance(approach_over_height_m, bool)
            or not isinstance(approach_over_height_m, int | float)
            or not np.isfinite(approach_over_height_m)
            or approach_over_height_m <= 0
        ):
            raise ValueError("Finite positive approach_over_height_m required")
        if max_batch_size is None:
            max_batch_size = 1 if planning_profile == "latency" else 10
        if planning_profile == "latency" and max_batch_size != 1:
            raise ValueError("The latency profile requires max_batch_size=1 (direct goalsets)")
        self.planning_profile = planning_profile
        self.approach_seed_policy = approach_seed_policy
        self.approach_order_m = check_approach_order(approach_order_m)
        #: The arm this service plans; its planner is built on that side's planning model.
        self.side = check_side(planner_options.get("side", "left"))
        self._sweep = sweep_boxes(gripper)
        self._options: dict = dict(
            gripper=gripper,
            max_goalset=max_goalset,
            max_batch_size=max_batch_size,
            planning_profile=planning_profile,
            approach_seed_policy=approach_seed_policy,
            approach_over_height_m=float(approach_over_height_m),
            **planner_options,
        )
        self._clock = clock
        self._factory = planner_factory
        self._planner = None
        self._estimate = None
        # Every planner call runs under it. A process whose other threads also
        # use the GPU passes one re-entrant lock they share (head_parts): cuRobo
        # captures CUDA graphs on (re)build and first use, and any allocation or
        # synchronization elsewhere in the process during a capture fails.
        self._lock = RLock() if gpu_lock is None else gpu_lock

    def prepare(self, feedback, estimate, *, carrying=False):
        """Initialize/prewarm or replace scene buffers, preserving depth age.

        ``carrying``: the hand holds an object that the caller attaches; its
        closed fingers are then covered by the attachment, not a rebuild.
        """
        with self._lock:
            if self._planner is not None and getattr(self._planner, "_closed", False):
                self.close()  # it closed itself after a failed scene load: rebuild now
            if self._planner is None or not self._locked_match(feedback, carrying):
                _report_rebuild(self._planner, feedback)
                if self._planner is not None:
                    self.close()
                self._planner = self._factory(feedback, estimate, estimate.received_at, **self._options)
                try:
                    self._planner.warmup()
                except BaseException:
                    self.close()
                    raise
            elif estimate is not self._estimate:
                try:
                    if carrying:
                        self._planner.update_scene(feedback, estimate, carrying=True)
                    else:
                        self._planner.update_scene(feedback, estimate)
                except BaseException:
                    self.close()
                    raise
            self._estimate = estimate

    def preload(self, feedback, estimate):
        """prepare plus the fine scene, ahead of plan (e.g. while grasp candidates are still inferred).

        plan with the same estimate then finds both in place; the fine mesh
        would otherwise be built and installed at its first fine check.
        """
        with self._lock:
            self.prepare(feedback, estimate)
            load = getattr(self._planner, "load_validation_scene", None)
            if load is not None:
                load()

    def needs_rebuild(self, feedback):
        """Whether the next empty-hand request would rebuild the planner; None while the service is busy.

        Never waits for the lock: it is held through every plan, and an idle
        check must not queue behind one.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            planner = self._planner
            return planner is None or getattr(planner, "_closed", False) or not planner.locked_joints_match(feedback)
        finally:
            self._lock.release()

    def refresh(self, feedback, estimate):
        """Rebuild between requests, so the rebuild does not land inside the next one.

        Returns whether it rebuilt. Like ``needs_rebuild``, it gives up at once
        (False) when a request holds the service.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not self.needs_rebuild(feedback):
                return False
            self.prepare(feedback, estimate)
            return True
        finally:
            self._lock.release()

    def _locked_match(self, feedback, carrying):
        if not carrying:
            return self._planner.locked_joints_match(feedback)
        return self._planner.locked_joints_match(feedback, carrying=True)

    @contextmanager
    def session(self, feedback, estimate, *, attached=None):
        """Hold exclusive access to one prepared snapshot through replay/checks.

        The yielded planner is borrowed only for the lifetime of this context;
        scene replacement and service shutdown wait until the caller exits.
        ``attached`` (palm-frame Nx4, the object last) is attached for the
        session and always detached at its end, so the next pick plans the
        empty hand.
        """
        with self._lock:
            self.prepare(feedback, estimate, carrying=attached is not None)
            assert self._planner is not None
            if attached is None:
                yield self._planner
                return
            planner = self._planner
            planner.attach_object(attached)
            try:
                yield planner
            finally:
                planner.detach_object()

    def plan(self, feedback, estimate, camera_from_grasps, confidence, base_from_camera, **options):
        """_plan, its inputs and outcome kept under $IBROBOT_X2_PLAN_INPUTS_DIR when set (plan_replay)."""
        directory = os.environ.get(PLAN_INPUTS_DIR_ENV)
        if not directory:
            return self._plan(feedback, estimate, camera_from_grasps, confidence, base_from_camera, **options)
        deadline = options.get("deadline")
        recorded = dict(
            feedback=feedback,
            camera_from_grasps=np.asarray(camera_from_grasps),
            confidence=np.asarray(confidence),
            base_from_camera=np.asarray(base_from_camera),
            options={key: value for key, value in options.items() if key not in ("deadline", "checkpoint")},
            budget_s=None if deadline is None else deadline - time.perf_counter(),
            service=dict(
                planning_profile=self.planning_profile,
                approach_seed_policy=self.approach_seed_policy,
                approach_over_height_m=self._options["approach_over_height_m"],
                approach_order_m=self.approach_order_m,
                max_goalset=self._options["max_goalset"],
            ),
        )
        began = time.perf_counter()
        try:
            result = self._plan(feedback, estimate, camera_from_grasps, confidence, base_from_camera, **options)
            recorded["outcome"] = dict(success=True, candidate_index=result[1].get("candidate_index"))
            return result
        except BaseException as exc:
            recorded["outcome"] = dict(success=False, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            recorded["outcome"]["wall_s"] = time.perf_counter() - began
            try:
                from .scene_estimation import LazyScene

                validation = estimate.validation_scene
                if isinstance(validation, LazyScene):
                    with self._lock:
                        validation = validation.materialize()
                save_plan_inputs(Path(directory), recorded, estimate, validation)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must never change the plan's outcome
                print(f"plan inputs not saved: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def _plan(
        self,
        feedback,
        estimate,
        camera_from_grasps,
        confidence,
        base_from_camera,
        *,
        screen_check=False,
        deadline=None,
        snapshot_now=None,
        target_points_camera=None,
        checkpoint=lambda: None,
    ):
        """Plan a goalset, optionally bounded by an absolute perf_counter deadline.

        Prepare/warm at startup. Deadline checks include lock wait and preparation,
        but cannot preempt an in-flight GPU call or planner initialization.
        With target_points_camera, candidates whose closing sweep holds fewer
        than MIN_ENCLOSED_POINTS target points are never planned. checkpoint runs
        after every attempt and may raise (e.g. cancellation) to stop the search.
        """
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("Finite monotonic planning deadline required")
        # A snapshot solve is geometry-only. The request owner validates intake
        # against real time and rechecks the target afterwards; the executor
        # checks feedback and robot state, not the scene.
        now = self._clock if snapshot_now is None else lambda: snapshot_now
        if snapshot_now is not None and not np.isfinite(snapshot_now):
            raise ValueError("Finite admitted snapshot time required")
        grasps = np.asarray(camera_from_grasps, dtype=float)
        scores = np.asarray(confidence, dtype=float)
        if (
            grasps.ndim != 3
            or grasps.shape[1:] != (4, 4)
            or not len(grasps)
            or scores.shape != (len(grasps),)
            or not np.isfinite(scores).all()
            or not np.isfinite(grasps).all()
        ):
            raise ValueError("Expected nonempty Kx4x4 grasps and K finite confidence scores")
        with self._lock:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Planning deadline exceeded")
            before = time.perf_counter()
            self.prepare(feedback, estimate)
            assert self._planner is not None
            prepared = time.perf_counter()
            if deadline is not None and prepared >= deadline:
                raise TimeoutError("Planning deadline exceeded")
            report: dict = {
                "selection_mode": "screen_then_goalset" if screen_check else "direct_goalset",
                "planning_profile": self.planning_profile,
                "approach_seed_policy": self.approach_seed_policy,
                "planning_clock": "source_clock" if snapshot_now is None else "admitted_snapshot",
                "execution_authorized": False,
            }
            screened = None
            # Per-request counters the planner creates on first use.
            for name, empty in (("fine_rescue_report", {}), ("linear_grasp_report", {}), ("vertical_lift_report", {})):
                if hasattr(self._planner, name):
                    setattr(self._planner, name, empty)
            try:
                screen_options = {} if deadline is None else {"deadline": deadline}
                feasible = (
                    self._planner.screen_grasps(feedback, grasps, base_from_camera, now(), **screen_options)
                    if screen_check
                    else None
                )
                screened = time.perf_counter()
                if target_points_camera is not None:
                    # Only screened candidates can be planned; count just those
                    # (all of them without a screen). Uncounted ones read -1.
                    counts = enclosed_points(
                        grasps,
                        target_points_camera,
                        self._sweep,
                        None if feasible is None else np.flatnonzero(feasible),
                    )
                    enclosing = counts >= MIN_ENCLOSED_POINTS
                    report["enclosure"] = {
                        "min_points": MIN_ENCLOSED_POINTS,
                        "rejected": np.flatnonzero(~enclosing & (counts >= 0)).tolist(),
                        "counts": counts.tolist(),
                    }
                    feasible = enclosing if feasible is None else np.asarray(feasible) & enclosing
                if feasible is not None:
                    report["screened_candidates"] = np.flatnonzero(feasible).tolist()
                # No support plane (or no target on it): confidence order only.
                support = getattr(estimate, "support", None)
                # The vertical lift rises off the support (straight up without one).
                self._planner.lift_up = (0.0, 0.0, 1.0) if support is None else tuple(map(float, support.normal))
                priority, report["approach_priority"] = approach_priority(
                    grasps,
                    base_from_camera,
                    [0.0, 0.0, 1.0] if support is None else support.normal,
                    None if support is None else support.target_heights_m.get("p95"),
                )
                if report["approach_priority"]["excluded"]:
                    allowed = np.ones(len(grasps), dtype=bool)
                    allowed[report["approach_priority"]["excluded"]] = False
                    feasible = allowed if feasible is None else np.asarray(feasible) & allowed
                    if "screened_candidates" in report:
                        report["screened_candidates"] = np.flatnonzero(feasible).tolist()
                if (
                    feasible is not None
                    and APPROACH_REACH_OFFSET_M is not None
                    and hasattr(self._planner, "approach_reachable")
                ):
                    # Candidates whose first approach pose the screen IK cannot
                    # reach go after all others (kept, never removed).
                    began = time.perf_counter()
                    indices = np.flatnonzero(feasible)
                    reachable = np.ones(len(grasps), dtype=bool)
                    reachable[indices] = self._planner.approach_reachable(
                        feedback, grasps[indices], base_from_camera, APPROACH_REACH_OFFSET_M
                    )
                    priority = np.asarray(priority) + 4 * ~reachable
                    report["approach_reachable"] = {
                        "offset_m": APPROACH_REACH_OFFSET_M,
                        "unreachable": np.flatnonzero(~reachable).tolist(),
                        "wall_s": time.perf_counter() - began,
                    }
                later = None
                if feasible is not None and GRASP_LINE_CANDIDATES and hasattr(self._planner, "grasp_line_clear"):
                    # Candidates whose open hand skims the scene on the way from
                    # approach to contact go after the clear ones, before the
                    # unreachable ones (kept, never removed).
                    began = time.perf_counter()
                    indices = np.array(
                        sorted(np.flatnonzero(feasible), key=lambda i: (priority[i], -scores[i]))[
                            :GRASP_LINE_CANDIDATES
                        ],
                        dtype=int,
                    )
                    clear = np.ones(len(grasps), dtype=bool)
                    matrix = np.asarray(base_from_camera, dtype=float)
                    points = (
                        None
                        if target_points_camera is None
                        else np.asarray(target_points_camera, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
                    )
                    # The line and its target clearance from one kinematics pass (planner.grasp_line_checks).
                    merged = (
                        TARGET_CLEARANCE_M is not None
                        and TARGET_CLEARANCE_LINE
                        and points is not None
                        and hasattr(self._planner, "grasp_line_checks")
                    )
                    clearance = np.full(len(grasps), np.inf)
                    if len(indices) and merged:
                        clear[indices], clearance[indices] = self._planner.grasp_line_checks(
                            indices, grasps[indices], base_from_camera, points
                        )
                    elif len(indices):
                        clear[indices] = self._planner.grasp_line_clear(indices, grasps[indices], base_from_camera)
                    priority = np.asarray(priority) + 2 * ~clear
                    report["grasp_line"] = {
                        "checked": indices.tolist(),
                        "blocked": np.flatnonzero(~clear).tolist(),
                        "wall_s": time.perf_counter() - began,
                    }
                    unreachable = np.flatnonzero(feasible & ~reachable) if "approach_reachable" in report else []
                    if (
                        APPROACH_REACH_FALLBACK_M is not None
                        and len(unreachable)
                        and not (feasible & reachable & clear).any()
                    ):
                        began = time.perf_counter()
                        rescued = unreachable[
                            self._planner.approach_reachable(
                                feedback, grasps[unreachable], base_from_camera, APPROACH_REACH_FALLBACK_M
                            )
                        ]
                        priority = np.asarray(priority).copy()
                        priority[rescued] -= 4
                        report["approach_reachable"]["fallback"] = {
                            "offset_m": APPROACH_REACH_FALLBACK_M,
                            "reachable": rescued.tolist(),
                            "wall_s": time.perf_counter() - began,
                        }
                    if (
                        TARGET_CLEARANCE_M is not None
                        and target_points_camera is not None
                        and len(indices)
                        and (merged or hasattr(self._planner, "contact_target_clearance"))
                    ):
                        began = time.perf_counter()
                        if not merged:
                            line = (grasps[indices], base_from_camera) if TARGET_CLEARANCE_LINE else ()
                            clearance[indices] = self._planner.contact_target_clearance(indices, points, *line)
                        later = tight = clearance < TARGET_CLEARANCE_M
                        report["target_clearance"] = {
                            "min_m": TARGET_CLEARANCE_M,
                            "tight": np.flatnonzero(tight).tolist(),
                            "clearance_m": {int(i): float(clearance[i]) for i in indices if np.isfinite(clearance[i])},
                            "wall_s": time.perf_counter() - began,
                            "with_grasp_line": merged,
                        }
                if PALM_LEFT_Y is not None and feasible is not None:
                    # The right arm's preference is the mirror image: its palm facing the robot's right
                    # (mirroring the palm by x mirrors the grasp frame by its y, test_mirror_side).
                    rotation = np.asarray(base_from_camera, dtype=float)[:3, :3]
                    outward = (grasps[:, :3, 0] @ rotation.T)[:, 1] * (1.0 if self.side == "left" else -1.0)
                    left = np.asarray(feasible, dtype=bool) & (outward > PALM_LEFT_Y)
                    level = np.zeros(len(grasps), dtype=int) if later is None else np.asarray(later, dtype=int)
                    later = np.where(left, np.maximum(level, 2), level)
                    report[f"palm_{self.side}"] = {"min_y": PALM_LEFT_Y, "later": np.flatnonzero(left).tolist()}
                if LOW_CONFIDENCE is not None and feasible is not None:
                    low = np.asarray(feasible, dtype=bool) & (scores < LOW_CONFIDENCE)
                    later = np.where(low, 3, np.zeros(len(grasps), dtype=int) if later is None else later)
                    report["low_confidence"] = {"min": LOW_CONFIDENCE, "later": np.flatnonzero(low).tolist()}
                plan, index, _ = select_plan(
                    self._planner,
                    feedback,
                    SimpleNamespace(grasps=grasps, confidence=scores),
                    base_from_camera,
                    feasible,
                    report,
                    checkpoint,
                    now=now,
                    deadline=deadline,
                    priority=priority,
                    goalset_group=GOALSET_GROUP,
                    first_tier_longest_first=(
                        SIDE_FIRST_LONGEST_APPROACH and report["approach_priority"]["mode"] == "side_first"
                    ),
                    approach_order_m=self.approach_order_m,
                    later=later,
                    later_alone=TARGET_CLEARANCE_ALONE,
                    mixed_retry=TARGET_CLEARANCE_MIXED_RETRY,
                )
            except (RuntimeError, TimeoutError, PlannerFault) as exc:
                # Keep failed attempts and their timing available to callers;
                # a benchmark must report rejected plans as well as successes.
                exc.planning_report = report
                if isinstance(exc, PlannerFault) or (isinstance(exc, RuntimeError) and device_fault(exc)):
                    # A CUDA/Warp failure poisons the planner's graphs and
                    # buffers: drop it so the next request builds a new one.
                    report["planner_fault"] = str(exc)
                    try:
                        self.close()
                    except Exception:  # noqa: BLE001 - the fault above is the error to report
                        self._planner = self._estimate = None
                raise
            finally:
                finished = time.perf_counter()
                if screen_check and hasattr(self._planner, "contact_screen_report"):
                    report["contact_screen"] = dict(self._planner.contact_screen_report)
                if hasattr(self._planner, "fine_rescue_report"):
                    report["fine_rescues"] = dict(self._planner.fine_rescue_report)
                if hasattr(self._planner, "linear_grasp_report"):
                    report["linear_grasp"] = dict(self._planner.linear_grasp_report)
                if hasattr(self._planner, "vertical_lift_report"):
                    report["vertical_lift"] = dict(self._planner.vertical_lift_report)
                if screened is None:
                    screened = finished
                report["timings_s"] = {
                    "prepare": prepared - before,
                    "screen": screened - prepared,
                    "plan": finished - screened,
                }
            report["candidate_index"] = index
            return plan, report

    def close(self):
        with self._lock:
            planner = self._planner
            self._planner = self._estimate = None
            if planner is not None:
                planner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
