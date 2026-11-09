"""Describe the object in the palm frame at contact, using CPU forward kinematics."""

import json
import sys
import time
import uuid
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from manipulation_execution.grasp_pipeline.contracts import HeldObject, Move

from .robot_model import config_directory
from .robot_spheres import RobotSphereModel

#: Silhouette pixels can carry the depth of the surface behind the target. They
#: sit a few centimetres off the object, close enough to survive the pick's
#: cluster filter (largest_cluster_mask joins voxels up to ~4.7 cm apart), and the
#: held radius is a maximum: on cap3, 7 such points of 517 raised it from 0.056
#: to 0.109 m, so every basket opening was rejected as too small.
SURFACE_LINK_M = 0.015
#: A piece holding less than this share of the cloud is fringe, not object.
FRINGE_MAX_SHARE = 0.05


def without_fringe(local):
    """The object cloud without small pieces detached from it; never empties it.

    Pieces are linked at SURFACE_LINK_M, or at four times the cloud's own median
    point spacing when that is coarser, so a sparse cloud is not cut apart.
    """
    local = np.asarray(local, float)
    if len(local) < 2:
        return local
    tree = cKDTree(local)
    spacing = float(np.median(tree.query(local, k=2)[0][:, 1]))
    pairs = tree.query_pairs(max(SURFACE_LINK_M, 4.0 * spacing), output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(local),) * 2)
    _, labels = connected_components(graph, directed=False)
    keep = np.bincount(labels)[labels] >= FRINGE_MAX_SHARE * len(local)
    return local[keep] if keep.any() else local


#: Statistical outlier removal: a point whose mean distance to its OUTLIER_NEIGHBOURS
#: nearest points exceeds the cloud's mean by OUTLIER_STD standard deviations is
#: depth noise, not object. The held radius is a maximum, so a few such points set
#: it: live 2026-11-06, four picks of one apple (least-squares sphere 0.039-0.042 m)
#: gave enclosing radii 0.044-0.049 m and a bowl was refused once (held 0.063 m with
#: its margin). Without the 2-3 % such points: 0.041-0.042 m on all four.
OUTLIER_NEIGHBOURS = 16
OUTLIER_STD = 2.0


def without_outliers(local):
    """The cloud without isolated noisy points; never empties it, keeps small clouds whole."""
    local = np.asarray(local, float)
    if len(local) <= OUTLIER_NEIGHBOURS:
        return local
    spacing = cKDTree(local).query(local, k=OUTLIER_NEIGHBOURS + 1)[0][:, 1:].mean(axis=1)
    keep = spacing <= spacing.mean() + OUTLIER_STD * spacing.std()
    return local[keep] if keep.any() else local


#: Bădoiu–Clarkson steps; the centre is within ~radius/sqrt(steps) of optimal.
CENTER_STEPS = 1000


def held_center(points):
    """Centre of an approximately smallest sphere around the palm-frame object cloud.

    The camera sees one side of the object, so the cloud's mean sits towards
    that side and the farthest point from it overstates the object: an apple of
    about 4 cm radius measured 0.060 m from its mean, 0.045 m from this centre.
    Whatever centre is returned, a radius measured from it encloses every point;
    the mean is kept as a candidate, so this is never worse than the mean.
    Producer and consumers of the held object must use this same centre.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    center = points.mean(axis=0)
    best, best_reach = center, float(np.max(np.linalg.norm(points - center, axis=1)))
    for step in range(1, CENTER_STEPS + 1):
        farthest = points[np.argmax(np.linalg.norm(points - center, axis=1))]
        center = center + (farthest - center) / (step + 1)
        reach = float(np.max(np.linalg.norm(points - center, axis=1)))
        if reach < best_reach:
            best, best_reach = center, reach
    return best


def return_path(moves):
    """The executed pick moves reversed into one path from the handover back to their start.

    Only timed trajectories qualify; anything else (e.g. a target segment)
    yields no path, and placement then plans its own way home.
    """
    if not moves or any(move.times is None for move in moves):
        return (), ()
    times, positions = [np.zeros(1)], [moves[-1].positions[-1:]]
    for segment in reversed(moves):
        back = segment.reversed()
        step = back.times[1] - back.times[0]
        times.append(times[-1][-1] + step + back.times[1:] - back.times[1])
        positions.append(back.positions[1:])
    return tuple(np.vstack(positions).ravel()), tuple(np.concatenate(times))


def filter_held_points(local):
    """Palm-frame object cloud -> (kept cloud, attachment indices, radius m, timings s).

    Pure and picklable, so it can run in a worker process while the arm lifts:
    on the 310P (2026-11-08) without_fringe alone took 1.3 s of a 1.9 s handover.
    """
    timings, tick = {}, [time.perf_counter()]

    def lap(name):
        now = time.perf_counter()
        timings[name] = round(now - tick[0], 4)
        tick[0] = now

    local = without_fringe(np.asarray(local, float))
    lap("without_fringe")
    local = without_outliers(local)
    lap("without_outliers")
    # Keep a bounded cloud, including extrema, for collision attachment.
    indices = np.unique(
        np.r_[
            np.linspace(0, len(local) - 1, min(len(local), 512), dtype=int),
            np.argmin(local, axis=0),
            np.argmax(local, axis=0),
        ]
    )
    # Consumers only see the published subset, so its centre is the sphere's.
    radius = float(np.max(np.linalg.norm(local - held_center(local[indices]), axis=1)) + 0.01)
    lap("held_center")
    return local, indices, radius, timings


def _filter_pool():
    """One spawned worker process: the filter must not take the GIL from the arm's command stream."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(filter_held_points, np.zeros((32, 3)) + np.arange(32)[:, None] * 1e-3)  # warm the worker's imports
    return pool


def held_builder(node, executor, effector, *, background=True):
    """``build(query, observation, proposals, plan)`` -> HeldObject; ``build.prepare`` starts it early.

    ``prepare`` (called when the hand has closed) computes the palm-frame cloud
    and hands its filtering to a worker process, so it runs while the arm
    lifts. The palm pose is the planned contact's, which neither the lift nor
    the fingers change, so the prepared cloud is the one build would compute.
    ``build`` uses the prepared result for the same plan, and computes inline
    when there is none or the worker failed.
    """
    models = {}
    pending = {}
    # Started with the node, so its imports are warm before the first lift.
    pool = [_filter_pool() if background else None]

    def palm_cloud(observation, proposals, plan):
        side = plan.report["arm_side"]
        if side not in models:
            models[side] = RobotSphereModel(directory=config_directory(side))
        model = models[side]
        moves = [step.segment for step in plan.steps if isinstance(step, Move)]
        contact = next(segment for segment in moves if segment.name == "grasp")
        sample = executor.io.feedback()
        hand = effector._measured()
        feedback = SimpleNamespace(
            vector=lambda names: np.array([sample.positions[name] for name in names]), hand=lambda _: hand
        )
        model.spheres(feedback, contact.end[None], hand=hand)
        palm = np.eye(4)
        palm[:3] = model.last_link_frames[0, model.link_indices["R_palm"]]
        points = proposals.target_points @ observation.base_from_camera[:3, :3].T + observation.base_from_camera[:3, 3]
        local = (points - palm[:3, 3]) @ palm[:3, :3]
        if not len(local) or not np.isfinite(local).all():
            raise ValueError("held object has no valid shape")
        return side, moves, contact, local

    def prepare(observation, proposals, plan):
        pending.clear()
        if not background:
            return
        try:
            started = time.perf_counter()
            side, moves, contact, local = palm_cloud(observation, proposals, plan)
            future = pool[0].submit(filter_held_points, local)
            pending[id(plan)] = (side, moves, contact, len(local), future, time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - build computes inline instead
            print(json.dumps({"held_object_prepare_error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)

    def build(query, observation, proposals, plan):
        started = time.perf_counter()
        report = {"prepared": False}
        filtered = None
        early = pending.pop(id(plan), None)
        if early is not None:
            side, moves, contact, points_in, future, report["prepare_s"] = early
            try:
                filtered = future.result(timeout=10.0)
                report["prepared"] = True
            except Exception as exc:  # noqa: BLE001 - fall back to the inline filter
                report["prepare_error"] = f"{type(exc).__name__}: {exc}"
            report["wait_s"] = round(time.perf_counter() - started, 4)
        if filtered is None:
            side, moves, contact, local = palm_cloud(observation, proposals, plan)
            points_in = len(local)
            filtered = filter_held_points(local)
        local, indices, radius, timings = filtered
        report.update(points_in=points_in, points_kept=len(local), filter_s=timings)
        report["total_s"] = round(time.perf_counter() - started, 4)
        print(json.dumps({"held_object": report}), file=sys.stderr, flush=True)
        if not np.isfinite(radius):
            raise ValueError("held object has no valid shape")
        return HeldObject(
            uuid.uuid4().hex,
            query,
            side,
            tuple(contact.joint_names),
            tuple(executor.measured(contact.joint_names)),
            tuple(moves[0].start),
            radius,
            tuple(local[indices].ravel()),
            effector.snapshot(),
            float(plan.report.get("carry_lift_m", 0.05)),
            str(getattr(observation.robot_state, "capture_id", observation.stamp_ns)),
            node.get_clock().now().nanoseconds,
            observation.base_frame,
            *return_path(moves),
        )

    build.prepare = prepare
    build.close = lambda: pool[0] is not None and pool[0].shutdown(wait=False, cancel_futures=True)
    return build
