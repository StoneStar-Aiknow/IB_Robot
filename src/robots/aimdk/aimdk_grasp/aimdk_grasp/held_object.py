"""Describe the object in the palm frame at contact, using CPU forward kinematics."""

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


def held_builder(node, executor, effector):
    models = {}

    def build(query, observation, proposals, plan):
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
        local = without_outliers(without_fringe(local))
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

    return build
