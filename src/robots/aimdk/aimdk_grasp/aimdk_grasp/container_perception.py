"""Estimate a visible container opening, rejecting a lid or an unseen cavity."""

import json
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from manipulation_execution.grasp_pipeline.contracts import ContainerEstimate, GraspFailure

from .place_request import placement_cover
from .place_targets import opening_centre
from .proposer import prepare_capture_scene
from .scene_estimation import NoSupportError, complete_scene, plane_basis


def opening_from_points(points, normal, *, radius_m, wall_margin_m=0.015):
    """A convex rim with visible depth below it and room for the held sphere.

    Heights are along the fitted support normal. Require cavity evidence in
    the centre of the rim, rather than treating a detector's box as an opening.
    """
    cloud, normal = np.asarray(points, float), np.asarray(normal, float)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or len(cloud) < 30 or not np.isfinite(cloud).all():
        raise GraspFailure("CONTAINER_NOT_OPEN", "too little valid container depth", retryable=True)
    basis = plane_basis(normal)
    local = cloud @ basis
    rim = float(np.quantile(local[:, 2], 0.95))
    upper = local[local[:, 2] >= rim - 0.015]
    try:
        hull = ConvexHull(upper[:, :2])
    except QhullError as exc:
        raise GraspFailure("CONTAINER_NOT_OPEN", "no measurable rim", retryable=True) from exc
    polygon = upper[hull.vertices, :2]
    planes = hull.equations
    # The largest inscribed circle, not the vertex mean: see opening_centre.
    _, inscribed = opening_centre(planes)
    if inscribed <= radius_m + wall_margin_m:
        raise GraspFailure(
            "CONTAINER_TOO_SMALL",
            f"opening has no clearance for the held object: inscribed radius {inscribed:.3f} m, "
            f"held radius {radius_m:.3f} m plus wall margin {wall_margin_m:.3f} m",
        )
    # Require an observed cavity well inside the rim. A solid lid's centre is
    # at rim height; missing depth is not evidence of an open container.
    inside = (local[:, :2] @ planes[:, :2].T + planes[:, 2] <= -wall_margin_m).all(axis=1)
    inner = local[inside]
    cavity = inner[inner[:, 2] < rim - 0.03]
    if len(cavity) < 10 or len(cavity) < 0.15 * len(inner):
        raise GraspFailure("CONTAINER_NOT_OPEN", "a visible cavity below the rim is required", retryable=True)
    opening = np.column_stack([polygon, np.full(len(polygon), rim)]) @ basis.T
    return ContainerEstimate(
        opening,
        normal,
        rim,
        {
            "cavity_points": len(cavity),
            "rim_points": len(upper),
            "clearance_m": inscribed,
            "wall_margin_m": wall_margin_m,
        },
    )


def cover_reach(points_camera, base_from_camera, centers, radii, margin):
    """Distance (m) from each camera-frame point to the nearest cover sphere's surface.

    Computed in the base frame (``centers`` are base-frame) exactly as a loop
    over all points would, but only for points inside the spheres' bounding
    box grown by the largest radius plus ``margin``: a point outside it is
    farther than ``margin`` from every sphere, so it gets ``inf`` and passes
    any ``reach > m`` filter with ``m <= margin``, as it did before. On the X2
    (2026-11-08) the full loop over 567k points and 13 spheres took 0.63 s.
    """
    points_camera = np.asarray(points_camera, dtype=float)
    centers, radii = np.asarray(centers, dtype=float).reshape(-1, 3), np.asarray(radii, dtype=float).reshape(-1)
    reach = np.full(len(points_camera), np.inf)
    if not len(centers) or not len(points_camera):
        return reach
    rotation, translation = base_from_camera[:3, :3], base_from_camera[:3, 3]
    local = (centers - translation) @ rotation  # sphere centres in the camera frame (rigid: same distances)
    pad = float(np.max(radii)) + float(margin) + 1e-6
    low, high = local.min(axis=0) - pad, local.max(axis=0) + pad
    near = np.flatnonzero(((points_camera >= low) & (points_camera <= high)).all(axis=1))
    base = points_camera[near] @ rotation.T + translation
    nearest = np.full(len(near), np.inf)
    for center, radius in zip(centers, radii, strict=True):
        nearest = np.minimum(nearest, np.linalg.norm(base - center, axis=1) - radius)
    reach[near] = nearest
    return reach


def estimate_container(observation, mask, held, model, settings, *, prepared=None, support_hint=None):
    """Use the image-time robot self filter for both geometry and collision scene.

    Prints one stderr JSON line ``{"estimate_container": ...}`` with its stage
    durations (s), also when it fails: on the X2 (2026-11-08) this took
    1.1-1.2 s of the ~4 s place planning. ``support_hint`` (the pick's torso-frame
    table plane) skips the support search when this frame confirms it.
    """
    started = time.perf_counter()
    timings: dict = {}
    tick = [started]
    scene = cover = None

    def lap(name):
        now = time.perf_counter()
        timings[name] = round(now - tick[0], 4)
        tick[0] = now

    try:
        if prepared is None:
            prepared = prepare_capture_scene(observation, model, settings.scene)
            lap("prepare")
        feedback = observation.robot_state.live.observation.image_feedback
        model.spheres(feedback, feedback.vector(model.arm)[None])
        palm = model.last_link_frames[0, model.link_indices["R_palm"]]
        # The planner starts with exactly these palm-frame spheres attached: the
        # closed fingers beyond the open-hand model, then the object. Whatever the
        # camera sees inside them is the hand and the object themselves (the real
        # fingers sit centimetres off the model), never an obstacle the start state
        # could avoid. Live 2026-11-06 such finger points collided with the finger
        # cover by 1.6 cm (world_links=['attached_object']) and no place could plan.
        cover = placement_cover(model, feedback, held)
        lap("cover")
        centers = cover[:, :3] @ palm[:3, :3].T + palm[:3, 3]
        rotation, translation = observation.base_from_camera[:3, :3], observation.base_from_camera[:3, 3]
        validation_margin = settings.scene.validation_self_filter_margin_m
        planning_margin = settings.scene.self_filter_margin_m
        reach = cover_reach(
            prepared.points,
            observation.base_from_camera,
            centers,
            cover[:, 3],
            max(validation_margin, planning_margin),
        )
        # Filtered like the robot's own spheres (_self_filter_masks): the fine
        # validation set with the validation margin, the coarse planning set with
        # the coarse margin its 3 cm voxels need.
        prepared = replace(
            prepared,
            keep=prepared.keep & (reach > validation_margin),
            planning_all=prepared.planning_all & (reach > planning_margin),
            spheres=np.vstack([prepared.spheres, np.column_stack([centers, cover[:, 3]])]),
        )
        selected = np.asarray(mask, bool).reshape(-1)[prepared.pixel] & prepared.keep
        points = prepared.points[selected]
        points = points @ rotation.T + translation
        lap("cover_filter")
        try:
            scene = complete_scene(prepared, mask, lazy_validation=False, support_hint=support_hint)
        except (NoSupportError, RuntimeError) as exc:
            raise GraspFailure("CONTAINER_NOT_FOUND", f"container scene unavailable: {exc}", retryable=True) from exc
        lap("complete_scene")
        container = opening_from_points(points, scene.support.normal, radius_m=held.radius_m)
        lap("opening")
        return replace(container, scene=scene)
    finally:
        timings["total"] = round(time.perf_counter() - started, 4)
        line = {
            "points": int(len(prepared.points)) if prepared is not None else None,
            "cover_spheres": int(len(cover)) if cover is not None else None,
            "timings_s": timings,
            "scene_timings_s": dict((scene.diagnostics or {}).get("timings_s", {})) if scene is not None else None,
            "support_reuse": (scene.diagnostics or {}).get("support_reuse") if scene is not None else None,
        }
        print(json.dumps({"estimate_container": line}, default=float), file=sys.stderr, flush=True)
