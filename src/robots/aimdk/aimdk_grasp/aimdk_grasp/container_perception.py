"""Estimate a visible container opening, rejecting a lid or an unseen cavity."""

from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from manipulation_execution.grasp_pipeline.contracts import ContainerEstimate, GraspFailure

from .place_request import placement_cover
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
    center = polygon.mean(axis=0)
    planes = hull.equations
    distances = -(center @ planes[:, :2].T + planes[:, 2])
    if distances.min() <= radius_m + wall_margin_m:
        raise GraspFailure(
            "CONTAINER_TOO_SMALL",
            f"opening has no clearance for the held object: inscribed radius {distances.min():.3f} m, "
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
            "clearance_m": float(distances.min()),
            "wall_margin_m": wall_margin_m,
        },
    )


def estimate_container(observation, mask, held, model, settings, *, prepared=None):
    """Use the image-time robot self filter for both geometry and collision scene."""
    if prepared is None:
        prepared = prepare_capture_scene(observation, model, settings.scene)
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
    centers = cover[:, :3] @ palm[:3, :3].T + palm[:3, 3]
    rotation, translation = observation.base_from_camera[:3, :3], observation.base_from_camera[:3, 3]
    base = prepared.points @ rotation.T + translation
    reach = np.full(len(base), np.inf)
    for center, radius in zip(centers, cover[:, 3], strict=True):
        reach = np.minimum(reach, np.linalg.norm(base - center, axis=1) - radius)
    # Filtered like the robot's own spheres (_self_filter_masks): the fine
    # validation set with the validation margin, the coarse planning set with
    # the coarse margin its 3 cm voxels need.
    prepared = replace(
        prepared,
        keep=prepared.keep & (reach > settings.scene.validation_self_filter_margin_m),
        planning_all=prepared.planning_all & (reach > settings.scene.self_filter_margin_m),
        spheres=np.vstack([prepared.spheres, np.column_stack([centers, cover[:, 3]])]),
    )
    selected = np.asarray(mask, bool).reshape(-1)[prepared.pixel] & prepared.keep
    points = prepared.points[selected]
    points = points @ rotation.T + translation
    try:
        scene = complete_scene(prepared, mask, lazy_validation=False)
    except (NoSupportError, RuntimeError) as exc:
        raise GraspFailure("CONTAINER_NOT_FOUND", f"container scene unavailable: {exc}", retryable=True) from exc
    container = opening_from_points(points, scene.support.normal, radius_m=held.radius_m)
    return replace(container, scene=scene)
