"""Palm goals whose carried object clears every edge of the measured opening."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from manipulation_execution.grasp_pipeline.contracts import GraspFailure

from .grasp_request import palm_reach
from .held_object import held_center
from .place_config import PLACE_EXTRA_POSITIONS, PlaceTargetConfig
from .scene_estimation import plane_basis

#: Palm orientations a placement may use, as rotations of the carrying palm: a
#: yaw about the opening's normal (full circle) after a tilt of the normal
#: itself. Live 2026-11-06 the grasp palm with five yaws (+-60 deg) reached no
#: opening pose for a bowl 20 cm toward the midline (r_place_14/17). Offline
#: IK on the recorded held objects: grasp palm +-60 deg reached 3-4 of 9 cells
#: and never the midline band; a 40 deg tilt and yaws every 60 deg reached 7.
PLACE_YAWS_DEG = tuple(range(0, 360, 30))
PLACE_TILTS_DEG = (0.0, 20.0, 40.0)
PLACE_TILT_DIRECTIONS = 4
__all__ = ["PLACE_EXTRA_POSITIONS", "PlaceTargetConfig", "place_orientations", "place_targets"]


def place_orientations(normal):
    """Rotations (applied before the carrying palm) ordered by how far they turn it.

    The least change first: the grasp palm itself, then small yaws and tilts.
    A tilt turns about an axis in the opening's plane; the yaw then turns
    about the (untilted) normal.
    """
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    basis = plane_basis(normal)
    rotations = []
    for tilt in PLACE_TILTS_DEG:
        directions = 1 if tilt == 0.0 else PLACE_TILT_DIRECTIONS
        for k in range(directions):
            angle = 2.0 * np.pi * k / directions
            axis = basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)
            tilted = Rotation.from_rotvec(axis * np.radians(tilt))
            for yaw in PLACE_YAWS_DEG:
                rotations.append((Rotation.from_rotvec(normal * np.radians(yaw)) * tilted).as_matrix())
    turned = [Rotation.from_matrix(rotation).magnitude() for rotation in rotations]
    return [rotations[i] for i in np.argsort(turned, kind="stable")]


def place_targets(container, held, current_palm, config: PlaceTargetConfig = PlaceTargetConfig()):
    basis = plane_basis(container.normal)
    local = container.opening @ basis
    hull = ConvexHull(local[:, :2])
    polygon = local[hull.vertices, :2]
    center = polygon.mean(axis=0)
    fraction = config.edge_fraction
    samples = np.vstack([center, (1.0 - fraction) * center + fraction * polygon])
    clearance = held.radius_m + float(container.report.get("wall_margin_m", 0.015))
    fits = (samples @ hull.equations[:, :2].T + hull.equations[:, 2] <= -clearance).all(axis=1)
    if config.extra_positions == 0:
        samples = samples[:1] if fits[0] else np.empty((0, 2))  # the centre only, and only if it fits
    else:
        samples = samples[fits]
        if len(samples) > 1 + config.extra_positions:
            # The centre (if it fits) and positions spread around the opening.
            keep = np.unique(np.round(np.linspace(1, len(samples) - 1, config.extra_positions)).astype(int))
            samples = samples[np.concatenate([[0], keep])]
    if not len(samples):
        raise GraspFailure(
            "CONTAINER_TOO_SMALL",
            f"no release position in the opening keeps the held object {clearance:.3f} m from the rim "
            f"(extra_positions {config.extra_positions}, edge_fraction {fraction:g})",
        )
    points = np.asarray(held.points_in_palm).reshape(-1, 3)
    if not len(points):
        raise GraspFailure("HELD_OBJECT_MISMATCH", "held object has no palm-frame geometry")
    middle = held_center(points)
    shoulder, reach = palm_reach(held.arm_side)
    targets = []
    for turn in place_orientations(container.normal):
        rotation = turn @ current_palm[:3, :3]
        transformed = points @ rotation.T
        for xy in samples:
            target = np.eye(4)
            target[:3, :3] = rotation
            # Centre the held object's enclosing sphere (not the one-sided cloud's
            # mean), keeping its lowest observed point bottom_clearance_m above the
            # rim. The extra centimetre covers cloud margins.
            target[:3, 3] = basis[:, :2] @ xy - rotation @ middle
            target[:3, 3] += container.normal * (
                container.rim_height_m
                + config.bottom_clearance_m
                + 0.01
                - np.min((transformed + target[:3, 3]) @ container.normal)
            )
            above = target.copy()
            above[:3, 3] += container.normal * config.transfer_clearance_m
            if max(np.linalg.norm(pose[:3, 3] - shoulder) for pose in (above, target)) <= reach:
                targets.append((above, target))
    if not targets:
        raise GraspFailure("CONTAINER_OUT_OF_REACH", "no opening pose within the holding arm's reach")
    return targets
