"""Verify observed target depth against the opening used by the place plan."""

import time

import numpy as np
from scipy.spatial import ConvexHull

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, PlaceVerdict

from .container_proposer import detect_masks
from .place_targets import opening_centre
from .proposer import PerceptionSettings
from .scene_estimation import plane_basis

INSIDE_FRACTION = 0.9
MIN_POINTS = 10


class PlaceVerifier:
    """Judge the placed object by the detected candidate nearest the opening's centre.

    The detector can rank another object outside the container first (an orange
    read as "an apple", live 2026-11-07), so every candidate is measured and the
    one the place aimed at -- nearest the opening it released into -- decides.
    ``settings`` are the pick's own perception settings, so the object is found
    with the thresholds that found it for the pick.
    """

    def __init__(self, services, *, settings=PerceptionSettings()):
        self.services, self.settings = services, settings

    def verify(self, observation, plan, held, *, deadline, cancelled):
        container = plan.container
        basis = plane_basis(container.normal)
        polygon = container.opening @ basis
        hull = ConvexHull(polygon[:, :2])
        centre = opening_centre(hull.equations)[0]
        intrinsics = observation.intrinsics
        candidates = []
        masks = detect_masks(self.services, observation, held.target_query, cancelled=cancelled, settings=self.settings)
        for index, mask in enumerate(masks):
            if cancelled() or time.monotonic() >= deadline:
                raise GraspFailure("CANCELED" if cancelled() else "SKILL_TIMEOUT", "place verification interrupted")
            y, x = np.nonzero(mask & np.isfinite(observation.depth_m) & (observation.depth_m > 0))
            if len(y) < MIN_POINTS:
                continue
            z = observation.depth_m[y, x]
            points = np.column_stack(
                [(x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z]
            )
            points = points @ observation.base_from_camera[:3, :3].T + observation.base_from_camera[:3, 3]
            local = points @ basis
            inside = (local[:, :2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 0).all(axis=1)
            below = local[:, 2] <= container.rim_height_m + 2 * held.radius_m
            candidates.append(
                {
                    "index": index,
                    "points": int(len(y)),
                    "centre_distance_m": round(float(np.linalg.norm(local[:, :2].mean(axis=0) - centre)), 4),
                    "inside_fraction": round(float(np.mean(inside & below)), 4),
                }
            )
        if not candidates:
            return PlaceVerdict(False, "PLACE_UNSEEN", {"candidates": []})
        chosen = min(candidates, key=lambda candidate: candidate["centre_distance_m"])
        report = {"inside_fraction": chosen["inside_fraction"], "chosen": chosen["index"], "candidates": candidates}
        if chosen["inside_fraction"] >= INSIDE_FRACTION:
            return PlaceVerdict(True, report=report)
        return PlaceVerdict(False, "PLACED_OUTSIDE", report)
