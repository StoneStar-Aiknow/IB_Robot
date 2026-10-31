"""Ordering and filtering of grasp candidates that does not depend on the planner.

Any robot's planner may use these on the grasp model's candidates before its
own search. Grasp poses follow the GraspGen convention: the gripper frame's z
axis is the approach direction. The closing sweep comes from the deployed
gripper description (``sweep_volume``), so it matches the hand the grasp model
was trained for.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Approach preference by target height above its support (p95 of the target
# points). A tall object is wrapped from the side: fingers close around it and
# the arm stays low (top-down grasps on the 25 cm bottle only reached the cap
# and needed 0.18-0.22 rad of gravity compensation). A low object leaves no
# room for side fingers above the table, so it is taken from above.
SIDE_FIRST_MIN_HEIGHT_M = 0.10
TOP_FIRST_MAX_HEIGHT_M = 0.06
# cos(approach, up): top-down within ~45 deg of straight down; side within
# ~24 deg of horizontal.
TOP_DOWN_MAX_COS = -0.7
SIDE_MAX_ABS_COS = 0.4
# A side_first request never plans a candidate that comes down onto the object
# more steeply than the side band: when few side candidates pass the screen,
# the ordering alone let one through (live 2026-10-13: cos -0.54 on a 19 cm
# bottle, 2 screened candidates; the hand pushed the cap). A tall object with
# no usable side candidate is reported as not graspable instead.
SIDE_FIRST_EXCLUDES_FROM_ABOVE = True


def approach_priority(camera_from_grasps, base_from_camera, up, height_m):
    """Per-candidate priority (0 first, 1 later) and a report of why.

    Unknown height or a height between the two limits gives every candidate
    priority 0, which leaves pure confidence order. Preference orders
    candidates; the only ones to remove are listed in the report's "excluded"
    (side_first: approach below the side band, see
    SIDE_FIRST_EXCLUDES_FROM_ABOVE). The caller removes them.
    """
    grasps = np.asarray(camera_from_grasps, dtype=float)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    approach = np.einsum("ij,nj->ni", np.asarray(base_from_camera, dtype=float)[:3, :3], grasps[:, :3, 2])
    cosine = approach @ up
    top_down = cosine <= TOP_DOWN_MAX_COS
    side = np.abs(cosine) <= SIDE_MAX_ABS_COS
    if height_m is None or not np.isfinite(height_m):
        mode, preferred = "none", np.ones(len(grasps), dtype=bool)
    elif height_m >= SIDE_FIRST_MIN_HEIGHT_M:
        mode, preferred = "side_first", side
    elif height_m < TOP_FIRST_MAX_HEIGHT_M:
        mode, preferred = "top_first", top_down
    else:
        mode, preferred = "none", np.ones(len(grasps), dtype=bool)
    excluded = (
        cosine < -SIDE_MAX_ABS_COS
        if mode == "side_first" and SIDE_FIRST_EXCLUDES_FROM_ABOVE
        else np.zeros(len(grasps), dtype=bool)
    )
    report = {
        "target_height_m": None if height_m is None else float(height_m),
        "mode": mode,
        "side": np.flatnonzero(side).tolist(),
        "top_down": np.flatnonzero(top_down).tolist(),
        "excluded": np.flatnonzero(excluded).tolist(),
    }
    return np.where(preferred, 0, 1), report


# A candidate must hold part of the target inside the gripper's closing sweep
# (the two boxes of the gripper config, in the grasp frame). GraspGen-X also
# returns candidates whose fingers face away from the target (plan3/plan4:
# 17-37 of 100, fingertip 5-13 cm from any target point); they pass IK and
# collision screening easily because they are nowhere near the object, and a
# side-first order put two of them first. Kept candidates held 1700-7000
# target points; the rejected ones held none.
MIN_ENCLOSED_POINTS = 50


def sweep_boxes(gripper: Mapping) -> tuple:
    """(offset, half extents) of the gripper description's two closing-sweep boxes."""
    sweep = gripper["sweep_volume"]
    return tuple(
        (np.asarray(sweep[offset], dtype=float), np.asarray(sweep[extents], dtype=float) / 2)
        for offset, extents in (("offset", "extents"), ("offset2", "extents2"))
    )


def enclosed_points(camera_from_grasps, points_camera, boxes, candidates=None):
    """Target points inside each candidate's closing sweep (``boxes``, union).

    ``boxes`` is ``sweep_boxes(gripper)``. With ``candidates`` (indices), only
    those are counted; the others are -1.
    """
    grasps = np.asarray(camera_from_grasps, dtype=float)
    points = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    counts = np.full(len(grasps), -1, dtype=int)
    indices = range(len(grasps)) if candidates is None else candidates
    for index in indices:
        grasp = grasps[index]
        local = (points - grasp[:3, 3]) @ grasp[:3, :3]
        inside = np.zeros(len(points), dtype=bool)
        for offset, extents in boxes:
            inside |= np.all(np.abs(local - offset) <= extents, axis=1)
        counts[index] = int(inside.sum())
    return counts
