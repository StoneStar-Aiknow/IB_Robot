"""Optional displacement limits for commissioning from the measured arm pose."""

from __future__ import annotations

import math

import numpy as np

from aimdk_robot.pose_backend import kin
from aimdk_robot.pose_backend.model import ARM_JOINTS, SIDES


def check_displacement(plan, start_q, max_joint_delta_rad=0.0, max_hand_displacement_m=0.0):
    """Return a refusal reason, or an empty string. Zero disables a limit.

    Check every trajectory point, including planner detours, relative to the
    actual measured start. This is an additional commissioning restriction;
    the existing collision, torque and runtime checks remain authoritative.
    """
    for value in (max_joint_delta_rad, max_hand_displacement_m):
        if not math.isfinite(value) or value < 0:
            raise ValueError("motion displacement limits must be finite and non-negative")
    if not (max_joint_delta_rad or max_hand_displacement_m):
        return ""
    positions = np.asarray(plan.traj.positions)
    if positions.ndim != 2 or positions.shape[1] != len(ARM_JOINTS) or not np.all(np.isfinite(positions)):
        return "invalid trajectory positions"
    start = np.asarray([start_q[n] for n in ARM_JOINTS])
    if not np.all(np.isfinite(start)):
        return "invalid measured start"
    delta = float(np.max(np.abs(positions - start)))
    if max_joint_delta_rad and delta > max_joint_delta_rad:
        row, column = np.unravel_index(np.argmax(np.abs(positions - start)), positions.shape)
        return (
            f"joint displacement {delta:.4f} rad exceeds commissioning limit {max_joint_delta_rad:.4f} rad: "
            f"{ARM_JOINTS[column]} from {start[column]:.4f} to {positions[row, column]:.4f} rad"
        )
    if max_hand_displacement_m:
        origin = kin.fk(start_q).points
        for row in positions:
            points = kin.fk(dict(zip(ARM_JOINTS, row, strict=True))).points
            for side in SIDES:
                distance = float(np.linalg.norm(points[f"{side}_hand"] - origin[f"{side}_hand"]))
                if distance > max_hand_displacement_m:
                    return f"{side} hand displacement {distance:.4f} m exceeds commissioning limit {max_hand_displacement_m:.4f} m"
    return ""
