"""Gravity compensation of a planned grasp for MC's kp=40 arm gain.

At kp 40 the arm settles below its command by its gravity torque / kp (r25
static-equilibrium fit; r48 reach1 faulted on the unmodelled sag). So that the
MEASURED arm follows the planned path, each waypoint is commanded above it by
the change in predicted sag relative to the segment start:

    commanded(q) = q + (sag(q) - sag(q_start))

The static sag at the start is already in MC's command board, which the
streaming executor carries as its command offset; GravityShaper gives it the
predicted sag per waypoint, and the executor tracks and judges arrival against
the raw waypoints. Live 2026-09-30 g1-g9 ran with this model (prepare_plan.py).

The chain's joint origins, axes, masses and centres are the official X2 Ultra
description's (aimdk_description, x2_ultra.urdf v1.3.0); only kp, the torso
tilt and the wrist payload are fitted.
"""

import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .joints import LEFT_ARM

GRAVITY = np.array([0.0, 0.0, -9.81])
# Arrival is judged against the raw plan, but the executor's tracking check
# compares the measured arm with the sent command, and at kp40 that command
# must lead the arm by the whole sag. The live runs measured sag plus lag up
# to 0.14 rad where the model predicted 0.117 (lag ~0.03); refuse plans whose
# predicted sag leaves less than 0.06 of the 0.3 rad tracking limit.
MAX_SAG_RAD = 0.24
# MC clamps commands to its joint limits; a clamped command differs from ours
# and the executor stops (g4, wrist roll). Commands stay this far inside (the
# planner runs wrist joints to within 0.001-0.002 rad of a limit, where the
# raise could cross it).
LIMIT_MARGIN_RAD = 0.005


def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


class ArmGravity:
    """Left-arm gravity torques (torso tilt and wrist payload from the r25 fit)."""

    def __init__(self, model):
        chain = model["chain"]
        if tuple(link["joint"] for link in chain) != LEFT_ARM:
            raise ValueError("Gravity model chain must be the seven left arm joints in order")
        self.kp = float(model["kp"])
        self.base = _rpy(model["torso_tilt_roll_rad"], model["torso_tilt_pitch_rad"], 0.0)
        self.payload = float(model["payload_kg"])
        self.payload_offset = np.array(model["payload_offset_m"], dtype=float)
        self.chain = [
            dict(
                xyz=np.array(link["xyz"], float),
                rpy=_rpy(*link["rpy"]),
                axis=np.array(link["axis"], float),
                mass=float(link["mass"]),
                centre=np.array(link["centre"], float),
            )
            for link in chain
        ]
        self.source = model.get("source", {})

    @classmethod
    def load(cls, path=None):
        if path is None:
            from .robot_model import config_directory

            path = config_directory() / "arm_gravity.json"
        with open(path) as stream:
            return cls(json.load(stream))

    def torques(self, q):
        """Motor torque holding the pose against gravity, per LEFT_ARM joint."""
        position, rotation = np.zeros(3), self.base.copy()
        origins, axes, masses, centres = [], [], [], []
        for link, angle in zip(self.chain, np.asarray(q, dtype=float), strict=False):
            position = position + rotation @ link["xyz"]
            rotation = rotation @ link["rpy"]
            origins.append(position.copy())
            axes.append(rotation @ link["axis"])
            rotation = rotation @ _axis_angle(link["axis"], angle)
            masses.append(link["mass"])
            centres.append(position + rotation @ link["centre"])
        if self.payload > 0.0:
            masses.append(self.payload)
            centres.append(position + rotation @ self.payload_offset)
        return np.array(
            [
                -sum(axes[i] @ np.cross(centres[k] - origins[i], masses[k] * GRAVITY) for k in range(i, len(masses)))
                for i in range(len(self.chain))
            ]
        )

    def sag(self, q):
        return self.torques(q) / self.kp

    def sags(self, positions):
        """sag() of every waypoint at once (N x 7); one waypoint at a time cost
        ~0.6 ms each on the workstation CPU, between plan ready and sending it."""
        q = np.asarray(positions, dtype=float)
        count = len(q)
        position = np.zeros((count, 3))
        rotation = np.broadcast_to(self.base, (count, 3, 3)).copy()
        origins, axes, masses, centres = [], [], [], []
        for index, link in enumerate(self.chain):
            position = position + rotation @ link["xyz"]
            rotation = rotation @ link["rpy"]
            origins.append(position)
            axes.append(rotation @ link["axis"])
            axis = link["axis"] / np.linalg.norm(link["axis"])
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            angle = q[:, index, None, None]
            rotation = rotation @ (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew))
            masses.append(link["mass"])
            centres.append(position + rotation @ link["centre"])
        if self.payload > 0.0:
            masses.append(self.payload)
            centres.append(position + rotation @ self.payload_offset)
        torques = np.zeros((count, len(self.chain)))
        for i in range(len(self.chain)):
            for k in range(i, len(masses)):
                torques[:, i] -= np.einsum("nj,nj->n", axes[i], np.cross(centres[k] - origins[i], masses[k] * GRAVITY))
        return torques / self.kp


def require_executable(
    segments: Mapping[str, np.ndarray],
    limits: Mapping[str, tuple[float, float]],
    gravity=None,
    *,
    max_sag_rad=MAX_SAG_RAD,
):
    """The largest predicted sag of planned left-arm segments ({name: N x 7}).

    Raises ValueError when a planned waypoint leaves MC's joint limits or the
    predicted sag leaves too little of the tracking limit: replan.
    """
    gravity = ArmGravity.load() if gravity is None else gravity
    lower, upper = (np.array(v, dtype=float) for v in zip(*(limits[n] for n in LEFT_ARM), strict=True))
    worst = 0.0
    for name, positions in segments.items():
        positions = np.asarray(positions, dtype=float)
        outside = (positions < lower) | (positions > upper)
        if outside.any():
            joints = sorted({LEFT_ARM[j] for j in np.flatnonzero(outside.any(0))})
            raise ValueError(f"Planned {name} leaves the MC joint limits at {joints}; replan")
        worst = max(worst, float(np.abs(gravity.sags(positions)).max()))
    if worst > max_sag_rad:
        raise ValueError(
            f"Predicted sag {worst:.3f} rad > {max_sag_rad} leaves too little of the tracking limit; replan"
        )
    return worst


class GravityShaper:
    """The streaming executor's command shaper: the predicted sag of each left-arm waypoint."""

    def __init__(self, gravity=None):
        self.gravity = ArmGravity.load() if gravity is None else gravity

    def raise_for(self, joint_names: Sequence[str], positions):
        if tuple(joint_names) != LEFT_ARM:
            raise ValueError("The gravity model shapes the seven left arm joints in order")
        return self.gravity.sags(positions)
