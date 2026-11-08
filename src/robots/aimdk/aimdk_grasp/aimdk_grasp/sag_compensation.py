"""Raise a task-pose trajectory by the arms' predicted gravity sag.

At MC's kp 40 the X2 arm settles below its command by gravity torque / kp: a
task pose commanded as configured lands about 0.11 rad short in shoulder roll
(right arm at grasp_ready, measured 2026-11-07) and fails its arrival check.
Sending ``q + sag(q)`` for every waypoint lands the measured arm on the planned
path instead. The planned path, the arrival check and every path guard keep
using the uncompensated positions; only the commands sent to the runtime move.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .arm_gravity import LIMIT_MARGIN_RAD, ArmGravity
from .joints import arm_joints


def deployed_hands(grasp_execution) -> tuple[str, ...] | None:
    """The hands a deployment declares mounted (``backend_options.installed_hands``), read on every start.

    The same reading as the grasp backend and the hand restorer (both hands when
    the list is omitted); None (both arms with hands) without a grasp execution.
    """
    if not isinstance(grasp_execution, Mapping):
        return None
    from .backend import grasp_installed_hands

    return grasp_installed_hands(grasp_execution)


class SagCompensation:
    """Command shaper for a both-arm trajectory in ``joint_names`` order."""

    def __init__(
        self,
        joint_names: Sequence[str],
        limits: Mapping | None = None,
        gravity: Mapping | None = None,
        *,
        installed_hands: Sequence[str] | None = None,
    ):
        """``installed_hands``: the arms that carry a hand (both when None); the others use a bare-wrist model."""
        names = tuple(joint_names)
        self.joint_names = names
        hands = ("left", "right") if installed_hands is None else tuple(installed_hands)
        if not set(hands) <= {"left", "right"}:
            raise ValueError("installed_hands must name left and/or right")
        self.arms = []
        for side in ("left", "right"):
            joints = arm_joints(side)
            if not set(joints) <= set(names):
                raise ValueError(f"sag compensation needs all seven {side} arm joints")
            model = (gravity or {}).get(side) or ArmGravity.load(side=side)
            if tuple(model.joints) != tuple(joints):
                raise ValueError(f"the {side} gravity model does not describe the {side} arm")
            if side not in hands:
                model = model.without_hand()
            self.arms.append(([names.index(joint) for joint in joints], model))
        self.installed_hands = hands
        limits = limits or {}
        self.lower = np.array([float(limits[n][0]) + LIMIT_MARGIN_RAD if n in limits else -np.inf for n in names])
        self.upper = np.array([float(limits[n][1]) - LIMIT_MARGIN_RAD if n in limits else np.inf for n in names])
        if np.any(self.lower >= self.upper):
            raise ValueError("arm limits leave no room inside the command margin")

    def describe(self) -> str:
        """Which model each arm uses, for the startup log."""
        return ", ".join(
            f"{side} {'with hand' if side in self.installed_hands else 'bare wrist'}" for side in ("left", "right")
        )

    def offsets(self, positions) -> np.ndarray:
        """Predicted sag per waypoint and joint (N x len(joint_names)); zero for joints of neither arm."""
        q = np.atleast_2d(np.asarray(positions, dtype=float))
        if q.shape[1] != len(self.joint_names):
            raise ValueError("trajectory width does not match the compensation joints")
        sag = np.zeros_like(q)
        for index, model in self.arms:
            sag[:, index] = model.sags(q[:, index])
        return sag

    def __call__(self, positions) -> np.ndarray:
        """Commands for the planned ``positions``: raised by the sag, kept inside the arm limits."""
        q = np.atleast_2d(np.asarray(positions, dtype=float))
        return np.clip(q + self.offsets(q), self.lower, self.upper)
