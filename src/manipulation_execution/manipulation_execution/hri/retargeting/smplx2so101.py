"""SMPL-X arm -> SO-101 arm joints ``"1".."5"``.

This is the only module that knows about the SO-101. Each frame is reduced to
five scalar readouts of one human arm, measured in a torso frame so that the
person turning around changes nothing, and each readout is bound to one robot
joint by a fixed affine band:

    q1 = C - pan               C = -90 (left arm) / +90 (right arm)
    q2 = elev                  working range -90 .. 0
    q3 = bend - 90             working range -90 .. +45
    q4 = wflex * 70 / 45       working range -70 .. +70
    q5 = -90 - (wroll + 90)/2  working range -135 .. -45, neutral -90

Values are returned unclamped; clamping to :attr:`Smplx2So101.working_range`
happens in the shared post-processing, where it is also counted.
"""

from __future__ import annotations

import math

import numpy as np

from .base import RetargetOutput, RetargetRequest

# Neutral SMPL-X skeleton: joint -> (parent, offset from the parent in metres),
# parents before children. Only the chain feeding the arm readouts is kept:
# spine to neck, both collars, both arms and four knuckles per hand.
_ROOT_POSITION = np.array((0.003123258056, -0.3514074395, 0.01203655087))
_CHAIN = {
    3: (0, (-0.002762692654, 0.109890584406, -0.027617633683)),
    6: (3, (0.009447697611, 0.131853244027, -0.005939982894)),
    9: (6, (-0.011330417621, 0.052235158709, 0.028446897711)),
    12: (9, (-0.012164458549, 0.165167067147, -0.031615341486)),
    13: (9, (0.046364158346, 0.084943726129, -0.007220482273)),
    14: (9, (-0.047694927949, 0.084338677081, -0.013399899226)),
    16: (13, (0.119239019379, 0.057728025451, -0.015460939348)),
    17: (14, (-0.102577731232, 0.053524439837, -0.012668528927)),
    18: (16, (0.254122860953, -0.072150517700, -0.042458851537)),
    19: (17, (-0.271149566376, -0.036492473179, -0.026467089686)),
    20: (18, (0.251986753784, 0.023221228165, -0.002472082928)),
    21: (19, (-0.249267447993, -0.004532550574, -0.015325182590)),
    25: (20, (0.101901881265, -0.008688142524, 0.019351139777)),
    28: (20, (0.109397525054, -0.006327671728, -0.003980820271)),
    31: (20, (0.084046906409, -0.014539334749, -0.043749611760)),
    34: (20, (0.097443982562, -0.009267637866, -0.027344661017)),
    40: (21, (-0.099880544280, -0.011782859804, 0.019599944550)),
    43: (21, (-0.107376261743, -0.009421943601, -0.003733819402)),
    46: (21, (-0.082026569238, -0.017634432833, -0.043502913056)),
    49: (21, (-0.095423248880, -0.012361931171, -0.027098749208)),
}
# Rotation index == SMPL-X body joint index for joints 0..21. Knuckle positions
# only need the wrist's global rotation, so no hand rotation is ever read.
_BODY_JOINT_COUNT = 22

_PELVIS, _NECK, _LEFT_COLLAR, _RIGHT_COLLAR = 0, 12, 13, 14
_ARM = {"left": (16, 18, 20), "right": (17, 19, 21)}  # shoulder, elbow, wrist
_KNUCKLES = {"left": (25, 28, 31, 34), "right": (40, 43, 46, 49)}
# Across-the-palm direction for wrist roll: first knuckle minus the last one of
# the four above. Any two knuckles far apart work; this pair is the one the
# band table below was measured with, so it must not change on its own.
_PALM_ACROSS = {"left": (25, 34), "right": (40, 49)}

_PAN_CENTER_DEG = {"left": -90.0, "right": 90.0}
_WROLL_CENTER_DEG = -90.0
# Readouts are held at the previous frame while the arm is too straight for its
# plane (and so pan, wrist flex and wrist roll) to be defined, or while the arm
# plane faces straight up or down, which leaves pan undefined.
_MIN_ARM_BEND_SIN = 0.3
_MIN_PLANE_HORIZONTAL = 0.3

_EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def neutral_fk(rotations: np.ndarray) -> dict[int, np.ndarray]:
    """Joint positions ``{joint: (N, 3)}`` of the neutral skeleton posed by ``rotations``."""
    count = rotations.shape[0]
    positions = {0: np.broadcast_to(_ROOT_POSITION, (count, 3))}
    orientations = {0: rotations[:, 0]}
    for joint, (parent, offset) in _CHAIN.items():
        positions[joint] = positions[parent] + orientations[parent] @ np.asarray(offset)
        if joint < _BODY_JOINT_COUNT:
            orientations[joint] = orientations[parent] @ rotations[:, joint]
    return positions


def _hold(values: np.ndarray, valid: np.ndarray, default: float) -> np.ndarray:
    """Replace invalid samples by the last valid one (the first valid one before it)."""
    valid = valid & np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, default)
    index = np.where(valid, np.arange(values.shape[0]), 0)
    np.maximum.accumulate(index, out=index)
    index[: int(np.argmax(valid))] = int(np.argmax(valid))
    return values[index]


def _continuous_angle(raw: np.ndarray, valid: np.ndarray, center: float) -> np.ndarray:
    """Hold, unwrap, and shift by whole turns so the median sits nearest ``center``."""
    angle = np.unwrap(_hold(raw, valid, center))
    turns = np.round((np.median(angle) - center) / (2.0 * math.pi))
    return angle - turns * 2.0 * math.pi


def arm_readouts(positions: dict[int, np.ndarray], side: str) -> dict[str, np.ndarray]:
    """The five arm readouts, in radians, of the human's ``side`` arm."""
    up = _unit(positions[_NECK] - positions[_PELVIS])
    across = positions[_LEFT_COLLAR] - positions[_RIGHT_COLLAR]
    left = _unit(across - _dot(across, up)[:, None] * up)
    forward = np.cross(left, up)
    if side == "left":
        outward = left
    else:
        # Mirror the lateral *and* the forward axis so that pan is measured the
        # same way on both arms; the per-side constant in q1 then flips sign.
        outward, forward = -left, -forward

    shoulder, elbow, wrist = (positions[j] for j in _ARM[side])
    upper = _unit(elbow - shoulder)
    fore = _unit(wrist - elbow)
    normal_raw = np.cross(upper, fore)
    normal = _unit(normal_raw)
    palm = _unit(np.mean([positions[j] for j in _KNUCKLES[side]], axis=0) - wrist)
    first, last = _PALM_ACROSS[side]
    across_palm = _unit(positions[first] - positions[last])

    plane_ok = np.linalg.norm(normal_raw, axis=-1) >= _MIN_ARM_BEND_SIN
    pan_ok = plane_ok & (np.hypot(_dot(normal, forward), _dot(normal, outward)) >= _MIN_PLANE_HORIZONTAL)
    pan_center = math.radians(_PAN_CENTER_DEG[side])
    wroll_center = math.radians(_WROLL_CENTER_DEG)
    return {
        "elev": np.arcsin(np.clip(_dot(upper, up), -1.0, 1.0)),
        "bend": np.arccos(np.clip(_dot(upper, fore), -1.0, 1.0)),
        "pan": _continuous_angle(np.arctan2(_dot(normal, outward), _dot(normal, forward)), pan_ok, pan_center),
        "wflex": _hold(np.arctan2(_dot(np.cross(fore, palm), normal), _dot(fore, palm)), plane_ok, 0.0),
        "wroll": _continuous_angle(
            np.arctan2(_dot(np.cross(normal, across_palm), palm), _dot(normal, across_palm)),
            plane_ok,
            wroll_center,
        ),
    }


def joint_angles(readouts: dict[str, np.ndarray], side: str) -> np.ndarray:
    """The band table: readouts of one arm -> unclamped ``(N, 5)`` joint angles in radians."""
    pan_center = math.radians(_PAN_CENTER_DEG[side])
    wroll_center = math.radians(_WROLL_CENTER_DEG)
    return np.stack(
        (
            pan_center - readouts["pan"],
            readouts["elev"],
            readouts["bend"] - math.pi / 2.0,
            readouts["wflex"] * (70.0 / 45.0),
            wroll_center - 0.5 * (readouts["wroll"] - wroll_center),
        ),
        axis=1,
    )


class Smplx2So101:
    """Retarget one SMPL-X arm to the five SO-101 arm joints."""

    name = "so101_arm_v1"
    channels = ("1", "2", "3", "4", "5")
    working_range = tuple(
        (math.radians(lo), math.radians(hi))
        for lo, hi in ((-118.5, 118.5), (-90.0, 0.0), (-90.0, 45.0), (-70.0, 70.0), (-135.0, -45.0))
    )
    # Idle sway: +-10 deg on joints 1-4; joint 5 stays at its neutral -90 deg.
    idle_amplitude = (math.radians(10.0),) * 4 + (0.0,)

    def retarget(self, request: RetargetRequest) -> RetargetOutput:
        side = request.side
        if side not in _ARM:
            raise ValueError(f"unknown arm side {side!r}")
        track = request.track
        values = joint_angles(arm_readouts(neutral_fk(track.rotations), side), side)
        return RetargetOutput(self.channels, values, np.asarray(track.t, dtype=np.float64))
