"""The seven source readouts: SMPL-X in, one angle per channel per side out.

Each function here returns the `source_rad` that the ledger's formula consumes.
The readout is the part of the mapping that was hard to get right, and each one
is what the operator actually confirmed by eye, so the shape of a function
matters as much as its value:

* THREE WRIST CHANNELS read an Euler component of the wrist's LOCAL matrix.
  Euler works there because the wrist is distal, small and has no better
  positional description available.
* FOUR PROXIMAL CHANNELS do not. `elbow`, `shoulder_roll`, `shoulder_pitch` and
  `shoulder_yaw` are all read geometrically. That is not four coincidences: the
  Euler decomposition is systematically unusable on proximal joints, because
  the component that names a motion is not the component that carries it.
* `shoulder_yaw` is the one channel whose readout needs the ROBOT, per frame.
  Its source is her twist measured against the twist the robot already has at
  yaw zero, and that coupling is not a constant -- it sweeps ~90 deg over this
  clip. It therefore lives in `retarget.py`, where the robot is in hand; the
  source half of it is `upper_arm_twist_deg` below.

All angles are degrees inside this module, radians only at the boundary
(`*_rad` wrappers), because degrees is what a human checking the numbers reads.
"""

from __future__ import annotations

import numpy as np

from .source import (
    ELBOW_JOINT,
    SHOULDER_JOINT,
    SHOULDER_ROTATION_INDEX,
    SourceAnimation,
    WRIST_JOINT,
    lateral,
    torso_coordinates,
    torso_frame,
    unit,
)

# Reference the upper-arm twist is measured from. Torso forward is the
# best-conditioned choice available: world down is nearly parallel to the bone
# in an arms-down clip, which collapses the projected reference and makes the
# measured slope flip sign between the arms -- a basis artefact, not physics.
TORSO_FORWARD = np.array([1.0, 0.0, 0.0])


# ---- wrists: Euler components of the local matrix ------------------------

def wrist_euler_deg(animation: SourceAnimation, body_index: int, axis: int) -> np.ndarray:
    """One unwrapped intrinsic-XYZ component of a wrist's local rotation."""
    return np.degrees(animation.local_euler_xyz(body_index)[:, axis])


# ---- elbow: the angle you can see ----------------------------------------

def elbow_flexion_deg(points: np.ndarray, side: str) -> np.ndarray:
    """How far the forearm has folded away from the line of the upper arm.

    0 = straight. Positions only; no rotation matrix is touched, which is what
    lets this check a matrix readout rather than merely restate it. Unsigned by
    construction, which suits a hinge and sidesteps the Euler singularity.
    """
    upper = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    fore = unit(points[:, WRIST_JOINT[side]] - points[:, ELBOW_JOINT[side]])
    return np.degrees(np.arccos(np.clip((upper * fore).sum(-1), -1.0, 1.0)))


# ---- shoulder_roll: lift in the coronal plane ----------------------------

def abduction_deg(points: np.ndarray, side: str) -> np.ndarray:
    """How far the upper arm has lifted away from hanging, in the torso frame.

    0 = hanging straight down, +90 = straight out sideways (T-pose), negative =
    pressed across the body. Signed anatomically -- outward is positive on both
    arms -- and deliberately not signed by the robot joint's own turning sense,
    which would make the sign question answer itself.
    """
    frame = torso_frame(points)
    arm = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    return np.degrees(np.arctan2((arm * lateral(frame, side)).sum(-1),
                                 -(arm * frame["up"]).sum(-1)))


def coronal_norm(points: np.ndarray, side: str) -> np.ndarray:
    """Conditioning of `abduction_deg`: an arm pointing straight forward has none."""
    frame = torso_frame(points)
    arm = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    return np.hypot((arm * lateral(frame, side)).sum(-1), (arm * frame["up"]).sum(-1))


# ---- shoulder_pitch: swing in the ROBOT's hinge plane --------------------

def transplanted_axis(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """The robot's hinge axis rebuilt inside her torso frame, per frame.

    The robot's torso frame is the world frame, so the axis's world components
    ARE its (fwd, left, up) coefficients; rebuilding it as
    c_fwd*fwd + c_left*left + c_up*up in her frame carries the axis's 12 deg
    tilt across without extracting an angle or assuming a convention.
    """
    frame = torso_frame(points)
    axis = (coefficients[0] * frame["fwd"] + coefficients[1] * frame["left"]
            + coefficients[2] * frame["up"])
    return unit(axis)


def pitch_deg(points: np.ndarray, side: str, coefficients: np.ndarray) -> np.ndarray:
    """Fore-aft swing of her upper arm, read in the hinge's own plane.

    0 = hanging, +90 = straight forward, negative = behind her. Reading it
    against plain world down and forward instead would measure a cone against a
    plane: it needs its slope fitted to -1.056 and still leaves 0.96 deg of
    curvature. In the true hinge plane the readout is exact with slope 1.
    """
    from .robot import hinge_basis  # local: keeps the source side import-light

    frame = torso_frame(points)
    axis = transplanted_axis(points, coefficients)
    down, forward = hinge_basis(axis, -frame["up"], frame["fwd"])
    arm = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    return np.degrees(np.arctan2((arm * forward).sum(-1), (arm * down).sum(-1)))


def perpendicular_norm(points: np.ndarray, side: str, coefficients: np.ndarray) -> np.ndarray:
    """Conditioning of `pitch_deg`: an arm along the hinge axis has no pitch."""
    axis = transplanted_axis(points, coefficients)
    arm = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    return np.sqrt(np.clip(1.0 - (arm * axis).sum(-1) ** 2, 0.0, 1.0))


# ---- shoulder_yaw: twist about the bone ----------------------------------

def rest_direction(points: np.ndarray, rotations: np.ndarray, side: str
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The upper-arm-fixed pair (bone, perpendicular) that readout A rides on.

    The clip average of her own plane normal pulled back through the shoulder's
    FK rotation, orthogonalized against the pulled-back bone. Averaging is what
    makes readouts A and B share a zero: at the average frame they agree
    exactly, so any later difference is her elbow leaving the robot's hinge
    plane and nothing else.

    Because it is a clip average, it makes readout A depend on the whole clip.
    That is the second reason a clip is converted as a whole (the first is the
    Euler unwrap).
    """
    bone = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    forearm = unit(points[:, WRIST_JOINT[side]] - points[:, ELBOW_JOINT[side]])
    normal = unit(np.cross(bone, forearm))
    bone_rest = unit(np.einsum("nji,nj->ni", rotations, bone).mean(0))
    normal_rest = unit(np.einsum("nji,nj->ni", rotations, normal).mean(0))
    normal_rest = unit(normal_rest - (normal_rest @ bone_rest) * bone_rest)
    return bone_rest, normal_rest


def twist_indicators(animation: SourceAnimation, side: str) -> dict[str, np.ndarray]:
    """Her bone and the two candidate perpendiculars, all in her torso frame.

    `spin`   -- readout A, a direction fixed in the upper arm carried by the
                shoulder's FK rotation. Everywhere defined; makes the robot's
                UPPER ARM spin as hers does.
    `plane`  -- readout B, the normal of her own shoulder-elbow-wrist plane,
                from joint positions only. Makes the robot's FOREARM point where
                hers points -- and the hand rides on the forearm, which is why
                this is the confirmed readout.

    `conditioning` is |sin(elbow flexion)|, the conditioning of B: a straight
    arm has no plane, and B degenerates there. A is the fallback if a future
    clip straightens the arm.
    """
    points = animation.points
    rotations = animation.link_rotations(SHOULDER_ROTATION_INDEX[side])
    transform = torso_coordinates(points)

    def to_torso(vectors: np.ndarray) -> np.ndarray:
        return np.einsum("nij,nj->ni", transform, vectors)

    bone = unit(points[:, ELBOW_JOINT[side]] - points[:, SHOULDER_JOINT[side]])
    forearm = unit(points[:, WRIST_JOINT[side]] - points[:, ELBOW_JOINT[side]])
    normal = np.cross(bone, forearm)
    _, normal_rest = rest_direction(points, rotations, side)
    return {
        "bone": to_torso(bone),
        "forearm": to_torso(forearm),
        "spin": to_torso(np.einsum("nij,j->ni", rotations, normal_rest)),
        "plane": to_torso(unit(normal)),
        "conditioning": np.linalg.norm(normal, axis=-1),
    }


def upper_arm_twist_deg(indicators: dict[str, np.ndarray], readout: str = "plane") -> np.ndarray:
    """Her upper-arm twist about her own bone, in her torso frame, degrees."""
    from .robot import twist_about

    if readout not in ("plane", "spin"):
        raise ValueError(f"readout must be 'plane' (B) or 'spin' (A), got {readout!r}")
    return twist_about(indicators["bone"], indicators[readout], TORSO_FORWARD)


def reference_norm(indicators: dict[str, np.ndarray]) -> np.ndarray:
    """How much of the torso-forward reference survives projection off the bone.

    The conditioning of the ANGLE, as distinct from `conditioning`, which is the
    conditioning of readout B's indicator. An arm pointing straight forward has
    no reference direction left and the twist would be meaningless there.
    """
    bone = indicators["bone"]
    return np.linalg.norm(np.cross(np.broadcast_to(TORSO_FORWARD, bone.shape), bone), axis=-1)
