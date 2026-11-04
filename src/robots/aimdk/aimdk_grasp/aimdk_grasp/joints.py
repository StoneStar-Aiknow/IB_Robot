"""X2 joint names the grasp pipeline reads, in the platform's own order."""

from __future__ import annotations

import numpy as np

ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")
LEFT_ARM = tuple(f"left_{part}_joint" for part in ARM_PARTS)
RIGHT_ARM = tuple(f"right_{part}_joint" for part in ARM_PARTS)
HEAD = ("head_yaw_joint", "head_pitch_joint")
#: The ten driven OmniHand finger joints; distal joints follow them mechanically
#: and are not measured.
HAND_PARTS = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)
LEFT_HAND = tuple(f"L_{part}_joint" for part in HAND_PARTS)
RIGHT_HAND = tuple(f"R_{part}_joint" for part in HAND_PARTS)
#: The left hand is mounted in the planning model as ``R_*`` aliases; an alias
#: position is the native one times its sign: joints turning about the link's x
#: or z axis are mirrored, flexion joints about y are not.
HAND_ALIAS_SIGNS = {
    "thumb_roll": -1,
    "thumb_abad": -1,
    "thumb_mcp": -1,
    "thumb_pip": -1,
    "thumb_dip": -1,
    "index_abad": -1,
    "ring_abad": -1,
    "pinky_abad": -1,
    **{f"{finger}_{joint}": 1 for finger in ("index", "middle", "ring", "pinky") for joint in ("pip", "dip")},
}
HAND_ALIASES = tuple(f"R_{part}_joint" for part in HAND_PARTS)
HAND_SIGNS = np.array([HAND_ALIAS_SIGNS[part] for part in HAND_PARTS], dtype=np.float64)
HAND_SIGNS.setflags(write=False)
#: Unmeasured distal joints as polynomials (low order first) of their driving
#: joint, both as aliases. Converted from the equality constraints of the AimDK
#: SDK OmniHand MuJoCo model (x2_31dof_omnihand_2.xml); the official URDF's
#: linear ``mimic`` approximates the same coupling.
DIP_FOLLOWERS = {
    "R_thumb_dip_joint": ("R_thumb_mcp_joint", (0.0, 1.846, -0.853, 0.280, 0.0)),
    **{
        f"R_{finger}_dip_joint": (f"R_{finger}_pip_joint", (0.0, 2.192, -1.425, 0.747, -0.167))
        for finger in ("index", "middle", "ring", "pinky")
    },
}
#: Every body joint a grasp observation records: both arms (the right arm and the
#: head are locked into the collision model) and the head.
BODY_JOINTS = LEFT_ARM + RIGHT_ARM + HEAD

#: The arms a grasp can use. The planning model of a side carries that side's
#: hand as the ``R_*`` aliases (the left hand mirrored, the right hand native).
SIDES = ("left", "right")
#: Right-arm angle = sign x left-arm angle for mirror poses about the torso's
#: sagittal (y = 0) plane, in ``ARM_PARTS`` order: pitch and elbow keep their
#: sign, roll and yaw joints flip it (checked on the planning URDF).
ARM_MIRROR_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
ARM_MIRROR_SIGNS.setflags(write=False)


def check_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return side


def arm_joints(side: str) -> tuple[str, ...]:
    """The seven arm joints of ``side``, in ``ARM_PARTS`` order."""
    return LEFT_ARM if check_side(side) == "left" else RIGHT_ARM


def hand_joints(side: str) -> tuple[str, ...]:
    """The ten driven finger joints of ``side``'s hand as the runtime names them, in ``HAND_PARTS`` order."""
    return LEFT_HAND if check_side(side) == "left" else RIGHT_HAND


def hand_alias_signs(side: str) -> np.ndarray:
    """Alias = native x sign, in ``HAND_PARTS`` order: the left hand is mounted
    mirrored as the ``R_*`` aliases, the right hand natively (all +1)."""
    return HAND_SIGNS if check_side(side) == "left" else _NATIVE_SIGNS


_NATIVE_SIGNS = np.ones(len(HAND_PARTS))
_NATIVE_SIGNS.setflags(write=False)


def other_side(side: str) -> str:
    return "right" if check_side(side) == "left" else "left"


def swap_side_name(name: str) -> str:
    """``left_*`` <-> ``right_*``; other names unchanged."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name
