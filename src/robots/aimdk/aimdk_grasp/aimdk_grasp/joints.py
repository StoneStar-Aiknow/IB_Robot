"""X2 joint names the grasp pipeline reads, in the platform's own order."""

from __future__ import annotations

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
#: Every body joint a grasp observation records: both arms (the right arm and the
#: head are locked into the collision model) and the head.
BODY_JOINTS = LEFT_ARM + RIGHT_ARM + HEAD
