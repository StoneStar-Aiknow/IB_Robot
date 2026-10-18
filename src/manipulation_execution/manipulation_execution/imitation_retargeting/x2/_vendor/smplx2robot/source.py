"""The SMPL-X side: load the animation, and rebuild the quantities readouts need.

Two independent descriptions of the same body come out of the fit, and the
retargeting uses both:

* JOINT POSITIONS (`ehm_joints_3d_model`, [N,145,3]) -- what you can see. Five
  of the seven channels are read from positions alone, which is what lets them
  check a matrix readout instead of merely restating it.
* LINK ROTATIONS (`global_orient_matrix` + `body_pose_matrix`) -- what positions
  cannot see. The upper arm's spin about its own long axis moves no joint
  centre at all, so `shoulder_yaw`'s candidate readout A can only come from
  here, and the three wrist channels are read as Euler components of the
  wrist's local matrix.

Nothing in this module knows about the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import pear

# Index into `ehm_joints_3d_model`. Body joints are 0..21; the hand joints from
# 25 are carried for completeness and are not used by any confirmed channel.
JOINT: dict[str, int] = {
    "pelvis": 0, "l_hip": 1, "r_hip": 2, "spine1": 3, "l_knee": 4, "r_knee": 5,
    "spine2": 6, "l_ankle": 7, "r_ankle": 8, "spine3": 9, "l_foot": 10, "r_foot": 11,
    "neck": 12, "l_collar": 13, "r_collar": 14, "head": 15, "l_shoulder": 16,
    "r_shoulder": 17, "l_elbow": 18, "r_elbow": 19, "l_wrist": 20, "r_wrist": 21,
}

# Kinematic parent of each of the 21 body_pose joints, in body_pose indexing.
BODY_PARENTS: tuple[int, ...] = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

SHOULDER_JOINT = {"left": JOINT["l_shoulder"], "right": JOINT["r_shoulder"]}
ELBOW_JOINT = {"left": JOINT["l_elbow"], "right": JOINT["r_elbow"]}
WRIST_JOINT = {"left": JOINT["l_wrist"], "right": JOINT["r_wrist"]}

# Index into the 22 entries `body_global_rotations` returns: entry 0 is the
# pelvis and entry i+1 is body_pose joint i.
SHOULDER_ROTATION_INDEX = {"left": 16, "right": 17}

REQUIRED_KEYS: tuple[str, ...] = (
    "frame_id",
    "timestamp_ns",
    "ehm_joints_3d_model",
    "body_pose_matrix",
    "global_orient_matrix",
)


def unit(vectors: np.ndarray) -> np.ndarray:
    """Normalise along the last axis. Batched over every leading axis."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def body_global_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """[22,3,3] model-frame rotation of every body link, for ONE frame.

    Entry 0 is the pelvis (the global orientation itself); entry i+1 composes
    body_pose joint i onto its parent.
    """
    out = np.empty((len(BODY_PARENTS) + 1, 3, 3), dtype=np.float64)
    out[0] = global_orient
    for body_index, parent in enumerate(BODY_PARENTS):
        out[body_index + 1] = out[parent] @ body_pose[body_index]
    return out


def continuous_intrinsic_xyz(rotations: np.ndarray) -> np.ndarray:
    """[N,3] intrinsic-XYZ Euler angles in radians, unwrapped along time.

    Unwrapping is not cosmetic. `as_euler` returns a principal value, so a joint
    that walks past +-180 deg comes back with a jump the size of the range, and
    a mapping applied to that produces a full-speed reversal on the robot. The
    unwrap makes the sequence continuous, which means the value on a frame
    depends on the frames before it -- deliberate, and the reason a clip must be
    converted as a whole rather than frame by frame.
    """
    angles = Rotation.from_matrix(np.asarray(rotations, dtype=np.float64)).as_euler(
        "XYZ", degrees=False
    )
    return np.unwrap(angles, axis=0)


def torso_frame(points: np.ndarray) -> dict[str, np.ndarray]:
    """An orthonormal torso frame from joint positions alone, per frame.

    Every shoulder readout is relative to the trunk: an arm 90 deg from vertical
    in the world is not abducted if she is lying down. Built from pelvis, neck
    and the two collars so it inherits no axis convention from SMPL-X, and
    explicitly orthogonalized -- the collar vector is not square to the spine.
    """
    points = np.asarray(points, dtype=np.float64)
    up = unit(points[:, JOINT["neck"]] - points[:, JOINT["pelvis"]])
    across = points[:, JOINT["l_collar"]] - points[:, JOINT["r_collar"]]
    left = unit(across - (across * up).sum(-1, keepdims=True) * up)
    return {"up": up, "left": left, "fwd": np.cross(left, up)}


def torso_coordinates(points: np.ndarray) -> np.ndarray:
    """[N,3,3] taking a model-frame direction to (fwd, left, up) coefficients.

    The robot's base is pinned and its waist is zero, so the robot's torso frame
    IS the world frame; expressing her directions in her own torso frame puts
    both sides of a comparison in the same coordinates without either inheriting
    an axis convention from the other.
    """
    frame = torso_frame(points)
    return np.stack([frame["fwd"], frame["left"], frame["up"]], axis=-2)


def lateral(frame: dict[str, np.ndarray], side: str) -> np.ndarray:
    """Outward direction for one arm: +left on the left, -left on the right."""
    return frame["left"] if side == "left" else -frame["left"]


@dataclass(frozen=True)
class SourceAnimation:
    """One clip of fitted SMPL-X body motion, with its timestamps."""

    path: Path
    frame_id: np.ndarray          # [N] int
    timestamp_ns: np.ndarray      # [N] int64
    points: np.ndarray            # [N,145,3] model-frame joint positions
    body_pose: np.ndarray         # [N,21,3,3] local rotations
    global_orient: np.ndarray     # [N,3,3] pelvis rotation

    @classmethod
    def load(cls, path: Path | str) -> "SourceAnimation":
        """Load a canonical source-animation npz, or a PEAR clip directory.

        A directory is accepted because that is the shape the fitter actually
        writes -- three files under one clip name. `pear.load_clip` renames its
        arrays and nothing else, so a directory and the npz `adapt` writes from
        it produce the same animation.
        """
        path = Path(path)
        if path.is_dir():
            if not pear.is_clip_directory(path):
                raise FileNotFoundError(
                    f"{path} is a directory but not a PEAR clip; a clip holds "
                    f"{list(pear.CLIP_FILES)}"
                )
            clip = pear.load_clip(path)
            animation = cls(
                path=path.resolve(),
                frame_id=clip["frame_id"],
                timestamp_ns=clip["timestamp_ns"],
                points=clip["ehm_joints_3d_model"],
                body_pose=clip["body_pose_matrix"],
                global_orient=clip["global_orient_matrix"],
            )
            animation.validate()
            return animation
        if not path.is_file():
            raise FileNotFoundError(f"source animation not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data]
            if missing:
                # ValueError, not KeyError: from the caller's point of view the
                # fault is "this npz is not a source animation", which is the
                # same fault a wrong array shape is. One exception type to catch.
                raise ValueError(f"{path.name} is missing keys {missing}; "
                                 f"present: {sorted(data.files)}")
            animation = cls(
                path=path.resolve(),
                frame_id=np.asarray(data["frame_id"]).astype(np.int64),
                timestamp_ns=np.asarray(data["timestamp_ns"]).astype(np.int64),
                points=np.asarray(data["ehm_joints_3d_model"], dtype=np.float64),
                body_pose=np.asarray(data["body_pose_matrix"], dtype=np.float64),
                global_orient=np.asarray(data["global_orient_matrix"], dtype=np.float64),
            )
        animation.validate()
        return animation

    def validate(self) -> None:
        frames = len(self.frame_id)
        shapes = {
            "timestamp_ns": (self.timestamp_ns.shape, (frames,)),
            "ehm_joints_3d_model": (self.points.shape[:1] + self.points.shape[2:], (frames, 3)),
            "body_pose_matrix": (self.body_pose.shape, (frames, 21, 3, 3)),
            "global_orient_matrix": (self.global_orient.shape, (frames, 3, 3)),
        }
        for key, (got, want) in shapes.items():
            if got != want:
                raise ValueError(f"{key} has shape {got}, expected {want}")
        if self.points.shape[1] <= max(JOINT.values()):
            raise ValueError(f"ehm_joints_3d_model has only {self.points.shape[1]} joints")
        for name, array in (("points", self.points), ("body_pose", self.body_pose),
                            ("global_orient", self.global_orient)):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if frames == 0:
            raise ValueError("animation has no frames")

    def __len__(self) -> int:
        return len(self.frame_id)

    @cached_property
    def rotations(self) -> np.ndarray:
        """[N,22,3,3] model-frame rotation of every body link, all frames."""
        return np.stack([
            body_global_rotations(self.global_orient[f], self.body_pose[f])
            for f in range(len(self))
        ])

    def link_rotations(self, index: int) -> np.ndarray:
        """[N,3,3] model-frame rotation of one link, in `rotations` indexing."""
        return self.rotations[:, index]

    def local_euler_xyz(self, body_index: int) -> np.ndarray:
        """[N,3] unwrapped intrinsic-XYZ of one joint's LOCAL matrix, radians."""
        return continuous_intrinsic_xyz(self.body_pose[:, body_index])

    @property
    def duration_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.timestamp_ns[-1] - self.timestamp_ns[0]) * 1e-9

    @property
    def time_s(self) -> np.ndarray:
        """Seconds since the first frame. Preserves the clip's real gaps."""
        return (self.timestamp_ns - self.timestamp_ns[0]).astype(np.float64) * 1e-9
