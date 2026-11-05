"""The X2 arms the composer plans for: the official MuJoCo model, pelvis welded.

``configure()`` is called once with the model path and the runtime profile; every
other module asks ``load()``. From the profile come the arm joint order, the
joint travel the runtime enforces (``vendor.arm.limits``), the measured rest pose
(``conformance.start_positions``) and the recorded gestures, whose extremes the
safe range keeps: the platform drives them itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

SIDES = ("left", "right")
ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)
ARM_JOINTS: tuple[str, ...] = tuple(f"{side}_{s}_joint" for side in SIDES for s in ARM_SUFFIXES)
# Rated joint torque from the official URDF (<limit effort>), Nm.
EFFORT_NM = {"shoulder_pitch": 36.0, "shoulder_roll": 36.0, "shoulder_yaw": 24.0, "elbow": 24.0,
             "wrist_yaw": 24.0, "wrist_pitch": 4.8, "wrist_roll": 4.8}  # fmt: skip
LIMIT_MARGIN_RAD = 0.05


@dataclass(frozen=True)
class Robot:
    model: mujoco.MjModel
    arm_joints: tuple[str, ...]
    qadr: dict[str, int]
    safe_limits: dict[str, tuple[float, float]]
    rest: dict[str, float]

    def new_data(self) -> mujoco.MjData:
        return mujoco.MjData(self.model)


_ROBOT: Robot | None = None


def _recorded_extremes(profile: dict, profile_dir: Path) -> dict[str, tuple[float, float]]:
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for rel in ((profile.get("vendor") or {}).get("named_motion") or {}).get("trajectories", {}).values():
        recorded = json.loads((profile_dir / rel).read_text())
        for i, name in enumerate(recorded["joints"]):
            values = [row[i] for row in recorded["positions"]]
            lo[name] = min(lo.get(name, values[0]), min(values))
            hi[name] = max(hi.get(name, values[0]), max(values))
    return {name: (lo[name], hi[name]) for name in lo}


def configure(mjcf_path: str | Path, profile_path: str | Path) -> Robot:
    """Load the model and the runtime profile; the result is what ``load()`` returns."""
    global _ROBOT
    profile_path = Path(profile_path)
    profile = yaml.safe_load(profile_path.read_text())
    if tuple(profile["arm_joints"]) != ARM_JOINTS:
        raise ValueError(f"unexpected arm joint order in {profile_path}")
    limits = ((profile.get("vendor") or {}).get("arm") or {}).get("limits") or {}
    if set(limits) != set(ARM_JOINTS):
        raise ValueError("the runtime profile must declare vendor.arm.limits for every arm joint")
    rest = {str(k): float(v) for k, v in ((profile.get("conformance") or {}).get("start_positions") or {}).items()}
    if set(rest) != set(ARM_JOINTS):
        raise ValueError("the runtime profile must declare conformance.start_positions (the rest pose)")
    recorded = _recorded_extremes(profile, profile_path.parent)
    safe = {}
    for name in ARM_JOINTS:
        low, high = float(limits[name][0]), float(limits[name][1])
        lo, hi = low + LIMIT_MARGIN_RAD, high - LIMIT_MARGIN_RAD
        rec_lo, rec_hi = recorded.get(name, (lo, hi))
        safe[name] = (max(low, min(lo, rec_lo, rest[name])), min(high, max(hi, rec_hi, rest[name])))
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(joint)
    model = spec.compile()
    qadr = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(model.njnt)}
    missing = sorted(set(ARM_JOINTS) - set(qadr))
    if missing:
        raise ValueError(f"model lacks arm joints {missing}")
    _ROBOT = Robot(model, ARM_JOINTS, qadr, safe, rest)
    return _ROBOT


def load() -> Robot:
    if _ROBOT is None:
        raise RuntimeError("pose_composer.model.configure() has not been called")
    return _ROBOT


def to_qpos(robot: Robot, q: dict[str, float]) -> np.ndarray:
    qpos = np.zeros(robot.model.nq)
    for name, value in q.items():
        qpos[robot.qadr[name]] = value
    return qpos
