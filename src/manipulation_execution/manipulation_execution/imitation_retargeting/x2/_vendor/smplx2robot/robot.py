"""The X2 side: forward kinematics on the shipped model, with no graphics.

Three things are read from the robot and never hardcoded:

* JOINT LIMITS, which bound the trajectory;
* HINGE AXES, in particular `shoulder_pitch`'s, which is 12 deg off world +y and
  is transplanted into her torso frame so the readout matches the hinge;
* GEOMETRY AT THE ZERO POSE -- the built-in bend, splay and forward hang that
  make up three of the seven offsets, and the per-frame twist coupling that
  `shoulder_yaw` is read against.

Everything is measured at joint ANCHORS (`data.xanchor`), never at link origins
(`data.xpos`). An anchor is the point the joint turns about; a link origin can
sit centimetres away and moves when the joint under test moves, which makes it
useless as a support point.

The base is pinned and every non-arm joint is held at zero, so the robot's torso
frame IS the world frame. That is what lets an axis measured in world
coordinates be read directly as (fwd, left, up) coefficients in her frame.

`mj_forward` only. Never `mj_step`: this is kinematics, not simulation, and a
dynamics step would let gravity move joints that were commanded elsewhere.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .paths import model_path
from .source import unit

SIDES: tuple[str, ...] = ("left", "right")

# arm_pos index order. This tuple IS the contract with the ledger's arm_pos
# indices and with every downstream consumer.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
ARM_DOF = len(ARM_JOINT_NAMES)

# Floating base held fixed: the robot neither falls nor drifts, and its torso
# frame stays aligned with the world.
BASE_QPOS: tuple[float, ...] = (0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0)

PITCH_JOINT = {side: f"{side}_shoulder_pitch_joint" for side in SIDES}
YAW_JOINT = {side: f"{side}_shoulder_yaw_joint" for side in SIDES}
# Bone endpoints. The roll anchor is the one point in the shoulder that neither
# pitch, roll nor yaw moves; the yaw anchor sits 12 cm down the arm and is not a
# support point.
BONE_PROXIMAL = {side: f"{side}_shoulder_roll_joint" for side in SIDES}
BONE_DISTAL = {side: f"{side}_elbow_joint" for side in SIDES}
FOREARM_DISTAL = {side: f"{side}_wrist_yaw_joint" for side in SIDES}

WORLD_DOWN = np.array([0.0, 0.0, -1.0])
WORLD_FORWARD = np.array([1.0, 0.0, 0.0])

# ---- mirror imitation -------------------------------------------------------
#
# Mirror imitation flips the COMMAND, never the picture: her right arm drives
# the robot's left arm and her left drives its right. Per channel that costs
# exactly one sign, and the signs come from the robot's own geometry.
#
# The base is pinned, so world = torso: x forward, y left, z up. The sagittal
# plane's normal is therefore y and the reflection is M = diag(1, -1, 1). A
# hinge axis is a PSEUDOVECTOR -- it picks up one extra sign flip under a
# reflection -- so it transforms as
#
#     w  ->  det(M) * M @ w  =  diag(-1, +1, -1) @ w
#
# An axis along y keeps its sign; an axis along x or z reverses. That single
# rule produces every entry below.
#
# Three independent sources agree on the result digit for digit: the axis table
# in `手写版.md` §1, measured on the physical robot; this model's own
# `jnt_axis`, which `mirror_signs_from_model` re-derives and a test pins these
# numbers to; and the ledger's own confirmed signs, via sign_L * sign_R * mu.
#
# §4.1 of that same document says the two arms' frames are COPIED, not mirrored,
# and the model agrees literally -- all 14 `jnt_axis` vectors are identical. That
# is the whole reason a sign table is needed: were the frames mirrored, every
# entry here would be +1 and mirroring would be a plain left/right swap.
MIRROR_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

# Indexed by arm_pos slot 0..6, which is also `ledger.CHANNEL_ORDER`.
MIRROR_SIGNS: tuple[int, ...] = (+1, -1, -1, +1, -1, +1, -1)

# Rigid probes for the mirror self-check. Four offsets on the wrist_roll link
# rather than one point, so the link's ORIENTATION is checked and not only where
# it sits; two anchors further up the arm so a fault in the elbow or forearm
# shows even when the hand happens to land right.
MIRROR_PROBE_BODY = {side: f"{side}_wrist_roll_link" for side in SIDES}
MIRROR_PROBE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
)
MIRROR_PROBE_ANCHORS = (BONE_DISTAL, FOREARM_DISTAL)


def hinge_basis(axis: np.ndarray, down: np.ndarray, forward: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal 2D basis for the plane a hinge turns in. Works batched.

    Reading a swing against plain down and forward measures a cone against a
    plane and needs a fitted slope; reading it in the plane the hinge actually
    turns in makes it an exact hinge, so the scale is 1 by construction.

    Both inputs are orthogonalized against the axis AND against each other. The
    second half costs nothing for these particular axes -- they have no forward
    component -- but that is a property of this URDF, not of the construction,
    and an atan2 taken in a skewed basis is not an angle.
    """
    axis = unit(axis)
    down = unit(down - (down * axis).sum(-1, keepdims=True) * axis)
    forward = forward - (forward * axis).sum(-1, keepdims=True) * axis
    return down, unit(forward - (forward * down).sum(-1, keepdims=True) * down)


def twist_about(axis: np.ndarray, vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Right-handed angle of `vector` about `axis`, from `reference` projected square.

    Deliberately not assembled out of `hinge_basis`: that helper leaves the
    handedness of its output up to its two inputs, and here handedness IS the
    measurement, so the second basis vector is built as `axis x e1` and cannot
    come out left-handed.

    Batched over leading axes. Returns NaN, loudly, if the reference is parallel
    to the axis, rather than a plausible-looking number.
    """
    axis = unit(axis)
    reference = np.asarray(reference, dtype=np.float64)
    e1 = unit(reference - (reference * axis).sum(-1, keepdims=True) * axis)
    e2 = np.cross(axis, e1)
    return np.degrees(np.arctan2((vector * e2).sum(-1), (vector * e1).sum(-1)))


def wrap_deg(values: np.ndarray | float) -> np.ndarray:
    """Fold an angle difference into (-180, 180]. atan2 readouts wrap; angles do not."""
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


class X2Kinematics:
    """Pose the arms, read anchors and axes. No renderer, no GL, no dynamics."""

    def __init__(self, model_file: Path | str | None = None) -> None:
        path = Path(model_file) if model_file is not None else model_path()
        if not Path(path).is_file():
            raise FileNotFoundError(f"robot model not found: {path}")
        self.model_file = Path(path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_file))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = {name: self._joint_id(name) for name in ARM_JOINT_NAMES}
        self._arm_adr = np.array(
            [self.model.jnt_qposadr[self._joint_ids[name]] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        limits = np.array(
            [self.model.jnt_range[self._joint_ids[name]] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        limits.setflags(write=False)
        self.limits_rad = limits
        self.set_arm(np.zeros(ARM_DOF))

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"joint {name!r} is not present in {self.model_file.name}")
        return joint_id

    @property
    def limits_deg(self) -> np.ndarray:
        return np.degrees(self.limits_rad)

    def set_arm(self, arm_pos_rad: np.ndarray) -> None:
        """Place the arms at one [14] pose in radians. Everything else at zero."""
        frame = np.asarray(arm_pos_rad, dtype=np.float64).reshape(-1)
        if frame.shape != (ARM_DOF,):
            raise ValueError(f"expected a [{ARM_DOF}] pose, got {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError("pose contains non-finite values")
        self.data.qpos[:] = 0.0
        self.data.qpos[:7] = BASE_QPOS
        self.data.qpos[self._arm_adr] = frame
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def anchor(self, joint_name: str) -> np.ndarray:
        """World position of a joint's rotation centre at the current pose."""
        return self.data.xanchor[self._joint_id(joint_name)].copy()

    def axis(self, joint_name: str) -> np.ndarray:
        """World direction of a joint's axis at the current pose."""
        return self.data.xaxis[self._joint_id(joint_name)].copy()

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """World (position, 3x3 rotation) of a link at the current pose.

        Unlike `anchor`, a link origin moves when the joint under it moves, so
        this is only ever used for points that are rigidly attached to the link
        and want the link's ORIENTATION -- never as a support point.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"body {body_name!r} is not present in {self.model_file.name}")
        return (self.data.xpos[body_id].copy(),
                self.data.xmat[body_id].reshape(3, 3).copy())

    # ---- geometry the readouts need -------------------------------------

    def pitch_axes(self) -> dict[str, np.ndarray]:
        """Each `shoulder_pitch` hinge axis in world = torso coordinates, at zero.

        These three numbers per side are directly the axis's (fwd, left, up)
        coefficients, which is what lets the same axis be rebuilt inside her
        torso frame without extracting an angle from either body.
        """
        self.set_arm(np.zeros(ARM_DOF))
        return {side: self.axis(PITCH_JOINT[side]) for side in SIDES}

    def yaw_axes(self) -> dict[str, np.ndarray]:
        """Each `shoulder_yaw` axis at zero. Reported, not transplanted: it sits
        4.06 deg off the bone, and it is the BONE both bodies can agree on."""
        self.set_arm(np.zeros(ARM_DOF))
        return {side: self.axis(YAW_JOINT[side]) for side in SIDES}

    def pitch_deg(self, side: str, arm_pos_rad: np.ndarray, axis: np.ndarray) -> float:
        """Fore-aft swing of the upper arm in the hinge's own plane, at a pose."""
        self.set_arm(arm_pos_rad)
        arm = unit(self.anchor(BONE_DISTAL[side]) - self.anchor(PITCH_JOINT[side]))
        down, forward = hinge_basis(axis, WORLD_DOWN, WORLD_FORWARD)
        return float(np.degrees(np.arctan2(arm @ forward, arm @ down)))

    def abduction_deg(self, side: str, arm_pos_rad: np.ndarray) -> float:
        """Lift of the upper arm away from hanging, at a pose. +90 = T-pose."""
        self.set_arm(arm_pos_rad)
        arm = unit(self.anchor(BONE_DISTAL[side]) - self.anchor(BONE_PROXIMAL[side]))
        sideways = 1.0 if side == "left" else -1.0
        return float(np.degrees(np.arctan2(sideways * arm[1], -arm[2])))

    def flexion_deg(self, side: str, arm_pos_rad: np.ndarray) -> float:
        """Elbow fold between upper arm and forearm, at a pose. 0 = straight."""
        self.set_arm(arm_pos_rad)
        upper = unit(self.anchor(BONE_DISTAL[side]) - self.anchor(BONE_PROXIMAL[side]))
        fore = unit(self.anchor(FOREARM_DISTAL[side]) - self.anchor(BONE_DISTAL[side]))
        return float(np.degrees(np.arccos(np.clip(upper @ fore, -1.0, 1.0))))

    def plane_twist_deg(self, side: str, arm_pos_rad: np.ndarray) -> float:
        """Twist of the robot's own shoulder-elbow-wrist plane about the bone.

        The readout the robot can express: its elbow is a single hinge, so the
        plane normal is very nearly the upper arm's own spin. "Very nearly", not
        exactly -- the hinge axis is 0.056 deg off perpendicular to the bone and
        the wrist anchor sits 0.10 mm off the plane, so sweeping the elbow moves
        this by up to ~1.6 deg. That is a measured tolerance, not an identity,
        and two orders of magnitude under what it is used to decide.
        """
        self.set_arm(arm_pos_rad)
        shoulder = self.anchor(BONE_PROXIMAL[side])
        elbow = self.anchor(BONE_DISTAL[side])
        wrist = self.anchor(FOREARM_DISTAL[side])
        bone = unit(elbow - shoulder)
        return float(twist_about(bone, unit(np.cross(bone, unit(wrist - elbow))),
                                 WORLD_FORWARD))

    def built_in_pitch_deg(self, axes: dict[str, np.ndarray] | None = None) -> dict[str, float]:
        """How far forward each upper arm already hangs at arm_pos = 0.

        Measured per side: the two differ by 0.35 deg because the URDF elbow
        anchors are not mirror images. Since the joint is an exact hinge, this
        one measurement pins the whole curve.
        """
        axes = self.pitch_axes() if axes is None else axes
        zero = np.zeros(ARM_DOF)
        return {side: self.pitch_deg(side, zero, axes[side]) for side in SIDES}

    def built_in_splay_deg(self) -> dict[str, float]:
        """How far each upper arm is already held away from the body at zero."""
        zero = np.zeros(ARM_DOF)
        return {side: self.abduction_deg(side, zero) for side in SIDES}

    def built_in_bend_deg(self) -> dict[str, float]:
        """How far each elbow is already folded at zero."""
        zero = np.zeros(ARM_DOF)
        return {side: self.flexion_deg(side, zero) for side in SIDES}

    def clip_to_limits(self, arm_pos_rad: np.ndarray) -> np.ndarray:
        """Clamp a [14] or [N,14] command into the model's own joint ranges."""
        array = as_arm_batch(arm_pos_rad)
        held = np.clip(array, self.limits_rad[:, 0], self.limits_rad[:, 1])
        return held.reshape(np.shape(arm_pos_rad))


def as_arm_batch(arm_pos: np.ndarray) -> np.ndarray:
    """Validate and normalise to [N,14] float64."""
    array = np.asarray(arm_pos, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != ARM_DOF:
        raise ValueError(
            f"arm_pos must have shape [{ARM_DOF}] or [N,{ARM_DOF}], got {np.shape(arm_pos)}"
        )
    if not np.isfinite(array).all():
        raise ValueError("arm_pos contains non-finite values")
    return array


def mirror_signs_from_model(kinematics: X2Kinematics) -> tuple[int, ...]:
    """Re-derive `MIRROR_SIGNS` from the model's hinge axes. No table consulted.

    Under the reflection M a rotation about axis w by angle q becomes a rotation
    about `det(M) * M @ w` by the SAME angle. So for each channel: reflect the
    left joint's world axis that way and ask which multiple of the right joint's
    world axis it is. +1 means the two joints must be commanded the same number,
    -1 means opposite.

    World axes, not the local `jnt_axis`. The local axes are identical on both
    sides -- the frames are copied, which is what `手写版.md` §4.1 says -- and
    the mirroring lives entirely in how the two shoulder bodies are mounted. In
    world coordinates that mounting is included, so this asks the question of
    the assembled robot rather than of a convention.

    Tolerance is 1e-4 on the dot product: the model's axes are stored single
    precision and the two sides agree to about 2e-5, while the decision being
    made is between +1 and -1.
    """
    kinematics.set_arm(np.zeros(ARM_DOF))
    pseudo = np.array([-1.0, 1.0, -1.0])       # det(M) * M for M = diag(1,-1,1)
    signs: list[int] = []
    for index in range(ARM_DOF // 2):
        reflected = pseudo * kinematics.axis(ARM_JOINT_NAMES[index])
        alignment = float(reflected @ kinematics.axis(ARM_JOINT_NAMES[index + ARM_DOF // 2]))
        if abs(abs(alignment) - 1.0) > 1e-4:
            raise ValueError(
                f"{ARM_JOINT_NAMES[index]} reflects to an axis {alignment:.6f} "
                f"aligned with its right-hand counterpart, not +-1 -- this "
                f"model's arms are not sagittally related and command-level "
                f"mirroring does not apply to it"
            )
        signs.append(1 if alignment > 0 else -1)
    return tuple(signs)


def mirror_plane_y(kinematics: X2Kinematics) -> float:
    """y of the sagittal plane, midway between the two shoulder pitch anchors.

    Measured rather than assumed to be zero, so the check stays honest on a
    model whose torso is not centred on its own origin. On this one it is 1e-6 m.
    """
    kinematics.set_arm(np.zeros(ARM_DOF))
    return 0.5 * float(kinematics.anchor(PITCH_JOINT["left"])[1]
                       + kinematics.anchor(PITCH_JOINT["right"])[1])


def _mirror_probes(kinematics: X2Kinematics, side: str) -> np.ndarray:
    """[6,3] world points on one arm at the pose the robot is already in.

    The left arm is probed at the local offsets; the RIGHT arm is probed at
    their reflections. That asymmetry is the whole subtlety of this check: a
    reflection turns a right-handed frame into a left-handed one, so the
    mirrored link's own axes are the reflections of the original's. Probing both
    sides at the same local offset instead measures that handedness flip and
    reports a constant error of twice the offset -- 200 mm here, which looks
    exactly like a sign error and is not one.
    """
    offsets = np.asarray(MIRROR_PROBE_OFFSETS, dtype=np.float64)
    if side == "right":
        offsets = offsets @ MIRROR_MATRIX.T
    position, rotation = kinematics.body_pose(MIRROR_PROBE_BODY[side])
    points = [position + rotation @ offset for offset in offsets]
    points.extend(kinematics.anchor(joint[side]) for joint in MIRROR_PROBE_ANCHORS)
    return np.asarray(points, dtype=np.float64)


def mirror_residual_mm(kinematics: X2Kinematics, arm_pos: np.ndarray,
                       mirrored_arm_pos: np.ndarray) -> dict[str, float]:
    """How far the mirrored command misses being a true reflection, in mm.

    Per frame the robot is posed twice -- at the command and at the mirrored
    command -- and both arms are probed in both poses. The mirrored pose's LEFT
    arm is compared with the reflection of the original's RIGHT arm, and its
    right with the reflection of the original's left, so all fourteen channels
    of the mirrored command are covered rather than only the left block.

    This is a geometric proof, not a regression test. Measured on the reference
    clip: the weakest single wrong entry in `MIRROR_SIGNS` -- wrist_pitch --
    still puts the residual at 24 mm rms and 94 mm max, while a correct mirror
    leaves 0.9 mm rms and 2.8 mm max. That floor is the robot's own asymmetry,
    almost all of it the 2.5 mm the two arms sit apart fore-aft, which is along
    x and so is not something a sign table could ever fix. Nothing lives in
    between the two.
    """
    original = as_arm_batch(arm_pos)
    mirrored = as_arm_batch(mirrored_arm_pos)
    if original.shape != mirrored.shape:
        raise ValueError(f"shapes differ: {original.shape} vs {mirrored.shape}")
    plane_y = mirror_plane_y(kinematics)

    def reflect(points: np.ndarray) -> np.ndarray:
        out = points.copy()
        out[:, 1] = 2.0 * plane_y - out[:, 1]
        return out

    errors = np.empty((len(original), 2 * len(MIRROR_PROBE_OFFSETS) + 4))
    for frame in range(len(original)):
        kinematics.set_arm(original[frame])
        source = {side: _mirror_probes(kinematics, side) for side in SIDES}
        kinematics.set_arm(mirrored[frame])
        image = {side: _mirror_probes(kinematics, side) for side in SIDES}
        errors[frame] = np.concatenate([
            np.linalg.norm(image["left"] - reflect(source["right"]), axis=1),
            np.linalg.norm(image["right"] - reflect(source["left"]), axis=1),
        ])
    return {"rms": float(np.sqrt(np.mean(errors ** 2)) * 1000.0),
            "max": float(errors.max() * 1000.0),
            "frames": int(len(original)),
            "points_per_frame": int(errors.shape[1])}


# The MJCF carries joint ranges but not rated speeds; the URDF beside it carries
# both. This is the vendor's own number and is read, never transcribed.
VELOCITY_URDF = "x2_ultra.urdf"


def vendor_velocity_limits(urdf_dir: Path | str | None = None) -> np.ndarray:
    """[14] rated joint speed in rad/s, in arm_pos order, from the vendor URDF.

    This is the MOTOR CEILING (about 860 deg/s), not a safe operating speed. It
    is the number a trajectory must not exceed even in principle; a first
    hardware run belongs one to two orders of magnitude below it. `safety.py`
    reports against a fraction of this for exactly that reason.
    """
    from .paths import find_urdf_dir

    path = find_urdf_dir(urdf_dir) / VELOCITY_URDF
    if not path.is_file():
        raise FileNotFoundError(f"vendor URDF not found: {path}")
    root = ET.parse(path).getroot()
    rated: dict[str, float] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name in ARM_JOINT_NAMES and limit is not None and limit.get("velocity"):
            rated[name] = float(limit.get("velocity"))
    missing = [name for name in ARM_JOINT_NAMES if name not in rated]
    if missing:
        raise ValueError(f"{path.name} has no velocity limit for {missing}")
    return np.array([rated[name] for name in ARM_JOINT_NAMES], dtype=np.float64)
