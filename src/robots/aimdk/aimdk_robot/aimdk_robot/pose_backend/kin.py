"""Forward kinematics, body keypoints and arm-segment directions in the body frame.

Body frame = world frame with the pelvis welded: x forward, y left, z up (metres).
Keypoints that are not joint origins (head top, chin, chest, waist sides, hand
centre) are taken from the official meshes once, at the zero pose, and then carried
by their bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import mujoco
import numpy as np

from aimdk_robot.pose_backend.model import SIDES, Robot, load, to_qpos


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def angle_deg(a, b) -> float:
    a, b = _unit(a), _unit(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _body_vertices(robot: Robot, data: mujoco.MjData, body: str, group: int = 1) -> np.ndarray:
    m = robot.model
    bid = m.body(body).id
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != group:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        out.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(out) if out else np.zeros((0, 3))


@dataclass(frozen=True)
class Anchor:
    body: str
    local: tuple[float, float, float]


@cache
def anchors() -> dict[str, Anchor]:
    robot = load()
    m = robot.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def anchor(body, world):
        bid = m.body(body).id
        local = d.xmat[bid].reshape(3, 3).T @ (np.asarray(world) - d.xpos[bid])
        return Anchor(body, tuple(float(x) for x in local))

    head = _body_vertices(robot, d, "head_pitch_link")
    z0, z1 = head[:, 2].min(), head[:, 2].max()
    top = head[np.argmax(head[:, 2])]
    upper = head[head[:, 2] > z0 + 0.6 * (z1 - z0)]
    lower = head[head[:, 2] < z0 + 0.25 * (z1 - z0)]
    forehead = upper[np.argmax(upper[:, 0])]
    chin = lower[np.argmax(lower[:, 0])]
    torso = _body_vertices(robot, d, "torso_link")
    shoulder_z = d.xpos[m.body("left_shoulder_roll_link").id][2]
    band = torso[(np.abs(torso[:, 1]) < 0.03) & (torso[:, 2] > shoulder_z - 0.2) & (torso[:, 2] < shoulder_z - 0.1)]
    chest = band[np.argmax(band[:, 0])]
    waist_band = np.concatenate([torso, _body_vertices(robot, d, "pelvis")])
    waist_band = waist_band[(waist_band[:, 2] > 0.80) & (waist_band[:, 2] < 0.90)]
    mid = head[(head[:, 2] > z0 + 0.35 * (z1 - z0)) & (head[:, 2] < z0 + 0.65 * (z1 - z0))]
    mouth_band = head[(head[:, 2] > z0 + 0.2 * (z1 - z0)) & (head[:, 2] < z0 + 0.4 * (z1 - z0))]
    belly_band = torso[(np.abs(torso[:, 1]) < 0.03) & (torso[:, 2] > 0.86) & (torso[:, 2] < 0.92)]
    out = {
        "left_ear": anchor("head_pitch_link", mid[np.argmax(mid[:, 1])]),
        "right_ear": anchor("head_pitch_link", mid[np.argmin(mid[:, 1])]),
        "back_of_head": anchor("head_pitch_link", mid[np.argmin(mid[:, 0])]),
        "mouth": anchor("head_pitch_link", mouth_band[np.argmax(mouth_band[:, 0])]),
        "belly": anchor("torso_link", belly_band[np.argmax(belly_band[:, 0])]),
        "head_top": anchor("head_pitch_link", top),
        "forehead": anchor("head_pitch_link", forehead),
        "chin": anchor("head_pitch_link", chin),
        "chest": anchor("torso_link", chest),
        "waist_left": anchor("torso_link", waist_band[np.argmax(waist_band[:, 1])]),
        "waist_right": anchor("torso_link", waist_band[np.argmin(waist_band[:, 1])]),
    }
    for side in SIDES:
        hand = _body_vertices(robot, d, f"{side}_wrist_roll_link")
        out[f"{side}_hand"] = anchor(f"{side}_wrist_roll_link", hand.mean(axis=0))
        for name, body in (
            ("shoulder", f"{side}_shoulder_roll_link"),
            ("elbow", f"{side}_elbow_link"),
            ("wrist", f"{side}_wrist_pitch_link"),
        ):
            out[f"{side}_{name}"] = Anchor(body, (0.0, 0.0, 0.0))
    return out


@dataclass
class Pose:
    q: dict[str, float]
    points: dict[str, np.ndarray]
    data: mujoco.MjData

    def upper(self, side: str) -> np.ndarray:
        return _unit(self.points[f"{side}_elbow"] - self.points[f"{side}_shoulder"])

    def fore(self, side: str) -> np.ndarray:
        return _unit(self.points[f"{side}_wrist"] - self.points[f"{side}_elbow"])

    def flex_deg(self, side: str) -> float:
        return angle_deg(self.upper(side), self.fore(side))

    def describe(self) -> dict:
        """The pose in the same body-frame vocabulary the planner prompts use."""
        out = {}
        for side in SIDES:
            sh = self.points[f"{side}_shoulder"]
            out[side] = {
                "upper": np.round(self.upper(side), 2).tolist(),
                "fore": np.round(self.fore(side), 2).tolist(),
                "elbow_flex_deg": round(self.flex_deg(side)),
                "wrist_rel_shoulder_m": np.round(self.points[f"{side}_wrist"] - sh, 2).tolist(),
            }
        return out


def fk(q: dict[str, float], robot: Robot | None = None, data: mujoco.MjData | None = None) -> Pose:
    robot = robot or load()
    data = data or robot.new_data()
    full = dict(robot.rest)
    full.update(q)
    data.qpos[:] = to_qpos(robot, full)
    mujoco.mj_forward(robot.model, data)
    points = {}
    for name, a in anchors().items():
        bid = robot.model.body(a.body).id
        points[name] = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array(a.local)
    return Pose(full, points, data)


@cache
def arm_lengths() -> dict[str, float]:
    pose = fk({})
    zero = fk({n: 0.0 for n in load().arm_joints})
    p = zero.points
    return {
        "upper": float(np.linalg.norm(p["left_elbow"] - p["left_shoulder"])),
        "fore": float(np.linalg.norm(p["left_wrist"] - p["left_elbow"])),
        "rest_flex_deg": pose.flex_deg("left"),
    }
