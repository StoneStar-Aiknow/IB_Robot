"""Turn a body-frame description of an arm (segment directions or keypoint positions)
into joint angles, within the safe limits, close to the current pose, clear of the body.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import mujoco
import numpy as np
from scipy.optimize import minimize

from pose_composer import kin, safety
from pose_composer.model import ARM_SUFFIXES, load, to_qpos

SOLVED = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
APPROX_DEG = 15.0
REG = 0.002
SEEDS = (
    (0.0, 0.0, 0.0, -0.05),
    (-1.5, 0.0, 0.0, -0.05),
    (-2.8, 0.0, 0.0, -0.05),
    (0.0, 1.5, 0.0, -0.05),
    (-1.0, 0.3, 0.0, -1.6),
)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero-length direction")
    return v / n


@dataclass
class ArmSolution:
    q: dict[str, float]
    cost: float
    residual_deg: dict[str, float]
    residual_m: dict[str, float]
    approx: bool


class _Arm:
    """Fast FK for one arm: only mj_kinematics, on a private MjData."""

    def __init__(self, side: str, base_q: dict[str, float]):
        self.robot = load()
        self.side = side
        self.names = [f"{side}_{s}_joint" for s in SOLVED]
        self.data = self.robot.new_data()
        self.base = dict(base_q)
        self.data.qpos[:] = to_qpos(self.robot, self.base)
        self.adr = [self.robot.qadr[n] for n in self.names]
        m = self.robot.model
        self.b_sh = m.body(f"{side}_shoulder_roll_link").id
        self.b_el = m.body(f"{side}_elbow_link").id
        self.b_wr = m.body(f"{side}_wrist_pitch_link").id
        self.b_hand = m.body(f"{side}_wrist_roll_link").id
        self.hand_local = np.array(kin.anchors()[f"{side}_hand"].local)
        self.bounds = [self.robot.safe_limits[n] for n in self.names]

    def points(self, x):
        self.data.qpos[self.adr] = x
        mujoco.mj_kinematics(self.robot.model, self.data)
        return self.data.xpos[self.b_sh].copy(), self.data.xpos[self.b_el].copy(), self.data.xpos[self.b_wr].copy()

    def hand(self):
        """Hand centre for the pose of the last ``points`` call."""
        return self.data.xpos[self.b_hand] + self.data.xmat[self.b_hand].reshape(3, 3) @ self.hand_local


def _full_q(base: dict[str, float], arm: _Arm, x) -> dict[str, float]:
    q = dict(base)
    q.update({n: float(v) for n, v in zip(arm.names, x, strict=True)})
    return q


def _clearance_penalty(q: dict[str, float], side: str) -> float:
    pose = kin.fk(q)
    m = load().model
    fromto = np.zeros(6)
    pen = 0.0
    for g, h, need, label in safety.pairs():
        if not label.startswith(side) and f"~{side}_" not in label:
            continue
        if safety.far(m, pose.data, g, h, need + 0.02):
            continue
        d = mujoco.mj_geomDistance(m, pose.data, g, h, need + 0.02, fromto)
        if d < need + 0.01:
            pen += (need + 0.01 - d) ** 2
    return pen


def _optimize(arm: _Arm, base: dict[str, float], err, current_x, with_collision: bool):
    def cost(x):
        c = err(x) + REG * float(np.sum((np.asarray(x) - current_x) ** 2))
        if with_collision:
            c += 2000.0 * _clearance_penalty(_full_q(base, arm, x), arm.side)
        return c

    best = None
    for seed in (tuple(current_x),) + SEEDS:
        x0 = np.clip(seed, [b[0] for b in arm.bounds], [b[1] for b in arm.bounds])
        res = minimize(cost, x0, bounds=arm.bounds, method="L-BFGS-B", options={"maxiter": 200})
        if (
            best is None
            or res.fun < best.fun - 1e-6
            or (
                abs(res.fun - best.fun) <= 1e-6 and np.sum((res.x - current_x) ** 2) < np.sum((best.x - current_x) ** 2)
            )
        ):
            best = res
    return best


def solve_dirs(side: str, upper, fore, base_q: dict[str, float], wrist_neutral: bool = True) -> ArmSolution:
    """Match shoulder->elbow and elbow->wrist directions."""
    u_t, f_t = _unit(upper), _unit(fore)
    return _solve(side, base_q, ("dirs", u_t, f_t), wrist_neutral)


def solve_points(side: str, elbow, wrist, base_q: dict[str, float], wrist_neutral: bool = True) -> ArmSolution:
    """Match elbow and wrist positions given relative to the shoulder (metres)."""
    return _solve(side, base_q, ("points", np.asarray(elbow, float), np.asarray(wrist, float)), wrist_neutral)


def solve_hand(side: str, hand, elbow_hint, base_q: dict[str, float], wrist_neutral: bool = True) -> ArmSolution:
    """Put the hand centre at ``hand`` (relative to the shoulder, metres); ``elbow_hint`` is an
    optional direction the upper arm should lean towards (shoulder -> elbow)."""
    hint = None if elbow_hint is None else _unit(elbow_hint)
    return _solve(side, base_q, ("hand", np.asarray(hand, float), hint), wrist_neutral)


def _solve(side, base_q, target, wrist_neutral) -> ArmSolution:
    base = dict(load().rest)
    base.update(base_q)
    if wrist_neutral:
        for s in ARM_SUFFIXES[4:]:
            base[f"{side}_{s}_joint"] = 0.0
    arm = _Arm(side, base)
    current_x = np.array([base[n] for n in arm.names])
    kind, a, b = target

    def err(x):
        sh, el, wr = arm.points(x)
        if kind == "hand":
            c = float(np.sum((arm.hand() - sh - a) ** 2) / 0.01)
            if b is not None:
                c += 0.3 * float(1 - ((el - sh) / np.linalg.norm(el - sh)) @ b)
            return c
        if kind == "dirs":
            u = (el - sh) / np.linalg.norm(el - sh)
            f = (wr - el) / np.linalg.norm(wr - el)
            return float((1 - u @ a) + (1 - f @ b))
        # The elbow is a hint (many elbow positions reach one wrist point); the wrist is the target.
        return float(0.3 * np.sum((el - sh - a) ** 2) / 0.04 + np.sum((wr - sh - b) ** 2) / 0.04)

    res = _optimize(arm, base, err, current_x, with_collision=False)
    q = _full_q(base, arm, res.x)
    if _clearance_penalty(q, side) > 0:
        res = _optimize(arm, base, err, current_x, with_collision=True)
        q = _full_q(base, arm, res.x)
    sh, el, wr = arm.points(res.x)
    if kind == "hand":
        r_m = {"hand": float(np.linalg.norm(arm.hand() - sh - a))}
        r_deg = {}
        approx = r_m["hand"] > 0.06
    elif kind == "dirs":
        u = (el - sh) / np.linalg.norm(el - sh)
        f = (wr - el) / np.linalg.norm(wr - el)
        r_deg = {"upper": kin.angle_deg(u, a), "fore": kin.angle_deg(f, b)}
        r_m = {}
        approx = max(r_deg.values()) > APPROX_DEG
    else:
        r_m = {"elbow": float(np.linalg.norm(el - sh - a)), "wrist": float(np.linalg.norm(wr - sh - b))}
        r_deg = {}
        approx = r_m["wrist"] > 0.06
    return ArmSolution({n: q[n] for n in q if n.startswith(side)}, float(res.fun), r_deg, r_m, approx)


@cache
def shoulder_points() -> dict[str, np.ndarray]:
    p = kin.fk({}).points
    return {"left": p["left_shoulder"], "right": p["right_shoulder"]}


def project_safe(q: dict[str, float]) -> tuple[dict[str, float], float]:
    """Nearest pose (in the shoulder and elbow joints) inside the limits and clear of the body.

    The method-independent fallback: whatever produced ``q``, this is what would run.
    Returns the pose and the largest joint change it needed (rad).
    """
    robot = load()
    names = [f"{side}_{s}_joint" for side in ("left", "right") for s in SOLVED]
    base = dict(robot.rest)
    base.update(q)
    for name, (lo, hi) in robot.safe_limits.items():
        base[name] = float(np.clip(base[name], lo, hi))
    x_prop = np.array([base[n] for n in names])
    bounds = [robot.safe_limits[n] for n in names]

    def full(x):
        out = dict(base)
        out.update(dict(zip(names, x, strict=True)))
        return out

    def penalty(x):
        qq = full(x)
        return _clearance_penalty(qq, "left") + _clearance_penalty(qq, "right")

    if penalty(x_prop) == 0:
        return base, float(max(abs(base[n] - q.get(n, base[n])) for n in robot.safe_limits))
    res = minimize(lambda x: float(np.sum((x - x_prop) ** 2)) + 5000.0 * penalty(x), x_prop, bounds=bounds,
                   method="L-BFGS-B", options={"maxiter": 300})  # fmt: skip
    out = full(res.x)
    return out, float(max(abs(out[n] - q.get(n, out[n])) for n in robot.safe_limits))
