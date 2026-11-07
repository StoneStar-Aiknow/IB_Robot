"""Safety checks every candidate pose and path goes through, whatever produced it."""

from __future__ import annotations

from functools import cache

import mujoco
import numpy as np

from aimdk_robot.pose_backend import kin
from aimdk_robot.pose_backend.model import SIDES, Robot, load
from robot_runtime.pose_backend import Report

CLEARANCE_M = 0.02
HAND_CLEARANCE_M = 0.03
# The shoulder yaw link sits 3.7 cm from the torso at rest by construction.
SHOULDER_TORSO_CLEARANCE_M = 0.01
MAX_SPEED_RADPS = 1.0
GRAVITY_TORQUE_SHARE = 0.30


def arm_efforts() -> dict[str, float]:
    """Rated joint torque from the official URDF, not the MJCF's +/-120 control range."""
    from aimdk_robot.pose_backend.model import ARM_JOINTS, EFFORT_NM

    return {name: EFFORT_NM[name.split("_", 1)[1].removesuffix("_joint")] for name in ARM_JOINTS}


def _col_geoms(m, body: str) -> list[int]:
    bid = m.body(body).id
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_group[g] == 3 and m.geom_contype[g]]


@cache
def pairs() -> list[tuple[int, int, float, str]]:
    """(geom, geom, required clearance, label) for every arm pair worth checking."""
    m = load().model
    trunk = ["pelvis", "torso_link", "head_pitch_link"] + [
        f"{s}_{p}_link" for s in SIDES for p in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ]
    moving = {
        side: [f"{side}_{p}_link" for p in ("shoulder_yaw", "elbow", "wrist_pitch", "wrist_roll")] for side in SIDES
    }
    out = []
    for side in SIDES:
        for arm_body in moving[side]:
            for other in trunk:
                need = CLEARANCE_M
                if arm_body.endswith("wrist_roll_link"):
                    need = HAND_CLEARANCE_M
                if arm_body.endswith("shoulder_yaw_link") and other == "torso_link":
                    need = SHOULDER_TORSO_CLEARANCE_M
                for g in _col_geoms(m, arm_body):
                    for h in _col_geoms(m, other):
                        out.append((g, h, need, f"{arm_body}~{other}"))
    for a in moving["left"]:
        for b in moving["right"]:
            need = HAND_CLEARANCE_M if "wrist_roll" in a or "wrist_roll" in b else CLEARANCE_M
            for g in _col_geoms(m, a):
                for h in _col_geoms(m, b):
                    out.append((g, h, need, f"{a}~{b}"))
    return out


def limit_violations(
    q: dict[str, float], robot: Robot | None = None, limits: dict[str, tuple[float, float]] | None = None
) -> list[str]:
    robot = robot or load()
    out = []
    for name, (lo, hi) in (limits or robot.safe_limits).items():
        v = q.get(name, robot.rest[name])
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            out.append(f"{name}={v:.3f} not in [{lo:.3f},{hi:.3f}]")
    return out


def far(m, data, g: int, h: int, margin: float) -> bool:
    """Broad phase: bounding spheres already further apart than the margin."""
    gap = np.linalg.norm(data.geom_xpos[g] - data.geom_xpos[h]) - m.geom_rbound[g] - m.geom_rbound[h]
    return gap > margin


# A path may pass this much closer than a pose may rest (hands >= 1.5 cm, arms >= 0.5 cm
# en route): a pose projected to sit on
# its clearance would otherwise make every path leaving it fail on its first sample.
PATH_SLACK_M = 0.015


def clearance(
    pose: kin.Pose, slack: float = 0.0, allowed: dict[str, float] | None = None
) -> tuple[float, list[tuple[str, float]]]:
    """Worst clearance, and the pairs closer than they may be.

    ``allowed`` maps a pair to a distance it may keep (a path leaving a pose that
    already sits closer than the rule, e.g. the measured start, may not get closer).
    """
    m = load().model
    fromto = np.zeros(6)
    worst = 1.0
    bad: dict[str, float] = {}
    for g, h, need, label in pairs():
        if far(m, pose.data, g, h, max(need, 0.05) + 0.01):
            continue
        d = mujoco.mj_geomDistance(m, pose.data, g, h, max(need, 0.05) + 0.01, fromto)
        worst = min(worst, d)
        floor = need - slack
        if allowed and label in allowed:
            floor = min(floor, allowed[label] - 0.005)
        if d < floor:
            bad[label] = min(bad.get(label, 1.0), d)
    return worst, sorted(bad.items(), key=lambda x: x[1])


def gravity_share(pose: kin.Pose) -> float:
    robot = load()
    m, d = robot.model, pose.data
    d.qvel[:] = 0
    # At zero velocity qfrc_bias is the static gravity torque. Inverse
    # dynamics also includes contact/constraint reactions and passive forces;
    # those are not gravity, and can dominate on a measured real-arm pose.
    mujoco.mj_fwdVelocity(m, d)
    share = 0.0
    for name in robot.arm_joints:
        effort = arm_efforts()[name]
        share = max(share, abs(float(d.qfrc_bias[m.joint(name).dofadr[0]])) / effort)
    return share


def check_pose(
    q: dict[str, float],
    slack: float = 0.0,
    allowed: dict[str, float] | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
) -> Report:
    if set(q) - set(load().arm_joints) or not all(np.isfinite(v) for v in q.values()):
        return Report(False, notes=["unknown joint or non-finite pose"])
    pose = kin.fk(q)
    limits = limit_violations(pose.q, limits=limits)
    worst, bad = clearance(pose, slack, allowed)
    share = gravity_share(pose)
    ok = not limits and not bad and share <= GRAVITY_TORQUE_SHARE
    return Report(ok, limits, bad, worst, share)


def check_path(times: np.ndarray, positions: np.ndarray, joints, samples: int = 50) -> Report:
    """Check a joint trajectory: every sampled point, plus speed.

    Pairs the start pose already holds closer than the rule may stay as close, never closer:
    the measured pose is where the robot is, not a target this path chose.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    if (
        times.ndim != 1
        or len(times) < 2
        or positions.shape != (len(times), len(joints))
        or len(set(joints)) != len(joints)
        or set(joints) != set(load().arm_joints)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(positions))
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        return Report(False, notes=["invalid trajectory shape, joints, timestamps or non-finite values"])
    start_q = dict(zip(joints, positions[0], strict=True))
    _, start_bad = clearance(kin.fk(start_q), PATH_SLACK_M)
    allowed = dict(start_bad)
    # Likewise a start outside the safe band may only move back towards it.
    ranges = {
        n: (min(lo, start_q.get(n, lo)), max(hi, start_q.get(n, hi))) for n, (lo, hi) in load().safe_limits.items()
    }
    robot = load()
    notes = []
    if len(times) > 1:
        speed = np.max(np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None])
        if speed > MAX_SPEED_RADPS + 1e-6:
            notes.append(f"peak speed {speed:.2f} rad/s > {MAX_SPEED_RADPS}")
    idx = np.unique(np.linspace(0, len(times) - 1, min(samples, len(times))).round().astype(int))
    worst_rep = None
    for i in idx:
        rep = check_pose(
            dict(zip(joints, positions[i], strict=True)), slack=PATH_SLACK_M, allowed=allowed, limits=ranges
        )
        if not rep.ok:
            rep.notes.append(f"at t={times[i]:.2f}s")
            worst_rep = rep
            break
    if worst_rep is None:
        worst_rep = Report(True)
    worst_rep.notes = notes + worst_rep.notes
    worst_rep.ok = worst_rep.ok and not notes
    del robot
    return worst_rep
