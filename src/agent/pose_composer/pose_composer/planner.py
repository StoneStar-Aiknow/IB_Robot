"""Collision-free joint paths: the straight path when it is clear, else through a via pose.

The straight minimum-jerk path between two safe poses can still sweep a hand
through the body (e.g. arms folded across the chest -> anything). Rather than a
general sampling planner, the arm is routed through a small set of "ready" via
poses with the hands held forward and away from the trunk, tried in order; the
first whose two segments are both clear is used. If none is, the motion is
refused rather than executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from pose_composer import kin, safety, solver, trajectory
from pose_composer.model import ARM_JOINTS, load

PATH_SAMPLES = 30


@dataclass
class Plan:
    traj: trajectory.Trajectory
    ok: bool
    via: list[str]
    report: safety.Report


@cache
def _ready_poses() -> dict[str, dict[str, float]]:
    rest = load().rest
    out = {}
    for name, upper, fore in (
        ("ready_forward", [0.6, 0.35, -0.7], [1.0, 0.0, 0.0]),
        ("ready_out", [0.2, 0.6, -0.75], [0.9, 0.3, 0.2]),
        ("ready_high", [0.7, 0.4, 0.0], [0.6, 0.0, 0.8]),
    ):
        q = dict(rest)
        for side in ("left", "right"):
            m = (lambda v: v) if side == "left" else (lambda v: [v[0], -v[1], v[2]])
            q.update(solver.solve_dirs(side, m(upper), m(fore), q).q)
        out[name] = q
    return out


def _segment_ok(a: dict, b: dict) -> bool:
    traj = trajectory.plan(a, [b], [0.0])
    return safety.check_path(traj.times, traj.positions, ARM_JOINTS, samples=PATH_SAMPLES).ok


def _blend_via(start: dict, goal: dict, ready: dict) -> dict:
    """Keep an arm that does not move where it is; only the moving arms go through the via."""
    via = dict(ready)
    for side in ("left", "right"):
        names = [n for n in ARM_JOINTS if n.startswith(side)]
        if max(abs(start[n] - goal[n]) for n in names) < 0.05:
            via.update({n: start[n] for n in names})
    return via


OFFSETS = ((0.12, 0.03, 0.02), (0.15, 0.12, 0.12), (0.05, 0.20, 0.10))


def pulled_forward(q: dict, offset=OFFSETS[0]) -> dict:
    """The same pose with each hand drawn away from the trunk by ``offset`` (left-arm frame)."""
    pose = kin.fk(q)
    out = dict(q)
    shoulders = solver.shoulder_points()
    for side in ("left", "right"):
        hand = pose.points[f"{side}_hand"]
        o = np.array(offset) * (np.array([1, 1, 1]) if side == "left" else np.array([1, -1, 1]))
        out.update(solver.solve_hand(side, hand + o - shoulders[side], None, out).q)
    return out


def _candidates(prev: dict, frame: dict):
    """Via sequences to try, cheapest first."""
    for k, off in enumerate(OFFSETS):
        yield f"approach{k}", [pulled_forward(frame, off)]
        yield f"depart{k}", [pulled_forward(prev, off)]
        yield f"depart+approach{k}", [pulled_forward(prev, off), pulled_forward(frame, off)]
    for name, ready in _ready_poses().items():
        yield name, [_blend_via(prev, frame, ready)]


def plan_safe(start: dict, frames: list[dict], holds: list[float]) -> Plan:
    vias: list[list[dict]] = []
    used: list[str] = []
    prev = start
    for frame in frames:
        if _segment_ok(prev, frame):
            vias.append([])
        else:
            for name, seq in _candidates(prev, frame):
                points = [prev, *seq, frame]
                if all(safety.check_pose(v).ok for v in seq) and all(
                    _segment_ok(a, b) for a, b in zip(points, points[1:], strict=False)
                ):
                    vias.append(seq)
                    used.append(name)
                    break
            else:
                traj = trajectory.plan(start, frames, holds)
                rep = safety.check_path(traj.times, traj.positions, ARM_JOINTS, samples=PATH_SAMPLES)
                rep.notes.append("no clear path through any via pose")
                return Plan(traj, False, used, rep)
        prev = frame
    traj = trajectory.plan(start, frames, holds, via=vias)
    rep = safety.check_path(traj.times, traj.positions, ARM_JOINTS, samples=PATH_SAMPLES * (1 + len(used)))
    return Plan(traj, rep.ok, used, rep)
