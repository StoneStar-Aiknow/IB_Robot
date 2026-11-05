"""X2 semantic targets to validated joint poses and paths."""

from __future__ import annotations

import numpy as np

from aimdk_robot.pose_backend import kin, planner, safety, solver
from aimdk_robot.pose_backend.model import ARM_JOINTS, SIDES, load
from robot_runtime.pose_backend import Composition, SpecError
from robot_runtime.task_poses import task_pose_arm_targets

LANDMARKS = {
    "head_top": ("头顶", (0.0, 0.0, 0.08)),
    "forehead": ("额头", (0.08, 0.0, 0.0)),
    "mouth": ("嘴", (0.08, 0.0, 0.0)),
    "chin": ("下巴", (0.07, 0.0, -0.03)),
    "left_ear": ("左耳", (0.0, 0.08, 0.0)),
    "right_ear": ("右耳", (0.0, -0.08, 0.0)),
    "back_of_head": ("后脑勺", (-0.08, 0.0, 0.0)),
    "chest": ("胸口", (0.07, 0.0, 0.0)),
    "front_of_chest": ("胸前正前方约 20 厘米处", (0.20, -0.02, 0.0)),
    "belly": ("肚子", (0.08, 0.0, 0.0)),
    "waist_left": ("左腰侧（叉腰的位置）", (0.0, 0.07, 0.0)),
    "waist_right": ("右腰侧（叉腰的位置）", (0.0, -0.07, 0.0)),
    "left_shoulder": ("左肩", (0.0, 0.0, 0.07)),
    "right_shoulder": ("右肩", (0.0, 0.0, 0.07)),
}
ELBOW_HINTS = {"out": (0, 1, 0), "down": (0, 0, -1), "forward": (1, 0, 0), "up": (0, 0, 1), "back": (-1, 0, 0)}


def describe(q: dict[str, float]) -> dict:
    """The current pose in the vocabulary of the prompt."""
    pose = kin.fk(q)
    return {
        side: {
            "upper": np.round(pose.upper(side), 2).tolist(),
            "fore": np.round(pose.fore(side), 2).tolist(),
            "elbow_bend_deg": round(pose.flex_deg(side)),
        }
        for side in SIDES
    }


def _mirror(v, side):
    v = np.asarray(v, float)
    return v if side == "left" else v * np.array([1, -1, 1])


def _vector(value, name: str) -> np.ndarray:
    v = np.asarray(value, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-6:
        raise SpecError(f"{name} must be a non-zero 3-vector")
    return v


def compose(data: dict, start_q: dict[str, float]) -> Composition:
    """Joint targets for a parsed description, each projected into the safe set."""
    out = Composition(
        str(data["kind"]), user_message=str(data.get("user_message", "")), reason=str(data.get("reason", ""))
    )
    if out.kind in ("rejected", "no_motion"):
        return out
    specs = list(data["frames"]) if out.kind == "sequence" else [data]
    if not specs or len(specs) > 4:
        raise SpecError("a sequence needs 1 to 4 frames")
    points = kin.fk({}).points
    shoulders = solver.shoulder_points()
    prev = dict(start_q)
    wrist_neutral = not bool(data.get("depends_on_current", False))
    for k, spec in enumerate(specs):
        q = dict(prev)
        # Resolve all named arms before IK, independent of left/right iteration
        # order. A task-pose arm is an exact fixed target, never keep-current.
        try:
            anchored, _ = task_pose_arm_targets(spec, load().task_poses)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
        q.update(anchored)
        for side in SIDES:
            arm = spec.get(side) or {"keep": True}
            if not isinstance(arm, dict) or arm.get("keep", False) or "task_pose" in arm:
                continue
            if "hand_to" in arm:
                name = str(arm["hand_to"])
                if name not in LANDMARKS:
                    raise SpecError(f"unknown body point {name!r}")
                offset = np.asarray(LANDMARKS[name][1], float)
                anchor = points["chest" if name == "front_of_chest" else name]
                if name == "front_of_chest":
                    offset = _mirror(offset, side)
                extra = _vector(arm["offset_cm"], "offset_cm") / 100.0 if arm.get("offset_cm") else np.zeros(3)
                if np.linalg.norm(extra) > 0.3:
                    raise SpecError("offset_cm is limited to 30 cm")
                hint = arm.get("elbow")
                hint_vec = _mirror(ELBOW_HINTS[hint], side) if hint in ELBOW_HINTS else None
                sol = solver.solve_hand(
                    side, anchor + offset + extra - shoulders[side], hint_vec, q, wrist_neutral=wrist_neutral
                )
                out.residual[f"{k}.{side}"] = {n: round(v, 3) for n, v in sol.residual_m.items()}
            elif "upper" in arm and "fore" in arm:
                sol = solver.solve_dirs(
                    side,
                    _vector(arm["upper"], "upper"),
                    _vector(arm["fore"], "fore"),
                    q,
                    wrist_neutral=wrist_neutral,
                )
                out.residual[f"{k}.{side}"] = {n: round(v, 1) for n, v in sol.residual_deg.items()}
            else:
                raise SpecError(f"{side}: give upper+fore, hand_to, or keep")
            q.update(sol.q)
            out.approx |= sol.approx
        q, _ = solver.project_safe(q)
        # Projection may perturb other joints to improve clearance. Named
        # anchors must remain exact; an incompatible active target is refused.
        q.update(anchored)
        if not safety.check_pose(q).ok:
            raise SpecError(f"frame {k} has no safe pose near the description")
        out.frames.append({n: float(q[n]) for n in ARM_JOINTS})
        out.holds.append(min(max(float(spec.get("hold_s", 1.0)), 0.0), 5.0))
        prev = q
    if out.approx:
        # Say so rather than let the robot claim a pose it could not reach.
        out.user_message = (
            out.user_message + " " if out.user_message else ""
        ) + "（有的位置够不到，我做到了最接近的姿势）"
    return out


def moved_arms(data: dict) -> list[str]:
    """The arms a single-pose description moves (the others keep where they are)."""
    if data.get("kind") != "pose":
        return []
    return [side for side in SIDES if isinstance(data.get(side), dict) and not data[side].get("keep", False)]


def targets_of(frame: dict[str, float], arms) -> dict[str, float]:
    return {n: float(frame[n]) for n in ARM_JOINTS if any(n.startswith(f"{side}_") for side in arms)}


def signature_of(frame: dict[str, float], arms) -> dict[str, list[float]]:
    """Where the moved arms are: elbow, wrist and hand relative to each shoulder (body frame, m)."""
    points = kin.fk(frame).points
    return {
        f"{side}_{part}": np.round(points[f"{side}_{part}"] - points[f"{side}_shoulder"], 4).tolist()
        for side in arms
        for part in ("elbow", "wrist", "hand")
    }


def recall(targets: dict[str, float], user_message: str, start_q: dict[str, float]) -> Composition:
    """A remembered pose from the current one: the moved arms go to their targets, the others stay."""
    q = dict(start_q)
    q.update(targets)
    q, _ = solver.project_safe(q)
    if not safety.check_pose(q).ok:
        raise SpecError("the remembered pose has no safe pose near it from here")
    return Composition("pose", [{n: float(q[n]) for n in ARM_JOINTS}], [1.0], user_message=user_message)


def resolve(description: dict) -> tuple[list[str], dict[str, float], dict[str, list[float]]] | None:
    """Solve a stored description again from the rest pose (no model call), or None if it no longer works."""
    from aimdk_robot.pose_backend.model import load

    try:
        composition = compose(description, dict(load().rest))
    except (SpecError, ValueError, KeyError, TypeError):
        return None
    arms = moved_arms(description)
    if composition.kind != "pose" or composition.approx or not arms:
        return None
    frame = composition.frames[-1]
    return arms, targets_of(frame, arms), signature_of(frame, arms)


def plan(composition: Composition, start_q: dict[str, float]) -> planner.Plan:
    full = [dict(start_q, **f) for f in composition.frames]
    return planner.plan_safe(dict(start_q), full, composition.holds)
