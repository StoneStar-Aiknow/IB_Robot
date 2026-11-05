"""The pose description a language model writes, and how it becomes joint targets.

Per arm the model chooses one of two forms (selected in simulation, round 2a/2c):

* directions — ``upper`` (shoulder -> elbow) and ``fore`` (elbow -> wrist) unit
  vectors in the body frame, for reaching, raising, pointing, spreading;
* a body point — ``hand_to`` one of ``LANDMARKS`` with an optional
  ``offset_cm`` and ``elbow`` hint, for poses whose meaning is where the hand
  is on the body (hands on hips, hand on the head, at the chin, on the ears).

The model never writes joint angles: the solver owns the kinematics, the joint
travel and the clearance from the body, and every result is checked again by
``safety`` and ``planner`` before anything is sent to the robot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from pose_composer import kin, planner, safety, solver
from pose_composer.model import ARM_JOINTS, SIDES

KINDS = ("pose", "sequence", "rejected", "no_motion")
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
PROMPT_VERSION = "pose-hybrid-v3"

SYSTEM_PROMPT = (
    "你是人形机器人 X2 的上肢姿态规划器。用户用自然语言让机器人摆手臂姿势，你只控制左右两条手臂"
    "（不控制手指、头、腰和腿）。只输出一个 JSON 对象，不要 markdown，不要解释。\n"
    "kind 取值：pose（摆出一个姿势）；sequence（用户明确要求依次做几个姿势时，按顺序给出多帧）；"
    "rejected（会打到或撞到自己、超出人体手臂能力、可能伤人时拒绝，给出 reason；"
    "这时 user_message 只说明不做的原因，不要承诺改做别的动作，机器人不会动）；"
    "no_motion（否定句、假设或提问、闲聊，不动）。\n"
    "每个回答都带 user_message：动作开始前播报给用户听的一句中文口语，不超过 30 个字，"
    "说要做什么（如「好的，双手向前伸直」），不要说已经做完。\n"
    "没被要求动的手臂保持当前姿态（keep=true）。相对指令（「再高一点」「左手也这样」）以当前姿态和对话历史为准。\n"
    "坐标系：以机器人自身为准，x 指向机器人正前方，y 指向机器人的左边，z 竖直向上。"
    "机器人的左手在 +y 一侧，右手在 −y 一侧。\n"
    "每条手臂二选一描述：\n"
    "1) 方向：upper = 肩→肘 的单位方向，fore = 肘→腕 的单位方向。适合伸、举、指、张开这类动作。"
    "参考：下垂 upper≈[0,0,-1]；向前平伸 upper=fore=[1,0,0]；向上举直 [0,0,1]；左臂侧平举 [0,1,0]，右臂 [0,-1,0]；"
    "肘弯 90° 时 upper 与 fore 垂直。\n"
    "2) 手到身体部位：hand_to = 下面的部位名，手会停在该部位外侧几厘米处（不会真的碰上）；"
    "可选 offset_cm = [前,左,上] 厘米微调；可选 elbow = out/down/forward/up/back 表示肘尖朝向。"
    "适合摸头、叉腰、托下巴、捂耳朵、手放胸口/肩上、合十这类手要放到身体某处的动作。\n"
    f"部位名：{json.dumps({k: v[0] for k, v in LANDMARKS.items()}, ensure_ascii=False)}\n"
    "肘只能像人一样朝一个方向弯，手臂不能穿过身体或头部，肩部向身体内侧收拢的幅度很小。\n"
    '输出格式：{"kind":"pose","left":{"upper":[x,y,z],"fore":[x,y,z]},'
    '"right":{"hand_to":"head_top","elbow":"out"},"user_message":"..."}；不动的手臂写 {"keep":true}；'
    'sequence 时：{"kind":"sequence","frames":[{"left":{...},"right":{...},"hold_s":1.0},...],"user_message":"..."}；'
    'rejected / no_motion 时：{"kind":"rejected","reason":"...","user_message":"..."}。\n'
)


class SpecError(ValueError):
    """A pose description that must not be executed."""


@dataclass
class Composition:
    kind: str
    frames: list[dict[str, float]] = field(default_factory=list)
    holds: list[float] = field(default_factory=list)
    user_message: str = ""
    reason: str = ""
    approx: bool = False
    residual: dict = field(default_factory=dict)


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


def build_messages(text: str, current_q: dict[str, float], history: list[tuple[str, dict]] | None = None) -> list[dict]:
    lines = []
    if history:
        lines.append("对话历史（从早到晚）：")
        for said, q in history:
            lines.append(f"- 用户说「{said}」，之后的姿态：{json.dumps(describe(q), ensure_ascii=False)}")
    lines.append(f"当前姿态：{json.dumps(describe(current_q), ensure_ascii=False)}")
    lines.append(f"用户指令：「{text}」")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def parse(reply: str) -> dict:
    """The single JSON object of a reply; a fenced block is tolerated, nothing else."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", reply, re.S)
    text = fenced.group(1) if fenced else reply
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise SpecError("no JSON object in the reply")
    data = json.loads(text[start : end + 1])
    if data.get("kind") not in KINDS:
        raise SpecError(f"unknown kind {data.get('kind')!r}")
    return data


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
    for k, spec in enumerate(specs):
        q = dict(prev)
        for side in SIDES:
            arm = spec.get(side) or {"keep": True}
            if not isinstance(arm, dict) or arm.get("keep", False):
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
                sol = solver.solve_hand(side, anchor + offset + extra - shoulders[side], hint_vec, q)
                out.residual[f"{k}.{side}"] = {n: round(v, 3) for n, v in sol.residual_m.items()}
            elif "upper" in arm and "fore" in arm:
                sol = solver.solve_dirs(side, _vector(arm["upper"], "upper"), _vector(arm["fore"], "fore"), q)
                out.residual[f"{k}.{side}"] = {n: round(v, 1) for n, v in sol.residual_deg.items()}
            else:
                raise SpecError(f"{side}: give upper+fore, hand_to, or keep")
            q.update(sol.q)
            out.approx |= sol.approx
        q, _ = solver.project_safe(q)
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


def plan(composition: Composition, start_q: dict[str, float]) -> planner.Plan:
    full = [dict(start_q, **f) for f in composition.frames]
    return planner.plan_safe(dict(start_q), full, composition.holds)
