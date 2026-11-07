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

from aimdk_robot.pose_backend.composition import (
    LANDMARKS,
    Composition,
    SpecError,
    compose,
    describe,
    moved_arms,
    plan,
    recall,
    resolve,
    signature_of,
    targets_of,
)

__all__ = [
    "LANDMARKS",
    "Composition",
    "SpecError",
    "compose",
    "describe",
    "moved_arms",
    "plan",
    "recall",
    "resolve",
    "signature_of",
    "targets_of",
    "build_messages",
    "parse",
    "memorable",
]

KINDS = ("pose", "sequence", "rejected", "no_motion")

PROMPT_VERSION = "pose-hybrid-v5"

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
    '姿势如果取决于当前姿态或之前的对话（相对指令、「回到刚才」「另一只手也这样」），加 "depends_on_current": true；'
    "只是让另一只手保持不动（keep）不算。\n"
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


def memorable(data: dict, composition: Composition, text: str, *, allow_approximate: bool = False) -> bool:
    """Whether a composed pose may be remembered and recalled for the same words later."""
    from pose_composer.memory import is_relative

    return (
        composition.kind == "pose"
        and (allow_approximate or not composition.approx)
        and not data.get("depends_on_current", False)
        and not is_relative(text)
        and bool(moved_arms(data))
    )
