"""Key actions as a multiple-choice question, read from InternVL's first token.

Plan v11 §3, methods A and A1. The question lists the five key actions and a
"none of these" option, one letter each, and asks for a letter. Nothing is
generated: InternVlVision.score reads the probability of each letter as the
reply's first token after one prefill, so a look costs one vision OM run and
one prefill per question.

A 1B model prefers some positions over others, so each question exists in
two orders (``ORDERS``): the actions forward and reversed, "none" last in
both. Averaging an action's share over the two cancels most of a position
preference; the offline evaluation measures whether that is worth the second
prefill.

Pure Python: the prompts, and turning a ``Choice`` into scores per action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from visual_interaction.phase0.prompts import chat, frame_header
from visual_interaction.session.key_actions import KEY_ACTIONS

NONE = "none"
LETTERS = ("A", "B", "C", "D", "E", "F")

CHOICE_SYSTEM = "你是展厅机器人的眼睛，只回答一个字母。"
OPTION_TEXT = {
    "yeah": "剪刀手（伸出食指和中指比V）",
    "thumb": "点赞（竖起大拇指）",
    "wave": "挥手打招呼",
    "heart": "比心（用手指或双手比出爱心）",
    "open": "张开双臂",
    NONE: "都不是",
}
SINGLE_LEAD = "图里离镜头最近的人在做下面哪个动作？"
MULTI_LEAD = "这几张图是同一个人连续拍的，最后一张是现在。这个人现在在做下面哪个动作？"

# The order of the actions behind the letters; "none" takes the letter after them.
ORDERS: dict[str, tuple[str, ...]] = {
    "fwd": KEY_ACTIONS,
    "rev": tuple(reversed(KEY_ACTIONS)),
}


def options(order: Sequence[str]) -> tuple[str, ...]:
    """The answers behind the letters, in letter order: the actions, then none."""
    return (*order, NONE)


def prompt(frames: int, order: Sequence[str]) -> str:
    """The whole question for ``frames`` pictures, image slots unexpanded."""
    lines = [f"{letter}. {OPTION_TEXT[answer]}" for letter, answer in zip(LETTERS, options(order), strict=False)]
    lead = SINGLE_LEAD if frames == 1 else MULTI_LEAD
    return chat(frame_header(frames) + lead + "\n" + "\n".join(lines) + "\n只回答一个字母。", system=CHOICE_SYSTEM)


def key(frames: int, order: str) -> str:
    return f"key.{frames}.{order}"


def questions(frames: Sequence[int] = (1, 4), orders: Sequence[str] = tuple(ORDERS)) -> dict[str, tuple]:
    """``{key: (text, letters)}`` for InternVlVision.register_choices."""
    out = {}
    for count in frames:
        for name in orders:
            answers = options(ORDERS[name])
            out[key(count, name)] = (prompt(count, ORDERS[name]), LETTERS[: len(answers)])
    return out


def by_answer(probs: Mapping[str, float], order: str) -> dict[str, float]:
    """A Choice's letter probabilities, keyed by what each letter stands for."""
    return {answer: float(probs[letter]) for letter, answer in zip(LETTERS, options(ORDERS[order]), strict=False)}


def shares(probs: Mapping[str, float]) -> dict[str, float]:
    """Each answer's probability over all the answers' (they need not sum to one)."""
    total = sum(probs.values())
    return {answer: (p / total if total > 0 else 0.0) for answer, p in probs.items()}


def average(*answers: Mapping[str, float]) -> dict[str, float]:
    """The mean share of each answer over several orders of one question."""
    parts = [shares(a) for a in answers]
    return {answer: sum(part[answer] for part in parts) / len(parts) for answer in parts[0]}


def calibrate(answers: Mapping[str, float], blank: Mapping[str, float]) -> dict[str, float]:
    """Shares divided by those of a blank picture, renormalised (contextual calibration)."""
    own, base = shares(answers), shares(blank)
    scaled = {answer: own[answer] / max(base[answer], 1e-6) for answer in own}
    return shares(scaled)


def action_scores(answer_shares: Mapping[str, float]) -> dict[str, float]:
    """The scores the key gate reads: an action's share, if it beats "none"."""
    none = answer_shares.get(NONE, 0.0)
    return {a: s for a, s in answer_shares.items() if a != NONE and s > none}


__all__ = [
    "CHOICE_SYSTEM",
    "LETTERS",
    "NONE",
    "OPTION_TEXT",
    "ORDERS",
    "action_scores",
    "average",
    "by_answer",
    "calibrate",
    "key",
    "options",
    "prompt",
    "questions",
    "shares",
]
