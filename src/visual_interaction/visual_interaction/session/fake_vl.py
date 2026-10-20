"""A scripted stand-in for the vision-language model.

The session is the part worth testing, and it has no opinion about what
produced an answer. This port hands back answers from a list, so a test can
drive a whole session — two-window confirmation, a settled topic, a visitor
swap — with no model, no NPU and no ROS.

It is also what the first version runs with: ``vl.implementation: fake`` in the
robot config keeps the skeleton launchable before the real loop is measured.

What it deliberately does *not* do: imitate the board demo's timing, retries or
sampling. Those are phase-zero measurements, and a plausible-looking fake would
invite someone to treat them as known. One answer per call, in order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from embodied_common.visual_interaction_contract import (
    CATEGORIES,
    KIND_DISPLAY,
    KIND_NONE,
    KINDS,
    REGIONS,
)
from visual_interaction.session.hooks import RoundInputs, VisionLanguagePort

# A reply naming each category, so a scripted answer passes the grounding check
# in ``semantics.check_reply`` without every test spelling one out.
GROUNDED_REPLIES: Mapping[str, str] = {
    "toy": "你举起的玩偶真有趣。",
    "sign": "你举着的牌子我看到了。",
    "hat": "你戴的帽子很有特点。",
    "badge": "你别着的徽章很醒目。",
}


def answer(
    category: str = "toy",
    *,
    region: str = "hand",
    kind: str = KIND_DISPLAY,
    reply: str | None = None,
) -> dict[str, object]:
    """Build one well-formed answer. Raises on a value outside the contract.

    Refusing an invalid value here is the point: a test that wants to check how
    a bad answer is handled should write the bad answer out, so the intent is
    visible rather than hidden behind a helper.
    """
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {list(KINDS)}")
    if category not in CATEGORIES:
        raise ValueError(f"category {category!r} is not one of {list(CATEGORIES)}")
    if region not in REGIONS:
        raise ValueError(f"region {region!r} is not one of {list(REGIONS)}")
    return {
        "kind": kind,
        "category": category,
        "region": region,
        "reply": reply if reply is not None else GROUNDED_REPLIES[category],
    }


def quiet() -> dict[str, object]:
    """The answer for "nothing new worth answering"."""
    return {"kind": KIND_NONE}


@dataclass
class ScriptedVision(VisionLanguagePort):
    """Returns prepared answers in order, then repeats the last one.

    Repeating rather than running out is what makes two-window confirmation
    easy to write: scripting one answer and calling twice is the common case.
    Pass ``repeat_last=False`` to answer ``none`` once the script is used up.

    Every call is recorded in :attr:`calls`, so a test can assert *when* the
    model was consulted — which is how the change gate and the synthesis
    interlock are checked.
    """

    answers: Sequence[str | Mapping[str, object] | None] = ()
    repeat_last: bool = True
    calls: list[RoundInputs] = field(default_factory=list)

    @classmethod
    def of(cls, *answers: str | Mapping[str, object] | None, repeat_last: bool = True) -> ScriptedVision:
        return cls(answers=tuple(answers), repeat_last=repeat_last)

    @classmethod
    def unreachable(cls) -> ScriptedVision:
        """A model that cannot be reached, reported as MODEL_NOT_READY."""
        return cls(answers=(None,), repeat_last=True)

    def extend(self, answers: Iterable[str | Mapping[str, object] | None]) -> None:
        self.answers = (*self.answers, *answers)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def observe(self, inputs: RoundInputs) -> str | dict[str, object] | None:
        index = len(self.calls)
        self.calls.append(inputs)
        if not self.answers:
            return quiet()
        if index < len(self.answers):
            chosen = self.answers[index]
        elif self.repeat_last:
            chosen = self.answers[-1]
        else:
            return quiet()
        return dict(chosen) if isinstance(chosen, Mapping) else chosen


__all__ = ["GROUNDED_REPLIES", "ScriptedVision", "answer", "quiet"]
