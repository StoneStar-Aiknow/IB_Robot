"""The rules that turn the inputs into candidate requests (``Trigger``); no ROS.

Every time here is camera time (``header.stamp`` of the image, PEAR and hand
frames), so the rules agree with each other whatever the node's own clock
does. ``Trigger.stamp`` names the frame the candidate should be.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .config import EventRule, GestureRule, UniformRule

# Trigger sources, in the order their default weights rank them.
GESTURE = "gesture"
EVENT = "event"
STILL = "still"
UNIFORM = "uniform"
SOURCES = (GESTURE, EVENT, STILL, UNIFORM)


@dataclass(frozen=True)
class Trigger:
    source: str
    label: str
    weight: float
    score: float
    # Camera time of the frame wanted; None: the newest frame held.
    stamp: float | None
    # Who asked, for an event (``TriggerRequest.source``).
    origin: str = ""

    @property
    def priority(self) -> bool:
        """Everything but the steady rhythm: only these may take the reserved slots."""
        return self.source != UNIFORM


class UniformTrigger:
    def __init__(self, rule: UniformRule):
        self.rule = rule
        self._last: float | None = None

    def reset(self) -> None:
        self._last = None

    def update(self, stamp: float) -> Trigger | None:
        if self._last is not None and stamp - self._last < self.rule.period_sec:
            return None
        self._last = stamp
        return Trigger(UNIFORM, UNIFORM, self.rule.weight, 1.0, stamp)


class GestureTrigger:
    """``GestureRule`` over hand frames: each listed label held on either hand."""

    def __init__(self, rule: GestureRule):
        self.rule = rule
        self.reset()

    def reset(self) -> None:
        self._since: dict[str, float] = {}
        self._fired: dict[str, float] = {}

    def update(self, stamp: float, observations: Iterable[tuple[str, float]]) -> list[Trigger]:
        """``observations``: ``(gesture, score)`` of each detected hand in one frame."""
        rule = self.rule
        best: dict[str, float] = {}
        for label, score in observations:
            if label in rule.labels and score >= rule.min_score:
                best[label] = max(score, best.get(label, 0.0))
        for label in list(self._since):
            if label not in best:
                del self._since[label]
        fired = []
        for label, score in best.items():
            since = self._since.setdefault(label, stamp)
            if stamp - since < rule.hold_sec:
                continue
            last = self._fired.get(label)
            if last is not None and stamp - last < rule.min_gap_sec:
                continue
            self._fired[label] = stamp
            fired.append(Trigger(GESTURE, label, rule.labels[label], score, stamp))
        return fired


def event_trigger(rule: EventRule, origin: str, label: str, score: float, stamp: float | None) -> Trigger:
    return Trigger(EVENT, label, rule.labels.get(label, rule.default_weight), score, stamp, origin)
