"""Key actions: when a stream of per-look scores counts as "seen".

Plan v11 §3. What scores a look varies (look_say's questions, InternVL's
choice-letter probabilities, ...); the rule that turns the scores into one
trigger per gesture does not, and the offline evaluation and the node share
it, or the numbers measured would not be the robot's.

* A look's candidate is its best-scoring action, if that score reaches the
  action's threshold. An action with no threshold is never a candidate.
* ``confirm`` looks in a row with the same candidate trigger it.
* A triggered action has to be absent for ``release`` looks before it can
  trigger again: someone holding a heart for ten seconds is one heart.
* And it waits ``cooldown_sec`` after its last trigger in any case.

Pure Python; the caller passes the clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# The five candidates of plan v11; the three ways to make a heart are one.
KEY_ACTIONS = ("yeah", "thumb", "wave", "heart", "open")


@dataclass(frozen=True)
class GateConfig:
    thresholds: Mapping[str, float]
    confirm: int = 2
    release: int = 1
    cooldown_sec: float = 8.0

    def __post_init__(self) -> None:
        unknown = sorted(set(self.thresholds) - set(KEY_ACTIONS))
        if unknown:
            raise ValueError(f"unknown key actions {unknown}; known: {KEY_ACTIONS}")
        if self.confirm < 1 or self.release < 1:
            raise ValueError("confirm and release must be at least 1")
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must not be negative")


@dataclass(frozen=True)
class Trigger:
    action: str
    at: float
    score: float


@dataclass
class KeyGate:
    """Feed it one look at a time with :meth:`update`; it returns a trigger or None."""

    config: GateConfig
    _streak: tuple[str | None, int] = field(default=(None, 0), init=False)
    _absent: dict = field(default_factory=dict, init=False)
    _armed: dict = field(default_factory=dict, init=False)
    _last: dict = field(default_factory=dict, init=False)

    def reset(self) -> None:
        """Forget everything, e.g. when a new person steps in."""
        self._streak = (None, 0)
        self._absent.clear()
        self._armed.clear()
        self._last.clear()

    def candidate(self, scores: Mapping[str, float]) -> tuple[str, float] | None:
        """The look's best action with its score, if it reaches its threshold."""
        best = None
        for action, threshold in self.config.thresholds.items():
            score = scores.get(action)
            if score is None or score < threshold:
                continue
            if best is None or score > best[1]:
                best = (action, float(score))
        return best

    def update(self, now: float, scores: Mapping[str, float]) -> Trigger | None:
        """One look's scores (empty for a look that saw nothing or failed)."""
        found = self.candidate(scores)
        action = found[0] if found else None
        for name in self.config.thresholds:
            if name == action:
                self._absent[name] = 0
                continue
            self._absent[name] = self._absent.get(name, 0) + 1
            if self._absent[name] >= self.config.release:
                self._armed[name] = True
        previous, count = self._streak
        self._streak = (action, count + 1 if action is not None and action == previous else int(action is not None))
        if action is None or self._streak[1] < self.config.confirm or not self._armed.get(action, True):
            return None
        last = self._last.get(action)
        if last is not None and now - last < self.config.cooldown_sec:
            return None
        self._armed[action] = False
        self._last[action] = now
        return Trigger(action, now, found[1])


__all__ = ["KEY_ACTIONS", "GateConfig", "KeyGate", "Trigger"]
