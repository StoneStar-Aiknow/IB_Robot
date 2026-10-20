"""Which topics have been settled, and which observation has been confirmed twice.

Two jobs, both about not repeating:

**Coverage** records what this session has already said. The key is the closed
pair ``(category, region)`` and never the reply text, because "挂坠很特别",
"你的项链很有意思" and "我注意到你胸前的小饰品啦" are one repetition to a
visitor and three different strings to a similarity check.

**Confirmation** holds a static observation back until two *different* sampling
windows agree on it. A small vision-language model will name a star-shaped
pendant it cannot actually resolve, and a second look is the cheapest check
available — static content may be spoken a little late anyway. It is a check,
not a proof: two rounds can repeat one hallucination.

Coverage outranks event numbering. Putting an object down and raising it again
mints a fresh ``episode_seq``, and the first version cannot tell a second toy
from the first one returning, so a covered topic stays silent however high that
climbs. The alternative — letting a new episode speak again — turns one
ambiguous object into repeated remarks, which is the failure a visitor notices.

Nothing here imports ROS, and the clock arrives as a parameter: every decision
is reproducible from the values passed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from embodied_common.visual_interaction_contract import (
    BLOCKING_COVERAGE_STATES,
    COVERAGE_STATES,
    COVERED_ESTIMATED,
    DEFAULT_CONFIRMATION_COUNT,
    IN_FLIGHT,
    SUPPRESS_DUPLICATE,
    SUPPRESS_IN_FLIGHT,
    SUPPRESS_UNCONFIRMED,
)

# Regions that mean "the same place" for de-duplication. A toy held up reads as
# ``hand`` in one window and ``upper_body`` in the next, so treating them apart
# would let one object be remarked on twice. The mapping is deliberately
# one-way and conservative: it collapses towards the broader region, and
# ``unknown`` collapses into nothing because it would otherwise match
# everything and silence unrelated topics.
_REGION_ALIASES: Mapping[str, str] = {
    "hand": "upper_body",
    "chest": "upper_body",
    "upper_body": "upper_body",
    "head": "head",
    "unknown": "unknown",
}


def canonical_region(region: str) -> str:
    """Collapse a region onto the one de-duplication uses.

    An unmapped value passes through unchanged rather than raising: the caller
    has already validated it against the contract, and failing open here would
    only widen what counts as a new topic.
    """
    return _REGION_ALIASES.get(region, region)


@dataclass(frozen=True, order=True)
class TopicKey:
    """What de-duplication treats as one subject: a category in a place.

    This merges two different toys held in the same hand one after the other.
    The first version accepts that bias — staying quiet about a second object
    is a missed answer, while treating a returning object as new is a repeat —
    and does not claim object-instance tracking. Adding an ``object_slot_id``
    waits for an association anyone has accepted.
    """

    category: str
    region: str

    @classmethod
    def of(cls, category: str, region: str) -> TopicKey:
        return cls(category=category, region=canonical_region(region))


@dataclass(frozen=True)
class Confirmation:
    """How many distinct windows have agreed on one topic so far.

    ``windows`` holds the evidence stamps already counted, so re-reading the
    same frames cannot confirm anything: a window only counts when its newest
    frame is newer than the last one counted.
    """

    key: TopicKey
    kind: str
    windows: tuple[float, ...] = ()

    @property
    def count(self) -> int:
        return len(self.windows)

    def with_window(self, stamp: float) -> Confirmation:
        return replace(self, windows=(*self.windows, stamp))


@dataclass(frozen=True)
class Decision:
    """Whether a candidate may be spoken, and why not when it may not."""

    allowed: bool
    reason: str = ""
    confirmations: int = 0


@dataclass
class TopicLedger:
    """Coverage and confirmation for one session. Reset by starting a new one.

    Mutable on purpose: it is the session's memory. Every method takes the
    values it needs, so the ledger never reads a clock of its own.
    """

    confirmation_count: int = DEFAULT_CONFIRMATION_COUNT
    _coverage: dict[TopicKey, str] = field(default_factory=dict)
    _confirmations: dict[TopicKey, Confirmation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confirmation_count < 2:
            # One window is a single frame set, which is what the second window
            # exists to check.
            raise ValueError("confirmation_count must be at least 2")

    # -- coverage ---------------------------------------------------------

    def coverage_of(self, key: TopicKey) -> str:
        return self._coverage.get(key, "")

    def is_blocked(self, key: TopicKey) -> bool:
        return self.coverage_of(key) in BLOCKING_COVERAGE_STATES

    def mark(self, key: TopicKey, state: str) -> None:
        """Record a coverage state, refusing a value outside the contract."""
        if state not in COVERAGE_STATES:
            raise ValueError(f"{state!r} is not one of {list(COVERAGE_STATES)}")
        self._coverage[key] = state

    def reserve(self, key: TopicKey) -> None:
        """Claim a topic before synthesis, so a second round cannot race it in."""
        self.mark(key, IN_FLIGHT)

    def release(self, key: TopicKey, state: str) -> None:
        """Settle a reserved topic once delivery finished, or give it back.

        An empty ``state`` clears the reservation and leaves the topic
        unspoken: that is cancellation before anything could be heard. Anything
        else must be a coverage state, so a failed delivery can never silently
        become a covered one.
        """
        if not state:
            self._coverage.pop(key, None)
            # The confirmation chain goes too: the evidence it rested on is the
            # evidence that just turned out not to hold.
            self._confirmations.pop(key, None)
            return
        self.mark(key, state)

    # -- confirmation -----------------------------------------------------

    def observe(self, key: TopicKey, kind: str, *, window_stamp: float) -> int:
        """Count one window towards confirming a topic; return the count so far.

        A window whose newest frame is not newer than the last one counted is
        ignored, so the same frames cannot confirm themselves. A changed
        ``kind`` restarts the chain: an object becoming a display is a
        different claim, not more evidence for the old one.
        """
        existing = self._confirmations.get(key)
        if existing is None or existing.kind != kind:
            self._confirmations[key] = Confirmation(key=key, kind=kind, windows=(window_stamp,))
            return 1
        if existing.windows and window_stamp <= existing.windows[-1]:
            return existing.count
        updated = existing.with_window(window_stamp)
        self._confirmations[key] = updated
        return updated.count

    def confirmations(self, key: TopicKey) -> int:
        existing = self._confirmations.get(key)
        return existing.count if existing else 0

    def break_chain(self, key: TopicKey) -> None:
        """Drop a topic's evidence: occlusion, a blur or a stale window."""
        self._confirmations.pop(key, None)

    def break_all_chains(self) -> None:
        """Drop every chain, for a target change or a camera dropout.

        Coverage survives: what was already said was still said.
        """
        self._confirmations.clear()

    # -- the decision -----------------------------------------------------

    def admit(self, key: TopicKey, kind: str, *, window_stamp: float) -> Decision:
        """Count this window and answer whether the candidate may be spoken.

        Coverage is checked before the window is counted: a settled topic needs
        no more evidence, and counting it would let a long look rebuild a chain
        that can never be used.
        """
        state = self.coverage_of(key)
        if state == IN_FLIGHT:
            return Decision(allowed=False, reason=SUPPRESS_IN_FLIGHT)
        if state in BLOCKING_COVERAGE_STATES:
            return Decision(allowed=False, reason=SUPPRESS_DUPLICATE)

        count = self.observe(key, kind, window_stamp=window_stamp)
        if count < self.confirmation_count:
            return Decision(allowed=False, reason=SUPPRESS_UNCONFIRMED, confirmations=count)
        return Decision(allowed=True, confirmations=count)

    # -- bookkeeping ------------------------------------------------------

    def covered_keys(self) -> tuple[TopicKey, ...]:
        """Settled topics, sorted. For diagnostics and the prompt experiment.

        Design decision 7.2 compares keeping coverage host-side against listing
        it in the prompt; a stable order keeps that comparison reproducible.
        """
        return tuple(sorted(key for key, state in self._coverage.items() if state in BLOCKING_COVERAGE_STATES))

    def spoken_count(self) -> int:
        return sum(1 for state in self._coverage.values() if state == COVERED_ESTIMATED)


__all__ = [
    "Confirmation",
    "Decision",
    "TopicKey",
    "TopicLedger",
    "canonical_region",
]
