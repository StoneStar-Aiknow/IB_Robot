"""One observation round, start to finish: look, decide, say it once, remember.

The session owns the order of the checks and nothing else. Reading an answer is
``semantics``, remembering what was said is ``topics``, deciding whether a
candidate is still about this visitor is ``validity``, and talking to a model or
a speaker is ``hooks``.

Every round ends in exactly one outcome, and a quiet one is a normal result.
The order the checks run in is itself a decision:

1. **The target first.** Without an unambiguous person there is nothing to be
   about, and a model call would be wasted.
2. **Then the gate**, then the model. A refused window costs nothing; a
   prefill costs the same whatever the picture shows.
3. **Then the answer**, then coverage, then confirmation. Coverage before
   confirmation so a settled topic stops collecting evidence it can never use.
4. **Then validity, again, right before delivery.** Not when the model
   finished: what matters is how old the picture is when the words would
   actually be heard.

The session never reads a clock. ``RoundInputs.now`` is passed in, so a test
drives minutes of interaction in milliseconds and the same inputs always give
the same decision.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from embodied_common.visual_interaction_contract import (
    CATEGORIES,
    COVERED_ESTIMATED,
    DEFAULT_ABSENT_RESET_SEC,
    DEFAULT_CONFIRMATION_COUNT,
    DEFAULT_MAX_EVIDENCE_AGE_SEC,
    DEFAULT_MAX_SESSION_SEC,
    DEFAULT_MIN_GAP_SEC,
    KIND_UNCERTAIN,
    PARTIAL_COVERED,
    SPEAKABLE_KINDS,
    SUPPRESS_AMBIGUOUS,
    SUPPRESS_COOLDOWN,
    SUPPRESS_MIRROR_INACTIVE,
    SUPPRESS_MODEL_NOT_READY,
    SUPPRESS_MUTED,
    SUPPRESS_NO_NEW_TOPIC,
    SUPPRESS_STALE,
    SUPPRESS_WAITING_NEXT_VISITOR,
    SUPPRESSED,
)
from visual_interaction.session.hooks import RoundInputs, SessionHooks
from visual_interaction.session.semantics import (
    Observation,
    SemanticsError,
    parse_observation,
    quiet_reason,
)
from visual_interaction.session.topics import TopicKey, TopicLedger
from visual_interaction.session.validity import (
    END_ABSENT,
    END_HANDOVER,
    ActorTracker,
    Candidate,
    SessionWindow,
    candidate_validity,
)


@dataclass(frozen=True)
class RoundResult:
    """What one round decided, and what the adapter should publish.

    ``spoken`` true means the text was delivered and the topic is settled.
    Otherwise ``reason`` says why not — always one of the contract's
    suppression reasons, so a reader never has to parse a message.
    """

    spoken: bool
    reason: str = ""
    # Why the reason, when there is more to say than the closed code. Carries
    # the SemanticsError detail for a refused or malformed answer, which is the
    # only record of what the model actually produced: diagnostics would
    # otherwise report SCHEMA_INVALID forty times with no way to tell whether
    # the model wrote prose, truncated, or answered a field wrong.
    #
    # For logs and diagnostics only. Nothing in the decision reads it, and the
    # published message keeps carrying the closed reason alone.
    detail: str = ""
    text: str = ""
    observation: Observation | None = None
    topic: TopicKey | None = None
    event_id: str = ""
    session_id: str = ""
    actor_epoch: int = 0
    confirmations: int = 0
    model_called: bool = False
    evidence_stamp: float = 0.0
    # Counts display episodes of this topic, for tracing and cancellation only;
    # coverage outranks it. Zero on a round that produced no candidate.
    episode_seq: int = 0
    # The latest moment speaking may START, derived from the evidence age and
    # the configured budget. Not a deadline for finishing playback, and zero
    # when there is nothing to say.
    latest_start_at: float = 0.0

    @property
    def quiet(self) -> bool:
        return not self.spoken


@dataclass
class SessionConfig:
    """Timing and limits. Every default is a proposal until phase zero measures.

    None of these is a measured value, and the names say what they bound rather
    than what was observed.
    """

    confirmation_count: int = DEFAULT_CONFIRMATION_COUNT
    max_evidence_age_sec: float = DEFAULT_MAX_EVIDENCE_AGE_SEC
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC
    absent_reset_sec: float = DEFAULT_ABSENT_RESET_SEC
    max_session_sec: float = DEFAULT_MAX_SESSION_SEC
    # Empty means the contract's whole whitelist. A deployment may narrow it,
    # never widen it: ``parse_observation`` raises on a value outside the
    # contract rather than honouring it.
    categories: tuple[str, ...] = CATEGORIES

    def __post_init__(self) -> None:
        if self.min_gap_sec < 0.0:
            raise ValueError("min_gap_sec cannot be negative")
        if self.max_evidence_age_sec <= 0.0:
            raise ValueError("max_evidence_age_sec must be positive")
        if not self.categories:
            raise ValueError("categories cannot be empty: a session with no subject can never speak")


@dataclass
class VisualInteractionSession:
    """The whole decision for one deployment. One round per :meth:`step`."""

    hooks: SessionHooks = field(default_factory=SessionHooks)
    config: SessionConfig = field(default_factory=SessionConfig)
    tracker: ActorTracker = field(default_factory=ActorTracker)
    _ledger: TopicLedger | None = field(default=None, init=False)
    _window: SessionWindow | None = field(default=None, init=False)
    _session_seq: int = field(default=0, init=False)
    _event_seq: int = field(default=0, init=False)
    _episode_seq: int = field(default=0, init=False)
    _quiet_until: float = field(default=0.0, init=False)
    _muted: bool = field(default=False, init=False)
    _awaiting_handover: bool = field(default=False, init=False)
    _last_window_stamp: float | None = field(default=None, init=False)

    # -- lifecycle --------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._window.session_id if self._window else ""

    @property
    def actor_epoch(self) -> int:
        return self.tracker.epoch

    @property
    def muted(self) -> bool:
        return self._muted

    def ledger(self) -> TopicLedger:
        if self._ledger is None:
            self._ledger = TopicLedger(confirmation_count=self.config.confirmation_count)
        return self._ledger

    def start_session(self, *, now: float, reason: str = "start") -> str:
        """Open a fresh de-duplication range and forget every covered topic."""
        previous = self._window
        if previous is not None:
            self.hooks.on_session_end(previous.session_id, reason)
        self._session_seq += 1
        self._window = SessionWindow(
            session_id=f"s{self._session_seq}",
            started_at=now,
            absent_reset_sec=self.config.absent_reset_sec,
            max_session_sec=self.config.max_session_sec,
        )
        self._ledger = TopicLedger(confirmation_count=self.config.confirmation_count)
        self._last_window_stamp = None
        self._awaiting_handover = False
        # The quiet gap and the mute survive: both are about the robot's own
        # behaviour, not about who is standing there.
        return self._window.session_id

    def next_visitor(self, *, now: float) -> str:
        """Explicit hand-over. The normal way a new session starts at a stand.

        A queue rarely produces a confirmed absence, so without this the
        absence timer and the duration limit are the only resets and every
        later visitor inherits the first one's covered topics.
        """
        return self.start_session(now=now, reason=END_HANDOVER)

    def mute(self) -> None:
        """Stop admitting candidates until explicitly resumed.

        A visitor asking for quiet, relayed by an attendant, must not be undone
        by a timer: only :meth:`resume` clears it.
        """
        self._muted = True

    def resume(self) -> None:
        self._muted = False

    # -- one round --------------------------------------------------------

    def step(self, inputs: RoundInputs) -> RoundResult:
        """Run one observation round and return what it decided."""
        if self._window is None:
            self.start_session(now=inputs.now)
        window = self._window
        assert window is not None

        window.observe(present=inputs.actor.present, known=inputs.actor.known, now=inputs.now)
        expiry = window.expired_reason(inputs.now)
        if expiry == END_ABSENT:
            # Nobody is there to inherit the covered topics, so a new range may
            # start on its own.
            self.start_session(now=inputs.now, reason=expiry)
            window = self._window
            assert window is not None
        elif expiry:
            # The duration limit is a missed-hand-over signal, not a reset:
            # clearing coverage while the same visitor is still standing there
            # would say the same things to them again every two minutes.
            # ``next_visitor`` is how an attendant ends it.
            self._awaiting_handover = True
            return self._finish(inputs, SUPPRESS_WAITING_NEXT_VISITOR)
        if self._awaiting_handover:
            return self._finish(inputs, SUPPRESS_WAITING_NEXT_VISITOR)
        if not inputs.mirror_active:
            # The speech context is absent (the mirror is not MIRRORING). The
            # window bookkeeping above still ran, so a visitor who leaves while
            # the feature is parked still rotates the session; skipping the
            # association is conservative and self-healing, because the first
            # round after a real pause sees a stamp gap that relocks the target
            # and breaks every chain collected before it.
            return self._finish(inputs, SUPPRESS_MIRROR_INACTIVE)

        association = self.tracker.update(
            present=inputs.actor.present,
            known=inputs.actor.known,
            box=inputs.actor.bbox,
            stamp=inputs.actor.stamp,
            others=inputs.actor.others,
        )
        if association.epoch_bumped:
            # Everything collected under the previous epoch rested on a target
            # this one cannot vouch for.
            self.ledger().break_all_chains()

        if association.ambiguous:
            return self._break_and_finish(inputs, SUPPRESS_AMBIGUOUS)
        if not association.associated:
            return self._break_and_finish(inputs, SUPPRESS_STALE)

        if self._muted:
            return self._finish(inputs, SUPPRESS_MUTED)
        if inputs.now < self._quiet_until:
            return self._finish(inputs, SUPPRESS_COOLDOWN)

        if not self.hooks.gate.should_observe(inputs):
            return self._finish(inputs, SUPPRESS_NO_NEW_TOPIC)

        raw = self.hooks.vision.observe(inputs)
        if raw is None:
            return self._break_and_finish(inputs, SUPPRESS_MODEL_NOT_READY, model_called=True)

        try:
            observation = parse_observation(raw, categories=self.config.categories)
        except SemanticsError as error:
            return self._break_and_finish(inputs, error.reason, model_called=True, detail=error.detail)

        if observation.kind not in SPEAKABLE_KINDS:
            if observation.kind == KIND_UNCERTAIN:
                # Cannot see it well enough is a break in the evidence, not a
                # gap to bridge: otherwise an occlusion lets an old look and a
                # new one add up to a confirmation.
                self.ledger().break_all_chains()
            return self._finish(inputs, quiet_reason(observation.kind), model_called=True, observation=observation)

        key = TopicKey.of(observation.category, observation.region)
        decision = self.ledger().admit(key, observation.kind, window_stamp=inputs.window.newest_stamp)
        if not decision.allowed:
            return self._finish(
                inputs,
                decision.reason,
                model_called=True,
                observation=observation,
                topic=key,
                confirmations=decision.confirmations,
            )

        return self._deliver(inputs, observation, key, decision.confirmations)

    # -- delivery ---------------------------------------------------------

    def _deliver(
        self,
        inputs: RoundInputs,
        observation: Observation,
        key: TopicKey,
        confirmations: int,
    ) -> RoundResult:
        self._event_seq += 1
        self._episode_seq += 1
        event_id = f"{self.session_id}-e{self._event_seq}"
        candidate = Candidate(
            event_id=event_id,
            session_id=self.session_id,
            actor_epoch=self.actor_epoch,
            evidence_stamp=inputs.window.newest_stamp,
            text=observation.reply,
        )

        # Asked again rather than reusing the round's start: generation takes
        # time, and the age that decides this is the age when the words would
        # be heard. A fake model returns instantly, which is precisely why the
        # clock is a port and not a captured value.
        now = self.hooks.clock.now(inputs)
        reason = candidate_validity(
            candidate,
            session_id=self.session_id,
            actor_epoch=self.actor_epoch,
            target_present=inputs.actor.present,
            topic_blocked=False,
            now=now,
            max_age_sec=self.config.max_evidence_age_sec,
        )
        if reason:
            return self._finish(
                inputs,
                reason,
                model_called=True,
                observation=observation,
                topic=key,
                confirmations=confirmations,
            )

        ledger = self.ledger()
        ledger.reserve(key)
        delivery = self.hooks.speech.deliver(observation.reply, event_id=event_id, now=now)

        if not delivery.accepted:
            # Nothing could have been heard, so the topic goes back to unspoken
            # and its evidence goes with it.
            ledger.release(key, "")
            return self._finish(
                inputs,
                SUPPRESS_STALE if delivery.detail == "stale" else SUPPRESS_MODEL_NOT_READY,
                model_called=True,
                observation=observation,
                topic=key,
                confirmations=confirmations,
            )

        if delivery.partial:
            # Some of it may have been heard; it is not re-said from the top.
            ledger.release(key, PARTIAL_COVERED)
        else:
            ledger.release(key, COVERED_ESTIMATED)

        self._quiet_until = delivery.estimated_end + self.config.min_gap_sec
        result = RoundResult(
            spoken=True,
            text=observation.reply,
            observation=observation,
            topic=key,
            event_id=event_id,
            session_id=self.session_id,
            actor_epoch=self.actor_epoch,
            confirmations=confirmations,
            model_called=True,
            evidence_stamp=inputs.window.newest_stamp,
            episode_seq=self._episode_seq,
            latest_start_at=inputs.window.newest_stamp + self.config.max_evidence_age_sec,
        )
        self._notify(inputs, "")
        return result

    def interrupt(self, topic: TopicKey, *, by_user: bool) -> None:
        """Stop an utterance mid-flight and record what that means for the topic.

        A visitor's stop is recorded as suppressed *and* mutes the session: the
        same remark must not come back on the next round, which is exactly what
        leaving the topic unspoken would do. A content-invalid cancel leaves it
        unspoken, because nothing was wrong with saying it.
        """
        if by_user:
            self.ledger().release(topic, SUPPRESSED)
            self.mute()
        else:
            self.ledger().release(topic, "")

    # -- helpers ----------------------------------------------------------

    def _break_and_finish(self, inputs: RoundInputs, reason: str, **kwargs) -> RoundResult:
        """End the round and drop every confirmation chain.

        For a round that saw nothing it can vouch for: an ambiguous target, a
        lost one, an unreachable model, an unreadable answer. Keeping the
        chains would let a look from before the break and one from after add
        up to "confirmed twice", which is not what two windows agreeing means.
        """
        self.ledger().break_all_chains()
        return self._finish(inputs, reason, **kwargs)

    def _finish(
        self,
        inputs: RoundInputs,
        reason: str,
        *,
        model_called: bool = False,
        observation: Observation | None = None,
        topic: TopicKey | None = None,
        confirmations: int = 0,
        detail: str = "",
    ) -> RoundResult:
        self._notify(inputs, reason)
        return RoundResult(
            spoken=False,
            reason=reason,
            detail=detail,
            observation=observation,
            topic=topic,
            session_id=self.session_id,
            actor_epoch=self.actor_epoch,
            confirmations=confirmations,
            model_called=model_called,
            evidence_stamp=inputs.window.newest_stamp,
        )

    def _notify(self, inputs: RoundInputs, outcome: str) -> None:
        # The round is over by the time this runs, so a broken diagnostics hook
        # must not turn a decision that already happened into an exception.
        with contextlib.suppress(Exception):
            self.hooks.on_round(inputs, outcome)


__all__ = ["RoundResult", "SessionConfig", "VisualInteractionSession"]
