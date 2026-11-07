"""Bridge final ASR transcripts into the incubating Agent request contract.

``voice_asr_node`` publishes plain final text (``std_msgs/String``); the Agent
only accepts the JSON envelope on ``/agent/request`` and blocks execution until
a client acknowledges the exact-plan ``presentation`` event.  This module is the
voice client of that contract: it wraps utterances into requests, routes stop
keywords to ``/agent/control`` (already from partial transcripts, so motion stops
before the utterance ends), renders presentations (log, optional TTS) before
acknowledging them, and speaks Agent replies.  While a request can still be
interrupted, speaking never mutes ASR: the stop keyword stays live and the
bridge's own replies (which therefore carry no stop word) are filtered out.

``VoiceBridgeCore`` is ROS-free and returns transport actions so the protocol
logic is unit-testable; ``VoiceBridgeNode`` only executes those actions.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from embodied_common.control_phrases import STOP_PHRASES
from embodied_common.text_normalization import normalize_trigger_text
from ibrobot_agent.addressee import (
    ADDRESSEE_TRIGGERS,
    AddresseeGate,
    AnchorCheck,
    Segment,
    SpeakerCheck,
    Voiceprint,
)
from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE, TERMINAL_REQUEST_STATES
from ibrobot_agent.execution_feedback import FeedbackTracker, execution_reply
from ibrobot_agent.presentation import presentation_digest
from ibrobot_agent.semantic_gate import SEMANTIC_GATE_MODES, SemanticGate

CONFIRM_MODES = ("auto", "voice_confirm")
_ACTIVE_ASR_STATES = {"listening", "recognizing"}
#: A refusal can contain a confirm word ("不执行" has 执行, "我不确定" has 确定); it must never
#: acknowledge a plan. "没" is left out on purpose: "没问题" is consent.
_NEGATION_MARKERS = ("不", "别")
_PHRASE_FIELDS = (
    "stop_phrases",
    "bypass_phrases",
    "confirm_phrases",
    "cancel_phrases",
    "wake_phrases",
    "silent_skills",
    "semantic_gate_lexicon",
    "semantic_gate_wake_prefixes",
)


@dataclass(frozen=True)
class VoiceBridgeConfig:
    channel_id: str = "voice_asr"
    principal_id: str = "local_operator"
    # Robot-specific and part of the Agent request key: always given by the launch builder.
    robot_scope: str = ""
    stop_phrases: tuple[str, ...] = STOP_PHRASES
    bypass_phrases: tuple[str, ...] = ()
    #: Empty keeps every utterance, which is what deployments without a
    #: wake word have always done. Set it and an utterance must either start
    #: with one of these or fall inside the window a previous one opened.
    wake_phrases: tuple[str, ...] = ()
    #: Skills whose output is speech itself (a recorded clip): the bridge announces nothing for
    #: them and says nothing afterwards, or it talks over the clip.
    silent_skills: tuple[str, ...] = ()
    #: Whether the planner's early reply is spoken as it arrives. Off when a deployment owns
    #: the speaker itself (silent_skills), because the preview cannot know the plan is silent.
    speak_reply_previews: bool = True
    addressee_trigger: str = "wake_phrase"
    #: Silence that ends a ``vendor_session`` session, and with it this gate's
    #: window; only the next wake event opens another. Must exceed the longest
    #: expected utterance plus the ASR's finalization lag, because the gap is
    #: measured between transcripts, not between the moments speech actually
    #: stopped. Too small, and the user's own follow-up after a pause needs the
    #: wake word again; too large, and a session outlives the conversation.
    session_gap_sec: float = 15.0
    #: ``wake_phrase``: how long a wake phrase keeps the window open for
    #: follow-ups. ``vendor_session``: how long a wake event stays pending for
    #: the transcript of the utterance it opened (the session itself is
    #: bounded by ``session_gap_sec``).
    wake_window_sec: float = 12.0
    #: Voiceprint check over the window the wake phrase opened. Needs
    #: ``wake_phrases``: the anchor is the utterance that said one. Requires
    #: ``voice_asr.speaker_embedding.enabled`` to be publishing measurements.
    speaker_check_enabled: bool = False
    #: Cosine below which a follow-up is treated as somebody else's. The
    #: default is the value the X2 was tested at (2026-10-24, campplus, >=2 s
    #: segments): other speakers against the anchor scored 0.28-0.56, the
    #: user's own follow-ups 0.67-0.84 once the anchor accumulates. The
    #: offline corpus numbers are not a basis: its labels were not one person
    #: per group. It still has to be re-measured per deployment; see
    #: SpeakerCheck and the calibration method in this package's README.
    speaker_min_similarity: float = 0.60
    #: A voiceprint whose transcript has not come within this long is dropped:
    #: the ASR discarded that segment. Voiceprints are paired with transcripts
    #: by segment, so this only bounds memory, never which voice judges whom.
    speaker_max_age_sec: float = 2.5
    #: An accepted voiceprint is folded into the anchor only when it scores at
    #: least this much above speaker_min_similarity; see SpeakerCheck.merge.
    speaker_merge_margin: float = 0.10
    #: With an anchor set, reject a follow-up that was too short to produce a
    #: voiceprint (``speaker_id_node`` published an empty one) rather than let it
    #: through unchecked. Measured on the X2 with real people (2026-10-26): the
    #: commands they actually give ("挥手", "握手握手") are about a second of
    #: speech, so with fail-open a second person got every command through.
    #: The cost is the user's own short commands: they need a few more words.
    #: A voiceprint that never arrives (the node is down) still fails open.
    speaker_reject_short: bool = False
    #: How long to hold a transcript for its own segment's measurement, when
    #: any check is enabled. Publisher ordering cannot be relied on: both the
    #: ASR and the voiceprint node finalize on the same END or the same stall,
    #: so which lands first is a coin flip. Past this the transcript is judged
    #: without it (fail-open). A stop keyword is never held.
    measurement_grace_sec: float = 0.4
    # Without knowing about TTS, the safe default is a spoken confirmation (see resolve_confirm_mode).
    confirm_mode: str = "voice_confirm"
    confirm_phrases: tuple[str, ...] = ("确认", "确定", "好的", "执行")
    cancel_phrases: tuple[str, ...] = ("取消", "算了", "不要")
    confirm_timeout_sec: float = 20.0
    session_idle_sec: float = 120.0
    min_text_chars: int = 2
    request_timeout_sec: float = 180.0
    speak_replies: bool = True
    # Consume execution feedback for this text channel without accepting speech commands.
    feedback_only: bool = False
    # Spoken while a request runs and the microphone stays open for stop words, so it must not
    # contain one (the bridge would hear itself and stop the robot).
    busy_message: str = "我正在执行上一个任务，请稍等。"
    # Domain term -> homophones the ASR model prefers (e.g. 夹爪 jiā zhǎo is heard as 家找/家长).
    text_corrections: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Semantic gate (``semantic_gate.py``): the L3 decision the wake/voiceprint gate cannot
    #: make -- whether the words were meant for the robot at all. ``off`` keeps the bridge
    #: behaving exactly as before it existed; ``shadow`` judges and records into
    #: ``semantic_gate_log_path`` without ever dropping an utterance, which is how the
    #: thresholds get calibrated on the deployment's own traffic; ``enforce`` drops the
    #: utterances the gate vetoes. Without ``wake_phrases`` or ``vendor_session`` the bridge
    #: admits every utterance, so this is the only thing standing between a meeting in the
    #: room and the planner.
    semantic_gate_mode: str = "off"
    #: Trained model (see ``tools/addressee``). Required as soon as the mode is not ``off``.
    semantic_gate_model_path: str = ""
    #: Below this score an utterance is only vetoed when no skill word or wake prefix matches.
    #: Deliberately low: the field corpus' scores are bimodal (half of them are 0.000, three
    #: quarters 1.000), so a low threshold costs little specificity and buys recall, and a
    #: dropped command is the failure the user feels.
    semantic_gate_threshold: float = 0.10
    #: Skill words that always win over the model (the catalog's ``aliases_zh``): an unseen
    #: skill scores badly by construction, see the module docstring.
    semantic_gate_lexicon: tuple[str, ...] = ()
    #: Leading wake-word spellings the ASR produces (灵犀 heard as 林星/年轻人/...): a sentence
    #: that starts with one was said to the robot, whatever the model thinks of the rest.
    semantic_gate_wake_prefixes: tuple[str, ...] = ()
    #: JSONL of every judgement (shadow and enforce alike). Empty: ROS log only.
    semantic_gate_log_path: str = ""
    #: A model that will not load disables the gate instead of failing the launch. The gate
    #: can only drop utterances it is confident about, so running without it is the safe side.
    semantic_gate_fail_open: bool = True

    def __post_init__(self) -> None:
        if not str(self.robot_scope).strip():
            raise ValueError("robot_scope must be the Agent's robot scope")
        if self.confirm_mode not in CONFIRM_MODES:
            raise ValueError(f"confirm_mode must be one of {CONFIRM_MODES}")
        if self.addressee_trigger not in ADDRESSEE_TRIGGERS:
            raise ValueError(f"addressee_trigger must be one of {ADDRESSEE_TRIGGERS}")
        if not float(self.wake_window_sec) > 0:
            raise ValueError("wake_window_sec must be positive")
        if not float(self.session_gap_sec) > 0:
            raise ValueError("session_gap_sec must be positive")
        if not 0.0 < float(self.speaker_min_similarity) <= 1.0:
            raise ValueError("speaker_min_similarity must be in (0, 1]")
        if not float(self.speaker_max_age_sec) > 0:
            raise ValueError("speaker_max_age_sec must be positive")
        if not 0.0 <= float(self.speaker_merge_margin) < 1.0:
            raise ValueError("speaker_merge_margin must be in [0, 1)")
        if not float(self.measurement_grace_sec) >= 0:
            raise ValueError("measurement_grace_sec must not be negative")
        for name in ("confirm_timeout_sec", "session_idle_sec", "request_timeout_sec"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be positive")
        if self.min_text_chars < 1:
            raise ValueError("min_text_chars must be at least 1")
        for name in _PHRASE_FIELDS:
            object.__setattr__(self, name, tuple(_normalize(item) for item in getattr(self, name) if _normalize(item)))
        if not self.stop_phrases:
            raise ValueError("stop_phrases must not be empty")
        if self.addressee_trigger == "vendor_session" and self.wake_phrases:
            # Two judgements of the same thing. Under vendor_session the wake
            # word never reaches the text, so a phrase list could only reject
            # every utterance.
            raise ValueError("addressee_trigger=vendor_session must not set wake_phrases")
        gate_is_on = self.addressee_trigger == "vendor_session" or bool(self.wake_phrases)
        if self.speaker_check_enabled and not gate_is_on:
            # Anchors are only ever captured from the utterance that proves
            # itself. With the gate off nothing ever captures one, and the check
            # would silently do nothing.
            raise ValueError("speaker_check_enabled requires an enabled gate: set wake_phrases or addressee_trigger")
        if any(phrase in _normalize(self.busy_message) for phrase in self.stop_phrases):
            raise ValueError("busy_message must not contain a stop phrase: the bridge would stop its own request")
        corrections = []
        for term, variants in self.text_corrections:
            term = str(term).strip()
            variants = tuple(str(item).strip() for item in variants if str(item).strip() and str(item).strip() != term)
            if not term or not variants:
                raise ValueError("text_corrections entries need a term and at least one misrecognized variant")
            corrections.append((term, variants))
        object.__setattr__(self, "text_corrections", tuple(corrections))
        if self.semantic_gate_mode not in SEMANTIC_GATE_MODES:
            raise ValueError(f"semantic_gate_mode must be one of {SEMANTIC_GATE_MODES}")
        if self.semantic_gate_mode != "off" and not str(self.semantic_gate_model_path).strip():
            raise ValueError("semantic_gate_model_path is required unless semantic_gate_mode is off")
        if not 0.0 <= float(self.semantic_gate_threshold) <= 1.0:
            raise ValueError("semantic_gate_threshold must be in [0, 1]")

    @property
    def pairs_by_segment(self) -> bool:
        """Whether the gate needs segment-tagged transcripts (``SpeechTranscript``).

        A voiceprint judges only the transcript of its own segment, and under
        ``vendor_session`` only the segment a wake event names opens a session,
        so either needs to know which segment a transcript came from.
        """
        return self.speaker_check_enabled or self.addressee_trigger == "vendor_session"

    def anchor_checks(self) -> tuple[AnchorCheck, ...]:
        """The enabled anchor checks, in the order an utterance is run through them."""
        checks: list[AnchorCheck] = []
        if self.speaker_check_enabled:
            checks.append(
                SpeakerCheck(
                    float(self.speaker_min_similarity),
                    float(self.speaker_max_age_sec),
                    float(self.speaker_merge_margin),
                    bool(self.speaker_reject_short),
                )
            )
        return tuple(checks)

    def correct(self, text: str) -> str:
        """Replace known misrecognitions, longest variant first so overlaps resolve deterministically."""
        pairs = sorted(
            ((variant, term) for term, variants in self.text_corrections for variant in variants),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        for variant, term in pairs:
            text = text.replace(variant, term)
        return text


@dataclass(frozen=True)
class PublishRequest:
    payload: dict[str, Any]


@dataclass(frozen=True)
class PublishControl:
    payload: dict[str, Any]


@dataclass(frozen=True)
class Speak:
    """Speak ``text``; run ``then`` only once the speech was rendered.

    When speech is disabled the text is logged and ``then`` runs immediately;
    when synthesis or playback fails ``then`` is dropped, so a presentation the
    user never heard is never acknowledged (the Agent fails closed on timeout).
    """

    text: str
    then: tuple[Any, ...] = ()
    execution_id: str = ""
    terminal: bool = False


@dataclass(frozen=True)
class Log:
    message: str


Action = PublishRequest | PublishControl | Speak | Log


@dataclass
class _PendingConfirmation:
    request_id: str
    acknowledgement: PublishControl
    deadline: float


@dataclass(frozen=True)
class _HeldTranscript:
    """A final transcript waiting for the measurement it will be judged by."""

    text: str
    heard_at: float
    #: Routed by then even if the measurement never arrives (fail-open).
    deadline: float
    segment: Segment | None


#: Request ids remembered for event/response matching. One id per utterance, so this bounds
#: memory for a long-running node while still covering any in-flight Agent request.
_REMEMBERED_REQUEST_IDS = 256


class _BoundedIdSet:
    """Insertion-ordered id set that forgets its oldest entries past ``limit``."""

    def __init__(self, limit: int = _REMEMBERED_REQUEST_IDS) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._limit = limit

    def add(self, request_id: str) -> None:
        self._ids[request_id] = None
        self._ids.move_to_end(request_id)
        while len(self._ids) > self._limit:
            self._ids.popitem(last=False)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)


@dataclass
class _BridgeState:
    session_id: str = ""
    last_activity: float = 0.0
    active_request_id: str = ""
    active_since: float = 0.0
    known_request_ids: _BoundedIdSet = field(default_factory=_BoundedIdSet)
    finished_request_ids: _BoundedIdSet = field(default_factory=_BoundedIdSet)
    # Released by the request timeout without a terminal event: the robot may still move, so a
    # later stop keyword must still reach it.
    timed_out_request_id: str = ""
    # The current utterance already stopped from a partial ("停" -> "停一下" grow word by word):
    # later partials and its final transcript belong to that stop. Cleared by every final.
    utterance_stopped: bool = False
    clarification_request_id: str = ""
    silent_request_id: str = ""
    pending: _PendingConfirmation | None = None
    # The reply already said ahead of the plan (``reply_preview``), so the
    # answer that repeats it is not said twice. Only the active request can
    # have one.
    preview_request_id: str = ""
    preview_text: str = ""
    # Stopped by this channel but not yet terminal: the Agent may not have seen
    # the stop when it publishes the preview, so the channel drops it itself.
    stopped_request_id: str = ""


def _normalize(text: str) -> str:
    return normalize_trigger_text(str(text))


def _new_session_id() -> str:
    return f"voice-session-{uuid.uuid4().hex[:12]}"


class VoiceBridgeCore:
    """Protocol state machine for one voice channel; not thread-safe."""

    def __init__(
        self,
        config: VoiceBridgeConfig,
        *,
        id_factory: Callable[[], str] | None = None,
        checks: Iterable[AnchorCheck] | None = None,
    ) -> None:
        self.config = config
        self._state = _BridgeState()
        self._feedback = FeedbackTracker()
        self._id_factory = id_factory or (lambda: f"voice-{uuid.uuid4().hex[:12]}")
        self._gate = AddresseeGate(
            trigger=config.addressee_trigger,
            wake_phrases=config.wake_phrases,
            wake_window_sec=config.wake_window_sec,
            session_gap_sec=config.session_gap_sec,
            checks=config.anchor_checks() if checks is None else checks,
        )
        #: Transcripts held for their measurement, in arrival order (``on_heard``).
        self._held: list[_HeldTranscript] = []
        #: L3 semantic gate, or None when off/unloadable; ``semantic_status`` says which and why,
        #: so the node can report a model that silently disabled itself.
        self._semantic, self.semantic_status = SemanticGate.from_config(config)

    @property
    def active_request_id(self) -> str:
        return self._state.active_request_id

    @property
    def awaiting_confirmation(self) -> bool:
        return self._state.pending is not None

    @property
    def interruptible(self) -> bool:
        """True while a stop keyword still has a target, so the microphone must stay open."""
        return bool(self._stop_target())

    def is_stop_text(self, text: str) -> bool:
        """True when ``text`` contains a stop keyword (after the domain corrections)."""
        return self._contains_stop(_normalize(self.config.correct(str(text).strip())))

    def on_heard(self, text: str, now: float, segment: Segment | None = None) -> list[Action]:
        """Route a final transcript now, or hold it until the measurement it will be judged by lands.

        ``segment`` is the speech segment the transcript came from
        (``SpeechTranscript``); the gate pairs it with that segment's
        measurements and wake event. Only a transcript the gate would compare
        against an anchor while its own measurement is still on its way is
        held; it is routed the moment that measurement lands
        (``on_measurement``), and ``measurement_grace_sec`` only bounds the wait
        (``tick``). Anything behind a held transcript queues, so the order is kept.

        A stop keyword is never held -- safety outruns addressing -- and it
        cancels every transcript still held: those were said before it, and
        letting one go out afterwards would move the robot right after the user
        told it to stop. Matching is by keyword, so an utterance like
        "不停地挥手" cancels too (the same trade-off as for stopping).
        """
        if self.config.feedback_only:
            return []
        if self.is_stop_text(text):
            cancelled = self._cancel_held(f"stop keyword in {str(text).strip()!r}")
            return [*cancelled, *self.on_transcript(text, now, segment=segment, cancels_held=bool(cancelled))]
        if self._held or self.should_hold(text, now, segment):
            self._held.append(_HeldTranscript(str(text), now, now + self.config.measurement_grace_sec, segment))
            return self.release_held(now)
        return self.on_transcript(text, now, segment=segment)

    def release_held(self, now: float) -> list[Action]:
        """Route held transcripts, in order, as soon as each no longer needs to wait."""
        actions: list[Action] = []
        while self._held:
            held = self._held[0]
            if now < held.deadline and self.should_hold(held.text, held.heard_at, held.segment):
                break
            self._held.pop(0)
            actions += self.on_transcript(held.text, now, heard_at=held.heard_at, segment=held.segment)
        return actions

    def on_transcript(
        self,
        text: str,
        now: float,
        *,
        heard_at: float | None = None,
        segment: Segment | None = None,
        cancels_held: bool = False,
    ) -> list[Action]:
        """Route one final transcript.

        ``heard_at`` is when it arrived, for one routed after being held; the
        gate judges it as of then. ``segment`` is the speech segment it came
        from. ``cancels_held`` marks a stop keyword that just cancelled held
        transcripts, so it is a stop even with nothing running yet.
        """
        if self.config.feedback_only:
            return []
        heard_at = now if heard_at is None else heard_at
        stopped_by_partial, self._state.utterance_stopped = self._state.utterance_stopped, False
        heard = str(text).strip()
        raw = self.config.correct(heard)
        normalized = _normalize(raw)
        # Every transcript, before any branch below can return: the segment a wake
        # event names opens the session and re-anchors whatever it turns out to be
        # -- a stop already sent from its partial, empty text, a stop, an answer --
        # and the wake is spent on it instead of passing to whoever speaks next.
        session_opened = self._gate.open_if_new_session(now, heard_at=heard_at, segment=segment)
        if not normalized:
            return []
        corrected = [Log(f"corrected transcript {heard!r} -> {raw!r}")] if raw != heard else []
        if stopped_by_partial and self._contains_stop(normalized):
            # Once the stop has finished the request, this sentence would otherwise reach the
            # Agent as a new command and occupy the channel ("帮我停一下" -> busy for "继续").
            return [*corrected, Log(f"final transcript {raw!r} of a stop already sent from its partial")]
        actions = self._route_transcript(
            raw, normalized, now, heard_at, segment, session_opened=session_opened, cancels_held=cancels_held
        )
        return corrected + actions if actions else corrected

    def on_partial(self, text: str, now: float) -> list[Action]:
        """Stop as soon as a partial transcript contains a stop keyword; ignore everything else.

        Final transcripts only arrive after VAD silence plus tail padding, which is
        seconds of extra motion when the user says "停". The stop also cancels
        every transcript still held for its measurement (see ``on_heard``); when
        that is all it stopped, its final transcript belongs to it as well.
        """
        if self.config.feedback_only or not self._is_stop_partial(text) or self._state.utterance_stopped:
            return []
        cancelled = self._cancel_held(f"stop keyword in partial {text!r}")
        if not self._stop_target():
            self._state.utterance_stopped = bool(cancelled)
            return cancelled
        self._state.utterance_stopped = True
        return [*cancelled, *self._stop(now, reason=f"stop keyword in partial {text!r}")]

    def required_measurements(self) -> tuple[str, ...]:
        """Measurement names the enabled checks need the node to feed."""
        return self._gate.required_measurements()

    def on_wake(self, now: float, segment: Segment | None = None) -> list[Action]:
        """The audio source reports that its wake word opened ``segment`` (see ``AddresseeGate.on_wake``).

        A transcript held for this wake may now be decidable, so it is routed here.
        """
        if not self._gate.on_wake(now, segment):
            return [Log(f"wake event for segment {segment} ignored: its transcript was already routed")]
        return self.release_held(now)

    def on_measurement(self, name: str, value: Any, now: float, segment: Segment | None = None) -> list[Action]:
        """Record one non-text observation of ``segment`` (see ``AddresseeGate.on_measurement``).

        The held transcript of that segment may now be decidable, so it is routed here.
        """
        self._gate.on_measurement(name, value, now, segment)
        return self.release_held(now)

    def should_hold(self, text: str, now: float, segment: Segment | None = None) -> bool:
        """True when routing ``text``, heard at ``now``, would judge it before what it needs has arrived.

        A stop and a confirmation answer never reach the gate, so they never
        wait; everything else waits only when the gate would compare it against
        an anchor while the measurement it needs is still on its way, or would
        reject it outside a session while its own wake event may still land
        (see ``AddresseeGate.needs_wait``).
        """
        normalized = _normalize(self.config.correct(str(text).strip()))
        if not normalized or self._contains_stop(normalized) or self._state.pending is not None:
            return False
        return self._gate.needs_wait(normalized, now, segment)

    def _cancel_held(self, reason: str) -> list[Action]:
        cancelled, self._held = self._held, []
        return [Log(f"{reason}: cancelled {held.text!r}, held for its measurement") for held in cancelled]

    def _is_stop_partial(self, text: str) -> bool:
        """A partial that is itself a stop keyword, or a prefix on its way to one.

        Partial matching must not be "contains": "停止照镜子" grows through the
        partial "停止", and stopping on that cancels the very request that stops
        the mirror. Keyword matching still applies to the final transcript,
        after tail padding.
        """
        normalized = _normalize(self.config.correct(str(text).strip()))
        if not normalized:
            return False
        if normalized in self.config.stop_phrases:
            return True
        return any(phrase.startswith(normalized) for phrase in self.config.stop_phrases)

    def _silent(self, request_id: str) -> bool:
        return bool(request_id) and request_id == self._state.silent_request_id

    def _is_silent_presentation(self, presentation: dict[str, Any]) -> bool:
        """True when every step of the plan is a skill whose output is speech itself."""
        if not self.config.silent_skills:
            return False
        steps = presentation.get("steps")
        if not isinstance(steps, list) or not steps:
            return False
        names = [
            _normalize(step.get("skill_name", ""))
            for step in steps
            if isinstance(step, dict) and step.get("skill_name")
        ]
        if not names:
            return False
        silent = {_normalize(name) for name in self.config.silent_skills}
        return all(name in silent for name in names)

    def _contains_stop(self, normalized: str) -> bool:
        # Keyword, not whole-sentence, matching: "帮我停一下" and "别动了谢谢" must interrupt.
        # Accepted trade-off: an utterance like "不停地挥手" also stops the robot.
        return any(phrase in normalized for phrase in self.config.stop_phrases)

    def _judge_semantics(
        self, raw: str, normalized: str, now: float, *, session_opened: bool
    ) -> list[Action] | None:
        """Run the L3 gate; returns the actions that stop the utterance, or None to let it through.

        Shadow mode records the verdict and returns None, which is what makes the log a
        labelled sample rather than a sample of rejections only: its false negatives (a
        command the gate would have dropped) are exactly what the next threshold has to
        account for.
        """
        gate = self._semantic
        if gate is None:
            return None
        # A session the addressee gate opened (wake word or vendor wake event) is what
        # lets a bare fragment through: "怎么" alone is a person mid-conversation, but
        # only the session says which conversation it belongs to.
        session_open = bool(self._state.session_id) and (
            now - self._state.last_activity <= self.config.session_idle_sec
        )
        decision = gate.judge(raw, session_open=session_open)
        mode = self.config.semantic_gate_mode
        gate.record(
            decision,
            self.config.semantic_gate_log_path,
            mode=mode,
            session_id=self._state.session_id,
            after_wake=bool(session_opened),
            active_request=bool(self._state.active_request_id),
            normalized=normalized,
        )
        if not decision.rejected or mode != "enforce":
            return None
        # Same wording as the addressee verdict on purpose: "not addressed to me" is one
        # concept, whichever layer decided it, and the shadow log carries the score for
        # the ones that only look that way.
        return [Log(f"not addressed to me (semantic {decision.score:.2f}: {decision.reason}); ignored {raw!r}")]

    def _route_transcript(
        self,
        raw: str,
        normalized: str,
        now: float,
        heard_at: float,
        segment: Segment | None,
        *,
        session_opened: bool,
        cancels_held: bool = False,
    ) -> list[Action]:
        state = self._state
        stops_something = cancels_held or self._stop_target() or normalized in self.config.stop_phrases
        if self._contains_stop(normalized) and stops_something:
            # With nothing to interrupt, a sentence that merely contains a keyword is a command.
            # Checked before the wake gate on purpose: a stop must never need addressing.
            return self._stop(now, reason=f"stop keyword in {raw!r}")
        if state.pending is not None:
            return self._answer_confirmation(normalized, raw, now)
        verdict = self._gate.judge(
            raw, normalized, now, session_opened=session_opened, heard_at=heard_at, segment=segment
        )
        if not verdict.addressed:
            return [Log(f"not addressed to me ({verdict.reason}); ignored {raw!r}")]
        raw, normalized = verdict.raw, verdict.normalized
        semantic_block = self._judge_semantics(raw, normalized, now, session_opened=session_opened)
        if semantic_block is not None:
            return semantic_block
        if any(phrase in normalized for phrase in self.config.bypass_phrases):
            return [Log(f"ignored bypass utterance {raw!r}")]
        if state.active_request_id:
            return [Log(f"busy with {state.active_request_id}; dropped {raw!r}"), *self._say(self.config.busy_message)]
        if len(normalized) < self.config.min_text_chars:
            return [Log(f"ignored short utterance {raw!r}")]
        self._gate.note_handled(heard_at)

        if not state.session_id or now - state.last_activity > self.config.session_idle_sec:
            state.session_id = _new_session_id()
            state.clarification_request_id = ""
        request_id = self._id_factory()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "session_id": state.session_id,
            "text": raw,
        }
        if state.clarification_request_id:
            payload["reply_to_request_id"] = state.clarification_request_id
            state.clarification_request_id = ""
        state.active_request_id = request_id
        state.active_since = now
        state.last_activity = now
        state.known_request_ids.add(request_id)
        return [Log(f"request {request_id}: {raw!r}"), PublishRequest(payload)]

    def on_response(self, payload: dict[str, Any], now: float) -> list[Action]:
        if self.config.feedback_only:
            return []
        request_id = str(payload.get("request_id", ""))
        if request_id not in self._state.known_request_ids:
            return []
        self._state.last_activity = now
        if payload.get("accepted") is False or payload.get("state") in TERMINAL_REQUEST_STATES:
            # A rejected admission is also emitted as a terminal event; speak only once.
            if not self._finish(request_id):
                return []
            message = str(payload.get("message", "")).strip()
            reason = payload.get("reason_code", "")
            spoken = self.config.busy_message if reason == "BUSY" else _spoken_reply(payload.get("state"), message)
            return [Log(f"request {request_id} not admitted: {reason} {message}"), *self._say(spoken)]
        return []

    def on_event(self, payload: dict[str, Any], now: float) -> list[Action]:
        key = payload.get("request_key")
        if self.config.feedback_only:
            if not isinstance(key, dict) or not all(
                key.get(name) == getattr(self.config, name) for name in ("channel_id", "principal_id", "robot_scope")
            ):
                return []
            return self._execution_speech(payload)
        if not isinstance(key, dict) or not self._owns(key):
            return []
        request_id = str(key["request_id"])
        self._state.last_activity = now
        event_type = payload.get("event_type")
        state = payload.get("state")
        if event_type == "presentation":
            return self._on_presentation(payload, request_id, now)
        if event_type == "continuation":
            # The Agent resolved "继续" against its ledgers; tell the user what resumes.
            message = str(payload.get("user_message", "")).strip()
            return [Log(f"request {request_id} continuation: {message}"), *self._say(message)]
        if event_type == "reply_preview":
            return self._on_reply_preview(payload, request_id)
        if state not in TERMINAL_REQUEST_STATES:
            return self._execution_speech(payload)
        silent = self._silent(request_id)
        previewed = self._take_preview(request_id)
        if not self._finish(request_id):
            return []
        if event_type == "needs_clarification":
            self._state.clarification_request_id = request_id
        message = str(payload.get("user_message", "")).strip()
        if isinstance(payload.get("detail"), dict) and payload["detail"].get("execution_id"):
            return self._execution_speech(payload)
        spoken = "" if silent else _spoken_reply(state, message)
        log = Log(f"request {request_id} {event_type} {state}: {message}")
        if previewed and spoken == previewed:
            return [log, Log(f"request {request_id} reply was already said as its preview")]
        return [log, *self._say(spoken)]

    def _execution_speech(self, payload: dict[str, Any]) -> list[Action]:
        if payload.get("event_type") not in {"execution_progress", "terminal", "quarantined", "failed", "cancelled"}:
            return []
        accepted = self._feedback.accept(payload)
        if accepted is None or not self.config.speak_replies:
            return []
        execution_id, text, terminal = accepted
        if self._silent(str(payload.get("request_id", ""))):
            return [Log(f"request {payload.get('request_id')} speech suppressed: its output is speech itself")]
        return [Speak(text, execution_id=execution_id, terminal=terminal)]

    def _on_reply_preview(self, payload: dict[str, Any], request_id: str) -> list[Action]:
        """Say the reply the planner is still producing, once per request.

        The plan behind it is not validated yet: if it is refused, the terminal
        event says so ("执行失败") instead of staying silent about it.
        """
        message = str(payload.get("user_message", "")).strip()
        if (
            request_id != self._state.active_request_id
            or self._state.preview_request_id == request_id
            or self._state.stopped_request_id == request_id
        ):
            return [Log(f"reply preview for {request_id} ignored")]
        if not message:
            return []
        if not self.config.speak_reply_previews:
            self._state.preview_request_id = request_id
            return [Log(f"request {request_id} reply preview held silent: {message}")]
        self._state.preview_request_id = request_id
        self._state.preview_text = message
        return [Log(f"request {request_id} reply preview: {message}"), *self._say(message)]

    def _take_preview(self, request_id: str) -> str:
        if self._state.preview_request_id != request_id:
            return ""
        text = self._state.preview_text
        self._state.preview_request_id = ""
        self._state.preview_text = ""
        return text

    def tick(self, now: float) -> list[Action]:
        state = self._state
        # A held transcript whose grace period ran out without a measurement.
        actions: list[Action] = self.release_held(now)
        if state.pending is not None and now >= state.pending.deadline:
            request_id = state.pending.request_id
            state.pending = None
            actions += [
                Log(f"confirmation for {request_id} timed out"),
                PublishControl({"operation": "stop", "request_id": request_id}),
                *self._say("确认超时，已取消。"),
            ]
        if state.active_request_id and now - state.active_since > self.config.request_timeout_sec:
            # Events are best-effort topics; never let a lost terminal event wedge the channel.
            # Releasing alone would leave the robot moving with no voice request to stop, so
            # stop it first and keep it as the stop target until its terminal event arrives.
            request_id = state.active_request_id
            state.active_request_id = ""
            if state.pending is not None and state.pending.request_id == request_id:
                state.pending = None
            state.timed_out_request_id = request_id
            actions += [
                Log(f"request {request_id} produced no terminal event; stopping and releasing channel"),
                PublishControl({"operation": "stop", "request_id": request_id}),
                # Still a stop target while this plays: the reply must not contain a stop word.
                *self._say("任务超时，正在中止。"),
            ]
        return actions

    def _owns(self, key: dict[str, Any]) -> bool:
        return (
            key.get("request_id") in self._state.known_request_ids
            and key.get("channel_id") == self.config.channel_id
            and key.get("principal_id") == self.config.principal_id
            and key.get("robot_scope") == self.config.robot_scope
        )

    def _on_presentation(self, payload: dict[str, Any], request_id: str, now: float) -> list[Action]:
        if request_id != self._state.active_request_id:
            # A late presentation for a finished or timed-out request must not unlock execution
            # the channel no longer tracks.
            return [Log(f"presentation for inactive request {request_id}; not acknowledging")]
        detail = payload.get("detail")
        if not isinstance(detail, dict):
            return [Log(f"presentation for {request_id} has no detail; not acknowledging")]
        presentation = detail.get("presentation")
        if not isinstance(presentation, dict) or presentation_digest(presentation) != detail.get("presentation_digest"):
            return [Log(f"presentation digest mismatch for {request_id}; not acknowledging")]
        summary = _presentation_summary(presentation, payload.get("user_message", ""))
        plan_log = Log("exact plan " + json.dumps(presentation, ensure_ascii=False, sort_keys=True))
        if self._is_silent_presentation(presentation):
            # The clip is the reply: announcing the plan and then saying "执行完成。" would talk over it.
            self._state.silent_request_id = request_id
            return [plan_log]
        if detail.get("async_display") is True or presentation.get("execution_mode") == ASYNC_PRESENTATION_MODE:
            # Typed Programs execute without waiting for any client: there is no receipt to give, and
            # asking for a confirmation that cannot hold the robot back would mislead the user.
            gate = (
                [Log(f"voice_confirm cannot gate {request_id}: the Agent executes typed Programs without a receipt")]
                if self.config.confirm_mode == "voice_confirm"
                else []
            )
            if self._state.preview_request_id == request_id:
                # The preview already told the user what is about to happen, and nothing waits on
                # this line; saying it again only delays the answer behind it.
                return [plan_log, *gate, Log(f"request {request_id} plan was already announced by its preview")]
            return [plan_log, *gate, *self._say(f"正在执行：{summary}。")]
        acknowledgement = PublishControl(
            {
                "operation": "presentation_rendered",
                "request_id": request_id,
                "request_key": payload["request_key"],
                "presentation_digest": detail["presentation_digest"],
                "receipt_token": detail.get("receipt_token", ""),
            }
        )
        if self.config.confirm_mode == "auto":
            return [plan_log, Speak(f"准备执行：{summary}。", then=(acknowledgement,))]
        self._state.pending = _PendingConfirmation(
            request_id=request_id,
            acknowledgement=acknowledgement,
            deadline=now + self.config.confirm_timeout_sec,
        )
        return [plan_log, Speak(f"准备执行：{summary}。请说确认或取消。")]

    def _answer_confirmation(self, normalized: str, raw: str, now: float) -> list[Action]:
        pending = self._state.pending
        assert pending is not None
        if any(phrase in normalized for phrase in self.config.cancel_phrases):
            return self._stop(now, reason=f"cancel phrase {raw!r}")
        if any(marker in normalized for marker in _NEGATION_MARKERS):
            # Keyword matching would read "不执行" as consent; a negated answer asks again instead.
            return [Log(f"negated answer {raw!r} is not a confirmation"), *self._say("请说确认或取消。")]
        if any(phrase in normalized for phrase in self.config.confirm_phrases):
            self._state.pending = None
            return [Log(f"confirmed {pending.request_id} by {raw!r}"), pending.acknowledgement]
        return [Log(f"awaiting confirmation; ignored {raw!r}"), *self._say("请说确认或取消。")]

    def _stop_target(self) -> str:
        state = self._state
        if state.pending is not None:
            return state.pending.request_id
        return state.active_request_id or state.timed_out_request_id

    def _stop(self, now: float, *, reason: str) -> list[Action]:
        state = self._state
        target = self._stop_target()
        state.pending = None
        state.last_activity = now
        if not target:
            return [Log(f"{reason}: no active request")]
        state.stopped_request_id = target
        return [Log(f"{reason}: stopping {target}"), PublishControl({"operation": "stop", "request_id": target})]

    def _finish(self, request_id: str) -> bool:
        """Release ``request_id``; return False when it was already finished."""
        if request_id in self._state.finished_request_ids:
            return False
        if self._state.silent_request_id == request_id:
            self._state.silent_request_id = ""
        self._state.finished_request_ids.add(request_id)
        if self._state.active_request_id == request_id:
            self._state.active_request_id = ""
        if self._state.timed_out_request_id == request_id:
            self._state.timed_out_request_id = ""
        if self._state.stopped_request_id == request_id:
            self._state.stopped_request_id = ""
        if self._state.pending is not None and self._state.pending.request_id == request_id:
            self._state.pending = None
        return True

    def _say(self, text: str) -> list[Action]:
        if not text or not self.config.speak_replies:
            return []
        return [Speak(text)]


def _spoken_reply(state: Any, message: str) -> str:
    """Execution terminals carry technical text; answers are already user-facing."""
    return execution_reply(str(state), fallback=message)


def _presentation_summary(presentation: dict[str, Any], fallback: Any) -> str:
    summary = presentation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return str(fallback or "").strip() or "当前计划"


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def resolve_confirm_mode(configured: str, *, tts_enabled: bool) -> str:
    """Pick the confirmation mode, defaulting to an explicit spoken confirmation without TTS.

    Only the legacy Agent Plan path waits for the receipt this decides; typed Programs execute
    without one. ``auto`` acknowledges the presentation once the exact plan has been rendered.
    Without TTS the only rendering is the log line, so an unset mode falls back to ``voice_confirm``
    and a human act gates that path; ``auto`` without TTS must be chosen explicitly (see the README).
    """
    if configured:
        return configured
    return "auto" if tts_enabled else "voice_confirm"


def node_parameter_defaults() -> dict[str, Any]:
    """ROS parameter defaults: transport names plus every ``VoiceBridgeConfig`` default, declared once."""
    defaults: dict[str, Any] = {
        "input_topic": "/voice_command",
        # voice_asr_node publishes/listens on these fixed names; an empty partial_topic disables
        # the realtime (partial-transcript) stop.
        "partial_topic": "/voice_partial",
        "asr_status_topic": "/voice_status",
        "asr_control_topic": "/voice_control",
        "request_topic": "/agent/request",
        "response_topic": "/agent/response",
        "event_topic": "/agent/event",
        "control_topic": "/agent/control",
        "tts_enabled": False,
        "tts_synthesis_service": "/voice_tts/synthesize",
        "tts_playback_service": "/voice_tts/play",
        "tts_timeout_sec": 60.0,
        # Empty: the launch builder derives it from voice_asr.speaker_embedding.output_topic,
        # so the producer's topic is the only copy. Required when the speaker check is on.
        "speaker_embedding_topic": "",
        # Empty: no wake events. Only read under addressee_trigger=vendor_session. The launch
        # builder fills it from the bound voice_bridge.wake_event interface, never from YAML.
        "wake_event_topic": "",
        # Empty: transcripts are plain text on input_topic. The launch builder fills it from
        # voice_asr.transcript_topic when the gate pairs by segment (VoiceBridgeConfig.pairs_by_segment);
        # the bridge then reads the segment-tagged transcripts there instead.
        "transcript_topic": "",
        "mute_asr_during_tts": True,
        "resume_settle_sec": 1.0,
    }
    for item in dataclasses.fields(VoiceBridgeConfig):
        if item.name in _PHRASE_FIELDS:
            defaults[f"{item.name}_json"] = json.dumps(list(item.default), ensure_ascii=False)
        elif item.name == "text_corrections":
            defaults["text_corrections_json"] = json.dumps(dict(item.default), ensure_ascii=False)
        elif item.name == "confirm_mode":
            # Empty: resolve_confirm_mode derives it from tts_enabled.
            defaults["confirm_mode"] = ""
        else:
            defaults[item.name] = item.default
    return defaults


def config_from_parameters(values: dict[str, Any]) -> VoiceBridgeConfig:
    return VoiceBridgeConfig(
        channel_id=str(values["channel_id"]),
        principal_id=str(values["principal_id"]),
        robot_scope=str(values["robot_scope"]),
        stop_phrases=_string_tuple(values["stop_phrases_json"], "stop_phrases"),
        bypass_phrases=_string_tuple(values["bypass_phrases_json"], "bypass_phrases"),
        confirm_mode=resolve_confirm_mode(str(values["confirm_mode"]), tts_enabled=bool(values["tts_enabled"])),
        confirm_phrases=_string_tuple(values["confirm_phrases_json"], "confirm_phrases"),
        cancel_phrases=_string_tuple(values["cancel_phrases_json"], "cancel_phrases"),
        silent_skills=_string_tuple(values["silent_skills_json"], "silent_skills"),
        wake_phrases=_string_tuple(values["wake_phrases_json"], "wake_phrases"),
        addressee_trigger=str(values["addressee_trigger"]),
        session_gap_sec=float(values["session_gap_sec"]),
        wake_window_sec=float(values["wake_window_sec"]),
        speaker_check_enabled=bool(values["speaker_check_enabled"]),
        speaker_min_similarity=float(values["speaker_min_similarity"]),
        speaker_max_age_sec=float(values["speaker_max_age_sec"]),
        speaker_merge_margin=float(values["speaker_merge_margin"]),
        speaker_reject_short=bool(values["speaker_reject_short"]),
        measurement_grace_sec=float(values["measurement_grace_sec"]),
        confirm_timeout_sec=float(values["confirm_timeout_sec"]),
        session_idle_sec=float(values["session_idle_sec"]),
        min_text_chars=int(values["min_text_chars"]),
        request_timeout_sec=float(values["request_timeout_sec"]),
        speak_replies=bool(values["speak_replies"]),
        speak_reply_previews=bool(values["speak_reply_previews"]),
        feedback_only=bool(values["feedback_only"]),
        busy_message=str(values["busy_message"]),
        text_corrections=_corrections(values.get("text_corrections_json", "{}")),
        semantic_gate_mode=str(values["semantic_gate_mode"]),
        semantic_gate_model_path=str(values["semantic_gate_model_path"]),
        semantic_gate_threshold=float(values["semantic_gate_threshold"]),
        semantic_gate_lexicon=_string_tuple(values["semantic_gate_lexicon_json"], "semantic_gate_lexicon"),
        semantic_gate_wake_prefixes=_string_tuple(
            values["semantic_gate_wake_prefixes_json"], "semantic_gate_wake_prefixes"
        ),
        semantic_gate_log_path=str(values["semantic_gate_log_path"]),
        semantic_gate_fail_open=bool(values["semantic_gate_fail_open"]),
    )


def _corrections(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("text_corrections must be a mapping of term -> list of misrecognized variants")
    return tuple((str(term), _string_tuple(variants, f"text_corrections[{term}]")) for term, variants in value.items())


class _Speaker:
    """Serialize TTS synthesize+play, muting ASR so the robot does not hear itself."""

    def __init__(self, node: Any, *, synthesis_service: str, playback_service: str, timeout_sec: float) -> None:
        from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech

        self._node = node
        self._timeout_sec = timeout_sec
        self._synthesis_type = SynthesizeSpeech
        self._playback_type = PlayAudioFile
        self._synthesize = node.create_client(SynthesizeSpeech, synthesis_service)
        self._playback = node.create_client(PlayAudioFile, playback_service)
        self._lock = threading.Lock()
        self._closing = threading.Event()

    def close(self) -> None:
        self._closing.set()

    def speak(self, text: str) -> bool:
        with self._lock:
            if not self._synthesize.wait_for_service(timeout_sec=2.0) or not self._playback.wait_for_service(
                timeout_sec=2.0
            ):
                self._node.get_logger().error("TTS services are unavailable")
                return False
            request = self._synthesis_type.Request()
            request.text = text
            result = self._call(self._synthesize, request)
            if result is None or not result.success or not result.audio_segments:
                self._node.get_logger().error(f"TTS synthesis failed: {getattr(result, 'message', 'timeout')}")
                return False
            for segment in result.audio_segments:
                descriptor, path = tempfile.mkstemp(prefix="ibrobot-voice-bridge-", suffix=".wav")
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(bytes(segment.audio_data))
                    play = self._playback_type.Request()
                    play.file_path = path
                    played = self._call(self._playback, play)
                    if played is None or not played.success:
                        self._node.get_logger().error(f"TTS playback failed: {getattr(played, 'message', 'timeout')}")
                        return False
                finally:
                    os.unlink(path)
            return True

    def _call(self, client: Any, request: Any) -> Any:
        future = client.call_async(request)
        deadline = time.monotonic() + self._timeout_sec
        while not future.done() and time.monotonic() < deadline and not self._closing.is_set():
            time.sleep(0.02)
        return future.result() if future.done() else None


def _build_node_class():
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from std_msgs.msg import String

    class VoiceBridgeNode(Node):
        """ROS transport for :class:`VoiceBridgeCore`."""

        def __init__(self, **node_options) -> None:
            super().__init__("agent_voice_bridge", **node_options)
            defaults = node_parameter_defaults()
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            values = {name: self.get_parameter(name).value for name in defaults}
            self._core = VoiceBridgeCore(config_from_parameters(values))
            self._core_lock = threading.Lock()
            self._asr_state = ""
            self._mute_asr = bool(values["mute_asr_during_tts"])
            self._speaker = (
                _Speaker(
                    self,
                    synthesis_service=str(values["tts_synthesis_service"]),
                    playback_service=str(values["tts_playback_service"]),
                    timeout_sec=float(values["tts_timeout_sec"]),
                )
                if values["tts_enabled"]
                else None
            )
            self._speech_queue: list[Speak] = []
            self._speech_wakeup = threading.Condition()
            self._speech_closed = False
            # Muting ASR still lets it publish the utterance it had already started, and that text
            # would be routed as a new command (possibly re-triggering this very reply). Ignore
            # transcripts until the ASR resume has settled.
            self._deaf_until = 0.0
            # While a request can be interrupted ASR is not muted; until then only stop words pass,
            # so the robot's own reply is not routed as a command.
            self._echo_until = 0.0
            self._resume_settle_sec = float(values["resume_settle_sec"])

            inbound = MutuallyExclusiveCallbackGroup()
            self._request_publisher = self.create_publisher(String, str(values["request_topic"]), 10)
            self._control_publisher = self.create_publisher(String, str(values["control_topic"]), 10)
            self._asr_control_publisher = self.create_publisher(String, str(values["asr_control_topic"]), 10)
            if self._core.config.feedback_only:
                transcript_topic = "feedback only"
            elif self._core.config.pairs_by_segment:
                from ibrobot_msgs.msg import SpeechTranscript

                if not values["transcript_topic"]:
                    # Plain text names no segment: every check would fail open and no
                    # wake event could ever open a session.
                    raise ValueError(
                        "the addressee gate pairs transcripts by speech segment and needs transcript_topic; "
                        "the embodied launch derives it from voice_asr.transcript_topic"
                    )
                transcript_topic = str(values["transcript_topic"])
                self.create_subscription(
                    SpeechTranscript, transcript_topic, self._on_segment_transcript, 10, callback_group=inbound
                )
            else:
                transcript_topic = str(values["input_topic"])
                self.create_subscription(String, transcript_topic, self._on_transcript, 10, callback_group=inbound)
            self.create_subscription(
                String, str(values["response_topic"]), self._on_response, 10, callback_group=inbound
            )
            self.create_subscription(String, str(values["event_topic"]), self._on_event, 10, callback_group=inbound)
            if values["partial_topic"] and not self._core.config.feedback_only:
                self.create_subscription(
                    String, str(values["partial_topic"]), self._on_partial, 10, callback_group=inbound
                )
            self.create_subscription(
                String, str(values["asr_status_topic"]), self._on_asr_status, 10, callback_group=inbound
            )
            if "speaker" in self._core.required_measurements():
                from ibrobot_msgs.msg import SpeakerEmbedding

                if not values["speaker_embedding_topic"]:
                    # Listening nowhere would accept every voice while reading as checked.
                    raise ValueError(
                        "speaker_check_enabled needs speaker_embedding_topic; the embodied launch "
                        "derives it from voice_asr.speaker_embedding.output_topic"
                    )

                self.create_subscription(
                    SpeakerEmbedding,
                    str(values["speaker_embedding_topic"]),
                    self._on_speaker_embedding,
                    10,
                    callback_group=inbound,
                )
            if self._core.config.addressee_trigger == "vendor_session":
                from ibrobot_msgs.msg import SpeechWakeEvent

                if not values["wake_event_topic"]:
                    # Only a wake event opens a session: without one the robot hears nobody.
                    raise ValueError(
                        "addressee_trigger=vendor_session needs wake_event_topic; bind "
                        "voice_bridge.wake_event to the runtime's audio.wake_event interface"
                    )
                self.create_subscription(
                    SpeechWakeEvent, str(values["wake_event_topic"]), self._on_wake_event, 10, callback_group=inbound
                )
            self.create_timer(0.2, self._on_tick, callback_group=inbound)
            self._speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
            self._speech_thread.start()
            confirm_mode = self._core.config.confirm_mode
            self.get_logger().info(
                f"voice bridge ready: {transcript_topic} -> {values['request_topic']} "
                f"(channel={values['channel_id']}, confirm_mode={confirm_mode}, "
                f"tts={'on' if self._speaker else 'off'})"
            )
            self.get_logger().info(self._core.semantic_status)
            if self._speaker is None:
                self.get_logger().warning(
                    "TTS is off: each exact plan is rendered only to this log; nobody hears it before the robot moves"
                )

        def _on_transcript(self, message: String) -> None:
            self._hear(message.data, None)

        def _on_segment_transcript(self, message: Any) -> None:
            self._hear(str(message.text), (str(message.source_id), int(message.segment_sequence)))

        def _hear(self, text: str, segment: Segment | None) -> None:
            if self._core.config.feedback_only:
                return
            now = time.monotonic()
            if now < self._deaf_until or (now < self._echo_until and not self._core.is_stop_text(text)):
                self.get_logger().info(f"ignored transcript heard during playback: {text!r}")
                return
            # The core holds a transcript that is still waiting for its measurement.
            self._run(lambda now: self._core.on_heard(text, now, segment))

        def _on_partial(self, message: String) -> None:
            if time.monotonic() < self._deaf_until:
                return
            self._run(lambda now: self._core.on_partial(message.data, now))

        def _on_response(self, message: String) -> None:
            if self._core.config.feedback_only:
                return
            payload = self._decode(message.data, "response")
            if payload is not None:
                self._run(lambda now: self._core.on_response(payload, now))

        def _on_event(self, message: String) -> None:
            payload = self._decode(message.data, "event")
            if payload is not None:
                self._run(lambda now: self._core.on_event(payload, now))

        def _on_wake_event(self, message: Any) -> None:
            segment = (str(message.source_id), int(message.segment_sequence))
            self._run(lambda now: self._core.on_wake(now, segment))

        def _on_speaker_embedding(self, message: Any) -> None:
            # Parks the voiceprint for the transcript of its segment (see
            # AddresseeGate.on_measurement); an empty one says the segment had
            # none, so its transcript need not wait. Either way that transcript,
            # if held, may now be decidable. The model identity travels with the
            # vector: only voiceprints of one model_id are compared.
            embedding = tuple(float(value) for value in message.embedding)
            voiceprint = Voiceprint(embedding, str(message.model_id)) if embedding else None
            segment = (str(message.source_id), int(message.segment_sequence))
            self._run(lambda now: self._core.on_measurement("speaker", voiceprint, now, segment))

        def _on_asr_status(self, message: String) -> None:
            self._asr_state = message.data

        def _on_tick(self) -> None:
            # Also releases transcripts whose grace period ran out without a measurement.
            self._run(self._core.tick)

        def _decode(self, data: str, kind: str) -> dict[str, Any] | None:
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                self.get_logger().warning(f"ignored malformed Agent {kind}")
                return None
            return payload if isinstance(payload, dict) else None

        def _run(self, step: Callable[[float], Iterable[Action]]) -> None:
            with self._core_lock:
                actions = list(step(time.monotonic()))
            self._execute(actions)

        def _execute(self, actions: Iterable[Any]) -> None:
            for action in actions:
                if isinstance(action, Log):
                    self.get_logger().info(action.message)
                elif isinstance(action, PublishRequest):
                    self._request_publisher.publish(String(data=json.dumps(action.payload, ensure_ascii=False)))
                elif isinstance(action, PublishControl):
                    self._control_publisher.publish(String(data=json.dumps(action.payload, ensure_ascii=False)))
                elif isinstance(action, Speak):
                    if self._speaker is None:
                        self.get_logger().info(f"[say] {action.text}")
                        self._execute(action.then)
                    else:
                        with self._speech_wakeup:
                            if action.terminal and action.execution_id:
                                self._speech_queue = [
                                    queued
                                    for queued in self._speech_queue
                                    if queued.execution_id != action.execution_id or queued.terminal
                                ]
                            self._speech_queue.append(action)
                            self._speech_wakeup.notify()

        def _speech_loop(self) -> None:
            while True:
                with self._speech_wakeup:
                    while not self._speech_queue and not self._speech_closed:
                        self._speech_wakeup.wait()
                    if self._speech_closed:
                        return
                    action = self._speech_queue.pop(0)
                try:
                    self._render_speech(action)
                except Exception as error:
                    # This thread is the only consumer of the speech queue. Letting one failed
                    # render end it would silence every later reply and withhold the plan
                    # acknowledgements chained behind them while the node still looks alive.
                    dropped = "; dependent actions dropped" if action.then else ""
                    self.get_logger().error(f"speech rendering failed: {error!r}{dropped}")

        def destroy_node(self):
            with self._speech_wakeup:
                self._speech_closed = True
                self._speech_queue.clear()
                self._speech_wakeup.notify_all()
            if self._speaker is not None:
                self._speaker.close()
            self._speech_thread.join(timeout=1.0)
            return super().destroy_node()

        def _render_speech(self, action: Speak) -> None:
            with self._core_lock:
                interruptible = self._core.interruptible
            # Muting ASR would also drop the stop keyword, so it is only done while nothing can move.
            # A reply that itself contains a stop word would stop the request it talks about, so it
            # still mutes (and says so).
            keep_listening = interruptible and not self._core.is_stop_text(action.text)
            if interruptible and not keep_listening:
                self.get_logger().warning(f"reply contains a stop word; ASR is muted while it plays: {action.text!r}")
            muted = self._mute_asr and not keep_listening and self._asr_state in _ACTIVE_ASR_STATES
            filter_echo = self._mute_asr and keep_listening
            if muted:
                self._asr_control_publisher.publish(String(data="stop"))
                self._deaf_until = float("inf")
            elif filter_echo:
                self._echo_until = float("inf")
            try:
                self.get_logger().info(f"[say] {action.text}")
                spoken = self._speaker.speak(action.text)
            finally:
                settled = time.monotonic() + self._resume_settle_sec
                if muted:
                    self._asr_control_publisher.publish(String(data="start"))
                    self._deaf_until = settled
                elif filter_echo:
                    self._echo_until = settled
            if spoken:
                self._execute(action.then)
            elif action.then:
                self.get_logger().error("speech was not rendered; dependent actions dropped")

    return VoiceBridgeNode


def main(args: list[str] | None = None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init(args=args)
    node = _build_node_class()()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
