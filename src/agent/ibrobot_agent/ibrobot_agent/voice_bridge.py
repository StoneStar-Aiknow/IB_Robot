"""Bridge final ASR transcripts into the incubating Agent request contract.

``voice_asr_node`` publishes final transcripts. This client wraps them into
Agent requests and sends stop controls immediately. Typed Programs display
plans asynchronously; only legacy Agent Plans wait for presentation receipts.
Synthesis and playback run as separate serial stages. Ordinary ready speech
is replaceable; dependent presentation receipts remain ordered.

``VoiceBridgeCore`` is ROS-free and returns transport actions so the protocol
logic is unit-testable; ``VoiceBridgeNode`` only executes those actions.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from embodied_common import mirror_session_contract as MIRROR_SESSION
from embodied_common.control_phrases import STOP_PHRASES
from embodied_common.recorded_clips import choose_clip_file, spoken_key
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
from ibrobot_agent.gift_workflow import GiftWorkflowSpeech
from ibrobot_agent.presentation import presentation_digest
from ibrobot_agent.semantic_gate import (
    GRADE_CONVERSATION,
    SEMANTIC_GATE_MODES,
    SemanticDecision,
    SemanticGate,
    correct_manipulation_transcript,
)
from ibrobot_agent.speaker_split import SpeakerSplit, TokenAlignment, WindowVoiceprints
from ibrobot_agent.speech_pipeline import SpeechPipeline
from ibrobot_agent.speech_policy import GraspSpeechPolicy, MirrorSpeechPolicy

CONFIRM_MODES = ("auto", "voice_confirm")
SPOKEN_LINE_EVENTS = (
    "gesture_start",
    "gesture_done",
    "busy",
    "busy_pick",
    "speech_stopped",
    "wake_ack",
    "wake_hint",
)
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
    "semantic_gate_dialogue_requests",
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
    # Require a paired voiceprint for every utterance, including stop/confirmation.
    require_verified_speech: bool = False
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
    #: One VAD segment can hold two voices: a bystander's remark and the owner's
    #: request said without a pause long enough for the VAD. With per-window
    #: voiceprints (``speaker_id_node``) and token times (the ASR), keep only the
    #: owner's words of such a segment (``speaker_split.py``). Off: the windows
    #: are still compared and logged, the transcript is used whole.
    speaker_split_enabled: bool = False
    #: A window at or above this is the owner, below the other line someone else,
    #: between them undecided. 2 s windows score lower than whole utterances, so
    #: both are calibrated on their own (see the package README).
    speaker_split_owner_similarity: float = 0.45
    speaker_split_other_similarity: float = 0.25
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
    persistent_session: bool = False
    mirror_session_enabled: bool = False
    min_text_chars: int = 2
    request_timeout_sec: float = 180.0
    speech_mode: str = "standard"
    speech_messages_json: str = "{}"
    gift_container_name: str = ""
    gift_container_label: str = "指定容器"
    speak_replies: bool = True
    # Consume execution feedback for this text channel without accepting speech commands.
    feedback_only: bool = False
    # Spoken while a request runs and the microphone stays open for stop words, so it must not
    # contain one (the bridge would hear itself and stop the robot).
    busy_message: str = "我正在执行上一个任务，请稍等。"
    # Domain term -> homophones the ASR model prefers (e.g. 夹爪 jiā zhǎo is heard as 家找/家长).
    text_corrections: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: ``(pattern, replacement)`` regular expressions applied after ``text_corrections``, for
    #: misrecognitions a plain substring would over-correct: "作为一个挥手的动作" is "做一个
    #: 挥手的动作", but "作为一个机器人你……" is not, so the rewrite is anchored to the shape.
    text_rewrites: tuple[tuple[str, str], ...] = ()
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
    #: Score a sentence with no skill word and no wake prefix must reach on its own. The
    #: low ``semantic_gate_threshold`` is a veto line, not an admission line: a character-bag
    #: model scores "清代你会过" 1.000 because it shares "你会" with "你会做什么", so a
    #: model-only decision needs a score that actually means something.
    semantic_gate_model_only_threshold: float = 0.90
    #: Skill words that always win over the model (the catalog's ``aliases_zh``): an unseen
    #: skill scores badly by construction, see the module docstring.
    semantic_gate_lexicon: tuple[str, ...] = ()
    #: Leading wake-word spellings the ASR produces (灵犀 heard as 林星/年轻人/...): a sentence
    #: that starts with one was said to the robot, whatever the model thinks of the rest.
    semantic_gate_dialogue_requests: tuple[str, ...] = ()
    semantic_gate_wake_prefixes: tuple[str, ...] = ()
    #: JSONL of every judgement (shadow and enforce alike). Empty: ROS log only.
    semantic_gate_log_path: str = ""
    #: A model that will not load disables the gate instead of failing the launch. The gate
    #: can only drop utterances it is confident about, so running without it is the safe side.
    semantic_gate_fail_open: bool = True
    #: Conversation admission (see ``semantic_gate``): inside a session, a second-person
    #: question, a reply inside the follow-up window and the trailing question of a glued
    #: segment are admitted without a skill word, as ``conversation``-graded requests the
    #: Agent answers and never executes. Off: the gate behaves as before it existed.
    semantic_gate_conversation: bool = False
    #: How long after the robot last spoke (its own speech ended, or a request of this
    #: channel finished) a reply-shaped sentence counts as an answer to it.
    semantic_gate_followup_sec: float = 20.0
    #: Lines the bridge says itself, by event, overriding its built-in wording (a deployment
    #: sets them to lines it has recorded, see ``speech_clips``): ``gesture_start`` and
    #: ``gesture_done`` (a gesture starts / succeeds), ``busy`` and ``busy_pick`` (a request
    #: arrives while another one, or a fruit delivery, runs), ``speech_stopped`` (after "别说了"),
    #: ``wake_ack`` (only the wake phrase was said) and ``wake_hint`` (the woken speaker talks
    #: after the window closed). An event without a line keeps the built-in behaviour; the two
    #: without one (``wake_ack``, ``wake_hint``, ``speech_stopped``) stay silent.
    spoken_lines_json: str = "{}"
    #: Fruit-delivery narration overrides by notice key (``gift_workflow.GiftWorkflowSpeech``).
    gift_messages_json: str = "{}"
    #: The deployment does not check the container after a release (``verify_placement: false``):
    #: a confirmed release is announced as the delivery. The launch derives it.
    gift_release_is_delivery: bool = False
    #: How long a sentence said while the robot answers (a clip or spoken reply, not a motion)
    #: is kept to be judged once that answer ends. 0 drops it, as before.
    pending_sentence_sec: float = 0.0
    #: Recorded lines (``embodied_common.recorded_clips.spoken_clips``): spoken text -> files.
    #: Speech whose text is one of them plays a recording instead of being synthesized; any
    #: other text is synthesized as before. The launch builds it from the clip manifest.
    speech_clips_json: str = "{}"
    #: English words and their reading for a TTS without an English front end
    #: ("openEuler" -> "欧喷欧拉"); applied before the remaining Latin words are dropped.
    tts_pronunciations_json: str = "{}"

    def __post_init__(self) -> None:
        GraspSpeechPolicy(self.speech_mode, json.loads(self.speech_messages_json))
        for name in ("persistent_session", "mirror_session_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
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
        if not 0.0 <= float(self.speaker_split_other_similarity) <= float(self.speaker_split_owner_similarity) <= 1.0:
            raise ValueError("speaker split thresholds need 0 <= other_similarity <= owner_similarity <= 1")
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
        if self.require_verified_speech and not self.speaker_check_enabled:
            raise ValueError("require_verified_speech requires speaker_check_enabled")
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
        rewrites = []
        for item in self.text_rewrites:
            if not isinstance(item, tuple | list) or len(item) != 2:
                raise ValueError("text_rewrites entries must be [pattern, replacement] pairs")
            pattern, replacement = str(item[0]), str(item[1])
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"text_rewrites pattern {pattern!r} is invalid: {error}") from error
            rewrites.append((pattern, replacement))
        object.__setattr__(self, "text_rewrites", tuple(rewrites))
        lines = json.loads(self.spoken_lines_json)
        if not isinstance(lines, dict) or any(
            key not in SPOKEN_LINE_EVENTS or not isinstance(value, str) for key, value in lines.items()
        ):
            raise ValueError(f"spoken_lines must map events of {SPOKEN_LINE_EVENTS} to text")
        if any(phrase in _normalize(value) for value in lines.values() for phrase in self.stop_phrases):
            raise ValueError("spoken_lines must not contain a stop phrase: the bridge would stop its own request")
        GiftWorkflowSpeech(messages=json.loads(self.gift_messages_json))
        clips = json.loads(self.speech_clips_json)
        if not isinstance(clips, dict) or any(
            not isinstance(files, list) or not files or any(not isinstance(item, str) for item in files)
            for files in clips.values()
        ):
            raise ValueError("speech_clips must map spoken text to a non-empty list of files")
        pronunciations = json.loads(self.tts_pronunciations_json)
        if not isinstance(pronunciations, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) for key, value in pronunciations.items()
        ):
            raise ValueError("tts_pronunciations must map words to their reading")
        if not float(self.pending_sentence_sec) >= 0:
            raise ValueError("pending_sentence_sec must not be negative")
        if not float(self.semantic_gate_followup_sec) >= 0:
            raise ValueError("semantic_gate_followup_sec must not be negative")
        if self.semantic_gate_mode not in SEMANTIC_GATE_MODES:
            raise ValueError(f"semantic_gate_mode must be one of {SEMANTIC_GATE_MODES}")
        if self.semantic_gate_mode != "off" and not str(self.semantic_gate_model_path).strip():
            raise ValueError("semantic_gate_model_path is required unless semantic_gate_mode is off")
        if not 0.0 <= float(self.semantic_gate_threshold) <= 1.0:
            raise ValueError("semantic_gate_threshold must be in [0, 1]")
        if not float(self.semantic_gate_threshold) <= float(self.semantic_gate_model_only_threshold) <= 1.0:
            raise ValueError("semantic_gate_model_only_threshold must be in [semantic_gate_threshold, 1]")

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
        for pattern, replacement in self.text_rewrites:
            text = re.sub(pattern, replacement, text)
        return correct_manipulation_transcript(text, wake_phrases=self.wake_phrases + self.semantic_gate_wake_prefixes)


@dataclass(frozen=True)
class PublishRequest:
    payload: dict[str, Any]


@dataclass(frozen=True)
class PublishControl:
    payload: dict[str, Any]


@dataclass(frozen=True)
class CancelSpeech:
    """Invalidate audio immediately; transport cancellation runs outside ROS callbacks.

    ``recorded`` also stops the executor-owned recorded clip (a global player stop),
    which plain cancellation never touches.
    """

    recorded: bool = False


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
    request_id: str = ""
    kind: str = "reply"
    expires_at: float = float("inf")
    notice: Any = None
    important: bool = False
    generation: int = 0
    session_id: str = ""
    priority: int | None = None
    phase: str = ""
    event_key: str = ""
    #: The addressee gate this speech belongs to (``VoiceBridgeCore._gate_epoch``): its end
    #: restarts that gate's window, never a newer one.
    gate_epoch: int = 0


@dataclass(frozen=True)
class Log:
    message: str


Action = PublishRequest | PublishControl | CancelSpeech | Speak | Log


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
    #: Token times of the transcript, for keeping one speaker's part of it.
    alignment: TokenAlignment | None = None


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
    task_label: str = "当前任务"
    last_busy_at: float = float("-inf")


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
        self._mirror_speech_policy = MirrorSpeechPolicy()
        self._speech_policy = GraspSpeechPolicy(config.speech_mode, json.loads(config.speech_messages_json))
        self._gift = GiftWorkflowSpeech(
            container_name=config.gift_container_name,
            container_label=config.gift_container_label,
            messages=json.loads(config.gift_messages_json),
            release_is_delivery=config.gift_release_is_delivery,
        )
        self._lines: dict[str, str] = json.loads(config.spoken_lines_json)
        #: Bumped whenever ``end_session`` replaces the addressee gate; a request or speech of an
        #: older epoch must not restart the new gate's window.
        self._gate_epoch = 0
        self._request_epoch: OrderedDict[str, int] = OrderedDict()
        #: The window (its ``wake_until``) the woken speaker was already told had closed.
        self._wake_hint_window = -1.0
        #: Requests that only answer (conversation admission, or a recorded clip): a sentence
        #: said meanwhile is kept for ``pending_sentence_sec`` instead of dropped.
        self._answer_only = _BoundedIdSet()
        self._pending_sentence: tuple[str, float, Segment | None, float] | None = None
        self._generation = 0
        self._latest_request_id = ""
        self._introduction_request_id = ""
        self._unsafe_request_id = ""
        self._stop_deadline = float("inf")
        self._stop_warning_sent = False
        self._ending_session = False
        self._mirror_state: dict[str, Any] | None = None
        self._mirror_start_request_id = ""
        self._mirror_seen_holding = False
        self._mirror_stop_request_id = ""
        self._mirror_busy_spoken = False
        self._mirror_stop_pending = False
        self._mirror_stop_failed = False
        self._mirror_interrupted_id = ""
        self._mirror_owned_requests = _BoundedIdSet()
        self._deferred_text = ""
        self._deferred_after = float("inf")
        self._request_sessions: dict[str, str] = {}
        self._deferred_submission: tuple[str, str] | None = None
        self._speech_suppressed: set[str] = set()
        self._receipt_due: dict[str, float] = {}
        self._announced_requests: set[str] = set()
        self._id_factory = id_factory or (lambda: f"voice-{uuid.uuid4().hex[:12]}")
        self._gate = AddresseeGate(
            require_measurements=config.require_verified_speech,
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
        self._recent_speech: deque[tuple[str, float]] = deque(maxlen=16)
        self._speech_windows: deque[list[float]] = deque(maxlen=16)
        self._echo_segments: OrderedDict[Segment, None] = OrderedDict()
        #: When the robot last finished a turn (its speech ended or a request finished): the
        #: start of the follow-up window conversation admission answers within.
        self._last_robot_turn = float("-inf")
        #: The per-window verdict of each recent segment (``_split_speakers``), for judging a
        #: question rescued from the end of that segment after the speaker check consumed it.
        self._splits: OrderedDict[Segment, SpeakerSplit] = OrderedDict()

    def _end_robot_turn(self, now: float, epoch: int | None) -> None:
        """A turn of the robot (speech, a clip, a motion) ended: restart the window it was taken in.

        Only for a turn of the current gate (``epoch``): one that outlived ``end_session`` belongs
        to a window nobody holds any more, and must not reopen it for whoever speaks next.
        """
        if epoch is None or epoch != self._gate_epoch:
            return
        self._last_robot_turn = max(self._last_robot_turn, now)
        if self.config.persistent_session:
            self._gate.note_robot_turn_end(now)

    def _not_addressed(self, verdict: Any, raw: str, heard_at: float, segment: Segment | None) -> list[Action]:
        """Ignore an utterance not meant for the robot; tell the woken speaker once that the window closed."""
        actions: list[Action] = [Log(f"not addressed to me ({verdict.reason}); ignored {raw!r}")]
        hint = self._lines.get("wake_hint")
        window = float(getattr(self._gate, "wake_until", 0.0))
        if (
            hint
            and verdict.reason == "no wake phrase"
            and window > 0.0
            and window != self._wake_hint_window
            and not self._state.active_request_id
            and self._gate.anchor_speaks(segment)
        ):
            self._wake_hint_window = window
            actions.append(Log("the woken speaker spoke after the window closed; telling them once"))
            if self.config.speak_replies:
                # Not a turn of the window: its end must not reopen what the hint says is closed.
                actions.append(self._speech(hint, kind="reply", gate_epoch=-1))
        return actions

    def on_mirror_state(self, payload: dict[str, Any], now: float, *, source_now: float) -> list[Action]:
        """Convert ROS source age to monotonic expiry without comparing clock domains."""
        try:
            stamp = payload["stamp"]
            age = source_now - float(stamp)
            valid = (
                payload.get("state") in MIRROR_SESSION.STATES
                and type(payload.get("holding")) is bool
                and type(payload.get("releasing")) is bool
                and not isinstance(stamp, bool)
                and math.isfinite(age)
                and 0 <= age <= MIRROR_SESSION.STATE_STALE_SEC
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        self._mirror_state = {**payload, "expires_at": now + MIRROR_SESSION.STATE_STALE_SEC - age} if valid else None
        if valid and payload["holding"]:
            self._mirror_seen_holding = True
        if self._mirror_idle(now):
            self._mirror_busy_spoken = False
        return []

    def _mirror_idle(self, now: float) -> bool:
        if not self.config.mirror_session_enabled:
            return True
        state = self._mirror_state
        if (
            state is None
            or now > state["expires_at"]
            or state["state"] != MIRROR_SESSION.STATE_WAIT
            or state["holding"]
            or state["releasing"]
            or self._mirror_stop_failed
            or self._mirror_stop_pending
            or self._mirror_stop_request_id
        ):
            return False
        # A start result may overtake the holding heartbeat. Old WAIT is not a release.
        if self._mirror_start_request_id:
            if not self._mirror_seen_holding:
                return False
            self._mirror_start_request_id = ""
        return True

    def _submit_mirror_stop(self, now: float) -> list[Action]:
        """Send the safety command through normal Agent and Gateway validation."""
        state = self._state
        if self._mirror_stop_request_id:
            return [Log("mirror stop already awaiting terminal confirmation")]
        self._mirror_stop_pending = False
        request_id = self._id_factory()
        self._mirror_stop_request_id = request_id
        self._mirror_owned_requests.add(request_id)
        self._latest_request_id = request_id
        if not state.session_id:
            state.session_id = _new_session_id()
        self._request_sessions[request_id] = state.session_id
        state.known_request_ids.add(request_id)
        state.active_request_id = request_id
        state.active_since = state.last_activity = now
        state.task_label = "模仿退出"
        state.last_busy_at = float("-inf")
        return [
            *self.cancel_speech(),
            PublishRequest(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "session_id": state.session_id,
                    "text": "停止照镜子",
                }
            ),
        ]

    def _observe_mirror_terminal(self, payload: dict[str, Any], request_id: str) -> None:
        if request_id == self._mirror_stop_request_id:
            self._mirror_stop_request_id = ""
            detail = payload.get("detail") or {}
            facts = detail.get("execution_detail") or {}
            # Version zero is the generated placeholder for executors without
            # phase/recovery facts. Its UNKNOWN defaults do not contradict a
            # confirmed mirror stop. Fresh unheld WAIT is still required below.
            if type(facts.get("schema_version")) is int and facts["schema_version"] == 0:
                facts = {}
            self._mirror_stop_failed = payload.get("state") != "SUCCEEDED" or any(
                item.get("recovery_state") in {"FAILED", "UNKNOWN"}
                or item.get("phase_state") == "UNKNOWN"
                or item.get("suggested_action") == "MANUAL_CHECK"
                for item in (detail, facts)
            )
            if (
                self._mirror_stop_failed
                and payload.get("state") == "UNKNOWN"
                and (not self._unsafe_request_id or self._unsafe_request_id == request_id)
            ):
                self._mirror_interrupted_id = request_id
            if not self._mirror_stop_failed:
                # Production stop confirms runtime idle before returning success.
                self._mirror_start_request_id = ""
                if self._unsafe_request_id == self._mirror_interrupted_id:
                    self._unsafe_request_id = ""
                    self._stop_deadline = float("inf")
                    self._stop_warning_sent = False
                self._mirror_interrupted_id = ""
        elif request_id == self._mirror_start_request_id and payload.get("accepted") is False:
            self._mirror_start_request_id = ""

    def note_speech(self, text: str, now: float, *, finished: bool = False, gate_epoch: int | None = None) -> None:
        self._recent_speech.append((_normalize(text), now))
        if finished:
            self._end_robot_turn(now, self._gate_epoch if gate_epoch is None else gate_epoch)
            if self._speech_windows:
                self._speech_windows[-1][1] = now + 1.0
        else:
            self._speech_windows.append([now, float("inf")])

    def note_audio(self, segment: Segment, captured_at: float) -> None:
        if any(start <= captured_at <= end for start, end in self._speech_windows):
            self._echo_segments[segment] = None
            while len(self._echo_segments) > 128:
                self._echo_segments.popitem(last=False)

    def is_recent_echo(self, text: str, now: float) -> bool:
        if (
            self.is_stop_text(text)
            or self._speech_interrupt(text, now) is not None
            or _normalize(text) in {"不玩了", "再见"}
        ):
            return False
        value = _normalize(text)
        return len(value) >= 6 and any(
            now - stamp <= 20.0 and SequenceMatcher(None, value, spoken).ratio() >= 0.78
            for spoken, stamp in self._recent_speech
        )

    def speech_is_current(self, action: Speak, now: float) -> bool:
        if now >= action.expires_at:
            return False
        if action.request_id in self._speech_suppressed:
            return False
        if action.generation != self._generation or (action.session_id and action.session_id != self._state.session_id):
            return False
        if action.request_id and action.request_id != self._latest_request_id and not self.config.feedback_only:
            return False
        if action.kind == "busy_mirror":
            return not self._mirror_idle(now) and not self._mirror_stop_pending and not self._mirror_stop_request_id
        if action.notice is not None and action.event_key.startswith("MIRROR_"):
            return self._mirror_speech_policy.is_current(action.notice, now)
        if action.notice is not None and self._gift.owns(action.request_id):
            return self._gift.is_current(action.notice, now)
        if action.notice is not None:
            if (
                not action.terminal
                and action.request_id
                and (
                    action.request_id in self._state.finished_request_ids
                    or self._state.stopped_request_id == action.request_id
                )
            ):
                return False
            return self._speech_policy.is_current(action.notice, now)
        return action.kind not in ("busy", "receipt") or (
            action.request_id == self.active_request_id
            and self._state.stopped_request_id != action.request_id
            and now < action.expires_at
        )

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

    def on_heard(
        self, text: str, now: float, segment: Segment | None = None, alignment: TokenAlignment | None = None
    ) -> list[Action]:
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
        if not self.config.require_verified_speech and (
            self.is_stop_text(text)
            or self._speech_interrupt(text, now) is not None
            or _normalize(text) in {"不玩了", "再见"}
        ):
            cancelled = self._cancel_held(f"stop keyword in {str(text).strip()!r}")
            return [*cancelled, *self.on_transcript(text, now, segment=segment, cancels_held=bool(cancelled))]
        if self._held or self.should_hold(text, now, segment):
            deadline = now + self.config.measurement_grace_sec
            self._held.append(_HeldTranscript(str(text), now, deadline, segment, alignment))
            return self.release_held(now)
        return self.on_transcript(text, now, segment=segment, alignment=alignment)

    def release_held(self, now: float) -> list[Action]:
        """Route held transcripts, in order, as soon as each no longer needs to wait."""
        actions: list[Action] = []
        while self._held:
            held = self._held[0]
            if now < held.deadline and self.should_hold(held.text, held.heard_at, held.segment):
                break
            self._held.pop(0)
            actions += self.on_transcript(
                held.text, now, heard_at=held.heard_at, segment=held.segment, alignment=held.alignment
            )
        return actions

    def on_transcript(
        self,
        text: str,
        now: float,
        *,
        heard_at: float | None = None,
        segment: Segment | None = None,
        cancels_held: bool = False,
        alignment: TokenAlignment | None = None,
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
        if self.is_recent_echo(text, now):
            return [Log(f"ignored recent playback echo: {text!r}")]
        stopped_by_partial, self._state.utterance_stopped = self._state.utterance_stopped, False
        heard, split_log = self._split_speakers(str(text).strip(), segment, alignment, heard_at)
        raw = self.config.correct(heard)
        normalized = _normalize(raw)
        # Every transcript, before any branch below can return: the segment a wake
        # event names opens the session and re-anchors whatever it turns out to be
        # -- a stop already sent from its partial, empty text, a stop, an answer --
        # and the wake is spent on it instead of passing to whoever speaks next.
        session_opened = self._gate.open_if_new_session(now, heard_at=heard_at, segment=segment)
        if not normalized:
            return split_log
        corrected = split_log + ([Log(f"corrected transcript {heard!r} -> {raw!r}")] if raw != heard else [])
        if stopped_by_partial and self._contains_stop(normalized) and not self._speech_interrupt(raw, now):
            # Once the stop has finished the request, this sentence would otherwise reach the
            # Agent as a new command and occupy the channel ("帮我停一下" -> busy for "继续").
            return [*corrected, Log(f"final transcript {raw!r} of a stop already sent from its partial")]
        actions = self._route_transcript(
            raw, normalized, now, heard_at, segment, session_opened=session_opened, cancels_held=cancels_held
        )
        return corrected + actions if actions else corrected

    def _split_speakers(
        self, heard: str, segment: Segment | None, alignment: TokenAlignment | None, heard_at: float
    ) -> tuple[str, list[Action]]:
        """Keep the owner's words of a segment that holds two voices; returns the text and its log.

        The windows are compared and logged whenever they exist, so a deployment
        can see how often segments mix before turning ``speaker_split_enabled``
        on. Only then is the transcript cut: to the owner's tokens, or to
        nothing when none of the owner's words were transcribed. A segment that
        is all the owner, or has no owner window at all, is left to the usual
        speaker check.
        """
        split = self._gate.split_owner(
            heard,
            segment,
            alignment,
            heard_at=heard_at,
            owner_similarity=float(self.config.speaker_split_owner_similarity),
            other_similarity=float(self.config.speaker_split_other_similarity),
            apply=bool(self.config.speaker_split_enabled),
        )
        if split is None:
            return heard, []
        if segment is not None:
            self._splits[segment] = split
            while len(self._splits) > 32:
                self._splits.popitem(last=False)
        similarities = " ".join(f"{value:.2f}" for value in split.similarities)
        if split.kind != "mixed":
            return heard, [Log(f"speaker windows [{similarities}] -> {split.kind}: {heard!r}")]
        verdict = (
            f"speaker windows [{similarities}] -> mixed, owner {max(split.span[0], 0.0):.1f}-{split.span[1]:.1f}s: "
            f"kept {split.text!r}, dropped {split.dropped!r}"
        )
        if not self.config.speaker_split_enabled:
            return heard, [Log(f"{verdict} (split off; using the whole transcript)")]
        return split.text, [Log(verdict)]

    def _tail_rejection(self, segment: Segment | None) -> str:
        """Why a question rescued from the end of ``segment`` is not the owner's; empty when it may be.

        Judged by the segment's per-window voiceprints (``_split_speakers`` keeps their
        verdict): a segment already cut to the owner's words is the owner's; otherwise the
        last window must not be clearly someone else. No windows, no anchor or a wake
        utterance: nothing to judge by, which accepts like every other missing measurement.
        """
        split = self._splits.pop(segment, None) if segment is not None else None
        if split is None or not split.similarities:
            return ""
        if split.kind == "mixed":
            # Split on, the transcript is already the owner's words. Split off, the question
            # at the end is the owner's only when the owner's stretch runs to the end.
            if self.config.speaker_split_enabled or split.span[1] == math.inf:
                return ""
            return f"speaker windows: the owner stops at {split.span[1]:.1f}s, before the question"
        if split.kind == "absent":
            return "speaker windows: no window is the owner's"
        last = split.similarities[-1]
        if last < float(self.config.speaker_split_other_similarity):
            return f"speaker windows: last window {last:.2f} < {self.config.speaker_split_other_similarity:.2f}"
        return ""

    def on_partial(self, text: str, now: float) -> list[Action]:
        """Stop as soon as a partial transcript contains a stop keyword; ignore everything else.

        Final transcripts only arrive after VAD silence plus tail padding, which is
        seconds of extra motion when the user says "停". The stop also cancels
        every transcript still held for its measurement (see ``on_heard``); when
        that is all it stopped, its final transcript belongs to it as well.
        """
        # Plain partials have no segment identity or final voiceprint. They cannot
        # authorize a stop when every voice command requires a verified speaker.
        if self.config.require_verified_speech:
            return []
        if self.config.feedback_only or not self._is_stop_partial(text) or self._state.utterance_stopped:
            return []
        if self._may_become_speech_interrupt(text):
            return [Log(f"partial {text!r} may still be a request to stop talking; waiting for the final")]
        cancelled = self._cancel_held(f"stop keyword in partial {text!r}")
        if not self._stop_target() and self._mirror_idle(now):
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
        if not normalized or (
            not self.config.require_verified_speech
            and (self._contains_stop(normalized) or self._state.pending is not None)
        ):
            return False
        return self._gate.needs_wait(normalized, now, segment)

    def _cancel_held(self, reason: str) -> list[Action]:
        cancelled, self._held = self._held, []
        return [Log(f"{reason}: cancelled {held.text!r}, held for its measurement") for held in cancelled]

    def _is_stop_partial(self, text: str) -> bool:
        """A partial that is itself a complete configured stop keyword.

        Partial matching must not be "contains": "停止照镜子" grows through the
        partial "停止", and stopping on that cancels the very request that stops
        the mirror. Keyword matching still applies to the final transcript,
        after tail padding. Do not match unfinished prefixes either: "好" can
        precede "好了我结束了" or an ordinary acknowledgement during a pick.
        """
        normalized = _normalize(self.config.correct(str(text).strip()))
        return bool(normalized) and normalized in self.config.stop_phrases

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
        self,
        raw: str,
        normalized: str,
        now: float,
        *,
        session_opened: bool,
        segment: Segment | None,
        addressed: bool = False,
    ) -> tuple[list[Action] | None, SemanticDecision | None]:
        """Run the L3 gate; returns the actions that stop the utterance (or None) and the verdict.

        Shadow mode records the verdict and returns None, which is what makes the log a
        labelled sample rather than a sample of rejections only: its false negatives (a
        command the gate would have dropped) are exactly what the next threshold has to
        account for. The verdict is returned only when it is enforced, because only then
        do its grade and admitted text shape the request.
        """
        gate = self._semantic
        if gate is None:
            return None, None
        # A session the addressee gate opened (wake word or vendor wake event) is what
        # lets a bare fragment through: "怎么" alone is a person mid-conversation, but
        # only the session says which conversation it belongs to.
        session_open = bool(self._state.session_id) and (
            now - self._state.last_activity <= self.config.session_idle_sec
        )
        # A wake-only utterance opens L1/L2 before there is an Agent session ID.
        # Only the caller's successful addressee verdict supplies that evidence;
        # merely configuring a wake phrase must not enable shape-based admission.
        session_open = session_open or (
            addressed and (bool(self.config.wake_phrases) or self.config.addressee_trigger == "vendor_session")
        )
        conversation = self.config.semantic_gate_conversation
        followup_open = conversation and now - self._last_robot_turn <= self.config.semantic_gate_followup_sec
        decision = gate.judge(raw, session_open=session_open, followup_open=followup_open, conversation=conversation)
        mode = self.config.semantic_gate_mode
        if not decision.rejected and decision.admitted_text:
            # The rescued question is the end of the segment, so the segment's voiceprint (its
            # first seconds) does not speak for it; the segment's last window does.
            tail_rejection = self._tail_rejection(segment)
            if tail_rejection:
                decision = dataclasses.replace(decision, rejected=True, reason=f"{decision.reason}; {tail_rejection}")
        gate.record(
            decision,
            self.config.semantic_gate_log_path,
            mode=mode,
            session_id=self._state.session_id,
            after_wake=bool(session_opened),
            active_request=bool(self._state.active_request_id),
            normalized=normalized,
            followup_open=bool(followup_open),
        )
        if mode != "enforce":
            return None, None
        if not decision.rejected:
            return None, decision
        # Same wording as the addressee verdict on purpose: "not addressed to me" is one
        # concept, whichever layer decided it, and the shadow log carries the score for
        # the ones that only look that way.
        return [Log(f"not addressed to me (semantic {decision.score:.2f}: {decision.reason}); ignored {raw!r}")], None

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
        admitted: bool = False,
    ) -> list[Action]:
        state = self._state
        heard_raw = raw
        verdict = None
        if self.config.require_verified_speech and not admitted:
            verdict = self._gate.judge(
                raw, normalized, now, session_opened=session_opened, heard_at=heard_at, segment=segment
            )
            if not verdict.addressed:
                return self._not_addressed(verdict, raw, heard_at, segment)
            raw, normalized = verdict.raw, verdict.normalized
            # An exact stop is an explicit control intent. A longer sentence
            # merely mentioning STOP must also pass the enabled intent gate.
            if self._contains_stop(normalized) and normalized not in self.config.stop_phrases:
                semantic_block, _ = self._judge_semantics(
                    heard_raw, normalized, now, session_opened=session_opened, segment=segment, addressed=True
                )
                if semantic_block is not None:
                    return semantic_block
        if normalized == "再见":
            return [Log("farewell retained dialogue and business state")]
        remainder = self._speech_interrupt(raw, now)
        if remainder is not None:
            # The robot's turn ends here: the window restarts, as when the turn ends by itself.
            self._end_robot_turn(now, self._request_epoch.get(self._stop_target(), self._gate_epoch))
            if self._introduction_request_id and self._stop_target() == self._introduction_request_id:
                # "别说了" silences the introduction; it does not stop the robot. The clip
                # executor ends the skill normally once its gesture finishes, so there is no
                # stop, no safety lock and no slow cancel-return of the gesture.
                self._deferred_text = remainder
                self._deferred_after = now
                return [
                    Log(f"introduction speech stopped; {self._introduction_request_id} finishes its gesture"),
                    *self.cancel_speech(recorded=True),
                    *(
                        self._say(self._lines["speech_stopped"])
                        if not remainder and self._lines.get("speech_stopped")
                        else []
                    ),
                ]
            actions = self.cancel_speech()
            if not remainder and self._lines.get("speech_stopped"):
                actions += self._say(self._lines["speech_stopped"])
            if remainder:
                return [
                    *actions,
                    *self._route_transcript(
                        remainder,
                        _normalize(remainder),
                        now,
                        heard_at,
                        segment,
                        session_opened=session_opened,
                        admitted=self.config.require_verified_speech,
                    ),
                ]
            return actions
        stops_something = (
            cancels_held
            or self._stop_target()
            or not self._mirror_idle(now)
            or normalized in self.config.stop_phrases
            or normalized == "不玩了"
        )
        if (self._contains_stop(normalized) or normalized == "不玩了") and stops_something:
            # With nothing to interrupt, a sentence that merely contains a keyword is a command.
            # Checked before the wake gate on purpose: a stop must never need addressing.
            held = self._cancel_held(f"verified stop keyword in {raw!r}") if self.config.require_verified_speech else []
            return [*held, *self._stop(now, reason=f"stop keyword in {raw!r}")]
        if self._unsafe_request_id or state.timed_out_request_id or self._ending_session:
            return [Log("previous request has no safe terminal; new intent held back")]
        if state.pending is not None:
            return self._answer_confirmation(normalized, raw, now)
        if verdict is None and not admitted:
            verdict = self._gate.judge(
                raw, normalized, now, session_opened=session_opened, heard_at=heard_at, segment=segment
            )
        if verdict is not None and not verdict.addressed:
            return self._not_addressed(verdict, raw, heard_at, segment)
        # The intent gate judges the sentence as heard: the addressee gate strips the wake
        # phrase, and a woken sentence must keep the gate's wake-prefix exemption.
        if verdict is not None:
            raw, normalized = verdict.raw, verdict.normalized
        if normalized == "再见":
            return [Log("farewell retained dialogue and business state")]
        if any(phrase in normalized for phrase in self.config.bypass_phrases):
            return [Log(f"ignored bypass utterance {raw!r}")]
        if (
            verdict is not None
            and not normalized
            and self._lines.get("wake_ack")
            and not state.active_request_id
            and self._mirror_idle(now)
        ):
            # Only the robot's name: answer that it listens, so the speaker knows to go on.
            return [Log(f"wake phrase only: {heard_raw!r}"), *self._say(self._lines["wake_ack"])]
        if not state.active_request_id and not self._mirror_idle(now):
            mirror = self._mirror_state or {}
            actions = [
                Log(
                    f"mirror ownership unresolved; dropped {raw!r} "
                    f"(state={mirror.get('state', 'unknown')}, "
                    f"fresh={now <= mirror.get('expires_at', float('-inf'))}, "
                    f"holding={mirror.get('holding')}, releasing={mirror.get('releasing')}, "
                    f"stop_failed={self._mirror_stop_failed}, stop_pending={self._mirror_stop_pending}, "
                    f"stop_request={self._mirror_stop_request_id!r})"
                )
            ]
            if not self._mirror_busy_spoken:
                self._mirror_busy_spoken = True
                actions.append(
                    self._speech(
                        "模仿功能还在运行或状态待确认，请稍等。", kind="busy_mirror", priority=0, expires_at=now + 5.0
                    )
                )
            return actions
        if state.active_request_id:
            if (
                self.config.pending_sentence_sec > 0
                and state.active_request_id in self._answer_only
                and len(normalized) >= self.config.min_text_chars
            ):
                # The robot is only answering: the next question waits for the answer to end
                # (``tick``) instead of being lost. One sentence, the latest one.
                self._pending_sentence = (raw, heard_at, segment, now + self.config.pending_sentence_sec)
                return [Log(f"answering {state.active_request_id}; kept {raw!r} for after the answer")]
            actions = [Log(f"busy with {state.active_request_id}; dropped {raw!r}")]
            if state.last_busy_at == float("-inf") and state.stopped_request_id != state.active_request_id:
                state.last_busy_at = now
                busy_line = self._lines.get("busy")
                if self._gift.owns(state.active_request_id) and self._lines.get("busy_pick"):
                    busy_line = self._lines["busy_pick"]
                actions.append(
                    self._speech(
                        busy_line or f"正在做{state.task_label}，请稍等。",
                        request_id=state.active_request_id,
                        kind="busy",
                        priority=0,
                        expires_at=now + 5.0,
                    )
                )
            return actions
        # The robot just asked a question of its own ("可以呀，现在就给你拿吗？"): a short reply
        # inside the follow-up window answers it, and "好"/"要" carry nothing an intent model
        # could judge. It still had to pass the wake window and the voiceprint above.
        answering = (
            bool(state.clarification_request_id)
            and now - self._last_robot_turn <= max(float(self.config.semantic_gate_followup_sec), 20.0)
            and len(normalized) <= _CLARIFICATION_REPLY_MAX_CHARS
        )
        semantic_block, semantic = (
            (None, None)
            if answering
            else self._judge_semantics(
                heard_raw,
                normalized,
                now,
                session_opened=session_opened,
                segment=segment,
                addressed=verdict is not None and verdict.addressed,
            )
        )
        if semantic_block is not None:
            # Vetoed as not meant for the robot, yet it passed the wake and voiceprint checks:
            # the woken speaker is still talking, so the window stays open as for a request.
            self._gate.note_handled(heard_at)
            return semantic_block
        admission_log: list[Action] = []
        if semantic is not None and semantic.admitted_text:
            admission_log.append(Log(f"admitted the trailing question {semantic.admitted_text!r} of {raw!r}"))
            raw, normalized = semantic.admitted_text, _normalize(semantic.admitted_text)
        if len(normalized) < self.config.min_text_chars and not answering:
            return [Log(f"ignored short utterance {raw!r}")]
        self._gate.note_handled(heard_at)

        if not state.session_id or (
            not self.config.persistent_session and now - state.last_activity > self.config.session_idle_sec
        ):
            state.session_id = _new_session_id()
            state.clarification_request_id = ""
        invalidation = self.cancel_speech()
        request_id = self._id_factory()
        self._latest_request_id = request_id
        self._request_sessions[request_id] = state.session_id
        while len(self._request_sessions) > _REMEMBERED_REQUEST_IDS:
            del self._request_sessions[next(iter(self._request_sessions))]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "session_id": state.session_id,
            "text": raw,
        }
        if state.clarification_request_id:
            payload["reply_to_request_id"] = state.clarification_request_id
            state.clarification_request_id = ""
        if semantic is not None and semantic.grade == GRADE_CONVERSATION:
            # Admitted as a question or a reply only: the Agent may answer it, never execute it.
            payload["admission"] = GRADE_CONVERSATION
            self._answer_only.add(request_id)
        self._request_epoch[request_id] = self._gate_epoch
        while len(self._request_epoch) > _REMEMBERED_REQUEST_IDS:
            self._request_epoch.popitem(last=False)
        self._pending_sentence = None
        state.active_request_id = request_id
        state.task_label = _spoken_label(raw)
        state.last_busy_at = float("-inf")
        state.active_since = now
        state.last_activity = now
        state.known_request_ids.add(request_id)
        admitted = f" ({semantic.reason}, conversation only)" if "admission" in payload and semantic else ""
        return [
            *admission_log,
            *invalidation,
            Log(f"request {request_id}: {raw!r}{admitted}"),
            PublishRequest(payload),
        ]

    def on_response(self, payload: dict[str, Any], now: float) -> list[Action]:
        if self.config.feedback_only:
            return []
        request_id = str(payload.get("request_id", ""))
        if request_id not in self._state.known_request_ids:
            return []
        if self._request_sessions.get(request_id) != self._state.session_id:
            return []
        self._state.last_activity = now
        if payload.get("accepted") is False or payload.get("state") in TERMINAL_REQUEST_STATES:
            self._observe_mirror_terminal(payload, request_id)
            # A rejected admission is also emitted as a terminal event; speak only once.
            if not self._finish(request_id):
                return []
            message = str(payload.get("message", "")).strip()
            reason = payload.get("reason_code", "")
            actions = [Log(f"request {request_id} not admitted: {reason} {message}")]
            if reason == "BUSY" and self._deferred_submission and self._deferred_submission[0] == request_id:
                self._deferred_text = self._deferred_submission[1]
                self._deferred_after = now + 0.4
            if reason != "BUSY":
                actions.extend(self._say(_spoken_reply(payload.get("state"), message)))
            return actions
        if payload.get("accepted") is True and request_id == self.active_request_id:
            # No spoken receipt: admission precedes the plan, so a receipt cannot know whether a
            # recorded clip, an answer or execution speech follows, and it lands out of order
            # before the clip or answer that already acknowledges the request.
            return [Log(f"request {request_id} accepted")]
        return []

    def on_event(self, payload: dict[str, Any], now: float) -> list[Action]:
        key = payload.get("request_key")
        if self.config.feedback_only:
            if not isinstance(key, dict) or not all(
                key.get(name) == getattr(self.config, name) for name in ("channel_id", "principal_id", "robot_scope")
            ):
                return []
            return self._execution_speech(payload, now)
        if not isinstance(key, dict) or not self._owns(key):
            return []
        request_id = str(key["request_id"])
        if self._request_sessions.get(request_id, self._state.session_id) != self._state.session_id:
            return []
        self._state.last_activity = now
        event_type = payload.get("event_type")
        state = payload.get("state")
        if state not in TERMINAL_REQUEST_STATES and (
            request_id != self.active_request_id
            or request_id in self._state.finished_request_ids
            or request_id in self._speech_suppressed
            or request_id == self._state.stopped_request_id
        ):
            return []
        if event_type == "presentation":
            return self._on_presentation(payload, request_id, now)
        if event_type == "continuation":
            # The Agent resolved "继续" against its ledgers; tell the user what resumes.
            message = str(payload.get("user_message", "")).strip()
            detail = payload.get("detail") or {}
            continuation = detail.get("continuation") or {}
            target = (continuation.get("interrupted_step") or {}).get("target_name")
            if target:
                self._state.task_label = _spoken_label(target)
            return [Log(f"request {request_id} continuation: {message}"), *self._say("从当前位置继续准备。")]
        if event_type == "reply_preview":
            return self._on_reply_preview(payload, request_id)
        if state not in TERMINAL_REQUEST_STATES:
            if request_id in self._state.finished_request_ids or request_id in self._speech_suppressed:
                return []
            return self._execution_speech(payload, now)
        self._observe_mirror_terminal(payload, request_id)
        if state == "UNKNOWN" and (not self._unsafe_request_id or self._unsafe_request_id == request_id):
            self._unsafe_request_id = request_id
        self._observe_terminal(payload, request_id, now)
        silent = self._silent(request_id) or request_id in self._speech_suppressed
        previewed = self._take_preview(request_id)
        if not self._finish(request_id):
            return []
        # A recorded clip (the fruit menu) never passes through this bridge's speech, so its
        # request's end is the robot's end of turn; spoken replies move it on again. A turn
        # whose outcome is unknown does not: nothing new may start until staff resolve it.
        if state != "UNKNOWN" and event_type != "quarantined":
            self._end_robot_turn(now, self._request_epoch.get(request_id))
        if request_id in self._speech_suppressed:
            return [Log(f"suppressed terminal speech for {request_id}")]
        if event_type == "needs_clarification":
            self._state.clarification_request_id = request_id
        message = str(payload.get("user_message", "")).strip()
        if isinstance(payload.get("detail"), dict) and payload["detail"].get("execution_id"):
            return self._execution_speech(payload, now, silent=silent)
        spoken = "" if silent else _spoken_reply(state, message)
        if state == "SUCCEEDED" and spoken == "执行完成。" and self._lines.get("gesture_done"):
            spoken = self._lines["gesture_done"]
        detail = payload.get("detail") or {}
        if state == "SUCCEEDED" and isinstance(detail, dict) and detail.get("save_prompt"):
            spoken = "执行完成。" + str(detail["save_prompt"])
        log = Log(f"request {request_id} {event_type} {state}: {message}")
        if previewed and spoken == previewed:
            return [log, Log(f"request {request_id} reply was already said as its preview")]

        actions = (
            [
                self._speech(
                    spoken,
                    request_id=request_id,
                    terminal=state != "ANSWERED",
                    kind="reply" if state == "ANSWERED" else "terminal",
                    priority=10 if state == "ANSWERED" else 20,
                )
            ]
            if spoken and self.config.speak_replies
            else []
        )
        facts = detail.get("execution_detail") or {}
        protected = (
            state in {"UNKNOWN", "CANCELED", "CANCELLED"}
            or facts.get("recovery_state") in {"FAILED", "UNKNOWN"}
            or facts.get("suggested_action") == "MANUAL_CHECK"
        )
        if protected and spoken and self.config.speak_replies:
            actions = [self._speech(spoken, request_id=request_id, terminal=True, kind="terminal", priority=40)]
        return [log, *actions]

    def _execution_speech(self, payload: dict[str, Any], now: float, *, silent: bool = False) -> list[Action]:
        if payload.get("event_type") not in {"execution_progress", "terminal", "quarantined", "failed", "cancelled"}:
            return []
        detail = payload.get("detail") or {}
        request_id = str((payload.get("request_key") or {}).get("request_id", ""))
        mirror = self._mirror_speech_policy.observe(payload, now)
        if mirror is not None:
            return self._notices(mirror) if self.config.speak_replies else []
        if self._gift.owns(request_id):
            notices = self._gift.accept(payload, now)
            return self._notices(notices) if self.config.speak_replies else []
        if detail.get("current_skill") == "pick_object":
            # Accept every ordered fact, including silent stages, before selecting speech.
            self._feedback.accept(payload)
            if not self.config.speak_replies:
                return []
            return self._notices(self._speech_policy.observe(payload, now))
        accepted = self._feedback.accept(payload)
        if accepted is None or not self.config.speak_replies:
            return []
        execution_id, text, terminal = accepted
        if payload.get("state") == "SUCCEEDED" and text == "执行完成。" and self._lines.get("gesture_done"):
            text = self._lines["gesture_done"]
        key = payload.get("request_key")
        request_id = str(key.get("request_id", "")) if isinstance(key, dict) else ""
        # Finishing a request clears its silent mark, so a terminal event passes it in. A clip
        # that failed never played: only the failure can tell the visitor anything happened.
        if (silent or self._silent(request_id)) and (not terminal or payload.get("state") == "SUCCEEDED"):
            return [Log(f"request {request_id} speech suppressed: its output is speech itself")]
        detail = payload.get("detail") or {}
        if payload.get("state") == "SUCCEEDED" and detail.get("save_prompt"):
            text += str(detail["save_prompt"])
        return [
            self._speech(
                text,
                request_id=request_id,
                execution_id=execution_id,
                terminal=terminal,
                kind="terminal" if terminal else "reply",
                priority=40
                if payload.get("state") in {"UNKNOWN", "CANCELED", "CANCELLED"}
                or (detail.get("execution_detail") or {}).get("recovery_state") in {"FAILED", "UNKNOWN"}
                else 20
                if terminal
                else 10,
            )
        ]

    def _notices(self, notices) -> list[Action]:
        return [
            self._speech(
                n.text,
                execution_id=n.execution_id,
                terminal=n.terminal,
                request_id=n.request_id,
                kind="terminal" if n.terminal else "reply",
                notice=n,
                important=n.important,
                priority=40 if n.token.split(":", 1)[0] == "E06" else getattr(n, "priority", 20 if n.terminal else 10),
                phase=getattr(n, "phase", ""),
                event_key=getattr(n, "event_key", n.token),
                expires_at=getattr(n, "expires_at", float("inf")),
            )
            for n in notices
        ]

    def note_playback(self, action: Speak, now: float) -> None:
        if (
            action.notice is not None
            and not self._gift.owns(action.request_id)
            and not action.event_key.startswith("MIRROR_")
        ):
            self._speech_policy.played(action.notice, now)

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
        if self._mirror_stop_pending and not state.active_request_id and not state.timed_out_request_id:
            actions += self._submit_mirror_stop(now)
        if (
            self._deferred_text
            and not self._unsafe_request_id
            and not state.active_request_id
            and self._mirror_idle(now)
            and now >= self._deferred_after
        ):
            text, self._deferred_text = self._deferred_text, ""
            submitted = self._route_transcript(text, _normalize(text), now, now, None, session_opened=True)
            actions += submitted
            for action in submitted:
                if isinstance(action, PublishRequest):
                    self._deferred_submission = (action.payload["request_id"], text)
        if self._pending_sentence is not None and not state.active_request_id:
            text, heard_at, segment, deadline = self._pending_sentence
            self._pending_sentence = None
            if now <= deadline and not self._unsafe_request_id and self._mirror_idle(now):
                actions.append(Log(f"answer ended; judging the kept sentence {text!r}"))
                actions += self._route_transcript(
                    text, _normalize(text), now, heard_at, segment, session_opened=False, admitted=True
                )
            else:
                actions.append(Log(f"kept sentence {text!r} expired before the answer ended"))
        if (
            self._ending_session
            and not self._unsafe_request_id
            and not state.active_request_id
            and self._mirror_idle(now)
        ):
            self._ending_session = False
            state.session_id = ""
        if self.config.speak_replies:
            actions += self._notices(self._speech_policy.tick(now))
        for request_id, due in list(self._receipt_due.items()):
            if now >= due:
                self._receipt_due.pop(request_id, None)
                if request_id == state.active_request_id and request_id not in self._announced_requests:
                    actions.append(
                        self._speech(
                            "好呀，我在准备了，稍等一下。", request_id=request_id, kind="receipt", expires_at=now + 3.0
                        )
                    )
        if state.pending is not None and now >= state.pending.deadline:
            request_id = state.pending.request_id
            state.pending = None
            actions += [
                *self._stop(now, reason=f"confirmation for {request_id} timed out"),
                *self._say("确认超时，正在取消。"),
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
            self._unsafe_request_id = request_id
            if self._stop_deadline == float("inf"):
                self._stop_deadline = now + 5.0
                self._stop_warning_sent = False
            actions += self.cancel_speech()
            actions += [
                Log(f"request {request_id} produced no terminal event; stopping and releasing channel"),
                PublishControl({"operation": "stop", "request_id": request_id}),
                # Still a stop target while this plays: the reply must not contain a stop word.
                *self._say("任务超时，正在中止。"),
            ]
        if self._unsafe_request_id and now >= self._stop_deadline and not self._stop_warning_sent:
            self._stop_warning_sent = True
            actions += [
                Log(f"E06: stop confirmation timed out for {self._unsafe_request_id}"),
                self._speech("停止状态尚未确认，请工作人员检查机器人和物体。", request_id="", important=True),
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
        if request_id != self._state.active_request_id or request_id in self._speech_suppressed:
            # A late presentation for a finished or timed-out request must not unlock execution
            # the channel no longer tracks.
            return [Log(f"presentation for inactive request {request_id}; not acknowledging")]
        detail = payload.get("detail")
        if not isinstance(detail, dict):
            return [Log(f"presentation for {request_id} has no detail; not acknowledging")]
        presentation = detail.get("presentation")
        if not isinstance(presentation, dict) or presentation_digest(presentation) != detail.get("presentation_digest"):
            return [Log(f"presentation digest mismatch for {request_id}; not acknowledging")]
        steps = presentation.get("steps", [])
        if self.config.mirror_session_enabled and any(
            isinstance(step, dict) and step.get("skill_name") == "mirror_start" for step in steps
        ):
            self._mirror_start_request_id = request_id
            self._mirror_owned_requests.add(request_id)
            self._mirror_seen_holding = False
            self._mirror_busy_spoken = False
        gift_owned = self._gift.register(request_id, self._state.session_id, steps)
        if any(
            isinstance(step, dict)
            and step.get("skill_name")
            in {
                "greet_visitor",
                "invite_visitor",
                "introduce_in_detail",
                "self_introduction",
                "introduce_agent",
                "introduce_platform",
                "runtime_play_audio",
                *self.config.silent_skills,
            }
            for step in steps
        ):
            self._introduction_request_id = request_id
        summary = _presentation_summary(presentation, payload.get("user_message", ""))
        self._state.task_label = _spoken_label(summary)
        self._receipt_due.pop(request_id, None)
        plan_log = Log("exact plan " + json.dumps(presentation, ensure_ascii=False, sort_keys=True))
        if self._is_silent_presentation(presentation):
            # The clip is the reply: announcing the plan and then saying "执行完成。" would talk over it.
            self._state.silent_request_id = request_id
            self._answer_only.add(request_id)
            return [plan_log]
        pick_steps = [
            step
            for step in presentation.get("steps", [])
            if isinstance(step, dict) and step.get("skill_name") == "pick_object"
        ]
        if gift_owned:
            self._announced_requests.add(request_id)
            return [plan_log]
        if pick_steps and (
            detail.get("async_display") is True or presentation.get("execution_mode") == ASYNC_PRESENTATION_MODE
        ):
            if request_id in self._announced_requests:
                return [plan_log]
            self._announced_requests.add(request_id)
            self._receipt_due.pop(request_id, None)
            target = str(pick_steps[0].get("target_name", ""))
            if not re.search(r"[\u4e00-\u9fff]", target):
                target = "这个东西"
            return (
                [plan_log, *self._notices([self._speech_policy.start(request_id, target, now)])]
                if self.config.speak_replies
                else [plan_log]
            )
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
            standalone_return = (
                len(presentation.get("steps", [])) == 1
                and presentation["steps"][0].get("skill_name") == "return_to_task_pose"
            )
            line = (
                "好呀，我先准备一下。"
                if standalone_return
                else self._lines.get("gesture_start") or f"正在执行：{summary}。"
            )
            return [plan_log, *gate, *self._say(line)]
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
            return [plan_log, self._speech(f"准备执行：{summary}。", then=(acknowledgement,))]
        self._state.pending = _PendingConfirmation(
            request_id=request_id,
            acknowledgement=acknowledgement,
            deadline=now + self.config.confirm_timeout_sec,
        )
        return [plan_log, self._speech(f"准备执行：{summary}。请说确认或取消。")]

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
        if self.config.mirror_session_enabled and (self._mirror_start_request_id or not self._mirror_idle(now)):
            if target and target == self._mirror_stop_request_id:
                return [Log("mirror stop already running"), *self.cancel_speech()]
            if not target:
                return self._submit_mirror_stop(now)
            self._mirror_stop_pending = True
            if target in self._mirror_owned_requests:
                self._mirror_interrupted_id = target
        if not target:
            return [Log(f"{reason}: no active request")]
        state.stopped_request_id = target
        # Stopping ends the robot's turn; the woken speaker usually goes on talking. A stop
        # whose terminal never comes keeps the channel held anyway (``_unsafe_request_id``).
        self._end_robot_turn(now, self._request_epoch.get(target))
        self._unsafe_request_id = target
        if self._stop_deadline == float("inf"):
            self._stop_deadline = now + 5.0
            self._stop_warning_sent = False
        return [
            Log(f"{reason}: stopping {target}"),
            PublishControl({"operation": "stop", "request_id": target}),
            *self.cancel_speech(),
        ]

    def _finish(self, request_id: str) -> bool:
        """Release ``request_id``; return False when it was already finished."""
        self._receipt_due.pop(request_id, None)
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

    def _speech_interrupt(self, text: str, now: float | None = None) -> str | None:
        """The rest of a request to stop talking ("好了别说了，挥个手" -> "挥个手"); None when it is none.

        A short sentence (at most ``_SPEECH_INTERRUPT_MAX_CHARS``) that contains an interrupt
        phrase anywhere is one ("行了别说了", "知道了别说了"); a longer one only with the phrase
        up front. While the robot is talking (``now`` given), a sentence made only of closing
        words ("好了", "知道啦", "好啦知道啦") stops the talk too; at any other time those words
        are ordinary answers and go to the intent gate.
        """
        text = str(text).strip()
        match = re.match(
            _SPEECH_INTERRUPT_LEAD + r"(?:" + "|".join(_SPEECH_INTERRUPT_PHRASES) + r")[，,。！!\s]*(.*)$",
            text,
        )
        if match:
            return match.group(1).strip()
        compact = re.sub(r"[，,。！!？?、\s~～]", "", text)
        if len(compact) <= _SPEECH_INTERRUPT_MAX_CHARS:
            for phrase in sorted(_SPEECH_INTERRUPT_PHRASES, key=len, reverse=True):
                at = compact.find(phrase)
                if at >= 0:
                    return compact[at + len(phrase) :]
        if now is not None and _closing_words(compact) and self._robot_speaking(now):
            return ""
        return None

    def _robot_speaking(self, now: float) -> bool:
        """A recorded answer is playing, or the bridge's own speech has not finished."""
        if self._introduction_request_id and self._stop_target() == self._introduction_request_id:
            return True
        return any(start <= now <= end for start, end in self._speech_windows)

    def _may_become_speech_interrupt(self, partial: str) -> bool:
        """During an introduction, a partial that is or may still grow into "别说了" is not a stop.

        "别" is a prefix of both "别动" and "别说了"; stopping on it would turn
        "silence the clip" into a full stop of the gesture. A real stop word that
        cannot become one ("别动", "停下") still stops from its partial.
        """
        if not (self._introduction_request_id and self._stop_target() == self._introduction_request_id):
            return False
        text = re.sub(r"[，,。！!？?、\s]", "", _normalize(self.config.correct(str(partial).strip())))
        # "好了别", "知道了别说": a closing word may come first ("好了别说了").
        text = re.sub(r"^(?:" + "|".join(_CLOSING_WORDS) + r")+", "", text) or text
        text = re.sub(_SPEECH_INTERRUPT_LEAD, "", text)
        return bool(text) and any(
            phrase.startswith(text) or text.startswith(phrase) for phrase in _SPEECH_INTERRUPT_PHRASES
        )

    def cancel_speech(self, *, recorded: bool = False) -> list[Action]:
        self._generation += 1
        self._receipt_due.clear()
        self._state.preview_text = ""
        self._state.preview_request_id = ""
        return [CancelSpeech(recorded=recorded)]

    def end_session(self, now: float) -> list[Action]:
        """End dialogue without clearing unresolved robot safety state."""
        self._gift.end_session(self._state.session_id)
        self._gate_epoch += 1
        self._pending_sentence = None
        # Retain configured checks, but end their wake window and speaker anchors.
        self._gate = AddresseeGate(
            trigger=self.config.addressee_trigger,
            wake_phrases=self.config.wake_phrases,
            wake_window_sec=self.config.wake_window_sec,
            session_gap_sec=self.config.session_gap_sec,
            checks=self._gate.checks,
        )
        self._held.clear()
        self._deferred_text = ""
        self._state.clarification_request_id = ""
        self._speech_suppressed.update(self._request_sessions)
        target = self._stop_target() or self._unsafe_request_id
        mirror_unresolved = not self._mirror_idle(now)
        self._ending_session = bool(target) or mirror_unresolved
        if mirror_unresolved:
            return [*self._stop(now, reason="staff session reset"), *self.cancel_speech()]
        if target:
            self._unsafe_request_id = target
            return [*self._stop(now, reason="session ended"), *self.cancel_speech()]
        self._state.session_id = ""
        return self.cancel_speech()

    def _observe_terminal(self, payload: dict[str, Any], request_id: str, now: float) -> None:
        if self._unsafe_request_id != request_id:
            return
        if request_id in self._mirror_owned_requests and (
            self._mirror_stop_failed or self._mirror_stop_pending or self._mirror_stop_request_id
        ):
            return
        detail = payload.get("detail") or {}
        facts = detail.get("execution_detail") or {}
        if payload.get("state") == "UNKNOWN" or any(
            item.get("recovery_state") in {"FAILED", "UNKNOWN"} for item in (detail, facts)
        ):
            return
        if (
            self._introduction_request_id == request_id
            and payload.get("state") in {"CANCELLED", "CANCELED"}
            and (
                facts.get("phase_state") != "CANCELED"
                or facts.get("reason_code") != "SKILL_CANCELLED"
                or facts.get("recovery_state") != "NOT_NEEDED"
            )
        ):
            return
        if payload.get("state") not in {"SUCCEEDED", "CANCELLED", "CANCELED", "CANCELLED_BEFORE_EXECUTION"}:
            return
        self._unsafe_request_id = ""
        self._stop_deadline = float("inf")
        self._stop_warning_sent = False
        self._introduction_request_id = ""
        self._deferred_after = now + 0.4
        if self._ending_session:
            self._ending_session = False
            self._state.session_id = ""

    def _speech(self, text: str, **kwargs) -> Speak:
        kwargs.setdefault("request_id", self._latest_request_id)
        kwargs.setdefault("priority", 20 if kwargs.get("kind") == "terminal" else 10)
        kwargs.setdefault("gate_epoch", self._request_epoch.get(kwargs["request_id"], self._gate_epoch))
        return Speak(text, generation=self._generation, session_id=self._state.session_id, **kwargs)

    def _say(self, text: str, *, kind: str = "reply") -> list[Action]:
        if not text or not self.config.speak_replies:
            return []
        return [self._speech(text, kind=kind)]


#: Phrases that silence a recorded introduction without stopping the robot, and the fillers
#: allowed before them ("好的，别说了").
_SPEECH_INTERRUPT_PHRASES = (
    "不用说了",
    "别讲了",
    "别说了",
    "停止播报",
    "停止讲解",
    "不用讲了",
    "别介绍了",
    "不用介绍了",
    "好了好了",
    "别说",
)
_SPEECH_INTERRUPT_LEAD = r"^\s*(?:(?:好(?:的)?|嗯|那个|行)[，,。！!\s]*)*"
#: Longest reply to the robot's own clarification question that skips the intent gate.
_CLARIFICATION_REPLY_MAX_CHARS = 12
#: Longest sentence in which an interrupt phrase counts anywhere ("后来知道了别说了" is 8).
_SPEECH_INTERRUPT_MAX_CHARS = 12
#: Words that end a talk only while the robot is talking ("好了", "知道啦"): elsewhere they are
#: answers. "好"/"好的"/"嗯" alone are not among them: a listener says them while listening.
_CLOSING_WORDS = (
    "好了",
    "好啦",
    "知道了",
    "知道啦",
    "行了",
    "行啦",
    "可以了",
    "可以啦",
    "好的好的",
    "好好",
    "好的",
    "好",
    "行",
    "嗯",
)
_STRONG_CLOSING = ("好了", "好啦", "知道了", "知道啦", "行了", "行啦", "可以了", "可以啦", "好的好的")


def _closing_words(compact: str) -> bool:
    """Whether ``compact`` is made only of closing words, at least one of them a strong one."""
    rest, strong = compact, False
    while rest:
        word = next((item for item in _CLOSING_WORDS if rest.startswith(item)), "")
        if not word:
            return False
        strong = strong or word in _STRONG_CLOSING
        rest = rest[len(word) :]
    return strong


def _spoken_reply(state: Any, message: str) -> str:
    """Execution terminals carry technical text; answers are already user-facing."""
    return execution_reply(str(state), fallback=message)


def _spoken_label(text: str) -> str:
    value = re.sub(r"[A-Za-z_][A-Za-z_0-9-]*", "", str(text).replace("V字", "胜利手势"))
    value = value.strip(" ：:，,。()（）")
    return value[:18] if re.search(r"[\u4e00-\u9fff]", value) else "当前动作"


def _tts_text(text: str, *, keep_english: bool = False, pronunciations: dict[str, str] | None = None) -> str:
    """Text the TTS can say: internal IDs never reach it, English words only when it reads English.

    ``keep_english`` is for a TTS whose English frontend is installed (ZipVoice with
    piper_phonemize); without it a sentence containing English fails to synthesize at all,
    so the default keeps dropping every Latin word -- after ``pronunciations`` replaced the
    ones a deployment knows how to say ("openEuler" -> "欧喷欧拉"), longest first.
    """
    text = str(text).replace("free_pose", "自由姿态").replace("(remembered)", "").replace("V字", "胜利手势")
    if pronunciations and not keep_english:
        for word in sorted(pronunciations, key=len, reverse=True):
            text = re.sub(re.escape(word), pronunciations[word], text, flags=re.IGNORECASE)
    if keep_english:
        # snake_case skill names and robot IDs are not words anyone should hear.
        text = re.sub(r"[A-Za-z0-9-]*_[A-Za-z0-9_-]*", "", text).strip()
        return text if re.search(r"[\u4e00-\u9fffA-Za-z]", text) else "正在处理。"
    text = re.sub(r"[A-Za-z_][A-Za-z_0-9-]*", "", text).strip()
    return text if re.search(r"[\u4e00-\u9fff]", text) else "正在处理。"


def _presentation_summary(presentation: dict[str, Any], fallback: Any) -> str:
    steps = presentation.get("steps") or []
    targets = [str(step.get("target_name", "")) for step in steps if isinstance(step, dict)]
    if any(re.search(r"[\u4e00-\u9fff]", target) for target in targets):
        return _spoken_label("、".join(targets))
    summary = presentation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _spoken_label(summary)
    return _spoken_label(fallback)


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
        "speech_chunk_topic": "",
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
        # True only where the TTS has its English frontend; otherwise English words are dropped.
        "tts_keep_english": False,
        "resume_settle_sec": 1.0,
    }
    for item in dataclasses.fields(VoiceBridgeConfig):
        if item.name in _PHRASE_FIELDS:
            defaults[f"{item.name}_json"] = json.dumps(list(item.default), ensure_ascii=False)
        elif item.name == "text_corrections":
            defaults["text_corrections_json"] = json.dumps(dict(item.default), ensure_ascii=False)
        elif item.name == "text_rewrites":
            defaults["text_rewrites_json"] = json.dumps([list(pair) for pair in item.default], ensure_ascii=False)
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
        require_verified_speech=bool(values["require_verified_speech"]),
        speaker_min_similarity=float(values["speaker_min_similarity"]),
        speaker_max_age_sec=float(values["speaker_max_age_sec"]),
        speaker_merge_margin=float(values["speaker_merge_margin"]),
        speaker_reject_short=bool(values["speaker_reject_short"]),
        speaker_split_enabled=bool(values["speaker_split_enabled"]),
        speaker_split_owner_similarity=float(values["speaker_split_owner_similarity"]),
        speaker_split_other_similarity=float(values["speaker_split_other_similarity"]),
        measurement_grace_sec=float(values["measurement_grace_sec"]),
        confirm_timeout_sec=float(values["confirm_timeout_sec"]),
        session_idle_sec=float(values["session_idle_sec"]),
        persistent_session=bool(values["persistent_session"]),
        mirror_session_enabled=bool(values["mirror_session_enabled"]),
        min_text_chars=int(values["min_text_chars"]),
        request_timeout_sec=float(values["request_timeout_sec"]),
        speech_mode=str(values["speech_mode"]),
        speech_messages_json=str(values["speech_messages_json"]),
        gift_container_name=str(values["gift_container_name"]),
        gift_container_label=str(values["gift_container_label"]),
        speak_replies=bool(values["speak_replies"]),
        speak_reply_previews=bool(values["speak_reply_previews"]),
        feedback_only=bool(values["feedback_only"]),
        busy_message=str(values["busy_message"]),
        text_corrections=_corrections(values.get("text_corrections_json", "{}")),
        text_rewrites=_rewrites(values.get("text_rewrites_json", "[]")),
        semantic_gate_mode=str(values["semantic_gate_mode"]),
        semantic_gate_model_path=str(values["semantic_gate_model_path"]),
        semantic_gate_threshold=float(values["semantic_gate_threshold"]),
        semantic_gate_model_only_threshold=float(values["semantic_gate_model_only_threshold"]),
        semantic_gate_lexicon=_string_tuple(values["semantic_gate_lexicon_json"], "semantic_gate_lexicon"),
        semantic_gate_dialogue_requests=_string_tuple(
            values["semantic_gate_dialogue_requests_json"], "semantic_gate_dialogue_requests"
        ),
        semantic_gate_wake_prefixes=_string_tuple(
            values["semantic_gate_wake_prefixes_json"], "semantic_gate_wake_prefixes"
        ),
        semantic_gate_log_path=str(values["semantic_gate_log_path"]),
        semantic_gate_fail_open=bool(values["semantic_gate_fail_open"]),
        semantic_gate_conversation=bool(values["semantic_gate_conversation"]),
        semantic_gate_followup_sec=float(values["semantic_gate_followup_sec"]),
        spoken_lines_json=str(values["spoken_lines_json"]),
        gift_messages_json=str(values["gift_messages_json"]),
        gift_release_is_delivery=bool(values["gift_release_is_delivery"]),
        pending_sentence_sec=float(values["pending_sentence_sec"]),
        speech_clips_json=str(values["speech_clips_json"]),
        tts_pronunciations_json=str(values["tts_pronunciations_json"]),
    )


def _rewrites(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(item, list) or len(item) != 2 for item in value):
        raise ValueError("text_rewrites must be a list of [pattern, replacement] pairs")
    return tuple((str(pattern), str(replacement)) for pattern, replacement in value)


def _corrections(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("text_corrections must be a mapping of term -> list of misrecognized variants")
    return tuple((str(term), _string_tuple(variants, f"text_corrections[{term}]")) for term, variants in value.items())


@dataclass(frozen=True)
class _RecordedAudio:
    """A recorded line chosen in place of synthesis (``speech_clips``)."""

    path: str


def choose_recorded_line(
    text: str, clips: dict[str, list[str]], last: dict[str, str], *, exists: Callable[[str], bool] = os.path.isfile
) -> str:
    """The file to play for ``text``, or empty to synthesize it: not a recorded line, or no file left."""
    key = spoken_key(text)
    files = clips.get(key)
    if not files:
        return ""
    path = choose_clip_file(files, last=last.get(key, ""), exists=exists)
    if path:
        last[key] = path
    return path


class _Speaker:
    """Independent blocking adapters for synthesis and whole-utterance playback."""

    def __init__(self, node: Any, *, synthesis_service: str, playback_service: str, timeout_sec: float) -> None:
        from std_srvs.srv import Trigger

        from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech

        self._node = node
        self._timeout_sec = timeout_sec
        self._synthesis_type = SynthesizeSpeech
        self._playback_type = PlayAudioFile
        self._synthesize = node.create_client(SynthesizeSpeech, synthesis_service)
        self._playback = node.create_client(PlayAudioFile, playback_service)
        self._stop_type = Trigger
        self._stop_client = node.create_client(Trigger, "/voice_tts/stop")
        self._play_lock = threading.Lock()
        self._play_generation = 0
        self._pending_stop = None
        self._pending_play = None
        self._closing = threading.Event()

    def stop(self, *, force: bool = False) -> None:
        with self._play_lock:
            self._play_generation += 1
            # The executor owns recorded introductions. Never send a global stop
            # without bridge playback to cancel, or a delayed command could stop it;
            # ``force`` is the deliberate "别说了" that silences such a clip.
            if not force and self._pending_play is None and self._pending_stop is None:
                return
        if not self._stop_client.wait_for_service(timeout_sec=0.2):
            raise TimeoutError("TTS stop service unavailable")
        if self._pending_stop is None:
            self._pending_stop = self._stop_client.call_async(self._stop_type.Request())
        future = self._pending_stop
        deadline = time.monotonic() + 0.5
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise TimeoutError("TTS stop was not acknowledged")
        self._pending_stop = None
        if not future.result().success:
            raise TimeoutError("TTS stop failed")
        # A successful stop can precede the old play response. The pipeline
        # waits for that callback and retries stop while it drains; do not turn
        # this normal ordering into a cancellation failure.

    def close(self) -> None:
        self._closing.set()
        self.stop()

    def synthesize(self, text: str) -> Any:
        if self._closing.is_set() or not self._synthesize.wait_for_service(timeout_sec=2.0):
            return None
        request = self._synthesis_type.Request()
        request.text = text
        result = self._call(self._synthesize, request)
        if result is None or not result.success or not result.audio_segments:
            self._node.get_logger().error(f"TTS synthesis failed: {getattr(result, 'message', 'timeout')}")
            return None
        return result.audio_segments

    def play_file(self, path: str) -> bool:
        """Play a recorded file through the same service, cancellable the same way."""
        with self._play_lock:
            generation = self._play_generation
        if self._closing.is_set() or not self._playback.wait_for_service(timeout_sec=2.0):
            return False
        play = self._playback_type.Request()
        play.file_path = path
        with self._play_lock:
            if generation != self._play_generation or self._closing.is_set():
                return False
            if self._pending_stop is not None or self._pending_play is not None:
                return False
            future = self._playback.call_async(play)
            self._pending_play = future
        played = self._wait(future)
        if future.done():
            self._pending_play = None
        if played is None or not played.success:
            self._node.get_logger().error(f"recorded line playback failed: {getattr(played, 'message', 'timeout')}")
            return False
        return True

    def play(self, segments: Any) -> bool:
        with self._play_lock:
            generation = self._play_generation
        if self._closing.is_set() or not self._playback.wait_for_service(timeout_sec=2.0):
            return False
        for segment in segments:
            if self._closing.is_set() or generation != self._play_generation:
                return False
            descriptor, path = tempfile.mkstemp(prefix="ibrobot-voice-bridge-", suffix=".wav")
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(bytes(segment.audio_data))
                play = self._playback_type.Request()
                play.file_path = path
                with self._play_lock:
                    if generation != self._play_generation or self._closing.is_set():
                        return False
                    if self._pending_stop is not None or self._pending_play is not None:
                        return False
                    future = self._playback.call_async(play)
                    self._pending_play = future
                played = self._wait(future)
                if future.done():
                    self._pending_play = None
                if played is None or not played.success:
                    self._node.get_logger().error(f"TTS playback failed: {getattr(played, 'message', 'timeout')}")
                    return False
            finally:
                os.unlink(path)
        return True

    def _call(self, client: Any, request: Any) -> Any:
        if self._closing.is_set():
            return None
        future = client.call_async(request)
        return self._wait(future)

    def _wait(self, future: Any) -> Any:
        deadline = time.monotonic() + self._timeout_sec
        while not future.done() and time.monotonic() < deadline and not self._closing.is_set():
            time.sleep(0.02)
        return future.result() if future.done() else None


def _build_node_class():
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    class VoiceBridgeNode(Node):
        """ROS transport for :class:`VoiceBridgeCore`."""

        _tts_keep_english = False  # set from the tts_keep_english parameter
        # Set from speech_clips_json / tts_pronunciations_json; empty: everything is synthesized.
        _spoken_clips: dict[str, list[str]] = {}  # noqa: RUF012
        _spoken_last: dict[str, str] = {}  # noqa: RUF012
        _pronunciations: dict[str, str] = {}  # noqa: RUF012

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
            self._tts_keep_english = bool(values["tts_keep_english"])
            self._spoken_clips: dict[str, list[str]] = json.loads(str(values["speech_clips_json"]))
            self._spoken_last: dict[str, str] = {}
            self._pronunciations: dict[str, str] = json.loads(str(values["tts_pronunciations_json"]))
            missing = sorted(
                {path for files in self._spoken_clips.values() for path in files if not os.path.isfile(path)}
            )
            if missing:
                # Not fatal: such a line is synthesized instead. The clips are installed per checkout.
                self.get_logger().warning(
                    f"{len(missing)} recorded line file(s) missing, those lines are synthesized: {missing[:5]}"
                )
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
            self._speech_pipeline = None
            self._speech_commands = queue.Queue()
            self._speech_transport_failed = False
            self._speech_transport = None
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
            self._active_audio_segments: dict[str, int] = {}
            if values["speech_chunk_topic"]:
                from ibrobot_msgs.msg import SpeechAudioChunk

                self.create_subscription(
                    SpeechAudioChunk,
                    str(values["speech_chunk_topic"]),
                    self._on_audio_chunk,
                    50,
                    callback_group=inbound,
                )
            if self._core.config.feedback_only:
                transcript_topic = "feedback only"
            elif self._core.config.pairs_by_segment or values["transcript_topic"]:
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
            if self._core.config.mirror_session_enabled:
                self.create_subscription(
                    String,
                    MIRROR_SESSION.STATE_TOPIC,
                    self._on_mirror_state,
                    QoSProfile(
                        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
                    ),
                    callback_group=inbound,
                )
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
            self.create_subscription(
                String, "/ibrobot_agent/voice_control", self._on_voice_control, 10, callback_group=inbound
            )
            self.create_timer(0.2, self._on_tick, callback_group=inbound)
            if self._speaker is not None:
                self._speech_pipeline = SpeechPipeline(
                    synthesize=self._prepare_speech,
                    play=self._render_speech,
                    stop=self._speaker.stop,
                    is_current=self._speech_is_current,
                    completed=lambda action: self._execute(action.then),
                    log=lambda message: self.get_logger().info(message),
                )
            if self._speaker is not None:
                self._speech_transport = threading.Thread(
                    target=self._speech_commands_loop, daemon=True, name="voice-cancellation"
                )
                self._speech_transport.start()
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
            text = str(message.text)
            alignment = TokenAlignment.build(
                getattr(message, "token_starts", ()), getattr(message, "token_ends", ()), text
            )
            self._hear(text, (str(message.source_id), int(message.segment_sequence)), alignment)

        def _on_audio_chunk(self, message: Any) -> None:
            source = str(message.source_id)
            if message.state == message.STATE_BEGIN:
                self._active_audio_segments[source] = int(message.sequence)
                if len(self._active_audio_segments) > 16:
                    del self._active_audio_segments[next(iter(self._active_audio_segments))]
            sequence = self._active_audio_segments.get(source)
            if sequence is not None:
                stamp = message.header.stamp
                age = self.get_clock().now().nanoseconds / 1e9 - (stamp.sec + stamp.nanosec / 1e9)
                captured_at = time.monotonic() - max(0.0, age) if stamp.sec else time.monotonic()
                with self._core_lock:
                    self._core.note_audio((source, sequence), captured_at)
            if message.state == message.STATE_END:
                self._active_audio_segments.pop(source, None)

        def _hear(self, text: str, segment: Segment | None, alignment: TokenAlignment | None = None) -> None:
            if self._core.config.feedback_only:
                return
            # The core holds a transcript that is still waiting for its measurement.
            self._run(lambda now: self._core.on_heard(text, now, segment, alignment))

        def _on_partial(self, message: String) -> None:
            self._run(lambda now: self._core.on_partial(message.data, now))

        def _on_voice_control(self, message: String) -> None:
            if message.data.strip() in {"end_session", "结束这轮", "下一位"}:
                self._run(self._core.end_session)
            elif message.data.strip() == "cancel_speech":
                self._run(lambda now: self._core.cancel_speech())

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

        def _on_mirror_state(self, message: String) -> None:
            payload = self._decode(message.data, "mirror state")
            source_now = self.get_clock().now().nanoseconds / 1e9
            self._run(lambda now: self._core.on_mirror_state(payload or {}, now, source_now=source_now))

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
            windows = WindowVoiceprints.from_flat(
                float(getattr(message, "window_seconds", 0.0)),
                getattr(message, "window_starts", ()),
                getattr(message, "window_embeddings", ()),
                len(embedding),
            )
            voiceprint = Voiceprint(embedding, str(message.model_id), windows) if embedding else None
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
                    if self._speaker is None:
                        self._request_publisher.publish(String(data=json.dumps(action.payload, ensure_ascii=False)))
                    else:
                        # A new recorded introduction shares the global player.
                        # It must not overtake cancellation of old bridge speech.
                        self._speech_commands.put(action)
                elif isinstance(action, PublishControl):
                    self._control_publisher.publish(String(data=json.dumps(action.payload, ensure_ascii=False)))
                elif isinstance(action, CancelSpeech):
                    if self._speaker is not None:
                        self._speech_commands.put(action)
                elif isinstance(action, Speak):
                    if self._speaker is None:
                        self.get_logger().info(f"[say] {action.text}")
                        self._execute(action.then)
                    else:
                        self._speech_commands.put(action)

        def _speech_commands_loop(self) -> None:
            while True:
                action = self._speech_commands.get()
                if action is None:
                    return
                try:
                    if isinstance(action, CancelSpeech):
                        # Queued in order with new requests, so this stop can never reach a later clip.
                        stop = (lambda: self._speaker.stop(force=True)) if action.recorded else self._speaker.stop
                        self._speech_pipeline.cancel(stop=stop, timeout=5.0)
                        self._speech_transport_failed = False
                    elif isinstance(action, PublishRequest):
                        with self._core_lock:
                            current = (
                                action.payload["request_id"] == self._core.active_request_id
                                and action.payload["session_id"] == self._core._state.session_id
                                and (
                                    not self._core._unsafe_request_id
                                    or (
                                        self._core.config.mirror_session_enabled
                                        and action.payload["request_id"] == self._core._mirror_stop_request_id
                                        and action.payload["text"] == "停止照镜子"
                                    )
                                )
                            )
                        if not self._speech_transport_failed and current:
                            self._request_publisher.publish(String(data=json.dumps(action.payload, ensure_ascii=False)))
                    elif not self._speech_transport_failed and self._speech_is_current(action):
                        self._speech_pipeline.submit(action)
                except Exception as error:
                    self._speech_transport_failed = True
                    self.get_logger().error(f"speech cancellation failed; speech held: {error}")

        def _speech_is_current(self, action: Speak) -> bool:
            with self._core_lock:
                return self._core.speech_is_current(action, time.monotonic())

        def destroy_node(self):
            if self._speech_transport is not None:
                self._speech_commands.put(None)
                self._speech_transport.join(timeout=6.0)
            if self._speech_pipeline is not None:
                self._speech_pipeline.close(stop=self._speaker.close)
                self._speech_pipeline = None
            return super().destroy_node()

        def _prepare_speech(self, action: Speak) -> Any:
            """A recorded line when ``action.text`` is one (and its file exists), else the synthesis."""
            path = choose_recorded_line(action.text, self._spoken_clips, self._spoken_last)
            if path:
                return _RecordedAudio(path)
            if spoken_key(action.text) in self._spoken_clips:
                self.get_logger().warning(f"no recorded file left for {action.text!r}; synthesizing it")
            return self._speaker.synthesize(
                _tts_text(action.text, keep_english=self._tts_keep_english, pronunciations=self._pronunciations)
            )

        def _render_speech(self, action: Speak, audio: Any) -> bool:
            text = _tts_text(action.text, keep_english=self._tts_keep_english, pronunciations=self._pronunciations)
            with self._core_lock:
                if not self._core.speech_is_current(action, time.monotonic()):
                    return False
                self._core.note_playback(action, time.monotonic())
                self._core.note_speech(action.text, time.monotonic())
            # Keep recognition live; the core filters actual text echoes, not all audio
            # captured during playback. Cancellation runs on the transport worker.
            try:
                if isinstance(audio, _RecordedAudio):
                    self.get_logger().info(f"[say] {action.text} (recorded {os.path.basename(audio.path)})")
                    spoken = self._speaker.play_file(audio.path)
                else:
                    self.get_logger().info(f"[say] {text}")
                    spoken = self._speaker.play(audio)
            finally:
                with self._core_lock:
                    self._core.note_speech(action.text, time.monotonic(), finished=True, gate_epoch=action.gate_epoch)
            return spoken

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
