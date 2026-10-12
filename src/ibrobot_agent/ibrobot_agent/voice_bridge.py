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
from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE, TERMINAL_REQUEST_STATES
from ibrobot_agent.presentation import presentation_digest

CONFIRM_MODES = ("auto", "voice_confirm")
_ACTIVE_ASR_STATES = {"listening", "recognizing"}
#: A refusal can contain a confirm word ("不执行" has 执行, "我不确定" has 确定); it must never
#: acknowledge a plan. "没" is left out on purpose: "没问题" is consent.
_NEGATION_MARKERS = ("不", "别")
_PHRASE_FIELDS = ("stop_phrases", "bypass_phrases", "confirm_phrases", "cancel_phrases")


@dataclass(frozen=True)
class VoiceBridgeConfig:
    channel_id: str = "voice_asr"
    principal_id: str = "local_operator"
    # Robot-specific and part of the Agent request key: always given by the launch builder.
    robot_scope: str = ""
    stop_phrases: tuple[str, ...] = STOP_PHRASES
    bypass_phrases: tuple[str, ...] = ()
    # Without knowing about TTS, the safe default is a spoken confirmation (see resolve_confirm_mode).
    confirm_mode: str = "voice_confirm"
    confirm_phrases: tuple[str, ...] = ("确认", "确定", "好的", "执行")
    cancel_phrases: tuple[str, ...] = ("取消", "算了", "不要")
    confirm_timeout_sec: float = 20.0
    session_idle_sec: float = 120.0
    min_text_chars: int = 2
    request_timeout_sec: float = 180.0
    speak_replies: bool = True
    # Spoken while a request runs and the microphone stays open for stop words, so it must not
    # contain one (the bridge would hear itself and stop the robot).
    busy_message: str = "我正在执行上一个任务，请稍等。"
    # Domain term -> homophones the ASR model prefers (e.g. 夹爪 jiā zhǎo is heard as 家找/家长).
    text_corrections: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not str(self.robot_scope).strip():
            raise ValueError("robot_scope must be the Agent's robot scope")
        if self.confirm_mode not in CONFIRM_MODES:
            raise ValueError(f"confirm_mode must be one of {CONFIRM_MODES}")
        for name in ("confirm_timeout_sec", "session_idle_sec", "request_timeout_sec"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be positive")
        if self.min_text_chars < 1:
            raise ValueError("min_text_chars must be at least 1")
        for name in _PHRASE_FIELDS:
            object.__setattr__(self, name, tuple(_normalize(item) for item in getattr(self, name) if _normalize(item)))
        if not self.stop_phrases:
            raise ValueError("stop_phrases must not be empty")
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


@dataclass(frozen=True)
class Log:
    message: str


Action = PublishRequest | PublishControl | Speak | Log


@dataclass
class _PendingConfirmation:
    request_id: str
    acknowledgement: PublishControl
    deadline: float


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
    pending: _PendingConfirmation | None = None


def _normalize(text: str) -> str:
    return normalize_trigger_text(str(text))


def _new_session_id() -> str:
    return f"voice-session-{uuid.uuid4().hex[:12]}"


class VoiceBridgeCore:
    """Protocol state machine for one voice channel; not thread-safe."""

    def __init__(self, config: VoiceBridgeConfig, *, id_factory: Callable[[], str] | None = None) -> None:
        self.config = config
        self._state = _BridgeState()
        self._id_factory = id_factory or (lambda: f"voice-{uuid.uuid4().hex[:12]}")

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

    def on_transcript(self, text: str, now: float) -> list[Action]:
        stopped_by_partial, self._state.utterance_stopped = self._state.utterance_stopped, False
        heard = str(text).strip()
        raw = self.config.correct(heard)
        normalized = _normalize(raw)
        if not normalized:
            return []
        corrected = [Log(f"corrected transcript {heard!r} -> {raw!r}")] if raw != heard else []
        if stopped_by_partial and self._contains_stop(normalized):
            # Once the stop has finished the request, this sentence would otherwise reach the
            # Agent as a new command and occupy the channel ("帮我停一下" -> busy for "继续").
            return [*corrected, Log(f"final transcript {raw!r} of a stop already sent from its partial")]
        actions = self._route_transcript(raw, normalized, now)
        return corrected + actions if actions else corrected

    def on_partial(self, text: str, now: float) -> list[Action]:
        """Stop as soon as a partial transcript contains a stop keyword; ignore everything else.

        Final transcripts only arrive after VAD silence plus tail padding, which is
        seconds of extra motion when the user says "停".
        """
        if not self.is_stop_text(text):
            return []
        if self._state.utterance_stopped or not self._stop_target():
            return []
        self._state.utterance_stopped = True
        return self._stop(now, reason=f"stop keyword in partial {text!r}")

    def _contains_stop(self, normalized: str) -> bool:
        # Keyword, not whole-sentence, matching: "帮我停一下" and "别动了谢谢" must interrupt.
        # Accepted trade-off: an utterance like "不停地挥手" also stops the robot.
        return any(phrase in normalized for phrase in self.config.stop_phrases)

    def _route_transcript(self, raw: str, normalized: str, now: float) -> list[Action]:
        state = self._state
        if self._contains_stop(normalized) and (self._stop_target() or normalized in self.config.stop_phrases):
            # With nothing to interrupt, a sentence that merely contains a keyword is a command.
            return self._stop(now, reason=f"stop keyword in {raw!r}")
        if state.pending is not None:
            return self._answer_confirmation(normalized, raw, now)
        if any(phrase in normalized for phrase in self.config.bypass_phrases):
            return [Log(f"ignored bypass utterance {raw!r}")]
        if state.active_request_id:
            return [Log(f"busy with {state.active_request_id}; dropped {raw!r}"), *self._say(self.config.busy_message)]
        if len(normalized) < self.config.min_text_chars:
            return [Log(f"ignored short utterance {raw!r}")]

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
        if state not in TERMINAL_REQUEST_STATES or not self._finish(request_id):
            return []
        if event_type == "needs_clarification":
            self._state.clarification_request_id = request_id
        message = str(payload.get("user_message", "")).strip()
        return [Log(f"request {request_id} {event_type} {state}: {message}"), *self._say(_spoken_reply(state, message))]

    def tick(self, now: float) -> list[Action]:
        state = self._state
        actions: list[Action] = []
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
        if detail.get("async_display") is True or presentation.get("execution_mode") == ASYNC_PRESENTATION_MODE:
            # Typed Programs execute without waiting for any client: there is no receipt to give, and
            # asking for a confirmation that cannot hold the robot back would mislead the user.
            gate = (
                [Log(f"voice_confirm cannot gate {request_id}: the Agent executes typed Programs without a receipt")]
                if self.config.confirm_mode == "voice_confirm"
                else []
            )
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
        return [Log(f"{reason}: stopping {target}"), PublishControl({"operation": "stop", "request_id": target})]

    def _finish(self, request_id: str) -> bool:
        """Release ``request_id``; return False when it was already finished."""
        if request_id in self._state.finished_request_ids:
            return False
        self._state.finished_request_ids.add(request_id)
        if self._state.active_request_id == request_id:
            self._state.active_request_id = ""
        if self._state.timed_out_request_id == request_id:
            self._state.timed_out_request_id = ""
        if self._state.pending is not None and self._state.pending.request_id == request_id:
            self._state.pending = None
        return True

    def _say(self, text: str) -> list[Action]:
        if not text or not self.config.speak_replies:
            return []
        return [Speak(text)]


def _spoken_reply(state: Any, message: str) -> str:
    """Execution terminals carry technical text; answers are already user-facing."""
    if state == "SUCCEEDED":
        return "执行完成。"
    if state in {"CANCELLED", "CANCELLED_BEFORE_EXECUTION"}:
        return "已停止。"
    if state == "UNKNOWN":
        return "执行结果未知，请人工检查机器人状态。"
    if state == "FAILED":
        # Gateway/planner failures carry English technical text (allowlist, registry, RPC);
        # it stays in the log while the user hears one actionable Chinese sentence.
        return "执行失败，详情见日志。"
    return message


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
        confirm_timeout_sec=float(values["confirm_timeout_sec"]),
        session_idle_sec=float(values["session_idle_sec"]),
        min_text_chars=int(values["min_text_chars"]),
        request_timeout_sec=float(values["request_timeout_sec"]),
        speak_replies=bool(values["speak_replies"]),
        busy_message=str(values["busy_message"]),
        text_corrections=_corrections(values.get("text_corrections_json", "{}")),
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
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.result() if future.done() else None


def _build_node_class():
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from std_msgs.msg import String

    class VoiceBridgeNode(Node):
        """ROS transport for :class:`VoiceBridgeCore`."""

        def __init__(self) -> None:
            super().__init__("agent_voice_bridge")
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
            self.create_subscription(
                String, str(values["input_topic"]), self._on_transcript, 10, callback_group=inbound
            )
            self.create_subscription(
                String, str(values["response_topic"]), self._on_response, 10, callback_group=inbound
            )
            self.create_subscription(String, str(values["event_topic"]), self._on_event, 10, callback_group=inbound)
            if values["partial_topic"]:
                self.create_subscription(
                    String, str(values["partial_topic"]), self._on_partial, 10, callback_group=inbound
                )
            self.create_subscription(
                String, str(values["asr_status_topic"]), self._on_asr_status, 10, callback_group=inbound
            )
            self.create_timer(0.2, self._on_tick, callback_group=inbound)
            self._speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
            self._speech_thread.start()
            confirm_mode = self._core.config.confirm_mode
            self.get_logger().info(
                f"voice bridge ready: {values['input_topic']} -> {values['request_topic']} "
                f"(channel={values['channel_id']}, confirm_mode={confirm_mode}, "
                f"tts={'on' if self._speaker else 'off'})"
            )
            if self._speaker is None:
                self.get_logger().warning(
                    "TTS is off: each exact plan is rendered only to this log; nobody hears it before the robot moves"
                )

        def _on_transcript(self, message: String) -> None:
            now = time.monotonic()
            if now < self._deaf_until or (now < self._echo_until and not self._core.is_stop_text(message.data)):
                self.get_logger().info(f"ignored transcript heard during playback: {message.data!r}")
                return
            self._run(lambda now: self._core.on_transcript(message.data, now))

        def _on_partial(self, message: String) -> None:
            if time.monotonic() < self._deaf_until:
                return
            self._run(lambda now: self._core.on_partial(message.data, now))

        def _on_response(self, message: String) -> None:
            payload = self._decode(message.data, "response")
            if payload is not None:
                self._run(lambda now: self._core.on_response(payload, now))

        def _on_event(self, message: String) -> None:
            payload = self._decode(message.data, "event")
            if payload is not None:
                self._run(lambda now: self._core.on_event(payload, now))

        def _on_asr_status(self, message: String) -> None:
            self._asr_state = message.data

        def _on_tick(self) -> None:
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
                            self._speech_queue.append(action)
                            self._speech_wakeup.notify()

        def _speech_loop(self) -> None:
            while True:
                with self._speech_wakeup:
                    while not self._speech_queue:
                        self._speech_wakeup.wait()
                    action = self._speech_queue.pop(0)
                try:
                    self._render_speech(action)
                except Exception as error:
                    # This thread is the only consumer of the speech queue. Letting one failed
                    # render end it would silence every later reply and withhold the plan
                    # acknowledgements chained behind them while the node still looks alive.
                    dropped = "; dependent actions dropped" if action.then else ""
                    self.get_logger().error(f"speech rendering failed: {error!r}{dropped}")

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
