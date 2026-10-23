"""Hermes-native response adapter. Installed alongside its pure-Python contract."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from html import escape
from pathlib import Path

from .hermes_response import ResponseFormatError, SpeechEvent, parse_response
from .hermes_speech_delivery import event_is_current, finish_speech_turn
from .stop_contract import is_stop

logger = logging.getLogger(__name__)


class ResponseAdapter:
    def __init__(self, *, speaker: str, protocol: str, timeout: float = 300.0) -> None:
        self.speaker = speaker
        self.protocol = protocol
        self.timeout = timeout
        # pre/transform/post are synchronous callbacks on the owning turn's
        # thread. Never share pending responses across simultaneous sessions.
        self._turn = threading.local()
        self._identities: dict[str, set[str]] = {}
        self._identity_lock = threading.Lock()

    def begin_turn(self, session_id: str = "", turn_id: str = "", **_kwargs):
        self._turn.identity = (session_id, turn_id)
        self._turn.pending = None
        if session_id and turn_id:
            with self._identity_lock:
                self._identities.setdefault(session_id, set()).add(turn_id)
        # Also refresh resumed sessions whose persisted system prompt predates
        # this protocol. The contract text still has a single published source.
        return {"context": self.protocol}

    def transform(self, response_text: str, session_id: str = "", **_kwargs) -> str:
        self._turn.pending = None
        identity = getattr(self._turn, "identity", None)
        if identity and identity[0] == session_id:
            finish_speech_turn(*identity)
        try:
            response = parse_response(response_text)
        except ResponseFormatError as exc:
            logger.warning("IBROBOT_RESPONSE_INVALID session=%s reason=%s", session_id, exc)
            return (
                "回复格式不完整，本次未进行语音播报。\n\n"
                f"<details><summary>原始回复</summary><pre>{escape(response_text)}</pre></details>"
            )
        rendered = response.render() or "（本回合无回复内容）"
        if identity and identity[0] == session_id and identity[1]:
            self._turn.pending = (identity, response, rendered)
        else:
            logger.warning("IBROBOT_RESPONSE_MISSING_TURN session=%s", session_id)
        return rendered

    def _speak(self, event: SpeechEvent) -> None:
        if not self.speaker or not event.response.speech:
            return
        try:
            result = subprocess.run(
                [self.speaker],
                input=json.dumps(event.to_dict(), ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode:
                logger.warning(
                    "IBROBOT_SPEECH_FAILED session=%s turn=%s phase=%s exit=%s",
                    event.session_id,
                    event.turn_id,
                    event.phase,
                    result.returncode,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("IBROBOT_SPEECH_FAILED session=%s reason=%s", event.session_id, type(exc).__name__)

    def finish_turn(self, session_id: str = "", turn_id: str = "", assistant_response: str = "", **_kwargs):
        pending = getattr(self._turn, "pending", None)
        self._turn.pending = None
        # Standard post_llm_call must reclaim state even without the optional
        # control adapter, including invalid replies and failed speech delivery.
        self.end_session(session_id=session_id, turn_id=turn_id)
        if pending is None:
            return
        identity, response, rendered = pending
        if identity != (session_id, turn_id) or rendered != assistant_response:
            logger.warning("IBROBOT_RESPONSE_IDENTITY_MISMATCH session=%s turn=%s", session_id, turn_id)
            return
        self._speak(SpeechEvent(session_id, turn_id, "final", response))

    def interim(self, text: str = "", session_id: str = "", turn_id: str = "", iteration: int = 0, **_kwargs):
        # Hermes invokes this observer off the control path. Ignore incomplete
        # streamed JSON, plain narration and events without a turn identity.
        if not session_id or not turn_id:
            return
        try:
            response = parse_response(text)
        except ResponseFormatError:
            return
        if type(iteration) is not int or not 0 <= iteration <= 100000:
            return
        event = SpeechEvent(session_id, turn_id, "interim", response, schema_version=2, sequence=iteration)
        if event_is_current(event):
            self._speak(event)

    def end_session(self, session_id: str = "", turn_id: str = "", **_kwargs):
        """Close the exact inner turn on Hermes' standard on_session_end hook."""
        identity = (session_id, turn_id)
        try:
            finish_speech_turn(*identity)
        finally:
            with self._identity_lock:
                turns = self._identities.get(session_id)
                if turns is not None:
                    turns.discard(turn_id)
                    if not turns:
                        self._identities.pop(session_id, None)
            if getattr(self._turn, "identity", None) == identity:
                self._turn.identity = None
                self._turn.pending = None

    def end_turn(self, session_id: str = "", turn_id: str = "", **_kwargs):
        # The outer control adapter uses the relay turn ID; speech hooks use
        # Hermes' inner conversation turn ID. Close the identity actually sent.
        identity = getattr(self._turn, "identity", None)
        if identity and identity[0] == session_id:
            self.end_session(session_id=identity[0], turn_id=identity[1])
        else:
            finish_speech_turn(session_id, turn_id)

    def stop_input(self, text: str = "", session_id: str = "", **_kwargs):
        if not is_stop(text):
            return
        with self._identity_lock:
            turns = tuple(self._identities.get(session_id, ()))
        for turn_id in turns:
            finish_speech_turn(session_id, turn_id)


def register(ctx) -> None:
    protocol = Path(__file__).with_name("RESPONSE_PROTOCOL.md").read_text(encoding="utf-8")
    adapter = ResponseAdapter(
        speaker=str(ctx.get_config("speaker", "")),
        protocol=protocol,
        timeout=float(ctx.get_config("timeout_sec", 300.0)),
    )
    if len(protocol) > 4000:
        raise ValueError("IB-Robot response protocol exceeds Hermes prompt-section limit")
    ctx.register_system_prompt_section("ibrobot.response.v1", protocol, position="after_memory", max_chars=4000)
    ctx.register_hook("pre_llm_call", adapter.begin_turn)
    ctx.register_hook("transform_llm_output", adapter.transform)
    ctx.register_hook("post_llm_call", adapter.finish_turn)
    ctx.register_hook("on_interim_message", adapter.interim)
    ctx.register_hook("on_session_end", adapter.end_session)
    ctx.register_hook("on_control_turn_end", adapter.end_turn)
    ctx.register_hook("on_control_input", adapter.stop_input)
