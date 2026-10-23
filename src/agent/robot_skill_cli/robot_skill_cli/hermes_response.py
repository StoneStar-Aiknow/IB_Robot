"""Versioned Hermes display/speech contract, independent of ROS and Hermes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from typing import Any

SCHEMA_VERSION = 1
MAX_SPEECH_CHARS = 300
MAX_DETAILS_CHARS = 16000
PHASES = frozenset({"final", "interim", "status_check_started", "planning_started", "plan_confirmed"})


class ResponseFormatError(ValueError):
    """A response cannot be routed without guessing its intended content."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseFormatError(f"duplicate field: {key}")
        result[key] = value
    return result


def load_json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ResponseFormatError("expected one complete JSON object") from exc


@dataclass(frozen=True)
class HermesResponse:
    speech: str
    details: str = ""
    schema_version: int = SCHEMA_VERSION
    completion: str = "continue"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.schema_version == 1:
            value.pop("completion")
        return value

    def render(self) -> str:
        parts = [self.speech] if self.speech else []
        if self.details:
            # Escape details so a model-generated closing tag cannot escape
            # the display-only region or inject markup into a chat client.
            parts.append(f"<details><summary>执行详情</summary>\n<pre>{escape(self.details)}</pre>\n</details>")
        return "\n\n".join(parts)


def parse_response(value: Any) -> HermesResponse:
    if isinstance(value, str):
        if len(value) > 100000:
            raise ResponseFormatError("response exceeds transport limit")
        value = load_json(value)
    if not isinstance(value, dict):
        raise ResponseFormatError("expected a response object")
    version = value.get("schema_version")
    if type(version) is not int or version not in {1, 2}:
        raise ResponseFormatError("unsupported response schema_version")
    fields = {"schema_version", "speech", "details"}
    if version == 2:
        fields.add("completion")
    if set(value) != fields:
        raise ResponseFormatError("unexpected response fields")
    completion = value.get("completion", "continue")
    if not isinstance(completion, str) or completion not in {"continue", "after_workflow"}:
        raise ResponseFormatError("unsupported completion policy")
    speech, details = value["speech"], value["details"]
    if not isinstance(speech, str) or not isinstance(details, str):
        raise ResponseFormatError("speech and details must be strings")
    if len(speech) > MAX_SPEECH_CHARS or len(details) > MAX_DETAILS_CHARS:
        raise ResponseFormatError("response field exceeds character limit")
    if any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in speech):
        raise ResponseFormatError("speech must be a single plain-text paragraph")
    if any(char in speech for char in "<>`{}"):
        raise ResponseFormatError("markup and structured data belong in details")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in details):
        raise ResponseFormatError("details contain invalid Unicode")
    return HermesResponse(speech=speech.strip(), details=details.strip(), schema_version=version, completion=completion)


@dataclass(frozen=True)
class SpeechEvent:
    session_id: str
    turn_id: str
    phase: str
    response: HermesResponse
    schema_version: int = SCHEMA_VERSION
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["response"] = self.response.to_dict()
        if self.schema_version == 1:
            value.pop("sequence")
        return value


def parse_event(value: Any) -> SpeechEvent:
    if not isinstance(value, dict):
        raise ResponseFormatError("expected a structured speech event")
    version = value.get("schema_version")
    if type(version) is not int or version not in {1, 2}:
        raise ResponseFormatError("unsupported speech event schema_version")
    fields = {"schema_version", "session_id", "turn_id", "phase", "response"}
    if version == 2:
        fields.add("sequence")
    if set(value) != fields:
        raise ResponseFormatError("unexpected speech event fields")
    sequence = value.get("sequence", 0)
    if type(sequence) is not int or not 0 <= sequence <= 100000:
        raise ResponseFormatError("invalid speech sequence")
    for name in ("session_id", "turn_id", "phase"):
        text = value[name]
        if not isinstance(text, str) or not text or len(text) > 256 or any(ord(c) < 32 for c in text):
            raise ResponseFormatError(f"invalid event {name}")
    if value["phase"] not in PHASES:
        raise ResponseFormatError("unknown speech phase")
    return SpeechEvent(
        value["session_id"], value["turn_id"], value["phase"], parse_response(value["response"]), version, sequence
    )
