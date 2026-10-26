"""The ``voice`` section of the X2 mirror session config, shared by its builder and the voice node.

The mirror session's launch builder (robot_config) checks the section at
launch and, when it is wrong, warns and starts no voice node; the voice node
(manipulation_execution) receives the same section as one JSON parameter and
checks it again. robot_config does not depend on manipulation_execution, so
both use this module. Keep it dependency-free: importing it must not pull in
any launch builder or ROS package.

``enabled``    start the voice node
``queue_max``  phrases waiting to be spoken, at most (1 to ``QUEUE_MAX_LIMIT``)
``phrases``    for each event in ``EVENTS``, a non-empty list of phrases; an
               event left out keeps its default. ZipVoice fails on ASCII
               letters, so a phrase must not contain any.
``utterance_topic``  empty (the default: transition phrases only, today's
               behaviour) or an absolute ROS name — the visual interaction
               utterance topic this voice also speaks for, on demand rather
               than from the phrase cache.

A section accepts only its known keys, so a misspelt key fails at launch
instead of being ignored. Every error is a ``ValueError`` that starts with the
path of the bad value, e.g. ``mirror_session.voice.phrases.ready[0]``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

NODE_NAME = "mirror_voice"

# The spoken events, in the order a round goes through them.
EVENTS = ("starting", "ready", "mirroring", "goodbye")
DEFAULT_PHRASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "starting": ("启动中",),
        "ready": ("我准备好咯",),
        "mirroring": ("开始模仿啦",),
        "goodbye": ("下次再见",),
    }
)

DEFAULT_QUEUE_MAX = 8
QUEUE_MAX_LIMIT = 64

# The synthesis service missing this long since the node first looked: one
# ERROR, then one more every repeat period until it appears.
SERVICE_WAIT_ERROR_SEC = 60.0
SERVICE_WAIT_REPEAT_SEC = 30.0

SECTION_PATH = "mirror_session.voice"
_KEYS = ("enabled", "queue_max", "phrases", "utterance_topic")
_ASCII_LETTER = re.compile(r"[A-Za-z]")


def has_ascii_letters(text: str) -> bool:
    """Whether ZipVoice cannot speak ``text``: it fails on ASCII letters.

    The phrase check above refuses such phrases at launch; an utterance arrives
    at run time, so its consumer filters with this instead.
    """
    return bool(_ASCII_LETTER.search(text))


@dataclass(frozen=True)
class VoiceSection:
    enabled: bool = False
    queue_max: int = DEFAULT_QUEUE_MAX
    # Every event in EVENTS order; read-only.
    phrases: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: DEFAULT_PHRASES)
    # The visual interaction utterance topic this voice also speaks for; empty
    # when the deployment has no such feature (transition phrases only).
    utterance_topic: str = ""


def parse_voice_section(section: Mapping[str, Any] | None) -> VoiceSection:
    """The section over the defaults; ``None`` is a section left out (voice off)."""
    if section is None:
        return VoiceSection()
    if not isinstance(section, Mapping):
        raise ValueError(f"{SECTION_PATH}: must be a mapping")
    unknown = sorted(str(key) for key in set(section) - set(_KEYS))
    if unknown:
        raise ValueError(f"{SECTION_PATH}: unknown keys {unknown}; known: {list(_KEYS)}")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{SECTION_PATH}.enabled: must be true or false")
    queue_max = section.get("queue_max", DEFAULT_QUEUE_MAX)
    if isinstance(queue_max, bool) or not isinstance(queue_max, int) or not 1 <= queue_max <= QUEUE_MAX_LIMIT:
        raise ValueError(f"{SECTION_PATH}.queue_max: must be an integer from 1 to {QUEUE_MAX_LIMIT}")
    utterance_topic = section.get("utterance_topic", "")
    if not isinstance(utterance_topic, str) or (utterance_topic and not utterance_topic.startswith("/")):
        raise ValueError(f"{SECTION_PATH}.utterance_topic: must be empty or an absolute ROS name")
    return VoiceSection(enabled, queue_max, _phrases(section.get("phrases")), utterance_topic)


def _phrases(value: Any) -> Mapping[str, tuple[str, ...]]:
    path = f"{SECTION_PATH}.phrases"
    if value is None:
        return DEFAULT_PHRASES
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: must be a mapping from event to phrases")
    unknown = sorted(str(key) for key in set(value) - set(EVENTS))
    if unknown:
        raise ValueError(f"{path}: unknown events {unknown}; known: {list(EVENTS)}")
    phrases = {}
    for event in EVENTS:
        if event not in value:
            phrases[event] = DEFAULT_PHRASES[event]
            continue
        texts = value[event]
        if not isinstance(texts, list | tuple) or not texts:
            raise ValueError(f"{path}.{event}: must be a non-empty list of phrases")
        checked = []
        for index, text in enumerate(texts):
            where = f"{path}.{event}[{index}]"
            if not isinstance(text, str):
                raise ValueError(f"{where}: must be a string, not {type(text).__name__}")
            text = text.strip()
            if not text:
                raise ValueError(f"{where}: must not be blank")
            if _ASCII_LETTER.search(text):
                raise ValueError(f"{where}: {text!r} has ASCII letters, which ZipVoice cannot speak")
            checked.append(text)
        phrases[event] = tuple(checked)
    return MappingProxyType(phrases)


def to_json(section: VoiceSection) -> str:
    """The section as the voice node's one string parameter."""
    data = {
        "enabled": section.enabled,
        "queue_max": section.queue_max,
        "phrases": {event: list(texts) for event, texts in section.phrases.items()},
        "utterance_topic": section.utterance_topic,
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def from_json(text: str) -> VoiceSection:
    """``to_json`` read back, checked again as ``parse_voice_section`` does."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SECTION_PATH}: not a JSON section ({exc})") from exc
    return parse_voice_section(data)
