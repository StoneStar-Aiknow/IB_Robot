"""Decide whether a transcript is addressed to the robot (the semantic gate, L3 of addressing).

``addressee.py`` answers "did somebody wake me, and is this the same voice". It cannot
tell "灵犀帮我挥挥手" from a colleague saying "回到抓取的初始状态": both are speech,
one is meant for the robot. This module scores the text itself with a small hashed
n-gram model trained on field ASR traffic, and vetoes only what is *confidently* not
addressed to the robot:

    reject  <=>  score < threshold  and no skill word  and no wake prefix

The veto direction is deliberate. Leaving-one-skill-out tests on the field corpus show
the model recalls an *unseen* skill's phrasing only 45-67% of the time, so a model that
also decided "yes" would drop half of the commands given to a newly added skill. Skill
words (from the catalog's ``aliases_zh``) and wake-word variants therefore win over the
model, while stop words, confirmations and cancellations never reach this gate at all
(``VoiceBridgeCore`` routes those first, and safety outruns addressing).

Everything here is ROS-free and standard-library only: the model is a JSON file of
sparse FNV-1a hashed weights, scoring one sentence takes tens of microseconds, and the
board needs no extra dependency. ``tools/addressee`` trains and exports that file; the
format is versioned so a stale model is refused instead of silently mis-scoring.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

#: ``off``: not built at all. ``shadow``: judge and record, never drop. ``enforce``: drop.
SEMANTIC_GATE_MODES = ("off", "shadow", "enforce")

_SCHEMA_VERSION = 1
_SUPPORTED_HASHES = ("fnv1a32",)
_PUNCT = re.compile(r"[\s，。！？、,.!?~～\"'“”‘’()（）\[\]【】:：;；\-—…]+")
#: Clamp for the logit, so an absurd weight cannot overflow ``exp``.
_LOGIT_LIMIT = 30.0
#: Address terms that point at someone else in the room. The field log caught this
#: failure first: "周哥举手的动作" scored 1.000 on the skill word 举手 although it was
#: said to a colleague. The model cannot know who 周哥 is, but a microphone transcript
#: that names a person in the room is evidence about the addressee, not the vocabulary.
_ADDRESS_TERMS = (
    "哥",
    "姐",
    "弟",
    "妹",
    "老师",
    "师傅",
    "老板",
    "经理",
    "主任",
    "同学",
    "各位",
    "大家",
    "你们",
    "他们",
    "咱们",
    "兄弟",
    "姐妹",
    "同志",
    "领导",
    "阿姨",
    "叔叔",
)
#: A skill word only counts as evidence in a command-sized sentence, or when the model
#: agrees at all: "这不是没开始吗智商没开始吗" carries 开始 and 镜子 and scored 0.000.
_LEXICON_MAX_BODY_CHARS = 10
#: Fragments this short carry no grammar, so the score is mostly the bias ("怎么" 0.999);
#: they only pass inside a session the addressee gate opened. Three characters still name
#: a skill ("挥挥手", "点点头"), which is why the line sits at three, not four.
_FRAGMENT_BODY_CHARS = 3


def _fnv1a32(text: str) -> int:
    """FNV-1a over UTF-8 bytes; the hash is part of the model format, never Python's ``hash()``."""
    digest = 0x811C9DC5
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x01000193) & 0xFFFFFFFF
    return digest


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/whitespace, matching the training pipeline."""
    return _PUNCT.sub("", str(text).strip().lower())


def strip_wake_words(text: str, wake_words: tuple[str, ...]) -> str:
    """Drop leading wake-word spellings (the ASR's own variants), twice at most.

    Under ``addressee_trigger=wake_phrase`` the transcript still carries the wake
    phrase; under ``vendor_session`` the vendor already consumed it. Training strips
    it too, so the gate scores the same body of text either way.
    """
    for _ in range(2):
        for wake in wake_words:
            # ``>=`` on purpose: a transcript that is only the wake word strips to
            # nothing, which the gate reads as "nothing to judge" and lets through.
            if wake and text.startswith(wake) and len(text) >= len(wake):
                text = text[len(wake):]
                break
        else:
            break
    return text


@dataclass(frozen=True)
class SemanticDecision:
    """One gate verdict, kept whole so the shadow log can be replayed as training data."""

    text: str
    score: float
    #: The model alone would reject this utterance; ``mode`` decides whether it is dropped.
    rejected: bool
    lexicon_hit: bool
    wake_prefix: bool
    reason: str
    #: The address term that made this "someone talking to someone else", if any.
    address_term: str = ""

    @property
    def accepted(self) -> bool:
        return not self.rejected


class SemanticGate:
    """Score transcripts against a trained model and veto the ones not addressed to the robot."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        threshold: float,
        lexicon: tuple[str, ...] = (),
        dialogue_requests: tuple[str, ...] = (),
        wake_prefixes: tuple[str, ...] = (),
        source: str = "",
    ) -> None:
        feature = payload.get("feature")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic model schema_version {payload.get('schema_version')!r}")
        if not isinstance(feature, dict):
            raise ValueError("semantic model has no feature section")
        if feature.get("hash") not in _SUPPORTED_HASHES:
            raise ValueError(f"unsupported semantic model hash {feature.get('hash')!r}")
        try:
            self._ngrams = tuple(int(item) for item in feature["ngrams"])
            self._dim = int(feature["dim"])
            self._bias = float(payload["bias"])
            self._weights = {int(key): float(value) for key, value in payload["weights"]}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed semantic model: {error}") from error
        if not self._ngrams or self._dim < 2:
            raise ValueError("semantic model needs at least one n-gram and a dimension of 2 or more")
        self._pad = str(feature.get("pad", "<>")) or "<>"
        self._wake_words = tuple(str(item) for item in payload.get("wake_words", ()))
        self.threshold = float(threshold)
        self.lexicon = tuple(item for item in (normalize_text(str(entry)) for entry in lexicon) if item)
        self.dialogue_requests = tuple(
            item for item in (normalize_text(str(entry)) for entry in dialogue_requests) if item
        )
        self.wake_prefixes = tuple(
            item for item in (normalize_text(str(entry)) for entry in wake_prefixes) if item
        )
        self.source = source
        self.created_at = str(payload.get("created_at", ""))
        self.training = dict(payload.get("training", {}))

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> SemanticGate:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("semantic model must be a JSON object")
        return cls(payload, source=path, **kwargs)

    @classmethod
    def from_config(cls, config: Any) -> tuple[SemanticGate | None, str]:
        """Build the gate described by ``config``; returns the gate and a status line for the log.

        ``off`` builds nothing. A model that cannot be read is reported, and then either
        disabled (``semantic_gate_fail_open``, the default: the bridge keeps working exactly
        as it did before this gate existed) or raised, which fails the launch early.
        """
        mode = str(config.semantic_gate_mode)
        if mode not in SEMANTIC_GATE_MODES:
            raise ValueError(f"semantic_gate_mode must be one of {SEMANTIC_GATE_MODES}")
        if mode == "off":
            return None, "semantic gate off"
        path = str(config.semantic_gate_model_path)
        try:
            gate = cls.load(
                path,
                threshold=float(config.semantic_gate_threshold),
                lexicon=tuple(config.semantic_gate_lexicon),
                dialogue_requests=tuple(config.semantic_gate_dialogue_requests),
                wake_prefixes=tuple(config.semantic_gate_wake_prefixes),
            )
        except (OSError, ValueError) as error:
            if not config.semantic_gate_fail_open:
                raise
            return None, f"semantic gate {mode} disabled: {error}"
        return gate, (
            f"semantic gate {mode}: {path} (threshold={gate.threshold:.2f}, "
            f"lexicon={len(gate.lexicon)}, trained={gate.created_at or 'unknown'})"
        )

    def _body(self, text: str) -> str:
        return strip_wake_words(normalize_text(text), self._wake_words)

    def features(self, text: str) -> Counter[int]:
        padded = f"{self._pad}{self._body(text)}{self._pad}"
        counts: Counter[int] = Counter()
        for size in self._ngrams:
            for start in range(len(padded) - size + 1):
                counts[_fnv1a32(padded[start:start + size]) % self._dim] += 1
        return counts

    def score(self, text: str) -> float:
        """Probability-like score in (0, 1) that ``text`` is addressed to the robot."""
        logit = self._bias
        for key, count in self.features(text).items():
            weight = self._weights.get(key)
            if weight:
                logit += weight * (1.0 + math.log(count))
        logit = max(-_LOGIT_LIMIT, min(_LOGIT_LIMIT, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def judge(self, text: str, *, session_open: bool = False) -> SemanticDecision:
        """Decide whether ``text`` is addressed to the robot; stop/confirm never come here.

        A skill word is evidence only in a command-sized sentence or when the model does
        not flatly disagree, a sentence naming someone else in the room is not addressed
        to the robot, and a bare fragment needs a session the addressee gate opened --
        the three failure classes the first field session produced on 2026-11-07.
        """
        body = self._body(text)
        raw_normalized = normalize_text(text)
        score = self.score(text)
        lexicon_hit = any(word in raw_normalized or word in body for word in self.lexicon)
        wake_prefix = any(raw_normalized.startswith(prefix) for prefix in self.wake_prefixes)
        address_term = next((term for term in _ADDRESS_TERMS if term in raw_normalized), "")
        if not body:
            reason = "empty after wake-word stripping"
            return SemanticDecision(text, score, False, lexicon_hit, wake_prefix, reason)
        if address_term and not wake_prefix:
            reason = f"names someone else ({address_term})"
            return SemanticDecision(text, score, True, lexicon_hit, wake_prefix, reason, address_term)
        rejected = False
        # A request to talk is addressed to whatever can talk back, and it can only ever
        # start a conversation, so it outranks the model instead of arguing with it.
        request_hit = next((phrase for phrase in self.dialogue_requests if phrase in raw_normalized), "")
        if request_hit:
            reason = "dialogue request"
        elif lexicon_hit and (len(body) <= _LEXICON_MAX_BODY_CHARS or score >= self.threshold):
            reason = "skill word"
        elif wake_prefix:
            reason = "wake prefix"
        elif score >= self.threshold:
            if len(body) < _FRAGMENT_BODY_CHARS and not session_open:
                reason = f"fragment of {len(body)} chars outside a session"
                rejected = True
            else:
                reason = "model"
        else:
            reason = f"score {score:.3f} below {self.threshold:.2f}"
            rejected = True
        return SemanticDecision(text, score, rejected, lexicon_hit, wake_prefix, reason, address_term)

    def record(self, decision: SemanticDecision, path: str, *, mode: str, **context: Any) -> None:
        """Append one JSON line describing ``decision``; never raises into the bridge.

        The line is the training signal for the next model: ``score`` and ``reason`` say
        what the gate thought, ``context`` (session, whether a wake event just opened it,
        whether the robot was speaking) says what the bridge knew, which the field corpus
        does not carry.
        """
        if not path:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": mode,
            "text": decision.text,
            "score": round(float(decision.score), 6),
            "reason": decision.reason,
            "lexicon_hit": decision.lexicon_hit,
            "wake_prefix": decision.wake_prefix,
            "rejected": decision.rejected,
            "model": os.path.basename(self.source),
            **context,
        }
        if decision.address_term:
            entry["address_term"] = decision.address_term
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # A full or read-only disk must not take the voice channel down with it.
            return
