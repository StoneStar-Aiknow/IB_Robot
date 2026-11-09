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

Conversation admission (``conversation=True``) adds three ways in that need no skill word
and no model score, because the corpus holds commands and meetings but almost no small
talk with the robot, so every question put to it scores 0.000: a second-person question
("你用的系统是什么"), a reply-shaped sentence inside the follow-up window the bridge opens
after the robot spoke ("是不是真的", "我想选香蕉"), and the trailing question of an
utterance the ASR glued onto chatter ("……跟他说的话你还会做什么动作"). Each of them is
graded ``conversation``: the bridge marks the request, and the Agent never turns such a
request into motion, so the worst a mistaken admission costs is one short spoken answer.

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
#: ``feature.vector_norm``: ``none`` scores raw ``1 + log(count)`` values (``addressee_zh_v1``
#: and every artefact without the key); ``l2`` divides them by their Euclidean norm first, so
#: a sentence's score no longer grows with its length. Grouped CV on the field corpus: the
#: same model drops 0.948 of the chatter with ``l2`` against 0.904 without, at equal recall.
_VECTOR_NORMS = ("none", "l2")
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
#: Where a skill word may start to read as an instruction addressed to the robot: "做个鼓掌"
#: yes, "这首歌举手的动作" (an ASR-mangled "周哥举手的动作") no. The field session of
#: 2026-11-07 shows why the bare word is not enough: the sentence is *about* the action.
#: A question put to the robot starts with "you" and asks something; used to let one carry a
#: skill word ("你怎么知道我举手了") past the remark veto without opening the veto to remarks.
#: No single-character "哪", "几" or "呢": "你看那个橙子呢" and "你看他在挥手呢" are remarks
#: that end like a question, and "哪"/"几" sit inside ordinary words ("哪个", "几个").
_QUESTION_STARTS = ("你", "您")
_QUESTION_MARKERS = ("什么", "怎么", "为什么", "谁", "多少", "吗", "是不是", "能不能")
_LEXICON_LEAD_CHARS = 3
#: Verbs and request words that make a sentence imperative even when the skill word sits
#: later: "我想看挥手" is a command, "这首歌举手的动作" is a remark about one.
_COMMAND_MARKERS = (
    "帮我",
    "给我",
    "帮忙",
    "麻烦",
    "请",
    "做个",
    "做一个",
    "做一次",
    "来一个",
    "来个",
    "一下",
    "能不能",
    "会不会",
    "要不要",
    "你会",
    "可以",
    "想",
    "吗",
    "吧",
)
#: Score a sentence with no skill word and no wake prefix must reach on its own. The model
#: is a character bag: "清代你会过" shares "你会" with "你会做什么" and scored 1.000, while
#: genuine requests without catalog vocabulary sit far lower. 0.90 is where the corpus'
#: bimodal scores actually separate the two.
_MODEL_ONLY_THRESHOLD = 0.90
#: Discourse fillers the ASR keeps in front of a sentence ("嗯你用的是什么芯片呢", "但是你看起来
#: ……", "那是怎么实现的呢"). They carry no addressee and only push the subject out of the
#: positions the rules look at. "那"/"就" only count before a word a sentence starts with, so
#: "那边的橘子" keeps its demonstrative.
_LEADING_FILLER = re.compile(
    r"^(?:嗯|啊|哦|噢|呃|哎|诶|唉|欸|额|那个|然后|但是|可是|所以|就是|那(?=[你您是我就给再])|就(?=[你您是我给都]))+"
)
#: Question words of a second-person question or a reply. Wider than ``_QUESTION_MARKERS``
#: because these never let a skill word past the remark veto on their own: they only admit a
#: ``conversation``-graded request, which the Agent answers and never executes.
_ASK_MARKERS = (
    "什么",
    "啥",
    "怎么",
    "为什么",
    "谁",
    "多少",
    "哪个",
    "哪一个",
    "哪些",
    "哪里",
    "哪儿",
    "在哪",
    "吗",
    "是不是",
    "能不能",
    "会不会",
    "有没有",
    "可不可以",
    "要不要",
)
#: Third-person pronouns: a sentence about "him"/"it" is said *about* the robot or a person,
#: to somebody else ("他实际的表现是什", "你看见它手指刚才抖了吗").
_THIRD_PERSON = ("他", "她", "它")
#: Answers to the robot's own question ("有什么水果" -> "我想选香蕉"): a choice verb and a
#: short object, nothing else. Admitted only inside the follow-up window.
_CHOICE = re.compile(r"^(?:我想选|我想要|我要|我选|给我个|给我一个|给我来个|来个|来一个|要个|要一个)\S{1,4}$")
#: Longest sentence the follow-up window admits on shape alone; longer ones are discussions.
_FOLLOWUP_MAX_CHARS = 20
#: Longest second-person question admitted on its shape: the field questions run to 16
#: characters ("你跟openeuler是什么关系"), while "你搞得再说上官网是个干嘛哪个你……" is a
#: colleague's rant that merely opens with "you".
_QUESTION_MAX_CHARS = 18
#: "你这不是跳甄姬吗" is a rhetorical remark, not a question; "是不是" is excluded from the match.
_RHETORICAL = re.compile(r"(?<!是)不是.*吗")
#: Bounds of the trailing question rescued from an utterance the ASR glued onto chatter. Six
#: characters keep "你还会干什么" and drop the tag questions people end their own stories with.
_TAIL_MIN_CHARS = 6
_TAIL_MAX_CHARS = 14
_TAIL_MIN_PREFIX_CHARS = 4
#: Tags that end a story told to somebody else ("……你知道吗", "……你说是吧").
_TAG_QUESTIONS = ("你知道吗", "你说是吧", "你说呢", "你说对吧", "你懂吗", "你信吗")
#: Grades of an admitted utterance. ``command`` is every rule this gate had before conversation
#: admission; ``conversation`` may only be answered, never executed (the Agent enforces it).
GRADE_COMMAND = "command"
GRADE_CONVERSATION = "conversation"


def strip_fillers(text: str) -> str:
    """Drop leading discourse fillers; a sentence that is nothing but fillers is kept as it is."""
    stripped = _LEADING_FILLER.sub("", text)
    return stripped or text


def collapse_repeats(text: str) -> str:
    """Fold runs of one character ("故故事", "有有"), for phrase matching only.

    ASR stutter doubles characters, and so does Chinese itself ("挥挥手"), so this is never
    applied to the text a request carries -- both sides of a comparison are folded instead.
    """
    return re.sub(r"(.)\1+", r"\1", text)


#: Placeholders that contain a question word without asking anything ("我是开了那个什么").
_PLACEHOLDERS = re.compile(r"那个什么|什么的|干什么的")


def _asks(body: str) -> bool:
    return any(marker in _PLACEHOLDERS.sub("", body) for marker in _ASK_MARKERS)


def _second_person_question(body: str) -> bool:
    """A short question put to "you" that is not about a third person and not rhetorical."""
    return (
        body.startswith(_QUESTION_STARTS)
        and len(body) <= _QUESTION_MAX_CHARS
        and _asks(body)
        and not any(p in body for p in _THIRD_PERSON)
        and not _RHETORICAL.search(body)
    )


def _reply_shaped(body: str) -> bool:
    """A sentence that reads as an answer or a reaction to what the robot just said."""
    if len(body) > _FOLLOWUP_MAX_CHARS or any(p in body for p in _THIRD_PERSON):
        return False
    return _asks(body) or body.endswith(("吗", "呢")) or body.startswith(_QUESTION_STARTS) or bool(_CHOICE.match(body))


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
                text = text[len(wake) :]
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
    #: ``command`` or ``conversation``; meaningful only when accepted.
    grade: str = GRADE_COMMAND
    #: The part of ``text`` that was admitted when it is not the whole of it (a rescued
    #: trailing question); empty when the whole utterance is the request.
    admitted_text: str = ""

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
        model_only_threshold: float = _MODEL_ONLY_THRESHOLD,
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
        self._vector_norm = str(feature.get("vector_norm", "none"))
        if self._vector_norm not in _VECTOR_NORMS:
            raise ValueError(f"unsupported semantic model vector_norm {self._vector_norm!r}")
        self._wake_words = tuple(str(item) for item in payload.get("wake_words", ()))
        self.threshold = float(threshold)
        self.model_only_threshold = float(model_only_threshold)
        self.lexicon = tuple(item for item in (normalize_text(str(entry)) for entry in lexicon) if item)
        self.dialogue_requests = tuple(
            item for item in (normalize_text(str(entry)) for entry in dialogue_requests) if item
        )
        self.wake_prefixes = tuple(item for item in (normalize_text(str(entry)) for entry in wake_prefixes) if item)
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
                # The bridge's own wake phrases are wake prefixes too: a deployment that renames
                # its wake word ("欧拉欧拉") must not need a second, drifting copy for this gate.
                wake_prefixes=tuple(config.semantic_gate_wake_prefixes) + tuple(getattr(config, "wake_phrases", ())),
                model_only_threshold=float(config.semantic_gate_model_only_threshold),
            )
        except (OSError, ValueError) as error:
            if not config.semantic_gate_fail_open:
                raise
            return None, f"semantic gate {mode} disabled: {error}"
        return gate, (
            f"semantic gate {mode}: {path} (threshold={gate.threshold:.2f}, "
            f"model-only={gate.model_only_threshold:.2f}, lexicon={len(gate.lexicon)}, "
            f"trained={gate.created_at or 'unknown'})"
        )

    def _body(self, text: str) -> str:
        return strip_wake_words(normalize_text(text), self._wake_words)

    def features(self, text: str) -> Counter[int]:
        padded = f"{self._pad}{self._body(text)}{self._pad}"
        counts: Counter[int] = Counter()
        for size in self._ngrams:
            for start in range(len(padded) - size + 1):
                counts[_fnv1a32(padded[start : start + size]) % self._dim] += 1
        return counts

    def vector(self, text: str) -> dict[int, float]:
        """The feature values ``score`` weighs; ``tools/addressee`` trains on exactly these."""
        values = {key: 1.0 + math.log(count) for key, count in self.features(text).items()}
        if self._vector_norm == "l2":
            norm = math.sqrt(sum(value * value for value in values.values()))
            if norm > 0.0:
                values = {key: value / norm for key, value in values.items()}
        return values

    def score(self, text: str) -> float:
        """Probability-like score in (0, 1) that ``text`` is addressed to the robot."""
        logit = self._bias
        for key, value in self.vector(text).items():
            weight = self._weights.get(key)
            if weight:
                logit += weight * value
        logit = max(-_LOGIT_LIMIT, min(_LOGIT_LIMIT, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def _leading_skill_word(self, body: str) -> bool:
        """Whether a catalog word opens the sentence, or a request word makes it imperative."""
        positions = [body.find(word) for word in self.lexicon if word in body]
        if not positions:
            return False
        if min(positions) < _LEXICON_LEAD_CHARS:
            return True
        return any(marker in body for marker in _COMMAND_MARKERS)

    def _asks_about_itself(self, body: str) -> bool:
        """Whether the sentence is a question put to the robot ("你怎么知道我举手了")."""
        if not body.startswith(_QUESTION_STARTS):
            return False
        return any(marker in body for marker in _QUESTION_MARKERS)

    def _phrase_hit(self, phrases: tuple[str, ...], *texts: str) -> str:
        """The first phrase found in any of ``texts``, comparing with character runs folded."""
        folded = [collapse_repeats(text) for text in texts]
        return next(
            (phrase for phrase in phrases if any(collapse_repeats(phrase) in text for text in folded)),
            "",
        )

    def judge(
        self,
        text: str,
        *,
        session_open: bool = False,
        followup_open: bool = False,
        conversation: bool = False,
    ) -> SemanticDecision:
        """Decide whether ``text`` is addressed to the robot; stop/confirm never come here.

        A skill word is evidence only in a command-sized sentence or when the model does
        not flatly disagree, a sentence naming someone else in the room is not addressed
        to the robot, and a bare fragment needs a session the addressee gate opened --
        the three failure classes the first field session produced on 2026-11-07.

        ``conversation`` turns on conversation admission (see the module docstring):
        ``session_open`` is required for the second-person question and the rescued tail,
        ``followup_open`` (the bridge's window after the robot spoke) for a reply.
        """
        body = strip_fillers(self._body(text))
        raw_normalized = normalize_text(text)
        score = self.score(text)
        lexicon_hit = bool(self._phrase_hit(self.lexicon, raw_normalized, body))
        wake_prefix = any(raw_normalized.startswith(prefix) for prefix in self.wake_prefixes)
        address_term = next((term for term in _ADDRESS_TERMS if term in raw_normalized), "")

        def decide(rejected: bool, reason: str, grade: str = GRADE_COMMAND, admitted: str = "") -> SemanticDecision:
            return SemanticDecision(
                text, score, rejected, lexicon_hit, wake_prefix, reason, address_term, grade, admitted
            )

        if not body:
            return decide(False, "empty after wake-word stripping")
        if address_term and not wake_prefix:
            return decide(True, f"names someone else ({address_term})")
        # A request to talk is addressed to whatever can talk back, and it can only ever
        # start a conversation, so it outranks the model instead of arguing with it.
        if self._phrase_hit(self.dialogue_requests, raw_normalized, body):
            return decide(False, "dialogue request")
        if (
            lexicon_hit
            and self._leading_skill_word(body)
            and (len(body) <= _LEXICON_MAX_BODY_CHARS or score >= self.threshold)
        ):
            return decide(False, "skill word")
        if wake_prefix:
            return decide(False, "wake prefix")
        if conversation and session_open and _second_person_question(body):
            # Inside a session the wake word and the voiceprint already opened, a question put
            # to "you" is the robot's to answer, skill word or not: "你是怎么抓水果的" and "你用的
            # 系统是什么" both scored 0.000 on a corpus without small talk. It is only ever
            # answered (conversation grade), which is what lets it skip the remark veto below.
            return decide(False, "question to the robot", GRADE_CONVERSATION)
        if lexicon_hit and self._asks_about_itself(body) and score >= self.threshold:
            # "你怎么知道我举手了" carries a skill word and a question. The veto below exists
            # for remarks about the action ("这首歌举手的动作"), and a question put to the robot
            # is the opposite: it can only be answered by it. The shape is only a hint, so the
            # model must not flatly disagree: unlike the skill-word branch, a short sentence
            # gets no pass on length ("你刚才拿的什么水果" is 9 characters and scores 0.000).
            return decide(False, "question about the robot")
        if lexicon_hit:
            # The word is there but the sentence is about the action, not a request for it,
            # and a character-bag model cannot tell the two apart ("这首歌举手的动作" 1.000).
            decision = decide(True, f"mentions a skill word without asking for it ({len(body)} chars)")
        elif score >= self.model_only_threshold:
            if len(body) < _FRAGMENT_BODY_CHARS and not session_open:
                decision = decide(True, f"fragment of {len(body)} chars outside a session")
            else:
                return decide(False, "model")
        elif score >= self.threshold:
            # High enough for the old rule, but the vocabulary overlap that produced it is
            # not evidence: without a skill word, only a clear score may open the pipeline.
            decision = decide(True, f"score {score:.3f} below {self.model_only_threshold:.2f} without a skill word")
        else:
            decision = decide(True, f"score {score:.3f} below {self.threshold:.2f}")
        if conversation and followup_open and _reply_shaped(body):
            # The robot has just spoken and its listener answers: "是不是真的", "系统是什么",
            # "我想选香蕉" have no subject and no catalog word, and only the turn says who they
            # are for. A statement ("其实这个功能可以再优化") is not reply-shaped and stays out.
            return decide(False, "follow-up to the robot", GRADE_CONVERSATION)
        if conversation and session_open:
            tail = self._trailing_question(body)
            if tail:
                # One VAD segment can hold two speakers or two listeners: the ASR glued "这都不是
                # 跟他说的话" to "你还会做什么动作". Only the trailing question is admitted, and the
                # request carries it alone; the bridge checks the voice of the segment's tail.
                return decide(False, f"trailing question {tail!r}", GRADE_CONVERSATION, tail)
        return decision

    def _trailing_question(self, body: str) -> str:
        """The longest second-person question (or talk request) that ends ``body`` after a prefix."""
        for start in range(_TAIL_MIN_PREFIX_CHARS, len(body) - _TAIL_MIN_CHARS + 1):
            tail = body[start:]
            if len(tail) > _TAIL_MAX_CHARS or not tail.startswith(_QUESTION_STARTS):
                continue
            if any(tail.endswith(tag) for tag in _TAG_QUESTIONS):
                continue
            if _second_person_question(tail) or (
                self._phrase_hit(self.dialogue_requests, tail) and not any(p in tail for p in _THIRD_PERSON)
            ):
                return tail
        return ""

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
        if not decision.rejected and decision.grade != GRADE_COMMAND:
            entry["grade"] = decision.grade
        if decision.admitted_text:
            entry["admitted_text"] = decision.admitted_text
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # A full or read-only disk must not take the voice channel down with it.
            return
