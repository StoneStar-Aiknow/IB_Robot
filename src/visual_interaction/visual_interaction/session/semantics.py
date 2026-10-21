"""Parse one model answer into a closed-set observation, or reject it.

The model writes JSON with four fields. Three are closed sets the host checks
against :mod:`embodied_common.visual_interaction_contract`; only ``reply`` is
free text, and it goes through :func:`check_reply` before anything may say it.

Two rules shape every function here:

* **Rejecting is cheap, guessing is not.** A malformed answer, an unknown
  category or a reply that reaches past what the closed fields support becomes
  silence with a recorded reason. Nothing is repaired into a different claim,
  and no field is filled in on the model's behalf.
* **Text in the frame is data.** A sign the visitor holds up may read like an
  instruction; it never becomes one. This module only ever treats model output
  as a value to validate.

Nothing here imports ROS: the whole module runs under plain pytest.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from embodied_common.visual_interaction_contract import (
    CATEGORIES,
    DEFAULT_MAX_REPLY_CHARS,
    DEFAULT_MIN_REPLY_CHARS,
    KIND_DISPLAY,
    KIND_NONE,
    KIND_OBJECT,
    KIND_UNCERTAIN,
    KINDS,
    REGIONS,
    SPEAKABLE_KINDS,
    SUPPRESS_NO_NEW_TOPIC,
    SUPPRESS_PRIVACY,
    SUPPRESS_SCHEMA_INVALID,
    SUPPRESS_UNCERTAIN,
)

# The fields the model may send. An extra key is a schema violation rather than
# something to ignore: it usually means the prompt and the host disagree, and
# silently dropping it would hide that.
_FIELDS = frozenset({"kind", "category", "region", "reply"})

# Fields that must be absent or empty when the kind carries no subject.
_SUBJECT_FIELDS = ("category", "region")

# Attributes outside the first version's verifiable range. A reply only ever
# expresses the category and the display relation, so wording that reaches for
# one of these is rejected whole: the claim is what is unverifiable, and
# deleting the word would leave the same sentence asserting it.
#
# This list is a net, not a proof. Keyword matching cannot enumerate every way
# to assert a colour or a mood, which is why it is only the second of three
# checks: the closed fields above decide the subject, this removes the common
# over-reach, and _REQUIRED_GROUNDING below refuses anything that stopped
# describing the confirmed category at all.
_OUT_OF_RANGE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Material and make.
    (r"木|金属|塑料|棉|丝|皮革|陶|玻璃|毛绒|手工|材质|做工", "material"),
    # Brand, price and provenance.
    (r"牌子是|品牌|名牌|多少钱|价格|贵|便宜|正品|限量|买的|产地", "commerce"),
    # Reading any text in frame out loud, however it is introduced.
    (r"写着|写的是|上面说|标语|字是|念出|读出|内容是|上面的字", "sign_text"),
    # Inferred inner state. The robot sees a picture, not a mood. "你" and the
    # state word are allowed to be far apart: "你看起来很开心" puts three
    # characters between them.
    (r"开心|高兴|难过|生气|紧张|害羞|心情|情绪|性格|喜欢|讨厌|想要|觉得|累了", "inner_state"),
    # Praise aimed at the person rather than at the thing.
    #
    # Praising the object is the point ("你的帽子很好看"), so this cannot refuse
    # the praise words themselves. It refuses them when the subject is the
    # visitor, in the shapes that carry that meaning:
    #
    #   你 … 好看          second person praised directly, at any distance
    #   你也好看            "and you too", after a fine object remark
    #   举着玩偶的你真可爱    a relative clause ending on 你
    #   你戴着帽子真好看      an adverbial clause: the hat is how, not what
    #
    # The first alternative is load-bearing: an adjacency-based pattern missed
    # "你戴的帽子好看，你也好看。", exactly what the default-off appearance
    # rule exists to stop. What makes the praise land on the object is a
    # determiner in front of the subject noun — either possessive 的
    # ("你戴的帽子") or a demonstrative with a measure word ("你戴这顶帽子").
    # The lookahead steps over a noun reached either way. With a bare
    # adverbial noun and no determiner ("你戴着帽子真好看") the praise is about
    # the visitor, so it stays refused.
    (
        r"你(?![^。！？，、]*?(?:的|这|那)[^。！？，、]*?"
        r"(?:帽子|玩偶|玩具|娃娃|公仔|牌子|标牌|纸板|徽章|胸针|别针|胸章))"
        r"[^。！？，、]*?(?:漂亮|好看|帅|可爱)"
        r"|你也(?:很|真|挺)?(?:漂亮|好看|帅|可爱)"
        r"|的你(?:真|很|好|挺)?(?:漂亮|好看|帅|可爱)",
        "person",
    ),
)

_OUT_OF_RANGE = tuple((re.compile(pattern), label) for pattern, label in _OUT_OF_RANGE_PATTERNS)

# The vocabulary a reply may be built from. This is the load-bearing check, and
# it is a whitelist on purpose.
#
# A blacklist cannot hold this boundary. "你举着的牌子写了电话号码13800138000。",
# "你举起的玩偶旁边有助听器。" and "你戴的徽章上有十字架。" each name the
# confirmed category, miss every out-of-range pattern, and each publishes
# something the first version promised never to say — personal data, a medical
# device, a religious symbol. Enumerating those three fixes those three.
#
# So the rule is inverted: a reply may only use characters from this set, and
# anything else — an unlisted word, a digit, a Latin letter — refuses it. The
# cost is that a perfectly good sentence can be refused for using a word nobody
# listed, which is the right direction to fail in. Widening the set is a review
# decision with the whole boundary in view, not an edit to a pattern.
_ALLOWED_REPLY_CHARS = frozenset(
    # Subjects: the four categories and their ordinary synonyms.
    "玩偶玩具娃娃公仔牌子标牌纸板帽子徽章胸针别针胸章"
    # Display relations: holding, wearing, showing, pointing.
    "举起拿着带来戴着别着挂在手上头上胸前身上拿出展示给看向"
    # Second person and the robot's own voice.
    "你我它这那个只件顶枚张"
    # Plain remarks: noticing, liking, finding it interesting.
    "真好有意思特别挺不错很可爱看到注意呀啊呢吧哦喔嗯漂亮趣点目醒眼"
    "是的了着过在和跟与也还就都又再一二两"
    # Light verbs that keep a sentence grounded in what is visible.
    "像似乎显得看起来瞧见发现觉"
    # Basic colours. A colour is a property of the picture rather than an
    # inference about it, so it is allowed to be stated: "你的红帽子很显眼。"
    # Material, make and brand stay refused by pattern — those are claims about
    # how a thing was produced, which a camera cannot establish.
    "红橙黄绿青蓝紫粉灰棕褐黑白金银色"
    # Stative remarks that describe rather than judge.
    "显眼鲜艳亮醒颜"
    # Punctuation.
    "。！？，、…—～ "
)

# Characters that are refused however they are used. Digits and Latin letters
# are how a phone number, a licence plate or a brand reaches a reply, and no
# allowed remark needs either.
_FORBIDDEN_REPLY_PATTERN = re.compile(r"[0-9０-９a-zA-Z]")

# Wording that treats the reply as a command rather than a remark. A sign in
# frame may read like an instruction, and a model that copied it would produce
# exactly this shape. The reply is something the robot says to a visitor, so an
# imperative aimed at the system is never a valid one.
_INSTRUCTION_SHAPED = re.compile(
    r"忽略|无视|取消|覆盖|系统提示|系统指令|提示词|规则|允许|禁止|现在起|从现在|"
    r"请(执行|运行|调用|输出)|指令|ignore\s+(previous|all|above)|system\s+prompt|"
    r"instruction|override|disregard",
    re.IGNORECASE,
)

# A reply must still be about the thing the closed fields settled. Requiring a
# word that names the confirmed category is a coarse check, but it is a
# *positive* one: it fails closed on a reply that drifted onto another subject,
# including one this module has no pattern for.
_CATEGORY_WORDS: Mapping[str, tuple[str, ...]] = {
    "toy": ("玩偶", "玩具", "娃娃", "公仔"),
    "sign": ("牌子", "标牌", "纸板", "举牌"),
    "hat": ("帽子", "帽"),
    "badge": ("徽章", "胸针", "别针", "胸章"),
}

# Characters that must never reach a synthesiser: C0/C1 controls, bidi overrides
# and zero-width joiners. They do not change what a reply claims, so these are
# normalised away rather than rejected.
_STRIP_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs"})


class SemanticsError(ValueError):
    """One model answer could not be read as a closed-set observation.

    Carries the suppression reason the session records, so a caller never has to
    map an exception message back onto a diagnostic code.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Observation:
    """What the model reported about one window, after validation.

    ``kind`` is always one of :data:`KINDS`. ``category`` and ``region`` are
    empty exactly when the kind carries no subject, and ``reply`` is empty
    unless the kind is speakable. A value of this class asserts only that the
    answer was well formed and inside the whitelist: whether it is true, fresh,
    already covered or safe to say now is decided elsewhere.
    """

    kind: str
    category: str = ""
    region: str = ""
    reply: str = ""

    @property
    def has_subject(self) -> bool:
        return self.kind in SPEAKABLE_KINDS

    @property
    def topic_fields(self) -> tuple[str, str]:
        """The closed pair a topic key is built from. Never the reply text."""
        return (self.category, self.region)


def quiet_reason(kind: str) -> str:
    """Map a non-speakable kind onto the reason the session reports.

    ``none`` and ``uncertain`` are both silence, and they are counted apart: one
    says the feature is working, the other says it cannot see well enough.
    """
    if kind == KIND_NONE:
        return SUPPRESS_NO_NEW_TOPIC
    if kind == KIND_UNCERTAIN:
        return SUPPRESS_UNCERTAIN
    raise ValueError(f"kind {kind!r} is not a quiet kind")


def normalise_text(value: str) -> str:
    """Strip invisible characters and collapse whitespace, changing no claim.

    Applied before length checks so a reply cannot pad itself past the band
    with zero-width characters, and before synthesis so control characters
    never reach the service.
    """
    without_controls = "".join(char for char in value if unicodedata.category(char) not in _STRIP_CATEGORIES)
    return re.sub(r"\s+", " ", without_controls).strip()


def check_reply(
    reply: str,
    *,
    category: str = "",
    min_chars: int = DEFAULT_MIN_REPLY_CHARS,
    max_chars: int = DEFAULT_MAX_REPLY_CHARS,
) -> str:
    """Return the reply to speak, or raise :class:`SemanticsError`.

    Five checks, all fail-closed, cheapest first:

    1. the length band, because a one-word answer carries no observation and a
       long one stops being a short remark;
    2. the instruction shape, because a sign in frame may read like a command
       and a reply is something said to a visitor, never to the system;
    3. no question, because the first version has no way to hear an answer;
    4. the vocabulary whitelist — the load-bearing one. A reply may only use
       characters from :data:`_ALLOWED_REPLY_CHARS`, so a phone number, a
       medical device or a religious symbol is refused for not being in the
       vocabulary rather than for matching a pattern somebody thought of;
    5. the out-of-range patterns and the category grounding, which catch a
       sentence built entirely from allowed characters that still over-reaches.

    Check 4 is why 1–3 and 5 are a net rather than the boundary: a blacklist
    cannot enumerate every way to say something out of range, and each of the
    three real bypasses found in review named its category, missed every
    pattern, and published something the first version promised never to say.
    """
    text = normalise_text(reply)
    if not text:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, "reply is empty")
    if len(text) < min_chars:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"reply is shorter than {min_chars} characters")
    if len(text) > max_chars:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"reply is longer than {max_chars} characters")
    if _INSTRUCTION_SHAPED.search(text):
        raise SemanticsError(SUPPRESS_PRIVACY, "reply is shaped like an instruction, not a remark")
    if "?" in text or "？" in text:
        # Asking invites an answer nothing here can hear, so the first version
        # states rather than asks.
        raise SemanticsError(SUPPRESS_PRIVACY, "reply asks a question; the first version only states")
    if _FORBIDDEN_REPLY_PATTERN.search(text):
        raise SemanticsError(SUPPRESS_PRIVACY, "reply contains digits or Latin letters, which no allowed remark needs")
    outside = sorted(set(text) - _ALLOWED_REPLY_CHARS)
    if outside:
        raise SemanticsError(SUPPRESS_PRIVACY, f"reply uses words outside the allowed vocabulary: {''.join(outside)}")
    for pattern, label in _OUT_OF_RANGE:
        if pattern.search(text):
            raise SemanticsError(SUPPRESS_PRIVACY, f"reply asserts {label}, which is out of range")
    if category:
        words = _CATEGORY_WORDS.get(category, ())
        if words and not any(word in text for word in words):
            raise SemanticsError(SUPPRESS_PRIVACY, f"reply does not describe the confirmed {category}")
    return text


def parse_observation(
    payload: str | Mapping[str, object],
    *,
    categories: tuple[str, ...] = CATEGORIES,
    min_chars: int = DEFAULT_MIN_REPLY_CHARS,
    max_chars: int = DEFAULT_MAX_REPLY_CHARS,
) -> Observation:
    """Validate one model answer. Raises :class:`SemanticsError` on anything odd.

    ``categories`` is the deployment's whitelist, which may be narrower than
    :data:`CATEGORIES` but never wider: a value outside the contract's own set
    is a configuration error, raised as such rather than honoured.
    """
    unknown_allowed = set(categories) - set(CATEGORIES)
    if unknown_allowed:
        raise ValueError(f"categories outside the contract: {sorted(unknown_allowed)}")

    data = _as_mapping(payload)

    extra = set(data) - _FIELDS
    if extra:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"unknown fields: {sorted(extra)}")

    kind = _as_string(data.get("kind"), "kind")
    if kind not in KINDS:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"kind {kind!r} is not one of {list(KINDS)}")

    if kind not in SPEAKABLE_KINDS:
        # A quiet answer must stay quiet: carrying a subject or a reply here
        # would let a "cannot see it" round speak anyway.
        for field in (*_SUBJECT_FIELDS, "reply"):
            if normalise_text(_as_string(data.get(field, ""), field)):
                raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"kind {kind!r} must not carry {field}")
        return Observation(kind=kind)

    category = _as_string(data.get("category"), "category")
    if category not in categories:
        # Unknown and not-enabled are the same answer here on purpose: both mean
        # the host has no verified way to talk about it.
        raise SemanticsError(SUPPRESS_PRIVACY, f"category {category!r} is not allowed")

    region = _as_string(data.get("region"), "region")
    if region not in REGIONS:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"region {region!r} is not one of {list(REGIONS)}")

    reply = check_reply(
        _as_string(data.get("reply"), "reply"),
        category=category,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    return Observation(kind=kind, category=category, region=region, reply=reply)


def _as_mapping(payload: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    if not isinstance(payload, str):
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"answer is {type(payload).__name__}, not JSON")
    text = payload.strip()
    if not text:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, "answer is empty")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        # Truncated output is the common case; an early stop on ``kind`` leaves
        # the rest of the object unclosed. It is not a candidate either way.
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"answer is not valid JSON: {error.msg}") from error
    if not isinstance(decoded, Mapping):
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"answer is a JSON {type(decoded).__name__}, not an object")
    return decoded


def _as_string(value: object, field: str) -> str:
    if value is None:
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"{field} is missing")
    if not isinstance(value, str):
        # bool is an int in Python, and both would otherwise stringify into a
        # plausible-looking value.
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"{field} is {type(value).__name__}, not a string")
    return value.strip()


__all__ = [
    "KIND_DISPLAY",
    "KIND_OBJECT",
    "Observation",
    "SemanticsError",
    "check_reply",
    "normalise_text",
    "parse_observation",
    "quiet_reason",
]
