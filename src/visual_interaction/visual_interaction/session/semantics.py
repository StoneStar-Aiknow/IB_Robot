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

# Values that name no subject, so finding one in a subject field means the model
# repeated a kind word rather than reporting something. Tolerated on a quiet
# answer (see parse_observation) and never a valid category in themselves —
# CATEGORIES excludes all of them.
_NON_SUBJECT_VALUES = frozenset({"none", "uncertain", "unknown", "null", "n/a", "无", "没有"})

# Specific risks a reply may not express. One family per risk somebody named,
# not a guess at vocabulary.
#
# This replaced a character whitelist, which constrained characters while the
# risk lives in meaning — a level mismatch that failed in both directions.
# Measured offline (visual_interaction.phase0.boundary_ablation) over fourteen
# sentences this feature exists to say and twenty-one that must stay refused:
#
#   character whitelist   11/14 said   15/21 refused
#   these filters         14/14 said   21/21 refused
#   both together         11/14 said   21/21 refused
#
# The whitelist refused "她正在展示一个和平手势" — an accurate description —
# and passed "你那个玩偶是我的。", built entirely from allowed characters.
# Keeping it alongside these filters bought nothing and cost three refusals, so
# expression is open and each risk is named instead.
#
# These are a net, not a proof: a risk nobody listed is not refused here. The
# closed fields still decide what may be discussed, and _CATEGORY_WORDS below
# refuses a reply that stopped describing the confirmed subject at all.
_OUT_OF_RANGE_PATTERNS: tuple[tuple[str, str], ...] = (
    # The visitor talked about rather than to. The board model did this on
    # every round ("她正在展示一个和平手势"), which reads as surveillance
    # commentary: the robot is speaking face to face. A second person is NOT
    # required — "戴着这顶帽子很显眼呀" addresses them by context, and six of
    # the fourteen desirable sentences carry no 你 at all.
    (r"她|他(?!们好)|这个人|那个人|那位|这位|此人|女士|先生|对方", "third_person"),
    # The room rather than the person in it. The closed fields cannot catch
    # this: they say what was confirmed, not what the sentence is about.
    (r"房间|地上|墙上|桌上|屋里|室内|背景|旁边有|摆着|放着", "scene_not_visitor"),
    # Material and make.
    (r"木|金属|塑料|棉|丝|皮革|陶|玻璃|毛绒|手工|材质|做工", "material"),
    # Brand, price and provenance.
    (r"牌子是|品牌|名牌|多少钱|价格|贵|便宜|正品|限量|买的|产地|进口", "commerce"),
    # Reading any text in frame out loud, however it is introduced.
    (r"写着|写的是|上面说|标语|字是|念出|读出|内容是|上面的字|电话|号码", "sign_text"),
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
        r"[^。！？，、]*?(?:漂亮|好看|帅|可爱|身材|年纪|岁)"
        r"|你也(?:很|真|挺)?(?:漂亮|好看|帅|可爱)"
        r"|的你(?:真|很|好|挺)?(?:漂亮|好看|帅|可爱)"
        # A simile routes praise back to the person around the determiner:
        # "你举起的玩偶真像你。" passed every check the whitelist build had.
        r"|像你",
        "person",
    ),
    # The robot inventing a history with the visitor, or a claim on what they
    # are holding. Found by composing whitelisted characters, so it is exactly
    # the class a vocabulary gate could not see.
    (r"是我的|我也有|我的东西|归我|送我|给我吧|我见过你|上次", "false_claim"),
    # Medical devices and religious or political symbols: named in review as
    # never-say, and the review bypass "你戴的徽章上有十字架。" lives here.
    (r"助听|轮椅|拐杖|药|病|残|十字架|佛|经文|党徽|国旗", "sensitive_attribute"),
    # Anyone other than the visitor brought into the remark.
    (r"和那个|和这个人|旁边的人|其他人|大家|他们", "third_party"),
)

_OUT_OF_RANGE = tuple((re.compile(pattern), label) for pattern, label in _OUT_OF_RANGE_PATTERNS)

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
#
# Each list has to carry the synonyms a model actually reaches for, or the
# check refuses accurate descriptions. The board run showed this: fourteen of
# twenty replies said 标志 for a sign, which was absent here, so every one was
# refused for "not describing the confirmed sign" while in fact describing it.
_CATEGORY_WORDS: Mapping[str, tuple[str, ...]] = {
    "toy": ("玩偶", "玩具", "娃娃", "公仔", "毛绒"),
    "sign": ("牌子", "标牌", "纸板", "举牌", "标志", "牌"),
    "hat": ("帽子", "帽"),
    "badge": ("徽章", "胸针", "别针", "胸章", "胸牌"),
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

    The boundary has three levels, each holding what it can actually hold:

    * **closed semantic range** — ``kind``, ``category`` and ``region`` decide
      what may be discussed at all. Done by :func:`parse_observation`.
    * **open expression** — any wording is allowed. There is no vocabulary
      gate: constraining characters while the risk lives in meaning refused
      accurate descriptions and passed inaccurate ones (see
      :data:`_OUT_OF_RANGE_PATTERNS` for the measurement).
    * **specific risk filters plus grounding** — this function. Each named
      risk is refused, and the reply must still describe the confirmed
      subject.

    Checks run cheapest first: the length band (a one-word answer carries no
    observation, a long one stops being a short remark), the instruction shape
    (a sign in frame may read like a command; a reply is said to a visitor,
    never to the system), no question (nothing here can hear an answer),
    digits and Latin letters, then each risk family, then grounding.

    These filters are a net, not a proof: a risk nobody listed is not refused.
    What bounds the damage is the level above — the closed fields — and the
    grounding check below, which refuses a reply that wandered off the subject
    those fields settled.
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
        # A quiet answer must stay quiet. What makes it quiet is that nothing
        # speakable survives, so the subject fields are dropped rather than
        # inspected — except for a real category, which is the one thing that
        # would mean the model contradicted itself.
        #
        # Measured on the board, thirteen of twenty quiet answers arrived as
        # {"kind": "none", "category": "uncertain", "region": "chest", ...},
        # some with none and uncertain swapped. The model meant "nothing to
        # report" and filled the fields with the words listed beside them in
        # the schema. Refusing those reported SCHEMA_INVALID — which reads as a
        # broken model — for an answer whose meaning was right.
        #
        # A named category still fails: "nothing to report, it is a hat" is a
        # contradiction, and honouring either half would be guessing which.
        category = normalise_text(_as_string(data.get("category", ""), "category"))
        if category and category not in _NON_SUBJECT_VALUES:
            raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"kind {kind!r} must not carry category")
        # A region alone names no subject — without a category nothing can be
        # said about it — and a reply here is a report to the host rather than
        # a remark to a visitor ("没有看到明显的物品。"). Both are discarded.
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


def _strip_code_fence(text: str) -> str:
    """Remove a markdown code fence around a JSON object, if there is one.

    Measured on the board: the model wraps its answer in ```json … ``` on
    roughly half the rounds, with the same object inside either way. Refusing
    those would throw away a well-formed answer over presentation, and no
    prompt wording reliably stops a chat-tuned model from doing it. The fence
    is formatting, so the host removes it rather than asking the model to be
    perfect.

    Only an outer fence is stripped. Anything else about the payload is still
    the schema check's business.
    """
    if not text.startswith("```"):
        return text
    body = text[3:]
    # An opening fence may name a language ("```json"); drop that first line.
    newline = body.find("\n")
    if newline != -1 and "{" not in body[:newline]:
        body = body[newline + 1 :]
    closing = body.rfind("```")
    return (body[:closing] if closing != -1 else body).strip()


def _as_mapping(payload: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    if not isinstance(payload, str):
        raise SemanticsError(SUPPRESS_SCHEMA_INVALID, f"answer is {type(payload).__name__}, not JSON")
    text = _strip_code_fence(payload.strip())
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
