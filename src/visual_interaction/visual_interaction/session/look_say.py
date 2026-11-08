"""Look-say: what InternVL names, matched to a prerecorded line, said on a rhythm.

The X2 mirror's ``internvl_voice`` mode. Each look runs the vision model once
on the newest frame and asks up to three short questions about it:

* **Q1**, every look: chatter's own open question, verified on the board — the
  model names the action freely, in a few characters.
* **Q2**, only when Q1 lands on a vague hand gesture ("做手势", "比划"): which
  gesture the fingers make.
* **Q3**, only right before a line that *names* a gesture (heart, thumbs up,
  victory, wave, open arms): a yes/no check. A "no" does not silence the
  robot; the line comes from the same family's generic pool instead.

The answers are mapped by keyword (:func:`classify`, reusing
``chatter.ACTIONS``) onto the pools of the clip table, and :class:`LookSay`
decides when a line is due and which one. Every line is a recording; the
model's words are never spoken.

Pure Python, no ROS, no model: the node passes the clock (every method takes
``now``), the voice's busy state and the answers in, and an RNG is injected,
so all of it is tested on a workstation.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from visual_interaction.session.chatter import ACTIONS, CHATTER_QUESTION

# ---------------------------------------------------------------------------
# The questions. Text only; the model's chat template and image header are the
# vision port's. Q1 is chatter's question verbatim (its system line,
# chatter.CHATTER_SYSTEM, goes with it), so the board's experience with it
# carries over.

Q1 = "q1"
Q2 = "q2"
Q3_HEART = "q3.heart"
Q3_THUMB = "q3.thumb"
Q3_YEAH = "q3.yeah"
Q3_WAVE = "q3.wave"
Q3_OPEN = "q3.open"


def _yes_no(gesture: str) -> str:
    return f"这个人是在{gesture}吗？只回答是或不是。"


QUESTIONS: dict[str, tuple[str, int]] = {
    Q1: (CHATTER_QUESTION, 16),
    # The model repeats "这个人的手指比的是" first: 6 tokens ended before the name.
    Q2: ("这个人的手指比的是什么手势？用不超过五个字回答。", 14),
    Q3_HEART: (_yes_no("用手比爱心"), 3),
    Q3_THUMB: (_yes_no("竖大拇指"), 3),
    Q3_YEAH: (_yes_no("比剪刀手"), 3),
    Q3_WAVE: (_yes_no("挥手打招呼"), 3),
    Q3_OPEN: (_yes_no("张开双臂"), 3),
}


# ---------------------------------------------------------------------------
# The mapping (plan §4.2).


@dataclass(frozen=True)
class Label:
    """What one answer means for the pools.

    ``family`` is the coarse class the switch rule compares (hand, big, pose,
    dance, other; plus ``down`` and ``none``, which are never candidates).
    ``pool`` is where its line comes from; a label with a ``verify_key``
    names a gesture and needs Q3's "yes" first, and ``coarse_pool`` is where
    it goes on a "no".
    """

    tag: str
    family: str
    pool: str | None = None
    coarse_pool: str | None = None
    verify_key: str | None = None

    @property
    def named(self) -> bool:
        return self.verify_key is not None


HEART = Label("hand/heart", "hand", "hand.heart", "hand.any", Q3_HEART)
THUMB = Label("hand/thumb", "hand", "hand.thumb", "hand.any", Q3_THUMB)
YEAH = Label("hand/yeah", "hand", "hand.yeah", "hand.any", Q3_YEAH)
HAND_ANY = Label("hand/any", "hand", "hand.any")
WAVE = Label("big/wave", "big", "big.wave", "big.any", Q3_WAVE)
OPEN = Label("big/open", "big", "big.open", "big.any", Q3_OPEN)
UP = Label("big/up", "big", "big.any")
POSE = Label("pose", "pose", "pose.any")
DANCE = Label("dance", "dance", "dance.any")
OTHER = Label("other", "other", "other.any")
DOWN = Label("down", "down")
NONE = Label("none", "none")

LABELS: tuple[Label, ...] = (HEART, THUMB, YEAH, HAND_ANY, WAVE, OPEN, UP, POSE, DANCE, OTHER, DOWN, NONE)
# The families a line can be about; none and down are not one.
FAMILIES = ("hand", "big", "pose", "dance", "other")

_CHATTER = {subject.id: subject.keywords for subject in ACTIONS}

# chatter's point keywords that are not about pointing at all: "伸出手指" is
# how the 1B model describes a finger gesture it cannot name, and the bare
# "指" is its riskiest single character. They are tried only after the vague
# gesture words below.
_POINT_LATE = ("伸出手指", "指")
_POINT_EARLY = tuple(word for word in _CHATTER["point"] if word not in _POINT_LATE)

# The words a vague hand gesture comes in, when nothing more specific named
# it; the plan's three plus their spelling variants.
VAGUE_GESTURE_WORDS = ("手势", "比划", "比画", "手指", "伸手指")
# Standing still. Tried only when no action matched at all.
STILL_WORDS = ("站着", "站立", "没有动作", "无动作", "没动", "不动", "静止", "双手下垂", "手臂下垂", "放下", "垂着")
# A few spellings of victory the model uses that chatter's list misses
# ("比出“V”字" is caught once the quotes are stripped).
_VICTORY_EXTRA = ("比V", "个V", "和平手势")
_NOTHING = ("无", "没有人", "没有", "不清楚", "看不清", "没有看到", "无人")

# First match wins, top to bottom: chatter's order for the specific actions,
# then the vague gesture words, the late point words and standing still.
# chatter's "smile" and its catch-all "gesture" (右手, 双手, 抬 ...) are not
# used: a face has no pool, and "双手下垂" must read as still, not as a hand.
RULES: tuple[tuple[Label, tuple[str, ...]], ...] = (
    (HEART, _CHATTER["heart"]),
    (THUMB, _CHATTER["thumbs_up"]),
    (YEAH, _CHATTER["victory"] + _VICTORY_EXTRA),
    (HAND_ANY, _CHATTER["ok"]),
    (OTHER, _CHATTER["clap"]),
    (WAVE, _CHATTER["wave"]),
    (OTHER, _CHATTER["fist_salute"]),
    (OTHER, _CHATTER["pray"]),
    (OTHER, _CHATTER["bow"]),
    (OPEN, _CHATTER["arms_open"]),
    (UP, _CHATTER["stretch"]),
    (POSE, _CHATTER["hands_hip"]),
    (POSE, _CHATTER["arms_crossed"]),
    (OTHER, _POINT_EARLY),
    (DANCE, _CHATTER["dance"]),
    (HAND_ANY, VAGUE_GESTURE_WORDS),
    (OTHER, _POINT_LATE),
    (DOWN, STILL_WORDS),
)

_STRIP = " \t\r\n。.!！?？,，、\"'“”‘’「」『』"
_QUOTES = "\"'“”‘’「」『』"


def _normalize(text: str | None) -> str:
    text = (text or "").strip().strip(_STRIP)
    for mark in _QUOTES:
        text = text.replace(mark, "")
    return text.upper()


def _match(text: str, rules: Sequence[tuple[Label, tuple[str, ...]]]) -> Label | None:
    for label, words in rules:
        if any(word.upper() in text for word in words):
            return label
    return None


def classify(answer: str | None) -> Label | None:
    """Q1's answer as a label; ``None`` when it names nothing we have a pool for.

    An object ("拿着玩偶"), a face ("微笑") or anything unknown is ``None``:
    a look, but no candidate. "无" or an empty answer is :data:`NONE`.
    """
    text = _normalize(answer)
    if not text or text in _NOTHING:
        return NONE
    found = _match(text, RULES)
    if found is not None:
        return found
    if text.startswith("无"):
        return NONE
    return None


def needs_refine(label: Label | None) -> bool:
    """Whether Q2 is worth asking: the answer was a hand gesture with no name."""
    return label is HAND_ANY


_REFINED = (HEART, THUMB, YEAH)


def refine(label: Label | None, answer: str | None) -> Label | None:
    """The label after Q2: one of the three named hand gestures, or unchanged."""
    found = _match(_normalize(answer), [rule for rule in RULES if rule[0] in _REFINED])
    return found if found is not None else label


def parse_yes(answer: str | None) -> bool:
    """Q3's answer: any 不 / 否 / 没 is a no; else only an answer opening with 是 is a yes."""
    text = _normalize(answer)
    if any(mark in text for mark in "不否没"):
        return False
    return text.startswith("是")


# ---------------------------------------------------------------------------
# When to speak and what.

SEQ_REPEAT = "seq.repeat"
SEQ_STILL = "seq.still"
SEQ_SWITCH = "seq.switch"
SEQ_INVITE = "seq.invite"
OTHER_ANY = "other.any"

# Looks slower than this are discarded when the look runs back to back
# (round_period_sec 0); otherwise the period itself is the limit.
DEFAULT_SLOW_LOOK_SEC = 2.5


@dataclass(frozen=True)
class LookConfig:
    """The ``look`` keys that shape the decisions (plan §2.1), plus the speech estimate."""

    window_sec: float = 0.0
    quick_gap_sec: float = 2.0
    refine_hand: bool = True
    verify: bool = True
    first_after_sec: tuple[float, float] = (2.0, 4.0)
    gap_sec: tuple[float, float] = (3.0, 6.0)
    max_per_round: int = 5
    fresh_sec: float = 5.0
    pool_cooldown_sec: float = 15.0
    fallback_after_sec: float = 2.5
    # Consecutive slow looks after which this round stops looking.
    max_slow_looks: int = 3
    # Speech-length estimate, for when the voice status is stale (chatter's).
    speech_lead_sec: float = 1.0
    speech_per_char_sec: float = 0.22
    # After our own line, how long past its estimated end the voice may stay
    # unconfirmed before the line counts as over anyway.
    speech_margin_sec: float = 0.5

    @classmethod
    def from_look(cls, look: Mapping[str, object]) -> LookConfig:
        """From ``parse_look_json``'s dict; keys it does not carry keep their defaults."""
        values = {}
        for name in cls.__dataclass_fields__:
            if name in look:
                value = look[name]
                values[name] = tuple(float(item) for item in value) if isinstance(value, list | tuple) else value
        return cls(**values)


@dataclass(frozen=True)
class Pick:
    """One line to say now."""

    pool: str
    clip_id: str
    text: str
    # Which rule chose it: named, general, other, seq.repeat, seq.still,
    # seq.switch or fallback.
    reason: str
    # Said before its time: a verified named gesture, quick_gap after the last line.
    quick: bool
    # 1-based within this round, and across the node's life (for event ids).
    seq: int
    total: int


@dataclass
class _Look:
    at: float
    label: Label | None
    # What Q3 needs to ask about this look's picture (the vision's Seen); kept
    # only for named labels and only while fresh.
    ref: object = None
    # Q3's verdict on this look: None until asked.
    verified: bool | None = None


Verify = Callable[[str, object], bool | None]


@dataclass
class LookSay:
    """The decisions of one mirror round after another; the node owns the I/O.

    Call :meth:`enter` on each MIRRORING entry and :meth:`leave` on exit;
    :meth:`voice` with the mirror voice's status (busy, and whether it is
    being heard at all); :meth:`looked` with each look's label; and
    :meth:`decide` to learn whether to say something now. ``verify(key, ref)``
    is Q3 — it returns the model's yes/no, or None when it could not ask.
    """

    config: LookConfig
    pools: Mapping[str, Sequence[tuple[str, str]]]
    rng: random.Random = field(default_factory=random.Random)
    total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._texts = {clip_id: text for entries in self.pools.values() for clip_id, text in entries}
        self._last_clip: str | None = None
        self._last_end = 0.0
        self._estimated_end = 0.0
        self._gap = 0.0
        self._clear()
        self._active = False

    # -- round ------------------------------------------------------------

    def _clear(self) -> None:
        self._looks: list[_Look] = []
        self._said_at: dict[str, float] = {}
        self._used: set[str] = set()
        self._order = {pool: [clip_id for clip_id, _ in entries] for pool, entries in self.pools.items()}
        for order in self._order.values():
            self.rng.shuffle(order)
        self.spoken = 0
        self._slow = 0
        self.stopped = False
        self._last_pick = float("-inf")
        self._busy_now = False
        self._heard = False
        # When our last line was published, until a later status shows it
        # played (busy, then idle) or its estimated end has passed: a status
        # from before the publish knows nothing of it.
        self._pending: float | None = None
        self._pending_busy = False

    def enter(self, now: float) -> None:
        """MIRRORING began: everything starts over, except the last line said."""
        self._clear()
        self._active = True
        self._last_end = now
        self._estimated_end = now
        self._gap = self.rng.uniform(*self.config.first_after_sec)

    def leave(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        """This round has said all it may; no more looking either."""
        return self.spoken >= self.config.max_per_round

    # -- voice ------------------------------------------------------------

    def voice(self, now: float, busy: bool, heard: bool, at: float | None = None) -> None:
        """The mirror voice's status: busy now, whether that report is fresh, and when it came (``at``, default now).

        A clip ends when a fresh status stops saying busy. While the status is
        stale, our own line's length estimate stands in. After our own line,
        only a status that came after it counts: it must say busy and then
        idle, or the line's estimated end (plus a margin) must pass.
        """
        at = now if at is None else at
        if self._pending is not None and heard and at > self._pending:
            if busy:
                self._pending_busy = True
            elif self._pending_busy:
                self._release(now)
        self._settle(now)
        self._heard = heard
        if not heard:
            self._busy_now = False
            return
        if busy or self._busy_now:
            self._last_end = max(self._last_end, now)
        self._busy_now = busy

    def busy(self, now: float) -> bool:
        self._settle(now)
        if self._pending is not None:
            return True
        if self._heard:
            return self._busy_now
        return now < self._estimated_end

    def _settle(self, now: float) -> None:
        """Our line counts as over once its estimated end and the margin have passed."""
        if self._pending is not None and now >= self._estimated_end + self.config.speech_margin_sec:
            self._release(self._estimated_end)

    def _release(self, ended: float) -> None:
        self._pending = None
        self._pending_busy = False
        self._last_end = max(self._last_end, ended)

    @property
    def last_end(self) -> float:
        """When the voice last finished (or is estimated to finish) speaking."""
        if self._heard:
            return self._last_end
        return max(self._last_end, self._estimated_end)

    @property
    def due_at(self) -> float:
        return self.last_end + self._gap

    # -- looking ----------------------------------------------------------

    def wants_look(self, now: float) -> bool:
        """The rhythm's half of the look gate (the node checks stream and actor).

        In MIRRORING, not done, not stopped, and inside the window: always with
        ``window_sec`` 0, otherwise from ``window_sec`` before the next line is
        due until it is said. The voice playing does not close it: a look
        during a clip is what lets the next line answer quickly.
        """
        if not self._active or self.done or self.stopped:
            return False
        window = self.config.window_sec
        return window <= 0 or now >= self.due_at - window

    def wants_refine(self, label: Label | None) -> bool:
        return self.config.refine_hand and needs_refine(label)

    def looked(self, now: float, label: Label | None, ref: object = None) -> None:
        """Record one look that finished in time; ``label`` None is a look with no candidate."""
        self._slow = 0
        keep = ref if label is not None and label.named else None
        self._looks.append(_Look(now, label, keep))
        for look in self._looks:
            if look.ref is not None and now - look.at > self.config.fresh_sec:
                look.ref = None

    def slow(self) -> None:
        """A look ran too long and its answer was discarded."""
        self._slow += 1
        if self._slow >= self.config.max_slow_looks:
            self.stopped = True

    @property
    def labels(self) -> list[Label | None]:
        return [look.label for look in self._looks]

    # -- deciding ---------------------------------------------------------

    def decide(self, now: float, verify: Verify | None = None) -> Pick | None:
        """The line to say now, if any; a returned pick counts as said.

        At most one Q3 per call. Nothing while the voice is busy or the round
        is full. When due, plan §4.4's order; before that, only a quick reply
        to a verified named gesture seen since the last line.
        """
        if not self._active or self.done or self.busy(now):
            return None
        due = self.due_at
        if now >= due:
            pick = self._choose(now, verify)
            if pick is None and now >= due + self.config.fallback_after_sec:
                pick = self._fallback(now)
        elif now >= self.last_end + self.config.quick_gap_sec:
            found = self._named(now, verify, since=self._last_pick)
            pick = (found[0], found[1], "named", True) if found else None
        else:
            pick = None
        if pick is None:
            return None
        return self._commit(now, *pick)

    def _choose(self, now: float, verify: Verify | None) -> tuple[str, str, str, bool] | None:
        found = self._named(now, verify)
        if found:
            return found[0], found[1], "named", False
        fresh = [
            look for look in reversed(self._looks) if look.label is not None and now - look.at <= self.config.fresh_sec
        ]
        for look in fresh:
            pool = self._general_pool(look)
            clip = self._next(pool, now) if pool != OTHER_ANY else None
            if clip is not None:
                return pool, clip, "general", False
        for look in fresh:
            clip = self._next(OTHER_ANY, now) if self._general_pool(look) == OTHER_ANY else None
            if clip is not None:
                return OTHER_ANY, clip, "other", False
        for pool, rule in ((SEQ_REPEAT, self._repeating), (SEQ_STILL, self._still), (SEQ_SWITCH, self._switched)):
            clip = self._next(pool, now) if rule(now) else None
            if clip is not None:
                return pool, clip, pool, False
        return None

    def _named(self, now: float, verify: Verify | None, since: float = float("-inf")) -> tuple[str, str] | None:
        """A fresh named gesture whose pool is free, after Q3 says yes (one Q3 at most)."""
        asked = False
        for look in reversed(self._looks):
            label = look.label
            if now - look.at > self.config.fresh_sec or look.at <= since:
                break
            if label is None or not label.named or look.verified is False:
                continue
            clip = self._next(label.pool, now)
            if clip is None:
                continue
            if not self.config.verify or look.verified:
                return label.pool, clip
            if asked or verify is None or look.ref is None:
                continue
            asked = True
            look.verified = bool(verify(label.verify_key, look.ref))
            look.ref = None
            if look.verified:
                return label.pool, clip
        return None

    def _general_pool(self, look: _Look) -> str | None:
        """Where a look lands when it is not said by name: named ones go to their family's any."""
        label = look.label
        if label.named:
            return label.coarse_pool
        return label.pool

    def _repeating(self, now: float) -> bool:
        last = self._looks[-3:]
        if len(last) < 3 or any(look.label is None for look in last):
            return False
        label = last[0].label
        if label.family not in FAMILIES or any(look.label.tag != label.tag for look in last):
            return False
        return any(self._cooling(pool, now) for pool in (label.pool, label.coarse_pool) if pool)

    def _still(self, now: float) -> bool:
        last = self._looks[-2:]
        return len(last) == 2 and all(look.label is DOWN for look in last)

    def _switched(self, now: float) -> bool:
        last = self._looks[-2:]
        if len(last) < 2 or any(look.label is None or look.label.family not in FAMILIES for look in last):
            return False
        return last[0].label.family != last[1].label.family

    def _fallback(self, now: float) -> tuple[str, str, str, bool] | None:
        for pool, cooled in ((SEQ_INVITE, True), (OTHER_ANY, True), (SEQ_INVITE, False), (OTHER_ANY, False)):
            clip = self._next(pool, now, ignore_cooldown=not cooled)
            if clip is not None:
                return pool, clip, "fallback", False
        return None

    # -- pools ------------------------------------------------------------

    def _cooling(self, pool: str, now: float) -> bool:
        said = self._said_at.get(pool)
        return said is not None and now - said < self.config.pool_cooldown_sec

    def _next(self, pool: str | None, now: float, *, ignore_cooldown: bool = False) -> str | None:
        """The pool's next unsaid clip this round, or None when cooling or used up."""
        if pool is None or pool not in self._order:
            return None
        if not ignore_cooldown and self._cooling(pool, now):
            return None
        for clip_id in self._order[pool]:
            if clip_id in self._used:
                continue
            if self.spoken == 0 and clip_id == self._last_clip:
                continue
            return clip_id
        return None

    def _commit(self, now: float, pool: str, clip_id: str, reason: str, quick: bool) -> Pick:
        self._used.add(clip_id)
        self._said_at[pool] = now
        self._last_clip = clip_id
        self._last_pick = now
        self.spoken += 1
        self.total += 1
        text = self._texts[clip_id]
        self._last_end = max(self._last_end, now)
        self._estimated_end = now + self.config.speech_lead_sec + self.config.speech_per_char_sec * len(text)
        self._pending = now
        self._pending_busy = False
        self._gap = self.rng.uniform(*self.config.gap_sec)
        return Pick(pool, clip_id, text, reason, quick, self.spoken, self.total)


__all__ = [
    "FAMILIES",
    "LABELS",
    "QUESTIONS",
    "RULES",
    "Label",
    "LookConfig",
    "LookSay",
    "Pick",
    "classify",
    "needs_refine",
    "parse_yes",
    "refine",
]
