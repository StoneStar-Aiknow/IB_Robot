"""Deployment answer rules that hold whatever the planner says: ask before a guess.

Gift eligibility is owned by the Agent gift state store, independently of dialogue
sessions. These wording rules decide when a proposal becomes an answer instead:

- ``only_one``: a request for two fruits, or all of them, is answered and nothing is picked;
- ``confirm_pick``: "你可以抓苹果吗" asks about an ability. The pick is held and the visitor
  asked; a yes runs exactly the held plan, without a second planning call. A polite request
  ("你能给我拿个橙子吗") and a choice right after the fruit menu are not asked again;
- ``recommend`` / ``random_pick``: "你推荐一个吧" is answered by offering to pick; "你帮我随便挑
  一个", or a yes to that offer, delivers a fruit the robot chooses;
- ``non_fruit_skill``: a pick of a fruit nobody named ("帮我拿一下我的手机" planned as an
  apple, 2026-11-10 rig) becomes this recorded answer instead, unless the visitor is answering
  the robot's own question or the fruit menu. A personal item named outright is refused even
  then: the menu is no licence to pick a phone;
- ``unavailable_fruit_skill``: a fruit the stand does not have ("给我一个梨") is answered by
  this recording, never by the personal-item refusal;
- ``contact_replies``: a request for physical contact ("你能抱抱我吗") is answered by one of
  these lines, never repeated twice in a row, and no motion runs: on 2026-11-10 the planner
  turned the same sentence into a blown kiss once and a handshake once;
- ``confirm_delivery``: "把橙子直接给我" asks for a hand-over the stand cannot do. The robot
  says how the gift arrives and asks; a yes runs the held pick and place;
- ``unsupported_skill``: a plan that names a skill this catalog does not offer is answered by
  this recording instead of failing in silence;
- ``repeat_skills``: a recorded answer that has a follow-up version (the price: first the
  open source answer, then the robot body's) plays the follow-up the second time.

ROS-free and deterministic apart from the random fruit (``rng``) and the reply chosen
(``choose_line``), both injectable.
"""

from __future__ import annotations

import random
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ibrobot_agent.contracts import PlannerOutcome
from ibrobot_agent.planner import planner_outcome_from_payload

LINE_KEYS = ("only_one", "already_given", "confirm_pick", "recommend", "random_pick", "confirm_delivery")
_FRUITS = {"柠檬": "a lemon", "苹果": "an apple", "橙子": "an orange", "橘子": "an orange", "桔子": "an orange"}
_QUANTITY = re.compile(r"两个|两份|三个|三份|全都|全部|都要|全要|每样|每种|各一个|各来一个|多拿|多要")
_GIFT_WORDS = re.compile(r"水果|礼物|柠檬|苹果|橙子|橘子|要|拿|给")
#: "两个游戏我都要玩" asks for both games (W06), not two fruits.
_GAME_WORDS = re.compile(r"游戏|玩|模仿|照镜子|动作")
#: "你可以/能/会 + 抓/拿 ... 吗": a question about an ability, not yet a request.
_ABILITY = re.compile(r"^(?:你)?(?:可以|能|会|能不能|可不可以|会不会)(?:帮|给)?.{0,4}(?:抓|拿|取|夹|递)")
_POLITE = ("给我", "帮我", "替我", "我要", "我想", "麻烦")
_RECOMMEND = re.compile(r"推荐|哪个好|哪个比较好|选哪个|挑哪个|哪种好")
_RANDOM_PICK = re.compile(
    r"(?:帮我|你|替我)(?:来)?(?:随便)?(?:选|挑|拿)(?:一个|一份|个|吧)|随便(?:选|挑|拿)(?:一个|一份|个)?|你(?:来)?(?:定|决定)吧?"
)
_YES = re.compile(
    r"^(?:嗯+|好+|好的|好呀|好啊|好吧|行+|可以|要|要的|对|是|是的|没问题|来吧|拿吧|快拿|当然|ok|okay)(?:呀|啊|吧|的|哒|呢)?$"
)
#: A pick is only for a fruit the visitor named, or the gift itself ("我要水果", "随便拿一个").
_NAMES_A_GIFT = re.compile(r"柠檬|苹果|橙子|橘子|桔子|水果|礼物|随便")
_MENU_SKILLS = frozenset({"offer_fruit_menu", "fruit_not_available"})
_PICK = "pick_object"
_PLACE = "place_in_container"
#: Fruits a visitor may ask for that the stand does not have.
_UNAVAILABLE_FRUITS = re.compile(r"梨|西瓜|香蕉|葡萄|菠萝|草莓|桃|芒果|荔枝|哈密瓜|柚子|榴莲|山竹|火龙果")
_PERSONAL_ITEMS = re.compile(
    r"手机|钱包|我的包|把包|拿包|背包|书包|包包|杯子|水杯|钥匙|眼镜|帽子|充电宝|耳机|证件|工牌"
)
_FETCH = re.compile(r"拿|递|取|抓|给我")
#: Physical contact with a person; the summit catalog has no contact skill (handshake is disabled).
_CONTACT = re.compile(
    r"抱抱|拥抱|抱我|抱你|抱一下|抱一抱|抱一个|搂我|搂一下|摸摸|摸一下|摸你|摸我|亲亲|亲我|亲你|亲一下"
    r"|握手|握个手|握一下手|牵手|牵着我|拉着我|拉我的手|扶我|扶一下"
)
#: Asking how the gift arrives: thrown, handed over, put in the hand.
_DELIVERY_METHOD = re.compile(r"扔|抛|投|递|(?:直接|亲手)给我|手里|手上")
#: "要不要", "能不能": a question, not a negation.
_A_NOT_A = re.compile(r"要不要|能不能|会不会|可不可以|可以不可以|行不行|好不好|愿不愿意")
_NEGATED = re.compile(r"不要|不用|别|不想|不需要|不必|甭")
#: Telling about something, not asking for it.
_NARRATION = re.compile(r"昨天|刚才|刚刚|以前|上次|已经|他|她|那个人")
_WHY = re.compile(r"为什么|为何|怎么")
# A conservative routing guard, not a parser/translator. Destination wording must
# reach the full planner intact instead of taking a default-only fruit shortcut.
_DESTINATION = re.compile(
    r"碗|盆|盘|篮|盒|箱|桶|杯|容器|放|送到|送进|装进"
    r"|\b(?:bowls?|basins?|plates?|trays?|baskets?|boxes|box|bins?|buckets?|cups?|containers?|put|place|into|inside)\b",
    re.IGNORECASE,
)


def _folded(text: str) -> str:
    return re.sub(r"[\s，,。.！!？?、~～]", "", str(text)).casefold()


def _asks(text: str, pattern: re.Pattern, *, questions: bool = True) -> bool:
    """``text`` asks for what ``pattern`` names: not negated, not told about, and (unless
    ``questions``) not asking why."""
    plain = _A_NOT_A.sub("", text)
    if not pattern.search(plain) or _NEGATED.search(plain) or _NARRATION.search(plain):
        return False
    return questions or not _WHY.search(plain)


def contact_request(text: str) -> bool:
    """A request for, or a question about, physical contact with the robot."""
    return _asks(_folded(text), _CONTACT)


def _delivery_request(folded: str) -> bool:
    return _asks(folded, _DELIVERY_METHOD, questions=False)


def _named_fruit(folded: str) -> str | None:
    return next((target for name, target in _FRUITS.items() if name in folded), None)


def _picks(outcome: PlannerOutcome) -> bool:
    return outcome.kind == "workflow" and any(step.skill_name == _PICK for step in outcome.steps)


@dataclass
class _Round:
    menu_open: bool = False
    held: tuple[str, PlannerOutcome | None] | None = None
    played: set[str] | None = None
    last_contact_reply: str = ""


class AnswerRules:
    """Per-round answer rules of one Agent; thread-safe."""

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        *,
        container_name: str = "",
        rng: Callable[[list[str]], str] | None = None,
        choose_line: Callable[[list[str]], str] | None = None,
        max_rounds: int = 256,
    ) -> None:
        config = dict(config or {})
        lines = dict(config.get("lines") or {})
        unknown = sorted(set(lines) - set(LINE_KEYS))
        if unknown or any(not isinstance(value, str) or not value.strip() for value in lines.values()):
            raise ValueError(f"answer_rules.lines must be texts for keys of {LINE_KEYS}; got {unknown}")
        repeat = dict(config.get("repeat_skills") or {})
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in repeat.items()):
            raise ValueError("answer_rules.repeat_skills must map a skill to its follow-up skill")
        contact = config.get("contact_replies") or []
        if not isinstance(contact, list) or any(not isinstance(line, str) or not line.strip() for line in contact):
            raise ValueError("answer_rules.contact_replies must be a list of non-empty texts")
        self.lines = lines
        self.repeat_skills = repeat
        self.contact_replies = tuple(dict.fromkeys(line.strip() for line in contact))
        self.non_fruit_skill = str(config.get("non_fruit_skill", "") or "")
        self.unavailable_fruit_skill = str(config.get("unavailable_fruit_skill", "") or "")
        self.unsupported_skill = str(config.get("unsupported_skill", "") or "")
        self.container_name = container_name
        self._rng = rng or random.choice
        self._choose_line = choose_line or random.choice
        self._rounds: OrderedDict[tuple[str, ...], _Round] = OrderedDict()
        self._max_rounds = max_rounds
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(
            self.lines
            or self.repeat_skills
            or self.non_fruit_skill
            or self.unavailable_fruit_skill
            or self.unsupported_skill
            or self.contact_replies
        )

    def may_override(self, text: str) -> bool:
        """Whether a planned answer to ``text`` may be replaced: say nothing of it before then."""
        if not self.enabled:
            return False
        folded = _folded(text)
        return bool(
            (self.contact_replies and _CONTACT.search(folded))
            or _DELIVERY_METHOD.search(folded)
            or _UNAVAILABLE_FRUITS.search(folded)
            or _PERSONAL_ITEMS.search(folded)
        )

    def _round(self, key: tuple[str, ...]) -> _Round:
        item = self._rounds.get(key)
        if item is None:
            item = self._rounds[key] = _Round(played=set())
            while len(self._rounds) > self._max_rounds:
                self._rounds.popitem(last=False)
        self._rounds.move_to_end(key)
        return item

    def _answer(self, line: str, kind: str = "conversation", missing: tuple[str, ...] = ()) -> PlannerOutcome:
        return PlannerOutcome(kind=kind, user_message=self.lines[line], missing_fields=missing)

    def _contact_answer(self, item: _Round) -> PlannerOutcome:
        choices = [line for line in self.contact_replies if line != item.last_contact_reply] or list(
            self.contact_replies
        )
        item.last_contact_reply = self._choose_line(choices)
        return PlannerOutcome(kind="conversation", user_message=item.last_contact_reply)

    @staticmethod
    def skill_answer(skill: str, message: str = "好的。") -> PlannerOutcome:
        """A one-step workflow of a recorded answer skill."""
        return planner_outcome_from_payload(
            {
                "kind": "workflow",
                "user_message": message,
                "summary": skill,
                "steps": [{"schema_version": 1, "skill_name": skill}],
            }
        )

    def _delivery(self, target: str, message: str) -> PlannerOutcome | None:
        if not self.container_name:
            return None
        return planner_outcome_from_payload(
            {
                "kind": "workflow",
                "user_message": message,
                "summary": "pick a fruit for the visitor",
                "steps": [
                    {"schema_version": 1, "skill_name": _PICK, "target_name": target},
                    {
                        "schema_version": 1,
                        "skill_name": _PLACE,
                        "target_name": target,
                        "container_name": self.container_name,
                    },
                ],
            }
        )

    def _random_delivery(self, line: str | None = None) -> PlannerOutcome | None:
        target = self._rng(sorted(set(_FRUITS.values())))
        return self._delivery(target, self.lines.get(line or "random_pick") or "好的，我帮你挑一个。")

    def _hold_delivery(self, item: _Round, outcome: PlannerOutcome | None) -> PlannerOutcome | None:
        """Ask how the gift arrives before a pick the visitor wants handed over."""
        if outcome is None:
            return None
        item.held = ("confirm_delivery", outcome)
        return self._answer("confirm_delivery", "needs_clarification", ("confirmation",))

    def before_planning(self, session: tuple[str, ...], text: str, *, replying: bool) -> PlannerOutcome | None:
        """An outcome decided from the words and the round alone, or None to plan as usual."""
        if not self.enabled:
            return None
        folded = _folded(text)
        with self._lock:
            item = self._round(session)
            held, item.held = item.held, None
            if replying and held is not None and _YES.match(folded):
                kind, outcome = held
                if kind in {"confirm_pick", "confirm_delivery"} and outcome is not None:
                    return outcome
                if kind == "recommend":
                    return self._random_delivery()
            if self.contact_replies and _asks(folded, _CONTACT):
                return self._contact_answer(item)
            named = _named_fruit(folded)
            if self.unavailable_fruit_skill and not named and _asks(folded, _UNAVAILABLE_FRUITS):
                return self.skill_answer(self.unavailable_fruit_skill)
            if self.non_fruit_skill and not named and _PERSONAL_ITEMS.search(folded) and _FETCH.search(folded):
                return self.skill_answer(self.non_fruit_skill)
            if (
                "only_one" in self.lines
                and _QUANTITY.search(folded)
                and _GIFT_WORDS.search(folded)
                and not _GAME_WORDS.search(folded)
            ):
                return self._answer("only_one")
            if _DESTINATION.search(text):
                # Preserve explicit containers, ambiguous references and changes
                # to a held plan. The planner sees the whole utterance/context.
                return None
            if "confirm_delivery" in self.lines and _delivery_request(folded):
                if named:
                    return self._hold_delivery(item, self._delivery(named, "好的。"))
                if _RANDOM_PICK.search(folded):
                    return self._hold_delivery(item, self._random_delivery())
                return None  # Which fruit is not said yet: the planner asks, or explains the rule.
            if not named and "random_pick" in self.lines and _RANDOM_PICK.search(folded):
                return self._random_delivery()
            if not named and "recommend" in self.lines and _RECOMMEND.search(folded):
                item.held = ("recommend", None)
                return self._answer("recommend", "needs_clarification", ("fruit",))
        return None

    def after_planning(
        self, session: tuple[str, ...], text: str, outcome: PlannerOutcome, *, replying: bool
    ) -> PlannerOutcome:
        """The outcome the round allows for the planner's (or a direct command's) ``outcome``."""
        if not self.enabled:
            return outcome
        folded = _folded(text)
        with self._lock:
            item = self._round(session)
            if self.contact_replies and outcome.kind == "workflow" and _CONTACT.search(folded):
                # Whatever motion the planner chose for words about contact, none of it runs.
                return self._contact_answer(item)
            named = _named_fruit(folded)
            if outcome.kind == "workflow" and len(outcome.steps) == 1:
                skill = outcome.steps[0].skill_name
                follow_up = self.repeat_skills.get(skill)
                if follow_up and skill in (item.played or set()):
                    step = outcome.steps[0].__class__(schema_version=1, skill_name=follow_up)
                    outcome = PlannerOutcome(
                        kind="workflow", user_message=outcome.user_message, summary=follow_up, steps=(step,)
                    )
            if _picks(outcome):
                if self.unavailable_fruit_skill and not named and _UNAVAILABLE_FRUITS.search(folded):
                    return self.skill_answer(self.unavailable_fruit_skill)
                if (
                    self.non_fruit_skill
                    and not named
                    and (
                        _PERSONAL_ITEMS.search(folded)
                        or (not replying and not item.menu_open and not _NAMES_A_GIFT.search(folded))
                    )
                ):
                    step = outcome.steps[0].__class__(schema_version=1, skill_name=self.non_fruit_skill)
                    return PlannerOutcome(
                        kind="workflow", user_message="我只拿水果礼物哦", summary=self.non_fruit_skill, steps=(step,)
                    )
                if "confirm_delivery" in self.lines and _delivery_request(folded):
                    return self._hold_delivery(item, outcome)
                if (
                    "confirm_pick" in self.lines
                    and not replying
                    and not item.menu_open
                    and _ABILITY.search(folded)
                    and (
                        folded.endswith(("吗", "嘛"))
                        or any(word in folded for word in ("能不能", "可不可以", "会不会"))
                    )
                    and not any(word in folded for word in _POLITE)
                ):
                    item.held = ("confirm_pick", outcome)
                    return self._answer("confirm_pick", "needs_clarification", ("confirmation",))
                return outcome
            return outcome

    def unsupported(self, catalog: Mapping[str, Any], outcome: PlannerOutcome) -> PlannerOutcome:
        """A workflow naming a skill the live catalog does not offer becomes the refusal recording."""
        if not self.unsupported_skill or outcome.kind != "workflow":
            return outcome
        skills = catalog.get("skills")
        if not isinstance(skills, list) or not skills:
            return outcome  # No catalog to judge by: the Gateway check decides.
        visible = {
            skill.get("name") for skill in skills if isinstance(skill, Mapping) and skill.get("planner_visible") is True
        }
        if self.unsupported_skill not in visible or all(step.skill_name in visible for step in outcome.steps):
            return outcome
        return self.skill_answer(self.unsupported_skill)

    def proposed(self, session: tuple[str, ...], outcome: PlannerOutcome) -> None:
        """The outcome is going to the robot: remember what the round has been told and shown."""
        if not self.enabled or outcome.kind != "workflow":
            return
        with self._lock:
            item = self._round(session)
            names = {step.skill_name for step in outcome.steps}
            item.played = (item.played or set()) | names
            item.menu_open = bool(names & _MENU_SKILLS)

    def answered(self, session: tuple[str, ...]) -> None:
        """Any other turn closes the fruit menu: a later ability question is asked again."""
        if not self.enabled:
            return
        with self._lock:
            self._round(session).menu_open = False
