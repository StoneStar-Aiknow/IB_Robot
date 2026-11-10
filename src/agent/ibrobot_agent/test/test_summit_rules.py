"""Answer rules and gate rules of the 2026-11-10 summit plan (1_需要修改的代码.md sections 1.4, 1.5, 2.4, 2.6).

The sentences are the rig's (runs 5 to 7): what was wrongly rejected must pass, the chatter of
the same runs must stay out, and the round rules must hold whatever the planner proposes.
"""

from __future__ import annotations

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent import semantic_gate as gate_module
from ibrobot_agent.answer_rules import AnswerRules
from ibrobot_agent.contracts import PlannerOutcome
from ibrobot_agent.planner import build_planner_messages, deployment_facts_from_config, direct_command_outcome
from ibrobot_agent.semantic_gate import GRADE_CONVERSATION, SemanticGate, normalize_text

LINES = {
    "only_one": "嘿嘿，这轮只能选一份哦～",
    "already_given": "这轮已经领过啦，下次再来哦～",
    "confirm_pick": "可以呀，现在就给你拿吗？",
    "recommend": "每个都有礼物哦，要我帮你挑吗？",
    "random_pick": "每个都有礼物，我帮你挑一个！",
}
ROUND = ("session-1", "local_operator", "voice_asr", "aimdk_x2_summit")


def _rules(**overrides):
    config = {"once_per_round": True, "lines": LINES, "repeat_skills": {"answer_price": "answer_body_price"}}
    config.update(overrides)
    return AnswerRules(config, container_name="an orange bowl", rng=lambda options: "an apple")


def _workflow(*names, target="an apple"):
    steps = tuple(
        CanonicalWorkflowStep(
            schema_version=1, skill_name=name, target_name=target if "pick" in name or "place" in name else ""
        )
        for name in names
    )
    return PlannerOutcome(kind="workflow", user_message="好的", summary="s", steps=steps)


PICK = _workflow("pick_object", "place_in_container")


# ---- 1.5 one gift per round --------------------------------------------------------------------


@pytest.mark.parametrize("text", ["这些水果我全都要", "请给我两个苹果", "不能要两个吗", "我全都要"])
def test_asking_for_more_than_one_fruit_is_answered(text):
    outcome = _rules().before_planning(ROUND, text, replying=False)
    assert outcome is not None and outcome.kind == "conversation" and outcome.user_message == LINES["only_one"]


@pytest.mark.parametrize("text", ["模仿和水果两个游戏我都想玩", "两个游戏我都要玩"])
def test_wanting_both_games_is_not_two_fruits(text):
    assert _rules().before_planning(ROUND, text, replying=False) is None


def test_a_second_pick_after_a_delivery_is_answered_and_a_new_round_may_pick():
    rules = _rules()
    rules.delivered(ROUND)
    assert rules.after_planning(ROUND, "再给我拿个苹果", PICK, replying=False).user_message == LINES["already_given"]
    other = ("session-2", *ROUND[1:])
    assert rules.after_planning(other, "我要一个苹果", PICK, replying=False) is PICK


def test_a_failed_delivery_does_not_count():
    rules = _rules()
    assert rules.after_planning(ROUND, "我要一个苹果", PICK, replying=False) is PICK
    assert not rules.has_delivered(ROUND)


# ---- 1.4 an ability question is asked back; a polite request is not -----------------------------


def test_an_ability_question_is_confirmed_and_a_yes_runs_the_held_plan():
    rules = _rules()
    asked = rules.after_planning(ROUND, "你可以抓苹果吗", PICK, replying=False)
    assert asked.kind == "needs_clarification" and asked.user_message == LINES["confirm_pick"]
    assert rules.before_planning(ROUND, "好的", replying=True) is PICK


def test_a_no_drops_the_held_plan():
    rules = _rules()
    rules.after_planning(ROUND, "你可以抓苹果吗", PICK, replying=False)
    assert rules.before_planning(ROUND, "不用了", replying=True) is None
    assert rules.before_planning(ROUND, "好的", replying=True) is None


@pytest.mark.parametrize("text", ["你能给我拿个橙子吗", "可以帮我拿个苹果吗", "我要一个苹果"])
def test_a_polite_request_runs_directly(text):
    assert _rules().after_planning(ROUND, text, PICK, replying=False) is PICK


def test_right_after_the_menu_an_ability_question_is_a_choice():
    rules = _rules()
    rules.proposed(ROUND, _workflow("offer_fruit_menu"))
    assert rules.after_planning(ROUND, "你可以抓苹果吗", PICK, replying=False) is PICK


# ---- Q03: recommend, or pick for the visitor -------------------------------------------------


def test_a_recommendation_offers_to_pick_and_a_yes_delivers_a_fruit():
    rules = _rules()
    offered = rules.before_planning(ROUND, "你帮我推荐一个水果吧", replying=False)
    assert offered.kind == "needs_clarification" and offered.user_message == LINES["recommend"]
    delivery = rules.before_planning(ROUND, "好呀", replying=True)
    assert [step.skill_name for step in delivery.steps] == ["pick_object", "place_in_container"]
    assert delivery.steps[1].container_name == "an orange bowl"


@pytest.mark.parametrize("text", ["你帮我选一个", "帮我随便挑一个", "随便拿一个"])
def test_picking_for_the_visitor_delivers_a_fruit_with_the_recorded_lead(text):
    outcome = _rules().before_planning(ROUND, text, replying=False)
    assert outcome.user_message == LINES["random_pick"]
    assert outcome.steps[0].target_name == outcome.steps[1].target_name == "an apple"


def test_a_named_fruit_is_not_a_random_pick():
    assert _rules().before_planning(ROUND, "帮我拿个苹果", replying=False) is None


# ---- Q17: the second price question is about the body ----------------------------------------


def test_the_price_answer_has_a_follow_up():
    rules = _rules()
    first = rules.after_planning(ROUND, "你多少钱一台", _workflow("answer_price"), replying=False)
    assert first.steps[0].skill_name == "answer_price"
    rules.proposed(ROUND, first)
    second = rules.after_planning(ROUND, "那多少钱", _workflow("answer_price"), replying=False)
    assert second.steps[0].skill_name == "answer_body_price"


def test_without_configuration_nothing_changes():
    rules = AnswerRules(None)
    assert rules.before_planning(ROUND, "我全都要", replying=False) is None
    assert rules.after_planning(ROUND, "你可以抓苹果吗", PICK, replying=False) is PICK


# ---- 2.4 / 2.6 the intent gate -----------------------------------------------------------------


def _gate():
    payload = {
        "schema_version": 1,
        "feature": {"ngrams": [1], "dim": 4096, "hash": "fnv1a32", "pad": "<>"},
        "wake_words": ["欧拉欧拉"],
        "bias": -4.0,
        "weights": [[gate_module._fnv1a32(char) % 4096, 6.0] for char in normalize_text("挥")],
    }
    return SemanticGate(
        payload,
        threshold=0.10,
        lexicon=("挥手", "苹果", "柠檬", "橙子", "水果", "抓进", "放进", "碗里"),
        dialogue_requests=("讲个故事", "作首诗", "自我介绍"),
        wake_prefixes=("欧拉",),
    )


@pytest.mark.parametrize(
    "text",
    [
        "除了这两个功能你还有其他功能吗",
        "你有多高呢",
        "是谁把你做出来的",
        "给大家做个自我介绍",
        "哈喽欧拉",
        "欧拉你好",
        "给我作首诗",
    ],
)
def test_the_rejected_questions_of_the_rig_are_admitted(text):
    decision = _gate().judge(text, session_open=True, followup_open=False, conversation=True)
    assert not decision.rejected, decision.reason


@pytest.mark.parametrize(
    "text", ["再来一个", "今天是几号", "那九十九乘以九十九等于几", "你不应该在北京的展台上吗你不在深圳的展台上啊"]
)
def test_follow_ups_are_admitted_in_the_window(text):
    decision = _gate().judge(text, session_open=True, followup_open=True, conversation=True)
    assert not decision.rejected and decision.grade == GRADE_CONVERSATION


@pytest.mark.parametrize("text", ["把苹果抓进黑色的碗里", "我要一个苹果"])
def test_fruit_commands_pass(text):
    assert not _gate().judge(text, session_open=True, conversation=True).rejected


@pytest.mark.parametrize("text", ["你跟open欧拉是什么关系他说的", "周哥举手的动作", "他实际的表现是什么"])
def test_chatter_stays_out(text):
    assert _gate().judge(text, session_open=True, followup_open=False, conversation=True).rejected


# ---- 3.1 the planner's facts and recorded clarifications -----------------------------------------


class _Request:
    text = "你在哪个城市"
    reply_to_request_id = None


def test_the_planner_is_given_the_facts_and_the_spoken_style():
    facts, lines = deployment_facts_from_config(
        {
            "deployment_facts": ["地点是北京中关村国际创新中心。"],
            "clarification_lines": {"confirm_pick": "可以呀，现在就给你拿吗？"},
        }
    )
    system, user = build_planner_messages(
        request=_Request(), context={}, catalog={}, deployment_facts=facts, clarification_lines=lines
    )
    assert "deployment.facts are the only facts" in system["content"] and "15 Chinese" in system["content"]
    assert "北京中关村国际创新中心" in user["content"] and "可以呀，现在就给你拿吗？" in user["content"]
    plain_system, _ = build_planner_messages(request=_Request(), context={}, catalog={})
    assert "deployment.facts" not in plain_system["content"]


def test_a_direct_command_ignores_case_spaces_and_punctuation():
    catalog = {"skills": [{"name": "explain_ib_robot", "planner_visible": True}]}
    outcome = direct_command_outcome("IB ROBOT是什么意思？", {"IB Robot是什么意思": "explain_ib_robot"}, catalog)
    assert outcome is not None and outcome.steps[0].skill_name == "explain_ib_robot"


def test_a_pick_of_something_that_is_not_a_fruit_becomes_the_recorded_refusal():
    rules = _rules(non_fruit_skill="decline_personal_item")
    refused = rules.after_planning(ROUND, "帮我拿一下我的手机", PICK, replying=False)
    assert [step.skill_name for step in refused.steps] == ["decline_personal_item"]
    assert rules.after_planning(ROUND, "我要一个苹果", PICK, replying=False) is PICK
    rules.proposed(ROUND, _workflow("offer_fruit_menu"))
    assert rules.after_planning(ROUND, "就要红的", PICK, replying=False) is PICK  # an answer to the menu


# ---- 2026-11-10 mock run: contact, delivery, fruit and catalog boundaries ----------------------

CONTACT = [
    "出于安全考虑，我暂时不支持这类肢体接触哦。",
    "为了安全，这类身体接触的互动暂时还不支持哦。",
    "这类接触互动暂时不支持，我们可以聊聊天。",
]


@pytest.mark.parametrize(
    "text",
    [
        "你能抱抱我吗",
        "给我一个拥抱",
        "你能不能抱抱我",
        "你会不会抱我",
        "我能摸摸你吗",
        "亲我一下",
        "握个手吧",
        "为什么你不能抱我",
    ],
)
def test_a_contact_request_is_answered_and_never_planned(text):
    outcome = _rules(contact_replies=CONTACT).before_planning(ROUND, text, replying=False)
    assert outcome is not None and outcome.kind == "conversation" and outcome.user_message in CONTACT


def test_the_contact_reply_does_not_repeat_twice_in_a_row():
    rules = _rules(contact_replies=CONTACT)
    said = [rules.before_planning(ROUND, "抱抱我", replying=False).user_message for _ in range(30)]
    assert all(a != b for a, b in zip(said, said[1:], strict=False)) and set(said) <= set(CONTACT)


@pytest.mark.parametrize("text", ["不要抱我", "刚才他抱了你吗", "抱歉，我没听清", "别摸我"])
def test_words_that_do_not_ask_for_contact_are_planned(text):
    assert _rules(contact_replies=CONTACT).before_planning(ROUND, text, replying=False) is None


@pytest.mark.parametrize("skill", ["blow_kiss", "handshake", "wave_hand"])
def test_a_motion_planned_for_words_about_contact_never_runs(skill):
    outcome = _rules(contact_replies=CONTACT).after_planning(ROUND, "别抱我", _workflow(skill), replying=False)
    assert outcome.kind == "conversation" and outcome.user_message in CONTACT and not outcome.steps


def test_a_blown_kiss_asked_for_plainly_still_runs():
    rules = _rules(contact_replies=CONTACT)
    kiss = _workflow("blow_kiss")
    assert rules.before_planning(ROUND, "来个飞吻", replying=False) is None
    assert rules.after_planning(ROUND, "来个飞吻", kiss, replying=False) is kiss


def test_without_contact_replies_contact_words_are_planned():
    assert _rules().before_planning(ROUND, "你能抱抱我吗", replying=False) is None


@pytest.mark.parametrize("text", ["给我一个梨", "可以选香蕉吗", "有没有葡萄", "我要西瓜"])
def test_a_fruit_the_stand_lacks_is_not_a_personal_item(text):
    rules = _rules(unavailable_fruit_skill="fruit_not_available", non_fruit_skill="decline_personal_item")
    outcome = rules.before_planning(ROUND, text, replying=False)
    assert [step.skill_name for step in outcome.steps] == ["fruit_not_available"]
    planned = rules.after_planning(ROUND, text, PICK, replying=False)
    assert [step.skill_name for step in planned.steps] == ["fruit_not_available"]


def test_a_named_fruit_wins_over_one_the_stand_lacks():
    rules = _rules(unavailable_fruit_skill="fruit_not_available")
    assert rules.before_planning(ROUND, "我不要香蕉，我要苹果", replying=False) is None


@pytest.mark.parametrize("text", ["帮我拿一下我的手机", "把我的包拿过来", "帮我拿一下那个杯子"])
def test_a_personal_item_is_refused_before_planning(text):
    rules = _rules(non_fruit_skill="decline_personal_item")
    outcome = rules.before_planning(ROUND, text, replying=False)
    assert [step.skill_name for step in outcome.steps] == ["decline_personal_item"]


def test_the_menu_is_no_licence_to_pick_a_personal_item():
    rules = _rules(non_fruit_skill="decline_personal_item")
    rules.proposed(ROUND, _workflow("offer_fruit_menu"))
    refused = rules.after_planning(ROUND, "我的钥匙", PICK, replying=True)
    assert [step.skill_name for step in refused.steps] == ["decline_personal_item"]


DELIVERY = "不能直接递给你哦，我放进碗里，滑到领取盒你自己拿，可以吗？"


def _delivery_rules():
    return _rules(lines={**LINES, "confirm_delivery": DELIVERY})


@pytest.mark.parametrize("text", ["把橙子直接给我", "把橙子扔给我", "把橙子递到我手里"])
def test_a_hand_over_is_explained_and_a_yes_runs_the_held_pick(text):
    rules = _delivery_rules()
    asked = rules.before_planning(ROUND, text, replying=False)
    assert asked.kind == "needs_clarification" and asked.user_message == DELIVERY and not asked.steps
    delivery = rules.before_planning(ROUND, "好的", replying=True)
    assert [step.skill_name for step in delivery.steps] == ["pick_object", "place_in_container"]
    assert {step.target_name for step in delivery.steps} == {"an orange"}
    assert delivery.steps[1].container_name == "an orange bowl"
    assert delivery.user_message not in LINES.values()  # no "I pick one for you" lead


def test_a_no_to_the_hand_over_drops_the_held_pick():
    rules = _delivery_rules()
    rules.before_planning(ROUND, "把橙子直接给我", replying=False)
    assert rules.before_planning(ROUND, "不要了", replying=True) is None
    assert rules.before_planning(ROUND, "好的", replying=True) is None


def test_a_hand_over_after_a_delivery_is_the_already_given_answer():
    rules = _delivery_rules()
    rules.delivered(ROUND)
    assert rules.before_planning(ROUND, "把苹果扔给我", replying=False).user_message == LINES["already_given"]


def test_a_planned_hand_over_is_held_too_and_why_questions_are_planned():
    rules = _delivery_rules()
    asked = rules.after_planning(ROUND, "你直接递给我吧", PICK, replying=False)
    assert asked.kind == "needs_clarification" and asked.user_message == DELIVERY
    assert rules.before_planning(ROUND, "为什么不能直接递给我", replying=False) is None
    assert rules.before_planning(ROUND, "直接递给我", replying=False) is None  # which fruit: the planner asks


def test_a_skill_outside_the_catalog_becomes_the_refusal_recording():
    rules = _rules(unsupported_skill="decline_unsupported")
    catalog = {
        "skills": [
            {"name": "wave_hand", "planner_visible": True},
            {"name": "decline_unsupported", "planner_visible": True},
        ]
    }
    refused = rules.unsupported(catalog, _workflow("nod_yes"))
    assert [step.skill_name for step in refused.steps] == ["decline_unsupported"]
    wave = _workflow("wave_hand")
    assert rules.unsupported(catalog, wave) is wave
    assert rules.unsupported({"skills": []}, _workflow("nod_yes")).steps[0].skill_name == "nod_yes"


def test_only_answers_the_rules_may_replace_are_held_back_from_preview():
    rules = _rules(contact_replies=CONTACT)
    assert rules.may_override("你能抱抱我吗") and rules.may_override("把橙子直接给我")
    assert not rules.may_override("你叫什么名字")
    assert not AnswerRules(None).may_override("你能抱抱我吗")


def test_the_generic_planner_prompt_names_no_skill_of_another_robot():
    system, _ = build_planner_messages(request=_Request(), context={}, catalog={})
    assert "nod_yes" not in system["content"]
