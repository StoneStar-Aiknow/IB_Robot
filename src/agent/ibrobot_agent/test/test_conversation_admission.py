"""Conversation admission: questions, replies and glued tails reach the planner as answer-only requests.

The sentences are the ones the 2026-11-09 local rig session logged as wrongly rejected
(``semantic_gate.jsonl``), and the room chatter of the same session must stay out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ibrobot_agent import semantic_gate as gate_module
from ibrobot_agent.addressee import Voiceprint
from ibrobot_agent.semantic_gate import GRADE_COMMAND, GRADE_CONVERSATION, SemanticGate, normalize_text
from ibrobot_agent.speaker_split import WindowVoiceprints
from ibrobot_agent.voice_bridge import Log, PublishRequest, VoiceBridgeConfig, VoiceBridgeCore, config_from_parameters

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "aimdk_x2"}
MODEL = "campplus@test"
ALICE = Voiceprint((1.0, 0.0, 0.0), MODEL)
BOB = Voiceprint((0.0, 1.0, 0.0), MODEL)
LEXICON = ("挥手", "举手", "橙子", "水果", "照镜子", "抓")
TALK = ("讲一个故事", "讲个故事")


def _payload(weights, *, bias=-4.0, dim=4096):
    return {
        "schema_version": 1,
        "feature": {"ngrams": [1], "dim": dim, "hash": "fnv1a32", "pad": "<>"},
        "wake_words": ["灵犀灵犀", "灵犀"],
        "bias": bias,
        "weights": weights,
    }


def _weigh(text, weight, dim=4096):
    return [[gate_module._fnv1a32(char) % dim, weight] for char in normalize_text(text)]


def _gate(**kwargs) -> SemanticGate:
    """Only "挥" scores; every question here sits near 0, as on the field model."""
    return SemanticGate(
        _payload(_weigh("挥", 6.0)),
        threshold=0.10,
        lexicon=kwargs.pop("lexicon", LEXICON),
        dialogue_requests=kwargs.pop("dialogue_requests", TALK),
        **kwargs,
    )


def _judge(text, **flags):
    return _gate().judge(text, **{"session_open": True, "conversation": True, **flags})


# --- the gate on its own ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "你用的计算芯片是哪一个",
        "你用的系统是什么",
        "你的AGENT的聪明在哪",
        "你的模型都是在本地上跑的吗",
        "嗯你都能干啥",
        "你还会做什么动作",
        "你还能干别的吗",
        "你怎么不说话呢",
        "你是死机了吗",
        "你是出BUG了吗",
        "你跟openEuler是什么关系",
        "你是怎么抓水果的",
        "你怎么抓水果",
        "你有什么牛逼的技术",
        "你身上跑了多少个模型",
        "你为什么在这个展会上",
        "嗯你用的是什么芯片呢",
    ],
)
def test_a_question_put_to_the_robot_is_admitted_for_an_answer(text):
    decision = _judge(text)

    assert decision.accepted, decision.reason
    assert decision.grade == GRADE_CONVERSATION
    assert decision.reason == "question to the robot"


def test_the_question_rule_needs_conversation_admission_and_a_session():
    assert _gate().judge("你用的系统是什么", session_open=True).rejected
    assert _judge("你用的系统是什么", session_open=False).rejected


@pytest.mark.parametrize(
    "text",
    [
        # A third person: said about the robot, to somebody else.
        "你看见它手指刚才抖了吗",
        "他实际的表现是什么",
        # A remark with a catalog word and no question word keeps the remark veto.
        "你看那个橙子呢",
        "你那个照镜子的视频呢",
        # Naming someone in the room still wins.
        "周哥你用的什么系统",
    ],
)
def test_remarks_and_questions_to_others_stay_out(text):
    assert _judge(text).rejected


def test_a_skill_word_command_keeps_the_command_grade():
    decision = _judge("帮我挥挥手")

    assert decision.accepted
    assert decision.grade == GRADE_COMMAND


def test_leading_fillers_do_not_hide_a_skill_word():
    # "那" is a filler only before a sentence opener; "嗯那" strips to the command itself.
    decision = _judge("嗯那挥手")

    assert decision.reason == "skill word"
    assert decision.grade == GRADE_COMMAND


def test_a_stuttered_talk_request_still_matches():
    decision = _judge("给我讲一个故故事")

    assert decision.accepted
    assert decision.reason == "dialogue request"


@pytest.mark.parametrize(
    "text", ["是不是真的", "系统是什么", "全都在机器人本地完成吗", "我想选香蕉", "给我个香蕉", "你骗人"]
)
def test_a_reply_inside_the_follow_up_window_is_admitted(text):
    assert _judge(text).rejected, "outside the window there is nobody it answers"

    decision = _judge(text, followup_open=True)

    assert decision.accepted, decision.reason
    assert decision.grade == GRADE_CONVERSATION
    assert decision.reason == "follow-up to the robot"


@pytest.mark.parametrize(
    "text",
    [
        "其实这个功能可以再优化优化",
        "这个也是TVS的不是这也走TTS",
        "是是被拦住了",
        "但是看起来有点闷闷的样子",
        "啊我没听着啊",
        "他做TTS的我们等会",
        "但是我现在又跟他说话他也没有分析那个意图识别呀",
        # "那个什么" is a placeholder, not a question (local rig, 2026-11-09 22:58).
        "那么我是开了那个什么",
    ],
)
def test_room_chatter_stays_out_of_the_follow_up_window(text):
    assert _judge(text, followup_open=True).rejected


def test_the_trailing_question_of_a_glued_segment_is_admitted_alone():
    glued = "就都过了呀你闹的呀赢了哦都忽略了呀这些都不是啊这都不是跟他说的话你还会做什么动作"

    decision = _judge(glued)

    assert decision.accepted
    assert decision.grade == GRADE_CONVERSATION
    assert decision.admitted_text == "你还会做什么动作"
    assert _judge("这也忽略哦你还会干什么").admitted_text == "你还会干什么"


@pytest.mark.parametrize(
    "text",
    [
        # No question at the end: nothing to rescue.
        "如果在这里运行的不是很好的话那会场是一点机会没有",
        # The tail asks about a third person.
        "我们刚才说的那个你看他是不是在动",
        # Too short a prefix: this is just a question, judged as one.
        "你看你是谁",
    ],
)
def test_no_tail_is_rescued_from_plain_chatter(text):
    decision = _judge(text)

    assert not decision.admitted_text


def test_no_tail_is_rescued_without_a_session():
    assert _judge("这也忽略哦你还会干什么", session_open=False).rejected


def test_the_question_once_vetoed_as_a_remark_is_now_answered():
    # deca0ee5 pinned "你刚才拿的什么水果" as a remark. With conversation admission (the summit)
    # it is a question to the robot, admitted for an answer only: the worst outcome is the
    # robot saying which fruit it took. Without it the remark veto still applies.
    assert _gate().judge("你刚才拿的什么水果", session_open=True).rejected
    decision = _judge("你刚才拿的什么水果")
    assert decision.accepted
    assert decision.grade == GRADE_CONVERSATION


def test_the_log_records_grade_and_admitted_text(tmp_path):
    gate = _gate()
    path = tmp_path / "gate.jsonl"
    gate.record(_judge("这也忽略哦你还会干什么"), str(path), mode="enforce")
    gate.record(_judge("帮我挥挥手"), str(path), mode="enforce")

    first, second = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())

    assert first["grade"] == GRADE_CONVERSATION
    assert first["admitted_text"] == "你还会干什么"
    assert "grade" not in second


# --- through the voice bridge -------------------------------------------------------------

WAKE = {"wake_phrases": ("灵犀灵犀",), "wake_window_sec": 60.0}


def _core(tmp_path: Path, **overrides) -> VoiceBridgeCore:
    model = tmp_path / "model.json"
    model.write_text(json.dumps(_payload(_weigh("挥", 6.0))), encoding="utf-8")
    ids = iter(f"voice-{index}" for index in range(100))
    config = VoiceBridgeConfig(
        **{
            "confirm_mode": "auto",
            **SCOPE,
            **WAKE,
            "semantic_gate_mode": "enforce",
            "semantic_gate_model_path": str(model),
            "semantic_gate_lexicon": LEXICON,
            "semantic_gate_conversation": True,
            "semantic_gate_followup_sec": 20.0,
            **overrides,
        }
    )
    return VoiceBridgeCore(config, id_factory=lambda: next(ids))


def _requests(actions):
    return [action.payload for action in actions if isinstance(action, PublishRequest)]


def _answer(core, request_id, now):
    core.on_event(
        {
            "request_key": {"request_id": request_id, **SCOPE},
            "event_type": "conversation",
            "state": "ANSWERED",
            "user_message": "好的。",
        },
        now=now,
    )


def test_a_question_goes_out_marked_answer_only(tmp_path):
    core = _core(tmp_path)
    (opener,) = _requests(core.on_transcript("灵犀灵犀挥个手", now=1.0))
    _answer(core, opener["request_id"], now=2.0)

    (question,) = _requests(core.on_transcript("你用的系统是什么", now=40.0))

    assert "admission" not in opener
    assert question["admission"] == GRADE_CONVERSATION
    assert question["text"] == "你用的系统是什么"


def test_the_follow_up_window_opens_when_the_robot_finishes_and_closes_after_it(tmp_path):
    core = _core(tmp_path)
    (opener,) = _requests(core.on_transcript("灵犀灵犀你好", now=1.0))
    _answer(core, opener["request_id"], now=5.0)

    (reply,) = _requests(core.on_transcript("是不是真的", now=15.0))
    _answer(core, reply["request_id"], now=16.0)
    late = core.on_transcript("是不是真的", now=37.0)

    assert reply["admission"] == GRADE_CONVERSATION
    assert _requests(late) == []
    assert any("not addressed to me (semantic" in action.message for action in late if isinstance(action, Log))


def test_the_robot_speech_ending_opens_the_follow_up_window(tmp_path):
    core = _core(tmp_path)
    core.on_transcript("灵犀灵犀", now=1.0)
    core.note_speech("有柠檬、苹果和橙子。", 2.0)
    core.note_speech("有柠檬、苹果和橙子。", 6.0, finished=True)

    (choice,) = _requests(core.on_transcript("我想选香蕉", now=10.0))

    assert choice["admission"] == GRADE_CONVERSATION


def test_a_glued_segment_sends_only_its_trailing_question(tmp_path):
    core = _core(tmp_path)
    (opener,) = _requests(core.on_transcript("灵犀灵犀挥个手", now=1.0))
    _answer(core, opener["request_id"], now=2.0)

    actions = core.on_transcript("这都不是跟他说的话你还会做什么动作", now=40.0)

    (request,) = _requests(actions)
    assert request["text"] == "你还会做什么动作"
    assert request["admission"] == GRADE_CONVERSATION
    assert any("trailing question" in action.message for action in actions if isinstance(action, Log))


def test_shadow_mode_never_marks_or_shortens_a_request(tmp_path):
    core = _core(tmp_path, semantic_gate_mode="shadow")
    (opener,) = _requests(core.on_transcript("灵犀灵犀挥个手", now=1.0))
    _answer(core, opener["request_id"], now=2.0)

    (request,) = _requests(core.on_transcript("这都不是跟他说的话你还会做什么动作", now=40.0))

    assert "admission" not in request
    assert request["text"] == "这都不是跟他说的话你还会做什么动作"


def test_conversation_admission_is_off_by_default(tmp_path):
    core = _core(tmp_path, semantic_gate_conversation=False)
    (opener,) = _requests(core.on_transcript("灵犀灵犀挥个手", now=1.0))
    _answer(core, opener["request_id"], now=2.0)

    assert _requests(core.on_transcript("你用的系统是什么", now=10.0)) == []


SPEAKER = {"speaker_check_enabled": True, "speaker_min_similarity": 0.5, "speaker_max_age_sec": 6.0}


def _seg(sequence):
    return ("runtime-1", sequence)


def _speaker_core(tmp_path):
    core = _core(tmp_path, **SPEAKER)
    core.on_measurement("speaker", ALICE, now=1.0, segment=_seg(1))
    (opener,) = _requests(core.on_transcript("灵犀灵犀挥个手", now=1.0, segment=_seg(1)))
    _answer(core, opener["request_id"], now=2.0)
    return core


def _with_windows(voiceprint, *voices):
    """The segment's voiceprint plus one 2 s window per second, each ALICE ("a") or BOB ("b")."""
    vectors = tuple({"a": ALICE, "b": BOB}[voice].vector for voice in voices)
    windows = WindowVoiceprints(2.0, tuple(float(index) for index in range(len(voices))), vectors)
    return Voiceprint(voiceprint.vector, voiceprint.model_id, windows)


GLUED = "这都不是跟他说的话你还会做什么动作"


def test_a_trailing_question_in_another_voice_is_refused(tmp_path):
    """The segment's voiceprint is its first seconds (the woken user talking to a colleague);
    its last windows say the question at its end was the colleague's."""
    core = _speaker_core(tmp_path)
    core.on_measurement("speaker", _with_windows(ALICE, "a", "a", "a", "b", "b"), now=39.0, segment=_seg(2))

    actions = core.on_transcript(GLUED, now=40.0, segment=_seg(2))

    assert _requests(actions) == []
    assert any("speaker windows" in action.message for action in actions if isinstance(action, Log))


def test_a_trailing_question_in_the_owner_s_voice_goes_out(tmp_path):
    """Split off: the windows are only judged, and the owner's voice ends the segment."""
    core = _speaker_core(tmp_path)
    core.on_measurement("speaker", _with_windows(ALICE, "b", "b", "b", "a", "a"), now=39.0, segment=_seg(2))

    (request,) = _requests(core.on_transcript(GLUED, now=40.0, segment=_seg(2)))

    assert request["text"] == "你还会做什么动作"


def test_a_segment_with_no_owner_window_gives_up_no_question(tmp_path):
    core = _speaker_core(tmp_path)
    core.on_measurement("speaker", _with_windows(ALICE, "b", "b", "b", "b"), now=39.0, segment=_seg(2))

    assert _requests(core.on_transcript(GLUED, now=40.0, segment=_seg(2))) == []


def test_the_segment_voiceprint_is_still_judged_first(tmp_path):
    core = _speaker_core(tmp_path)
    core.on_measurement("speaker", _with_windows(BOB, "b", "b", "a", "a"), now=39.0, segment=_seg(2))

    # The whole segment's voiceprint is somebody else's and the split is off: the addressee
    # gate refuses it before any window is looked at.
    assert _requests(core.on_transcript(GLUED, now=40.0, segment=_seg(2))) == []


def test_a_trailing_question_without_windows_fails_open(tmp_path):
    core = _speaker_core(tmp_path)
    core.on_measurement("speaker", ALICE, now=39.0, segment=_seg(2))

    (request,) = _requests(core.on_transcript("这都不是跟他说的话你还会做什么动作", now=40.0, segment=_seg(2)))

    assert request["text"] == "你还会做什么动作"


# --- text corrections and parameters ------------------------------------------------------


def test_text_rewrites_fix_only_the_command_shape():
    config = VoiceBridgeConfig(**SCOPE, text_rewrites=(("^作为一个(.{1,6})的动作$", r"做一个\1的动作"),))

    assert config.correct("作为一个挥手的动作") == "做一个挥手的动作"
    assert config.correct("作为一个机器人你会做什么") == "作为一个机器人你会做什么"


def test_an_invalid_rewrite_is_refused():
    with pytest.raises(ValueError, match="text_rewrites"):
        VoiceBridgeConfig(**SCOPE, text_rewrites=(("(", "x"),))


def test_the_new_settings_come_through_ros_parameters():
    from ibrobot_agent.voice_bridge import node_parameter_defaults

    values = node_parameter_defaults()
    values.update(
        robot_scope="aimdk_x2",
        semantic_gate_conversation=True,
        semantic_gate_followup_sec=12.0,
        text_rewrites_json=json.dumps([["^作为一个(.{1,6})的动作$", "做一个\\1的动作"]], ensure_ascii=False),
    )

    config = config_from_parameters(values)

    assert config.semantic_gate_conversation is True
    assert config.semantic_gate_followup_sec == 12.0
    assert config.correct("作为一个飞吻的动作") == "做一个飞吻的动作"
