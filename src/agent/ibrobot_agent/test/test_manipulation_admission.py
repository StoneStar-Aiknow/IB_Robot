"""Manipulation admission through the semantic gate, verified voice core and ROS transport."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from ibrobot_agent.addressee import Voiceprint
from ibrobot_agent.semantic_gate import GRADE_COMMAND, GRADE_CONVERSATION, SemanticGate, correct_manipulation_transcript
from ibrobot_agent.voice_bridge import PublishRequest, VoiceBridgeConfig, VoiceBridgeCore

SCOPE = {"channel_id": "mock_voice", "principal_id": "mock_operator", "robot_scope": "mock_robot"}
OWNER = Voiceprint((1.0, 0.0, 0.0), "mock-speaker")
OTHER = Voiceprint((0.0, 1.0, 0.0), "mock-speaker")
COMMANDS = (
    "把苹果抓进黑色的碗里",
    "把橘子抓进黑色的碗里",
    "把柠檬放进白色的盒子里",
    "将绿色积木放入透明的容器内",
    "请帮我把红色积木放到蓝色的箱子里面吧",
    "麻烦你把乒乓球抓到左边的碗里",
    "抓取苹果放进黑色的碗里",
    "帮我抓取橙子然后放进白色的碗里",
    "拿起黄色积木再放到透明的盒子里面",
    "嗯，把苹果抓进黑色的碗里。",
    "把知了玩具放进白色的盒子里",
    "把金刚玩具放进黑色的碗里",
    "把以前买的苹果放进黑色的碗里",
    "把已经准备好的苹果放进黑色的碗里",
)
COMPLETED_TRANSFERS = (
    "把苹果放进了黑色的碗里",
    "把苹果刚刚放进黑色的碗里",
    "把苹果已经放进黑色的碗里",
    "把苹果刚放进黑色的碗里",
    "把苹果曾经放进黑色的碗里",
    "把苹果放进过黑色的碗里",
    "把苹果放进黑色的碗里了",
    "已经把苹果放进黑色的碗里",
    "我已经把苹果放进黑色的碗里",
    "刚刚把苹果放进黑色的碗里",
    "抓取苹果然后放进了黑色的碗里",
    "拿起了苹果再放进黑色的碗里",
    "抓取过苹果然后放进黑色的碗里",
    "把苹果抓紧了黑色的碗里",
)
NON_COMMANDS = (
    "不要把苹果抓进黑色的碗里",
    "别把苹果抓进黑色的碗里",
    "把苹果不要抓进黑色的碗里",
    "把苹果抓进黑色的碗里吗",
    "把苹果抓进黑色的碗里？",
    "你能把苹果抓进黑色的碗里吗",
    "刚才他说把苹果抓进黑色的碗里",
    "如果把苹果抓进黑色的碗里会怎样",
    "假如把苹果抓进黑色的碗里",
    "把苹果抓进黑色的碗里这句话没进意图",
    "把苹果照进黑色的碗里这句话没进没进意图里面",
    "把苹果说成橘子再抓进黑色的碗里",
    "把苹果抓进他刚才说的黑碗里",
    "把苹果抓进黑色的碗里然后挥手",
    "把苹果抓紧不要松手",
    "把苹果放进心里",
    "把苹果抓进里",
    "把苹果抓进黑色的碗里把橘子也放进白碗里",
    "“把苹果抓进黑色的碗里”",
    "他说“把苹果抓进黑色的碗里”",
)


@pytest.fixture
def quiet_model(tmp_path):
    # Deliberately reject every sentence on score alone: an unrelated toy skill or
    # today's downloaded model must not accidentally make these grammar tests pass.
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature": {"ngrams": [1], "dim": 4096, "hash": "fnv1a32", "pad": "<>", "vector_norm": "l2"},
                "wake_words": ["欧拉欧拉"],
                "bias": -10.0,
                "weights": [],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("text", COMMANDS)
def test_complete_manipulation_command_is_admitted_inside_a_session(quiet_model, text):
    gate = SemanticGate.load(str(quiet_model), threshold=0.1, lexicon=("抓取物体", "放进容器"))
    assert gate.score(text) < gate.threshold
    assert gate.judge(text).rejected
    decision = gate.judge(text, session_open=True)
    assert decision.accepted and decision.grade == GRADE_COMMAND
    assert decision.reason == "manipulation command"


@pytest.mark.parametrize("text", NON_COMMANDS)
def test_a_negation_quotation_question_or_discussion_gets_no_manipulation_exception(quiet_model, text):
    gate = SemanticGate.load(str(quiet_model), threshold=0.1, lexicon=("抓取物体", "放进容器"))
    assert gate.judge(text, session_open=True).rejected


def test_a_question_about_manipulation_can_only_get_conversation_admission(quiet_model):
    gate = SemanticGate.load(str(quiet_model), threshold=0.1)
    decision = gate.judge("你能把苹果抓进黑色的碗里吗", session_open=True, conversation=True)
    assert decision.accepted and decision.grade == GRADE_CONVERSATION


@pytest.mark.parametrize(
    "text",
    [
        "不要把苹果抓进黑色的碗里",
        "请不要把苹果抓进黑色的碗里",
        "把苹果不要抓进黑色的碗里",
        "把苹果不放进黑色的碗里",
        "刚才他说把苹果抓进黑色的碗里",
        "如果把苹果抓进黑色的碗里会怎样",
        "假如把苹果抓进黑色的碗里",
        "把苹果抓进黑色的碗里这句话没进意图",
        "“把苹果抓进黑色的碗里”",
        "把苹果抓进黑色的碗里？",
    ],
)
@pytest.mark.parametrize("wake", ["", "欧拉欧拉"])
def test_manipulation_non_commands_are_vetoed_even_when_the_model_is_overconfident(quiet_model, text, wake):
    payload = json.loads(quiet_model.read_text())
    payload["bias"] = 10.0
    gate = SemanticGate(payload, threshold=0.1, wake_prefixes=("欧拉欧拉",))
    text = wake + text
    assert gate.score(text) > 0.99
    decision = gate.judge(text, session_open=True)
    assert decision.rejected and "not an instruction" in decision.reason


@pytest.mark.parametrize("text", COMPLETED_TRANSFERS)
@pytest.mark.parametrize("wake", ["", "欧拉欧拉"])
def test_completed_transfers_are_not_new_commands_even_with_a_high_model_score(quiet_model, text, wake):
    payload = json.loads(quiet_model.read_text())
    payload["bias"] = 10.0
    gate = SemanticGate(payload, threshold=0.1, wake_prefixes=("欧拉欧拉",))
    decision = gate.judge(wake + text, session_open=True)
    assert gate.score(text) > 0.99
    assert decision.rejected and "not an instruction" in decision.reason


@pytest.mark.parametrize("text", COMPLETED_TRANSFERS)
def test_completed_transfer_reports_are_not_rewritten(text):
    assert correct_manipulation_transcript(text) == text


@pytest.mark.parametrize("wake", ["", "欧拉欧拉，"])
def test_grip_homophone_is_corrected_only_in_a_complete_transfer_command(wake):
    text = wake + "请把苹果抓紧黑色的碗里。"
    assert correct_manipulation_transcript(text, wake_phrases=("欧拉欧拉",)) == wake + "请把苹果抓进黑色的碗里。"


@pytest.mark.parametrize(
    "text",
    [
        "抓紧苹果不要松手",
        "抓紧时间准备抓取",
        "不要把苹果抓紧黑色的碗里",
        "如果把苹果抓紧黑色的碗里",
        "把苹果抓紧黑色的碗里吗",
        "“把苹果抓紧黑色的碗里”",
    ],
)
def test_legitimate_grip_requests_and_non_commands_are_not_rewritten(text):
    assert correct_manipulation_transcript(text) == text


def _bridge(quiet_model):
    core = VoiceBridgeCore(
        VoiceBridgeConfig(
            **SCOPE,
            confirm_mode="auto",
            wake_phrases=("欧拉欧拉",),
            wake_window_sec=12.0,
            speaker_check_enabled=True,
            speaker_min_similarity=0.6,
            require_verified_speech=True,
            semantic_gate_mode="enforce",
            semantic_gate_model_path=str(quiet_model),
            semantic_gate_fail_open=False,
        )
    )
    core.on_measurement("speaker", OWNER, now=1.0, segment=("mock", 1))
    assert not _requests(core.on_heard("欧拉欧拉", now=1.0, segment=("mock", 1)))
    return core


def _requests(actions):
    return [a.payload for a in actions if isinstance(a, PublishRequest)]


@pytest.mark.parametrize("text", [COMMANDS[0], "把苹果抓紧黑色的碗里"])
def test_verified_follow_up_reaches_the_agent_as_a_corrected_command(quiet_model, text):
    core = _bridge(quiet_model)
    core.on_measurement("speaker", OWNER, now=2.0, segment=("mock", 2))
    (request,) = _requests(core.on_heard(text, now=2.0, segment=("mock", 2)))
    assert request["text"] == COMMANDS[0]
    assert request.get("admission", GRADE_COMMAND) == GRADE_COMMAND


@pytest.mark.parametrize("voice", [OTHER, None, Voiceprint(OWNER.vector, "another-model")])
def test_manipulation_command_still_requires_a_matching_voiceprint(quiet_model, voice):
    core = _bridge(quiet_model)
    core.on_measurement("speaker", voice, now=2.0, segment=("mock", 2))
    assert not _requests(core.on_heard(COMMANDS[0], now=2.0, segment=("mock", 2)))


def test_manipulation_command_still_requires_a_voiceprint_before_the_deadline(quiet_model):
    core = _bridge(quiet_model)
    assert not _requests(core.on_heard(COMMANDS[0], now=2.0, segment=("mock", 2)))
    assert not _requests(core.tick(now=3.0))


def test_manipulation_command_still_requires_an_unexpired_wake_window(quiet_model):
    core = _bridge(quiet_model)
    core.on_measurement("speaker", OWNER, now=20.0, segment=("mock", 2))
    assert not _requests(core.on_heard(COMMANDS[0], now=20.0, segment=("mock", 2)))


@pytest.mark.parametrize("text", NON_COMMANDS)
def test_verified_speaker_does_not_turn_discussion_into_an_agent_command(quiet_model, text):
    core = _bridge(quiet_model)
    core.on_measurement("speaker", OWNER, now=2.0, segment=("mock", 2))
    assert not _requests(core.on_heard(text, now=2.0, segment=("mock", 2)))


@pytest.mark.parametrize("text", COMPLETED_TRANSFERS)
def test_verified_speaker_does_not_repeat_a_reported_completed_transfer(quiet_model, text):
    core = _bridge(quiet_model)
    core.on_measurement("speaker", OWNER, now=2.0, segment=("mock", 2))
    assert not _requests(core.on_heard(text, now=2.0, segment=("mock", 2)))


def test_mock_ros_transcripts_and_voiceprints_publish_only_the_owners_command(quiet_model):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import String

    from ibrobot_agent.voice_bridge import _build_node_class
    from ibrobot_msgs.msg import SpeakerEmbedding, SpeechTranscript

    assert os.environ["ROS_LOCALHOST_ONLY"] == "1"
    domain = int(os.environ["IBROBOT_TEST_ROS_DOMAIN_ID"])
    context = Context()
    rclpy.init(context=context, domain_id=domain)
    prefix = "/mock_grasp_admission"
    params = {
        **SCOPE,
        "wake_phrases_json": json.dumps(["欧拉欧拉"]),
        "wake_window_sec": 60.0,
        "speaker_check_enabled": True,
        "require_verified_speech": True,
        "semantic_gate_mode": "enforce",
        "semantic_gate_model_path": str(quiet_model),
        "semantic_gate_fail_open": False,
        "confirm_mode": "auto",
        "transcript_topic": prefix + "/transcript",
        "speaker_embedding_topic": prefix + "/speaker",
        "request_topic": prefix + "/request",
        "event_topic": prefix + "/event",
        "tts_enabled": False,
    }
    bridge = _build_node_class()(
        context=context, parameter_overrides=[Parameter(k, value=v) for k, v in params.items()]
    )
    source = Node("mock_grasp_voice_source", context=context)
    transcripts = source.create_publisher(SpeechTranscript, params["transcript_topic"], 10)
    speakers = source.create_publisher(SpeakerEmbedding, params["speaker_embedding_topic"], 10)
    events = source.create_publisher(String, params["event_topic"], 10)
    requests = []
    source.create_subscription(String, params["request_topic"], lambda m: requests.append(json.loads(m.data)), 10)
    executor = MultiThreadedExecutor(num_threads=3, context=context)
    executor.add_node(bridge)
    executor.add_node(source)
    errors = []

    def spin():
        try:
            executor.spin()
        except Exception as error:
            errors.append(error)

    def wait_for(predicate):
        until = time.monotonic() + 8
        while not predicate():
            assert time.monotonic() < until, "mock ROS transport timed out"
            threading.Event().wait(0.01)

    def say(text, seq, voice=OWNER):
        speakers.publish(
            SpeakerEmbedding(
                source_id="mock", segment_sequence=seq, model_id=voice.model_id, embedding=list(voice.vector)
            )
        )
        transcripts.publish(SpeechTranscript(source_id="mock", segment_sequence=seq, text=text))

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        wait_for(lambda: all(p.get_subscription_count() for p in (transcripts, speakers, events)))
        say("欧拉欧拉介绍一下自己", 1)
        wait_for(lambda: len(requests) == 1)
        first = requests[0]
        events.publish(
            String(
                data=json.dumps(
                    {
                        "request_key": {**SCOPE, "request_id": first["request_id"]},
                        "event_type": "terminal",
                        "state": "SUCCEEDED",
                    }
                )
            )
        )
        wait_for(lambda: not bridge._core.active_request_id)
        say("把苹果抓进黑色的碗里", 2, OTHER)
        say("刚才他说把苹果抓进黑色的碗里", 3)
        say("不要把苹果抓进黑色的碗里", 4)
        say("把苹果放进了黑色的碗里", 5)
        say("把苹果刚刚放进黑色的碗里", 6)
        say("把苹果已经放进黑色的碗里", 7)
        say("把香蕉抓紧白色的碗里", 8)
        wait_for(lambda: len(requests) >= 2)
        assert [r["text"] for r in requests] == ["介绍一下自己", "把香蕉抓进白色的碗里"]
        assert requests[-1].get("admission", GRADE_COMMAND) == GRADE_COMMAND
    finally:
        bridge.destroy_node()
        stopped = executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        source.destroy_node()
        context.try_shutdown()
        assert stopped and not thread.is_alive() and not errors
