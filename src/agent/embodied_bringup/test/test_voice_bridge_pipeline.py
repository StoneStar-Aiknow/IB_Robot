"""Service-level voice scheduling contracts, owned by the integration package.

Audio is deliberately opaque bytes: these tests exercise ROS services and the
bridge's file adapter, not a WAV decoder or real audio hardware.
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from ibrobot_agent.voice_bridge import PublishControl, Speak, _build_node_class
from ibrobot_msgs.msg import SynthesizedAudio
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech

_TIMEOUT = 8.0


def _wait(predicate, description):
    deadline = time.monotonic() + _TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, description
        threading.Event().wait(0.01)


class _Services:
    def __init__(self, context):
        # Distinct default callback groups let playback block without blocking synthesis.
        self.synthesis = Node("pipeline_synthesis", context=context)
        self.playback = Node("pipeline_playback", context=context)
        self.closing = threading.Event()
        self.synth_entered = {text: threading.Event() for text in ("A", "B", "C", "D")}
        self.play_entered = {text: threading.Event() for text in ("A", "B", "C", "D")}
        self.synth_gates = {}
        self.play_gates = {}
        self.segments = {}
        self.fail_synthesis = set()
        self.fail_playback = set()
        self.played = []
        self.paths = []
        self.errors = []
        self.controls = []
        self.asr_controls = []
        self.requests = []
        self.synthesis.create_service(SynthesizeSpeech, "/pipeline/synthesize", self._synthesize)
        self.playback.create_service(PlayAudioFile, "/pipeline/play", self._play)
        self.synthesis.create_subscription(
            String, "/pipeline/control", lambda msg: self.controls.append(json.loads(msg.data)), 10
        )
        self.synthesis.create_subscription(
            String, "/pipeline/asr_control", lambda msg: self.asr_controls.append(msg.data), 10
        )
        self.synthesis.create_subscription(
            String, "/pipeline/request", lambda msg: self.requests.append(json.loads(msg.data)), 10
        )

    def _gate(self, gates, text):
        gate = gates.get(text)
        if gate is None:
            return True
        deadline = time.monotonic() + _TIMEOUT
        while not gate.wait(0.01):
            if self.closing.is_set():
                return False
            if time.monotonic() >= deadline:
                self.errors.append(f"callback timed out: {text}")
                return False
        return not self.closing.is_set()

    def _synthesize(self, request, response):
        # The real Chinese frontend strips Latin letters; use Chinese utterances
        # on the wire while keeping short ASCII labels for opaque audio bytes.
        text = request.text.translate(str.maketrans("甲乙丙丁", "ABCD"))
        self.synth_entered[text].set()
        response.success = self._gate(self.synth_gates, text) and text not in self.fail_synthesis
        if response.success:
            response.audio_segments = [
                SynthesizedAudio(index=index, audio_data=list(marker), audio_format="wav")
                for index, marker in enumerate(self.segments.get(text, [text.encode()]))
            ]
        return response

    def _play(self, request, response):
        try:
            marker = Path(request.file_path).read_bytes()
            text = marker.decode()[0]
            self.paths.append(request.file_path)
            self.played.append(marker)
            self.play_entered[text].set()
            response.success = self._gate(self.play_gates, text) and text not in self.fail_playback
        except Exception as error:
            self.errors.append(repr(error))
            response.success = False
        return response


@pytest.fixture
def rig():
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "root conftest must isolate the ROS graph"
    assert os.environ["ROS_LOCALHOST_ONLY"] == "1"
    context = Context()
    rclpy.init(context=context, domain_id=int(os.environ["IBROBOT_TEST_ROS_DOMAIN_ID"]))
    services = _Services(context)
    node = _build_node_class()(
        context=context,
        parameter_overrides=[
            Parameter(name, value=value)
            for name, value in {
                "tts_enabled": True,
                "tts_synthesis_service": "/pipeline/synthesize",
                "tts_playback_service": "/pipeline/play",
                "tts_timeout_sec": _TIMEOUT,
                "control_topic": "/pipeline/control",
                "request_topic": "/pipeline/request",
                "asr_control_topic": "/pipeline/asr_control",
                "resume_settle_sec": 0.0,
                "robot_scope": "pipeline_test",
            }.items()
        ],
    )
    executor = MultiThreadedExecutor(num_threads=6, context=context)
    for participant in (node, services.synthesis, services.playback):
        executor.add_node(participant)
    spin_errors = []

    def spin():
        try:
            executor.spin()
        except Exception as error:
            spin_errors.append(error)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        assert node._speaker._synthesize.wait_for_service(timeout_sec=_TIMEOUT)
        assert node._speaker._playback.wait_for_service(timeout_sec=_TIMEOUT)
        for publisher in (node._control_publisher, node._request_publisher, node._asr_control_publisher):
            _wait(lambda publisher=publisher: publisher.get_subscription_count() > 0, "observer discovery")
        yield node, services
    finally:
        services.closing.set()
        try:
            node.destroy_node()
        finally:
            stopped = executor.shutdown(timeout_sec=_TIMEOUT)
            thread.join(timeout=_TIMEOUT)
            try:
                assert stopped and not thread.is_alive(), "executor did not stop"
                assert not spin_errors, spin_errors
                assert not services.errors, services.errors
            finally:
                services.synthesis.destroy_node()
                services.playback.destroy_node()
                context.try_shutdown()


def _say(node, text, *, then=False):
    utterance = text.translate(str.maketrans("ABCD", "甲乙丙丁"))
    node._execute([Speak(utterance, then=(PublishControl({"heard": text}),) if then else ())])


def test_latest_ready_replaces_only_waiting_audio(rig):
    node, services = rig
    services.play_gates["A"] = threading.Event()
    # Observe replacement after the real C service response has been admitted as
    # ready, not merely after its service callback returned (a scheduling race).
    replaced = threading.Event()
    original_log = node._speech_pipeline._log

    def observe(message):
        original_log(message)
        if "drop reason=replaced" in message:
            replaced.set()

    node._speech_pipeline._log = observe
    _say(node, "A")
    assert services.play_entered["A"].wait(_TIMEOUT)
    _say(node, "B")
    assert services.synth_entered["B"].wait(_TIMEOUT)
    _say(node, "C")
    assert replaced.wait(_TIMEOUT), "C never became ready and replaced B"
    services.play_gates["A"].set()
    assert services.play_entered["C"].wait(_TIMEOUT)
    # A protected completion fence proves no discarded B remains for later play.
    _say(node, "D", then=True)
    _wait(lambda: services.controls == [{"heard": "D"}], "completion fence")
    assert services.played == [b"A", b"C", b"D"]


def test_ready_audio_plays_without_waiting_for_newer_synthesis(rig):
    node, services = rig
    services.play_gates["A"] = threading.Event()
    services.synth_gates["C"] = threading.Event()
    _say(node, "A")
    assert services.play_entered["A"].wait(_TIMEOUT)
    _say(node, "B")
    assert services.synth_entered["B"].wait(_TIMEOUT)
    _say(node, "C")
    # A single synthesis worker entering C proves B was admitted ready.
    assert services.synth_entered["C"].wait(_TIMEOUT)
    services.play_gates["A"].set()
    assert services.play_entered["B"].wait(_TIMEOUT), "pending C blocked ready B"
    assert not services.play_entered["C"].is_set()
    services.synth_gates["C"].set()
    _say(node, "D", then=True)
    _wait(lambda: services.controls == [{"heard": "D"}], "completion fence")
    assert services.played == [b"A", b"B", b"C", b"D"]


def test_protected_then_waits_for_all_segments_and_keeps_order(rig):
    node, services = rig
    services.segments["B"] = [b"B-first", b"B-second"]
    services.play_gates["A"] = threading.Event()
    services.play_gates["B"] = threading.Event()
    _say(node, "A")
    assert services.play_entered["A"].wait(_TIMEOUT)
    _say(node, "B", then=True)
    _say(node, "C", then=True)
    assert services.synth_entered["C"].wait(_TIMEOUT)
    services.play_gates["A"].set()
    assert services.play_entered["B"].wait(_TIMEOUT)
    assert services.controls == [], "acknowledgement preceded successful playback"
    services.play_gates["B"].set()
    _wait(lambda: services.controls == [{"heard": "B"}, {"heard": "C"}], "ordered acknowledgements")
    assert services.played == [b"A", b"B-first", b"B-second", b"C"]
    _wait(lambda: all(not Path(path).exists() for path in services.paths), "temporary audio cleanup")


@pytest.mark.parametrize("interruptible, stop_reply", [(False, False), (True, False), (True, True)])
def test_asr_only_mutes_for_playback_under_existing_stop_rules(rig, interruptible, stop_reply):
    node, services = rig
    node._on_asr_status(String(data="listening"))
    if interruptible:
        node._on_transcript(String(data="请挥手"))
        _wait(lambda: len(services.requests) == 1, "active request")
    services.synth_gates["A"] = threading.Event()
    services.play_gates["A"] = threading.Event()
    # Keep the service marker stable while exercising the stop-keyword policy.
    action_text = "停止A" if stop_reply else "A"
    if stop_reply:
        services.synth_entered[action_text] = services.synth_entered["A"]
        services.synth_gates[action_text] = services.synth_gates["A"]
        services.segments[action_text] = [b"A"]
    _say(node, action_text, then=True)
    assert services.synth_entered["A"].wait(_TIMEOUT)
    assert services.asr_controls == [], "ASR muted during synthesis"
    services.synth_gates["A"].set()
    assert services.play_entered["A"].wait(_TIMEOUT)
    muted = not interruptible or stop_reply
    if muted:
        _wait(lambda: services.asr_controls == ["stop"], "playback mute")
    else:
        assert services.asr_controls == [], "stop must remain audible during motion"
    services.play_gates["A"].set()
    _wait(lambda: services.controls == [{"heard": action_text}], "speech completion")
    if muted:
        _wait(lambda: services.asr_controls == ["stop", "start"], "ASR resume")
    else:
        assert services.asr_controls == []


@pytest.mark.parametrize("phase", ["synthesis", "playback"])
def test_close_unblocks_service_waits_and_suppresses_then(rig, phase):
    node, services = rig
    gates = services.synth_gates if phase == "synthesis" else services.play_gates
    entered = services.synth_entered if phase == "synthesis" else services.play_entered
    gates["A"] = threading.Event()
    _say(node, "A", then=True)
    assert entered["A"].wait(_TIMEOUT)
    pipeline = node._speech_pipeline
    pipeline.close(stop=node._speaker.close, timeout=2.0)
    # Rejection and idempotent close are public lifecycle contracts.
    assert not pipeline.submit(Speak("B", then=(PublishControl({"heard": "B"}),)))
    pipeline.close(stop=node._speaker.close, timeout=2.0)
    gates["A"].set()
    assert services.controls == []
    assert not services.synth_entered["B"].is_set()
    assert not services.play_entered["B"].is_set()
    assert services.played == ([] if phase == "synthesis" else [b"A"])
