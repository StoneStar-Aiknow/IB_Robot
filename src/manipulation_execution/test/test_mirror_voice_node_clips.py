"""The voice node in prerecorded mode (``clips_json``): board only, it needs ibrobot_msgs.

No synthesis client at all, the ASR control publisher and the ASR status
subscription, the 「用 <dir>，找到 n / N」 count, the ``noclip`` refusal, no look
line once the wrap-up (and with it the goodbye) has begun, and a
status published the moment the voice starts or stops speaking. And the last
``start`` of a real process, sent after SIGINT or SIGTERM.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import wave

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs.msg")

from test_mirror_voice_node import utterance  # noqa: E402

from embodied_common.mirror_session_contract import STATE_MIRRORING  # noqa: E402
from embodied_common.mirror_voice_contract import parse_voice_section, to_json  # noqa: E402
from embodied_common.visual_interaction_contract import UTTERANCE_TOPIC  # noqa: E402

CLIPS = {"evt_ready_01": "我准备好咯！", "look_wave_01": "收到你的招呼啦！", "evt_goodbye_01": "下次再见吧！"}


def clips_json(folder, asr_mode="mute"):
    return json.dumps(
        {
            "version": 1,
            "clip_dirs": [str(folder / "absent"), str(folder)],
            "wav_suffix": "_v1.wav",
            "clips": CLIPS,
            "events": {"starting": [], "ready": ["evt_ready_01"], "mirroring": [], "goodbye": ["evt_goodbye_01"]},
            "standby": {"ids": [], "first_after_sec": [1.0, 3.0], "gap_sec": [2.0, 5.0]},
            "asr": {"mode": asr_mode, "resume_delay_sec": 0.8, "topic": "/voice_control"},
        },
        ensure_ascii=False,
    )


@pytest.fixture
def node(tmp_path):
    from rclpy.parameter import Parameter

    from manipulation_execution.mirror_voice_node import MirrorVoiceNode

    for clip_id in ("evt_ready_01", "look_wave_01"):  # evt_goodbye_01 is missing
        with wave.open(str(tmp_path / f"{clip_id}_v1.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(b"\x00\x00" * 240)
    rclpy.init()
    try:
        voice = to_json(parse_voice_section({"enabled": True, "utterance_topic": UTTERANCE_TOPIC}))
        made = MirrorVoiceNode(
            parameter_overrides=[
                Parameter("voice_json", value=voice),
                Parameter("clips_json", value=clips_json(tmp_path)),
            ]
        )
        assert made.speaker is not None
        yield made
        made.destroy_node()
    finally:
        rclpy.shutdown()


def test_no_synthesis_client_and_the_asr_topics(node):
    services = {client.srv_name for client in node.clients}
    assert services == {"/voice_tts/play"}
    assert "/voice_control" in {publisher.topic_name for publisher in node.publishers}
    assert "/voice_status" in {sub.topic_name for sub in node.subscriptions}


def test_the_clips_found_are_counted(node, tmp_path):
    report = node._clip_report
    assert (report.directory, report.found, report.total) == (str(tmp_path), 2, 3)


def mirroring(phase):
    from std_msgs.msg import String

    return String(data=json.dumps({"state": STATE_MIRRORING, "phase": phase, "last_transition": None}))


def test_an_utterance_without_a_clip_is_refused_as_noclip(node):
    node._on_state(mirroring("running"))
    node._on_utterance(utterance("这句没有录过"))

    assert node._utterance_refused["noclip"] == 1
    assert node.speaker.synthesis_calls == 0


def test_a_look_line_is_refused_once_the_wrap_up_began(node):
    node._on_state(mirroring("running"))
    assert node._mirroring_now()
    node._on_state(mirroring("wrap_up"))
    node._on_utterance(utterance(CLIPS["look_wave_01"]))

    assert not node._mirroring_now()
    assert node._utterance_refused["mirror"] == 1


def test_speaking_publishes_the_status_at_once(node):
    sent = []
    node._status_publisher.publish = lambda message: sent.append(json.loads(message.data))
    node._on_speaking(CLIPS["look_wave_01"])  # the speaker's callback, at a clip's start

    assert len(sent) == 1, "a status went out at once, not at the next period"
    assert sent[-1]["clips"] == {"dir": node._clip_report.directory, "found": 2, "total": 3}
    assert sent[-1]["synthesis_calls"] == 0


class TestExit:
    """``main``'s signal handling: the context must outlive the signal in prerecorded mode."""

    @pytest.mark.parametrize("muted,taken_over", [(True, True), (False, False)], ids=["clips", "baseline"])
    def test_only_prerecorded_mode_replaces_rclpys_handlers(self, monkeypatch, muted, taken_over):
        from manipulation_execution import mirror_voice_node as module

        class FakeNode:
            speaker = None
            files = None
            mute = type("Mute", (), {"shutdown": lambda self: None})() if muted else None

            def destroy_node(self):
                pass

        class FakeExecutor:
            def remove_node(self, node):
                pass

            def shutdown(self):
                pass

        calls = []
        monkeypatch.setattr(module.rclpy, "init", lambda args=None: None)
        monkeypatch.setattr(module.rclpy, "try_shutdown", lambda: None)
        monkeypatch.setattr(module, "MirrorVoiceNode", FakeNode)
        monkeypatch.setattr(module, "SingleThreadedExecutor", FakeExecutor)
        monkeypatch.setattr(module, "_keep_context_past_signals", lambda: calls.append(True))

        module.main()

        assert bool(calls) is taken_over

    @pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
    def test_a_signalled_process_still_starts_the_asr(self, tmp_path, monkeypatch, signum):
        from std_msgs.msg import String

        domain = 100 + os.getpid() % 100
        monkeypatch.setenv("ROS_DOMAIN_ID", str(domain))
        monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
        voice = to_json(parse_voice_section({"enabled": True}))
        params = tmp_path / "params.json"
        # ``off`` sends stop at activation, so shutdown owes a start with no clip played.
        params.write_text(
            json.dumps({"/**": {"ros__parameters": {"voice_json": voice, "clips_json": clips_json(tmp_path, "off")}}})
        )
        rclpy.init()
        process = None
        try:
            listener = rclpy.create_node("asr_control_listener")
            received = []
            listener.create_subscription(String, "/voice_control", lambda m: received.append(m.data), 10)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from manipulation_execution.mirror_voice_node import main; main()",
                    "--ros-args",
                    "--params-file",
                    str(params),
                ]
            )
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and listener.count_publishers("/voice_control") == 0:
                rclpy.spin_once(listener, timeout_sec=0.1)
            assert listener.count_publishers("/voice_control") == 1, "the node never came up"
            time.sleep(1.0)  # discovery from both sides

            process.send_signal(signum)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and "start" not in received:
                rclpy.spin_once(listener, timeout_sec=0.1)

            assert "start" in received
            assert process.wait(timeout=10.0) == 0
            listener.destroy_node()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            rclpy.shutdown()
