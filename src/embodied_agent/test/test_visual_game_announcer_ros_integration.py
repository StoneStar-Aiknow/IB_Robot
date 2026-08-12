"""ROS integration test for terminal visual-game announcements."""

import io
import time
import uuid
import wave
from array import array
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from embodied_agent.visual_game_announcer_node import VisualGameAnnouncerNode
from embodied_agent.visual_game_qos import visual_game_event_qos
from ibrobot_msgs.msg import SynthesizedAudio, VisualGameEvent
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech


def _audio_segment():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16)
    return SynthesizedAudio(
        index=0,
        text="announcement",
        audio_data=array("B", output.getvalue()),
        audio_format="wav_pcm_s16le",
        sample_rate=16000,
        channels=1,
        duration_sec=0.001,
    )


def test_terminal_game_event_reaches_tts_service():
    rclpy.init()
    announcer = None
    peer = None
    executor = None
    try:
        suffix = f"id_{uuid.uuid4().hex}"
        event_topic = f"/visual_game_announcer_test/{suffix}/events"
        tts_service = f"/visual_game_announcer_test/{suffix}/synthesize"
        playback_service = f"/visual_game_announcer_test/{suffix}/play"
        announcer = VisualGameAnnouncerNode(
            parameter_overrides=[
                Parameter("event_topic", Parameter.Type.STRING, event_topic),
                Parameter("tts_service", Parameter.Type.STRING, tts_service),
                Parameter("playback_service", Parameter.Type.STRING, playback_service),
            ]
        )
        peer = rclpy.create_node(f"visual_game_announcer_peer_{suffix}")
        observed_text = []
        observed_playback = []

        def synthesize(request, response):
            observed_text.append(request.text)
            response.success = True
            response.audio_segments = [_audio_segment()]
            return response

        def play(request, response):
            path = Path(request.file_path)
            observed_playback.append((path, path.read_bytes()))
            response.success = True
            return response

        peer.create_service(SynthesizeSpeech, tts_service, synthesize)
        peer.create_service(PlayAudioFile, playback_service, play)
        publisher = peer.create_publisher(VisualGameEvent, event_topic, visual_game_event_qos())
        executor = SingleThreadedExecutor()
        executor.add_node(announcer)
        executor.add_node(peer)

        deadline = time.monotonic() + 2.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert publisher.get_subscription_count() > 0

        event = VisualGameEvent()
        event.request_id = f"game-{suffix}"
        event.execution_id = f"execution-{suffix}"
        event.game_name = "sorting_hat"
        event.handler = "sorting_hat_v1"
        event.announce = True
        event.state = "succeeded"
        event.success = True
        event.scene_summary = "赫奇帕奇"
        publisher.publish(event)

        deadline = time.monotonic() + 2.0
        while (not observed_playback or observed_playback[0][0].exists()) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)

        assert observed_text == ["赫奇帕奇"]
        assert observed_playback[0][1].startswith(b"RIFF")
        assert not observed_playback[0][0].exists()
    finally:
        if executor is not None:
            executor.shutdown()
        if announcer is not None:
            announcer.close()
            announcer.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()


def test_no_person_terminal_event_reaches_tts_service_with_guidance():
    rclpy.init()
    announcer = None
    peer = None
    executor = None
    try:
        suffix = f"id_{uuid.uuid4().hex}"
        event_topic = f"/visual_game_announcer_test/{suffix}/events"
        tts_service = f"/visual_game_announcer_test/{suffix}/synthesize"
        playback_service = f"/visual_game_announcer_test/{suffix}/play"
        announcer = VisualGameAnnouncerNode(
            parameter_overrides=[
                Parameter("event_topic", Parameter.Type.STRING, event_topic),
                Parameter("tts_service", Parameter.Type.STRING, tts_service),
                Parameter("playback_service", Parameter.Type.STRING, playback_service),
            ]
        )
        peer = rclpy.create_node(f"visual_game_announcer_peer_{suffix}")
        observed_text = []
        observed_playback = []

        def synthesize(request, response):
            observed_text.append(request.text)
            response.success = True
            response.audio_segments = [_audio_segment()]
            return response

        def play(request, response):
            path = Path(request.file_path)
            observed_playback.append((path, path.read_bytes()))
            response.success = True
            return response

        peer.create_service(SynthesizeSpeech, tts_service, synthesize)
        peer.create_service(PlayAudioFile, playback_service, play)
        publisher = peer.create_publisher(VisualGameEvent, event_topic, visual_game_event_qos())
        executor = SingleThreadedExecutor()
        executor.add_node(announcer)
        executor.add_node(peer)

        deadline = time.monotonic() + 2.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert publisher.get_subscription_count() > 0

        event = VisualGameEvent()
        event.request_id = f"game-{suffix}"
        event.execution_id = f"execution-{suffix}"
        event.game_name = "sorting_hat"
        event.handler = "sorting_hat_v1"
        event.announce = True
        event.state = "failed"
        event.success = False
        event.error_code = "NO_PERSON"
        publisher.publish(event)

        deadline = time.monotonic() + 2.0
        while (not observed_playback or observed_playback[0][0].exists()) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)

        assert observed_text == ["暂未识别到新生，请走入画面中央"]
        assert observed_playback[0][1].startswith(b"RIFF")
        assert not observed_playback[0][0].exists()
    finally:
        if executor is not None:
            executor.shutdown()
        if announcer is not None:
            announcer.close()
            announcer.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()


def test_late_announcer_does_not_replay_retained_terminal_event():
    rclpy.init()
    announcer = None
    peer = None
    executor = None
    try:
        suffix = f"id_{uuid.uuid4().hex}"
        event_topic = f"/visual_game_announcer_test/{suffix}/events"
        tts_service = f"/visual_game_announcer_test/{suffix}/synthesize"
        playback_service = f"/visual_game_announcer_test/{suffix}/play"
        peer = rclpy.create_node(f"visual_game_announcer_peer_{suffix}")
        publisher = peer.create_publisher(VisualGameEvent, event_topic, visual_game_event_qos())
        event = VisualGameEvent()
        event.request_id = f"game-{suffix}"
        event.execution_id = f"execution-{suffix}"
        event.game_name = "sorting_hat"
        event.handler = "sorting_hat_v1"
        event.announce = True
        event.state = "succeeded"
        event.success = True
        event.scene_summary = "斯莱特林"
        publisher.publish(event)

        observed_text = []

        def synthesize(request, response):
            observed_text.append(request.text)
            response.success = True
            response.audio_segments = [_audio_segment()]
            return response

        peer.create_service(SynthesizeSpeech, tts_service, synthesize)
        peer.create_service(PlayAudioFile, playback_service, lambda _request, response: response)
        announcer = VisualGameAnnouncerNode(
            parameter_overrides=[
                Parameter("event_topic", Parameter.Type.STRING, event_topic),
                Parameter("tts_service", Parameter.Type.STRING, tts_service),
                Parameter("playback_service", Parameter.Type.STRING, playback_service),
            ]
        )
        executor = SingleThreadedExecutor()
        executor.add_node(peer)
        executor.add_node(announcer)

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)

        assert observed_text == []
    finally:
        if executor is not None:
            executor.shutdown()
        if announcer is not None:
            announcer.close()
            announcer.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()
