"""Real ROS serialization and ordered delivery of atomic speech chunks, localhost only."""

import os
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor

from ibrobot_msgs.msg import SpeechAudioChunk
from voice_asr_service.external_vad_input import ExternalVadInput


def test_atomic_audio_loopback():
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("speech_chunk_loopback", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    samples, finals = [], []
    asr = SimpleNamespace(
        start_streaming=lambda: True,
        accept_waveform=lambda values: samples.append(values.copy()),
        get_final_result=lambda: SimpleNamespace(text="recognized"),
        reset=lambda: None,
    )
    try:
        voice = ExternalVadInput(
            node=node, asr=asr, chunk_topic="/test/atomic_speech", min_samples=1, on_final=finals.append
        )
        voice.enabled = True
        publisher = node.create_publisher(SpeechAudioChunk, "/test/atomic_speech", 50)
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1
        for sequence, state in ((1, 1), (2, 2), (3, 3)):
            msg = SpeechAudioChunk()
            msg.source_id, msg.sequence, msg.state = "loopback", sequence, state
            msg.audio_data = [sequence, 0]
            publisher.publish(msg)
        deadline = time.monotonic() + 5
        while not finals and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert finals == ["recognized"]
        np.testing.assert_array_equal(np.concatenate(samples), np.array([1, 2, 3]) / 32768)
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown(timeout_sec=0)
        context.shutdown()
