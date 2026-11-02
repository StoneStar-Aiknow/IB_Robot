"""speaker_id_node keeps running when the model fails on an utterance."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs.msg")

from voice_asr_service.speaker_id import node as speaker_node  # noqa: E402
from voice_asr_service.speaker_id.utterance import Utterance  # noqa: E402

MODEL = "campplus@test"


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _Embedder:
    """Fails on the utterances whose PCM it is told to, embeds the rest."""

    model_id = MODEL

    def __init__(self, failing=()) -> None:
        self.failing = set(failing)
        self.closed = False

    def embed(self, pcm):
        if pcm in self.failing:
            raise RuntimeError("ONNX Runtime: invalid input")
        return (1.0, 0.0, 0.0)

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _node(embedder):
    node = speaker_node.SpeakerIdNode.__new__(speaker_node.SpeakerIdNode)
    node.get_logger = lambda: logging.getLogger("speaker-id-test")
    node._values = {"bundle_path": "models/campplus-speaker", "deployment": "torch_cpu"}
    node._min_speech_seconds = 1.0
    node._failures = 0
    node._last_reload = -speaker_node.RELOAD_INTERVAL_SEC
    node._embedder = embedder
    node._model_id = MODEL
    node._publisher = _Publisher()
    return node


def _utterance(pcm: bytes, sequence: int) -> Utterance:
    from std_msgs.msg import Header

    return Utterance(pcm=pcm, speech_seconds=2.0, source_id="runtime-1", segment_sequence=sequence, header=Header())


def test_a_failure_on_one_utterance_publishes_nothing_for_it_and_the_next_is_embedded():
    """Published empty, the failure would read as "too short" and, with
    speaker_reject_short, turn the user away; published nothing, the gate judges
    that one transcript without a voiceprint after its grace period."""
    node = _node(_Embedder(failing={b"bad"}))

    node._publish(_utterance(b"bad", 1))
    node._publish(_utterance(b"good", 2))

    assert [message.segment_sequence for message in node._publisher.messages] == [2]
    assert list(node._publisher.messages[0].embedding) == [1.0, 0.0, 0.0]


def test_repeated_failures_reload_the_model(monkeypatch):
    broken = _Embedder(failing={b"bad"})
    reloaded = _Embedder()
    monkeypatch.setattr(speaker_node, "SpeakerEmbedder", lambda *_args: reloaded)
    node = _node(broken)

    for sequence in range(1, speaker_node.RELOAD_AFTER_FAILURES + 1):
        node._publish(_utterance(b"bad", sequence))
    node._publish(_utterance(b"bad", 10))

    assert broken.closed and node._embedder is reloaded
    assert [message.segment_sequence for message in node._publisher.messages] == [10]


def test_a_reload_that_fails_leaves_the_node_up_without_a_model(monkeypatch):
    def _broken_load(*_args):
        raise RuntimeError("bundle missing")

    monkeypatch.setattr(speaker_node, "SpeakerEmbedder", _broken_load)
    node = _node(_Embedder(failing={b"bad"}))

    for sequence in range(1, speaker_node.RELOAD_AFTER_FAILURES + 2):
        node._publish(_utterance(b"bad", sequence))

    assert node._embedder is None and node._publisher.messages == []


def test_a_reloaded_model_of_another_identity_is_refused(monkeypatch):
    """Its vectors could not be compared with the anchors the gate already holds."""
    other = _Embedder()
    other.model_id = "campplus@other"
    monkeypatch.setattr(speaker_node, "SpeakerEmbedder", lambda *_args: other)
    node = _node(_Embedder(failing={b"bad"}))

    for sequence in range(1, speaker_node.RELOAD_AFTER_FAILURES + 1):
        node._publish(_utterance(b"bad", sequence))

    assert node._embedder is None and other.closed
