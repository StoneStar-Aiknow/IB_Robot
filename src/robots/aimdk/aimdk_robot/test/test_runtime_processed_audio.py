"""Unit tests for the runtime-internal processed-audio (enhanced speech) bridge.

The runtime owns this path, so the tests drive the node's own methods instead of
a separate process. The vendor message type is injected through ``sys.modules``
so the suite runs without the AimDK overlay, matching the no-rclpy pattern the
package already uses elsewhere.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from aimdk_robot.runtime_node import AimdkRuntimeNode  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402

SPEECH_INTERFACE = {
    "capability": "interaction.audio_capture",
    "kind": "topic",
    "direction": "publish",
    "endpoint": "/audio/speech_chunk",
    "message_type": "ibrobot_msgs/msg/SpeechAudioChunk",
    "qos": {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 50},
}

#: The diagnostic republish is a public endpoint like any other, so its
#: endpoint and QoS come from its own interface declaration.
ENHANCED_CAPTURE_INTERFACE = {
    "capability": "interaction.audio_capture",
    "kind": "topic",
    "direction": "publish",
    "endpoint": "/audio/enhanced_capture_stamped",
    "message_type": "audio_common_msgs/msg/AudioDataStamped",
    "qos": {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10},
}


class Publisher:
    def __init__(self, topic, qos):
        self.topic, self.qos = topic, qos
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Timer:
    def __init__(self, period, callback):
        self.period, self.callback = period, callback


class Logger:
    def __init__(self):
        self.records = []

    def info(self, message):
        self.records.append(("info", message))

    def warning(self, message):
        self.records.append(("warning", message))

    def error(self, message):
        self.records.append(("error", message))


class Clock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=2))


class Stamped:
    """Stand-in for audio_common_msgs/AudioDataStamped, absent on some hosts."""

    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.audio = types.SimpleNamespace(data=[])


@pytest.fixture()
def vendor_module(monkeypatch):
    """Inject the optional message packages so the path is testable without an overlay."""
    package = types.ModuleType("aimdk_msgs")
    messages = types.ModuleType("aimdk_msgs.msg")
    messages.ProcessedAudioOutput = type("ProcessedAudioOutput", (), {})
    monkeypatch.setitem(sys.modules, "aimdk_msgs", package)
    monkeypatch.setitem(sys.modules, "aimdk_msgs.msg", messages)

    audio_package = types.ModuleType("audio_common_msgs")
    audio_messages = types.ModuleType("audio_common_msgs.msg")
    audio_messages.AudioDataStamped = Stamped
    monkeypatch.setitem(sys.modules, "audio_common_msgs", audio_package)
    monkeypatch.setitem(sys.modules, "audio_common_msgs.msg", audio_messages)


def runtime(interfaces=None, vendor=None, clock=None, enhanced_capture=True):
    """A runtime object wired just enough to exercise the processed-audio path."""
    node = AimdkRuntimeNode.__new__(AimdkRuntimeNode)
    node._vendor = (
        vendor
        if vendor is not None
        else {
            "topics": {"processed_audio": "/agent/process_audio_output"},
            "qos": {"processed_audio": {"reliability": "best_effort", "durability": "volatile"}},
        }
    )
    declared = {"audio.speech": dict(SPEECH_INTERFACE)}
    if enhanced_capture:
        declared["audio.enhanced_capture"] = dict(ENHANCED_CAPTURE_INTERFACE)
    declared.update(interfaces or {})
    node._speech_description = {"interfaces": declared}
    node._speech_enabled = False
    node._speech_last_receipt = None
    node._speech_last_seen = None
    import threading

    node._speech_state_lock = threading.Lock()
    node._speech_cb = object()
    # rclpy.Node already exposes read-only `publishers`/`subscriptions`, so the
    # recording lists use private names of our own.
    node._rec_publishers, node._rec_subscriptions, node._rec_timers = [], [], []
    node.logger = Logger()
    node.get_logger = lambda: node.logger
    node.get_clock = lambda: clock or Clock()
    node.create_publisher = lambda msg_type, topic, qos=None, **kwargs: (
        node._rec_publishers.append(Publisher(str(topic), qos)) or node._rec_publishers[-1]
    )
    node.create_subscription = lambda msg_type, topic, cb, qos, **kwargs: (
        node._rec_subscriptions.append((msg_type, str(topic), cb, qos)) or node._rec_subscriptions[-1]
    )
    node.create_timer = lambda period, cb, **kwargs: node._rec_timers.append(Timer(period, cb)) or node._rec_timers[-1]
    return node


def vendor_message(pcm=b"\x01\x02", state=2, sequence=1, stream_id=7):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=Time(sec=3, nanosec=4), frame_id="vendor", sequence=sequence),
        stream_id=stream_id,
        audio_vad_state=types.SimpleNamespace(value=state),
        audio_data=pcm,
    )


@pytest.mark.parametrize(
    ("capability", "topic", "expected"),
    [
        (True, "/agent/process_audio_output", True),
        (True, None, False),
        (False, "/agent/process_audio_output", False),
    ],
)
def test_processed_audio_gate_needs_both_capability_and_topic(vendor_module, capability, topic, expected):
    node = runtime()
    node._state = types.SimpleNamespace(has_capability=lambda _name: capability)
    assert node._wants_processed_audio({} if topic is None else {"processed_audio": topic}) is expected


def test_a_profile_without_the_vendor_topic_creates_nothing(vendor_module):
    # Drive the real decision path, not a restatement of the condition: a
    # profile that never names the vendor topic must not add a subscription,
    # a publisher or a heartbeat timer.
    node = runtime()
    node._state = types.SimpleNamespace(has_capability=lambda _name: True)
    assert node._setup_processed_audio_if_declared({}) is False
    assert node._rec_subscriptions == [] and node._rec_publishers == [] and node._rec_timers == []
    assert node._speech_enabled is False


def test_processed_audio_requires_the_declared_project_interface(vendor_module):
    # A profile that names the vendor topic but publishes no audio.speech is a
    # wiring error: fail fast instead of silently dropping the stream.
    node = runtime()
    node._speech_description = {"interfaces": {}}
    with pytest.raises(RuntimeError, match="no\\s+audio.speech interface"):
        node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})


def test_processed_audio_rejects_a_non_chunk_interface(vendor_module):
    node = runtime()
    node._speech_description = {
        "interfaces": {"audio.speech": {**SPEECH_INTERFACE, "message_type": "audio_common_msgs/msg/AudioDataStamped"}}
    }
    with pytest.raises(RuntimeError, match="SpeechAudioChunk"):
        node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})


def test_processed_audio_wires_vendor_qos_and_public_endpoint(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})

    chunk_pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    diag_pub = next(p for p in node._rec_publishers if p.topic == "/audio/enhanced_capture_stamped")
    # Public endpoint and QoS come from each interface's own declaration, not a
    # literal and not borrowed from a sibling interface.
    assert chunk_pub.qos.reliability.name == "RELIABLE" and chunk_pub.qos.depth == 50
    assert diag_pub.qos.reliability.name == "RELIABLE" and diag_pub.qos.depth == 10
    (_type, topic, _cb, qos) = node._rec_subscriptions[0]
    assert topic == "/agent/process_audio_output"
    # The vendor subscription keeps the vendor's own QoS, and never asks for the
    # latched history that would replay pre-start speech.
    assert qos.reliability.name == "BEST_EFFORT" and qos.durability.name == "VOLATILE"
    assert node._rec_timers and node._speech_enabled is True


def test_diagnostic_publisher_follows_its_interface_declaration(vendor_module):
    # Undeclared means unpublished: the vendor section must not be able to
    # create a public endpoint the runtime description does not advertise.
    node = runtime(enhanced_capture=False)
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    assert [p.topic for p in node._rec_publishers] == ["/audio/speech_chunk"]
    assert node._speech_capture_pub is None

    node = runtime(interfaces={"audio.enhanced_capture": {**ENHANCED_CAPTURE_INTERFACE, "endpoint": "/diag/x"}})
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    diag = next(p for p in node._rec_publishers if p.topic == "/diag/x")
    assert diag.qos.depth == 10


def test_heartbeat_carries_no_audio_and_advances_sequence(vendor_module):
    node = runtime(clock=Clock())
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._publish_speech_heartbeat()
    node._publish_speech_heartbeat()
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 2]
    assert all(m.state == 0 for m in pub.messages)
    assert all(list(m.audio_data) == [] for m in pub.messages)
    assert all(m.source_id == node._speech_source_id for m in pub.messages)


def test_empty_end_is_still_published_as_a_boundary(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(pcm=b"", state=3))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.state for m in pub.messages] == [3]
    assert pub.messages[0].sequence == 1
    # Nothing to mirror into the diagnostic stream when there is no PCM.
    diag = next(p for p in node._rec_publishers if p.topic == "/audio/enhanced_capture_stamped")
    assert diag.messages == []


def test_source_id_is_stable_across_messages(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(sequence=1))
    node._on_processed_audio(vendor_message(sequence=2))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    ids = {m.source_id for m in pub.messages}
    assert len(ids) == 1 and ids == {node._speech_source_id}


def test_contiguous_vendor_sequence_keeps_the_project_sequence_contiguous(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    for sequence in (1, 2, 3):
        node._on_processed_audio(vendor_message(sequence=sequence))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 2, 3]


def test_vendor_sequence_jump_marks_a_project_gap(vendor_module):
    # Only a strict forward skip counts, and it is what a firmware honouring
    # the IDL would emit on BEST_EFFORT loss. The consumer must then see a
    # project-side gap and drop the affected segment. X2 itself does not
    # exercise this path today: its firmware sends a constant (see the test
    # below), so this pins the contract rather than an observed field.
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(sequence=1))
    node._on_processed_audio(vendor_message(sequence=5))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 3]
    assert any(level == "warning" for level, _ in node.logger.records)


def test_constant_vendor_sequence_does_not_mark_a_gap(vendor_module):
    # The X2 firmware fills header.sequence with a constant despite the IDL
    # documenting a consecutively increasing ID (observed on hardware,
    # 2026-10-15). An "anything but last + 1" test flagged every message and
    # dropped all speech; only a strict forward skip is loss evidence.
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    for _ in range(3):
        node._on_processed_audio(vendor_message(sequence=0))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 2, 3]
    assert not any(level == "warning" for level, _ in node.logger.records)


def test_new_vendor_stream_restarts_the_gap_baseline(vendor_module):
    # stream_id names the vendor stream; its first message cannot be a gap.
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(sequence=9, stream_id=1))
    node._on_processed_audio(vendor_message(sequence=1, stream_id=2))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 2]


def test_repeated_and_reset_vendor_sequence_do_not_mark_a_gap(vendor_module):
    # Switching to "only a strict forward skip" also changed how repeats and
    # resets are treated, so pin that half of the contract too: treating a
    # non-increasing value as loss is what dropped every utterance on the robot.
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    for sequence in (5, 5, 2, 2, 3):
        node._on_processed_audio(vendor_message(sequence=sequence))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert [m.sequence for m in pub.messages] == [1, 2, 3, 4, 5]
    assert not any(level == "warning" for level, _ in node.logger.records)


def test_malformed_pcm_is_forwarded_for_the_consumer_to_abort(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(pcm=b"\x00", state=2))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    # Forwarded verbatim: the ASR adapter owns the "odd S16LE block" decision.
    assert list(pub.messages[0].audio_data) == [0]


def test_oversized_vendor_chunk_is_bounded(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(pcm=b"\x00" * 40000, state=2))
    pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    assert len(pub.messages[0].audio_data) == 16000


# --- the wake event read off the platform's own acknowledgement ---

WAKE_INTERFACE = {
    "capability": "interaction.audio_capture",
    "kind": "topic",
    "direction": "publish",
    "endpoint": "/audio/wake_event",
    "message_type": "ibrobot_msgs/msg/SpeechWakeEvent",
    "qos": {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10},
}
VENDOR_WITH_PLAYBACK = {
    "topics": {"processed_audio": "/agent/process_audio_output", "audio_playback": "/aima/hal/audio/playback"},
    "qos": {"processed_audio": {"reliability": "best_effort", "durability": "volatile"}},
    "wake_ack": {"pkg_names": ["tts"], "pair_window_s": 1.5, "token_pattern": "^_[0-9]+_"},
}


@pytest.fixture()
def playback_module(vendor_module):
    sys.modules["aimdk_msgs.msg"].AudioPlayback = type("AudioPlayback", (), {})


def _playback(pkg_name, token_id):
    return types.SimpleNamespace(pkg_name=pkg_name, token_id=token_id)


def test_a_profile_without_the_wake_interface_does_not_read_the_playback_topic(playback_module):
    node = runtime(vendor=dict(VENDOR_WITH_PLAYBACK))
    node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])

    assert not any(topic == "/aima/hal/audio/playback" for _type, topic, _cb, _qos in node._rec_subscriptions)
    assert not any(p.topic == "/audio/wake_event" for p in node._rec_publishers)


def test_the_wake_interface_needs_the_vendor_playback_topic(playback_module):
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)})
    with pytest.raises(RuntimeError, match="audio_playback"):
        node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})


def test_the_wake_interface_must_be_a_speech_wake_event(playback_module):
    node = runtime(
        interfaces={"audio.wake_event": {**WAKE_INTERFACE, "message_type": "std_msgs/msg/String"}},
        vendor=dict(VENDOR_WITH_PLAYBACK),
    )
    with pytest.raises(RuntimeError, match="ibrobot_msgs/msg/SpeechWakeEvent"):
        node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])


def test_the_wake_interface_needs_the_acknowledgement_token_form(playback_module):
    """Without it any playback token pairs, the platform's dialog answers included."""
    vendor = {**VENDOR_WITH_PLAYBACK, "wake_ack": {"pkg_names": ["tts"], "pair_window_s": 1.5}}
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)}, vendor=vendor)
    with pytest.raises(RuntimeError, match="token_pattern"):
        node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])


def _wake_paths(node):
    node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])
    wake_pub = next(p for p in node._rec_publishers if p.topic == "/audio/wake_event")
    speech_pub = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    on_playback = next(cb for _type, topic, cb, _qos in node._rec_subscriptions if topic == "/aima/hal/audio/playback")
    return wake_pub, speech_pub, on_playback


def test_an_acknowledged_speech_begin_publishes_one_wake_event_naming_its_segment(playback_module):
    """The event names the woken segment the way SpeakerEmbedding names one: the
    source and the sequence of its BEGIN chunk."""
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)}, vendor=dict(VENDOR_WITH_PLAYBACK))
    wake_pub, speech_pub, on_playback = _wake_paths(node)

    on_playback(_playback("tts", "_1792835430393_1ack"))
    node._on_processed_audio(vendor_message(state=1))
    node._on_processed_audio(vendor_message(state=2, sequence=2))
    on_playback(_playback("tts", "_1792835430393_1ack"))

    begin = next(chunk for chunk in speech_pub.messages if chunk.state == 1)
    assert len(wake_pub.messages) == 1
    event = wake_pub.messages[0]
    assert event.header.stamp == Time(sec=1, nanosec=2)
    assert event.source_id == begin.source_id
    assert event.segment_sequence == begin.sequence


def test_an_acknowledgement_after_the_begin_still_names_that_segment(playback_module):
    """Measured on the X2, the acknowledgement can land up to 1.0 s after the BEGIN,
    by which time later chunks of the segment have been published."""
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)}, vendor=dict(VENDOR_WITH_PLAYBACK))
    wake_pub, speech_pub, on_playback = _wake_paths(node)

    node._on_processed_audio(vendor_message(state=1))
    node._on_processed_audio(vendor_message(state=2, sequence=2))
    node._on_processed_audio(vendor_message(state=2, sequence=3))
    on_playback(_playback("tts", "_1792835430393_1ack"))

    begin = next(chunk for chunk in speech_pub.messages if chunk.state == 1)
    assert [event.segment_sequence for event in wake_pub.messages] == [begin.sequence]


def test_the_acknowledgement_is_read_best_effort_even_though_playback_is_published_reliable(playback_module):
    """Both X2 profiles set vendor.qos.audio_playback RELIABLE for the runtime's
    own publisher on that topic; the wake-ack subscription used to inherit it."""
    qos = {**VENDOR_WITH_PLAYBACK["qos"], "audio_playback": {"reliability": "reliable", "durability": "volatile"}}
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)}, vendor={**VENDOR_WITH_PLAYBACK, "qos": qos})
    node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])

    (subscription_qos,) = [
        qos for _type, topic, _cb, qos in node._rec_subscriptions if topic == "/aima/hal/audio/playback"
    ]
    assert subscription_qos.reliability.name == "BEST_EFFORT"


def test_speech_without_an_acknowledgement_publishes_no_wake_event(playback_module):
    node = runtime(interfaces={"audio.wake_event": dict(WAKE_INTERFACE)}, vendor=dict(VENDOR_WITH_PLAYBACK))
    node._setup_processed_audio(VENDOR_WITH_PLAYBACK["topics"])
    wake_pub = next(p for p in node._rec_publishers if p.topic == "/audio/wake_event")
    on_playback = next(cb for _type, topic, cb, _qos in node._rec_subscriptions if topic == "/aima/hal/audio/playback")

    node._on_processed_audio(vendor_message(state=1))
    on_playback(_playback("ibrobot.aimdk_robot", "own-playback"))

    assert wake_pub.messages == []


@pytest.mark.parametrize(("pcm_bytes", "state"), [(19200, 1), (20800, 3)])
def test_a_vendor_backfill_burst_is_forwarded_whole(vendor_module, pcm_bytes, state):
    # Measured on the X2 (2026-10-20): segments open and close with bursts of up to 20800 bytes,
    # and a 16000-byte cap cut speech out of the utterance, so the recognizer heard another word.
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    pcm = (bytes(range(256)) * (pcm_bytes // 256 + 1))[:pcm_bytes]
    node._on_processed_audio(vendor_message(pcm=pcm, state=state))
    speech = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    diag = next(p for p in node._rec_publishers if p.topic == "/audio/enhanced_capture_stamped")
    assert [(m.state, bytes(m.audio_data)) for m in speech.messages] == [(state, pcm)]
    assert [bytes(m.audio.data) for m in diag.messages] == [pcm]


def test_a_vendor_message_past_the_bound_is_dropped_and_marks_a_gap(vendor_module):
    node = runtime()
    node._setup_processed_audio({"processed_audio": "/agent/process_audio_output"})
    node._on_processed_audio(vendor_message(sequence=1))
    node._on_processed_audio(vendor_message(pcm=b"\x00" * (2 * 16000 * 2 + 2), state=2, sequence=2))
    node._on_processed_audio(vendor_message(sequence=3))
    speech = next(p for p in node._rec_publishers if p.topic == "/audio/speech_chunk")
    diag = next(p for p in node._rec_publishers if p.topic == "/audio/enhanced_capture_stamped")
    # None of it is published, and the gap left behind makes the consumer drop the segment
    # rather than hear it with speech cut out.
    assert [m.sequence for m in speech.messages] == [1, 3]
    assert len(diag.messages) == 2
    assert any(level == "warning" for level, _ in node.logger.records)
