import math
import struct
import sys
import types
import wave

import pytest

from voice_tts_service.audio_playback import AudioFilePlayer, AudioPlaybackConfig, AudioPlaybackError
from voice_tts_service.audio_playback_node import (
    AUDIO_DATA_STAMPED_TYPE,
    AUDIO_DATA_TYPE,
    AudioPlaybackNode,
    _audio_message_class,
    _publisher_qos,
)


def _write_wav(path, *, sample_rate=16000, channels=1, sample_width=2, frames=160):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00" * sample_width * channels * frames)


@pytest.mark.parametrize(
    ("file_path", "error_code"),
    [
        ("", "INVALID_PATH"),
        ("relative.wav", "INVALID_PATH"),
    ],
)
def test_player_rejects_invalid_paths(file_path, error_code):
    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer.validate_file(file_path)
    assert error.value.code == error_code


def test_player_reports_missing_and_invalid_files(tmp_path):
    with pytest.raises(AudioPlaybackError) as missing:
        AudioFilePlayer.validate_file(str(tmp_path / "missing.wav"))
    assert missing.value.code == "FILE_NOT_FOUND"

    invalid_path = tmp_path / "invalid.wav"
    invalid_path.write_bytes(b"not wave data")
    with pytest.raises(AudioPlaybackError) as invalid:
        AudioFilePlayer.validate_file(str(invalid_path))
    assert invalid.value.code == "INVALID_AUDIO_FILE"


def test_shared_playback_validates_fixed_pcm_contract(tmp_path):
    compatible = tmp_path / "compatible.wav"
    incompatible = tmp_path / "incompatible.wav"
    _write_wav(compatible, sample_rate=24000)
    _write_wav(incompatible, sample_rate=16000)

    assert AudioFilePlayer.validate_pcm_format(str(compatible), sample_rate=24000, channels=1) == compatible
    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer.validate_pcm_format(str(incompatible), sample_rate=24000, channels=1)
    assert error.value.code == "UNSUPPORTED_AUDIO_FORMAT"
    assert "got 16000 Hz" in str(error.value)


class _FakeAudioPublisher:
    def __init__(self, subscribers=1, ack=True):
        self.subscribers = subscribers
        self.ack = ack
        self.messages = []
        self.ack_timeouts = []

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, message):
        self.messages.append(message)

    def wait_for_all_acked(self, timeout=None):
        self.ack_timeouts.append(timeout)
        return self.ack


class _FakeAudioData:
    def __init__(self):
        self.data = []


class _FakeHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class _FakeAudioDataStamped:
    def __init__(self):
        self.header = _FakeHeader()
        self.audio = _FakeAudioData()


class _FakeClock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: "stamp")


def _make_playback_node(
    publisher,
    *,
    timeout_sec=1.0,
    sample_rate=24000,
    resample=False,
    message_type=AUDIO_DATA_TYPE,
    chunk_ms=50,
    prebuffer_ms=0,
):
    node = AudioPlaybackNode.__new__(AudioPlaybackNode)
    node._player = AudioFilePlayer(AudioPlaybackConfig(timeout_sec=timeout_sec))
    node._audio_pub = publisher
    node._playback_sample_rate = sample_rate
    node._playback_channels = 1
    node._resample = resample
    node._chunk_ms = chunk_ms
    node._prebuffer_ms = prebuffer_ms
    node._audio_message_type = message_type
    node._audio_message_class = _FakeAudioDataStamped if message_type == AUDIO_DATA_STAMPED_TYPE else _FakeAudioData
    node.get_clock = _FakeClock
    return node


def _install_fake_audio_common_msgs(monkeypatch):
    msg_module = types.ModuleType("audio_common_msgs.msg")
    msg_module.AudioData = _FakeAudioData
    msg_module.AudioDataStamped = _FakeAudioDataStamped
    package_module = types.ModuleType("audio_common_msgs")
    package_module.msg = msg_module
    monkeypatch.setitem(sys.modules, "audio_common_msgs", package_module)
    monkeypatch.setitem(sys.modules, "audio_common_msgs.msg", msg_module)


def test_shared_playback_waits_for_subscriber_and_publishes_all_chunks(monkeypatch, tmp_path):
    audio_path = tmp_path / "compatible.wav"
    _write_wav(audio_path, sample_rate=24000, frames=2400)
    publisher = _FakeAudioPublisher()
    node = _make_playback_node(publisher)
    _install_fake_audio_common_msgs(monkeypatch)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", lambda _seconds: None)

    assert node._publish_audio(str(audio_path)) == str(audio_path)
    assert len(publisher.messages) == 2
    assert all(len(message.data) == 2400 for message in publisher.messages)
    assert len(publisher.ack_timeouts) == 1
    assert all(timeout.nanoseconds > 0 for timeout in publisher.ack_timeouts)


def test_shared_playback_reports_missing_subscriber(monkeypatch, tmp_path):
    audio_path = tmp_path / "compatible.wav"
    _write_wav(audio_path, sample_rate=24000)
    publisher = _FakeAudioPublisher(subscribers=0)
    node = _make_playback_node(publisher, timeout_sec=0.01)
    clock = iter((0.0, 0.02))
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", lambda _seconds: None)

    with pytest.raises(AudioPlaybackError) as error:
        node._publish_audio(str(audio_path))
    assert error.value.code == "PLAYER_NOT_READY"


def test_shared_playback_reports_ack_timeout(monkeypatch, tmp_path):
    audio_path = tmp_path / "compatible.wav"
    _write_wav(audio_path, sample_rate=24000)
    publisher = _FakeAudioPublisher(ack=False)
    node = _make_playback_node(publisher)
    _install_fake_audio_common_msgs(monkeypatch)

    with pytest.raises(AudioPlaybackError) as error:
        node._publish_audio(str(audio_path))
    assert error.value.code == "PLAYBACK_TIMEOUT"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0.0
        self.now += seconds


def test_sample_clock_does_not_accumulate_ack_or_publish_latency(monkeypatch, tmp_path):
    """20 ms of work + 50 ms ACK latency must not turn each 50 ms chunk into 120 ms."""
    audio_path = tmp_path / "speech.wav"
    _write_wav(audio_path, sample_rate=24000, frames=24000)
    clock = _Clock()
    published_at = []
    acknowledged_at = []

    class SlowPublisher(_FakeAudioPublisher):
        def publish(self, message):
            published_at.append(clock.now)
            clock.sleep(0.02)
            super().publish(message)

        def wait_for_all_acked(self, timeout=None):
            acknowledged_at.append(clock.now)
            clock.sleep(0.05)
            return super().wait_for_all_acked(timeout)

    publisher = SlowPublisher()
    node = _make_playback_node(publisher, timeout_sec=3.0)
    _install_fake_audio_common_msgs(monkeypatch)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    node._publish_audio(str(audio_path))

    assert published_at == pytest.approx([index * 0.05 for index in range(20)])
    assert acknowledged_at == pytest.approx([1.0])
    assert clock.now == pytest.approx(1.05)
    assert len(publisher.messages) == 20


def test_sample_clock_preserves_partial_chunk_and_pcm_bytes(monkeypatch, tmp_path):
    audio_path = tmp_path / "partial.wav"
    pcm = bytes(range(256)) * 10
    with wave.open(str(audio_path), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)
    clock = _Clock()
    publisher = _FakeAudioPublisher()
    node = _make_playback_node(publisher)
    _install_fake_audio_common_msgs(monkeypatch)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    node._publish_audio(str(audio_path))

    assert b"".join(bytes(message.data) for message in publisher.messages) == pcm
    assert [len(message.data) for message in publisher.messages] == [2400, 160]
    assert clock.now == pytest.approx(len(pcm) / 48000)
    assert len(publisher.ack_timeouts) == 1


def test_publish_stall_cannot_run_past_playback_budget(monkeypatch, tmp_path):
    audio_path = tmp_path / "stall.wav"
    _write_wav(audio_path, sample_rate=24000, frames=2400)
    clock = _Clock()

    class StalledPublisher(_FakeAudioPublisher):
        def publish(self, message):
            clock.sleep(2.0)
            super().publish(message)

    publisher = StalledPublisher()
    node = _make_playback_node(publisher, timeout_sec=1.0)
    _install_fake_audio_common_msgs(monkeypatch)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    with pytest.raises(AudioPlaybackError) as error:
        node._publish_audio(str(audio_path))

    assert error.value.code == "PLAYBACK_TIMEOUT"
    assert len(publisher.messages) == 1


def _write_tone(path, *, sample_rate, seconds, frequency=1000.0, channels=1):
    frames = int(sample_rate * seconds)
    values = [round(0.5 * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(frames)]
    interleaved = [value for value in values for _ in range(channels)]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))


def test_load_without_resample_keeps_rejecting_rate_mismatch(tmp_path):
    audio_path = tmp_path / "zipvoice.wav"
    _write_wav(audio_path, sample_rate=24000)

    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer.load_pcm_for_playback(str(audio_path), sample_rate=16000, channels=1, resample=False)
    assert error.value.code == "UNSUPPORTED_AUDIO_FORMAT"
    assert "got 24000 Hz" in str(error.value)


def test_load_with_resample_converts_to_playback_rate(tmp_path):
    audio_path = tmp_path / "zipvoice.wav"
    _write_tone(audio_path, sample_rate=24000, seconds=0.5)

    pcm = AudioFilePlayer.load_pcm_for_playback(str(audio_path), sample_rate=16000, channels=1, resample=True)

    assert len(pcm) == 8000 * 2


def test_load_with_resample_still_rejects_channel_mismatch(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    _write_tone(audio_path, sample_rate=24000, seconds=0.1, channels=2)

    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer.load_pcm_for_playback(str(audio_path), sample_rate=16000, channels=1, resample=True)
    assert error.value.code == "UNSUPPORTED_AUDIO_FORMAT"


def test_resampled_stamped_playback_is_paced_on_the_target_rate(monkeypatch, tmp_path):
    """24 kHz ZipVoice audio reaches a 16 kHz runtime speaker as stamped 50 ms chunks."""
    audio_path = tmp_path / "zipvoice.wav"
    _write_tone(audio_path, sample_rate=24000, seconds=1.0)
    clock = _Clock()
    publisher = _FakeAudioPublisher()
    node = _make_playback_node(
        publisher, timeout_sec=3.0, sample_rate=16000, resample=True, message_type=AUDIO_DATA_STAMPED_TYPE
    )
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    node._publish_audio(str(audio_path))

    assert all(isinstance(message, _FakeAudioDataStamped) for message in publisher.messages)
    assert all(message.header.stamp == "stamp" for message in publisher.messages)
    assert [len(message.audio.data) for message in publisher.messages] == [1600] * 20
    expected = AudioFilePlayer.load_pcm_for_playback(str(audio_path), sample_rate=16000, channels=1, resample=True)
    assert b"".join(bytes(message.audio.data) for message in publisher.messages) == expected
    assert clock.now == pytest.approx(1.0)


def test_chunk_ms_sets_the_publication_granularity(monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    _write_wav(audio_path, sample_rate=16000, frames=16000)
    clock = _Clock()
    publisher = _FakeAudioPublisher()
    node = _make_playback_node(publisher, timeout_sec=3.0, sample_rate=16000, chunk_ms=100)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    node._publish_audio(str(audio_path))

    assert [len(message.data) for message in publisher.messages] == [3200] * 10


def test_unknown_audio_message_type_is_rejected(monkeypatch):
    _install_fake_audio_common_msgs(monkeypatch)

    assert _audio_message_class(AUDIO_DATA_STAMPED_TYPE) is _FakeAudioDataStamped
    with pytest.raises(ValueError, match="audio_message_type"):
        _audio_message_class("std_msgs/msg/String")


def _paced_playback(monkeypatch, tmp_path, *, seconds, chunk_ms, prebuffer_ms):
    """Play ``seconds`` of audio on a virtual clock; return each write's (time, audio already written)."""
    audio_path = tmp_path / "reply.wav"
    _write_wav(audio_path, sample_rate=16000, frames=int(16000 * seconds))
    clock = _Clock()
    writes = []

    class RecordingPublisher(_FakeAudioPublisher):
        def publish(self, message):
            writes.append((clock.now, len(self.messages) * chunk_ms / 1000))
            super().publish(message)

    publisher = RecordingPublisher()
    node = _make_playback_node(
        publisher, timeout_sec=60.0, sample_rate=16000, chunk_ms=chunk_ms, prebuffer_ms=prebuffer_ms
    )
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.monotonic", clock.monotonic)
    monkeypatch.setattr("voice_tts_service.audio_playback_node.time.sleep", clock.sleep)

    node._publish_audio(str(audio_path))

    return writes, clock.now


def test_the_lead_is_kept_for_the_whole_reply_not_only_its_start(monkeypatch, tmp_path):
    """A player waiting on every write reports PLAYING/STOPPED around it; a lead keeps it fed."""
    writes, returned_at = _paced_playback(monkeypatch, tmp_path, seconds=12.0, chunk_ms=200, prebuffer_ms=1000)

    # Once the first second is queued, every later write finds a full second not yet played.
    lead_at_each_write = [written - now for now, written in writes if written >= 1.0]
    assert lead_at_each_write and min(lead_at_each_write) == pytest.approx(1.0)
    assert returned_at == pytest.approx(12.0)


def test_a_reply_shorter_than_the_lead_returns_once_it_has_played(monkeypatch, tmp_path):
    # The voice bridge resumes listening when this returns; returning on the write would let
    # the robot hear its own reply.
    writes, returned_at = _paced_playback(monkeypatch, tmp_path, seconds=0.6, chunk_ms=200, prebuffer_ms=1000)

    assert [now for now, _written in writes] == [0.0, 0.0, 0.0]
    assert returned_at == pytest.approx(0.6)


def test_the_bound_speaker_qos_reaches_the_publisher():
    """The X2 runtime subscribes best_effort; RELIABLE is compatible, which is why it stayed silent."""
    import json

    from rclpy.qos import QoSProfile, ReliabilityPolicy

    qos = _publisher_qos(
        json.dumps({"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 10})
    )

    assert isinstance(qos, QoSProfile)
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.depth == 10


def test_an_undeclared_speaker_qos_keeps_the_rclpy_default():
    assert _publisher_qos("") == 10


def test_malformed_qos_json_is_rejected_instead_of_falling_back():
    with pytest.raises(ValueError, match="audio_qos_json"):
        _publisher_qos("{not json")
