import sys
import types
import wave

import pytest

from voice_tts_service.audio_playback import AudioFilePlayer, AudioPlaybackConfig, AudioPlaybackError
from voice_tts_service.audio_playback_node import AudioPlaybackNode


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


def _make_playback_node(publisher, *, timeout_sec=1.0):
    node = AudioPlaybackNode.__new__(AudioPlaybackNode)
    node._player = AudioFilePlayer(AudioPlaybackConfig(timeout_sec=timeout_sec))
    node._audio_pub = publisher
    node._playback_sample_rate = 24000
    node._playback_channels = 1
    return node


def _install_fake_audio_common_msgs(monkeypatch):
    class FakeAudioData:
        def __init__(self):
            self.data = []

    msg_module = types.ModuleType("audio_common_msgs.msg")
    msg_module.AudioData = FakeAudioData
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
