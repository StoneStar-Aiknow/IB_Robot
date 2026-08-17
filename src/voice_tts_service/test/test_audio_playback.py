import subprocess
import wave

import pytest

from voice_tts_service.audio_playback import AudioFilePlayer, AudioPlaybackConfig, AudioPlaybackError


def _write_wav(path):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 160)


def test_player_validates_and_invokes_aplay_without_shell(monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    _write_wav(audio_path)
    calls = []

    monkeypatch.setattr("voice_tts_service.audio_playback.shutil.which", lambda command: "/usr/bin/aplay")

    def fake_run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("voice_tts_service.audio_playback.subprocess.run", fake_run)

    result = AudioFilePlayer(AudioPlaybackConfig(timeout_sec=12.0)).play(str(audio_path))

    assert result == audio_path
    assert calls == [
        (
            ["/usr/bin/aplay", "-q", str(audio_path)],
            {"check": False, "capture_output": True, "text": True, "timeout": 12.0},
        )
    ]


@pytest.mark.parametrize(
    ("file_path", "error_code"),
    [
        ("", "INVALID_PATH"),
        ("relative.wav", "INVALID_PATH"),
    ],
)
def test_player_rejects_invalid_paths(file_path, error_code):
    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer().play(file_path)
    assert error.value.code == error_code


def test_player_reports_missing_and_invalid_files(tmp_path):
    with pytest.raises(AudioPlaybackError) as missing:
        AudioFilePlayer().play(str(tmp_path / "missing.wav"))
    assert missing.value.code == "FILE_NOT_FOUND"

    invalid_path = tmp_path / "invalid.wav"
    invalid_path.write_bytes(b"not wave data")
    with pytest.raises(AudioPlaybackError) as invalid:
        AudioFilePlayer().play(str(invalid_path))
    assert invalid.value.code == "INVALID_AUDIO_FILE"


def test_player_reports_process_failures(monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    _write_wav(audio_path)
    monkeypatch.setattr("voice_tts_service.audio_playback.shutil.which", lambda command: "/usr/bin/aplay")
    monkeypatch.setattr(
        "voice_tts_service.audio_playback.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "device busy"),
    )

    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer().play(str(audio_path))
    assert error.value.code == "PLAYBACK_FAILED"
    assert "device busy" in str(error.value)


def test_player_reports_timeout(monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    _write_wav(audio_path)
    monkeypatch.setattr("voice_tts_service.audio_playback.shutil.which", lambda command: "/usr/bin/aplay")

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("voice_tts_service.audio_playback.subprocess.run", time_out)
    with pytest.raises(AudioPlaybackError) as error:
        AudioFilePlayer(AudioPlaybackConfig(timeout_sec=1.0)).play(str(audio_path))
    assert error.value.code == "PLAYBACK_TIMEOUT"
