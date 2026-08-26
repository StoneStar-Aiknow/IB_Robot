"""Validate WAV files before publishing them through audio_common."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


class AudioPlaybackError(RuntimeError):
    """Stable audio playback failure returned by the ROS service."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """Configuration for one synchronous audio_common publication."""

    timeout_sec: float = 300.0


class AudioFilePlayer:
    """Validate server-local WAV files for the shared playback topic."""

    def __init__(self, config: AudioPlaybackConfig | None = None) -> None:
        self.config = config or AudioPlaybackConfig()

    @staticmethod
    def validate_file(file_path: str) -> Path:
        value = file_path.strip()
        if not value:
            raise AudioPlaybackError("INVALID_PATH", "file_path must not be empty")

        path = Path(value)
        if not path.is_absolute():
            raise AudioPlaybackError("INVALID_PATH", "file_path must be an absolute path on the playback host")
        if not path.exists():
            raise AudioPlaybackError("FILE_NOT_FOUND", f"audio file does not exist: {path}")
        if not path.is_file():
            raise AudioPlaybackError("NOT_A_FILE", f"audio path is not a regular file: {path}")
        if path.suffix.lower() != ".wav":
            raise AudioPlaybackError("UNSUPPORTED_FORMAT", "only WAV audio files are supported")

        try:
            with wave.open(str(path), "rb") as stream:
                if stream.getnchannels() <= 0 or stream.getframerate() <= 0 or stream.getnframes() <= 0:
                    raise AudioPlaybackError("INVALID_AUDIO_FILE", "WAV file contains invalid audio metadata")
        except (EOFError, OSError, wave.Error) as exc:
            raise AudioPlaybackError("INVALID_AUDIO_FILE", f"invalid WAV file: {exc}") from exc
        return path

    @classmethod
    def validate_pcm_format(
        cls,
        file_path: str,
        *,
        sample_rate: int,
        channels: int,
        sample_width: int = 2,
    ) -> Path:
        """Validate a WAV against the fixed raw PCM contract used by audio_play."""

        path = cls.validate_file(file_path)
        try:
            with wave.open(str(path), "rb") as stream:
                actual = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
        except (EOFError, OSError, wave.Error) as exc:
            raise AudioPlaybackError("INVALID_AUDIO_FILE", f"invalid WAV file: {exc}") from exc
        expected = (sample_rate, channels, sample_width, "NONE")
        if actual != expected:
            raise AudioPlaybackError(
                "UNSUPPORTED_AUDIO_FORMAT",
                "audio_common playback requires "
                f"{sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit PCM; "
                f"got {actual[0]} Hz, {actual[1]} channel(s), {actual[2] * 8}-bit, {actual[3]}",
            )
        return path


__all__ = ["AudioFilePlayer", "AudioPlaybackConfig", "AudioPlaybackError"]
