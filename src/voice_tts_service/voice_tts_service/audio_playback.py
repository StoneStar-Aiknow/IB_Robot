"""Validated local WAV playback through ALSA."""

from __future__ import annotations

import shutil
import subprocess
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
    """Configuration for one synchronous ALSA playback operation."""

    timeout_sec: float = 300.0
    player: str = "aplay"


class AudioFilePlayer:
    """Validate and play server-local WAV files without invoking a shell."""

    def __init__(self, config: AudioPlaybackConfig | None = None) -> None:
        self.config = config or AudioPlaybackConfig()

    @staticmethod
    def _validate_file(file_path: str) -> Path:
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

    def play(self, file_path: str) -> Path:
        """Play one WAV file and return its validated path after completion."""

        path = self._validate_file(file_path)
        player = shutil.which(self.config.player)
        if player is None:
            raise AudioPlaybackError("PLAYER_NOT_FOUND", f"audio player is unavailable: {self.config.player}")
        if self.config.timeout_sec <= 0:
            raise AudioPlaybackError("INVALID_TIMEOUT", "audio playback timeout must be positive")

        command = [player, "-q", str(path)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioPlaybackError(
                "PLAYBACK_TIMEOUT",
                f"audio playback exceeded {self.config.timeout_sec:g} seconds",
            ) from exc
        except OSError as exc:
            raise AudioPlaybackError("PLAYBACK_FAILED", f"failed to start audio player: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            message = f"audio player exited with status {completed.returncode}"
            if detail:
                message = f"{message}: {detail}"
            raise AudioPlaybackError("PLAYBACK_FAILED", message)
        return path


__all__ = ["AudioFilePlayer", "AudioPlaybackConfig", "AudioPlaybackError"]
