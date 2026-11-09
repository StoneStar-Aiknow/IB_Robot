"""Validate WAV files before publishing them through audio_common."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from voice_tts_service.audio_utils import resample_pcm16


class AudioPlaybackError(RuntimeError):
    """Stable audio playback failure returned by the ROS service."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """Configuration for one synchronous audio_common publication."""

    timeout_sec: float = 300.0
    player: str = "aplay"


class AudioFilePlayer:
    """Validate server-local WAV files for the shared playback topic."""

    def __init__(self, config: AudioPlaybackConfig | None = None) -> None:
        self.config = config or AudioPlaybackConfig()
        self._state_lock = threading.Lock()
        self._play_lock = threading.Lock()
        self._generation = 0
        self._process: subprocess.Popen | None = None

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def stop(self) -> None:
        """Cancel active/queued offline playback and reap the current child."""
        with self._state_lock:
            self._generation += 1
            if self._process is not None:
                self._terminate(self._process)

    def play(self, file_path: str, cancel: threading.Event | None = None) -> Path:
        """Play a validated WAV through the configured child, retaining the old API."""
        with self._state_lock:
            generation = self._generation

        def check_cancel():
            with self._state_lock:
                canceled = generation != self._generation or (cancel is not None and cancel.is_set())
            if canceled:
                raise AudioPlaybackError("PLAYBACK_CANCELED", "audio file playback canceled")

        with self._play_lock:
            check_cancel()
            path = self.validate_file(file_path)
            player = shutil.which(self.config.player)
            if player is None:
                raise AudioPlaybackError("PLAYER_NOT_FOUND", f"audio player is unavailable: {self.config.player}")
            if self.config.timeout_sec <= 0:
                raise AudioPlaybackError("INVALID_TIMEOUT", "audio playback timeout must be positive")
            process = None
            deadline = time.monotonic() + self.config.timeout_sec
            try:
                with self._state_lock:
                    if generation != self._generation or (cancel is not None and cancel.is_set()):
                        raise AudioPlaybackError("PLAYBACK_CANCELED", "audio file playback canceled")
                    process = subprocess.Popen(
                        [player, "-q", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    self._process = process
                while True:
                    check_cancel()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AudioPlaybackError(
                            "PLAYBACK_TIMEOUT", f"audio playback exceeded {self.config.timeout_sec:g} seconds"
                        )
                    try:
                        stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancel()
                if process.returncode != 0:
                    detail = (stderr or stdout).strip()
                    message = f"audio player exited with status {process.returncode}"
                    raise AudioPlaybackError("PLAYBACK_FAILED", f"{message}: {detail}" if detail else message)
                return path
            except OSError as exc:
                raise AudioPlaybackError("PLAYBACK_FAILED", f"failed to start audio player: {exc}") from exc
            finally:
                if process is not None:
                    with self._state_lock:
                        self._terminate(process)
                        self._process = None
                    process.communicate()

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

    @classmethod
    def load_pcm_for_playback(
        cls,
        file_path: str,
        *,
        sample_rate: int,
        channels: int,
        resample: bool,
        sample_width: int = 2,
    ) -> bytes:
        """Return the WAV's PCM in the playback contract, resampling only when explicitly enabled.

        Channel count and sample width are never converted. A sample-rate mismatch is rejected unless
        ``resample`` is set, so audio is never silently played at the wrong rate.
        """

        if not resample:
            path = cls.validate_pcm_format(
                file_path, sample_rate=sample_rate, channels=channels, sample_width=sample_width
            )
            with wave.open(str(path), "rb") as stream:
                return stream.readframes(stream.getnframes())

        path = cls.validate_file(file_path)
        try:
            with wave.open(str(path), "rb") as stream:
                actual = (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                source_rate = stream.getframerate()
                pcm = stream.readframes(stream.getnframes())
        except (EOFError, OSError, wave.Error) as exc:
            raise AudioPlaybackError("INVALID_AUDIO_FILE", f"invalid WAV file: {exc}") from exc
        if actual != (channels, sample_width, "NONE"):
            raise AudioPlaybackError(
                "UNSUPPORTED_AUDIO_FORMAT",
                "audio_common playback requires "
                f"{channels} channel(s), {sample_width * 8}-bit PCM (any rate, resampled to {sample_rate} Hz); "
                f"got {actual[0]} channel(s), {actual[1] * 8}-bit, {actual[2]}",
            )
        return resample_pcm16(pcm, channels=channels, source_rate=source_rate, target_rate=sample_rate)


__all__ = ["AudioFilePlayer", "AudioPlaybackConfig", "AudioPlaybackError"]
