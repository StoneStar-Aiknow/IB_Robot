"""Coordinate H.264 RTP stream recording across episode lifecycle.

Manages recorder instances for all RTP observation streams, synchronized
with episode action goal/result lifecycle from episode_recorder.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_service.h264_stream_recorder import H264StreamRecorder


class VideoRecordingCoordinator:
    """Coordinate RTP video recording for all observation streams in an episode.

    Thread-safe: Can be called from action server callbacks while RTP receiver
    threads are writing frames.
    """

    def __init__(self) -> None:
        """Initialize coordinator with no active recorders."""
        self._lock = Lock()
        self._recorders: dict[str, H264StreamRecorder] = {}
        self._episode_dir: Path | None = None
        self._active = False

    def register_recorder(self, observation_key: str, recorder: H264StreamRecorder) -> None:
        """Register a recorder for an observation stream.

        Args:
            observation_key: Observation key (e.g., "observation.images.top")
            recorder: H264StreamRecorder instance (should be wired into H264RtpReceiver)

        Raises:
            ValueError: If observation_key already registered
        """
        with self._lock:
            if observation_key in self._recorders:
                raise ValueError(f"Recorder already registered for {observation_key}")
            self._recorders[observation_key] = recorder

    def start_episode(self, episode_dir: Path) -> None:
        """Start recording for all registered streams.

        Args:
            episode_dir: Episode directory path (e.g., /tmp/episodes/my_dataset/episodes/episode_000042)

        Raises:
            ValueError: If already recording
            OSError: If episode directory does not exist or files cannot be created
        """
        with self._lock:
            if self._active:
                raise ValueError("Cannot start new episode while previous episode is active")

            self._episode_dir = Path(episode_dir)
            if not self._episode_dir.exists():
                raise OSError(f"Episode directory does not exist: {self._episode_dir}")

            started = []
            try:
                for obs_key, recorder in self._recorders.items():
                    recorder.start_episode(self._episode_dir, obs_key)
                    started.append(recorder)
            except Exception:
                for recorder in reversed(started):
                    recorder.stop_episode()
                self._episode_dir = None
                raise

            self._active = True

    def stop_episode(self) -> bool:
        """Stop recording for all registered streams.

        Returns:
            True if all recorders preserved their files (success or tolerant mode with gaps);
            False if any recorder discarded files (strict mode with gaps).

        The return value should drive the episode action result:
        - True: episode succeeded, files on disk
        - False: episode failed, some RTP streams had unrecoverable gaps

        Raises:
            ValueError: If not currently recording
        """
        with self._lock:
            if not self._active:
                raise ValueError("No active episode to stop")

            all_valid = True
            for recorder in self._recorders.values():
                kept = recorder.stop_episode()
                if not kept:
                    all_valid = False

            if not all_valid:
                for recorder in self._recorders.values():
                    recorder.discard_files()

            self._active = False
            self._episode_dir = None

            return all_valid

    def is_recording(self) -> bool:
        """Check if currently recording an episode."""
        with self._lock:
            return self._active

    def recorder_count(self) -> int:
        """Return number of registered recorders."""
        with self._lock:
            return len(self._recorders)
