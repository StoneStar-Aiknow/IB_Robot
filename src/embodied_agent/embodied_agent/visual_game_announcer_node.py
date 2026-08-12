"""Request terminal visual-game announcements from the shared TTS service."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node

from embodied_agent.visual_game_qos import visual_game_event_consumer_qos
from embodied_common.visual_game_contracts import get_visual_game_announcement
from ibrobot_msgs.msg import VisualGameEvent
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech

TTS_SERVICE_UNAVAILABLE = "TTS_SERVICE_UNAVAILABLE"
PLAYBACK_SERVICE_UNAVAILABLE = "PLAYBACK_SERVICE_UNAVAILABLE"
SERVICE_UNAVAILABLE_GRACE_SEC = 3.0


def announcement_text(event: VisualGameEvent) -> str | None:
    """Return the handler-owned announcement for one terminal game event."""
    if not event.announce or not event.handler.strip():
        return None
    try:
        result = json.loads(event.result_json) if event.result_json else {"scene_summary": event.scene_summary.strip()}
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    try:
        return get_visual_game_announcement(
            event.handler.strip(),
            state=event.state,
            success=event.success,
            error_code=event.error_code,
            result=result,
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class _Announcement:
    request_id: str
    text: str
    attempts: int = 0
    execution_id: str = ""
    game_name: str = ""
    service_unavailable_since_monotonic: float | None = None

    @property
    def deduplication_key(self) -> str:
        return self.execution_id or self.request_id


class VisualGameAnnouncerNode(Node):
    """Deduplicate terminal events, synthesize speech, and play it locally."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("visual_game_announcer_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("event_topic", "/embodied/visual_game_events")
        self.declare_parameter("tts_service", "/voice_tts/synthesize")
        self.declare_parameter("playback_service", "/voice_tts/play")
        self.declare_parameter("announcement_queue_capacity", 8)
        self.declare_parameter("deduplication_capacity", 128)
        self.declare_parameter("max_delivery_attempts", 3)
        self.declare_parameter("tts_timeout_sec", 15.0)
        self.declare_parameter("playback_timeout_sec", 300.0)
        self.declare_parameter("debug_tracing", False)

        event_topic = self.get_parameter("event_topic").value
        tts_service = self.get_parameter("tts_service").value
        playback_service = self.get_parameter("playback_service").value
        self._announcement_queue_capacity = max(1, int(self.get_parameter("announcement_queue_capacity").value))
        self._deduplication_capacity = max(1, int(self.get_parameter("deduplication_capacity").value))
        self._max_delivery_attempts = max(1, int(self.get_parameter("max_delivery_attempts").value))
        self._tts_timeout_sec = max(0.1, float(self.get_parameter("tts_timeout_sec").value))
        self._playback_timeout_sec = max(0.1, float(self.get_parameter("playback_timeout_sec").value))
        self._debug = bool(self.get_parameter("debug_tracing").value)
        self._completed_announcements: OrderedDict[str, None] = OrderedDict()
        self._active_announcements: set[str] = set()
        self._pending_announcements: OrderedDict[str, _Announcement] = OrderedDict()
        self._tts_inflight = False
        self._tts_deadline_monotonic = 0.0
        self._tts_future = None
        self._tts_announcement = None
        self._tts_generation = 0
        self._playback_inflight = False
        self._playback_deadline_monotonic = 0.0
        self._playback_future = None
        self._playback_generation = 0
        self._playback_announcement = None
        self._playback_paths: list[Path] = []
        self._playback_pauses_sec: list[float] = []
        self._playback_index = 0
        self._playback_next_ready_monotonic = 0.0
        self._temp_dir: Path | None = None

        self._tts_client = self.create_client(SynthesizeSpeech, tts_service)
        self._playback_client = self.create_client(PlayAudioFile, playback_service)
        self.create_subscription(
            VisualGameEvent,
            event_topic,
            self._handle_event,
            visual_game_event_consumer_qos(depth=self._deduplication_capacity * 2),
        )
        self.create_timer(0.5, self._service_announcement_queues)
        self.get_logger().info(
            "[embodied-debug] visual game announcer ready: "
            f"event_topic={event_topic}, tts_service={tts_service}, playback_service={playback_service}"
        )

    def _handle_event(self, event: VisualGameEvent) -> str | None:
        request_id = event.request_id.strip()
        execution_id = event.execution_id.strip()
        game_name = event.game_name.strip()
        if event.state == "accepted" and request_id and game_name:
            self._supersede_unplayed_announcements(game_name, keep_execution_id=execution_id)
            self._dispatch_pending_announcement()
            return None

        text = announcement_text(event)
        announcement = _Announcement(
            request_id=request_id,
            text=text or "",
            execution_id=execution_id,
            game_name=game_name,
        )
        if (
            text is None
            or not request_id
            or announcement.deduplication_key in self._active_announcements
            or announcement.deduplication_key in self._completed_announcements
        ):
            return None
        if len(self._pending_announcements) >= self._announcement_queue_capacity:
            self.get_logger().error(f"visual game announcement queue is full: {request_id}")
            return "ANNOUNCEMENT_QUEUE_FULL"
        self._active_announcements.add(announcement.deduplication_key)
        self._pending_announcements[announcement.deduplication_key] = announcement
        self._dispatch_pending_announcement()
        return None

    def _supersede_unplayed_announcements(self, game_name: str, *, keep_execution_id: str) -> None:
        for key, announcement in tuple(self._pending_announcements.items()):
            if announcement.game_name != game_name or announcement.execution_id == keep_execution_id:
                continue
            self._pending_announcements.pop(key, None)
            self._complete_announcement(announcement)
            self.get_logger().info(
                f"superseded stale visual game announcement before playback: {announcement.request_id}"
            )

        announcement = self._tts_announcement
        if (
            announcement is None
            or announcement.game_name != game_name
            or announcement.execution_id == keep_execution_id
        ):
            return
        self._tts_generation += 1
        self._tts_inflight = False
        if self._tts_future is not None and not self._tts_future.done():
            with contextlib.suppress(Exception):
                self._tts_future.cancel()
        self._tts_future = None
        self._tts_announcement = None
        self._complete_announcement(announcement)
        self.get_logger().info(f"superseded stale visual game TTS request before playback: {announcement.request_id}")

    def _service_announcement_queues(self) -> None:
        self._expire_tts_request()
        self._expire_playback_request()
        self._dispatch_playback()
        self._dispatch_pending_announcement()

    def _expire_tts_request(self) -> None:
        if not self._tts_inflight or self._tts_deadline_monotonic > time.monotonic():
            return
        announcement = self._tts_announcement
        self._tts_generation += 1
        self._tts_inflight = False
        self._tts_future = None
        self._tts_announcement = None
        if announcement is not None:
            self._retry_or_finish(announcement, f"TTS request timed out after {self._tts_timeout_sec:.1f}s")

    def _retry_or_finish(self, announcement: _Announcement, error: str | None = None) -> None:
        if error is None:
            self._complete_announcement(announcement)
            return
        if announcement.attempts < self._max_delivery_attempts:
            self.get_logger().warning(
                f"visual game announcement attempt {announcement.attempts} failed for "
                f"{announcement.request_id}; retrying: {error}"
            )
            self._pending_announcements[announcement.deduplication_key] = announcement
            return
        self._complete_announcement(announcement)
        self.get_logger().error(
            f"visual game announcement failed after {announcement.attempts} attempts for "
            f"{announcement.request_id}: {error}"
        )

    def _finish_playback(self, announcement: _Announcement, error: str | None = None) -> None:
        self._cleanup_playback_files()
        self._playback_inflight = False
        self._playback_future = None
        self._playback_announcement = None
        if error is None:
            self._complete_announcement(announcement)
        else:
            self._complete_announcement(announcement)
            self.get_logger().error(f"visual game announcement playback failed for {announcement.request_id}: {error}")
        self._dispatch_pending_announcement()

    def _cleanup_playback_files(self) -> None:
        for path in self._playback_paths:
            with contextlib.suppress(OSError):
                path.unlink()
        self._playback_paths.clear()
        self._playback_pauses_sec.clear()
        self._playback_index = 0
        self._playback_next_ready_monotonic = 0.0
        if self._temp_dir is not None:
            with contextlib.suppress(OSError):
                self._temp_dir.rmdir()
            self._temp_dir = None

    def _prepare_playback_files(self, announcement: _Announcement, audio_segments) -> list[Path]:
        segments = sorted(audio_segments, key=lambda segment: int(segment.index))
        if not segments:
            raise ValueError("TTS returned no audio segments")
        if [int(segment.index) for segment in segments] != list(range(len(segments))):
            raise ValueError("TTS audio segment indexes must be contiguous from zero")
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ibrobot-visual-game-audio-"))
        file_stem = hashlib.sha256(announcement.deduplication_key.encode()).hexdigest()
        paths = []
        pauses_sec = []
        try:
            for segment in segments:
                if str(segment.audio_format).strip().lower() != "wav_pcm_s16le":
                    raise ValueError(f"unsupported TTS audio format: {segment.audio_format}")
                audio_data = bytes(segment.audio_data)
                if not audio_data.startswith(b"RIFF") or b"WAVE" not in audio_data[:16]:
                    raise ValueError(f"TTS audio segment {segment.index} is not a WAV file")
                path = self._temp_dir / f"{file_stem}-{int(segment.index):04d}.wav"
                path.write_bytes(audio_data)
                paths.append(path)
                pauses_sec.append(max(0.0, int(segment.pause_after_ms) / 1000.0))
        except (OSError, ValueError):
            for path in paths:
                with contextlib.suppress(OSError):
                    path.unlink()
            with contextlib.suppress(OSError):
                self._temp_dir.rmdir()
            self._temp_dir = None
            raise
        self._playback_pauses_sec = pauses_sec
        return paths

    def _dispatch_playback(self) -> None:
        if self._playback_inflight or self._playback_announcement is None:
            return
        if self._playback_index >= len(self._playback_paths):
            self._finish_playback(self._playback_announcement)
            return
        if self._playback_next_ready_monotonic > time.monotonic():
            return
        if not self._playback_client.service_is_ready():
            self._finish_playback(self._playback_announcement, PLAYBACK_SERVICE_UNAVAILABLE)
            return
        request = PlayAudioFile.Request()
        request.file_path = str(self._playback_paths[self._playback_index])
        self._playback_inflight = True
        self._playback_generation += 1
        generation = self._playback_generation
        self._playback_deadline_monotonic = time.monotonic() + self._playback_timeout_sec
        try:
            future = self._playback_client.call_async(request)
        except Exception as exc:
            self._playback_inflight = False
            self._finish_playback(self._playback_announcement, f"playback request failed: {exc}")
            return
        self._playback_future = future
        future.add_done_callback(
            lambda completed, request_generation=generation: self._handle_playback_response(
                completed, request_generation
            )
        )

    def _handle_playback_response(self, future, generation: int | None = None) -> None:
        if generation is not None and generation != self._playback_generation:
            return
        if generation is not None and self._playback_future is not None and future is not self._playback_future:
            return
        self._playback_inflight = False
        self._playback_future = None
        announcement = self._playback_announcement
        if announcement is None:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._finish_playback(announcement, f"playback request failed: {exc}")
            return
        if not response.success:
            self._finish_playback(announcement, f"playback failed: {response.error_code}: {response.message}")
            return
        completed_path = self._playback_paths[self._playback_index]
        with contextlib.suppress(OSError):
            completed_path.unlink()
        self._playback_next_ready_monotonic = time.monotonic() + self._playback_pauses_sec[self._playback_index]
        self._playback_index += 1
        self._dispatch_playback()

    def _expire_playback_request(self) -> None:
        if not self._playback_inflight or self._playback_deadline_monotonic > time.monotonic():
            return
        announcement = self._playback_announcement
        self._playback_generation += 1
        self._playback_inflight = False
        self._playback_future = None
        if announcement is not None:
            self._finish_playback(
                announcement,
                f"playback request timed out after {self._playback_timeout_sec:.1f}s",
            )

    def _complete_announcement(self, announcement: _Announcement) -> None:
        self._active_announcements.discard(announcement.deduplication_key)
        self._completed_announcements[announcement.deduplication_key] = None
        while len(self._completed_announcements) > self._deduplication_capacity:
            self._completed_announcements.popitem(last=False)

    def _dispatch_pending_announcement(self) -> None:
        if self._tts_inflight or self._playback_inflight or self._playback_announcement is not None:
            return
        if not self._pending_announcements:
            return
        now = time.monotonic()
        tts_ready = self._tts_client.service_is_ready()
        playback_ready = self._playback_client.service_is_ready()
        if not tts_ready or not playback_ready:
            code = TTS_SERVICE_UNAVAILABLE if not tts_ready else PLAYBACK_SERVICE_UNAVAILABLE
            for key in tuple(self._pending_announcements):
                announcement = self._pending_announcements[key]
                unavailable_since = announcement.service_unavailable_since_monotonic
                if unavailable_since is None:
                    self._pending_announcements[key] = _Announcement(
                        request_id=announcement.request_id,
                        text=announcement.text,
                        attempts=announcement.attempts,
                        execution_id=announcement.execution_id,
                        game_name=announcement.game_name,
                        service_unavailable_since_monotonic=now,
                    )
                    self.get_logger().warning(
                        f"[{code}] deferring visual game announcement for up to "
                        f"{SERVICE_UNAVAILABLE_GRACE_SEC:.1f}s: {announcement.request_id}"
                    )
                    announcement = self._pending_announcements[key]
                unavailable_since = announcement.service_unavailable_since_monotonic
                if unavailable_since is not None and now - unavailable_since >= SERVICE_UNAVAILABLE_GRACE_SEC:
                    self.get_logger().warning(
                        f"skipping visual game announcement after service wait timed out: {announcement.request_id}"
                    )
                    self._pending_announcements.pop(key, None)
                    self._complete_announcement(announcement)
            return
        for key, announcement in tuple(self._pending_announcements.items()):
            unavailable_since = announcement.service_unavailable_since_monotonic
            if unavailable_since is None:
                continue
            if now - unavailable_since >= SERVICE_UNAVAILABLE_GRACE_SEC:
                self.get_logger().warning(
                    f"skipping visual game announcement after service wait timed out: {announcement.request_id}"
                )
                self._pending_announcements.pop(key, None)
                self._complete_announcement(announcement)
                continue
            self._pending_announcements[key] = _Announcement(
                request_id=announcement.request_id,
                text=announcement.text,
                attempts=announcement.attempts,
                execution_id=announcement.execution_id,
                game_name=announcement.game_name,
            )
        if not self._pending_announcements:
            return
        _deduplication_key, pending = self._pending_announcements.popitem(last=False)
        announcement = _Announcement(
            request_id=pending.request_id,
            text=pending.text,
            attempts=pending.attempts + 1,
            execution_id=pending.execution_id,
            game_name=pending.game_name,
        )
        request = SynthesizeSpeech.Request()
        request.text = announcement.text
        request.prompt_audio = []
        request.prompt_audio_format = ""
        request.prompt_text = ""
        self._tts_inflight = True
        self._tts_generation += 1
        generation = self._tts_generation
        self._tts_announcement = announcement
        self._tts_deadline_monotonic = time.monotonic() + self._tts_timeout_sec
        try:
            future = self._tts_client.call_async(request)
        except Exception as exc:
            self._tts_inflight = False
            self._tts_announcement = None
            self._retry_or_finish(announcement, f"TTS request failed: {exc}")
            return
        self._tts_future = future
        future.add_done_callback(
            lambda completed, item=announcement, request_generation=generation: self._handle_tts_response(
                item, completed, request_generation
            )
        )

    def _handle_tts_response(self, announcement: _Announcement, future, generation: int | None = None) -> None:
        if generation is not None and generation != self._tts_generation:
            return
        if generation is not None and self._tts_future is not None and future is not self._tts_future:
            return
        self._tts_inflight = False
        self._tts_future = None
        self._tts_announcement = None
        try:
            response = future.result()
        except Exception as exc:
            self._retry_or_finish(announcement, f"TTS request failed: {exc}")
            self._dispatch_pending_announcement()
            return
        if not response.success:
            self._retry_or_finish(announcement, f"TTS failed: {response.error_code}: {response.message}")
            self._dispatch_pending_announcement()
            return
        try:
            self._playback_paths = self._prepare_playback_files(announcement, response.audio_segments)
        except (OSError, ValueError) as exc:
            self._retry_or_finish(announcement, f"invalid synthesized audio: {exc}")
            self._dispatch_pending_announcement()
            return
        self._playback_announcement = announcement
        self._playback_index = 0
        self._dispatch_playback()

    def close(self) -> None:
        """Cancel in-flight TTS requests and release async announcer state.

        The node's executor is owned by the caller (``rclpy.spin``), so this
        only drains this node's own async state; ROS resources are released by
        :meth:`destroy_node`. Bumping ``_tts_generation`` first invalidates any
        in-flight ``done_callback`` so a late cancel/failure cannot rearm state.
        """
        self._tts_generation += 1
        self._tts_inflight = False
        if self._tts_future is not None and not self._tts_future.done():
            with contextlib.suppress(Exception):
                self._tts_future.cancel()
        self._tts_future = None
        self._tts_announcement = None
        self._playback_generation += 1
        self._playback_inflight = False
        if self._playback_future is not None and not self._playback_future.done():
            with contextlib.suppress(Exception):
                self._playback_future.cancel()
        self._playback_future = None
        self._playback_announcement = None
        self._cleanup_playback_files()
        self._pending_announcements.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualGameAnnouncerNode()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
