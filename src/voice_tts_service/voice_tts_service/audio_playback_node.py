"""ROS 2 service for playing a WAV file on the local speaker."""

from __future__ import annotations

import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import PlayAudioFile

from .audio_playback import AudioFilePlayer, AudioPlaybackConfig, AudioPlaybackError

AUDIO_DATA_TYPE = "audio_common_msgs/msg/AudioData"
AUDIO_DATA_STAMPED_TYPE = "audio_common_msgs/msg/AudioDataStamped"
SUPPORTED_AUDIO_MESSAGE_TYPES = (AUDIO_DATA_TYPE, AUDIO_DATA_STAMPED_TYPE)
#: Publication granularity bounds, matching the speaker interface's write_chunk_ms bounds.
MIN_CHUNK_MS = 20
MAX_CHUNK_MS = 200
#: Lead kept ahead of the sample clock, matching the speaker interface's prebuffer_ms bounds.
MIN_PREBUFFER_MS = 0
MAX_PREBUFFER_MS = 2000


def _audio_message_class(message_type: str):
    """Resolve the configured playback message type; anything else is a configuration error."""

    from audio_common_msgs.msg import AudioData, AudioDataStamped

    if message_type == AUDIO_DATA_TYPE:
        return AudioData
    if message_type == AUDIO_DATA_STAMPED_TYPE:
        return AudioDataStamped
    raise ValueError(f"audio_message_type must be one of {SUPPORTED_AUDIO_MESSAGE_TYPES}, got {message_type!r}")


class AudioPlaybackNode(Node):
    """Expose synchronous, serialized local WAV playback."""

    def __init__(self) -> None:
        super().__init__("voice_tts_audio_player")
        self.declare_parameter("service_name", "/voice_tts/play")
        self.declare_parameter("timeout_sec", 300.0)
        self.declare_parameter("audio_topic", "/audio/play")
        self.declare_parameter("playback_sample_rate", 24000)
        self.declare_parameter("playback_channels", 1)
        # Off by default: a rate mismatch stays an error unless the robot config opts in.
        self.declare_parameter("resample_to_playback_rate", False)
        # AudioDataStamped for runtime-owned speakers (e.g. the aimdk audio.play interface).
        self.declare_parameter("audio_message_type", AUDIO_DATA_TYPE)
        # How the speaker wants PCM written; robot launches take both from its audio.play interface.
        self.declare_parameter("chunk_ms", 50)
        self.declare_parameter("prebuffer_ms", 0)

        self._player = AudioFilePlayer(
            AudioPlaybackConfig(
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
        )
        self._audio_topic = str(self.get_parameter("audio_topic").value)
        self._playback_sample_rate = int(self.get_parameter("playback_sample_rate").value)
        self._playback_channels = int(self.get_parameter("playback_channels").value)
        self._resample = bool(self.get_parameter("resample_to_playback_rate").value)
        # Bounded here as well as in the interface schema: a standalone launch has no description
        # to reject a value that would publish a DDS message per millisecond.
        self._chunk_ms = int(self.get_parameter("chunk_ms").value)
        if not MIN_CHUNK_MS <= self._chunk_ms <= MAX_CHUNK_MS:
            raise ValueError(f"chunk_ms must be between {MIN_CHUNK_MS} and {MAX_CHUNK_MS}, got {self._chunk_ms}")
        self._prebuffer_ms = int(self.get_parameter("prebuffer_ms").value)
        if not MIN_PREBUFFER_MS <= self._prebuffer_ms <= MAX_PREBUFFER_MS:
            raise ValueError(
                f"prebuffer_ms must be between {MIN_PREBUFFER_MS} and {MAX_PREBUFFER_MS}, got {self._prebuffer_ms}"
            )
        self._audio_message_type = str(self.get_parameter("audio_message_type").value)
        self._audio_message_class = _audio_message_class(self._audio_message_type)

        self._audio_pub = self.create_publisher(self._audio_message_class, self._audio_topic, 10)
        service_name = str(self.get_parameter("service_name").value)
        self._service = self.create_service(
            PlayAudioFile,
            service_name,
            self._play,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(f"Audio playback service ready at {service_name!r}")

    def _play(self, request, response):
        try:
            path = self._publish_audio(request.file_path)
            response.success = True
            response.error_code = ""
            response.message = f"played audio file: {path}"
        except AudioPlaybackError as exc:
            response.success = False
            response.error_code = exc.code
            response.message = str(exc)
            self.get_logger().error(f"Audio playback failed [{exc.code}]: {exc}")
        except Exception as exc:  # Preserve a stable response for unexpected runtime failures.
            response.success = False
            response.error_code = "INTERNAL_ERROR"
            response.message = str(exc)
            self.get_logger().error(f"Unexpected audio playback failure: {exc}")
        return response

    def _publish_audio(self, file_path: str) -> str:
        """Publish PCM chunks for audio_play instead of opening ALSA here."""
        timeout_sec = self._player.config.timeout_sec
        if timeout_sec <= 0:
            raise AudioPlaybackError("INVALID_TIMEOUT", "audio playback timeout must be positive")
        deadline = time.monotonic() + timeout_sec

        while self._audio_pub.get_subscription_count() <= 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AudioPlaybackError(
                    "PLAYER_NOT_READY",
                    f"audio playback subscriber did not become ready within {timeout_sec:g} seconds",
                )
            time.sleep(min(0.01, remaining))

        pcm = self._player.load_pcm_for_playback(
            file_path,
            sample_rate=self._playback_sample_rate,
            channels=self._playback_channels,
            resample=self._resample,
        )
        # Pace on the playback contract's rate: after resampling the WAV header rate no longer applies.
        rate = self._playback_sample_rate
        frame_bytes = self._playback_channels * 2
        chunk_bytes = max(1, rate * self._chunk_ms // 1000) * frame_bytes
        lead_frames = rate * self._prebuffer_ms // 1000
        playback_start = time.monotonic()
        published_frames = 0
        for offset in range(0, len(pcm), chunk_bytes):
            data = pcm[offset : offset + chunk_bytes]
            published_frames += len(data) // frame_bytes
            chunk_end = playback_start + published_frames / rate
            if chunk_end > deadline or time.monotonic() >= deadline:
                raise AudioPlaybackError(
                    "PLAYBACK_TIMEOUT",
                    f"audio playback exceeded {timeout_sec:g} seconds",
                )
            self._audio_pub.publish(self._make_message(data))
            # A player that has to wait for each write can report a PLAYING/STOPPED pair around
            # it (the X2's hal_audio did, once per chunk, audible as stutter), so the stream keeps
            # prebuffer_ms ahead of the sample clock for its whole length, not only at the start.
            # DDS acknowledgement can take an entire PCM chunk period; waiting for it here would
            # starve the player between chunks: measured against the real GStreamer sink,
            # a 2.6 s clip did not finish in 90 s. Absolute sample-clock deadlines also avoid
            # accumulating serialization and publish overhead into the sound duration.
            self._sleep_until(playback_start + max(0, published_frames - lead_frames) / rate, deadline, timeout_sec)
        # Return once the last sample's playback time is reached, not when it was written: callers
        # take the return as "spoken" (the voice bridge resumes listening on it), and with a lead
        # the last chunk is written up to prebuffer_ms early.
        self._sleep_until(playback_start + published_frames / rate, deadline, timeout_sec)
        # Retain reliable-delivery confirmation without putting ACK latency
        # between PCM chunks. This is not a hardware playback-complete event.
        wait_for_all_acked = getattr(self._audio_pub, "wait_for_all_acked", None)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (wait_for_all_acked is not None and not wait_for_all_acked(Duration(seconds=remaining))):
            raise AudioPlaybackError(
                "PLAYBACK_TIMEOUT",
                f"audio playback acknowledgement exceeded {timeout_sec:g} seconds",
            )
        return file_path.strip()

    @staticmethod
    def _sleep_until(when: float, deadline: float, timeout_sec: float) -> None:
        if deadline - time.monotonic() <= 0:
            raise AudioPlaybackError("PLAYBACK_TIMEOUT", f"audio playback exceeded {timeout_sec:g} seconds")
        delay = when - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _make_message(self, chunk: bytes):
        message = self._audio_message_class()
        if self._audio_message_type == AUDIO_DATA_STAMPED_TYPE:
            message.header.stamp = self.get_clock().now().to_msg()
            message.audio.data = list(chunk)
        else:
            message.data = list(chunk)
        return message


def main(args=None):
    rclpy.init(args=args)
    node = AudioPlaybackNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
