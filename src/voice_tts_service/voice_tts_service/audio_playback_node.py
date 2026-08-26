"""ROS 2 service for playing a WAV file on the local speaker."""

from __future__ import annotations

import time
import wave

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import PlayAudioFile

from .audio_playback import AudioFilePlayer, AudioPlaybackConfig, AudioPlaybackError


class AudioPlaybackNode(Node):
    """Expose synchronous, serialized local WAV playback."""

    def __init__(self) -> None:
        super().__init__("voice_tts_audio_player")
        self.declare_parameter("service_name", "/voice_tts/play")
        self.declare_parameter("timeout_sec", 300.0)
        self.declare_parameter("audio_topic", "/audio/play")
        self.declare_parameter("playback_sample_rate", 24000)
        self.declare_parameter("playback_channels", 1)

        self._player = AudioFilePlayer(
            AudioPlaybackConfig(
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
        )
        self._audio_topic = str(self.get_parameter("audio_topic").value)
        self._playback_sample_rate = int(self.get_parameter("playback_sample_rate").value)
        self._playback_channels = int(self.get_parameter("playback_channels").value)
        from audio_common_msgs.msg import AudioData

        self._audio_pub = self.create_publisher(AudioData, self._audio_topic, 10)
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

        path = self._player.validate_pcm_format(
            file_path,
            sample_rate=self._playback_sample_rate,
            channels=self._playback_channels,
        )
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            from audio_common_msgs.msg import AudioData

            chunk_frames = max(1, rate // 20)
            while True:
                data = wav.readframes(chunk_frames)
                if not data:
                    break
                duration = len(data) / (rate * channels * 2)
                remaining = deadline - time.monotonic()
                if remaining < duration:
                    raise AudioPlaybackError(
                        "PLAYBACK_TIMEOUT",
                        f"audio playback exceeded {timeout_sec:g} seconds",
                    )
                message = AudioData()
                message.data = list(data)
                self._audio_pub.publish(message)
                wait_for_all_acked = getattr(self._audio_pub, "wait_for_all_acked", None)
                if wait_for_all_acked is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not wait_for_all_acked(Duration(seconds=remaining)):
                        raise AudioPlaybackError(
                            "PLAYBACK_TIMEOUT",
                            f"audio playback acknowledgement exceeded {timeout_sec:g} seconds",
                        )
                time.sleep(duration)
        return str(path)


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
