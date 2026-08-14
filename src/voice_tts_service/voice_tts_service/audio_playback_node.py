"""ROS 2 service for playing a WAV file on the local speaker."""

from __future__ import annotations

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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

        self._player = AudioFilePlayer(
            AudioPlaybackConfig(
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
        )
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
            path = self._player.play(request.file_path)
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
