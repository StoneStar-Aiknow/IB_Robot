"""ROS adapter for the X2 mirror voice (``mirror_voice`` package).

The voice node says a short phrase when the mirror session starts, is ready,
starts mirroring and goes back to WAIT. All the deciding is in
``mirror_voice.core`` and ``mirror_voice.speaker``; this node only
subscribes, makes the calls the speaker asks for and publishes ``~/status``.

Parameters (filled by launch, see ``VOICE_PARAMETERS``):

``voice_json``            the checked ``mirror_session.voice`` section
                          (``mirror_voice_contract.to_json``); not readable
                          or not enabled: ERROR and a clean exit
``tts_service``           ``SynthesizeSpeech`` (``voice_tts.service_name``)
``playback_service``      ``PlayAudioFile`` (``voice_tts.playback_service_name``)
``tts_timeout_sec``       per synthesis call
``playback_timeout_sec``  per played segment
``status_period_sec``     ``~/status`` period (default 1 s)

Interfaces:

* subscribes ``/mirror_session/state`` (reliable + transient_local, depth 1,
  as the operator); a message that is not JSON counts as malformed;
* calls ``tts_service`` and ``playback_service``, one call at a time;
* writes the WAV files in a directory of its own, removed on exit;
* ``~/status`` (``std_msgs/String``, JSON snapshot, every
  ``status_period_sec``).

Nothing else: no session service, no other topic. Every call is
asynchronous on one thread; ``speaker`` times calls out and ignores a late
answer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from embodied_common.mirror_session_contract import STATE_TOPIC
from embodied_common.mirror_voice_contract import NODE_NAME, VoiceSection, from_json
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech
from manipulation_execution.mirror_voice.speaker import Done, Speaker, SynthesisDone, WavDirectory

TICK_PERIOD_S = 0.1

# Every parameter the node declares, with its default. Launch may pass no other
# name and no other type: rclpy drops an override it has no declaration for
# without a warning, and refuses one of the wrong type at startup.
VOICE_PARAMETERS = {
    "voice_json": "",
    "tts_service": "/voice_tts/synthesize",
    "playback_service": "/voice_tts/play",
    "tts_timeout_sec": 15.0,
    "playback_timeout_sec": 300.0,
    "status_period_sec": 1.0,
}
_ROS_NAMES = ("tts_service", "playback_service")
_PERIODS = ("tts_timeout_sec", "playback_timeout_sec", "status_period_sec")


@dataclass(frozen=True)
class VoiceSettings:
    """The parameters, checked. No ROS."""

    section: VoiceSection
    tts_service: str
    playback_service: str
    tts_timeout_sec: float
    playback_timeout_sec: float
    status_period_sec: float

    @classmethod
    def from_parameters(cls, values: Mapping[str, Any]) -> VoiceSettings:
        section = from_json(values["voice_json"])
        if not section.enabled:
            raise ValueError("voice_json: the voice section is not enabled")
        for name in _ROS_NAMES:
            value = values[name]
            if not isinstance(value, str) or value.strip() != value or not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute ROS name")
        for name in _PERIODS:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0.0:
                raise ValueError(f"{name} must be positive")
        return cls(
            section=section,
            tts_service=values["tts_service"],
            playback_service=values["playback_service"],
            **{name: float(values[name]) for name in _PERIODS},
        )


class RosVoiceCalls:
    """``SpeakerCalls`` over two ROS service clients.

    Nothing here waits: each call is sent and its answer comes back through
    ``done``.
    """

    def __init__(self, *, tts, playback):
        self._tts = tts
        self._playback = playback

    def tts_ready(self) -> bool:
        return self._tts.service_is_ready()

    def playback_ready(self) -> bool:
        return self._playback.service_is_ready()

    def synthesize(self, text: str, done: SynthesisDone) -> None:
        request = SynthesizeSpeech.Request()
        request.text = text
        request.prompt_audio = []
        request.prompt_audio_format = ""
        request.prompt_text = ""

        def step(response) -> None:
            if not response.success:
                done(False, f"{response.error_code}: {response.message}", ())
                return
            done(True, "", list(response.audio_segments))

        self._then(self._tts.call_async(request), step, done, "synthesis")

    def play(self, path: str, done: Done) -> None:
        request = PlayAudioFile.Request()
        request.file_path = path

        def step(response) -> None:
            if not response.success:
                done(False, f"{response.error_code}: {response.message}")
                return
            done(True, "")

        self._then(self._playback.call_async(request), step, done, "playback")

    @staticmethod
    def _then(future, step: Callable[[Any], None], done: Callable, what: str) -> None:
        def answered(future) -> None:
            try:
                answer = future.result()
            except Exception as exc:  # noqa: BLE001 - the call failed: no answer
                done(None, f"{what} failed: {exc}")
                return
            if answer is None:
                done(None, f"{what}: no answer")
                return
            try:
                step(answer)
            except Exception as exc:  # noqa: BLE001 - an answer that cannot be read is no answer
                done(None, f"{what}: {exc}")

        future.add_done_callback(answered)


class MirrorVoiceNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        for name, default in VOICE_PARAMETERS.items():
            self.declare_parameter(name, default)
        self.speaker: Speaker | None = None
        self.files: WavDirectory | None = None
        try:
            settings = VoiceSettings.from_parameters(
                {name: self.get_parameter(name).value for name in VOICE_PARAMETERS}
            )
        except ValueError as exc:
            self.get_logger().error(f"mirror voice off, the mirror goes on without it: {exc}")
            return
        self._settings = settings
        self.files = WavDirectory()
        calls = RosVoiceCalls(
            tts=self.create_client(SynthesizeSpeech, settings.tts_service),
            playback=self.create_client(PlayAudioFile, settings.playback_service),
        )
        self.speaker = Speaker(
            settings.section,
            calls,
            self.files,
            tts_timeout_s=settings.tts_timeout_sec,
            playback_timeout_s=settings.playback_timeout_sec,
            log=self._log,
        )
        self.create_subscription(
            String,
            STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self.create_timer(TICK_PERIOD_S, self.speaker.tick)
        self.create_timer(settings.status_period_sec, self._publish_status)
        self.get_logger().info(
            f"mirror voice: {STATE_TOPIC} -> synthesis {settings.tts_service} "
            f"(timeout {settings.tts_timeout_sec:g} s), playback {settings.playback_service} "
            f"(timeout {settings.playback_timeout_sec:g} s); queue_max {settings.section.queue_max}; "
            f"WAV files in {self.files.path}"
        )

    def _log(self, level: str, message: str) -> None:
        # rclpy pins the severity of each call site; keep one call per level.
        if level == "error":
            self.get_logger().error(message)
        elif level == "warning":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _on_state(self, message: String) -> None:
        self.speaker.on_state(message.data)

    def _publish_status(self) -> None:
        status = self.speaker.status()
        status["tts_service"] = self._settings.tts_service
        status["playback_service"] = self._settings.playback_service
        self._status_publisher.publish(String(data=json.dumps(status, ensure_ascii=False, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    node = None
    try:
        node = MirrorVoiceNode()
        if node.speaker is not None:
            executor.add_node(node)
            executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if node.files is not None:
                node.files.remove()
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
