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
``clips_json``            empty (the default): the behaviour above. Otherwise
                          ``internvl_voice_contract.playback_json``: the
                          prerecorded mode below

Interfaces:

* subscribes ``/mirror_session/state`` (reliable + transient_local, depth 1,
  as the operator); a message that is not JSON counts as malformed;
* when ``voice_json`` carries an ``utterance_topic``: subscribes it
  (``ibrobot_msgs/VisualInteractionUtterance``, depth 10 as the publisher)
  and admits one visual interaction reply per round — speakable kind, not
  suppressed, no ASCII letters, the mirror still MIRRORING and the moment
  still inside ``latest_start_at``. What survives is queued with the state
  phrases, synthesized on demand, and its files are removed once spoken. The
  refusal counts land in ``~/status`` as ``utterance_refused``;
* calls ``tts_service`` and ``playback_service``, one call at a time;
* writes the WAV files in a directory of its own, removed on exit;
* ``~/status`` (``std_msgs/String``, JSON snapshot, every
  ``status_period_sec``).

Nothing else: no session service. Every call is asynchronous on one thread;
``speaker`` times calls out and ignores a late answer.

Prerecorded mode (``clips_json`` set; ``internvl_voice`` on): the clips are
checked once at startup (24 kHz mono 16 bit) in the first ``clip_dirs``
entry that has any, logged as 「用 <dir>，找到 n / N」; only the usable ones
are ever played. No synthesis client is made and nothing is synthesized: an
utterance whose text has no clip is refused as ``noclip``. Around every
playback the ASR is muted through ``asr_mute.AsrMute`` on the ``asr.topic``
(``/voice_control``, as the self introduction does), reading its state from
``/voice_status``; and ``~/status`` is also published the moment
``speaking`` changes.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import uninstall_signal_handlers
from std_msgs.msg import String

from embodied_common.internvl_voice_contract import parse_playback_json
from embodied_common.mirror_session_contract import STATE_MIRRORING, STATE_STALE_SEC, STATE_TOPIC
from embodied_common.mirror_voice_contract import NODE_NAME, VoiceSection, from_json, has_ascii_letters
from embodied_common.visual_interaction_contract import UTTERANCE_KINDS
from ibrobot_msgs.msg import VisualInteractionUtterance
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech
from manipulation_execution.mirror_voice.asr_mute import AsrMute
from manipulation_execution.mirror_voice.speaker import (
    ClipFiles,
    ClipReport,
    Done,
    Speaker,
    SynthesisDone,
    WavDirectory,
    load_clips,
)

TICK_PERIOD_S = 0.1

# Why an utterance can be refused at the door, counted in ``~/status`` as
# ``utterance_refused``. The order is the order the checks run in.
UTTERANCE_REFUSALS = ("kind", "suppressed", "empty", "ascii", "mirror", "late")
# Prerecorded mode adds one: no clip has the utterance's text.
CLIPS_REFUSALS = (*UTTERANCE_REFUSALS, "noclip")

# Fixed by voice_asr_node, as the voice bridge reads it.
ASR_STATUS_TOPIC = "/voice_status"

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
    "clips_json": "",
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
    # Empty: the synthesized phrases; else the prerecorded mode's playback_json.
    clips_json: str = ""

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
        clips_json = values.get("clips_json", "")
        return cls(
            section=section,
            tts_service=values["tts_service"],
            playback_service=values["playback_service"],
            **{name: float(values[name]) for name in _PERIODS},
            clips_json=clips_json.strip() if isinstance(clips_json, str) else "",
        )


class RosVoiceCalls:
    """``SpeakerCalls`` over two ROS service clients.

    Nothing here waits: each call is sent and its answer comes back through
    ``done``. Without a synthesis client (prerecorded mode) the synthesis
    service is never ready and a synthesis call fails.
    """

    def __init__(self, *, tts, playback):
        self._tts = tts
        self._playback = playback

    def tts_ready(self) -> bool:
        return self._tts is not None and self._tts.service_is_ready()

    def playback_ready(self) -> bool:
        return self._playback.service_is_ready()

    def synthesize(self, text: str, done: SynthesisDone) -> None:
        if self._tts is None:
            raise RuntimeError("prerecorded mode: no synthesis")
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
    def __init__(self, **kwargs):
        # Keyword-only forward: launch passes no arguments, and a test passes
        # parameter_overrides the same way the visual interaction node's tests do.
        super().__init__(NODE_NAME, **kwargs)
        for name, default in VOICE_PARAMETERS.items():
            self.declare_parameter(name, default)
        self.speaker: Speaker | None = None
        self.files: WavDirectory | ClipFiles | None = None
        self.mute: AsrMute | None = None
        try:
            settings = VoiceSettings.from_parameters(
                {name: self.get_parameter(name).value for name in VOICE_PARAMETERS}
            )
        except ValueError as exc:
            self.get_logger().error(f"mirror voice off, the mirror goes on without it: {exc}")
            return
        self._settings = settings
        # The current session state and when its message arrived, for the
        # utterance gate. The transition filter inside the speaker keeps its
        # own view; this one answers "is the mirror following someone now".
        self._state = ""
        self._state_at: float | None = None
        self._clip_report: ClipReport | None = None
        if settings.clips_json:
            self._start_clips(settings)
            return
        self.files = WavDirectory()
        self._utterance_refused = dict.fromkeys(UTTERANCE_REFUSALS, 0)
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
            utterance_gate=self._mirroring_now,
        )
        self.create_subscription(
            String,
            STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        if settings.section.utterance_topic:
            self.create_subscription(
                VisualInteractionUtterance,
                settings.section.utterance_topic,
                self._on_utterance,
                10,
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
        if settings.section.utterance_topic:
            self.get_logger().info(f"mirror voice: utterances from {settings.section.utterance_topic}")

    def _log(self, level: str, message: str) -> None:
        # rclpy pins the severity of each call site; keep one call per level.
        if level == "error":
            self.get_logger().error(message)
        elif level == "warning":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _start_clips(self, settings: VoiceSettings) -> None:
        """Prerecorded mode: check the clips, mute the ASR around them, never synthesize."""
        playback, warnings = parse_playback_json(settings.clips_json)
        for warning in warnings:
            self._log("warning", f"mirror voice: {warning}")
        clips, report = load_clips(playback)
        self._clip_report = report
        if report.directory is None:
            self._log(
                "error",
                f"mirror voice clips: 找到 0 / {report.total}, no clip WAV in {list(report.tried)}; "
                "the mirror goes on without voice (nothing is synthesized)",
            )
        else:
            self._log(
                "info" if report.found == report.total else "warning",
                f"mirror voice clips: 用 {report.directory}，找到 {report.found} / {report.total}",
            )
            for clip_id, reason in report.problems.items():
                self._log("warning", f"mirror voice clips: {clip_id}: {reason}; it is never said")
        for event in report.silent_events:
            self._log("error", f"mirror voice clips: no usable clip for {event}; that event is silent")

        asr = playback["asr"]
        publisher = None if asr["mode"] == "keep" else self.create_publisher(String, asr["topic"], 10)
        self.mute = AsrMute(
            asr["mode"],
            asr["resume_delay_sec"],
            lambda command: publisher.publish(String(data=command)),
            log=self._log,
        )
        if publisher is not None:
            self.create_subscription(String, ASR_STATUS_TOPIC, self._on_asr_status, 10)

        self.files = ClipFiles()
        self._utterance_refused = dict.fromkeys(CLIPS_REFUSALS, 0)
        # No synthesis client at all: nothing here can reach the synthesis service.
        calls = RosVoiceCalls(tts=None, playback=self.create_client(PlayAudioFile, settings.playback_service))
        self.speaker = Speaker(
            settings.section,
            calls,
            self.files,
            tts_timeout_s=settings.tts_timeout_sec,
            playback_timeout_s=settings.playback_timeout_sec,
            log=self._log,
            utterance_gate=self._mirroring_now,
            clips=clips,
            on_playback_start=self.mute.playback_started,
            on_playback_end=self.mute.playback_ended,
            on_speaking=self._on_speaking,
        )
        self.create_subscription(
            String,
            STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        if settings.section.utterance_topic:
            self.create_subscription(
                VisualInteractionUtterance,
                settings.section.utterance_topic,
                self._on_utterance,
                10,
            )
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self.create_timer(TICK_PERIOD_S, self._clips_tick)
        self.create_timer(settings.status_period_sec, self._publish_status)
        self.mute.activate()
        self.get_logger().info(
            f"mirror voice: {STATE_TOPIC} -> prerecorded clips, playback {settings.playback_service} "
            f"(timeout {settings.playback_timeout_sec:g} s), no synthesis; queue_max {settings.section.queue_max}; "
            f"ASR {asr['mode']} on {asr['topic']} (resume {asr['resume_delay_sec']:g} s after a clip)"
        )
        if settings.section.utterance_topic:
            self.get_logger().info(f"mirror voice: utterances from {settings.section.utterance_topic}, clips only")

    def _clips_tick(self) -> None:
        self.speaker.tick()
        self.mute.tick()

    def _on_asr_status(self, message: String) -> None:
        self.mute.on_asr_status(message.data)

    def _on_speaking(self, text: str | None) -> None:
        # Readers of ~/status (the look) learn at once that the voice is busy or free.
        if getattr(self, "_status_publisher", None) is not None:
            self._publish_status()

    def _on_state(self, message: String) -> None:
        # The current state and its arrival are recorded beside the transition
        # the speaker reads. A message that is not JSON leaves both alone: the
        # recorded moment keeps aging, so the gate closes on staleness rather
        # than on a parse.
        try:
            payload = json.loads(message.data)
        except ValueError:
            payload = None
        state = payload.get("state") if isinstance(payload, dict) else None
        if isinstance(state, str):
            self._state = state
            self._state_at = self._now()
        self.speaker.on_state(message.data)

    def _on_utterance(self, message: VisualInteractionUtterance) -> None:
        """One visual interaction round: admit its reply, or refuse it with a count.

        The session has already decided the reply is worth saying. This is the
        door: the mirror must still be MIRRORING (it may have stopped since
        the round began), the text must still be one ZipVoice can speak, and
        the moment must still be inside ``latest_start_at``. What survives is
        queued behind the state phrases, and the speaker asks the gate once
        more when it would actually start synthesizing.
        """
        if message.kind not in UTTERANCE_KINDS:
            self._refuse("kind")
            return
        if message.suppress_reason:
            self._refuse("suppressed")
            return
        text = message.text.strip()
        if not text:
            self._refuse("empty")
            return
        if has_ascii_letters(text):
            self._refuse("ascii", text)
            return
        if not self._mirroring_now():
            self._refuse("mirror", text)
            return
        latest = message.latest_start_at.sec + message.latest_start_at.nanosec * 1e-9
        if latest != 0.0 and self._now() > latest:
            self._refuse("late", text)
            return
        if self._clip_report is not None and not self.speaker.has_clip(text):
            # Prerecorded mode says only what it has a clip of; never synthesized.
            self._refuse("noclip", text)
            return
        self.speaker.on_utterance(text)

    def _refuse(self, reason: str, text: str = "") -> None:
        self._utterance_refused[reason] += 1
        # A refusal with text dropped words someone could have heard; the
        # rest are the round-by-round quiet the topic reports anyway.
        if text:
            self._log("info", f"utterance refused ({reason}): {text!r}")

    def _mirroring_now(self) -> bool:
        # Arrival time, not the publisher's clock: no cross-host agreement to
        # lean on. Two missed heartbeats count as unknown, and unknown gates
        # the utterance off — the conservative side.
        return (
            self._state == STATE_MIRRORING
            and self._state_at is not None
            and self._now() - self._state_at <= STATE_STALE_SEC
        )

    def _now(self) -> float:
        # The ROS clock, the same one the visual interaction node stamps its
        # rounds by on this host; a monotonic clock would not compare against
        # latest_start_at at all.
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_status(self) -> None:
        status = self.speaker.status()
        status["tts_service"] = self._settings.tts_service
        status["playback_service"] = self._settings.playback_service
        status["mirror_state"] = self._state
        status["utterance_refused"] = dict(self._utterance_refused)
        report = self._clip_report
        if report is not None:
            status["clips"] = {"dir": report.directory, "found": report.found, "total": report.total}
            status["noclip"] = self._utterance_refused["noclip"]
            status["asr"] = self.mute.status()
        self._status_publisher.publish(String(data=json.dumps(status, ensure_ascii=False, sort_keys=True)))


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def _keep_context_past_signals() -> None:
    """SIGINT and SIGTERM end the spin but leave the context up, as in mirror_session_node.

    rclpy's own handlers shut the context down inside the signal, before
    ``main`` can publish, and without calling ``Context.on_shutdown`` callbacks
    (which on Humble run after the context is gone anyway). The ASR's last
    ``start`` needs a working publisher.
    """
    uninstall_signal_handlers()
    signal.signal(signal.SIGTERM, _interrupt)


def main(args=None) -> None:
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    node = None
    try:
        node = MirrorVoiceNode()
        if node.mute is not None:  # prerecorded mode only: the baseline keeps rclpy's handlers
            _keep_context_past_signals()
        if node.speaker is not None:
            executor.add_node(node)
            executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if node.mute is not None:
                # Best effort: another node's rclpy.shutdown() can still beat it.
                node.mute.shutdown()
            if node.files is not None:
                node.files.remove()
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
