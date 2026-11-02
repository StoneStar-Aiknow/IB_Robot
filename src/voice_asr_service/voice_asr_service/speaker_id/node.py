"""Publish one speaker embedding per utterance, for the agent's addressee gate."""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import SpeakerEmbedding, SpeechAudioChunk
from voice_asr_service.speaker_id.defaults import SPEAKER_ID_DEFAULTS
from voice_asr_service.speaker_id.embedder import SpeakerEmbedder
from voice_asr_service.speaker_id.utterance import UtteranceAssembler

#: Consecutive inference failures after which the deployment is loaded again.
RELOAD_AFTER_FAILURES = 3
#: Least time between two reload attempts while the model keeps failing.
RELOAD_INTERVAL_SEC = 30.0


class SpeakerIdNode(Node):
    """Embed every utterance on the chunk topic and publish the vector.

    Runs beside ``voice_asr_node`` on the same ``SpeechAudioChunk`` stream
    rather than inside it, so the vector is published at the end of speech,
    while the transcript is still waiting for VAD silence plus the tail
    padding, and extracting here costs the ASR no latency at all. The consumer
    pairs the two by segment (``source_id``, ``segment_sequence``).

    A failure of the model on one utterance stays with that utterance: it is
    logged and nothing is published for it, so the gate judges its transcript
    without a voiceprint once its grace period runs out (fail-open), exactly as
    if the voiceprint had been lost. After ``RELOAD_AFTER_FAILURES`` failures
    in a row the deployment is loaded again; while that keeps failing, the node
    stays up publishing nothing and retries every ``RELOAD_INTERVAL_SEC``.

    Nothing is persisted. A vector only ever exists in this process's memory and
    in the message it is published in; voiceprints are biometric data.
    """

    def __init__(self) -> None:
        super().__init__("speaker_id_node")
        for name, default in SPEAKER_ID_DEFAULTS.items():
            self.declare_parameter(name, default)
        values = {name: self.get_parameter(name).value for name in SPEAKER_ID_DEFAULTS}

        self._values = values
        self._min_speech_seconds = float(values["min_speech_seconds"])
        self._failures = 0
        self._last_reload = -RELOAD_INTERVAL_SEC
        self._assembler = UtteranceAssembler(
            sample_rate=int(values["sample_rate"]),
            max_seconds=float(values["max_speech_seconds"]),
            idle_timeout_sec=float(values["idle_timeout_sec"]),
        )
        self._embedder = self._load(values)
        if self._embedder is None:
            return
        self._model_id = self._embedder.model_id

        self._publisher = self.create_publisher(SpeakerEmbedding, str(values["output_topic"]), 10)
        # Reliable and deep enough for a burst of chunks: a dropped BEGIN costs
        # the whole utterance, which the gate then has to fail open on.
        qos = QoSProfile(depth=50)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self._subscription = self.create_subscription(SpeechAudioChunk, str(values["chunk_topic"]), self._on_chunk, qos)
        self._timer = self.create_timer(0.5, self._on_idle_check)
        self.get_logger().info(
            f"speaker_id_node: {values['chunk_topic']} -> {values['output_topic']} "
            f"(model={self._embedder.model_id}, dim={self._embedder.dim}, "
            f"deployment={values['deployment']})"
        )

    def _load(self, values: dict) -> SpeakerEmbedder | None:
        bundle_path = str(values["bundle_path"]).strip()
        deployment = str(values["deployment"]).strip()
        embedder = None
        try:
            if not bundle_path or not deployment:
                raise ValueError("speaker_id needs bundle_path and deployment; see voice_asr_service/README.md")
            embedder = SpeakerEmbedder(bundle_path, deployment)
            embedder.initialize()
            return embedder
        except Exception as error:  # noqa: BLE001 - the reason is reported either way
            if embedder is not None:
                embedder.close()
            if values["exit_on_init_failure"]:
                raise RuntimeError(f"speaker_id_node initialization failed: {error}") from error
            # The gate fails open without measurements, so a missing model
            # degrades the robot to the wake-phrase-only behaviour rather than
            # taking the voice pipeline down with it.
            self.get_logger().error(f"speaker_id_node disabled: {error}")
            return None

    def _on_chunk(self, message: SpeechAudioChunk) -> None:
        utterance = self._assembler.push(
            int(message.state),
            bytes(message.audio_data),
            at=time.monotonic(),
            source_id=str(message.source_id),
            sequence=int(message.sequence),
            header=message.header,
        )
        if self._assembler.discarded:
            # Mirrors the ASR's own discard log, so a missing voiceprint and a
            # missing transcript can be matched up in the launch log.
            self.get_logger().info(f"speaker_id_node: {self._assembler.discarded}; discarding utterance")
        if utterance is None:
            return
        self._publish(utterance)

    def _on_idle_check(self) -> None:
        utterance = self._assembler.flush_stalled(time.monotonic())
        if utterance is not None:
            self.get_logger().warning("speaker_id_node: utterance had no END; embedding what arrived")
            self._publish(utterance)

    def _publish(self, utterance) -> None:
        # Every utterance that ends gets one message, unless the model failed on
        # it. One that is too short (or that the model returned nothing for) gets
        # an empty embedding: the consumer then knows no voiceprint is coming and
        # decides at once, instead of holding the transcript for its grace
        # period (~1.5 s on the X2).
        embedding: tuple[float, ...] = ()
        if utterance.speech_seconds < self._min_speech_seconds:
            self.get_logger().debug(f"too short to embed: {utterance.speech_seconds:.2f}s")
        else:
            embedded = self._embed(utterance.pcm)
            if embedded is None:
                return
            embedding = embedded
            if not embedding:
                self.get_logger().warning("speaker embedding came back empty; publishing none")
        out = SpeakerEmbedding()
        if utterance.header is not None:
            out.header.stamp = utterance.header.stamp
            out.header.frame_id = utterance.header.frame_id
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        out.source_id = utterance.source_id
        out.segment_sequence = int(utterance.segment_sequence)
        out.embedding = list(embedding)
        out.model_id = self._model_id
        out.speech_seconds = float(utterance.speech_seconds)
        self._publisher.publish(out)

    def _embed(self, pcm) -> tuple[float, ...] | None:
        """The utterance's vector, or None when the model is down or failed on it."""
        if self._embedder is None:
            self._reload()
            if self._embedder is None:
                return None
        try:
            embedding = self._embedder.embed(pcm)
        except Exception as error:  # noqa: BLE001 - any failure of the model stays with this utterance
            self._failures += 1
            self.get_logger().error(
                f"speaker_id_node: embedding failed ({self._failures} in a row), publishing no voiceprint "
                f"for this utterance: {error}"
            )
            if self._failures >= RELOAD_AFTER_FAILURES:
                self._reload()
            return None
        self._failures = 0
        return embedding

    def _reload(self) -> None:
        """Load the deployment again; on failure stay up without one until the next attempt."""
        now = time.monotonic()
        if now - self._last_reload < RELOAD_INTERVAL_SEC:
            return
        self._last_reload = now
        old, self._embedder = self._embedder, None
        if old is not None:
            old.close()
        embedder = None
        try:
            embedder = SpeakerEmbedder(str(self._values["bundle_path"]), str(self._values["deployment"]))
            embedder.initialize()
        except Exception as error:  # noqa: BLE001 - reported, and retried later
            if embedder is not None:
                embedder.close()
            self.get_logger().error(
                f"speaker_id_node: reloading the model failed, publishing no voiceprints "
                f"(the gate fails open) and retrying in {RELOAD_INTERVAL_SEC:.0f} s: {error}"
            )
            return
        if embedder.model_id != self._model_id:
            # Another model would make every vector incomparable with the anchors
            # the consumer already holds; refuse it rather than switch silently.
            embedder.close()
            self.get_logger().error(
                f"speaker_id_node: the reloaded model is {embedder.model_id}, not {self._model_id}; "
                "keeping the node without a model"
            )
            return
        self._embedder = embedder
        self._failures = 0
        self.get_logger().warning("speaker_id_node: model reloaded after repeated failures")

    def close(self) -> None:
        """Release the loaded deployment; safe before and after a failed load."""
        embedder, self._embedder = getattr(self, "_embedder", None), None
        if embedder is not None:
            embedder.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SpeakerIdNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C from the launch shuts the context down under spin(); both are a
        # normal stop, not a failure.
        pass
    except RuntimeError as error:
        print(f"[speaker_id_node] {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        if node is not None:
            # The loaded deployment owns an ONNX Runtime session and the
            # process-local runtime providers; destroy_node does not reach them.
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
