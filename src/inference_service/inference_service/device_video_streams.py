"""Device-side ownership of configured observation video encoders and RTP senders."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from inference_service.distributed.types import PROTOCOL_VERSION, StreamReference
from inference_service.distributed.video_streams import (
    VideoStreamDescriptor,
    VideoStreamDiagnosticSnapshot,
    VideoStreamRuntimeStatus,
)
from inference_service.video_codec import (
    VideoCodecRegistry,
    VideoEncoder,
    VideoFrame,
    create_default_video_codec_registry,
)
from inference_service.video_rtp import H264RtpSender
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import ObservationTransportSpec, effective_observation_transport
from tensormsg.converter import ros_image_to_hwc_uint8

_RTP_CLOCK_RATE = 90_000
_RTP_PAYLOAD_TYPE = 96


@dataclass(slots=True)
class _EdgeStream:
    spec: SpecView
    transport: ObservationTransportSpec
    encoder: VideoEncoder
    sender: H264RtpSender
    selected_backend: str
    ssrc: int
    lock: threading.Lock
    sender_factory: Callable[..., H264RtpSender]
    last_capture_timestamp_ns: int = 0
    last_input_capture_timestamp_ns: int = 0
    last_rtp_timestamp: int = 0
    keyframe_sent: bool = False
    dropped_frames: int = 0
    last_error: str = ""


class DeviceVideoStreamManager:
    """Encode configured RTP observations continuously for one edge pipeline."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        contract_fingerprint: str,
        deployment_fingerprint: str,
        observation_specs: Iterable[SpecView],
        codec_registry: VideoCodecRegistry | None = None,
        sender_factory: Callable[..., H264RtpSender] = H264RtpSender,
    ) -> None:
        if not pipeline_id or not contract_fingerprint or not deployment_fingerprint:
            raise ValueError("device video streams require pipeline and fingerprint identity")
        self.pipeline_id = pipeline_id
        self.contract_fingerprint = contract_fingerprint
        self.deployment_fingerprint = deployment_fingerprint
        self._session_id = ""
        self._session_generation = 0
        self._lock = threading.RLock()
        registry = codec_registry or create_default_video_codec_registry()
        self._streams: dict[str, _EdgeStream] = {}
        try:
            for spec in observation_specs:
                transport = effective_observation_transport(spec.transport)
                if transport.mode != "rtp":
                    continue
                self._streams[spec.key] = self._create_stream(spec, transport, registry, sender_factory)
        except Exception:
            self.close()
            raise

    @property
    def stream_references(self) -> tuple[StreamReference, ...]:
        return tuple(
            StreamReference(observation_key, stream.transport.stream_id or "")
            for observation_key, stream in sorted(self._streams.items())
        )

    @property
    def observation_keys(self) -> frozenset[str]:
        return frozenset(self._streams)

    @property
    def session(self) -> tuple[str, int]:
        with self._lock:
            return self._session_id, self._session_generation

    def diagnostic_snapshots(self) -> tuple[VideoStreamDiagnosticSnapshot, ...]:
        snapshots = []
        for stream in sorted(self._streams.values(), key=lambda item: item.spec.key):
            transport = stream.transport
            assert transport.stream_id is not None
            assert transport.endpoint is not None
            with stream.lock:
                status = stream.sender.status
                snapshots.append(
                    VideoStreamDiagnosticSnapshot(
                        observation_key=stream.spec.key,
                        stream_id=transport.stream_id,
                        mode=transport.mode,
                        configured_encoder_backend=transport.encoder_backend,
                        selected_encoder_backend=stream.selected_backend,
                        configured_decoder_backend=transport.decoder_backend,
                        selected_decoder_backend="not-local",
                        endpoint=(transport.endpoint.host, transport.endpoint.port),
                        contract_fingerprint=self.contract_fingerprint,
                        deployment_fingerprint=self.deployment_fingerprint,
                        security="none/trusted-network-only",
                        lifecycle_state=status.state.value,
                        ready=status.ready and stream.keyframe_sent,
                    )
                )
        return tuple(snapshots)

    def bind_session(self, session_id: str, session_generation: int) -> bool:
        if not session_id or session_generation < 1:
            raise ValueError("device video streams require a live session")
        with self._lock:
            if (session_id, session_generation) == (self._session_id, self._session_generation):
                return False
            self._session_id = ""
            self._session_generation = 0
            for stream in tuple(self._streams.values()):
                stream.sender.close()
                with stream.lock:
                    stream.encoder.reset()
                    stream.ssrc = secrets.randbits(32)
                    stream.sender = self._create_sender(stream)
                    stream.last_capture_timestamp_ns = 0
                    stream.last_input_capture_timestamp_ns = 0
                    stream.last_rtp_timestamp = 0
                    stream.keyframe_sent = False
                    stream.last_error = ""
            self._session_id = session_id
            self._session_generation = session_generation
        return True

    def clear_session(self) -> None:
        with self._lock:
            self._session_id = ""
            self._session_generation = 0

    def submit_ros_image(
        self,
        observation_key: str,
        message: object,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
    ) -> bool:
        stream = self._streams.get(observation_key)
        if stream is None:
            return False
        with self._lock:
            if not self._session_id or not self._session_generation:
                return False
            with stream.lock:
                if capture_timestamp_ns <= stream.last_input_capture_timestamp_ns:
                    stream.dropped_frames += 1
                    return False
                try:
                    frame = ros_image_to_hwc_uint8(
                        message,
                        output_encoding="rgb8",
                        resize=(stream.transport.media.height, stream.transport.media.width),
                    )
                    packets = stream.encoder.encode(
                        VideoFrame(
                            frame,
                            capture_timestamp_ns,
                            receive_timestamp_ns,
                            stream.transport.media.width,
                            stream.transport.media.height,
                            "rgb24",
                            color_space=stream.transport.media.color_space,
                            color_range=stream.transport.media.color_range,
                        )
                    )
                    for packet in packets:
                        stream.sender.enqueue(packet)
                    stream.last_input_capture_timestamp_ns = capture_timestamp_ns
                    stream.last_error = ""
                except Exception as exc:
                    stream.last_error = str(exc)
                    raise
        return True

    def descriptors(self) -> tuple[VideoStreamDescriptor, ...]:
        with self._lock:
            session_id = self._session_id
            session_generation = self._session_generation
        if not session_id or session_generation < 1:
            return ()
        return tuple(
            self._descriptor(stream, session_id, session_generation)
            for stream in sorted(self._streams.values(), key=lambda item: item.spec.key)
        )

    def statuses(self) -> tuple[VideoStreamRuntimeStatus, ...]:
        with self._lock:
            session_id = self._session_id
            session_generation = self._session_generation
        if not session_id or session_generation < 1:
            return ()
        return tuple(
            self._status(stream, session_id, session_generation)
            for stream in sorted(self._streams.values(), key=lambda item: item.spec.key)
        )

    def reset(self) -> None:
        for stream in tuple(self._streams.values()):
            with stream.lock:
                stream.encoder.reset()
                stream.sender.reset()
                stream.last_capture_timestamp_ns = 0
                stream.last_input_capture_timestamp_ns = 0
                stream.last_rtp_timestamp = 0
                stream.keyframe_sent = False
                stream.last_error = ""

    def close(self, timeout_s: float = 1.0) -> None:
        streams = tuple(getattr(self, "_streams", {}).values())
        for stream in streams:
            try:
                stream.sender.close(timeout_s)
            finally:
                stream.encoder.close(timeout_s)
        if hasattr(self, "_streams"):
            self._streams.clear()
        if hasattr(self, "_lock"):
            self.clear_session()

    @staticmethod
    def _create_stream(
        spec: SpecView,
        transport: ObservationTransportSpec,
        registry: VideoCodecRegistry,
        sender_factory: Callable[..., H264RtpSender],
    ) -> _EdgeStream:
        if transport.stream_id is None or transport.endpoint is None:
            raise ValueError(f"RTP observation {spec.key!r} is missing stream identity or endpoint")
        if transport.h264 is None or transport.media is None or transport.buffer is None:
            raise ValueError(f"RTP observation {spec.key!r} has unresolved codec, media, or buffer settings")
        resolved = registry.resolve(transport.encoder_backend, "encoder")
        encoder = resolved.create(
            width=transport.media.width,
            height=transport.media.height,
            frame_rate_hz=transport.media.frame_rate_hz,
            bitrate_bps=transport.h264.bitrate_bps,
            gop_frames=transport.h264.gop_frames,
            input_pixel_format="rgb24",
            profile=transport.h264.profile,
            color_space=transport.media.color_space,
            color_range=transport.media.color_range,
        )
        ssrc = secrets.randbits(32)
        holder: dict[str, _EdgeStream] = {}

        def mark_sent(packet) -> None:
            stream = holder.get("stream")
            if stream is None:
                return
            with stream.lock:
                stream.last_capture_timestamp_ns = packet.capture_timestamp_ns
                stream.last_rtp_timestamp = packet.rtp_timestamp
                stream.keyframe_sent = stream.keyframe_sent or packet.keyframe

        sender = sender_factory(
            stream_id=transport.stream_id,
            endpoint=(transport.endpoint.host, transport.endpoint.port),
            ssrc=ssrc,
            queue_capacity=transport.buffer.sender_queue_frames,
            payload_type=_RTP_PAYLOAD_TYPE,
            selected_backend=resolved.name,
            on_sent=mark_sent,
        )
        try:
            sender.start()
        except Exception:
            encoder.close()
            raise
        stream = _EdgeStream(
            spec,
            transport,
            encoder,
            sender,
            resolved.name,
            ssrc,
            threading.RLock(),
            sender_factory,
        )
        holder["stream"] = stream
        return stream

    @staticmethod
    def _create_sender(stream: _EdgeStream) -> H264RtpSender:
        transport = stream.transport
        assert transport.stream_id is not None
        assert transport.endpoint is not None
        assert transport.buffer is not None

        def mark_sent(packet) -> None:
            with stream.lock:
                stream.last_capture_timestamp_ns = packet.capture_timestamp_ns
                stream.last_rtp_timestamp = packet.rtp_timestamp
                stream.keyframe_sent = stream.keyframe_sent or packet.keyframe

        sender = stream.sender_factory(
            stream_id=transport.stream_id,
            endpoint=(transport.endpoint.host, transport.endpoint.port),
            ssrc=stream.ssrc,
            queue_capacity=transport.buffer.sender_queue_frames,
            payload_type=_RTP_PAYLOAD_TYPE,
            selected_backend=stream.selected_backend,
            on_sent=mark_sent,
        )
        sender.start()
        return sender

    def _descriptor(self, stream: _EdgeStream, session_id: str, session_generation: int) -> VideoStreamDescriptor:
        transport = stream.transport
        assert transport.stream_id is not None
        assert transport.endpoint is not None
        assert transport.h264 is not None
        assert transport.media is not None
        return VideoStreamDescriptor(
            protocol_version=PROTOCOL_VERSION,
            pipeline_id=self.pipeline_id,
            session_id=session_id,
            session_generation=session_generation,
            observation_key=stream.spec.key,
            stream_id=transport.stream_id,
            endpoint_host=transport.endpoint.host,
            endpoint_port=transport.endpoint.port,
            ssrc=stream.ssrc,
            payload_type=_RTP_PAYLOAD_TYPE,
            codec=transport.codec,
            codec_profile=transport.h264.profile,
            width=transport.media.width,
            height=transport.media.height,
            frame_rate_hz=transport.media.frame_rate_hz,
            rtp_clock_rate=_RTP_CLOCK_RATE,
            pixel_format=transport.media.pixel_format,
            color_space=transport.media.color_space,
            color_range=transport.media.color_range,
            encoder_backend=stream.selected_backend,
            contract_fingerprint=self.contract_fingerprint,
            deployment_fingerprint=self.deployment_fingerprint,
        )

    def _status(self, stream: _EdgeStream, session_id: str, session_generation: int) -> VideoStreamRuntimeStatus:
        with stream.lock:
            sender_status = stream.sender.status
            sender_metrics = sender_status.metrics
            encoder_metrics = stream.encoder.metrics
            return VideoStreamRuntimeStatus(
                protocol_version=PROTOCOL_VERSION,
                pipeline_id=self.pipeline_id,
                session_id=session_id,
                session_generation=session_generation,
                observation_key=stream.spec.key,
                stream_id=stream.transport.stream_id or "",
                lifecycle_state=sender_status.state.value,
                ready=sender_status.ready and stream.keyframe_sent,
                selected_backend=stream.selected_backend,
                timestamp_mapping_valid=stream.last_capture_timestamp_ns > 0,
                mapping_rtp_timestamp=stream.last_rtp_timestamp,
                mapping_capture_timestamp_ns=stream.last_capture_timestamp_ns,
                keyframe_ready=stream.keyframe_sent,
                encoded_frames=encoder_metrics.input_frames,
                sent_packets=sender_metrics.sent_packets,
                dropped_frames=sender_metrics.dropped_frames + stream.dropped_frames,
                sender_queue_depth=sender_metrics.queued_frames,
                reconnect_count=sender_metrics.reconnect_count,
                last_error=stream.last_error or sender_status.last_error,
            )
