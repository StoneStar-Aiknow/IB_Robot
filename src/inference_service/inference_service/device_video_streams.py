"""Device-side ownership of configured observation video encoders and RTP senders."""

from __future__ import annotations

import queue
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field

from inference_service.distributed.types import PROTOCOL_VERSION, StreamReference
from inference_service.distributed.video_streams import (
    VideoStreamDescriptor,
    VideoStreamDiagnosticSnapshot,
    VideoStreamRuntimeStatus,
)
from inference_service.video_codec import (
    CodecLifecycleState,
    ResolvedCodecBackend,
    VideoCodecRegistry,
    VideoEncoder,
    VideoFrame,
    create_default_video_codec_registry,
)
from inference_service.video_rtp import H264RtpSender, StreamLifecycleState
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import ObservationTransportSpec, effective_observation_transport
from tensormsg.converter import ros_image_to_hwc_uint8

_RTP_CLOCK_RATE = 90_000
_RTP_PAYLOAD_TYPE = 96
# Depth of the per-stream encode queue.  The worker encodes far faster than
# the publish rate, so this only absorbs arrival bursts; overflow drops the
# oldest frame, which the next keyframe heals.
_ENCODE_QUEUE_FRAMES = 4
_ENCODE_POLL_S = 0.1


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
    # Guards the submit-side timestamp gate and enqueue; the encode worker
    # never takes it, so executor callbacks never queue behind an in-flight
    # encode.  The gate and the enqueue must share one critical section:
    # with a reentrant callback group two concurrent callbacks could
    # otherwise pass the gate in order and enqueue out of order.
    submit_gate: threading.Lock = field(default_factory=threading.Lock)
    encode_queue: queue.Queue[tuple[object, int, int, int, int]] = field(
        default_factory=lambda: queue.Queue(maxsize=_ENCODE_QUEUE_FRAMES)
    )
    encode_stop: threading.Event = field(default_factory=threading.Event)
    encode_worker: threading.Thread | None = None
    # Bumped by reset()/bind_session()/clear_session() so frames dequeued
    # before a lifecycle boundary are dropped by the worker instead of
    # surfacing as the first frame of the new epoch/session.
    lifecycle_epoch: int = 0
    active_session_generation: int = 0
    last_capture_timestamp_ns: int = 0
    last_input_capture_timestamp_ns: int = 0
    last_rtp_timestamp: int = 0
    last_sent_mapping: tuple[int, int] = (0, 0)
    keyframe_sent: bool = False
    dropped_frames: int = 0
    failed_frames: int = 0
    submitted_frames: int = 0
    metrics_started_ns: int = field(default_factory=time.monotonic_ns)
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
        rtp_specs = []
        for spec in observation_specs:
            transport = effective_observation_transport(spec.transport)
            if transport.mode == "rtp":
                rtp_specs.append((spec, transport))
        resolved_backends = {
            spec.key: registry.resolve(transport.encoder_backend, "encoder") for spec, transport in rtp_specs
        }
        # Ascend DVPP channel IDs must stay dense (1..N) below the hardware
        # limit even when software-encoded streams are interleaved, so number
        # only the Ascend streams; other backends ignore the channel.
        encoder_channel_ids = {
            observation_key: channel_id
            for channel_id, observation_key in enumerate(
                sorted(key for key, resolved in resolved_backends.items() if resolved.name == "ascend"),
                start=1,
            )
        }
        if encoder_channel_ids:
            max_channel = max(encoder_channel_ids.values())
            if max_channel > 127:
                raise ValueError(
                    f"Ascend DVPP requires at most 128 VENC channels per device; got channel_id={max_channel}"
                )
        try:
            for spec, transport in rtp_specs:
                stream = self._create_stream(
                    spec,
                    transport,
                    resolved_backends[spec.key],
                    sender_factory,
                    encoder_channel_ids.get(spec.key, 0),
                )
                stream.encode_worker = threading.Thread(
                    target=self._encode_worker,
                    args=(stream,),
                    name=f"video-encode-{stream.transport.stream_id or spec.key}",
                    daemon=True,
                )
                stream.encode_worker.start()
                self._streams[spec.key] = stream
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

    def latest_sent_capture_ns(self, observation_key: str) -> int:
        """Capture timestamp of the newest access unit actually put on the wire.

        Freshness decisions on the device side must reflect what the compute
        side can actually see, not what the local subscription received:
        frames that failed to encode or never left the sender queue do not
        exist remotely, so they must not count as "new".  The value updates
        only from the sender thread's post-send callback and reads as 0 when
        the stream is unknown or nothing has been sent yet -- including right
        after a session rollover, which clears the record so pre-rollover
        frames are never mistaken for fresh.
        """
        stream = self._streams.get(observation_key)
        if stream is None:
            return 0
        return stream.last_sent_mapping[1]

    def latest_sent_mapping(self, observation_key: str) -> tuple[int, int]:
        stream = self._streams.get(observation_key)
        return stream.last_sent_mapping if stream is not None else (0, 0)

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
        """Bind or rebind the live cloud session without restarting healthy encoders.

        Session rollovers used to close and recreate every sender and reset every
        encoder, which respawns the Ascend FFmpeg process and its DVPP channel.
        Heartbeat flaps and cloud reconnects churned those resets faster than the
        pipeline could encode a single frame.  Rebinding now keeps healthy FFmpeg
        processes and sender threads alive: only the SSRC, the sender queue, and
        per-stream bookkeeping rotate, while senders or encoders that actually
        died are recreated as a recovery path.

        The bind is atomic (prepare/commit/rollback): the lifecycle epoch is
        advanced first so frames dequeued but not yet encoded die, the encode
        queue is drained, output still inside the encoder pipeline is
        discarded, and the sender SSRC rotates so in-flight sent-callbacks of
        the retired session are rejected.  Only when every stream prepared
        successfully is the new generation committed; on any failure the
        manager stays session-less and the next rollover retries, instead of
        advertising a session whose streams are silently inactive.
        """
        if not session_id or session_generation < 1:
            raise ValueError("device video streams require a live session")
        with self._lock:
            if (session_id, session_generation) == (self._session_id, self._session_generation):
                return False
            prepared: list[_EdgeStream] = []
            try:
                for stream in tuple(self._streams.values()):
                    with stream.lock:
                        stream.lifecycle_epoch += 1
                        self._drain_encode_queue(stream)
                        stream.ssrc = secrets.randbits(32)
                        self._rotate_sender(stream)
                        if stream.encoder.state is not CodecLifecycleState.RUNNING:
                            stream.encoder.reset()
                        else:
                            # Healthy encoder: drop access units of the retired
                            # session still in flight through the DVPP
                            # pipeline instead of respawning the process.
                            stream.encoder.discard_pending_output()
                        stream.last_capture_timestamp_ns = 0
                        stream.last_input_capture_timestamp_ns = 0
                        stream.last_rtp_timestamp = 0
                        stream.last_sent_mapping = (0, 0)
                        stream.keyframe_sent = False
                        stream.last_error = ""
                        stream.failed_frames = 0
                        stream.submitted_frames = 0
                        stream.metrics_started_ns = time.monotonic_ns()
                        stream.active_session_generation = 0
                        # Register for rollback before releasing the lock so a
                        # failure in a later stream's prepare still rolls this
                        # one back instead of leaving it half-bound.
                        prepared.append(stream)
            except Exception as exc:
                # Rollback: a partially bound session must not be advertised.
                # Prepared streams stay inactive and the epoch is advanced
                # again so nothing they already queued survives either.
                for stream in prepared:
                    with stream.lock:
                        stream.lifecycle_epoch += 1
                        self._drain_encode_queue(stream)
                        stream.active_session_generation = 0
                        stream.last_error = f"session bind rolled back: {exc}"
                raise
            for stream in prepared:
                with stream.lock:
                    stream.active_session_generation = session_generation
            self._session_id = session_id
            self._session_generation = session_generation
        return True

    def _rotate_sender(self, stream: _EdgeStream) -> None:
        """Rotate the sender SSRC, replacing senders whose worker thread died."""
        sender = stream.sender
        sender.ssrc = stream.ssrc
        if sender.status.state not in {StreamLifecycleState.FAILED, StreamLifecycleState.STOPPED}:
            sender.reset()
            return
        # A sender thread that exited on a send error cannot be revived by
        # reset(); replace the sender entirely.  close() only joins a thread
        # that is already dead here, so this stays cheap.
        with suppress(Exception):
            sender.close()
        stream.sender = self._create_sender(stream)

    def clear_session(self) -> None:
        with self._lock:
            for stream in tuple(self._streams.values()):
                with stream.lock:
                    # Quiesce pending input so frames of the cleared session
                    # are not encoded and sent after the heartbeat already
                    # expired; full resource cleanup stays with reset()/close().
                    stream.lifecycle_epoch += 1
                    self._drain_encode_queue(stream)
                    stream.active_session_generation = 0
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
        """Accept one ROS image for asynchronous encoding.

        Encoding (conversion + the Ascend drain wait) used to run inline in
        the executor callback while holding ``stream.lock``.  Blocked
        callbacks piled up on the bounded worker pool, stalling the executor's
        dispatch loop until the best-effort DDS reader queues overflowed and
        dropped frames wholesale.  The callback now only runs the cheap
        timestamp gate and enqueues the raw message; a dedicated per-stream
        worker thread performs the encode.  Overflow drops the oldest queued
        frame, which the next keyframe heals.

        The timestamp gate and the enqueue share one critical section: the
        observation callback runs in a reentrant callback group on a
        multithreaded executor, so two concurrent callbacks for the same
        observation could otherwise pass the gate in timestamp order but
        enqueue out of order, tripping the encoders' monotonic-timestamp
        validation and desynchronizing the RTP timestamp FIFO.
        """
        stream = self._streams.get(observation_key)
        if stream is None:
            return False
        with self._lock:
            if not self._session_id or not self._session_generation:
                return False
            session_generation = self._session_generation
        with stream.submit_gate:
            if stream.active_session_generation != session_generation:
                return False
            if capture_timestamp_ns <= stream.last_input_capture_timestamp_ns:
                stream.dropped_frames += 1
                return False
            stream.last_input_capture_timestamp_ns = capture_timestamp_ns
            stream.submitted_frames += 1
            item = (message, capture_timestamp_ns, receive_timestamp_ns, session_generation, stream.lifecycle_epoch)
            while True:
                try:
                    stream.encode_queue.put_nowait(item)
                    return True
                except queue.Full:
                    try:
                        stream.encode_queue.get_nowait()
                        stream.encode_queue.task_done()
                    except queue.Empty:
                        pass
                    stream.dropped_frames += 1

    def _encode_worker(self, stream: _EdgeStream) -> None:
        while not stream.encode_stop.is_set():
            try:
                item = stream.encode_queue.get(timeout=_ENCODE_POLL_S)
            except queue.Empty:
                continue
            message, capture_timestamp_ns, receive_timestamp_ns, session_generation, epoch = item
            try:
                with stream.lock:
                    if stream.lifecycle_epoch != epoch or stream.active_session_generation != session_generation:
                        # A reset or session rollover retired this frame while
                        # it waited in the queue; drop it instead of letting it
                        # surface as the first frame of the new epoch.
                        continue
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
                        stream.last_error = ""
                    except Exception as exc:
                        # Propagate the failure: clear the keyframe flag so
                        # status stops reporting ready, count the frame as
                        # failed, and attempt a controlled encoder recovery
                        # so subsequent frames are not doomed to repeat it.
                        stream.last_error = str(exc)
                        stream.failed_frames += 1
                        stream.keyframe_sent = False
                        self._recover_encoder(stream)
            finally:
                stream.encode_queue.task_done()

    @staticmethod
    def _recover_encoder(stream: _EdgeStream) -> None:
        """Try to bring a failed encoder back so later frames can succeed."""
        if stream.encoder.state is not CodecLifecycleState.FAILED:
            return
        try:
            stream.encoder.reset()
        except Exception as exc:
            stream.last_error = f"encoder reset failed: {exc}"

    def flush(self, timeout_s: float = 2.0) -> bool:
        """Block until every queued frame has been encoded (test helper)."""
        deadline = time.monotonic() + timeout_s
        for stream in self._streams.values():
            while stream.encode_queue.unfinished_tasks > 0:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.005)
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

    def sender_diagnostics(self) -> tuple[dict[str, object], ...]:
        diagnostics = []
        for stream in sorted(self._streams.values(), key=lambda item: item.spec.key):
            elapsed_s = max((time.monotonic_ns() - stream.metrics_started_ns) / 1e9, 1e-9)
            encoder_metrics = stream.encoder.metrics
            sender_metrics = stream.sender.status.metrics
            diagnostics.append(
                {
                    "observation": stream.spec.key,
                    "submitted_fps": stream.submitted_frames / elapsed_s,
                    "encoded_fps": encoder_metrics.output_frames / elapsed_s,
                    "sent_fps": sender_metrics.sent_frames / elapsed_s,
                    "submitted_frames": stream.submitted_frames,
                    "encoded_frames": encoder_metrics.output_frames,
                    "sent_frames": sender_metrics.sent_frames,
                    "sent_packets": sender_metrics.sent_packets,
                    "encode_queue_depth": stream.encode_queue.qsize(),
                    "sender_queue_depth": sender_metrics.queued_frames,
                    "dropped_frames": sender_metrics.dropped_frames + stream.dropped_frames + stream.failed_frames,
                }
            )
        return tuple(diagnostics)

    def reset(self) -> None:
        for stream in tuple(self._streams.values()):
            with stream.lock:
                # Advance the epoch first: a frame the worker already dequeued
                # but has not encoded yet must not survive the reset and
                # become the first frame of the new epoch.
                stream.lifecycle_epoch += 1
                self._drain_encode_queue(stream)
                self._rotate_sender(stream)
                stream.encoder.reset()
                stream.last_capture_timestamp_ns = 0
                stream.last_input_capture_timestamp_ns = 0
                stream.last_rtp_timestamp = 0
                stream.last_sent_mapping = (0, 0)
                stream.keyframe_sent = False
                stream.last_error = ""
                stream.failed_frames = 0
                stream.submitted_frames = 0
                stream.metrics_started_ns = time.monotonic_ns()

    @staticmethod
    def _drain_encode_queue(stream: _EdgeStream) -> None:
        while True:
            try:
                stream.encode_queue.get_nowait()
            except queue.Empty:
                return
            stream.encode_queue.task_done()

    def close(self, timeout_s: float = 1.0) -> None:
        deadline = time.monotonic() + timeout_s
        streams = tuple(getattr(self, "_streams", {}).values())
        for stream in streams:
            stream.encode_stop.set()
        for stream in streams:
            worker = stream.encode_worker
            if worker is not None:
                worker.join(max(0.0, deadline - time.monotonic()))
        for stream in streams:
            worker = stream.encode_worker
            stuck = worker is not None and worker.is_alive()
            if stuck:
                # The worker is wedged (typically blocked writing to the
                # FFmpeg stdin pipe).  Close the encoder first so the blocked
                # write fails and the worker can exit; closing the sender and
                # other resources while the worker still references them
                # would be a use-after-close.
                with suppress(Exception):
                    stream.encoder.close(max(0.0, deadline - time.monotonic()))
                worker.join(max(0.0, deadline - time.monotonic()))
                stuck = worker.is_alive()
            if stuck:
                # Still wedged: leave the sender untouched rather than close
                # resources a live worker may still touch.
                continue
            try:
                stream.sender.close(max(0.0, deadline - time.monotonic()))
            finally:
                stream.encoder.close(max(0.0, deadline - time.monotonic()))
        if hasattr(self, "_streams"):
            self._streams.clear()
        if hasattr(self, "_lock"):
            self.clear_session()

    @staticmethod
    def _create_stream(
        spec: SpecView,
        transport: ObservationTransportSpec,
        resolved: ResolvedCodecBackend,
        sender_factory: Callable[..., H264RtpSender],
        channel_id: int,
    ) -> _EdgeStream:
        if transport.stream_id is None or transport.endpoint is None:
            raise ValueError(f"RTP observation {spec.key!r} is missing stream identity or endpoint")
        if transport.h264 is None or transport.media is None or transport.buffer is None:
            raise ValueError(f"RTP observation {spec.key!r} has unresolved codec, media, or buffer settings")
        encoder_options: dict[str, object] = {
            "width": transport.media.width,
            "height": transport.media.height,
            "frame_rate_hz": transport.media.frame_rate_hz,
            "bitrate_bps": transport.h264.bitrate_bps,
            "gop_frames": transport.h264.gop_frames,
            "input_pixel_format": "rgb24",
            "profile": transport.h264.profile,
            "color_space": transport.media.color_space,
            "color_range": transport.media.color_range,
        }
        if resolved.name == "ascend":
            encoder_options["channel_id"] = channel_id
        encoder = resolved.create(
            **encoder_options,
        )
        ssrc = secrets.randbits(32)
        holder: dict[str, _EdgeStream] = {}

        def mark_sent(packet) -> None:
            stream = holder.get("stream")
            if stream is None:
                return
            if stream.sender is not sender:
                # This sender was replaced by a rollover/recovery; its late
                # callback belongs to the retired sender and must not
                # repopulate the new session's bookkeeping.
                return
            _mark_sent_unlocked(stream, packet)

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
            if stream.sender is not sender:
                return
            _mark_sent_unlocked(stream, packet)

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
        sender_status = stream.sender.status
        sender_metrics = sender_status.metrics
        encoder_metrics = stream.encoder.metrics
        mapping_rtp_timestamp, mapping_capture_timestamp_ns = stream.last_sent_mapping
        # Ready requires the whole chain to be healthy: the sender, an
        # acknowledged keyframe, a running encoder, a live encode worker, and
        # no outstanding encode error.  A dead encoder or a swallowed encode
        # exception used to leave status ready=True on historical keyframes.
        worker = stream.encode_worker
        worker_alive = worker is None or worker.is_alive()
        encoder_running = stream.encoder.state is CodecLifecycleState.RUNNING
        return VideoStreamRuntimeStatus(
            protocol_version=PROTOCOL_VERSION,
            pipeline_id=self.pipeline_id,
            session_id=session_id,
            session_generation=session_generation,
            observation_key=stream.spec.key,
            stream_id=stream.transport.stream_id or "",
            lifecycle_state=sender_status.state.value if encoder_running and worker_alive else "failed",
            ready=(
                sender_status.ready
                and stream.keyframe_sent
                and encoder_running
                and worker_alive
                and not stream.last_error
            ),
            selected_backend=stream.selected_backend,
            timestamp_mapping_valid=mapping_capture_timestamp_ns > 0,
            mapping_rtp_timestamp=mapping_rtp_timestamp,
            mapping_capture_timestamp_ns=mapping_capture_timestamp_ns,
            keyframe_ready=stream.keyframe_sent,
            encoded_frames=encoder_metrics.input_frames,
            sent_packets=sender_metrics.sent_packets,
            dropped_frames=sender_metrics.dropped_frames + stream.dropped_frames + stream.failed_frames,
            sender_queue_depth=sender_metrics.queued_frames,
            reconnect_count=sender_metrics.reconnect_count,
            last_error=stream.last_error or sender_status.last_error,
        )


def _mark_sent_unlocked(stream: _EdgeStream, packet) -> None:
    """Record one sent access unit without ever taking ``stream.lock``.

    The sender thread invokes this after every UDP send.  Taking the stream
    lock here would stall the sender behind ``submit_ros_image`` while it
    encodes, so the bookkeeping uses plain attribute stores: the RTP timestamp
    and capture timestamp are single-word writes, and keyframe acknowledgement
    is a monotonic flag.
    """
    stream.last_sent_mapping = (packet.rtp_timestamp, packet.capture_timestamp_ns)
    stream.last_rtp_timestamp, stream.last_capture_timestamp_ns = stream.last_sent_mapping
    if packet.keyframe:
        stream.keyframe_sent = True
