"""RFC 6184 H.264 RTP transport with bounded real-time queues."""

from __future__ import annotations

import secrets
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from inference_service.h264_stream_recorder import H264StreamRecorder

from inference_service.observation_sync import ObservationSynchronizationError, RtpTimestampMapper
from inference_service.video_codec import EncodedPacket, VideoCodecError, VideoDecoder, VideoFrame
from robot_config.contract_utils import StreamBuffer

_RTP_VERSION = 2
_RTP_HEADER = struct.Struct("!BBHII")
_ANNEX_B_START = b"\x00\x00\x00\x01"


class StreamLifecycleState(str, Enum):
    CONFIGURED = "configured"
    STARTING = "starting"
    WAITING_FOR_KEYFRAME = "waiting_for_keyframe"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    queued_frames: int = 0
    queued_packets: int = 0
    sent_frames: int = 0
    sent_packets: int = 0
    dropped_frames: int = 0
    received_packets: int = 0
    dropped_packets: int = 0
    lost_packets: int = 0
    decoded_frames: int = 0
    decode_errors: int = 0
    errors: int = 0
    reconnect_count: int = 0


@dataclass(frozen=True, slots=True)
class StreamStatus:
    stream_id: str
    state: StreamLifecycleState
    ready: bool
    selected_backend: str
    metrics: StreamMetrics
    last_error: str = ""


class VideoRtpError(RuntimeError):
    def __init__(self, code: str, message: str, *, stream_id: str, recoverable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.stream_id = stream_id
        self.recoverable = recoverable


@dataclass(frozen=True, slots=True)
class RtpPacket:
    payload_type: int
    marker: bool
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must fit in 7 bits")
        if not 0 <= self.sequence <= 0xFFFF:
            raise ValueError("RTP sequence must fit in uint16")
        if not 0 <= self.timestamp <= 0xFFFF_FFFF or not 0 <= self.ssrc <= 0xFFFF_FFFF:
            raise ValueError("RTP timestamp and SSRC must fit in uint32")
        if not self.payload:
            raise ValueError("RTP payload cannot be empty")

    def to_bytes(self) -> bytes:
        second = self.payload_type | (0x80 if self.marker else 0)
        return _RTP_HEADER.pack(0x80, second, self.sequence, self.timestamp, self.ssrc) + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> RtpPacket:
        if len(data) < _RTP_HEADER.size:
            raise ValueError("RTP datagram is shorter than the fixed header")
        first, second, sequence, timestamp, ssrc = _RTP_HEADER.unpack_from(data)
        if first >> 6 != _RTP_VERSION:
            raise ValueError("unsupported RTP version")
        csrc_count = first & 0x0F
        offset = _RTP_HEADER.size + csrc_count * 4
        if len(data) < offset:
            raise ValueError("truncated RTP CSRC list")
        if first & 0x10:
            if len(data) < offset + 4:
                raise ValueError("truncated RTP extension header")
            extension_words = struct.unpack_from("!H", data, offset + 2)[0]
            offset += 4 + extension_words * 4
            if len(data) < offset:
                raise ValueError("truncated RTP extension payload")
        end = len(data)
        if first & 0x20:
            padding = data[-1]
            if padding == 0 or padding > end - offset:
                raise ValueError("invalid RTP padding")
            end -= padding
        return cls(second & 0x7F, bool(second & 0x80), sequence, timestamp, ssrc, data[offset:end])


def packetize_h264(
    access_unit: EncodedPacket,
    *,
    ssrc: int,
    payload_type: int,
    sequence: int,
    max_payload_size: int,
) -> tuple[list[RtpPacket], int]:
    """Packetize one Annex-B access unit using single NAL or FU-A packets."""
    if max_payload_size < 3:
        raise ValueError("max_payload_size must leave room for FU-A headers")
    nal_units = split_annex_b(access_unit.payload)
    if not nal_units:
        raise ValueError("H.264 access unit contains no NAL units")
    packets: list[RtpPacket] = []
    current_sequence = sequence
    for nal_index, nal in enumerate(nal_units):
        if not nal:
            continue
        is_last_nal = nal_index == len(nal_units) - 1
        if len(nal) <= max_payload_size:
            packets.append(
                RtpPacket(
                    payload_type,
                    is_last_nal,
                    current_sequence,
                    access_unit.rtp_timestamp,
                    ssrc,
                    nal,
                )
            )
            current_sequence = (current_sequence + 1) & 0xFFFF
            continue
        nal_header = nal[0]
        fu_indicator = (nal_header & 0xE0) | 28
        nal_type = nal_header & 0x1F
        chunks = [nal[index : index + max_payload_size - 2] for index in range(1, len(nal), max_payload_size - 2)]
        for chunk_index, chunk in enumerate(chunks):
            fu_header = nal_type
            if chunk_index == 0:
                fu_header |= 0x80
            if chunk_index == len(chunks) - 1:
                fu_header |= 0x40
            packets.append(
                RtpPacket(
                    payload_type,
                    is_last_nal and chunk_index == len(chunks) - 1,
                    current_sequence,
                    access_unit.rtp_timestamp,
                    ssrc,
                    bytes((fu_indicator, fu_header)) + chunk,
                )
            )
            current_sequence = (current_sequence + 1) & 0xFFFF
    return packets, current_sequence


def split_annex_b(payload: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index <= len(payload) - 3:
        if payload[index : index + 4] == _ANNEX_B_START:
            starts.append((index, 4))
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    if not starts:
        return [payload] if payload else []
    return [
        payload[start + prefix : starts[item_index + 1][0] if item_index + 1 < len(starts) else len(payload)]
        for item_index, (start, prefix) in enumerate(starts)
        if start + prefix < (starts[item_index + 1][0] if item_index + 1 < len(starts) else len(payload))
    ]


@dataclass(frozen=True, slots=True)
class H264AccessUnit:
    timestamp: int
    payload: bytes
    has_sps: bool
    has_pps: bool
    keyframe: bool


class H264Depacketizer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._expected_sequence: int | None = None
        self._timestamp: int | None = None
        self._nal_units: list[bytes] = []
        self._fragment: bytearray | None = None
        self._damaged = False

    def push(self, packet: RtpPacket) -> tuple[H264AccessUnit | None, int]:
        lost_packets = 0
        if self._expected_sequence is not None and packet.sequence != self._expected_sequence:
            lost_packets = (packet.sequence - self._expected_sequence) & 0xFFFF
            self._discard_access_unit()
        self._expected_sequence = (packet.sequence + 1) & 0xFFFF
        if self._timestamp is not None and packet.timestamp != self._timestamp:
            self._discard_access_unit()
        self._timestamp = packet.timestamp

        nal_type = packet.payload[0] & 0x1F
        if 1 <= nal_type <= 23:
            self._nal_units.append(packet.payload)
        elif nal_type == 24:
            self._append_stap_a(packet.payload)
        elif nal_type == 28:
            self._append_fu_a(packet.payload)
        else:
            self._damaged = True

        if not packet.marker:
            return None, lost_packets
        if self._fragment is not None:
            self._damaged = True
        access_unit = None
        if not self._damaged and self._nal_units:
            nal_types = [nal[0] & 0x1F for nal in self._nal_units]
            access_unit = H264AccessUnit(
                packet.timestamp,
                b"".join(_ANNEX_B_START + nal for nal in self._nal_units),
                7 in nal_types,
                8 in nal_types,
                5 in nal_types,
            )
        self._discard_access_unit()
        return access_unit, lost_packets

    def _append_stap_a(self, payload: bytes) -> None:
        offset = 1
        while offset + 2 <= len(payload):
            size = struct.unpack_from("!H", payload, offset)[0]
            offset += 2
            if size == 0 or offset + size > len(payload):
                self._damaged = True
                return
            self._nal_units.append(payload[offset : offset + size])
            offset += size
        if offset != len(payload):
            self._damaged = True

    def _append_fu_a(self, payload: bytes) -> None:
        if len(payload) < 3:
            self._damaged = True
            return
        indicator, header = payload[0], payload[1]
        start = bool(header & 0x80)
        end = bool(header & 0x40)
        if start:
            self._fragment = bytearray(((indicator & 0xE0) | (header & 0x1F),))
        elif self._fragment is None:
            self._damaged = True
            return
        assert self._fragment is not None
        self._fragment.extend(payload[2:])
        if end:
            self._nal_units.append(bytes(self._fragment))
            self._fragment = None

    def _discard_access_unit(self) -> None:
        self._timestamp = None
        self._nal_units.clear()
        self._fragment = None
        self._damaged = False


class DatagramSender(Protocol):
    def sendto(self, data: bytes, endpoint: tuple[str, int]) -> int: ...

    def close(self) -> None: ...


class DatagramReceiver(Protocol):
    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...

    def settimeout(self, timeout: float) -> None: ...

    def close(self) -> None: ...


class H264RtpSender:
    """Per-stream sender whose frame queue never blocks the producer."""

    def __init__(
        self,
        *,
        stream_id: str,
        endpoint: tuple[str, int],
        ssrc: int,
        queue_capacity: int,
        payload_type: int = 96,
        max_datagram_size: int = 1200,
        selected_backend: str = "software",
        datagram_sender: DatagramSender | None = None,
        initial_sequence: int | None = None,
        on_sent: Callable[[EncodedPacket], None] | None = None,
    ) -> None:
        if not stream_id or not endpoint[0] or not 1 <= endpoint[1] <= 65535:
            raise ValueError("RTP sender requires stream identity and a valid endpoint")
        if queue_capacity <= 0 or max_datagram_size <= _RTP_HEADER.size + 2:
            raise ValueError("RTP sender queue and datagram size must be positive")
        self.stream_id = stream_id
        self.endpoint = endpoint
        self.ssrc = ssrc
        self.payload_type = payload_type
        self.max_payload_size = max_datagram_size - _RTP_HEADER.size
        self.selected_backend = selected_backend
        self._queue_capacity = queue_capacity
        self._queue: deque[EncodedPacket] = deque()
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._socket = datagram_sender or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = secrets.randbelow(1 << 16) if initial_sequence is None else initial_sequence
        self._on_sent = on_sent
        self._state = StreamLifecycleState.CONFIGURED
        self._metrics = StreamMetrics()
        self._last_error = ""
        self._stopping = False
        self._thread: threading.Thread | None = None
        # Bumped by reset() so an access unit dequeued before a session
        # rollover no longer fires its on_sent callback afterwards; the
        # callback would otherwise re-populate bookkeeping that the rollover
        # just cleared, mixing old-session data into the new session.
        self._epoch = 0

    @property
    def status(self) -> StreamStatus:
        with self._condition:
            return StreamStatus(
                self.stream_id,
                self._state,
                self._state is StreamLifecycleState.READY,
                self.selected_backend,
                self._metrics,
                self._last_error,
            )

    def start(self) -> None:
        with self._condition:
            if self._state is not StreamLifecycleState.CONFIGURED:
                raise VideoRtpError("invalid_state", f"sender is {self._state.value}", stream_id=self.stream_id)
            self._state = StreamLifecycleState.STARTING
            self._thread = threading.Thread(target=self._run, name=f"rtp-send-{self.stream_id}", daemon=True)
            self._thread.start()

    def enqueue(self, access_unit: EncodedPacket) -> None:
        with self._condition:
            if self._state in {StreamLifecycleState.FAILED, StreamLifecycleState.STOPPED}:
                raise VideoRtpError("invalid_state", f"sender is {self._state.value}", stream_id=self.stream_id)
            dropped = 0
            if len(self._queue) >= self._queue_capacity:
                self._queue.popleft()
                dropped = 1
            self._queue.append(access_unit)
            self._metrics = replace(
                self._metrics,
                queued_frames=len(self._queue),
                dropped_frames=self._metrics.dropped_frames + dropped,
            )
            self._condition.notify()

    def reset(self) -> None:
        with self._send_lock, self._condition:
            self._queue.clear()
            self._epoch += 1
            self._metrics = replace(
                self._metrics,
                queued_frames=0,
                reconnect_count=self._metrics.reconnect_count + 1,
            )
            self._last_error = ""
            if self._state not in {StreamLifecycleState.CONFIGURED, StreamLifecycleState.STOPPED}:
                self._state = StreamLifecycleState.STARTING

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._condition:
            self._stopping = True
            self._queue.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise VideoRtpError("shutdown_timeout", "sender did not stop in time", stream_id=self.stream_id)
        self._socket.close()
        with self._condition:
            self._state = StreamLifecycleState.STOPPED
            self._metrics = replace(self._metrics, queued_frames=0)

    def send_pending(self) -> bool:
        """Send one queued access unit; exposed for deterministic integration tests."""
        with self._send_lock:
            with self._condition:
                if not self._queue:
                    return False
                access_unit = self._queue.popleft()
                epoch = self._epoch
                self._metrics = replace(self._metrics, queued_frames=len(self._queue))
            try:
                packets, self._sequence = packetize_h264(
                    access_unit,
                    ssrc=self.ssrc,
                    payload_type=self.payload_type,
                    sequence=self._sequence,
                    max_payload_size=self.max_payload_size,
                )
                for packet in packets:
                    self._socket.sendto(packet.to_bytes(), self.endpoint)
            except OSError as exc:
                with self._condition:
                    self._state = StreamLifecycleState.FAILED
                    self._last_error = str(exc)
                    self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
                raise VideoRtpError("send_failed", str(exc), stream_id=self.stream_id, recoverable=True) from exc
            with self._condition:
                self._state = StreamLifecycleState.READY
                self._metrics = replace(
                    self._metrics,
                    sent_frames=self._metrics.sent_frames + 1,
                    sent_packets=self._metrics.sent_packets + len(packets),
                )
            # The epoch check stays inside _send_lock so reset() cannot rotate
            # the session between the send and the callback, which would let a
            # retired access unit re-populate bookkeeping the rollover cleared.
            if self._on_sent is not None and epoch == self._epoch:
                self._on_sent(access_unit)
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    if self._state is StreamLifecycleState.STARTING:
                        self._state = StreamLifecycleState.READY
                    self._condition.wait()
                if self._stopping:
                    return
            try:
                self.send_pending()
            except VideoRtpError:
                return


class H264RtpReceiver:
    """Validate, reconstruct, decode, map, and buffer one H.264 RTP stream."""

    def __init__(
        self,
        *,
        stream_id: str,
        observation_key: str,
        ssrc: int,
        decoder: VideoDecoder,
        frame_buffer: StreamBuffer,
        timestamp_mapper: RtpTimestampMapper,
        session_generation: int,
        packet_queue_capacity: int,
        payload_type: int = 96,
        selected_backend: str = "software",
        endpoint: tuple[str, int] | None = None,
        datagram_receiver: DatagramReceiver | None = None,
        max_datagram_size: int = 65535,
        recorder: H264StreamRecorder | None = None,
        decode: bool = True,
    ) -> None:
        if not stream_id or not observation_key or session_generation < 1 or packet_queue_capacity <= 0:
            raise ValueError("RTP receiver requires stream identity, session, and positive queue capacity")
        self.stream_id = stream_id
        self.observation_key = observation_key
        self.ssrc = ssrc
        self.payload_type = payload_type
        self.decoder = decoder
        self.frame_buffer = frame_buffer
        self.timestamp_mapper = timestamp_mapper
        self.session_generation = session_generation
        self.selected_backend = selected_backend
        self._recorder = recorder
        self._decode = decode
        self._frame_count = 0
        self._recording_generation: int | None = None
        if max_datagram_size <= _RTP_HEADER.size:
            raise ValueError("max_datagram_size must exceed the RTP header size")
        if datagram_receiver is not None and endpoint is None:
            raise ValueError("an injected datagram receiver requires an endpoint")
        if endpoint is not None and (not endpoint[0] or not 1 <= endpoint[1] <= 65535):
            raise ValueError("RTP receiver endpoint must have a host and port in 1..65535")
        self.endpoint = endpoint
        self._socket = datagram_receiver
        self._max_datagram_size = max_datagram_size
        self._capacity = packet_queue_capacity
        self._queue: deque[tuple[bytes, int]] = deque()
        self._condition = threading.Condition(threading.RLock())
        self._lock = self._condition
        self._processing_lock = threading.RLock()
        self._depacketizer = H264Depacketizer()
        self._state = StreamLifecycleState.CONFIGURED
        self._metrics = StreamMetrics()
        self._last_error = ""
        self._have_sps = False
        self._have_pps = False
        self._keyframe_ready = False
        self._stopping = False
        self._receive_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

    @property
    def status(self) -> StreamStatus:
        with self._lock:
            return StreamStatus(
                self.stream_id,
                self._state,
                self._state is StreamLifecycleState.READY,
                self.selected_backend,
                self._metrics,
                self._last_error,
            )

    def start(self) -> None:
        with self._lock:
            if self._state is not StreamLifecycleState.CONFIGURED:
                raise VideoRtpError("invalid_state", f"receiver is {self._state.value}", stream_id=self.stream_id)
            self._state = StreamLifecycleState.STARTING
            if self.endpoint is not None:
                if self._socket is None:
                    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    udp_socket.bind(self.endpoint)
                    self._socket = udp_socket
                self._socket.settimeout(0.1)
                self._receive_thread = threading.Thread(
                    target=self._receive_loop,
                    name=f"rtp-recv-{self.stream_id}",
                    daemon=True,
                )
                self._process_thread = threading.Thread(
                    target=self._process_loop,
                    name=f"rtp-decode-{self.stream_id}",
                    daemon=True,
                )
                self._receive_thread.start()
                self._process_thread.start()
            self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME

    def enqueue_datagram(self, datagram: bytes, *, receive_time_ns: int | None = None) -> None:
        with self._lock:
            if self._state in {StreamLifecycleState.FAILED, StreamLifecycleState.STOPPED}:
                raise VideoRtpError("invalid_state", f"receiver is {self._state.value}", stream_id=self.stream_id)
            dropped = 0
            if len(self._queue) >= self._capacity:
                self._queue.popleft()
                dropped = 1
            self._queue.append((datagram, time.time_ns() if receive_time_ns is None else receive_time_ns))
            self._metrics = replace(
                self._metrics,
                queued_packets=len(self._queue),
                dropped_packets=self._metrics.dropped_packets + dropped,
            )
            self._condition.notify()

    def process_pending(self) -> bool:
        with self._processing_lock:
            pending = self._pop_pending()
            if pending is None:
                return False
            datagram, receive_time_ns = pending
            self._process_datagram_locked(datagram, receive_time_ns=receive_time_ns)
            return True

    def _pop_pending(self) -> tuple[bytes, int] | None:
        with self._lock:
            if not self._queue:
                return None
            pending = self._queue.popleft()
            self._metrics = replace(self._metrics, queued_packets=len(self._queue))
            return pending

    def process_datagram(self, datagram: bytes, *, receive_time_ns: int) -> list[VideoFrame]:
        with self._processing_lock:
            return self._process_datagram_locked(datagram, receive_time_ns=receive_time_ns)

    def _process_datagram_locked(self, datagram: bytes, *, receive_time_ns: int) -> list[VideoFrame]:
        try:
            packet = RtpPacket.from_bytes(datagram)
        except ValueError as exc:
            self._degrade("invalid_rtp_packet", str(exc), dropped_packets=1)
            return []
        if packet.ssrc != self.ssrc or packet.payload_type != self.payload_type:
            self._degrade(
                "stream_identity_mismatch", "RTP SSRC or payload type does not match descriptor", dropped_packets=1
            )
            return []
        with self._lock:
            self._metrics = replace(self._metrics, received_packets=self._metrics.received_packets + 1)
        access_unit, lost_packets = self._depacketizer.push(packet)
        if lost_packets:
            self._have_sps = False
            self._have_pps = False
            self._keyframe_ready = False
            self._degrade("packet_loss", f"lost {lost_packets} RTP packets", lost_packets=lost_packets)
        if access_unit is None:
            if lost_packets:
                self._record_access_unit(
                    b"",
                    capture_timestamp_ns=None,
                    rtp_timestamp=packet.timestamp,
                    keyframe=False,
                    lost_packets=lost_packets,
                    dropped="rtp_sequence_gap",
                )
            return []
        self._have_sps = self._have_sps or access_unit.has_sps
        self._have_pps = self._have_pps or access_unit.has_pps
        if not self._keyframe_ready:
            if not (access_unit.keyframe and self._have_sps and self._have_pps):
                with self._lock:
                    if self._state is not StreamLifecycleState.DEGRADED:
                        self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
                return []
            self._keyframe_ready = True
        try:
            capture_timestamp_ns = self.timestamp_mapper.map(
                access_unit.timestamp,
                now_ns=receive_time_ns,
                session_generation=self.session_generation,
            )
        except ObservationSynchronizationError as exc:
            self._record_access_unit(
                access_unit.payload,
                capture_timestamp_ns=None,
                rtp_timestamp=access_unit.timestamp,
                keyframe=access_unit.keyframe,
                lost_packets=lost_packets,
                dropped="timestamp_unmapped",
            )
            self._keyframe_ready = False
            self._degrade("timestamp_mapping_unavailable", str(exc))
            return []
        self._record_access_unit(
            access_unit.payload,
            capture_timestamp_ns=capture_timestamp_ns,
            rtp_timestamp=access_unit.timestamp,
            keyframe=access_unit.keyframe,
            lost_packets=lost_packets,
        )
        if not self._decode:
            with self._lock:
                self._state = StreamLifecycleState.READY
                self._last_error = ""
            return []
        try:
            frames = self.decoder.decode(
                EncodedPacket(
                    access_unit.payload,
                    access_unit.timestamp,
                    capture_timestamp_ns,
                    keyframe=access_unit.keyframe,
                )
            )
        except VideoCodecError as exc:
            self._keyframe_ready = False
            self._have_sps = False
            self._have_pps = False
            self._degrade("decode_failed", str(exc), decode_errors=1)
            return []
        received_frames = []
        for frame in frames:
            received_frame = replace(
                frame,
                receive_timestamp_ns=receive_time_ns,
            )
            self.frame_buffer.push(
                received_frame.capture_timestamp_ns,
                received_frame,
                receive_time_ns=receive_time_ns,
            )
            received_frames.append(received_frame)
        with self._lock:
            if frames:
                self._state = StreamLifecycleState.READY
                self._last_error = ""
            elif self._state not in {StreamLifecycleState.READY, StreamLifecycleState.DEGRADED}:
                # Pipelined hardware decoders may accept an access unit without
                # producing its frame synchronously. Once a frame has made the
                # stream ready, an empty drain is not a new keyframe boundary.
                self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
            self._metrics = replace(self._metrics, decoded_frames=self._metrics.decoded_frames + len(frames))
        return received_frames

    def reset(self, session_generation: int) -> None:
        if session_generation < 1:
            raise ValueError("session_generation must be positive")
        with self._processing_lock, self._lock:
            self._queue.clear()
            self._depacketizer.reset()
            self.decoder.reset()
            self.frame_buffer.reset()
            self.timestamp_mapper.reset(session_generation)
            self.session_generation = session_generation
            self._frame_count = 0
            self._have_sps = False
            self._have_pps = False
            self._keyframe_ready = False
            self._last_error = ""
            self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
            self._metrics = replace(
                self._metrics,
                queued_packets=0,
                reconnect_count=self._metrics.reconnect_count + 1,
            )

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._socket is not None:
            self._socket.close()
        deadline = time.monotonic() + timeout_s
        for thread in (self._receive_thread, self._process_thread):
            if thread is None:
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise VideoRtpError("shutdown_timeout", "receiver did not stop in time", stream_id=self.stream_id)
        with self._processing_lock:
            self.decoder.close(timeout_s)
        with self._lock:
            self._queue.clear()
            self._state = StreamLifecycleState.STOPPED
            self._metrics = replace(self._metrics, queued_packets=0)

    def _record_access_unit(
        self,
        payload: bytes,
        *,
        capture_timestamp_ns: int | None,
        rtp_timestamp: int,
        keyframe: bool,
        lost_packets: int,
        dropped: str | None = None,
    ) -> None:
        """Record one reconstructed access unit without disrupting reception on I/O errors."""
        if self._recorder is None:
            return
        recording_generation = self._recorder.recording_generation()
        if recording_generation is None:
            self._recording_generation = None
            return
        if recording_generation != self._recording_generation:
            self._frame_count = 0
            self._recording_generation = recording_generation
        frame_index = self._frame_count
        self._frame_count += 1
        try:
            self._recorder.write_access_unit(
                payload=payload,
                capture_timestamp_ns=capture_timestamp_ns,
                rtp_timestamp=rtp_timestamp,
                frame_index=frame_index,
                keyframe=keyframe,
                lost_packets=lost_packets,
                session_generation=self.session_generation,
                dropped=dropped,
            )
        except (OSError, ValueError) as exc:
            with self._lock:
                self._last_error = f"recording_failed: {exc}"

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while True:
            with self._lock:
                if self._stopping:
                    return
            try:
                datagram, _source = self._socket.recvfrom(self._max_datagram_size)
            except TimeoutError:
                continue
            except OSError as exc:
                with self._lock:
                    if self._stopping:
                        return
                self.fail(exc)
                return
            self.enqueue_datagram(datagram)

    def _process_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
            try:
                self.process_pending()
            except Exception as exc:
                self.fail(exc)
                return

    def fail(self, error: Exception) -> None:
        with self._lock:
            self._state = StreamLifecycleState.FAILED
            self._last_error = str(error)
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)

    def _degrade(
        self,
        code: str,
        message: str,
        *,
        dropped_packets: int = 0,
        lost_packets: int = 0,
        decode_errors: int = 0,
    ) -> None:
        with self._lock:
            self._state = StreamLifecycleState.DEGRADED
            self._last_error = f"{code}: {message}"
            self._metrics = replace(
                self._metrics,
                dropped_packets=self._metrics.dropped_packets + dropped_packets,
                lost_packets=self._metrics.lost_packets + lost_packets,
                decode_errors=self._metrics.decode_errors + decode_errors,
                errors=self._metrics.errors + 1,
            )
