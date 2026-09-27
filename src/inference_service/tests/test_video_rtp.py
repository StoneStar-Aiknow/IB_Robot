from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass

import numpy as np
import pytest

from inference_service.h264_stream_recorder import H264StreamRecorder
from inference_service.observation_sync import RtpTimestampMapper
from inference_service.software_video_codec import SoftwareH264Decoder, SoftwareH264Encoder
from inference_service.video_codec import CodecLifecycleState, CodecMetrics, EncodedPacket, VideoDecoder, VideoFrame
from inference_service.video_rtp import (
    H264Depacketizer,
    H264RtpReceiver,
    H264RtpSender,
    RtpPacket,
    StreamLifecycleState,
    VideoRtpError,
    packetize_h264,
    split_annex_b,
)
from observation_transport.rtp_sender import PACKETIZATION_ACCESS_UNIT, packetize_h264_datagrams
from robot_config.contract_utils import StreamBuffer

pytest.importorskip("av")

_SSRC = 0x1020_3040
_ENDPOINT = ("127.0.0.1", 5004)


@dataclass
class _MemoryDatagramSender:
    datagrams: list[bytes]
    closed: bool = False

    def sendto(self, data: bytes, _endpoint: tuple[str, int]) -> int:
        self.datagrams.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FailingDatagramSender(_MemoryDatagramSender):
    def sendto(self, _data: bytes, _endpoint: tuple[str, int]) -> int:
        raise OSError("network unavailable")


class _OneFrameDelayedDecoder(VideoDecoder):
    """Return each packet's frame on the next decode call."""

    def __init__(self) -> None:
        self._pending: EncodedPacket | None = None

    @property
    def state(self) -> CodecLifecycleState:
        return CodecLifecycleState.RUNNING

    @property
    def metrics(self) -> CodecMetrics:
        return CodecMetrics()

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        output = self._pending
        self._pending = packet
        if output is None:
            return []
        return [
            VideoFrame(
                np.zeros((48, 64, 3), dtype=np.uint8),
                output.capture_timestamp_ns,
                output.capture_timestamp_ns,
                64,
                48,
                "rgb24",
                keyframe=output.keyframe,
            )
        ]

    def reset(self) -> None:
        self._pending = None

    def close(self, timeout_s: float = 1.0) -> None:
        self._pending = None


class _ResetCountingDecoder(VideoDecoder):
    """Wrap a decoder and count reset() invocations."""

    def __init__(self, inner: VideoDecoder) -> None:
        self._inner = inner
        self.reset_count = 0

    @property
    def state(self) -> CodecLifecycleState:
        return self._inner.state

    @property
    def metrics(self) -> CodecMetrics:
        return self._inner.metrics

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        return self._inner.decode(packet)

    def reset(self) -> None:
        self.reset_count += 1
        self._inner.reset()

    def close(self, timeout_s: float = 1.0) -> None:
        self._inner.close(timeout_s)


class _StarvedDecoder(VideoDecoder):
    """Accept every access unit and never produce output."""

    def __init__(self, reset_error: Exception | None = None) -> None:
        self.reset_count = 0
        self._reset_error = reset_error

    @property
    def state(self) -> CodecLifecycleState:
        return CodecLifecycleState.RUNNING

    @property
    def metrics(self) -> CodecMetrics:
        return CodecMetrics()

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        return []

    def reset(self) -> None:
        self.reset_count += 1
        if self._reset_error is not None:
            raise self._reset_error

    def close(self, timeout_s: float = 1.0) -> None:
        pass


def test_rtp_packet_round_trip_validates_fixed_header_and_identity():
    packet = RtpPacket(96, True, 65535, 0xFFFF_FFFE, _SSRC, b"payload")

    parsed = RtpPacket.from_bytes(packet.to_bytes())

    assert parsed == packet
    with pytest.raises(ValueError, match="version"):
        RtpPacket.from_bytes(bytes((0,)) + packet.to_bytes()[1:])


def test_split_annex_b_returns_payload_unchanged_without_a_start_code():
    assert split_annex_b(b"\x41no start code here") == [b"\x41no start code here"]
    assert split_annex_b(b"") == []


def test_split_annex_b_handles_three_and_four_byte_start_codes():
    payload = b"\x00\x00\x01" + b"\x67abc" + b"\x00\x00\x00\x01" + b"\x65defg"

    assert split_annex_b(payload) == [b"\x67abc", b"\x65defg"]


def test_split_annex_b_treats_an_extra_leading_zero_as_a_four_byte_start_code():
    # 00 00 00 00 01 must anchor at offset 1 so the surplus zero stays outside
    # the NAL body, matching how H.264 Annex-B trailing_zero_8bits is emitted.
    assert split_annex_b(b"\x00\x00\x00\x00\x01\x65payload") == [b"\x65payload"]


def test_split_annex_b_discards_bytes_before_the_first_start_code():
    assert split_annex_b(b"junk\x00\x00\x00\x01\x65body") == [b"\x65body"]


def test_split_annex_b_skips_empty_nal_units_between_adjacent_start_codes():
    payload = b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x65body" + b"\x00\x00\x00\x01"

    assert split_annex_b(payload) == [b"\x65body"]


def test_split_annex_b_splits_a_megabyte_access_unit_without_per_byte_scanning():
    # The sender splits every access unit inline on the encode worker, so a
    # per-byte Python scan shows up directly as lost capture frame rate on the
    # edge board. 1 MiB must stay far below one frame period at 30 FPS.
    payload = b"".join(b"\x00\x00\x00\x01\x41" + bytes(4095) for _ in range(256))

    start = time.perf_counter()
    nal_units = split_annex_b(payload)
    elapsed_s = time.perf_counter() - start

    assert len(nal_units) == 256
    assert elapsed_s < 0.020, f"split_annex_b took {elapsed_s * 1000:.1f} ms for {len(payload)} bytes"


def test_h264_packetization_round_trip_handles_single_nal_and_fu_a():
    small = b"\x67" + b"s" * 8
    large = b"\x65" + bytes(range(256)) * 8
    access_unit = EncodedPacket(
        b"\x00\x00\x01" + small + b"\x00\x00\x00\x01" + large,
        90_000,
        1_000_000_000,
        keyframe=True,
    )

    packets, next_sequence = packetize_h264(
        access_unit,
        ssrc=_SSRC,
        payload_type=96,
        sequence=65534,
        max_payload_size=300,
    )
    depacketizer = H264Depacketizer()
    reconstructed = None
    for packet in packets:
        reconstructed, lost = depacketizer.push(RtpPacket.from_bytes(packet.to_bytes()))
        assert lost == 0

    assert next_sequence == (65534 + len(packets)) & 0xFFFF
    assert reconstructed is not None
    assert split_annex_b(reconstructed.payload) == [small, large]
    assert reconstructed.has_sps is True
    assert reconstructed.keyframe is True


def test_depacketizer_discards_incomplete_access_unit_after_packet_loss():
    access_unit = EncodedPacket(b"\x00\x00\x00\x01\x65" + b"x" * 2000, 90, 1_000, keyframe=True)
    packets, _ = packetize_h264(
        access_unit,
        ssrc=_SSRC,
        payload_type=96,
        sequence=10,
        max_payload_size=300,
    )
    depacketizer = H264Depacketizer()

    output = None
    total_lost = 0
    for packet in [packets[0], *packets[2:]]:
        output, lost = depacketizer.push(packet)
        total_lost += lost

    assert output is None
    assert total_lost == 1


def test_sender_queue_is_bounded_drop_oldest_and_never_blocks_producer():
    datagram_sender = _MemoryDatagramSender([])
    sender = _sender(datagram_sender, queue_capacity=2)
    first = _encoded(1, b"\x41first")
    second = _encoded(2, b"\x41second")
    third = _encoded(3, b"\x41third")

    sender.enqueue(first)
    sender.enqueue(second)
    sender.enqueue(third)

    assert sender.status.metrics.queued_frames == 2
    assert sender.status.metrics.dropped_frames == 1
    assert sender.send_pending()
    assert sender.send_pending()
    timestamps = [RtpPacket.from_bytes(item).timestamp for item in datagram_sender.datagrams]
    assert timestamps == [2, 3]
    sender.close()
    assert datagram_sender.closed is True
    assert sender.status.state is StreamLifecycleState.STOPPED


def test_sender_failure_is_observable_and_fail_closed():
    datagram_sender = _FailingDatagramSender([])
    sender = _sender(datagram_sender, queue_capacity=1)
    sender.enqueue(_encoded(1, b"\x41frame"))

    with pytest.raises(VideoRtpError, match="network unavailable"):
        sender.send_pending()

    assert sender.status.state is StreamLifecycleState.FAILED
    assert sender.status.ready is False
    assert sender.status.metrics.errors == 1
    assert "network unavailable" in sender.status.last_error
    sender.close()


def test_receiver_validates_stream_identity_and_bounds_packet_queue():
    receiver, _ = _receiver(packet_queue_capacity=2)
    receiver.start()
    wrong = RtpPacket(96, True, 1, 90, _SSRC + 1, b"\x41data").to_bytes()

    receiver.enqueue_datagram(wrong, receive_time_ns=1)
    receiver.enqueue_datagram(wrong, receive_time_ns=2)
    receiver.enqueue_datagram(wrong, receive_time_ns=3)

    assert receiver.status.metrics.queued_packets == 2
    assert receiver.status.metrics.dropped_packets == 1
    assert receiver.process_pending()
    assert receiver.status.metrics.queued_packets == 1
    assert receiver.status.state is StreamLifecycleState.DEGRADED
    assert receiver.status.metrics.errors == 1
    assert "stream_identity_mismatch" in receiver.status.last_error
    receiver.close()


def test_sender_reset_suppresses_on_sent_of_retired_access_units():
    """reset() drops queued access units and rotates the session epoch so an
    access unit already dequeued by a send in flight cannot fire on_sent
    afterwards: the late callback would re-populate bookkeeping that a
    session rollover just cleared."""
    callbacks: list[EncodedPacket] = []
    datagram_sender = _MemoryDatagramSender([])
    sender = H264RtpSender(
        stream_id="top",
        endpoint=_ENDPOINT,
        ssrc=_SSRC,
        queue_capacity=4,
        datagram_sender=datagram_sender,
        on_sent=callbacks.append,
    )
    sender.enqueue(_encoded(1, b"\x41first"))
    sender.enqueue(_encoded(2, b"\x41second"))

    epoch_before = sender._epoch
    sender.reset()

    # The rollover cleared the queue and advanced the epoch.
    assert sender._epoch == epoch_before + 1
    assert sender.status.metrics.queued_frames == 0

    # Nothing is left to send, and no retired access unit fires on_sent.
    assert not sender.send_pending()
    assert callbacks == []

    # A fresh access unit of the new session flows normally.
    sender.enqueue(_encoded(3, b"\x41third"))
    assert sender.send_pending()
    assert [packet.rtp_timestamp for packet in callbacks] == [3]
    sender.close()


def test_software_rtp_interoperability_preserves_count_timestamps_and_quality():
    encoder = _encoder(gop_frames=2)
    datagram_sender = _MemoryDatagramSender([])
    sender = _sender(datagram_sender, queue_capacity=2)
    receiver, buffer = _receiver()
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)
    source_frames = []

    for index in range(5):
        capture_ns = 1_000_000_000 + index * 50_000_000
        image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        source_frames.append(image)
        for access_unit in encoder.encode(VideoFrame(image, capture_ns, capture_ns, 64, 48, "rgb24")):
            sender.enqueue(access_unit)
            sender.send_pending()

    decoded = _deliver(datagram_sender.datagrams, receiver, start_receive_ns=2_000_000_000)

    assert len(decoded) == 5
    assert [item[0] for item in buffer.history] == [
        1_000_000_000,
        1_050_000_000,
        1_100_000_000,
        1_150_000_000,
        1_200_000_000,
    ]
    assert receiver.status.state is StreamLifecycleState.READY
    assert receiver.status.metrics.decoded_frames == 5
    assert (
        max(
            np.mean(np.abs(frame.data.astype(np.int16) - source.astype(np.int16)))
            for frame, source in zip(decoded, source_frames, strict=True)
        )
        < 3
    )
    encoder.close()
    sender.close()
    receiver.close()


def test_receiver_preserves_capture_timestamp_from_delayed_decoder_output():
    datagrams, encoder, sender = _encoded_stream(frame_count=2, gop_frames=1)
    receiver, buffer = _receiver(decoder=_OneFrameDelayedDecoder())
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    decoded = _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert len(decoded) == 1
    assert decoded[0].capture_timestamp_ns == 1_000_000_000
    assert buffer.history[0][0] == 1_000_000_000
    assert receiver.status.state is StreamLifecycleState.READY
    encoder.close()
    sender.close()
    receiver.close()


def test_delayed_decoder_output_is_fresh_from_decode_availability():
    datagrams, encoder, sender = _encoded_stream(frame_count=2, gop_frames=1)
    decode_available_ns = 2_000_700_000
    receiver, buffer = _receiver(decoder=_OneFrameDelayedDecoder(), clock=lambda: decode_available_ns)
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    decoded = _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert len(decoded) == 1
    assert decoded[0].capture_timestamp_ns == 1_000_000_000
    assert decoded[0].receive_timestamp_ns == decode_available_ns
    entry, issue = buffer.select_entry(1_000_000_000, now_ns=decode_available_ns + 400_000_000)
    assert issue is None
    assert entry is not None
    assert entry[0] == 1_000_000_000
    assert entry[1] == decode_available_ns
    encoder.close()
    sender.close()
    receiver.close()


def test_late_join_waits_for_repeated_headers_and_next_idr():
    datagrams, encoder, sender = _encoded_stream(frame_count=5, gop_frames=2)
    receiver, buffer = _receiver()
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)
    packets = [RtpPacket.from_bytes(item) for item in datagrams]
    first_keyframe_timestamp = packets[0].timestamp
    late_datagrams = [
        item for item, packet in zip(datagrams, packets, strict=True) if packet.timestamp != first_keyframe_timestamp
    ]

    decoded = _deliver(late_datagrams, receiver, start_receive_ns=2_050_000_000)

    assert decoded
    assert decoded[0].keyframe is True
    assert buffer.history[0][0] == 1_100_000_000
    encoder.close()
    sender.close()
    receiver.close()


def test_packet_loss_degrades_stream_then_next_repeated_header_idr_recovers():
    datagrams, encoder, sender = _encoded_stream(frame_count=5, gop_frames=2, max_datagram_size=180)
    receiver, _ = _receiver()
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)
    packets = [RtpPacket.from_bytes(item) for item in datagrams]
    damaged_timestamp = sorted({packet.timestamp for packet in packets})[1]
    dropped = False
    delivered = []
    for datagram, packet in zip(datagrams, packets, strict=True):
        if packet.timestamp == damaged_timestamp and not dropped:
            dropped = True
            continue
        delivered.append(datagram)

    decoded = _deliver(delivered, receiver, start_receive_ns=2_000_000_000)

    assert receiver.status.state is StreamLifecycleState.READY
    assert receiver.status.metrics.lost_packets == 1
    assert receiver.status.metrics.decoded_frames >= 3
    assert any(frame.keyframe and frame.capture_timestamp_ns >= 1_100_000_000 for frame in decoded)
    encoder.close()
    sender.close()
    receiver.close()


def test_keyframe_recovery_after_loss_does_not_reset_decoder():
    datagrams, encoder, sender = _encoded_stream(frame_count=5, gop_frames=2, max_datagram_size=180)
    decoder = _ResetCountingDecoder(SoftwareH264Decoder())
    receiver, _ = _receiver(decoder=decoder)
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)
    packets = [RtpPacket.from_bytes(item) for item in datagrams]
    damaged_timestamp = sorted({packet.timestamp for packet in packets})[1]
    dropped = False
    delivered = []
    for datagram, packet in zip(datagrams, packets, strict=True):
        if packet.timestamp == damaged_timestamp and not dropped:
            dropped = True
            continue
        delivered.append(datagram)

    _deliver(delivered, receiver, start_receive_ns=2_000_000_000)

    assert decoder.reset_count == 0
    assert receiver.status.state is StreamLifecycleState.READY
    assert receiver.status.metrics.decoded_frames >= 3
    encoder.close()
    sender.close()
    receiver.close()


def test_starved_decoder_triggers_rate_limited_reset():
    datagrams, encoder, sender = _encoded_stream(frame_count=40, gop_frames=1, max_datagram_size=180)
    decoder = _StarvedDecoder()
    receiver, _ = _receiver(decoder=decoder)
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert decoder.reset_count == 1
    assert receiver.status.state is StreamLifecycleState.WAITING_FOR_KEYFRAME
    encoder.close()
    sender.close()
    receiver.close()


def test_starved_decoder_reset_failure_degrades_stream():
    datagrams, encoder, sender = _encoded_stream(frame_count=40, gop_frames=1, max_datagram_size=180)
    from observation_transport.video_codec import VideoCodecError

    decoder = _StarvedDecoder(reset_error=VideoCodecError("decode_failed", "decoder wedged", backend="software"))
    receiver, _ = _receiver(decoder=decoder)
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert decoder.reset_count == 1
    assert receiver.status.state is StreamLifecycleState.DEGRADED
    assert receiver.status.metrics.decode_errors == 1
    encoder.close()
    sender.close()
    receiver.close()


def test_receiver_reset_clears_buffer_mapping_and_readiness():
    receiver, buffer = _receiver()
    receiver.start()
    receiver.timestamp_mapper.update(90, 1_000, 2_000, session_generation=1)
    buffer.push(1_000, object(), receive_time_ns=2_000)

    receiver.reset(2)

    assert buffer.history == []
    assert receiver.timestamp_mapper.ready is False
    assert receiver.status.state is StreamLifecycleState.WAITING_FOR_KEYFRAME
    assert receiver.status.metrics.reconnect_count == 1
    receiver.close()


def test_recording_only_receiver_preserves_source_timestamps_without_mapping(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    recorder.start_episode(tmp_path, "observation.images.top")
    receiver, _ = _receiver(recorder=recorder, decode=False)
    receiver.start()
    encoder = _encoder(gop_frames=1)
    memory = _MemoryDatagramSender([])
    sender = _sender(memory, queue_capacity=2)

    capture_ns = 1_000_000_000
    receiver.timestamp_mapper.update(90_000, capture_ns, 2_000_000_000, session_generation=1)
    for packet in encoder.encode(
        VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), capture_ns, capture_ns, 64, 48, "rgb24")
    ):
        sender.enqueue(packet)
        sender.send_pending()
    assert _deliver(memory.datagrams, receiver, start_receive_ns=2_000_000_000) == []

    receiver.timestamp_mapper.reset(1)
    memory.datagrams.clear()
    capture_ns += 50_000_000
    for packet in encoder.encode(
        VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), capture_ns, capture_ns, 64, 48, "rgb24")
    ):
        sender.enqueue(packet)
        sender.send_pending()
    _deliver(memory.datagrams, receiver, start_receive_ns=2_050_000_000)

    assert recorder.stop_episode() is True
    sidecar = tmp_path / "observation.images.top.h264.json"
    entries = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert entries[0]["capture_timestamp_ns"] == 1_000_000_000
    assert entries[1]["capture_timestamp_ns"] == 1_050_000_000
    assert not entries[1].get("dropped")
    encoder.close()
    sender.close()
    receiver.close()


def test_recording_receiver_resets_frame_index_for_each_episode(tmp_path):
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    receiver, _ = _receiver(recorder=recorder, decode=False)
    receiver.start()
    encoder = _encoder(gop_frames=1)
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    for episode in range(2):
        episode_dir = tmp_path / str(episode)
        recorder.start_episode(episode_dir, "observation.images.top")
        memory = _MemoryDatagramSender([])
        sender = _sender(memory, queue_capacity=2)
        capture_ns = 1_000_000_000 + episode * 50_000_000
        for packet in encoder.encode(
            VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), capture_ns, capture_ns, 64, 48, "rgb24")
        ):
            sender.enqueue(packet)
            sender.send_pending()
        _deliver(memory.datagrams, receiver, start_receive_ns=2_000_000_000 + episode * 50_000_000)
        assert recorder.stop_episode() is True
        sender.close()

    for episode in range(2):
        sidecar = tmp_path / str(episode) / "observation.images.top.h264.json"
        assert json.loads(sidecar.read_text().splitlines()[0])["frame_index"] == 0
    encoder.close()
    receiver.close()


@pytest.mark.parametrize(("integrity_mode", "kept"), [("strict", False), ("tolerant", True)])
def test_recording_packet_loss_injection_applies_integrity_policy(tmp_path, integrity_mode, kept):
    recorder = H264StreamRecorder(integrity_mode=integrity_mode)
    recorder.start_episode(tmp_path, "observation.images.top")
    receiver, _ = _receiver(recorder=recorder, decode=False)
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)
    datagrams, encoder, sender = _encoded_stream(frame_count=3, gop_frames=1, max_datagram_size=180)
    packets = [RtpPacket.from_bytes(item) for item in datagrams]
    damaged_timestamp = sorted({packet.timestamp for packet in packets})[1]
    dropped = False
    delivered = []
    for datagram, packet in zip(datagrams, packets, strict=True):
        if packet.timestamp == damaged_timestamp and not dropped:
            dropped = True
            continue
        delivered.append(datagram)

    _deliver(delivered, receiver, start_receive_ns=2_000_000_000)

    assert recorder.stop_episode() is kept
    sidecar = tmp_path / "observation.images.top.h264.json"
    if kept:
        entries = [json.loads(line) for line in sidecar.read_text().splitlines()]
        assert any(entry["lost_packets"] > 0 and entry["dropped"] == "rtp_sequence_gap" for entry in entries)
    else:
        assert not sidecar.exists()
    encoder.close()
    sender.close()
    receiver.close()


def test_access_unit_packetization_sends_one_datagram_per_access_unit():
    small = b"\x67" + b"s" * 8
    large = b"\x65" + bytes(range(256)) * 40
    access_unit = EncodedPacket(b"\x00\x00\x00\x01" + small + b"\x00\x00\x01" + large, 90_000, 1, keyframe=True)

    datagrams, next_sequence = packetize_h264_datagrams(
        access_unit,
        ssrc=_SSRC,
        payload_type=96,
        sequence=65535,
        max_payload_size=1188,
        packetization=PACKETIZATION_ACCESS_UNIT,
    )

    assert len(datagrams) == 1
    assert next_sequence == 0
    packet = RtpPacket.from_bytes(datagrams[0])
    assert packet.marker is True
    assert packet.payload == access_unit.payload
    reconstructed, lost = H264Depacketizer().push(packet)
    assert lost == 0
    assert reconstructed is not None
    assert reconstructed.payload == access_unit.payload
    assert (reconstructed.has_sps, reconstructed.has_pps, reconstructed.keyframe) == (True, False, True)


def test_access_unit_packetization_falls_back_to_fu_a_beyond_one_udp_datagram():
    access_unit = EncodedPacket(b"\x00\x00\x00\x01\x65" + bytes(70_000), 90, 1, keyframe=True)

    datagrams, _ = packetize_h264_datagrams(
        access_unit,
        ssrc=_SSRC,
        payload_type=96,
        sequence=0,
        max_payload_size=1388,
        packetization=PACKETIZATION_ACCESS_UNIT,
    )

    assert len(datagrams) > 1
    assert max(len(item) for item in datagrams) <= 1400
    depacketizer = H264Depacketizer()
    outputs = [depacketizer.push(RtpPacket.from_bytes(item))[0] for item in datagrams]
    assert outputs[-1] is not None
    assert outputs[-1].payload == access_unit.payload


def test_depacketizer_rejects_an_access_unit_datagram_inside_a_fragmented_access_unit():
    depacketizer = H264Depacketizer()
    fragment = RtpPacket(96, False, 1, 90, _SSRC, bytes((0x7C, 0x85)) + b"x" * 10)
    whole = RtpPacket(96, True, 2, 90, _SSRC, b"\x00\x00\x00\x01\x41body")

    assert depacketizer.push(fragment) == (None, 0)
    assert depacketizer.push(whole) == (None, 0)


def test_software_stream_decodes_through_access_unit_packetization():
    datagrams, encoder, sender = _encoded_stream(frame_count=5, gop_frames=2, packetization=PACKETIZATION_ACCESS_UNIT)
    receiver, buffer = _receiver()
    receiver.start()
    receiver.timestamp_mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    decoded = _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert len(datagrams) == 5
    assert len(decoded) == 5
    assert receiver.status.metrics.received_packets == 5
    assert receiver.status.state is StreamLifecycleState.READY
    encoder.close()
    sender.close()
    receiver.close()


def test_receiver_reports_every_lifecycle_transition_to_its_listener():
    transitions = []
    datagrams, encoder, sender = _encoded_stream(frame_count=3, gop_frames=2)
    buffer = StreamBuffer("hold", 50_000_000, max_age_ns=1_000_000_000, retention_ns=2_000_000_000)
    mapper = RtpTimestampMapper(2_000_000_000, observation_key="observation.images.top", stream_id="top")
    receiver = H264RtpReceiver(
        stream_id="top",
        observation_key="observation.images.top",
        ssrc=_SSRC,
        decoder=SoftwareH264Decoder(),
        frame_buffer=buffer,
        timestamp_mapper=mapper,
        session_generation=1,
        packet_queue_capacity=64,
        on_state_change=lambda: transitions.append(receiver.status.state),
    )
    receiver.start()
    mapper.update(90_000, 1_000_000_000, 2_000_000_000, session_generation=1)

    _deliver(datagrams, receiver, start_receive_ns=2_000_000_000)

    assert transitions == [
        StreamLifecycleState.STARTING,
        StreamLifecycleState.WAITING_FOR_KEYFRAME,
        StreamLifecycleState.READY,
    ]
    encoder.close()
    sender.close()
    receiver.close()


@pytest.mark.parametrize("packetization", ["rfc6184", PACKETIZATION_ACCESS_UNIT])
def test_local_udp_threads_deliver_bursts_without_waiting_for_the_poll_timeout(packetization):
    udp_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_receiver.bind(("127.0.0.1", 0))
    endpoint = udp_receiver.getsockname()
    buffer = StreamBuffer("hold", 50_000_000, max_age_ns=1_000_000_000, retention_ns=2_000_000_000)
    mapper = RtpTimestampMapper(2_000_000_000, observation_key="observation.images.top", stream_id="top")
    receiver = H264RtpReceiver(
        stream_id="top",
        observation_key="observation.images.top",
        ssrc=_SSRC,
        decoder=SoftwareH264Decoder(),
        frame_buffer=buffer,
        timestamp_mapper=mapper,
        session_generation=1,
        packet_queue_capacity=256,
        endpoint=endpoint,
        datagram_receiver=udp_receiver,
    )
    # A small datagram budget forces multi-packet RFC 6184 bursts.
    sender = H264RtpSender(
        stream_id="top",
        endpoint=endpoint,
        ssrc=_SSRC,
        queue_capacity=4,
        max_datagram_size=200,
        packetization=packetization,
    )
    encoder = _encoder(gop_frames=2)
    receiver.start()
    sender.start()
    mapper.update(90_000, 1_000_000_000, time.time_ns(), session_generation=1)

    started = time.monotonic()
    for index in range(4):
        capture_ns = 1_000_000_000 + index * 50_000_000
        image = np.random.default_rng(index).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for access_unit in encoder.encode(VideoFrame(image, capture_ns, capture_ns, 64, 48, "rgb24")):
            sender.enqueue(access_unit)
    deadline = started + 2.0
    while len(buffer) < 3 and time.monotonic() < deadline:
        time.sleep(0.002)
    elapsed_s = time.monotonic() - started

    assert len(buffer) >= 3
    assert elapsed_s < 0.1, f"burst delivery took {elapsed_s * 1000:.0f} ms"
    assert receiver.status.metrics.lost_packets == 0
    encoder.close()
    sender.close()
    receiver.close()
    assert receiver.status.state is StreamLifecycleState.STOPPED


def test_local_udp_sender_receiver_threads_deliver_stream_and_stop_cleanly():
    udp_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_receiver.bind(("127.0.0.1", 0))
    endpoint = udp_receiver.getsockname()
    buffer = StreamBuffer("hold", 50_000_000, max_age_ns=1_000_000_000, retention_ns=2_000_000_000)
    mapper = RtpTimestampMapper(
        2_000_000_000,
        observation_key="observation.images.top",
        stream_id="top",
    )
    receiver = H264RtpReceiver(
        stream_id="top",
        observation_key="observation.images.top",
        ssrc=_SSRC,
        decoder=SoftwareH264Decoder(),
        frame_buffer=buffer,
        timestamp_mapper=mapper,
        session_generation=1,
        packet_queue_capacity=64,
        endpoint=endpoint,
        datagram_receiver=udp_receiver,
    )
    sender = H264RtpSender(
        stream_id="top",
        endpoint=endpoint,
        ssrc=_SSRC,
        queue_capacity=2,
        initial_sequence=10,
    )
    encoder = _encoder(gop_frames=2)
    receiver.start()
    sender.start()
    mapper.update(90_000, 1_000_000_000, time.time_ns(), session_generation=1)

    for index in range(3):
        capture_ns = 1_000_000_000 + index * 50_000_000
        image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        for access_unit in encoder.encode(VideoFrame(image, capture_ns, capture_ns, 64, 48, "rgb24")):
            sender.enqueue(access_unit)

    deadline = time.monotonic() + 2.0
    while len(buffer) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert len(buffer) >= 2
    assert receiver.status.state is StreamLifecycleState.READY
    assert sender.status.state is StreamLifecycleState.READY
    encoder.close()
    sender.close()
    receiver.close()
    assert sender.status.state is StreamLifecycleState.STOPPED
    assert receiver.status.state is StreamLifecycleState.STOPPED


def _encoded(timestamp: int, nal: bytes) -> EncodedPacket:
    return EncodedPacket(b"\x00\x00\x00\x01" + nal, timestamp, timestamp * 1000)


def _sender(
    datagram_sender: _MemoryDatagramSender,
    *,
    queue_capacity: int,
    max_datagram_size: int = 1200,
    packetization: str = "rfc6184",
) -> H264RtpSender:
    return H264RtpSender(
        stream_id="top",
        endpoint=_ENDPOINT,
        ssrc=_SSRC,
        queue_capacity=queue_capacity,
        datagram_sender=datagram_sender,
        initial_sequence=10,
        max_datagram_size=max_datagram_size,
        packetization=packetization,
    )


def _receiver(
    *,
    packet_queue_capacity: int = 64,
    recorder=None,
    decode: bool = True,
    decoder: VideoDecoder | None = None,
    clock=None,
) -> tuple[H264RtpReceiver, StreamBuffer]:
    buffer = StreamBuffer("hold", 50_000_000, max_age_ns=1_000_000_000, retention_ns=2_000_000_000)
    mapper = RtpTimestampMapper(
        2_000_000_000,
        observation_key="observation.images.top",
        stream_id="top",
    )
    return (
        H264RtpReceiver(
            stream_id="top",
            observation_key="observation.images.top",
            ssrc=_SSRC,
            decoder=decoder or SoftwareH264Decoder(),
            frame_buffer=buffer,
            timestamp_mapper=mapper,
            session_generation=1,
            packet_queue_capacity=packet_queue_capacity,
            recorder=recorder,
            decode=decode,
            clock=clock,
        ),
        buffer,
    )


def _encoder(*, gop_frames: int) -> SoftwareH264Encoder:
    return SoftwareH264Encoder(
        width=64,
        height=48,
        frame_rate_hz=20.0,
        bitrate_bps=300_000,
        gop_frames=gop_frames,
    )


def _encoded_stream(
    *,
    frame_count: int,
    gop_frames: int,
    max_datagram_size: int = 1200,
    packetization: str = "rfc6184",
) -> tuple[list[bytes], SoftwareH264Encoder, H264RtpSender]:
    encoder = _encoder(gop_frames=gop_frames)
    memory = _MemoryDatagramSender([])
    sender = _sender(memory, queue_capacity=2, max_datagram_size=max_datagram_size, packetization=packetization)
    for index in range(frame_count):
        capture_ns = 1_000_000_000 + index * 50_000_000
        rng = np.random.default_rng(index)
        image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for access_unit in encoder.encode(VideoFrame(image, capture_ns, capture_ns, 64, 48, "rgb24")):
            sender.enqueue(access_unit)
            sender.send_pending()
    return memory.datagrams, encoder, sender


def _deliver(datagrams: list[bytes], receiver: H264RtpReceiver, *, start_receive_ns: int) -> list[VideoFrame]:
    decoded = []
    for index, datagram in enumerate(datagrams):
        decoded.extend(receiver.process_datagram(datagram, receive_time_ns=start_receive_ns + index * 1_000_000))
    return decoded


def test_recording_receiver_keeps_frame_index_monotonic_across_session_resets(tmp_path):
    """A heartbeat flap re-handshakes the RTP session mid-episode.

    frame_index belongs to the recording, not to the RTP session, so it must keep
    counting across the reset. Restarting at 0 makes the whole episode unconvertible.
    """
    recorder = H264StreamRecorder(integrity_mode="tolerant")
    receiver, _ = _receiver(recorder=recorder, decode=False)
    receiver.start()
    encoder = _encoder(gop_frames=1)
    recorder.start_episode(tmp_path, "observation.images.top")

    for generation in (1, 2, 3):
        if generation > 1:
            receiver.reset(generation)  # what a heartbeat expiry triggers
        receiver.timestamp_mapper.update(
            90_000 * generation,
            1_000_000_000 * generation,
            2_000_000_000 * generation,
            session_generation=generation,
        )
        memory = _MemoryDatagramSender([])
        sender = _sender(memory, queue_capacity=2)
        capture_ns = 1_000_000_000 * generation
        for packet in encoder.encode(
            VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), capture_ns, capture_ns, 64, 48, "rgb24")
        ):
            sender.enqueue(packet)
            sender.send_pending()
        _deliver(memory.datagrams, receiver, start_receive_ns=2_000_000_000 * generation)
        sender.close()

    assert recorder.stop_episode() is True
    sidecar = tmp_path / "observation.images.top.h264.json"
    indices = [json.loads(line)["frame_index"] for line in sidecar.read_text().splitlines()]
    assert len(indices) == 3
    assert indices == list(range(len(indices)))
    encoder.close()
    receiver.close()


@pytest.mark.parametrize("packetization", ["rfc6184", "access_unit"])
def test_source_timestamp_survives_codec_and_wire_without_mapping(packetization):
    encoder = _encoder(gop_frames=1)
    receiver, buffer = _receiver()
    receiver.start()
    timestamps = [1_039_999_877, 1_100_000_321]
    try:
        sequence = 1
        for timestamp in timestamps:
            frame = VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), timestamp, timestamp, 64, 48, "rgb24")
            for encoded in encoder.encode(frame):
                datagrams, sequence = packetize_h264_datagrams(
                    encoded,
                    ssrc=_SSRC,
                    payload_type=96,
                    sequence=sequence,
                    max_payload_size=300,
                    packetization=packetization,
                )
                _deliver(datagrams, receiver, start_receive_ns=timestamp + 100_000)
        assert [entry[0] for entry in buffer.entries()] == timestamps
        assert not receiver.timestamp_mapper.ready
        assert receiver.status.ready
    finally:
        encoder.close()
        receiver.close()


def test_fragments_with_conflicting_source_timestamps_are_not_decoded():
    from dataclasses import replace

    encoded = EncodedPacket(b"\x00\x00\x00\x01\x65" + b"x" * 2000, 90000, 1_000_000_123)
    packets, _ = packetize_h264(encoded, ssrc=_SSRC, payload_type=96, sequence=1, max_payload_size=300)
    packets[1] = replace(packets[1], capture_timestamp_ns=1_000_000_124)
    depacketizer = H264Depacketizer()
    for packet in packets:
        access_unit, _ = depacketizer.push(packet)
        assert access_unit is None


def test_receiver_rejects_missing_source_timestamp():
    receiver, buffer = _receiver()
    try:
        receiver.start()
        packet = RtpPacket(96, True, 1, 90000, _SSRC, b"\x65body")
        assert receiver.process_datagram(packet.to_bytes(), receive_time_ns=1_000_000_000) == []
        assert not buffer.entries()
        assert receiver.status.last_error.startswith("capture_timestamp_missing:")
    finally:
        receiver.close()


@pytest.mark.parametrize("payload_size", [40, 2000, 65484, 65485, 70000])
@pytest.mark.parametrize("packetization", ["rfc6184", "access_unit"])
def test_capture_extension_respects_mtu_and_whole_access_unit_boundary(payload_size, packetization):
    encoded = EncodedPacket(b"\x00\x00\x00\x01\x65" + b"x" * (payload_size - 5), 0xFFFFFFFF, 1_790_000_000_123_456_789)
    wire, _ = packetize_h264_datagrams(
        encoded, ssrc=_SSRC, payload_type=96, sequence=65535, max_payload_size=1188, packetization=packetization
    )
    if packetization == "access_unit" and payload_size <= 65483:
        assert len(wire) == 1 and len(wire[0]) <= 65507
    else:
        assert max(map(len, wire)) <= 1200
    receiver = H264Depacketizer()
    for data in wire:
        packet = RtpPacket.from_bytes(data)
        assert packet.capture_timestamp_ns == encoded.capture_timestamp_ns
        output, lost = receiver.push(packet)
        assert lost == 0
    assert output.payload == encoded.payload
    assert output.capture_timestamp_ns == encoded.capture_timestamp_ns


def test_capture_extension_rejects_invalid_length_and_truncation():
    import struct

    packet = RtpPacket(96, True, 1, 90000, _SSRC, b"\x65body", 1_000_000_123)
    wire = bytearray(packet.to_bytes())
    struct.pack_into("!H", wire, 14, 1)
    with pytest.raises(ValueError, match="capture timestamp extension"):
        RtpPacket.from_bytes(bytes(wire))
    with pytest.raises(ValueError, match="capture timestamp extension"):
        RtpPacket.from_bytes(packet.to_bytes()[:20])


def test_sequence_gap_does_not_label_intact_recovery_frame_as_dropped():
    receiver, buffer = _receiver()
    encoder = _encoder(gop_frames=1)
    receiver.start()
    try:
        old_time, target = 1_000_000_123, 1_040_000_123
        for sequence, stamp in [(1, old_time), (10, target)]:
            image = VideoFrame(np.zeros((48, 64, 3), dtype=np.uint8), stamp, stamp, 64, 48, "rgb24")
            for encoded in encoder.encode(image):
                datagrams, _ = packetize_h264_datagrams(
                    encoded,
                    ssrc=_SSRC,
                    payload_type=96,
                    sequence=sequence,
                    max_payload_size=1188,
                    packetization="access_unit",
                )
                _deliver(datagrams, receiver, start_receive_ns=stamp + 1)
        assert buffer.entries()[-1][0] == target
        assert receiver.status.metrics.last_dropped_capture_timestamp_ns != target
        assert receiver.status.ready
    finally:
        encoder.close()
        receiver.close()
