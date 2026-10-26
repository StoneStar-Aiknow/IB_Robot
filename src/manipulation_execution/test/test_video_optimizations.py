import queue
import threading
from array import array
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage, Image

from manipulation_execution.compressed_image_decode import CompressedImageDecoder, byte_array, decode_compressed_bgr8
from manipulation_execution.motion_imitation_stream_node import MotionImitationStreamNode
from observation_transport.video_codec import CodecLifecycleState, VideoFrame


def encoded_message(codec=".jpg", width=64, height=48):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = [20, 120, 230]
    success, data = cv2.imencode(codec, image)
    assert success
    message = CompressedImage(format=codec[1:], data=byte_array(data))
    message.header.stamp.sec = 42
    message.header.frame_id = "camera"
    return message


def test_array_assignment_preserves_cpu_bytes_header_and_wire_contract():
    message = encoded_message()
    image = decode_compressed_bgr8(message)
    expected = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert isinstance(image.data, array)
    assert image.data.typecode == "B"
    assert bytes(image.data) == expected.tobytes()
    received = deserialize_message(serialize_message(image), Image)
    assert received == image
    assert received.header == message.header


def test_dvpp_preserves_packed_nv12_and_source_timestamp():
    calls = []
    fake = SimpleNamespace(
        state=CodecLifecycleState.RUNNING,
        decode=lambda packet: [
            VideoFrame(
                np.zeros((72, 64), np.uint8),
                packet.capture_timestamp_ns,
                1,
                64,
                48,
                "nv12",
                color_space="bt601",
                color_range="full",
            )
        ],
        close=lambda: calls.append("close"),
    )
    decoder = CompressedImageDecoder("dvpp", factory=lambda: fake)
    message = encoded_message()
    image = decoder.decode(message)
    assert (image.width, image.height, image.step, image.encoding) == (64, 48, 64, "nv12_full_range")
    assert len(image.data) == 48 * 64 * 3 // 2
    assert deserialize_message(serialize_message(image), Image) == image
    assert image.header == message.header
    decoder.close()
    decoder.close()
    assert calls == ["close"]


def test_auto_falls_back_once_when_acl_is_unavailable_and_cpu_does_not_load_acl():
    calls = []

    def unavailable():
        calls.append("load")
        raise ImportError("ACL unavailable")

    decoder = CompressedImageDecoder("auto", factory=unavailable)
    for _index in range(2):
        assert decoder.decode(encoded_message()).encoding == "bgr8"
    assert calls == ["load"]
    assert "ACL unavailable" in decoder.last_error
    assert CompressedImageDecoder("cpu", factory=unavailable).decode(encoded_message()).encoding == "bgr8"
    assert calls == ["load"]


def test_strict_dvpp_does_not_silently_fallback_on_decode_failure():
    def fail(_payload):
        raise RuntimeError("decode failed")

    decoder = CompressedImageDecoder(
        "dvpp", factory=lambda: SimpleNamespace(state=CodecLifecycleState.RUNNING, decode=fail)
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        decoder.decode(encoded_message())
    assert decoder.cpu_frames == 0


def test_auto_preserves_png_cpu_decode_and_refuses_depth():
    decoder = CompressedImageDecoder("auto", factory=lambda: pytest.fail("must not load DVPP for PNG"))
    message = encoded_message(".png")
    assert decoder.decode(message).encoding == "bgr8"
    message.format = "16UC1; png compressed"
    assert decoder.decode(message) is None


def test_callback_drops_before_decode_and_rejects_old_capture_timestamps(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("manipulation_execution.motion_imitation_stream_node.time.monotonic", lambda: clock[0])
    harness = SimpleNamespace(
        _source_available=lambda: True,
        _stats_lock=threading.RLock(),
        _source_generation=0,
        _source_healthy=True,
        _next_pear_at=0.0,
        _source_geometry=None,
        _yolox_detections=None,
        _yolox_result_ready=False,
        _pear_queue=queue.Queue(maxsize=1),
        _yolox_queue=queue.Queue(maxsize=1),
        _received=0,
        _window_received=0,
        _compressed_dropped=0,
        _out_of_order=0,
        _last_capture_stamp=None,
        _last_encoded_at=0,
        _rpc_timeout=2,
        _decode_queue=queue.Queue(maxsize=1),
    )
    harness._invalidate_source_locked = lambda: MotionImitationStreamNode._invalidate_source_locked(harness)
    frames = [encoded_message() for _index in range(3)]
    for message, seconds in zip(frames, [10, 12, 11], strict=True):
        message.header.stamp.sec = seconds
        MotionImitationStreamNode._on_compressed_frame(harness, message)
    assert harness._decode_queue.get_nowait().message is frames[1]
    assert harness._compressed_dropped == 1
    assert harness._out_of_order == 1
    clock[0] = 11
    frames[0].header.stamp.sec = 1
    MotionImitationStreamNode._on_compressed_frame(harness, frames[0])
    assert harness._decode_queue.empty()
    clock[0] = 12
    MotionImitationStreamNode._on_compressed_frame(harness, frames[0])
    assert harness._decode_queue.empty()
    clock[0] = 13
    frames[0].header.stamp.sec = 1
    MotionImitationStreamNode._on_compressed_frame(harness, frames[0])
    assert harness._decode_queue.get_nowait().message is frames[0]


def test_stream_rejects_conflicting_publishers_before_decoding():
    harness = SimpleNamespace(
        _source_available=lambda: False,
        _stats_lock=threading.Lock(),
        _source_rejected=0,
        _yolox_detections=object(),
        _yolox_result_ready=True,
        _decode_queue=queue.Queue(maxsize=1),
    )
    MotionImitationStreamNode._on_compressed_frame(harness, encoded_message())
    assert harness._source_rejected == 1
    assert harness._decode_queue.empty()
    assert harness._yolox_detections is None
    assert not harness._yolox_result_ready


def test_bridge_uses_the_shared_packet_timestamp_and_retries_a_failed_decoder():
    packets = []
    codec = SimpleNamespace(state=CodecLifecycleState.FAILED, close=lambda: None)

    def reset():
        codec.state = CodecLifecycleState.RUNNING

    def decode(packet):
        packets.append(packet)
        return [
            VideoFrame(
                np.zeros((72, 64), np.uint8),
                packet.capture_timestamp_ns,
                1,
                64,
                48,
                "nv12",
                color_space="bt601",
                color_range="full",
            )
        ]

    codec.reset = reset
    codec.decode = decode
    transport = CompressedImageDecoder("dvpp", factory=lambda: codec)
    message = encoded_message()
    message.header.stamp.nanosec = 123
    image = transport.decode(message)
    assert packets[0].capture_timestamp_ns == 42_000_000_123
    assert packets[0].payload == bytes(message.data)
    assert image.header == message.header
    assert transport.dvpp_frames == 1
    transport.close()
    with pytest.raises(RuntimeError, match="closed"):
        transport.decode(message)
