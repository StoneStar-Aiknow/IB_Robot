from types import SimpleNamespace

import numpy as np
import pytest

from observation_transport.ascend_jpeg_codec import AscendJpegDecoder, register_ascend_jpeg_backend
from observation_transport.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    EncodedPacket,
    VideoCodecError,
    VideoDecoder,
    create_default_video_codec_registry,
)


class FakeDvpp:
    def __init__(self, fail=None):
        self.fail = fail
        self.events = []
        self.next_pointer = 100
        self.acl = SimpleNamespace(
            rt=SimpleNamespace(
                create_stream=lambda: (20, self.result("create_stream")),
                destroy_stream=lambda _handle: self.result("destroy_stream"),
                synchronize_stream=lambda _handle: self.result("synchronize"),
                memcpy=self.memcpy,
            ),
            util=SimpleNamespace(bytes_to_ptr=lambda payload: payload, numpy_to_ptr=lambda host: host),
        )
        self.lease = SimpleNamespace(
            acl=self.acl, bind_current_thread=lambda: None, close=lambda: self.events.append("lease_close")
        )

    def result(self, name):
        self.events.append(name)
        return 1 if name == self.fail else 0

    def memcpy(self, target, _capacity, _source, _length, direction):
        if direction == 2:
            target.fill(128)
        return self.result("copy")

    def dvpp_malloc(self, _size):
        self.next_pointer += 1
        return self.next_pointer, self.result("malloc")

    def dvpp_create_channel_desc(self):
        return 1

    def dvpp_create_channel(self, _handle):
        return self.result("create_channel")

    def dvpp_destroy_channel(self, _handle):
        return self.result("destroy_channel")

    def dvpp_destroy_channel_desc(self, _handle):
        return self.result("destroy_channel_desc")

    def dvpp_jpeg_get_image_info(self, _pointer, _length):
        return 1344, 972, 3, self.result("info")

    def dvpp_jpeg_predict_dec_size(self, *_args):
        return 1408 * 976 * 3 // 2, self.result("size")

    def dvpp_create_pic_desc(self):
        return 2

    def dvpp_destroy_pic_desc(self, _handle):
        return self.result("destroy_picture")

    def dvpp_free(self, _handle):
        return self.result("free")

    def dvpp_jpeg_decode_async(self, *_args):
        return self.result("decode")

    def dvpp_get_pic_desc_ret_code(self, _handle):
        return self.result("picture_result")

    def dvpp_get_pic_desc_width_stride(self, _handle):
        return 1344

    def dvpp_get_pic_desc_height_stride(self, _handle):
        return 976

    def __getattr__(self, name):
        if name.startswith("dvpp_set_pic_desc_"):
            return lambda *_args: self.result("set_picture")
        raise AttributeError(name)


def test_reads_actual_dvpp_stride_and_removes_height_padding():
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=fake, owns_lease=True)
    frame = decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    assert (frame.width, frame.height, frame.data.shape[1]) == (1344, 972, 1344)
    assert frame.data.shape == (1458, 1344)
    assert np.all(frame.data == 128)
    assert fake.events.count("free") == 2
    decoder.close()
    decoder.close()
    assert fake.events.count("lease_close") == 1


@pytest.mark.parametrize(
    "failure", ["create_channel", "create_stream", "copy", "info", "size", "set_picture", "decode", "picture_result"]
)
def test_decode_failures_release_resources_and_allow_close(failure):
    fake = FakeDvpp(failure)
    decoder = AscendJpegDecoder(fake.lease, media=fake, owns_lease=True)
    with pytest.raises(RuntimeError):
        decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    fake.fail = None
    decoder.close()
    assert not decoder._buffers and not decoder._pictures
    assert fake.events[-1] == "lease_close"


def test_failed_synchronize_keeps_buffers_until_a_successful_retry():
    fake = FakeDvpp("synchronize")
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    with pytest.raises(RuntimeError):
        decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    assert len(decoder._buffers) == 2
    assert "free" not in fake.events
    fake.fail = None
    decoder.close()
    assert not decoder._buffers


def test_horizontal_padding_is_not_transmitted():
    fake = FakeDvpp()
    fake.dvpp_get_pic_desc_width_stride = lambda _handle: 1408
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    frame = decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    assert frame.data.shape[1] == frame.width == 1344
    assert frame.data.nbytes == 1344 * 972 * 3 // 2
    decoder.close()


@pytest.mark.parametrize("failure", ["destroy_picture", "free"])
def test_frame_cleanup_failures_are_retryable(failure):
    fake = FakeDvpp(failure)
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    with pytest.raises(RuntimeError):
        decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    assert decoder._buffers or decoder._pictures
    fake.fail = None
    decoder.close()
    assert not decoder._buffers and not decoder._pictures


def test_cleanup_failures_preserve_handles_for_retry_and_release_independently():
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=fake, owns_lease=True)
    decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 12345, keyframe=True))[0]
    fake.fail = "destroy_stream"
    with pytest.raises(RuntimeError):
        decoder.close()
    assert decoder._stream == 20
    assert "destroy_channel_desc" in fake.events
    assert "lease_close" not in fake.events
    fake.fail = None
    decoder.close()
    assert decoder._stream is None
    assert fake.events.count("lease_close") == 1


def test_shared_decoder_contract_preserves_timestamps_color_and_metrics():
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    assert isinstance(decoder, VideoDecoder)
    assert decoder.state == CodecLifecycleState.CREATED
    packet = EncodedPacket(b"\xff\xd8jpeg", 0, 987654321, keyframe=True)
    frame = decoder.decode(packet)[0]
    assert frame.capture_timestamp_ns == packet.capture_timestamp_ns
    assert frame.receive_timestamp_ns > 0
    assert (frame.pixel_format, frame.color_space, frame.color_range) == ("nv12", "bt601", "full")
    assert decoder.state == CodecLifecycleState.RUNNING
    assert decoder.metrics.input_frames == decoder.metrics.output_frames == 1
    assert decoder.metrics.errors == 0
    decoder.reset()
    assert decoder.state == CodecLifecycleState.CREATED
    assert "lease_close" not in fake.events
    assert decoder.decode(packet)[0].width == 1344
    decoder.close()
    assert decoder.state == CodecLifecycleState.CLOSED
    with pytest.raises(VideoCodecError, match="closed"):
        decoder.decode(packet)


def test_failed_frame_can_reset_without_releasing_the_injected_lease():
    fake = FakeDvpp("decode")
    decoder = AscendJpegDecoder(fake.lease, media=fake, owns_lease=True)
    packet = EncodedPacket(b"\xff\xd8jpeg", 0, 1)
    with pytest.raises(VideoCodecError):
        decoder.decode(packet)
    assert decoder.state == CodecLifecycleState.FAILED
    fake.fail = None
    decoder.reset()
    assert "lease_close" not in fake.events
    assert len(decoder.decode(packet)) == 1
    assert decoder.metrics.input_frames == 2
    assert decoder.metrics.errors == 1
    decoder.close()
    assert fake.events.count("lease_close") == 1


@pytest.mark.parametrize("payload, complete", [(b"not jpeg", True), (b"\xff\xd8jpeg", False)])
def test_invalid_packet_fails_before_allocating(payload, complete):
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    with pytest.raises(VideoCodecError, match="complete JPEG"):
        decoder.decode(EncodedPacket(payload, 0, 1, end_of_frame=complete))
    assert not fake.events
    decoder.close()


def test_explicit_jpeg_registration_preserves_default_video_registry(monkeypatch):
    monkeypatch.setattr("observation_transport.ascend_jpeg_codec.importlib.util.find_spec", lambda _name: object())
    fake = FakeDvpp()
    acquisitions = []

    def lease_factory():
        acquisitions.append(True)
        return fake.lease

    registry = create_default_video_codec_registry()
    register_ascend_jpeg_backend(registry, lease_factory=lease_factory)
    backend = registry.resolve("ascend_jpeg", "decoder", codec="jpeg")
    assert backend.capabilities.codecs == ("jpeg",)
    assert backend.capabilities.hardware_accelerated
    assert acquisitions == []
    decoder = backend.create(media=fake)
    assert acquisitions == [True]
    assert decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 42))[0].capture_timestamp_ns == 42
    decoder.close()
    assert fake.events.count("lease_close") == 1
    with pytest.raises(VideoCodecError, match="not registered"):
        create_default_video_codec_registry().resolve("ascend_jpeg", "decoder", codec="jpeg")
    with pytest.raises(VideoCodecError, match="unavailable"):
        registry.resolve("ascend_jpeg", "encoder", codec="jpeg")


def test_auto_h264_does_not_select_a_higher_priority_jpeg_backend(monkeypatch):
    from observation_transport.video_codec import VideoCodecRegistry

    monkeypatch.setattr("observation_transport.ascend_jpeg_codec.importlib.util.find_spec", lambda _name: object())
    registry = VideoCodecRegistry()
    register_ascend_jpeg_backend(registry, lease_factory=lambda: pytest.fail("resolution must not acquire hardware"))
    registry.register("h264_test", priority=0, probe=lambda _kind: CodecCapabilities(), decoder_factory=lambda: None)
    assert registry.resolve("auto", "decoder").name == "h264_test"
    assert registry.resolve("auto", "decoder", codec="jpeg").name == "ascend_jpeg"
    with pytest.raises(VideoCodecError, match="not supported"):
        registry.resolve("ascend_jpeg", "decoder")


@pytest.mark.parametrize("codec", ["", " ", None, 123])
def test_registry_rejects_invalid_codec_names(codec):
    with pytest.raises(ValueError, match="codec"):
        create_default_video_codec_registry().resolve("auto", "decoder", codec=codec)


def test_missing_media_api_does_not_prevent_releasing_an_unused_lease():
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=SimpleNamespace(), owns_lease=True)
    with pytest.raises(AttributeError):
        decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 1))
    decoder.close()
    assert decoder.state == CodecLifecycleState.CLOSED
    assert fake.events == ["lease_close"]


@pytest.mark.parametrize("payload, complete", [(b"not jpeg", True), (b"\xff\xd8jpeg", False)])
def test_recoverable_bad_frame_keeps_the_channel_for_the_next_good_frame(payload, complete):
    fake = FakeDvpp()
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    good = EncodedPacket(b"\xff\xd8jpeg", 0, 1, keyframe=True)
    assert decoder.decode(good)
    with pytest.raises(VideoCodecError, match="complete JPEG"):
        decoder.decode(EncodedPacket(payload, 0, 2, keyframe=True, end_of_frame=complete))
    assert decoder.state == CodecLifecycleState.RUNNING
    assert decoder.decode(good)
    assert "destroy_channel" not in fake.events
    assert "destroy_stream" not in fake.events
    decoder.close()


def test_truncated_jpeg_rejection_reuses_the_channel_without_reset():
    fake = FakeDvpp("info")
    decoder = AscendJpegDecoder(fake.lease, media=fake)
    with pytest.raises(VideoCodecError, match="get_image_info"):
        decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 1, keyframe=True))
    assert decoder.state == CodecLifecycleState.RUNNING
    fake.fail = None
    assert decoder.decode(EncodedPacket(b"\xff\xd8jpeg", 0, 2, keyframe=True))[0].width == 1344
    assert "destroy_channel" not in fake.events
    assert fake.events.count("create_channel") == 1
    decoder.close()
