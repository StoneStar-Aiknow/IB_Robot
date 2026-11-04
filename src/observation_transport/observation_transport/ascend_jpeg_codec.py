"""Optional JPEG decoder using the shared codec contract and an injected ACL lease."""

from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import replace

import numpy as np

from observation_transport.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    EncodedPacket,
    VideoCodecError,
    VideoDecoder,
    VideoFrame,
)

_BACKEND = "ascend_jpeg"


def _check(result, operation):
    if result not in (None, 0):
        raise VideoCodecError("runtime_error", f"{operation} returned {result}", backend=_BACKEND)


class AscendJpegDecoder(VideoDecoder):
    """Decode one JPEG per call; join the decode worker before closing its lease."""

    def __init__(self, lease, *, media=None, owns_lease=False):
        self._lease = lease
        self._owns_lease = owns_lease
        self._acl = lease.acl
        self._media = media
        self._channel = None
        self._channel_created = False
        self._stream = None
        self._buffers = []
        self._pictures = []
        self._pending = False
        self._state = CodecLifecycleState.CREATED
        self._metrics = CodecMetrics()

    @property
    def state(self):
        return self._state

    @property
    def metrics(self):
        return self._metrics

    def _open(self):
        self._lease.bind_current_thread()
        if self._media is None:
            self._media = importlib.import_module("acl.media")
        if self._channel is None:
            self._channel = self._media.dvpp_create_channel_desc()
            if not self._channel:
                raise VideoCodecError("allocation_failed", "missing JPEG channel descriptor", backend=_BACKEND)
        if not self._channel_created:
            _check(self._media.dvpp_create_channel(self._channel), "dvpp_create_channel")
            self._channel_created = True
        if self._stream is None:
            stream, result = self._acl.rt.create_stream()
            _check(result, "acl.rt.create_stream")
            self._stream = stream
        self._state = CodecLifecycleState.RUNNING

    def _allocate(self, size):
        if size <= 0:
            raise VideoCodecError("invalid_frame", "JPEG allocation size must be positive", backend=_BACKEND)
        pointer, result = self._media.dvpp_malloc(size)
        _check(result, "dvpp_malloc")
        if not pointer:
            raise VideoCodecError("allocation_failed", "missing JPEG buffer", backend=_BACKEND)
        self._buffers.append(pointer)
        return pointer

    def _free_frames(self):
        if self._pending:
            _check(self._acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream")
            self._pending = False
        if self._media is None:
            return
        errors = []
        for resources, operation in (
            (self._pictures, "dvpp_destroy_pic_desc"),
            (self._buffers, "dvpp_free"),
        ):
            for resource in list(resources):
                try:
                    _check(getattr(self._media, operation)(resource), operation)
                except Exception as exc:
                    errors.append(str(exc))
                else:
                    resources.remove(resource)
        if errors:
            raise VideoCodecError("cleanup_failed", "; ".join(errors), backend=_BACKEND)

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        if self._state in {CodecLifecycleState.CLOSED, CodecLifecycleState.FAILED}:
            raise VideoCodecError("invalid_state", f"decoder is {self._state.value}", backend=_BACKEND)
        self._metrics = replace(self._metrics, input_frames=self._metrics.input_frames + 1)
        try:
            if not packet.payload.startswith(b"\xff\xd8") or not packet.end_of_frame:
                raise VideoCodecError("invalid_frame", "expected one complete JPEG", backend=_BACKEND, recoverable=True)
            self._open()
            try:
                frame = self._decode_frame(packet)
            finally:
                self._free_frames()
        except VideoCodecError as exc:
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            if not exc.recoverable:
                # Only runtime failures poison the channel: a recoverable
                # input error leaves no resources behind (the finally above
                # already freed them), and rebuilding the DVPP channel per bad
                # frame costs far more than the decode itself.
                self._state = CodecLifecycleState.FAILED
            raise
        except Exception:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise
        self._metrics = replace(self._metrics, output_frames=self._metrics.output_frames + 1)
        return [frame]

    def _decode_frame(self, packet):
        media, acl = self._media, self._acl
        jpeg = packet.payload
        padded = jpeg + bytes(8)
        source = self._allocate(len(padded))
        _check(acl.rt.memcpy(source, len(padded), acl.util.bytes_to_ptr(padded), len(padded), 1), "memcpy JPEG")
        width, height, _components, result = media.dvpp_jpeg_get_image_info(source, len(jpeg))
        if result not in (None, 0):
            # A truncated or corrupt JPEG is an input problem: the buffers
            # allocated above are released by the decode finally, so the next
            # frame can reuse the channel instead of rebuilding it.
            raise VideoCodecError(
                "invalid_frame", "dvpp_jpeg_get_image_info rejected the JPEG", backend=_BACKEND, recoverable=True
            )
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise VideoCodecError("invalid_frame", "JPEG NV12 dimensions must be positive and even", backend=_BACKEND)
        size, result = media.dvpp_jpeg_predict_dec_size(source, len(jpeg), 1)
        _check(result, "dvpp_jpeg_predict_dec_size")
        output = self._allocate(size)
        picture = media.dvpp_create_pic_desc()
        if not picture:
            raise VideoCodecError("allocation_failed", "missing JPEG picture descriptor", backend=_BACKEND)
        self._pictures.append(picture)
        for field, value in (
            ("data", output),
            ("size", size),
            ("format", 1),
            ("width", width),
            ("height", height),
            ("width_stride", (width + 127) // 128 * 128),
            ("height_stride", (height + 15) // 16 * 16),
        ):
            _check(getattr(media, "dvpp_set_pic_desc_" + field)(picture, value), "dvpp_set_pic_desc_" + field)
        self._pending = True
        _check(media.dvpp_jpeg_decode_async(self._channel, source, len(jpeg), picture, self._stream), "JPEG decode")
        _check(acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream")
        self._pending = False
        get_result = getattr(media, "dvpp_get_pic_desc_ret_code", None)
        if get_result is not None:
            _check(get_result(picture), "JPEG picture result")
        stride = int(media.dvpp_get_pic_desc_width_stride(picture))
        rows = int(media.dvpp_get_pic_desc_height_stride(picture))
        if stride < width or rows < height or stride % 2 or rows % 2 or stride * rows * 3 // 2 > size:
            raise VideoCodecError("invalid_frame", "JPEG decoder returned invalid strides", backend=_BACKEND)
        host = np.empty(size, dtype=np.uint8)
        _check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), size, output, size, 2), "memcpy NV12")
        luma = host[: stride * rows].reshape(rows, stride)[:height, :width]
        chroma = host[stride * rows : stride * rows * 3 // 2].reshape(rows // 2, stride)[: height // 2, :width]
        return VideoFrame(
            data=np.concatenate((luma, chroma)),
            capture_timestamp_ns=packet.capture_timestamp_ns,
            receive_timestamp_ns=time.time_ns(),
            width=width,
            height=height,
            pixel_format="nv12",
            color_space="bt601",
            color_range="full",
            keyframe=True,
        )

    def _dispose(self):
        self._lease.bind_current_thread()
        self._free_frames()
        errors = []
        if self._stream is not None:
            try:
                _check(self._acl.rt.destroy_stream(self._stream), "acl.rt.destroy_stream")
            except Exception as exc:
                errors.append(str(exc))
            else:
                self._stream = None
        if self._channel_created:
            try:
                _check(self._media.dvpp_destroy_channel(self._channel), "dvpp_destroy_channel")
            except Exception as exc:
                errors.append(str(exc))
            else:
                self._channel_created = False
        if self._channel is not None and not self._channel_created:
            try:
                _check(self._media.dvpp_destroy_channel_desc(self._channel), "dvpp_destroy_channel_desc")
            except Exception as exc:
                errors.append(str(exc))
            else:
                self._channel = None
        if errors:
            raise VideoCodecError("cleanup_failed", "; ".join(errors), backend=_BACKEND)

    def reset(self):
        if self._state == CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "cannot reset a closed JPEG decoder", backend=_BACKEND)
        try:
            self._dispose()
        except Exception:
            self._state = CodecLifecycleState.FAILED
            raise
        self._state = CodecLifecycleState.CREATED

    def close(self, timeout_s=1.0):
        if self._state == CodecLifecycleState.CLOSED:
            return
        try:
            self._dispose()
            if self._owns_lease:
                self._lease.close()
        except Exception:
            self._state = CodecLifecycleState.FAILED
            raise
        self._state = CodecLifecycleState.CLOSED


def register_ascend_jpeg_backend(registry, *, lease_factory):
    """Register JPEG explicitly without changing the default H.264 auto policy."""

    def probe(kind):
        if kind != "decoder" or importlib.util.find_spec("acl") is None:
            return None
        return CodecCapabilities(codecs=("jpeg",), pixel_formats=("nv12",), hardware_accelerated=True)

    def create(**options):
        lease = lease_factory()
        try:
            return AscendJpegDecoder(lease, owns_lease=True, **options)
        except Exception:
            lease.close()
            raise

    registry.register(_BACKEND, priority=100, probe=probe, decoder_factory=create)
