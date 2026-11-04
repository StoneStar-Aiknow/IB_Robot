"""JPEG/PNG transport decode; model-semantic preprocessing stays in adapters.

The legacy pure function emits BGR. The optional stateful decoder emits
explicit full-range NV12 for updated HRI services, retaining source geometry.
"""

from __future__ import annotations

import array
import logging
import time

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

from observation_transport.video_codec import CodecLifecycleState, EncodedPacket

_LOGGER = logging.getLogger(__name__)

# Pixel encodings that are not 8-bit colour. These matter because
# ``cv2.imdecode(..., IMREAD_COLOR)`` on a 16-bit single-channel PNG -- how
# depth frames are usually carried -- succeeds and returns three channels of
# garbage rather than failing. Screening on the codec name cannot catch that:
# such a frame's ``format`` is e.g. "16UC1; png compressed", which names png.
# So the discriminator is the pixel encoding, not the compression.
_NON_COLOR_ENCODINGS = (
    "16uc1",
    "16sc1",
    "32fc1",
    "64fc1",
    "mono16",
    "16bit",
)

_BGR8_CHANNELS = 3

# Retry cadence for a failed DVPP decoder factory: 1 s doubling up to 30 s, so
# a device that recovers is picked back up without per-frame create attempts.
_CREATE_RETRY_MIN_SEC = 1.0
_CREATE_RETRY_MAX_SEC = 30.0


def is_decodable_format(format_text: str) -> bool:
    """Whether ``CompressedImage.format`` describes something safe to decode as colour.

    Permissive on purpose. An unexpected-but-benign format string must not
    silently drop every frame of a stream, and ``cv2.imdecode`` already fails
    closed on bytes that are not an image at all. Only encodings known to
    decode into plausible-looking nonsense are refused.
    """
    return not any(marker in str(format_text).lower() for marker in _NON_COLOR_ENCODINGS)


def decode_compressed_bgr8(message: CompressedImage) -> Image | None:
    """Decode ``message`` to a ``bgr8`` ``Image``, or ``None`` if it cannot be decoded.

    The returned message carries the source header unchanged: the stamp is the
    camera's observation time, never the time this decode ran.
    """
    if not is_decodable_format(message.format):
        return None
    if not message.data:
        return None
    try:
        frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if frame is None:
        return None
    if frame.ndim != 3 or frame.shape[2] != _BGR8_CHANNELS:
        return None
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        return None
    image = Image()
    image.header = message.header
    image.height = height
    image.width = width
    image.encoding = "bgr8"
    image.is_bigendian = 0
    # Set explicitly rather than left to a conversion helper: the service
    # rejects a message whose step is smaller than width * 3.
    image.step = width * _BGR8_CHANNELS
    # MUST be array.array('B'). The generated Image.data setter early-returns
    # for that type and validates nothing; every other accepted type falls
    # through to a per-element `isinstance(v, int) and 0 <= v < 256` check over
    # the whole buffer. On a 2688x1944 frame that is 15.7M elements and was
    # measured at 7.7 s per frame against 16 ms here -- a 474x difference that
    # looks exactly like a slow decoder. The numpy fast path some ROS builds
    # have does not exist in this one.
    image.data = array.array("B", np.ascontiguousarray(frame).tobytes())
    return image


def byte_array(frame):
    payload = array.array("B")
    payload.frombytes(np.ascontiguousarray(frame).tobytes())
    return payload


class CompressedImageDecoder:
    """Optional DVPP transport with an unchanged CPU path for legacy consumers."""

    def __init__(self, backend="cpu", device_id=0, *, factory=None):
        if backend not in {"cpu", "auto", "dvpp"}:
            raise ValueError("image_decoder must be cpu, auto or dvpp")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("dvpp_device_id must be a non-negative integer")
        self.backend = backend
        self.device_id = device_id
        self._factory = factory or self._create_dvpp
        self._decoder = None
        self._next_create_attempt_at = 0.0
        self._create_retry_backoff_sec = _CREATE_RETRY_MIN_SEC
        self._closed = False
        self.last_error = ""
        self.cpu_frames = 0
        self.dvpp_frames = 0

    def _create_dvpp(self):
        from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
        from observation_transport.ascend_jpeg_codec import register_ascend_jpeg_backend
        from observation_transport.video_codec import create_default_video_codec_registry

        manager = AclRuntimeManager()
        registry = create_default_video_codec_registry()
        register_ascend_jpeg_backend(registry, lease_factory=lambda: manager.acquire(self.device_id))
        return registry.resolve("ascend_jpeg", "decoder", codec="jpeg").create()

    def _ensure_decoder(self) -> None:
        """Create the DVPP decoder, retrying a failed factory with capped backoff."""
        now = time.monotonic()
        if now < self._next_create_attempt_at:
            if self.backend == "dvpp":
                raise RuntimeError(self.last_error)
            return
        self._next_create_attempt_at = now + self._create_retry_backoff_sec
        self._create_retry_backoff_sec = min(_CREATE_RETRY_MAX_SEC, self._create_retry_backoff_sec * 2.0)
        try:
            self._decoder = self._factory()
        except Exception as exc:
            first_failure = not self.last_error
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.backend == "auto" and first_failure:
                _LOGGER.warning(
                    "DVPP JPEG decode unavailable on device %d, falling back to CPU; next retry in %.0f s: %s",
                    self.device_id,
                    self._create_retry_backoff_sec,
                    self.last_error,
                )
            if self.backend == "dvpp":
                raise
        else:
            self._create_retry_backoff_sec = _CREATE_RETRY_MIN_SEC
            if self.last_error:
                _LOGGER.info("DVPP JPEG decode recovered on device %d", self.device_id)
            self.last_error = ""

    def decode(self, message):
        if self._closed:
            raise RuntimeError("compressed image decoder is closed")
        if not is_decodable_format(message.format) or not message.data:
            return None
        is_jpeg = bytes(message.data[:2]) == b"\xff\xd8"
        if self.backend != "cpu" and is_jpeg:
            if self._decoder is None:
                self._ensure_decoder()
            if self._decoder is not None:
                try:
                    if self._decoder.state == CodecLifecycleState.FAILED:
                        self._decoder.reset()
                    stamp = message.header.stamp
                    packet = EncodedPacket(
                        payload=bytes(message.data),
                        rtp_timestamp=0,
                        capture_timestamp_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                        keyframe=True,
                    )
                    frames = self._decoder.decode(packet)
                    if len(frames) != 1:
                        raise ValueError("JPEG decoder must return exactly one frame")
                    decoded = frames[0]
                    if (
                        decoded.pixel_format != "nv12"
                        or decoded.color_range != "full"
                        or decoded.color_space != "bt601"
                    ):
                        raise ValueError("JPEG decoder must return full-range BT.601 NV12")
                    image = Image()
                    image.header = message.header
                    image.width, image.height = int(decoded.width), int(decoded.height)
                    image.encoding = "nv12_full_range"
                    image.step = int(decoded.width)
                    image.data = byte_array(decoded.data)
                    self.dvpp_frames += 1
                    self.last_error = ""
                    return image
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    if self.backend == "dvpp":
                        raise
            elif self.backend == "dvpp":
                raise RuntimeError(self.last_error)
        image = decode_compressed_bgr8(message)
        if image is not None:
            self.cpu_frames += 1
        return image

    def close(self):
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None
        self._closed = True


__all__ = ["CompressedImageDecoder", "byte_array", "decode_compressed_bgr8", "is_decodable_format"]
