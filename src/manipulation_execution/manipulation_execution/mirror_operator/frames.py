"""Camera messages -> ``GrayFrame``, and the frame rate limit. No ROS.

``camera_kind`` ``compressed`` (``sensor_msgs/CompressedImage``) is decoded
straight to gray at 1/8 size (``IMREAD_REDUCED_GRAYSCALE_8``; only JPEG is
reduced while decoding, other formats decode at full size). ``image``
(``sensor_msgs/Image``, ``bgr8``/``rgb8``/``mono8``) is turned gray, then
reduced to 1/8. ``stamp_s`` is the camera time from the message header.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy

from manipulation_execution.mirror_operator.judge import GrayFrame

REDUCTION = 8
_CHANNELS = {"bgr8": 3, "rgb8": 3, "mono8": 1}
_TO_GRAY = {"bgr8": cv2.COLOR_BGR2GRAY, "rgb8": cv2.COLOR_RGB2GRAY}


def stamp_s(message: Any) -> float:
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def _bytes(data: Any) -> numpy.ndarray:
    try:
        return numpy.frombuffer(data, dtype=numpy.uint8)
    except TypeError:  # a list of ints
        return numpy.asarray(data, dtype=numpy.uint8)


def gray_from_compressed(message: Any) -> GrayFrame:
    """``ValueError`` when the data does not decode."""
    data = _bytes(message.data)
    image = cv2.imdecode(data, cv2.IMREAD_REDUCED_GRAYSCALE_8) if data.size else None
    if image is None:
        raise ValueError(f"compressed frame ({message.format or 'no format'}, {data.size} bytes) does not decode")
    return GrayFrame.of(image, stamp_s(message))


def gray_from_image(message: Any) -> GrayFrame:
    """``ValueError`` for an encoding other than bgr8/rgb8/mono8 or a short buffer."""
    channels = _CHANNELS.get(message.encoding)
    if channels is None:
        raise ValueError(f"image encoding {message.encoding!r} is not one of {sorted(_CHANNELS)}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    data = _bytes(message.data)
    if height <= 0 or width <= 0 or step < width * channels or data.size < height * step:
        raise ValueError(f"image {width}x{height} step {step} does not match {data.size} bytes")
    rows = data[: height * step].reshape(height, step)[:, : width * channels]
    image = rows.reshape(height, width, channels)
    gray = image[:, :, 0] if channels == 1 else cv2.cvtColor(image, _TO_GRAY[message.encoding])
    size = (math.ceil(width / REDUCTION), math.ceil(height / REDUCTION))
    return GrayFrame.of(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), stamp_s(message))


class FrameThrottle:
    """Passes at most ``max_hz`` frames a second, before decoding.

    A frame up to 10 % early still passes, so a camera running at exactly
    ``max_hz`` is not halved by arrival jitter.
    """

    def __init__(self, max_hz: float):
        self._period = 1.0 / max_hz
        self._last: float | None = None

    def admit(self, now: float) -> bool:
        if self._last is not None and now - self._last < 0.9 * self._period and now >= self._last:
            return False
        self._last = now
        return True
