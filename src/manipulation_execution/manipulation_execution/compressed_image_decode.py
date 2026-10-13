"""Software decode of a JPEG/PNG ``CompressedImage`` into a raw ``bgr8`` ``Image``.

The perception model services take a raw ``sensor_msgs/Image`` only
(``perception_service/model_contracts.py`` accepts ``bgr8``/``rgb8``), so a
caller fed an encoded camera stream has to decode before it can ask for a
detection. This module is the only place in ``manipulation_execution`` that
touches pixels, and it is deliberately the only one that imports ``cv2``: model
*semantic* preprocessing -- letterboxing, channel order, cropping, resizing,
normalisation -- stays inside the perception adapters, which is what the rule
in ``perception_service/README.md`` is about. Transport decode is not that.

Pure functions: no node, no logging, no counters. A frame that cannot be
decoded returns ``None`` and the caller decides how to count and report it.
"""

from __future__ import annotations

import array

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

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
    frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
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
    # Must be array.array('B'). The generated Image.data setter accepts that
    # type without validation; every other accepted type is checked element
    # by element (`isinstance(v, int) and 0 <= v < 256`). On a 2688x1944 frame
    # (15.7M elements) that check takes seconds per frame, against about
    # 16 ms for the whole conversion here. The targeted ROS distribution has
    # no numpy fast path for this setter.
    image.data = array.array("B", np.ascontiguousarray(frame).tobytes())
    return image


__all__ = ["decode_compressed_bgr8", "is_decodable_format"]
