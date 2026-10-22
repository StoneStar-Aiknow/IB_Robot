"""Transport decode of the platform's encoded camera stream.

The decisive assertion here is that a decoded frame passes
``perception_service.model_contracts.validate_image_message`` -- the exact
check the YOLOX and PEAR services run on every request. A frame that decodes
but carries a wrong ``step`` is rejected service-side with a message that says
nothing about the decoder, so this is the cheapest place to catch it.
"""

import array

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage

from manipulation_execution.compressed_image_decode import decode_compressed_bgr8, is_decodable_format
from perception_service.model_contracts import validate_image_message


def _jpeg_message(width: int = 64, height: int = 48, *, image_format: str = "jpeg") -> CompressedImage:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = (10, 120, 230)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    message = CompressedImage()
    message.format = image_format
    message.data = encoded.tobytes()
    message.header.frame_id = "head_front_center"
    message.header.stamp.sec = 7
    message.header.stamp.nanosec = 123456789
    return message


def test_decoded_frame_satisfies_the_service_image_contract():
    image = decode_compressed_bgr8(_jpeg_message())

    assert image is not None
    validate_image_message(image)


def test_decode_sets_bgr8_geometry_and_keeps_the_source_header():
    image = decode_compressed_bgr8(_jpeg_message(width=64, height=48))

    assert image is not None
    assert (image.width, image.height) == (64, 48)
    assert image.encoding == "bgr8"
    assert image.step == 64 * 3
    assert len(image.data) == image.step * image.height
    # The stamp is the camera's observation time, never the decode time.
    assert (image.header.stamp.sec, image.header.stamp.nanosec) == (7, 123456789)
    assert image.header.frame_id == "head_front_center"


@pytest.mark.parametrize(
    "image_format",
    ["16UC1; png compressed", "32FC1; png compressed", "mono16; png compressed"],
)
def test_depth_frames_are_refused_rather_than_decoded_as_colour(image_format):
    # These decode to three channels of plausible-looking garbage instead of
    # failing, so the codec name cannot be the discriminator: the format above
    # names png.
    assert not is_decodable_format(image_format)
    assert decode_compressed_bgr8(_jpeg_message(image_format=image_format)) is None


@pytest.mark.parametrize("image_format", ["jpeg", "jpg", "png", "rgb8; jpeg compressed bgr8", ""])
def test_unsurprising_formats_are_accepted(image_format):
    # The platform's exact format string is not documented, so an unexpected
    # but benign value must not drop every frame of the stream.
    assert is_decodable_format(image_format)


def test_empty_payload_is_refused():
    message = _jpeg_message()
    message.data = b""

    assert decode_compressed_bgr8(message) is None


def test_truncated_payload_is_refused():
    message = _jpeg_message()
    message.data = message.data[: len(message.data) // 4]

    assert decode_compressed_bgr8(message) is None


def test_non_image_payload_is_refused():
    message = _jpeg_message()
    message.data = b"this is not an image"

    assert decode_compressed_bgr8(message) is None


def test_payload_uses_the_setter_fast_path():
    """Pins the one detail that decides whether this is 16 ms or 7.7 s a frame.

    The generated ``Image.data`` setter early-returns for ``array.array('B')``
    and validates nothing. Hand it anything else accepted -- bytes, a numpy
    array on a build without the numpy branch -- and it validates every element
    of the buffer, which on a 5 MP frame was measured at 7.7 s. A regression
    here presents as a mysteriously slow camera, not as a failure, so assert
    the type rather than a duration.
    """
    image = decode_compressed_bgr8(_jpeg_message())

    assert image is not None
    assert isinstance(image.data, array.array)
    assert image.data.typecode == "B"
