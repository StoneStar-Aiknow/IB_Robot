"""Error-path regressions for the DVPP frame decoders.

The review asked for two properties the one-shot ``_attempted`` flag broke:
a failed DVPP factory must not grow ``last_error`` frame over frame, and
creation must be retried with backoff instead of once per process. The
``auto`` variant must also say why it fell back to the CPU instead of going
quiet, and say when hardware decode comes back.
"""

import logging
from types import SimpleNamespace

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

from manipulation_execution.compressed_image_decode import CompressedImageDecoder, byte_array
from manipulation_execution.frame_decoder import AutoDecoder, DvppDecoder
from observation_transport.video_codec import CodecLifecycleState, VideoFrame

_DECODER_LOGGER = "manipulation_execution.compressed_image_decode"


def _encoded_message():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    success, data = cv2.imencode(".jpg", image)
    assert success
    message = CompressedImage(format="jpeg", data=byte_array(data))
    return message


def _healthy_dvpp_fake():
    return SimpleNamespace(
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
        close=lambda: None,
    )


def test_dvpp_create_failure_keeps_last_error_stable_across_frames():
    decoder = DvppDecoder({})
    attempts = []

    def unavailable():
        attempts.append(1)
        raise RuntimeError("acl device busy")

    decoder._transport = CompressedImageDecoder("dvpp", 0, factory=unavailable)
    message = _encoded_message()

    for _ in range(4):
        assert decoder.decode(message) is None

    assert attempts == [1]
    assert decoder.last_error == "RuntimeError: acl device busy"


def test_dvpp_create_retries_after_the_backoff_window():
    decoder = DvppDecoder({})
    attempts = []

    def unavailable():
        attempts.append(1)
        raise RuntimeError("device busy")

    decoder._transport = CompressedImageDecoder("dvpp", 0, factory=unavailable)
    message = _encoded_message()

    assert decoder.decode(message) is None
    assert decoder.decode(message) is None
    assert attempts == [1]
    decoder._transport._next_create_attempt_at = 0.0
    assert decoder.decode(message) is None
    assert attempts == [1, 1]


def test_auto_fallback_warns_once_and_recovers_with_info(caplog):
    decoder = AutoDecoder({})
    attempts = []
    state = {"fail": True}

    def factory():
        attempts.append(1)
        if state["fail"]:
            raise ImportError("ACL not loaded")
        return _healthy_dvpp_fake()

    decoder._transport = CompressedImageDecoder("auto", 0, factory=factory)
    message = _encoded_message()

    with caplog.at_level(logging.WARNING, logger=_DECODER_LOGGER):
        assert decoder.decode(message).encoding == "bgr8"
        assert decoder.decode(message).encoding == "bgr8"
    assert attempts == [1]
    assert caplog.text.count("ACL not loaded") == 1
    assert "device 0" in caplog.text

    decoder._transport._next_create_attempt_at = 0.0
    state["fail"] = False
    with caplog.at_level(logging.INFO, logger=_DECODER_LOGGER):
        image = decoder.decode(message)
    assert image.encoding == "nv12_full_range"
    assert "recovered" in caplog.text
    assert decoder.last_error == ""
