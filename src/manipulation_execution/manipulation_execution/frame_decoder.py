"""Replaceable transport decode for the motion-imitation frame stream.

The stream node subscribes an encoded camera stream and needs a raw ``bgr8``
``Image`` for the perception services. How that decode is done is expected to
change -- a faster CPU path, Ascend DVPP, a smaller stream from another compute
unit -- so the node does not call a decoder directly: it builds one by name
from configuration (``frame_source.decoder`` / ``decoder_params``) and only
ever talks to the ``FrameDecoder`` interface.

The contract every implementation keeps:

* ``decode`` returns ``None`` for a frame it cannot decode and never raises for
  a bad frame. The caller counts failures; a decoder does not log them.
* The returned ``Image`` carries the source header unchanged, so the stamp is
  the camera's observation time.
* Only the newest frame matters. A decoder is handed one frame at a time and
  keeps no queue; the node's depth-1 slot does the shedding.

An unknown name or a parameter a decoder does not accept fails when the node
starts, with the key named, rather than on the first frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from sensor_msgs.msg import CompressedImage, Image

from manipulation_execution.compressed_image_decode import decode_compressed_bgr8


class FrameDecoder(ABC):
    """Decode one ``CompressedImage`` into a ``bgr8`` ``Image``, or ``None``."""

    name = ""

    @abstractmethod
    def decode(self, message: CompressedImage) -> Image | None:
        raise NotImplementedError


class MockCpuDecoder(FrameDecoder):
    """Single-threaded OpenCV software decode on the CPU.

    On the 2688x1944 head stream it takes about 90 ms per frame on the X2
    compute unit, so it cannot keep up with 30 Hz. The node's status reports
    decode rate and time so that another registered decoder can be compared
    against it.
    """

    name = "mock_cpu"

    def __init__(self, params: Mapping) -> None:
        if params:
            raise ValueError(f"decoder_params: {self.name} takes no parameters, got {sorted(params)}")

    def decode(self, message: CompressedImage) -> Image | None:
        return decode_compressed_bgr8(message)


DECODERS: dict[str, type[FrameDecoder]] = {MockCpuDecoder.name: MockCpuDecoder}


def create_frame_decoder(name: object, params: object) -> FrameDecoder:
    """Build the decoder registered under ``name``, failing with the key named."""
    if not isinstance(name, str) or name not in DECODERS:
        raise ValueError(f"decoder: unknown decoder {name!r}; registered: {sorted(DECODERS)}")
    if not isinstance(params, Mapping):
        raise ValueError("decoder_params must be a mapping")
    return DECODERS[name](params)


__all__ = ["DECODERS", "FrameDecoder", "MockCpuDecoder", "create_frame_decoder"]
