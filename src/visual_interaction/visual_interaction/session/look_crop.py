"""The upper-body crop look_say shows the model, as one pure function.

The vision OM sees a single 448² image. Handing it the whole half-resolution
frame (1344×972 on the X2 head camera) spends most of those pixels on the hall;
the person's hands, which is what the questions are about, end up a few dozen
pixels wide. So the frame is cut down to the actor's upper body first:

1. the top 65 % of the actor's box (head to about the hips);
2. widened by 15 % of the box width on each side, so arms held out stay in,
   and raised by 5 % of the box height, so a hand above the head stays in;
3. padded to a square around that region's centre, with anything outside the
   frame filled black rather than clamped, so the person is never stretched
   and the crop never slides away from them at an edge;
4. resized to 448² with PIL bicubic, the resize the board demo's own
   preprocessing uses, so the model sees what it was validated on.

A box that is missing, not finite, inverted, outside the frame or smaller than
:data:`MIN_BOX_PX` on either side once clipped (or whose upper-body region
keeps less than that of the frame's height) gives the whole frame instead,
resized the same way the demo resizes it (stretched, not padded), and
``fallback`` says so.

Coordinates: ``ActorState.bbox`` is ``[x_min, y_min, x_max, y_max]`` in the
pixels of the image the motion imitation stream decoded. YOLOX letterboxes to
640 internally and divides its boxes back by that ratio, then clips them to
the source size, so the box is in source pixels. In the deployed configs that
stream and the visual interaction node read the same topic
(``rgb_head_front_center/rgb_image_half/compressed``), so the box already
matches the frame. ``aimdk_x2_voice_mirror.yaml`` is the exception: its stream
reads full resolution while the node keeps the half stream. ``source_size``
covers that case: given the (width, height) the box was measured in, the box is
scaled to the frame.

No ROS, no NPU: numpy and PIL only, so it is tested on a workstation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SIDE = 448
MIN_BOX_PX = 32.0
UPPER_FRACTION = 0.65
SIDE_MARGIN = 0.15
TOP_MARGIN = 0.05


@dataclass(frozen=True)
class Crop:
    """``image`` is SIDE×SIDE×3 uint8 in the frame's channel order.

    ``square`` is ``(x0, y0, side)`` of the square taken from the frame, in
    frame pixels, and may reach outside it; None when the whole frame was used.
    """

    image: np.ndarray
    fallback: bool
    square: tuple[int, int, int] | None


def upper_body_square(
    bbox: Sequence[float] | None,
    frame_size: tuple[int, int],
    *,
    source_size: tuple[int, int] | None = None,
) -> tuple[int, int, int] | None:
    """The square ``(x0, y0, side)`` to cut, or None when the box is unusable.

    ``frame_size`` and ``source_size`` are (width, height).
    """
    width, height = frame_size
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if source_size is not None:
        source_width, source_height = source_size
        if source_width <= 0 or source_height <= 0:
            return None
        sx, sy = width / float(source_width), height / float(source_height)
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    # Judge size on the part of the box that is in the frame: a box mostly
    # outside it has nothing to show.
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(width)), min(y2, float(height))
    if cx2 - cx1 < MIN_BOX_PX or cy2 - cy1 < MIN_BOX_PX:
        return None

    box_width, box_height = x2 - x1, y2 - y1
    left = x1 - SIDE_MARGIN * box_width
    right = x2 + SIDE_MARGIN * box_width
    top = y1 - TOP_MARGIN * box_height
    bottom = y1 + UPPER_FRACTION * box_height
    # A box that is mostly below the frame's top edge can still leave its
    # upper body outside the frame; a black square would show nothing.
    if min(bottom, float(height)) - max(top, 0.0) < MIN_BOX_PX:
        return None
    side = max(right - left, bottom - top)
    centre_x, centre_y = (left + right) / 2.0, (top + bottom) / 2.0
    side_px = int(math.ceil(side))
    x0 = int(math.floor(centre_x - side_px / 2.0))
    y0 = int(math.floor(centre_y - side_px / 2.0))
    return x0, y0, side_px


def _check_frame(frame: object) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be an H×W×3 uint8 array")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame is empty")


def _resize(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    if image.shape[0] == SIDE and image.shape[1] == SIDE:
        return np.ascontiguousarray(image)
    # Channel order does not matter to a per-channel resize, so the BGR array
    # goes through PIL as is and comes back BGR.
    resized = Image.fromarray(np.ascontiguousarray(image)).resize((SIDE, SIDE), Image.BICUBIC)
    return np.asarray(resized)


def crop_upper_body(
    frame: np.ndarray,
    bbox: Sequence[float] | None,
    *,
    source_size: tuple[int, int] | None = None,
) -> Crop:
    """Cut the actor's upper body out of ``frame`` (H×W×3 uint8) as a 448² square."""
    _check_frame(frame)
    height, width = frame.shape[:2]
    square = upper_body_square(bbox, (width, height), source_size=source_size)
    if square is None:
        return Crop(image=_resize(frame), fallback=True, square=None)

    x0, y0, side = square
    canvas = np.zeros((side, side, 3), np.uint8)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + side, width), min(y0 + side, height)
    canvas[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return Crop(image=_resize(canvas), fallback=False, square=square)


def whole_frame(frame: np.ndarray) -> np.ndarray:
    """The whole frame at 448², stretched the way the demo stretches it."""
    _check_frame(frame)
    return _resize(frame)


__all__ = [
    "MIN_BOX_PX",
    "SIDE",
    "SIDE_MARGIN",
    "TOP_MARGIN",
    "UPPER_FRACTION",
    "Crop",
    "crop_upper_body",
    "upper_body_square",
    "whole_frame",
]
