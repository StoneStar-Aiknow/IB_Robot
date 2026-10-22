"""Explicit full-range JPEG NV12 transport for the two HRI adapters only."""

from __future__ import annotations

import math

import cv2
import numpy as np

NV12_FULL_RANGE = "nv12_full_range"


class Nv12Image:
    def __init__(self, message):
        width, height, stride = int(message.width), int(message.height), int(message.step)
        if message.encoding != NV12_FULL_RANGE or message.is_bigendian:
            raise ValueError("expected little-endian full-range NV12")
        if width <= 0 or height <= 0 or width % 2 or height % 2 or stride < width or stride % 2:
            raise ValueError("NV12 dimensions and stride must be positive and even")
        if len(message.data) != stride * height * 3 // 2:
            raise ValueError("NV12 payload length does not match its geometry")
        self.width, self.height = width, height
        self.shape = (height, width, 3)
        storage = np.frombuffer(message.data, dtype=np.uint8).reshape(height * 3 // 2, stride)
        self._luma = storage[:height, :width]
        self._chroma = storage[height:, :width]

    @staticmethod
    def _bgr(luma, blue_chroma, red_chroma):
        return cv2.cvtColor(cv2.merge((luma, red_chroma, blue_chroma)), cv2.COLOR_YCrCb2BGR)

    def small_bgr(self, size=640):
        ratio = float(size) / max(self.width, self.height)
        shape = (max(1, int(self.width * ratio)), max(1, int(self.height * ratio)))
        return self._bgr(
            cv2.resize(self._luma, shape, interpolation=cv2.INTER_LINEAR),
            cv2.resize(self._chroma[:, 0::2], shape, interpolation=cv2.INTER_LINEAR),
            cv2.resize(self._chroma[:, 1::2], shape, interpolation=cv2.INTER_LINEAR),
        )

    def roi_rgb(self, box, margin):
        left, top, right, bottom = (float(value) for value in box)
        if not all(math.isfinite(value) for value in (left, top, right, bottom, margin)):
            raise ValueError("person bbox and crop margin must be finite")
        if right <= left or bottom <= top or margin <= 0:
            raise ValueError("person bbox and crop margin must have positive extent")
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        half = max(right - left, bottom - top) * margin / 2
        origin_x = max(0, min(self.width - 2, math.floor(center_x - half - 8) // 2 * 2))
        origin_y = max(0, min(self.height - 2, math.floor(center_y - half - 8) // 2 * 2))
        end_x = min(self.width, max(origin_x + 2, math.ceil((center_x + half + 8) / 2) * 2))
        end_y = min(self.height, max(origin_y + 2, math.ceil((center_y + half + 8) / 2) * 2))
        luma = self._luma[origin_y:end_y, origin_x:end_x]
        chroma = self._chroma[origin_y // 2 : end_y // 2, origin_x:end_x]
        shape = (end_x - origin_x, end_y - origin_y)
        bgr = self._bgr(
            luma,
            cv2.resize(chroma[:, 0::2], shape, interpolation=cv2.INTER_LINEAR),
            cv2.resize(chroma[:, 1::2], shape, interpolation=cv2.INTER_LINEAR),
        )
        local_box = np.asarray(box, dtype=np.float32) - np.array(
            [origin_x, origin_y, origin_x, origin_y], dtype=np.float32
        )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), local_box
