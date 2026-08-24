"""Normalized-cross-correlation template tracker with composite quality gates.

The design doc prefers OpenCV CSRT, but the workspace venv ships base
``opencv-python`` without the contrib tracking module. This module provides an
equivalent CPU tracker built on ``cv2.matchTemplate`` with multi-scale search,
motion-prior windows, and exposed quality components so callers can run the
same admission policy the design reserves for CSRT.
"""

from dataclasses import dataclass

import cv2
import numpy as np

_MIN_SIDE_PX = 8
_DEFAULT_SCALES = (0.85, 1.0, 1.18)


@dataclass(frozen=True)
class VisualUpdate:
    """One visual tracking result with its raw quality components."""

    bbox: tuple[float, float, float, float]
    match_score: float
    scale: float

    @property
    def center(self) -> tuple[float, float]:
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    @property
    def area(self) -> float:
        x_min, y_min, x_max, y_max = self.bbox
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


class TemplateTracker:
    """Track one target patch across grayscale frames via bounded NCC search."""

    def __init__(
        self,
        *,
        match_threshold: float = 0.35,
        scales: tuple[float, ...] = _DEFAULT_SCALES,
        scale_jump_limit: float = 1.35,
        template_refresh_score: float = 0.9,
        min_area_ratio: float = 0.25,
    ):
        if not 0.0 < match_threshold < 1.0:
            raise ValueError("match_threshold must be in (0, 1)")
        self.match_threshold = float(match_threshold)
        self.scales = tuple(float(s) for s in scales)
        self.scale_jump_limit = float(scale_jump_limit)
        self.template_refresh_score = float(template_refresh_score)
        self.min_area_ratio = float(min_area_ratio)
        self._template: np.ndarray | None = None
        self._template_scale = 1.0
        self._bbox: tuple[float, float, float, float] | None = None

    @property
    def initialized(self) -> bool:
        return self._template is not None and self._bbox is not None

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        return self._bbox

    def initialize(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
        """Capture the template from ``bbox`` in ``gray`` (uint8, single channel)."""
        patch = self._extract_patch(gray, bbox)
        if patch is None:
            return False
        self._template = patch
        self._template_scale = 1.0
        self._bbox = self._clip_bbox(gray, bbox)
        return True

    def update(
        self,
        gray: np.ndarray,
        *,
        search_center: tuple[float, float] | None = None,
        search_radius_px: float = 60.0,
    ) -> VisualUpdate | None:
        """Locate the template near its previous position (or ``search_center``)."""
        if not self.initialized:
            return None
        x_min, y_min, x_max, y_max = self._bbox
        center = search_center if search_center is not None else ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        best: VisualUpdate | None = None
        for scale in self.scales:
            candidate = self._search_at_scale(gray, center, search_radius_px, scale)
            if candidate is not None and (best is None or candidate.match_score > best.match_score):
                best = candidate
        if best is None or best.match_score < self.match_threshold:
            return None
        if best.scale > self._template_scale * self.scale_jump_limit:
            return None
        # Guard against template collapse: a shrinking match that keeps a high
        # NCC score on a tiny sliver of texture is a classic runaway lock.
        assert self._template is not None
        template_area = float(self._template.shape[0] * self._template.shape[1])
        if best.area < self.min_area_ratio * template_area:
            return None

        self._bbox = self._clip_bbox(gray, best.bbox)
        if best.match_score >= self.template_refresh_score:
            patch = self._extract_patch(gray, self._bbox)
            if patch is not None:
                self._template = patch
                self._template_scale = best.scale
        return VisualUpdate(self._bbox, best.match_score, best.scale)

    def _search_at_scale(
        self,
        gray: np.ndarray,
        center: tuple[float, float],
        search_radius_px: float,
        scale: float,
    ) -> VisualUpdate | None:
        assert self._template is not None
        template = (
            self._template
            if scale == 1.0
            else cv2.resize(self._template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        )
        th, tw = template.shape[:2]
        if th < _MIN_SIDE_PX or tw < _MIN_SIDE_PX or th >= gray.shape[0] or tw >= gray.shape[1]:
            return None

        center_x, center_y = center
        half_w = tw / 2.0 + search_radius_px
        half_h = th / 2.0 + search_radius_px
        left = int(max(0, np.floor(center_x - half_w)))
        top = int(max(0, np.floor(center_y - half_h)))
        right = int(min(gray.shape[1], np.ceil(center_x + half_w)))
        bottom = int(min(gray.shape[0], np.ceil(center_y + half_h)))
        if right - left < tw or bottom - top < th:
            return None

        window = gray[top:bottom, left:right]
        scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(scores)
        if not np.isfinite(max_score):
            return None
        x_min = left + max_loc[0]
        y_min = top + max_loc[1]
        return VisualUpdate((x_min, y_min, x_min + tw, y_min + th), float(max_score), scale)

    def _extract_patch(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        x_min, y_min, x_max, y_max = self._clip_bbox(gray, bbox)
        if x_max - x_min < _MIN_SIDE_PX or y_max - y_min < _MIN_SIDE_PX:
            return None
        return gray[y_min:y_max, x_min:x_max]

    @staticmethod
    def _clip_bbox(gray: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        height, width = gray.shape[:2]
        x_min = int(np.clip(np.floor(bbox[0]), 0, width - 1))
        y_min = int(np.clip(np.floor(bbox[1]), 0, height - 1))
        x_max = int(np.clip(np.ceil(bbox[2]), x_min + 1, width))
        y_max = int(np.clip(np.ceil(bbox[3]), y_min + 1, height))
        return (x_min, y_min, x_max, y_max)
