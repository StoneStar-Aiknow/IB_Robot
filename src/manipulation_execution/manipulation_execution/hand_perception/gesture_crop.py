"""Where the two-hand gesture model looks: gating, the square crop, rate limits. **No ROS, no cv2.**

The hand landmarks of a frame (21 image points per hand, source pixels, MediaPipe
order) decide whether the gesture model runs on that frame and on which square:

- both hands: skip when the two hand centres (mean of the 21 points) are further
  apart than ``far_ratio`` x the person box height (the box is the PEAR person
  box the hands were found in; upright, its height is about the arm span, so the
  ruler only ever errs towards running). Otherwise the square is the union of the
  two hands' point boxes, scaled by ``k`` about its centre, side = the long edge;
- one hand: a square of ``one_hand_palms`` palms centred on that hand (wide
  enough to take in a second hand the landmark model lost), at most
  ``one_hand_max_rate_hz``;
- no hand: no run.

A square under ``min_side_px`` is skipped (the person is too far for 320 px to
show anything); one over the image's long edge is cut to it, about the same
centre (only a landmark glitch gets there; it bounds the copy). All runs together are at most ``max_rate_hz``, by camera stamp.
Palm = max(|wrist -> middle MCP|, 1.25 x |index MCP -> pinky MCP|) in pixels, the
larger hand's when there are two; it sizes the one-hand square only.

The square may reach past the image: ``extract_square`` pads it with 114, so the
model sees a letterbox with nothing to fit (the square is already square).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
PAD_VALUE = 114
MAX_SIDE_PX = 4096  # without the image size: a square this big would only be a landmark glitch
# Camera stamps come in 33 ms steps at 30 fps; a period of exactly two frames
# (66.7 ms at 15 Hz) must not lose to a fraction of a millisecond of jitter.
RATE_SLACK_NS = 5_000_000
BOTH, ONE = "both", "one"


@dataclass(frozen=True)
class CropConfig:
    far_ratio: float = 0.6
    k: float = 1.5
    one_hand_palms: float = 4.0
    min_side_px: float = 64.0
    max_rate_hz: float = 15.0
    one_hand_max_rate_hz: float = 5.0
    # A newer hand frame waiting only yields up to this rate, it never starves the
    # gesture: with the hands saturated (out ~17 fps) 10 Hz is every 2nd frame,
    # ~5 runs in the matcher's 0.6 s vote window (it needs 4).
    busy_rate_hz: float = 10.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"gesture {f.name} must be a finite positive number, got {value!r}")


@dataclass(frozen=True)
class CropPlan:
    """``run`` false: ``reason`` says why (``hands`` / ``box`` / ``far`` / ``small``)."""

    run: bool
    reason: str
    mode: str | None = None
    x0: int = 0
    y0: int = 0
    side: int = 0
    box_h: float = float("nan")
    dist_ratio: float = float("nan")  # two-hand centre distance / person box height; NaN with one hand
    palm: float = float("nan")

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x0 + self.side, self.y0 + self.side)


def valid_hands(lms) -> list[np.ndarray]:
    """The hands with 21 finite 2D points, as (21,2) float arrays."""
    out = []
    for lm in lms:
        a = np.asarray(lm, dtype=np.float64)
        if a.shape == (21, 2) and np.isfinite(a).all():
            out.append(a)
    return out


def hand_centre(lm: np.ndarray) -> np.ndarray:
    return np.asarray(lm, dtype=np.float64).mean(axis=0)


def palm_px(lm: np.ndarray) -> float:
    lm = np.asarray(lm, dtype=np.float64)
    length = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
    width = float(np.linalg.norm(lm[PINKY_MCP] - lm[INDEX_MCP]))
    return max(length, 1.25 * width)


def _square(cx: float, cy: float, side: float) -> tuple[int, int, int]:
    s = int(round(side))
    return int(round(cx - s / 2.0)), int(round(cy - s / 2.0)), s


def plan_crop(lms, person_box, cfg: CropConfig, image_hw=None) -> CropPlan:
    """The square to run on for this frame's hands (source pixels), or why not.

    ``image_hw`` (height, width): the side is at most the longer of the two."""
    hands = valid_hands(lms)[:2]
    if not hands:
        return CropPlan(False, "hands")
    box = np.asarray(person_box, dtype=np.float64).reshape(-1)
    box_h = float(box[3] - box[1]) if box.size == 4 else float("nan")
    if not math.isfinite(box_h) or box_h <= 0:
        return CropPlan(False, "box")
    palm = max(palm_px(h) for h in hands)
    if len(hands) == 2:
        dist = float(np.linalg.norm(hand_centre(hands[0]) - hand_centre(hands[1])))
        ratio = dist / box_h
        if ratio > cfg.far_ratio:
            return CropPlan(False, "far", BOTH, box_h=box_h, dist_ratio=ratio, palm=palm)
        pts = np.concatenate(hands)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        (cx, cy), long_edge = (lo + hi) / 2.0, float(max(hi - lo))
        mode, side = BOTH, cfg.k * long_edge
    else:
        ratio = float("nan")
        cx, cy = hand_centre(hands[0])
        mode, side = ONE, cfg.one_hand_palms * palm
    x0, y0, s = _square(float(cx), float(cy), side)
    if not math.isfinite(side) or side < cfg.min_side_px:
        return CropPlan(False, "small", mode, x0, y0, s, box_h, ratio, palm)
    if image_hw is not None and s > max(image_hw):
        x0, y0, s = _square(float(cx), float(cy), float(max(image_hw)))
    if s > MAX_SIDE_PX:
        return CropPlan(False, "large", mode, x0, y0, s, box_h, ratio, palm)
    return CropPlan(True, "", mode, x0, y0, s, box_h, ratio, palm)


class RateGate:
    """All runs <= ``max_rate_hz``, one-hand runs <= ``one_hand_max_rate_hz``, by camera stamp (ns).

    ``allow`` only asks; ``mark`` records a run. A stamp going backwards forgets
    the history (the next frame may run). ``allow_busy``: may a frame run while a
    newer hand frame is waiting (``busy_rate_hz`` since the last run)."""

    def __init__(self, cfg: CropConfig):
        self._any_ns = int(1e9 / cfg.max_rate_hz) - RATE_SLACK_NS
        self._one_ns = int(1e9 / cfg.one_hand_max_rate_hz) - RATE_SLACK_NS
        self._busy_ns = int(1e9 / cfg.busy_rate_hz) - RATE_SLACK_NS
        self.last_any: int | None = None
        self.last_one: int | None = None

    def allow(self, mode: str, stamp_ns: int) -> bool:
        for last in (self.last_any, self.last_one):
            if last is not None and stamp_ns < last:
                self.last_any = self.last_one = None
                return True
        if self.last_any is not None and stamp_ns - self.last_any < self._any_ns:
            return False
        return not (mode == ONE and self.last_one is not None and stamp_ns - self.last_one < self._one_ns)

    def allow_busy(self, stamp_ns: int) -> bool:
        last = self.last_any
        return last is None or stamp_ns < last or stamp_ns - last >= self._busy_ns

    def mark(self, mode: str, stamp_ns: int) -> None:
        self.last_any = stamp_ns
        if mode == ONE:
            self.last_one = stamp_ns


def extract_square(img: np.ndarray, x0: int, y0: int, side: int, pad: int = PAD_VALUE) -> np.ndarray:
    """``img[y0:y0+side, x0:x0+side]`` with the part outside the image filled with ``pad``."""
    h, w = img.shape[:2]
    out = np.full((side, side, *img.shape[2:]), pad, dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out
