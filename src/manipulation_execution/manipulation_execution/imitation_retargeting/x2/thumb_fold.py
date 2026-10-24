"""Thumb fold with the fingers open: a layer on hand_core's 20 joint angles, off unless configured.

hand_core's tracked thumb is a function of the four-finger mean (thumb follow can
only open it further), so a thumb folded into the palm with the fingers straight
("four") shows on the robot as a thumb spread open. This adds the fold:

- feature: thumb tip to pinky MCP over palm length (world keypoints). 10-24 run
  171354: thumb in the palm 0.27-0.53, thumb out (open palm, six, thumbs up)
  0.96-1.35. ``fold = clip((pk_out - d) / (pk_out - pk_in), 0, 1)``.
- only in tracking mode (no gesture preset) and only while the four pips are open
  (<= PIP_CAP): the thumb goes along the line open -> fist thumb, to
  ``max(fold, hand_core's own)``.
- collision guard, hand_core's 3d-thumb rule (10-23 real contact: thumb across the
  palm, pips commanded 0.67 stuck at 0.50, thumb_roll fault): a thumb that went
  past |abad| ABAD_IN with the fingers open keeps the pips <= PIP_CAP; when the
  fingers want to curl, the thumb first backs out to ABAD_IN along its line to
  open. A thumb may only pass ABAD_IN with the fingers open or with the fingers
  it covers already folded to 90 % of their target (hand_core's gesture order).
- rate limit: each joint follows hand_core's own motion plus at most hand_core's
  B2 step per tick (thumb joints scaled together), so where the layer does not
  act the output is hand_core's exactly.
"""

from __future__ import annotations

from collections import deque

import numpy as np

PIPS = (4, 5, 7, 9)
THUMB = (0, 1, 2)
ABAD = 1
ABAD_IN = 0.6  # hand_core T3_ABAD_IN
PIP_CAP = 0.4  # hand_core T3_PIP_CAP
FOLD_FRAC = 0.9  # hand_core GestureConfig.fold_frac
WINDOW_S = 0.25  # feature samples before t_out that are medianed


def thumb_pinky(world_landmarks) -> float:
    """21 x 3 world keypoints -> thumb tip to pinky MCP over palm length; NaN when missing or degenerate."""
    a = np.asarray(world_landmarks, dtype=np.float64)
    if a.size != 63 or not np.all(np.isfinite(a)):
        return float("nan")
    a = a.reshape(21, 3)
    palm = float(np.linalg.norm(a[9] - a[0]))
    if palm < 1e-4:
        return float("nan")
    return float(np.linalg.norm(a[17] - a[4]) / palm)


class ThumbFold:
    def __init__(self, cfg, out_hz: float, pk_in: float, pk_out: float):
        if not 0.0 < pk_in < pk_out:
            raise ValueError(f"thumb fold needs 0 < pk_in < pk_out, got {pk_in} / {pk_out}")
        self.pk_in, self.pk_out = float(pk_in), float(pk_out)
        sides = ("left", "right")
        self.q_open = np.stack([np.asarray(cfg.q_open[s], dtype=np.float64) for s in sides])
        span = np.stack([np.asarray(cfg.q_fist[s], dtype=np.float64) for s in sides]) - self.q_open
        self.span = span
        travel = np.abs(span)
        tt = np.array([cfg.full_travel_s_thumb if j in THUMB else cfg.full_travel_s for j in range(10)])
        self.step = travel / (tt * float(out_hz))
        self._buf: list[deque] = [deque(), deque()]  # robot hand -> (t_camera, fold)
        self._prev: np.ndarray | None = None  # (2, 10) last output
        self._hc_prev: np.ndarray | None = None  # (2, 10) last hand_core input
        self._in_open = [False, False]  # thumb went past ABAD_IN with the fingers open
        self.active = [False, False]  # the layer moved the thumb this tick (status)

    def push(self, t_camera: float, d_robot) -> None:
        """One frame's feature per robot hand (mirror already applied); NaN = no hand."""
        for h in range(2):
            d = float(d_robot[h])
            if np.isfinite(d):
                self._buf[h].append((t_camera, float(np.clip((self.pk_out - d) / (self.pk_out - self.pk_in), 0.0, 1.0))))
            while self._buf[h] and self._buf[h][0][0] < t_camera - 3.0:
                self._buf[h].popleft()

    def _fold_at(self, h: int, t_out: float) -> float:
        xs = [f for t, f in self._buf[h] if t_out - WINDOW_S <= t <= t_out]
        return float(np.median(xs)) if xs else float("nan")

    def _along(self, h: int, thumb: np.ndarray, abad_max: float) -> np.ndarray:
        """Scale a thumb pose toward open (its line) so that |abad| <= abad_max."""
        rel = thumb - self.q_open[h, list(THUMB)]
        a = abs(thumb[ABAD])
        if a <= abad_max or a < 1e-9:
            return thumb
        k = max(0.0, (abad_max - abs(self.q_open[h, ABAD])) / (a - abs(self.q_open[h, ABAD])))
        return self.q_open[h, list(THUMB)] + k * rel

    def apply(self, q20, t_out: float, modes) -> np.ndarray:
        hc = np.asarray(q20, dtype=np.float64).reshape(2, 10)
        if self._prev is None:
            self._prev, self._hc_prev = hc.copy(), hc.copy()
        out = np.empty_like(hc)
        pips, th = list(PIPS), list(THUMB)
        for h in range(2):
            prev = self._prev[h]
            tgt = hc[h].copy()
            fingers_open = max(abs(tgt[pips]).max(), abs(prev[pips]).max()) <= PIP_CAP + 1e-9
            fold = self._fold_at(h, t_out) if (modes[h] == "track") else float("nan")
            self.active[h] = False
            if np.isfinite(fold) and fingers_open:
                own = self.q_open[h, th] + fold * self.span[h, th]
                if abs(own[ABAD]) > abs(tgt[ABAD]):
                    tgt[th] = own
                    self.active[h] = True
            # a thumb in the palm with the fingers open: fingers stay open; to curl them it backs out first
            if self._in_open[h]:
                if abs(tgt[pips]).max() > PIP_CAP:
                    tgt[th] = self._along(h, tgt[th], ABAD_IN)
                tgt[pips] = np.clip(tgt[pips], -PIP_CAP, PIP_CAP)
            # the thumb passes ABAD_IN only with the fingers open or the covered fingers already folded
            if abs(tgt[ABAD]) > ABAD_IN:
                open_ok = abs(tgt[pips]).max() <= PIP_CAP + 1e-9 and abs(prev[pips]).max() <= PIP_CAP + 1e-9
                ahead_ok = all(abs(prev[j]) >= FOLD_FRAC * abs(tgt[j]) for j in pips if abs(tgt[j]) > PIP_CAP)
                if not (open_ok or ahead_ok):
                    # no deeper than now (a thumb already past ABAD_IN is held, not yanked out)
                    tgt[th] = self._along(h, tgt[th], max(ABAD_IN, abs(prev[ABAD])))
            # follow hand_core's own motion, catch up at most one B2 step; thumb joints scaled together
            lim = self.step[h] + np.abs(hc[h] - self._hc_prev[h])
            d = tgt - prev
            r = np.max(np.abs(d[th]) / np.maximum(lim[th], 1e-12))
            if r > 1.0:
                d[th] /= r
            rest = [j for j in range(10) if j not in THUMB]
            d[rest] = np.clip(d[rest], -lim[rest], lim[rest])
            out[h] = prev + d
            a_out = abs(out[h, ABAD])
            if a_out <= ABAD_IN + 1e-9:
                self._in_open[h] = False
            elif not self._in_open[h] and abs(out[h, pips]).max() <= PIP_CAP + 1e-9:
                self._in_open[h] = True
        self._prev, self._hc_prev = out.copy(), hc.copy()
        return out.reshape(20)
