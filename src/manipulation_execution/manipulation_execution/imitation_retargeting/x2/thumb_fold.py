"""Thumb driven by how close its tip is to the other fingers: a layer on hand_core's 20 joint angles.

hand_core's tracked thumb is a function of the four-finger mean, so the robot thumb
stays open whenever the fingers are straight. Here, in tracking mode, the thumb
follows its own distance to the fingers instead (zll 10-24):

- features (world keypoints over palm length): thumb tip to the index / middle /
  ring / pinky tips, and to the pinky MCP ("four"). 10-24 run 175919: touching
  0.06-0.25, open palm >= 0.60 (index tip), six >= 0.80; four, thumb to pinky MCP
  0.13-0.44, open palm >= 0.88.
- closeness per finger ``c = clip((out - d) / (out - in))``; pinky = max of its tip and
  its MCP, the MCP only as far as no other tip is close (the thumb crossing the palm in
  "ok" / touching the middle is 0.40-0.53 from the pinky MCP too); which finger =
  closeness-weighted stage s in 1..4, how far = max closeness.
- stage poses (robot thumb roll / abad / mcp): 1 index = the "ok" pose (zll's
  gesture files), 3 ring = the Victory preset's thumb (lies on the curled ring and
  pinky), 2 middle = halfway between, 4 pinky / four = the fist thumb. Thumb =
  open + closeness * (pose at s - open). Far from every finger = open.
- collision guard, generalising hand_core's 3d-thumb rule (10-23 real contact:
  thumb across the palm, pips commanded 0.67 stuck at 0.50, thumb_roll fault):
  past |abad| ABAD_IN the thumb is "in". Each stage allows the fingers it covers to
  curl up to the stage pose's own value (ALLOW), the others PIP_CAP. The thumb goes
  in only with every finger either open or (a covered one) already curled to 90 %
  of its target, covered fingers' targets clipped to the allowance; while in, no
  finger crosses between open and curled; a finger that wants to, or a stage
  change that no longer allows a curled finger, makes the thumb back out to
  ABAD_IN along its line first. A finger the stage does not cover curled (a fist,
  "six"): the thumb stays out (ABAD_IN at most), or hand_core's thumb if that is
  deeper (the fist, with hand_core's own order).
- gesture presets and the exit of one: hand_core's output, after backing out.
- rate limit: each joint follows hand_core's own motion plus at most hand_core's
  B2 step per tick (thumb joints scaled together).
"""

from __future__ import annotations

from collections import deque

import numpy as np

PIPS = (4, 5, 7, 9)  # index, middle, ring, pinky
THUMB = (0, 1, 2)
ABAD = 1
ABAD_IN = 0.6  # hand_core T3_ABAD_IN
PIP_CAP = 0.4  # hand_core T3_PIP_CAP
FOLD_FRAC = 0.9  # hand_core GestureConfig.fold_frac
WINDOW_S = 0.25  # feature samples before t_out that are medianed
_EPS = 1e-9

# Robot thumb (roll, abad, mcp) per stage. ok: zll's gesture_ok_{left,right} (omnihand
# gesture studio, URDF -> X2: left as granger's e5_tour_poses, right per proposal v6
# §7.2). Victory: dexhand_robot.json's preset, run on the robot since 10-23.
_TH_OK = {"left": (-0.035, 1.64, -0.739), "right": (0.14, -1.64, 0.646)}
_TH_VICTORY = {"left": (-0.80, 1.64, -0.838), "right": (0.80, -1.64, 0.838)}
# Pips (index, middle, ring, pinky) a thumb in at each stage allows. ok: its own index
# (0.877 left, 1.024 right); middle: same as ok, for middle and ring (not run yet);
# ring: the Victory preset's ring and pinky; four: ring and pinky as a fist's (the
# pinky tip stage, zll's "four" touching the curled pinky), index and middle hand_core's 3d rule.
_OK_INDEX = {"left": 0.877, "right": 1.024}


def thumb_dists(world_landmarks) -> np.ndarray:
    """21 x 3 world keypoints -> thumb tip to index / middle / ring / pinky tips and pinky MCP,
    over palm length; NaNs when missing or degenerate."""
    a = np.asarray(world_landmarks, dtype=np.float64)
    if a.size != 63 or not np.all(np.isfinite(a)):
        return np.full(5, np.nan)
    a = a.reshape(21, 3)
    palm = float(np.linalg.norm(a[9] - a[0]))
    if palm < 1e-4:
        return np.full(5, np.nan)
    return np.linalg.norm(a[[8, 12, 16, 20, 17]] - a[4], axis=1) / palm


class ThumbFold:
    def __init__(self, cfg, out_hz: float, pk_in: float, pk_out: float, tip_in: float = 0.25, tip_out: float = 0.60):
        if not (0.0 < pk_in < pk_out and 0.0 < tip_in < tip_out):
            raise ValueError(f"thumb fold needs 0 < in < out, got pk {pk_in}/{pk_out} tip {tip_in}/{tip_out}")
        self.pk = (float(pk_in), float(pk_out))
        self.tip = (float(tip_in), float(tip_out))
        sides = ("left", "right")
        self.q_open = np.stack([np.asarray(cfg.q_open[s], dtype=np.float64) for s in sides])
        q_fist = np.stack([np.asarray(cfg.q_fist[s], dtype=np.float64) for s in sides])
        travel = np.abs(q_fist - self.q_open)
        tt = np.array([cfg.full_travel_s_thumb if j in THUMB else cfg.full_travel_s for j in range(10)])
        self.step = travel / (tt * float(out_hz))
        th = list(THUMB)
        # (hand, stage 0..3, thumb joint)
        ok, vic = (np.array([_TH_OK[s] for s in sides]), np.array([_TH_VICTORY[s] for s in sides]))
        self.stage_thumb = np.stack([ok, (ok + vic) / 2.0, vic, q_fist[:, th]], axis=1)
        # (hand, stage 0..3, pip index/middle/ring/pinky)
        allow = np.full((2, 4, 4), PIP_CAP)
        for h, s in enumerate(sides):
            allow[h, 0, 0] = _OK_INDEX[s]
            allow[h, 1, 1] = allow[h, 1, 2] = _OK_INDEX[s]
            allow[h, 2, 2] = allow[h, 2, 3] = allow[h, 3, 2] = allow[h, 3, 3] = 1.57
        self.allow = allow
        self._buf: list[deque] = [deque(), deque()]  # robot hand -> (t_camera, 5 distances)
        self._prev: np.ndarray | None = None  # (2, 10) last output
        self._hc_prev: np.ndarray | None = None  # (2, 10) last hand_core input
        self._in = [False, False]  # the layer's thumb is past ABAD_IN
        self._caps = np.full((2, 4), PIP_CAP)  # per pip, while in
        self.active = [False, False]  # the thumb target came from this layer this tick (status)
        self.stage = [float("nan"), float("nan")]  # status: s in 1..4, NaN = far
        self.closeness = [0.0, 0.0]

    def push(self, t_camera: float, d_robot) -> None:
        """One frame's thumb_dists per robot hand (mirror already applied); NaNs = no hand."""
        for h in range(2):
            d = np.asarray(d_robot[h], dtype=np.float64)
            if d.shape == (5,) and np.all(np.isfinite(d)):
                self._buf[h].append((t_camera, d))
            while self._buf[h] and self._buf[h][0][0] < t_camera - 3.0:
                self._buf[h].popleft()

    def _dists_at(self, h: int, t_out: float) -> np.ndarray | None:
        xs = [d for t, d in self._buf[h] if t_out - WINDOW_S <= t <= t_out]
        return np.median(np.stack(xs), axis=0) if xs else None

    def _reach(self, h: int, d: np.ndarray):
        """Distances -> (thumb pose, allowed pips, stage s, closeness)."""
        (ti, to), (pi, po) = self.tip, self.pk
        c = np.clip((to - d[:4]) / (to - ti), 0.0, 1.0)
        c_mcp = float(np.clip((po - d[4]) / (po - pi), 0.0, 1.0)) * (1.0 - c[:3].max())
        c[3] = max(c[3], c_mcp)
        w = c ** 4
        th = self.q_open[h, list(THUMB)]
        if w.sum() < 1e-12:
            return th.copy(), np.full(4, PIP_CAP), float("nan"), 0.0
        s = float(np.dot(np.arange(4), w) / w.sum())  # 0..3
        k = min(int(s), 2)
        f = s - k
        pose = (1 - f) * self.stage_thumb[h, k] + f * self.stage_thumb[h, k + 1]
        allow = (1 - f) * self.allow[h, k] + f * self.allow[h, k + 1]
        cm = float(c.max())
        return th + cm * (pose - th), allow, s + 1.0, cm

    def _along(self, h: int, thumb: np.ndarray, abad_max: float) -> np.ndarray:
        """Scale a thumb pose toward open (its line) so that |abad| <= abad_max."""
        o = self.q_open[h, list(THUMB)]
        a, a0 = abs(thumb[ABAD]), abs(o[ABAD])
        if a <= abad_max or a - a0 < 1e-9:
            return thumb
        return o + max(0.0, (abad_max - a0) / (a - a0)) * (thumb - o)

    def apply(self, q20, t_out: float, modes) -> np.ndarray:
        hc = np.asarray(q20, dtype=np.float64).reshape(2, 10)
        if self._prev is None:
            self._prev, self._hc_prev = hc.copy(), hc.copy()
        out = np.empty_like(hc)
        pips, th = list(PIPS), list(THUMB)
        for h in range(2):
            prev, tgt = self._prev[h], hc[h].copy()
            self.active[h], self.stage[h], self.closeness[h] = False, float("nan"), 0.0
            mine = None  # (thumb, allowed pips) when the layer drives the thumb
            if modes[h] == "track":
                d = self._dists_at(h, t_out)
                if d is not None:
                    thumb, allow, self.stage[h], self.closeness[h] = self._reach(h, d)
                    mine = (thumb, allow)
            want = np.abs(tgt[pips])
            if self._in[h]:
                caps = self._caps[h] if mine is None else np.minimum(self._caps[h], mine[1])
                if mine is not None:  # a covered finger past its allowance is clipped, not a reason to back out
                    want = np.where(mine[1] > PIP_CAP + _EPS, np.minimum(want, mine[1]), want)
                out_first = (mine is None or np.any(want > caps + _EPS)
                             or np.any(np.abs(prev[pips]) > caps + _EPS))
                if out_first:
                    # back out along the current line; fingers no further than now until out
                    tgt[th] = self._along(h, prev[th], ABAD_IN)
                    hold = np.maximum(self._caps[h], np.abs(prev[pips]))
                    tgt[pips] = np.clip(tgt[pips], -hold, hold)
                else:
                    tgt[th] = mine[0]
                    tgt[pips] = np.clip(tgt[pips], -caps, caps)
                    self.active[h] = True
            elif mine is not None:
                thumb, allow = mine
                covered = allow > PIP_CAP + _EPS
                p = np.abs(prev[pips])
                if np.any(want[~covered] > PIP_CAP + _EPS) or np.any(p[~covered] > PIP_CAP + _EPS):
                    # an uncovered finger curled: the thumb stays out; a fist keeps hand_core's thumb
                    out_t = self._along(h, thumb, ABAD_IN)
                    o = self.q_open[h, th]
                    if np.linalg.norm(out_t - o) > np.linalg.norm(tgt[th] - o):
                        tgt[th] = out_t
                        self.active[h] = True
                elif abs(thumb[ABAD]) <= ABAD_IN:
                    tgt[th] = thumb  # not in: the fingers are hand_core's
                    self.active[h] = True
                else:
                    goal = np.where(covered, np.minimum(want, allow), want)
                    ready = np.all((p <= PIP_CAP + _EPS) & (goal <= PIP_CAP + _EPS)
                                   | covered & (goal > PIP_CAP) & (p >= FOLD_FRAC * goal - _EPS) & (p <= allow + _EPS))
                    tgt[th] = thumb if ready else self._along(h, thumb, ABAD_IN)
                    tgt[pips] = np.sign(tgt[pips]) * goal
                    self.active[h] = True
            # follow hand_core's own motion, catch up at most one B2 step; thumb joints scaled together
            lim = self.step[h] + np.abs(hc[h] - self._hc_prev[h])
            dq = tgt - prev
            r = np.max(np.abs(dq[th]) / np.maximum(lim[th], 1e-12))
            if r > 1.0:
                dq[th] /= r
            rest = [j for j in range(10) if j not in THUMB]
            dq[rest] = np.clip(dq[rest], -lim[rest], lim[rest])
            out[h] = prev + dq
            if abs(out[h, ABAD]) <= ABAD_IN + _EPS:
                self._in[h] = False
            elif not self._in[h] and self.active[h]:
                # caps from the fingers as the thumb goes in: open stays open, curled stays within allowance
                self._in[h] = True
                p = np.abs(out[h, pips])
                self._caps[h] = np.where(p <= PIP_CAP + _EPS, PIP_CAP, np.maximum(p, mine[1]))
        self._prev, self._hc_prev = out.copy(), hc.copy()
        return out.reshape(20)
