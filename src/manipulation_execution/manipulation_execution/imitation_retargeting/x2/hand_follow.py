"""Dexterous-hand following beside the arm stream: HandFrame observations in, 20 hand radians out.

The arm path is untouched. This holds its own ``hand_core.HandStream``
(video2x2_runtime 570c53c, where ``FollowMeStream`` holds it). ``hand_core.py``
and its ``data/*.json`` are not in this repository: they are loaded at run time
from ``extern_dir`` (``hand_perception.npu_hands.load_extern``).
It runs on the same output clock idea as the arm: ``t_out = now - offset``,
``offset`` = median age of the camera frames + ``latency_s``. The ages are the
PEAR frames' when they arrive (the arm's clock), else the hand frames' own.

Everything else is hand_core as upstream: curl per finger -> joint map, window
smoothing, B2 rate limits, a hand whose data is invalid for ``stale_s`` going
to neutral, and the gesture layer (labels with hysteresis -> presets from
``dexhand_robot.json``). ``tick`` returns None until the first hand frame.

``thumb_follow`` (off by default) wires hand_core's ThumbFollowConfig: the
tracked thumb, otherwise a function of the four-finger mean only, may open
further when the person's thumb is out (``thumb_tuck`` of the world
keypoints); closing stays capped by the four-finger curve.
``thumb_fold`` (off by default) adds the other direction with the fingers
open, after hand_core (``thumb_fold.ThumbFold``).
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from collections.abc import Sequence

import numpy as np

from ...hand_perception.npu_hands import DEFAULT_EXTERN_DIR, HAND_CORE_PY, load_extern
from .thumb_fold import ThumbFold, thumb_pinky

HAND_COUNT = 20
AGE_WINDOW = 50  # frames in the age median
HISTORY_S = 1.0  # followme_core's history_s
_THUMB_NEAR = [6, 7, 10, 11]  # index / middle PIP and DIP
_PALM = (0, 9)  # wrist -> middle MCP


def thumb_tuck(world_landmarks) -> float:
    """21 x 3 world keypoints -> thumb tip's nearest distance to the index / middle PIP / DIP over palm
    length (video2x2 hand_keypoints.thumb_tuck). NaN when missing or degenerate."""
    a = np.asarray(world_landmarks, dtype=np.float64)
    if a.size != 63 or not np.all(np.isfinite(a)):
        return float("nan")
    a = a.reshape(21, 3)
    palm = float(np.linalg.norm(a[_PALM[1]] - a[_PALM[0]]))
    if palm < 1e-4:
        return float("nan")
    return float(np.linalg.norm(a[_THUMB_NEAR] - a[4], axis=1).min() / palm)


class HandFollow:
    def __init__(
        self,
        *,
        latency_s: float,
        lookahead_s: float,
        mirror: bool,
        out_hz: float,
        stale_s: float = 1.0,
        gestures: bool = True,
        gesture_labels: Sequence[str] | None = None,
        gesture_theta_in: float | None = None,
        thumb_follow: tuple[float, float] | None = None,
        thumb_fold: tuple[float, float] | None = None,
        calib_path=None,
        robot_path=None,
        extern_dir=DEFAULT_EXTERN_DIR,
    ):
        hand_core = load_extern(extern_dir, HAND_CORE_PY)
        cfg = hand_core.load_config(calib_path or hand_core.CALIB_MEDIAPIPE_PATH, robot_path)
        self.cfg = cfg
        self.latency_s = float(latency_s)
        self.mirror = bool(mirror)
        self.stream = hand_core.HandStream(
            cfg, mirror=self.mirror, history_s=HISTORY_S, lookahead_s=lookahead_s, out_hz=out_hz,
            buf_span=latency_s + HISTORY_S + 1.5, stale_s=stale_s,
            gesture=(hand_core.GestureConfig(**({} if gesture_theta_in is None else {"theta_in": float(gesture_theta_in)}))
                     if gestures else None),
            gesture_labels=hand_core.GESTURE_LABELS if gesture_labels is None else gesture_labels,
            thumb_follow=None if thumb_follow is None else hand_core.ThumbFollowConfig(*map(float, thumb_follow)),
        )
        self.fold = None if thumb_fold is None else ThumbFold(cfg, out_hz, *map(float, thumb_fold))
        self._lock = threading.Lock()
        self._pear_ages: deque[float] = deque(maxlen=AGE_WINDOW)
        self._hand_ages: deque[float] = deque(maxlen=AGE_WINDOW)
        self._last_t: float | None = None
        self.frames = 0
        self.detected = [0, 0]
        self.out_of_order = 0
        self.last_t_out: float | None = None

    @property
    def neutral(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.stream.q_neutral)

    def on_pear(self, now: float, t_camera: float) -> None:
        with self._lock:
            self._pear_ages.append(now - t_camera)

    def on_hand(self, now: float, t_camera: float, left, right) -> None:
        """One HandFrame. ``left`` / ``right`` (the person's) are ``(curl_rad[5], label, score[, world_landmarks[63]])``
        or None."""
        with self._lock:
            self._hand_ages.append(now - t_camera)
            self.frames += 1
            if self._last_t is not None and t_camera < self._last_t:
                self.out_of_order += 1
            self._last_t = t_camera if self._last_t is None else max(self._last_t, t_camera)
            curls = []
            for side, obs in enumerate((left, right)):
                if obs is None:
                    curls.append(None)
                    continue
                self.detected[side] += 1
                curls.append(np.asarray(obs[0], dtype=np.float64))
            self.stream.push(t_camera, self.stream.curl_external(curls[0], curls[1]))
            if self.stream.gesture is not None:
                human = [
                    (obs[1], float(obs[2]), True) if obs is not None else ("", 0.0, False)
                    for obs in (left, right)
                ]
                self.stream.push_gesture(t_camera, human[::-1] if self.mirror else human)
            if self.stream.thumb_follow is not None:
                d = [thumb_tuck(obs[3]) if obs is not None and len(obs) > 3 else float("nan") for obs in (left, right)]
                self.stream.push_thumb(t_camera, d[::-1] if self.mirror else d)
            if self.fold is not None:
                d = [thumb_pinky(obs[3]) if obs is not None and len(obs) > 3 else float("nan") for obs in (left, right)]
                self.fold.push(t_camera, d[::-1] if self.mirror else d)

    def tick(self, now: float) -> list[float] | None:
        with self._lock:
            if self._last_t is None:
                return None
            ages = self._pear_ages or self._hand_ages
            t_out = now - (statistics.median(ages) + self.latency_s)
            self.last_t_out = t_out
            q = self.stream.emit(t_out, 1.0, False)
            if self.fold is not None:
                modes = [m for m, _ in self.stream.gesture_state()] or ["track", "track"]
                q = self.fold.apply(q, t_out, modes)
        return [float(v) for v in q]

    def status(self, now: float) -> dict:
        with self._lock:
            gestures = self.stream.gesture_state() if self.stream.gesture is not None else []
            return {
                "hand_frames": self.frames,
                "hand_detected": list(self.detected),
                "hand_out_of_order": self.out_of_order,
                "hand_age_ms": round(statistics.median(self._hand_ages) * 1e3, 1) if self._hand_ages else None,
                "pear_age_ms": round(statistics.median(self._pear_ages) * 1e3, 1) if self._pear_ages else None,
                "hand_newest_behind_t_out_ms": (
                    round((self._last_t - self.last_t_out) * 1e3, 1)
                    if self._last_t is not None and self.last_t_out is not None else None
                ),
                "hand_stale": [bool(v) for v in self.stream.stale_last],
                "gesture": gestures,
                "thumb_fold_active": list(self.fold.active) if self.fold is not None else None,
            }
