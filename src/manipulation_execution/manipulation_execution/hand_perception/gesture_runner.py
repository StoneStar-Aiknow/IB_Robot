"""The two-hand gesture model on the hand worker's frames, after the HandFrame is out. **No ROS.**

``hand_perception_node`` owns one ``GestureRunner`` when ``gesture_enabled`` is
set and the model path is not empty. The node's worker thread calls ``load``
once, right after the hand backend is up, and then ``step`` once per frame
*after* that frame's HandFrame is published, with the frame it already decoded,
its person box and the 21 image points of each hand found. ``step`` never raises.

Per ``step``: skip if the slot already holds a newer frame (the hands come
first); ``gesture_crop.plan_crop`` picks the square or a skip; ``RateGate``
keeps the runs under the rates; then pad + resize + om + ``post_fast`` and one
JSON message (``x2_hand_gesture/v1``) per run.

Fail silent, by design: the hands never wait on the gesture model.

- the om does not load: disabled, one INFO line;
- a run raises, or the output is not a finite [1,38,2100] tensor: result
  dropped, rate-limited INFO; ``fail_limit`` in a row: disabled;
- a run takes over ``slow_ms`` (the whole step: crop, resize, NPU, post)
  ``slow_limit`` times in a row: disabled (that run's result is still sent).

Disabled is for the rest of the process; the node then drops the runner. No image
is ever written or logged; the log line is numbers only.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, fields

from . import gesture_crop as gc
from . import gesture_om as go

SCHEMA = "x2_hand_gesture/v1"


@dataclass(frozen=True)
class RunnerConfig:
    conf: float = 0.25
    slow_ms: float = 20.0
    slow_limit: int = 3
    fail_limit: int = 5
    log_period_s: float = 1.0  # summary line period; <= 0: no summary

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                raise ValueError(f"gesture {f.name} must be a finite number, got {v!r}")
        if not 0 < self.conf < 1:
            raise ValueError(f"gesture conf must be in (0, 1), got {self.conf!r}")
        if self.slow_ms <= 0:
            raise ValueError(f"gesture slow_ms must be positive, got {self.slow_ms!r}")
        for name in ("slow_limit", "fail_limit"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"gesture {name} must be an integer >= 1, got {v!r}")


def payload(stamp_ns: int, actor_id: int, cls: str, score: float, mode: str, roi, infer_ms: float) -> dict:
    return {
        "schema": SCHEMA,
        "stamp_ns": int(stamp_ns),
        "actor_id": int(actor_id),
        "cls": cls,
        "score": round(float(score), 4),
        "crop": mode,
        "roi": [int(v) for v in roi],
        "infer_ms": round(float(infer_ms), 2),
    }


class GestureRunner:
    """``publish(text)`` sends one JSON string; ``log(text)`` writes one INFO line."""

    def __init__(
        self,
        crop: gc.CropConfig,
        cfg: RunnerConfig,
        *,
        model_path,
        device: int,
        publish,
        log,
        clock=time.monotonic,
        om_factory=go.GestureOm,
    ):
        self.crop, self.cfg = crop, cfg
        self.model_path, self.device = model_path, int(device)
        self._publish, self._log, self._clock, self._om_factory = publish, log, clock, om_factory
        self.om = None
        self.enabled = False
        self.disabled_reason = ""
        self.rate = gc.RateGate(crop)
        self._fails = self._slows = 0
        self._last_fail_log = -math.inf
        self._period_t0 = clock()
        self._runs: Counter = Counter()
        self._skips: Counter = Counter()
        self._last: dict = {}

    # -- lifecycle (worker thread) -------------------------------------------
    def load(self) -> bool:
        try:
            self.om = self._om_factory(self.model_path, self.device)
        except Exception as e:  # noqa: BLE001  any load failure only turns the gesture off
            self.om = None
            self.disabled_reason = "load"
            self._log(f"gesture model off: load failed ({type(e).__name__}: {e}); hands unaffected")
            return False
        self.enabled = True
        self._log(
            f"gesture model {getattr(self.om, 'path', self.model_path)} ready; conf {self.cfg.conf:g}, "
            f"rate <= {self.crop.max_rate_hz:g} Hz (one hand <= {self.crop.one_hand_max_rate_hz:g} Hz)"
        )
        return True

    def close(self) -> None:
        om, self.om = self.om, None
        self.enabled = False
        if om is not None:
            with contextlib.suppress(Exception):
                om.close()

    def _disable(self, reason: str, detail: str) -> None:
        self.disabled_reason = reason
        self._log(f"gesture model off: {detail}; hands unaffected")
        self.close()

    # -- per frame (worker thread, after the HandFrame publish) -------------------
    def step(self, stamp_ns, actor_id, img, box, lms, busy: bool) -> dict | None:
        """One frame; the message sent, or None. Never raises."""
        if not self.enabled:
            return None
        try:
            return self._step(int(stamp_ns), actor_id, img, box, lms, busy)
        except Exception as e:  # noqa: BLE001
            self._failed(e)
            return None
        finally:
            with contextlib.suppress(Exception):  # a log line must not reach the hands
                self._maybe_log()

    def _step(self, stamp_ns, actor_id, img, box, lms, busy):
        if busy:
            self._skips["busy"] += 1
            return None
        if actor_id is None:
            self._skips["actor"] += 1
            return None
        t0 = time.perf_counter()
        plan = gc.plan_crop(lms, box, self.crop, img.shape[:2])
        if not plan.run:
            self._skips[plan.reason] += 1
            if plan.reason in ("far", "small"):
                self._last.update(box_h=plan.box_h, dist=plan.dist_ratio, palm=plan.palm, side=plan.side)
            return None
        if not self.rate.allow(plan.mode, stamp_ns):
            self._skips["rate"] += 1
            return None
        self.rate.mark(plan.mode, stamp_ns)
        square = gc.extract_square(img, plan.x0, plan.y0, plan.side)
        x = go.preprocess(square)
        t1 = time.perf_counter()
        y = self.om.infer(x)
        npu_ms = (time.perf_counter() - t1) * 1000
        cls, score, _ = go.best_gesture(go.post_fast(y, conf=self.cfg.conf), self.cfg.conf)
        total_ms = (time.perf_counter() - t0) * 1000
        msg = payload(stamp_ns, actor_id, cls, score, plan.mode, plan.roi, total_ms)
        self._publish(json.dumps(msg, separators=(",", ":")))
        self._fails = 0
        self._runs[plan.mode] += 1
        self._last = dict(
            box_h=plan.box_h, dist=plan.dist_ratio, palm=plan.palm, side=plan.side,
            cls=cls, score=score, npu_ms=npu_ms, total_ms=total_ms,
        )  # fmt: skip
        if total_ms > self.cfg.slow_ms:
            self._slows += 1
            if self._slows >= self.cfg.slow_limit:
                self._disable("slow", f"{self._slows} runs in a row over {self.cfg.slow_ms:g} ms (last {total_ms:.1f})")
        else:
            self._slows = 0
        return msg

    def _failed(self, e: Exception) -> None:
        self._fails += 1
        self._skips["error"] += 1
        now = self._clock()
        if now - self._last_fail_log >= max(self.cfg.log_period_s, 1.0):
            self._last_fail_log = now
            self._log(f"gesture result dropped: {type(e).__name__}: {e} ({self._fails} in a row)")
        if self._fails >= self.cfg.fail_limit:
            self._disable("errors", f"{self._fails} failed runs in a row ({type(e).__name__})")

    def _maybe_log(self) -> None:
        period = self.cfg.log_period_s
        now = self._clock()
        if period <= 0 or now - self._period_t0 < period or not (self._runs or self._skips):
            return
        last = self._last

        def num(key, fmt):
            v = last.get(key)
            return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)

        runs = " ".join(f"{k}={v}" for k, v in sorted(self._runs.items())) or "0"
        skips = " ".join(f"{k}={v}" for k, v in sorted(self._skips.items())) or "0"
        self._log(
            f"gesture {now - self._period_t0:.1f}s runs {runs} skips {skips} | last box_h {num('box_h', '.0f')} "
            f"dist/h {num('dist', '.2f')} palm {num('palm', '.0f')} side {num('side', 'd')} "
            f"cls {last.get('cls', '-')} score {num('score', '.2f')} npu_ms {num('npu_ms', '.1f')} "
            f"total_ms {num('total_ms', '.1f')}"
        )
        self._period_t0 = now
        self._runs.clear()
        self._skips.clear()
