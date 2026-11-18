"""The two-hand gesture model on its own thread, fed from the hand worker's frames. **No ROS.**

``hand_perception_node`` owns one ``GestureRunner`` when ``gesture_enabled`` is
set and the model path is not empty. The node's worker thread calls ``start``
once, right after the hand backend is up; that only starts the gesture thread,
which loads the om (ACL contexts are per thread: the session is built, used and
freed there only). Then, once per frame *after* that frame's HandFrame is
published, the worker calls ``post`` with the frame it already decoded, its
person box and the 21 image points of each hand found.

``post`` (hand worker) never waits and never raises: nothing until the om is
loaded; skip if the hand slot already holds a newer frame (the hands come
first); ``gesture_crop.plan_crop`` picks the square or a skip; ``RateGate``
keeps the runs under the rates; then it copies the square out of the frame and
drops it in a one-slot mailbox (latest only: a job not yet taken is replaced).
The gesture thread takes it, resizes, runs the om and ``post_fast``, and
publishes one JSON message (``x2_hand_gesture/v1``) per run. The two threads
share one small lock for the mailbox, the state and the counters, never held
across a run.

Fail silent, by design: the hands never wait on the gesture model.

- the om does not load: off, one INFO line;
- a run raises, or the output is not a finite [1,38,2100] tensor: result
  dropped, rate-limited INFO; ``fail_limit`` in a row: off;
- ``slow_limit`` of the last ``slow_window`` runs took over ``slow_ms`` (crop,
  resize, NPU, post): off (that run's result is still sent);
- a run has not returned after ``watchdog_ms`` (``post`` compares its start
  time, it never waits on it): off; the thread is a daemon and is left behind.

Off is for the rest of the process: no more posts, nothing more published; the
node then drops the runner. No image is ever written or logged; the log line is
numbers only.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, fields

from . import gesture_crop as gc
from . import gesture_om as go

SCHEMA = "x2_hand_gesture/v1"
LOADING, ON, OFF = "loading", "on", "off"


@dataclass(frozen=True)
class RunnerConfig:
    conf: float = 0.25
    slow_ms: float = 20.0
    slow_window: int = 10  # the last this many runs ...
    slow_limit: int = 3  # ... with this many over slow_ms: off
    fail_limit: int = 5  # failed runs in a row: off
    watchdog_ms: float = 200.0  # one run not back after this: off
    log_period_s: float = 1.0  # summary line period; <= 0: no summary

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                raise ValueError(f"gesture {f.name} must be a finite number, got {v!r}")
        if not 0 < self.conf < 1:
            raise ValueError(f"gesture conf must be in (0, 1), got {self.conf!r}")
        for name in ("slow_ms", "watchdog_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"gesture {name} must be positive, got {getattr(self, name)!r}")
        for name in ("slow_window", "slow_limit", "fail_limit"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"gesture {name} must be an integer >= 1, got {v!r}")
        if self.slow_limit > self.slow_window:
            raise ValueError(f"gesture slow_limit {self.slow_limit} is more than slow_window {self.slow_window}")


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


@dataclass(frozen=True)
class Job:
    """One square for the gesture thread: a copy, never a view of the hand worker's frame."""

    stamp_ns: int
    actor_id: int
    plan: gc.CropPlan
    square: object  # (side, side, 3) uint8
    crop_ms: float


class GestureRunner:
    """``publish(text)`` sends one JSON string; ``log(text)`` writes one INFO line (both from either thread)."""

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
        self.state = LOADING
        self.disabled_reason = ""
        self.thread: threading.Thread | None = None
        self.rate = gc.RateGate(crop)
        self._lock = threading.Lock()  # mailbox, state, counters; never held across a run
        self._job: Job | None = None
        self._has_job = threading.Event()
        self._stop = threading.Event()
        self._busy_since: float | None = None  # clock() when the running job started; None when idle
        self._fails = 0
        self._slow = collections.deque(maxlen=cfg.slow_window)
        self._last_fail_log = -math.inf
        self._period_t0 = clock()
        self._runs: Counter = Counter()
        self._skips: Counter = Counter()
        self._last: dict = {}

    @property
    def enabled(self) -> bool:
        return self.state == ON

    @property
    def off(self) -> bool:
        return self.state == OFF

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        """Hand worker: start the gesture thread (it loads the om); returns at once."""
        self.thread = threading.Thread(target=self._main, name="hand-gesture", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Shutdown: no more runs; waits at most ``timeout`` for the thread to free the om."""
        with self._lock:
            if self.state != OFF:
                self.state, self.disabled_reason = OFF, "stopped"
            self._job = None
        self._stop.set()
        self._has_job.set()
        t = self.thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    def _main(self) -> None:
        try:
            if not self.load():
                return
            while not self._stop.is_set():
                job = self.take(0.5)
                if job is not None:
                    self.run(job)
        finally:
            self._close()

    def load(self) -> bool:
        """Gesture thread: open the om session; on any failure the gesture is off."""
        try:
            om = self._om_factory(self.model_path, self.device)
        except Exception as e:  # noqa: BLE001  any load failure only turns the gesture off
            self._off("load", f"load failed ({type(e).__name__}: {e})")
            return False
        self.om = om
        with self._lock:
            if self.state != LOADING:  # stopped while loading
                return False
            self.state = ON
        self._log(
            f"gesture model {getattr(om, 'path', self.model_path)} ready; conf {self.cfg.conf:g}, "
            f"rate <= {self.crop.max_rate_hz:g} Hz (one hand <= {self.crop.one_hand_max_rate_hz:g} Hz), "
            f"watchdog {self.cfg.watchdog_ms:g} ms"
        )
        return True

    def _close(self) -> None:
        om, self.om = self.om, None
        if om is not None:
            with contextlib.suppress(Exception):
                om.close()

    def _off(self, reason: str, detail: str) -> None:
        """Either thread: off for good, one log line; the gesture thread then frees the om."""
        with self._lock:
            if self.state == OFF:
                return
            self.state, self.disabled_reason = OFF, reason
            self._job = None
        self._stop.set()
        self._has_job.set()
        self._log(f"gesture model off: {detail}; hands unaffected")

    # -- per frame (hand worker, after the HandFrame publish) -----------------
    def post(self, stamp_ns, actor_id, img, box, lms, busy: bool) -> bool:
        """One frame: crop it and leave the square for the gesture thread. True if posted. Never waits or raises."""
        if self.state != ON:
            return False
        try:
            if self._hung():
                return False
            return self._post(int(stamp_ns), actor_id, img, box, lms, busy)
        except Exception as e:  # noqa: BLE001
            self._failed(e)
            return False
        finally:
            with contextlib.suppress(Exception):  # a log line must not reach the hands
                self._maybe_log()

    def _hung(self) -> bool:
        started = self._busy_since
        if started is None or self._clock() - started <= self.cfg.watchdog_ms / 1000.0:
            return False
        self._off("hung", f"a run has not returned in {self.cfg.watchdog_ms:g} ms (its thread is left behind)")
        return True

    def _skip(self, reason: str, **last) -> bool:
        with self._lock:
            self._skips[reason] += 1
            self._last.update(last)
        return False

    def _post(self, stamp_ns, actor_id, img, box, lms, busy):
        if busy:
            return self._skip("busy")
        if actor_id is None:
            return self._skip("actor")
        t0 = time.perf_counter()
        plan = gc.plan_crop(lms, box, self.crop, img.shape[:2])
        if not plan.run:
            if plan.reason in ("far", "small"):
                return self._skip(plan.reason, box_h=plan.box_h, dist=plan.dist_ratio, palm=plan.palm, side=plan.side)
            return self._skip(plan.reason)
        if not self.rate.allow(plan.mode, stamp_ns):
            return self._skip("rate")
        self.rate.mark(plan.mode, stamp_ns)
        square = gc.extract_square(img, plan.x0, plan.y0, plan.side)
        job = Job(stamp_ns, int(actor_id), plan, square, (time.perf_counter() - t0) * 1000)
        with self._lock:
            if self.state != ON:
                return False
            if self._job is not None:
                self._skips["replaced"] += 1
            self._job = job
        self._has_job.set()
        return True

    # -- gesture thread ----------------------------------------------------------
    def take(self, timeout: float) -> Job | None:
        """The newest posted job (and the mailbox empty), or None after ``timeout`` s."""
        if not self._has_job.wait(timeout):
            return None
        with self._lock:
            job, self._job = self._job, None
            self._has_job.clear()
        return job

    def run(self, job: Job) -> dict | None:
        """One job: om + post + publish; the message sent, or None. Never raises."""
        self._busy_since = self._clock()
        try:
            return self._run(job)
        except Exception as e:  # noqa: BLE001
            self._failed(e)
            return None
        finally:
            self._busy_since = None

    def _run(self, job: Job):
        plan = job.plan
        t0 = time.perf_counter()
        x = go.preprocess(job.square)
        t1 = time.perf_counter()
        y = self.om.infer(x)
        npu_ms = (time.perf_counter() - t1) * 1000
        cls, score, _ = go.best_gesture(go.post_fast(y, conf=self.cfg.conf), self.cfg.conf)
        total_ms = job.crop_ms + (time.perf_counter() - t0) * 1000
        if self.state != ON:  # turned off while this run was out (watchdog, stop): nothing more goes out
            return None
        msg = payload(job.stamp_ns, job.actor_id, cls, score, plan.mode, plan.roi, total_ms)
        self._publish(json.dumps(msg, separators=(",", ":")))
        with self._lock:
            self._fails = 0
            self._runs[plan.mode] += 1
            self._last = dict(
                box_h=plan.box_h, dist=plan.dist_ratio, palm=plan.palm, side=plan.side,
                cls=cls, score=score, npu_ms=npu_ms, total_ms=total_ms,
            )  # fmt: skip
        self._slow.append(total_ms > self.cfg.slow_ms)
        slow = sum(self._slow)
        if slow >= self.cfg.slow_limit:
            self._off(
                "slow",
                f"{slow} of the last {len(self._slow)} runs over {self.cfg.slow_ms:g} ms (last {total_ms:.1f})",
            )
        return msg

    # -- either thread -----------------------------------------------------------
    def _failed(self, e: Exception) -> None:
        now = self._clock()
        with self._lock:
            self._fails += 1
            self._skips["error"] += 1
            fails = self._fails
            log = now - self._last_fail_log >= max(self.cfg.log_period_s, 1.0)
            if log:
                self._last_fail_log = now
        if log:
            self._log(f"gesture result dropped: {type(e).__name__}: {e} ({fails} in a row)")
        if fails >= self.cfg.fail_limit:
            self._off("errors", f"{fails} failed runs in a row ({type(e).__name__})")

    def _maybe_log(self) -> None:
        period = self.cfg.log_period_s
        now = self._clock()
        if period <= 0 or now - self._period_t0 < period:
            return
        with self._lock:
            if not (self._runs or self._skips):
                return
            runs = " ".join(f"{k}={v}" for k, v in sorted(self._runs.items())) or "0"
            skips = " ".join(f"{k}={v}" for k, v in sorted(self._skips.items())) or "0"
            last = dict(self._last)
            self._runs.clear()
            self._skips.clear()
        elapsed, self._period_t0 = now - self._period_t0, now

        def num(key, fmt):
            v = last.get(key)
            return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)

        self._log(
            f"gesture {elapsed:.1f}s runs {runs} skips {skips} | last box_h {num('box_h', '.0f')} "
            f"dist/h {num('dist', '.2f')} palm {num('palm', '.0f')} side {num('side', 'd')} "
            f"cls {last.get('cls', '-')} score {num('score', '.2f')} npu_ms {num('npu_ms', '.1f')} "
            f"total_ms {num('total_ms', '.1f')}"
        )
