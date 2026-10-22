"""The live retarget node's engine, kept free of rclpy so it can be tested without ROS.

The subscription hands frames to ``on_frame``; a worker thread runs
``process_one`` so the slow ``stream.retarget`` never blocks a callback; the
50 Hz timer calls ``tick`` and publishes whatever it returns. Every cheap stream
call and every counter sits under one lock; ``retarget`` runs outside it.
"""

from __future__ import annotations

import collections
import math
import threading
import time

import numpy as np

from .x2.safety import RateLimiter

_YOLOX = "yolox"
_LATENCY_WINDOW = 200


class LivePlayer:
    """Drives a RetargetStream for the live node: frame gate, worker queue, play/pause, rate cap."""

    def __init__(self, stream, *, low_rad, high_rad, rated_rad_s, vel_scale, rate_hz, staleness_s, base_pose,
                 queue_size=4, on_mapped=None):
        self._stream = stream
        # Called on the worker with (t_camera, t_received, pose) for every mapped frame, outside the lock.
        self._on_mapped = on_mapped
        self._low = np.asarray(low_rad, dtype=np.float64)
        self._high = np.asarray(high_rad, dtype=np.float64)
        self._limiter = RateLimiter(self._low, self._high, rated_rad_s, vel_scale, 1.0 / float(rate_hz))
        self._staleness_s = float(staleness_s)
        self._base_pose = np.asarray(base_pose, dtype=np.float64)

        self._lock = threading.Lock()
        self._queue = collections.deque(maxlen=int(queue_size))
        self._queue_cv = threading.Condition()
        self._closing = False
        self._thread: threading.Thread | None = None

        self._last_yolox_at: float | None = None
        self._playing = False
        self._play_requests = 0
        self._input_frames = 0
        self._yolox_frames = 0
        self._dropped = 0
        self._processed = 0
        self._retarget_errors = 0
        self._last_error: str | None = None
        self._retarget_ms = collections.deque(maxlen=_LATENCY_WINDOW)
        self._published = 0
        self._published_this_round = False
        self._reentries = 0
        self._held_stale = 0
        self._last_command: list[float] | None = None

    # -- input (subscription callback) -----------------------------------------
    def on_frame(self, now, t_camera, frame_id, source, smplx_pose_raw) -> None:
        with self._lock:
            self._input_frames += 1
            if source != _YOLOX:
                # center_fallback means nobody is in the picture: not a sign of life.
                return
            self._yolox_frames += 1
            reset_first = self._last_yolox_at is None or now - self._last_yolox_at > self._staleness_s
            self._last_yolox_at = now
            with self._queue_cv:
                evicted = self._queue[0] if len(self._queue) == self._queue.maxlen else None
                self._queue.append((reset_first, t_camera, frame_id, smplx_pose_raw, now))
                if evicted is not None:
                    self._dropped += 1
                    if evicted[0]:
                        # The frame that opened the gap was dropped; its successor carries the reset.
                        self._queue[0] = (True,) + self._queue[0][1:]
                self._queue_cv.notify()

    # -- worker ------------------------------------------------------------------
    def process_one(self, timeout=None) -> bool:
        with self._queue_cv:
            if not self._queue:
                self._queue_cv.wait(timeout)
            if not self._queue:
                return False
            reset_first, t_camera, frame_id, raw, t_received = self._queue.popleft()
        try:
            if reset_first:
                with self._lock:
                    self._stream.reset_input()
                    if self._playing and self._published_this_round:
                        # The person is back after a gap: hold the last command, then
                        # blend into them as a round does, instead of a full-speed chase.
                        self._stream.start_round(from_pose=self._last_command)
                        self._reentries += 1
            started = time.perf_counter()
            pose = self._stream.retarget(t_camera, frame_id, raw)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            with self._lock:
                self._stream.push_pose(t_camera, pose)
                self._retarget_ms.append(elapsed_ms)
                self._processed += 1
            if self._on_mapped is not None:
                self._on_mapped(t_camera, t_received, pose)
        except Exception as exc:  # noqa: BLE001 - a bad frame must never kill the worker
            with self._lock:
                self._retarget_errors += 1
                self._last_error = repr(exc)[:200]
        return True

    def _run(self) -> None:
        while not self._closing:
            self.process_one(timeout=0.1)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live_player_worker", daemon=True)
        self._thread.start()

    def close(self) -> None:
        with self._queue_cv:
            self._closing = True
            self._queue_cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock, self._queue_cv:
            # Frames never processed count as dropped, so input always balances.
            self._dropped += len(self._queue)
            self._queue.clear()
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    # -- output (service and timers) -----------------------------------------------
    def play(self, on: bool) -> None:
        with self._lock:
            if on:
                self._stream.start_round()
                self._limiter.reset(self._base_pose)
                self._published_this_round = False
                self._playing = True
                self._play_requests += 1
            else:
                self._playing = False

    def tick(self, now) -> list[float] | None:
        with self._lock:
            if not self._playing:
                return None
            poses = self._stream.poll(now)
            if not poses:
                return None
            if self._last_yolox_at is None or now - self._last_yolox_at > self._staleness_s:
                # The poll above keeps the output clock moving; the limiter stays on the
                # last published pose so the next command is capped against what was sent.
                self._held_stale += 1
                return None
            # One capped step per published command: when the poll catches up on several
            # samples, stepping through each would multiply the speed cap.
            out = self._limiter.step(poses[-1])
            command = out.tolist()
            self._published += 1
            self._published_this_round = True
            self._last_command = command
            return command

    def status(self, now) -> dict:
        with self._lock:
            latencies = sorted(self._retarget_ms)
            status = dict(self._stream.status())
            status.update({
                "playing": self._playing,
                "play_requests": self._play_requests,
                "input_frames": self._input_frames,
                "yolox_frames": self._yolox_frames,
                "input_age_sec": round(now - self._last_yolox_at, 2) if self._last_yolox_at is not None else None,
                "dropped": self._dropped,
                "processed": self._processed,
                "retarget_errors": self._retarget_errors,
                "last_error": self._last_error,
                "retarget_ms_p50": round(latencies[len(latencies) // 2], 1) if latencies else None,
                "retarget_ms_max": round(latencies[-1], 1) if latencies else None,
                "published": self._published,
                "held_stale": self._held_stale,
                "reentries": self._reentries,
                "cap_trips": int(self._limiter.cap_trips.sum()),
                "limit_trips": int(self._limiter.limit_trips.sum()),
                "limits_deg": [[round(math.degrees(lo), 1), round(math.degrees(hi), 1)]
                               for lo, hi in zip(self._low, self._high)],
                "last_command": [round(v, 4) for v in self._last_command]
                if self._last_command is not None
                else None,
            })
        return status
