"""Per-tick ``SessionInputs`` from the perception messages.

The subscriptions hand messages in as they arrive; ``take`` is called once per
tick and returns what the hooks see on that tick:

* ``window``: the ``PearFrame`` messages since the previous tick, oldest
  first, at most ``window_max`` (older ones are dropped and counted);
* ``actor``: the latest ``ActorState`` as an :class:`ActorSignal`, with its age
  since receipt, so a stream that went quiet shows up as an old signal rather
  than as no signal;
* ``camera_frame``: the latest camera message, only on the first tick after it
  arrived. It is passed on undecoded; decoding is left to the detector.

Messages are read by attribute only, so nothing here imports ROS.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from manipulation_execution.mirror_session.hooks import ActorSignal, PearWindow, SessionInputs

DEFAULT_WINDOW_MAX = 64


def actor_signal(message, age_s: float) -> ActorSignal:
    """An ``ibrobot_msgs/ActorState`` as the hooks see it."""
    stamp = message.header.stamp
    actor_id = int(message.actor_id)
    return ActorSignal(
        present=bool(message.actor_present),
        actor_id=None if actor_id < 0 else actor_id,
        stamp_s=stamp.sec + stamp.nanosec * 1e-9,
        start_ready=bool(message.start_ready),
        reason=str(message.reason),
        age_s=age_s,
    )


class InputCollector:
    def __init__(self, window_max: int = DEFAULT_WINDOW_MAX, clock: Callable[[], float] = time.monotonic):
        if window_max < 1:
            raise ValueError("window_max must be at least 1")
        self._clock = clock
        self._lock = threading.Lock()
        self._window: deque = deque(maxlen=window_max)
        self._actor = None
        self._actor_at = 0.0
        self._camera = None
        self.pear_received = 0
        self.pear_dropped = 0
        self.actor_received = 0
        self.camera_received = 0

    def pear(self, message) -> None:
        with self._lock:
            if len(self._window) == self._window.maxlen:
                self.pear_dropped += 1
            self._window.append(message)
            self.pear_received += 1

    def actor(self, message) -> None:
        with self._lock:
            self._actor, self._actor_at = message, self._clock()
            self.actor_received += 1

    def camera(self, message) -> None:
        with self._lock:
            self._camera = message
            self.camera_received += 1

    def take(self) -> SessionInputs:
        with self._lock:
            frames = tuple(self._window)
            self._window.clear()
            actor, actor_at = self._actor, self._actor_at
            camera, self._camera = self._camera, None
            now = self._clock()
        signal = None if actor is None else actor_signal(actor, now - actor_at)
        return SessionInputs(window=PearWindow(frames), actor=signal, camera_frame=camera)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pear_received": self.pear_received,
                "pear_dropped": self.pear_dropped,
                "actor_received": self.actor_received,
                "camera_received": self.camera_received,
            }
