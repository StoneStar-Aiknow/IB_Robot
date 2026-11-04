"""Subscriptions read when a caller needs them, not by an executor.

An executor wakes for every sample of every subscription it holds and runs a
Python callback for it. Where only the newest sample (or the last few) is ever
read, that work is spent on samples nobody looks at: the X2 runtime took every
~500 Hz joint state of the platform to publish 50 Hz views, and the idle grasp
pipeline held its GIL ~30% of the time on runtime views, /tf and camera frames
(2026-10-22), so the samples a request needed arrived late.

A subscription here is never added to an executor. Its samples wait in the
reader queue (KEEP_LAST, as deep as its reader needs history) until
:meth:`OnDemandSubscriptions.drain` takes them, and each keeps the time it was
received, not the time it was taken. A sample whose receive time the middleware
did not record is delivered with ``None`` instead of a time: a freshness gate
must treat it as not fresh, and drain never pretends it just arrived.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from copy import copy
from typing import Any


def received_monotonic(info: dict, *, now_monotonic: float | None = None, now_ns: int | None = None) -> float | None:
    """``time.monotonic()`` when the middleware received a sample, or ``None`` when it did not record one.

    The sample may have waited in the reader queue for seconds before the drain,
    so an absent receive time must not become "received now": that would pass
    every freshness gate on data of unknown age. Callers that gate on freshness
    refuse ``None``.
    """
    received_ns = int(info.get("received_timestamp") or 0)
    if received_ns <= 0:
        return None
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    wall_ns = time.time_ns() if now_ns is None else now_ns
    return monotonic - max(0, wall_ns - received_ns) * 1e-9


class OnDemandSubscriptions:
    """Subscriptions on a node of their own that no executor spins; :meth:`drain` takes what has queued."""

    def __init__(self, context, name: str):
        import rclpy

        # The process's global arguments still apply, so a launch's namespace and
        # topic remaps reach these readers as they reach any node. Only the name is
        # pinned: a launch's -r __node:= renames every node of the process, and this
        # one would then share its owner's name. A node's own remap rules take
        # precedence over the global ones.
        self._node = rclpy.create_node(
            name,
            context=context,
            cli_args=["--ros-args", "-r", f"__node:={name}"],
            enable_rosout=False,
            start_parameter_services=False,
        )
        self._lock = threading.Lock()
        self._entries: list[tuple[Any, Callable[[Any, float | None], None], int]] = []

    def subscribe(
        self,
        msg_type,
        topic: str,
        qos,
        depth: int,
        on_sample: Callable[[Any, float | None], None],
        *,
        raw: bool = False,
    ) -> None:
        """``on_sample(message, received_monotonic_s)`` runs in :meth:`drain`.

        ``received_monotonic_s`` is ``None`` for a sample the middleware did not
        time; a freshness gate treats that as not fresh.

        ``depth`` is both the reader's KEEP_LAST history and the most samples one
        :meth:`drain` takes from this subscription, so a caller sizes it by the
        history its reader needs and gets a bounded cost per call either way.

        ``qos`` None subscribes with the ROS defaults (reliable, volatile).
        """
        from rclpy.qos import HistoryPolicy, QoSProfile

        profile = QoSProfile(depth=int(depth)) if qos is None else copy(qos)
        profile.history, profile.depth = HistoryPolicy.KEEP_LAST, int(depth)
        subscription = self._node.create_subscription(msg_type, topic, lambda _message: None, profile, raw=raw)
        with self._lock:
            self._entries.append((subscription, on_sample, int(depth)))

    def drain(self) -> int:
        """Take up to each subscription's ``depth``, pass them on oldest first; returns how many.

        Bounded on purpose. Taking until a reader reports empty makes one call cost
        whatever has arrived, so on a host that cannot convert samples faster than the
        publishers produce them the loop only returns when it momentarily wins the race:
        measured on the X2's 310P, one unbounded drain of ~1000 samples/s of vendor joint
        state took 1.9 s, and the 50 Hz timer that calls it fired at 0.5 Hz while every
        consumer downstream saw 1-2 s old data. A bound costs at most ``depth`` samples
        per subscription per call, and what does not fit is dropped by KEEP_LAST, which
        is what that history means.
        """
        with self._lock:
            taken = []
            for subscription, on_sample, depth in self._entries:
                for _ in range(max(depth, 1)):
                    with subscription.handle:
                        sample = subscription.handle.take_message(subscription.msg_type, subscription.raw)
                    if sample is None:
                        break
                    message, info = sample
                    taken.append((received_monotonic(info), on_sample, message))
            # A sample without a receive time cannot be proven newer than
            # anything, so it goes first and a known-time sample overwrites it
            # in newest-wins state.
            taken.sort(key=lambda item: (item[0] is not None, item[0] if item[0] is not None else 0.0))
            for received, on_sample, message in taken:
                on_sample(message, received)
        return len(taken)

    def close(self) -> None:
        with self._lock:
            self._entries = []
            self._node.destroy_node()
