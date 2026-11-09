"""Vendor pub/sub channel wrappers with cached state and explicit close.

These are transport-level wrappers over the vendor ``ChannelPublisher`` /
``ChannelSubscriber``: they add last-sample caching with a monotonic stamp
(state feeds), dict-or-IDL command acceptance (command writers) and
lifecycle/refuse-after-close checks. Topic names, rates and QoS semantics are
runtime concerns (PR 2) — the vendor channel factory applies its own QoS.

Threading (verified vendor behavior): the SDK invokes a subscriber handler on
its DDS reader thread (or its queue thread when ``queue_len > 0``), never on
the caller's thread. Handlers registered here must therefore only update
in-memory state under the feed's lock, like the caching done below.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from unitree_g1_adapter import vendor as vendor_module


class StateFeed:
    """A vendor subscriber that caches the newest sample behind a lock.

    ``handler``, when given, runs after the cache update on the same DDS
    callback thread; it must not block. ``now`` is injectable for tests.
    """

    def __init__(
        self,
        topic: str,
        type_key: str,
        *,
        vendor: Mapping[str, Any] | None = None,
        handler: Callable[[Any], None] | None = None,
        queue_len: int = 0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._vendor = vendor if vendor is not None else vendor_module.default_vendor()
        self.topic = str(topic)
        self.type_key = str(type_key)
        self._now = now
        self._lock = threading.Lock()
        self._last_sample: Any = None
        self._last_stamp = 0.0
        self._closed = False
        self._user_handler = handler
        self._subscriber = self._vendor["channel_subscriber"](self.topic, self.type_key)
        # Init starts the reader; the SDK offers no way to swap handlers later.
        self._subscriber.Init(self._on_sample, int(queue_len))

    def _on_sample(self, sample: Any) -> None:
        with self._lock:
            self._last_sample = sample
            self._last_stamp = self._now()
        if self._user_handler is not None:
            self._user_handler(sample)

    @property
    def last_sample(self) -> Any:
        """The newest vendor sample, or ``None`` before the first delivery."""
        with self._lock:
            return self._last_sample

    @property
    def last_stamp(self) -> float:
        """Monotonic time of the newest sample; ``0.0`` before the first."""
        with self._lock:
            return self._last_stamp

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._subscriber.Close()


class CommandWriter:
    """A vendor publisher that accepts plain dicts or ready IDL samples.

    The vendor ``Write(sample, timeout)`` returns ``False`` when the write
    failed or the optional publication-matched wait elapsed; that boolean is
    passed through so callers can count failures instead of assuming success.
    """

    def __init__(
        self,
        topic: str,
        type_key: str,
        *,
        vendor: Mapping[str, Any] | None = None,
    ) -> None:
        self._vendor = vendor if vendor is not None else vendor_module.default_vendor()
        self.topic = str(topic)
        self.type_key = str(type_key)
        self._closed = False
        self._publisher = self._vendor["channel_publisher"](self.topic, self.type_key)
        self._publisher.Init()

    def write(self, sample: Any, *, timeout: float | None = None) -> bool:
        """Write one command frame; ``False`` means the vendor refused it.

        For the command types (``low_cmd``/``hand_cmd``) a ``list`` is the
        per-slot plain-dict form and is built through
        :mod:`unitree_g1_adapter.vendor`; a ``LowCmdFrame`` mapping is the same
        slot list plus the frame-level ``mode_pr``/``mode_machine``/``crc``
        the vendor LowCmd header carries. Anything else must already be a
        vendor IDL sample. A bare mapping is refused — the converter takes the
        slot list itself, and accepting a dict would silently build a frame
        from its keys.
        """
        if self._closed:
            raise RuntimeError(f"command writer {self.topic!r} is closed")
        message = self._build(sample)
        return bool(self._publisher.Write(message, timeout))

    def _build(self, sample: Any) -> Any:
        if self.type_key in ("low_cmd", "hand_cmd"):
            if isinstance(sample, list):
                builder = self._vendor["make_lowcmd" if self.type_key == "low_cmd" else "make_handcmd"]
                return builder(sample)
            if isinstance(sample, Mapping):
                if self.type_key != "low_cmd" or "motor_cmd" not in sample:
                    raise TypeError(
                        f"{self.type_key} frames take the slot list, not a mapping; pass the "
                        "per-slot dicts as a list, a vendor IDL sample, or (low_cmd only) a "
                        "frame mapping with a 'motor_cmd' slot list"
                    )
                return self._vendor["make_lowcmd"](
                    sample["motor_cmd"],
                    mode_pr=sample.get("mode_pr", 0),
                    mode_machine=sample.get("mode_machine", 0),
                    crc=sample.get("crc"),
                )
            return sample
        if isinstance(sample, list | Mapping):
            raise TypeError(f"plain-data frames are not supported for type {self.type_key!r}; pass a vendor IDL sample")
        return sample

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.Close()
