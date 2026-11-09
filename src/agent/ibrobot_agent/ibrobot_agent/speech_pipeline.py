"""ROS-free, bounded single-synthesizer/single-player speech scheduling.

Explicit higher-priority speech preempts claimed lower-priority playback. ``then`` entries partition latest-ready
selection; terminals are protected too. Callbacks run outside the state lock.
``close`` linearizes against completion callbacks: a callback already admitted
finishes before closing starts, otherwise it is suppressed. Consequently a
completion callback must not wait for a concurrent close (nor call close itself).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class _Entry:
    sequence: int
    action: Any
    state: str = "pending"
    audio: Any = None
    valid: bool = True

    @property
    def priority(self) -> int:
        return getattr(self.action, "priority", None) or 0

    @property
    def barrier(self) -> bool:
        return bool(self.action.then)

    @property
    def protected(self) -> bool:
        return (
            self.barrier
            or self.priority >= 20
            or self.action.terminal
            or self.action.kind == "terminal"
            or getattr(self.action, "important", False)
        )


class SpeechPipeline:
    """Inject whole-Speak synthesis/playback, not individual audio segments.

    ``completed(action)`` is called once for every successful, still-valid play,
    including actions without ``then``. Adapters decide how to dispatch ``then``.
    Capacity includes claimed playback and synthesis (even invalidated synthesis
    until it returns). Rejected submissions have no queue-cleaning side effects.
    External callbacks should return promptly; close reports stuck workers.
    """

    def __init__(
        self,
        *,
        synthesize: Callable[[Any], Any],
        play: Callable[[Any, Any], bool],
        is_current: Callable[[Any], bool] = lambda action: True,
        completed: Callable[[Any], None] = lambda action: None,
        log: Callable[[str], None] = lambda message: None,
        capacity: int = 32,
        stop: Callable[[], None] = lambda: None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._synthesize = synthesize
        self._play = play
        self._is_current = is_current
        self._completed = completed
        self._log = log
        self._capacity = capacity
        self._stop = stop
        self._seen: OrderedDict[tuple, None] = OrderedDict()
        self._condition = threading.Condition()
        self._completion_gate = threading.Lock()
        self._cancel_gate = threading.RLock()
        self._canceling = False
        self._entries: list[_Entry] = []
        self._synthesizing: _Entry | None = None
        self._playing: _Entry | None = None
        self._closed = False
        self._sequence = 0
        self._workers = [
            threading.Thread(target=self._synth_loop, name="speech-synthesis", daemon=True),
            threading.Thread(target=self._play_loop, name="speech-playback", daemon=True),
        ]
        for worker in self._workers:
            worker.start()

    def _emit(self, event: str, entry: _Entry | None = None) -> None:
        identity = "" if entry is None else f" seq={entry.sequence} execution_id={entry.action.execution_id!r}"
        # A broken logger must not kill either consumer.
        with suppress(Exception):
            self._log(f"speech t={time.monotonic():.6f}{identity} {event}")

    def _drop_locked(self, entry: _Entry) -> None:
        entry.valid = False
        entry.audio = None
        if entry in self._entries:
            self._entries.remove(entry)
        self._condition.notify_all()

    def submit(self, action: Any) -> bool:
        if getattr(action, "priority", None) is None:
            return self._submit(action)
        if not self._is_current(action):
            return False
        # Keep playback gated from priority invalidation until admission completes.
        with self._cancel_gate:
            if not self._is_current(action):
                return False
            key = (
                getattr(action, "request_id", ""),
                action.execution_id,
                getattr(action, "phase", ""),
                getattr(action, "event_key", ""),
            )
            with self._condition:
                if key[-1] and key in self._seen:
                    return False
                self._canceling = True
            try:
                return self._submit(action)
            finally:
                with self._condition:
                    self._canceling = False
                    self._condition.notify_all()

    def _submit(self, action: Any) -> bool:
        if getattr(action, "priority", None) is not None:
            self.preempt(action, _validated=True)
        dropped: list[tuple[_Entry, str]] = []
        with self._condition:
            key = (
                getattr(action, "request_id", ""),
                action.execution_id,
                getattr(action, "phase", ""),
                getattr(action, "event_key", ""),
            )
            if key[-1] and key in self._seen:
                return False
            self._sequence += 1
            entry = _Entry(self._sequence, action)
            if self._closed:
                reason = "closed"
            else:
                victims: list[tuple[_Entry, str]] = []
                for old in self._entries:
                    if old.state == "playing" or old.barrier:
                        continue
                    if (
                        action.kind == "terminal"
                        or action.terminal
                        and action.execution_id
                        and (old.action.execution_id == action.execution_id and not old.protected)
                    ):
                        if getattr(action, "priority", None) is None or (
                            getattr(action, "request_id", "") == getattr(old.action, "request_id", "")
                            and old.priority < (getattr(action, "priority", None) or 0)
                        ):
                            victims.append((old, "terminal"))
                    elif old.state == "pending" and not old.protected:
                        kinds = ("busy", "receipt") if action.kind in ("busy", "receipt") else ("busy",)
                        if old.action.kind in kinds:
                            victims.append((old, "replaced"))
                # Invalidated synthesis still occupies a worker/retains its action.
                retained = [old for old in self._entries if all(old is not victim for victim, _ in victims)]
                occupancy = len(retained)
                if self._synthesizing is not None and self._synthesizing not in retained:
                    occupancy += 1
                while occupancy >= self._capacity:
                    victim = next(
                        (old for old in retained if not old.protected and old.state not in ("playing", "synthesizing")),
                        None,
                    )
                    if victim is None:
                        break
                    retained.remove(victim)
                    victims.append((victim, "capacity"))
                    occupancy -= 1
                if occupancy >= self._capacity:
                    reason = "capacity rejected (protected or in-flight entries)"
                else:
                    reason = ""
                    for old, cause in victims:
                        self._drop_locked(old)
                        dropped.append((old, cause))
                    if key[-1]:
                        self._seen[key] = None
                        while len(self._seen) > 2048:
                            self._seen.popitem(last=False)
                    if action.kind == "terminal":
                        self._entries.insert(0, entry)
                    else:
                        self._entries.append(entry)
                    self._condition.notify_all()
        for old, cause in dropped:
            self._emit(f"drop reason={cause}", old)
        self._emit(f"drop reason={reason}" if reason else "submitted", entry)
        return not reason

    def _current(self, entry: _Entry) -> bool:
        # Never acquire an adapter/core lock while holding the scheduling lock.
        with self._condition:
            if self._closed or not entry.valid:
                return False
        try:
            current = self._is_current(entry.action)
        except Exception as error:
            self._emit(f"validity failed: {error!r}", entry)
            current = False
        with self._condition:
            if self._closed or not entry.valid:
                return False
            if current:
                return True
            self._drop_locked(entry)
        self._emit("drop reason=expired", entry)
        return False

    def _synth_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or any(entry.state == "pending" for entry in self._entries)
                )
                if self._closed:
                    return
                entry = max(
                    (entry for entry in self._entries if entry.state == "pending"),
                    key=lambda e: (e.priority, -e.sequence),
                )
                entry.state = "synthesizing"
                self._synthesizing = entry
            audio = None
            try:
                if self._current(entry):
                    self._emit("synthesis start", entry)
                    audio = self._synthesize(entry.action)
                    self._emit("synthesis returned", entry)
                    if not self._current(entry):
                        audio = None
            except Exception as error:
                self._emit(f"synthesis failed: {error!r}", entry)
            dropped = []
            with self._condition:
                self._synthesizing = None
                if audio is None or self._closed or not entry.valid:
                    self._drop_locked(entry)
                else:
                    entry.audio = audio
                    entry.state = "ready"
                    # Collapse only ordinary ready items in this barrier partition.
                    index = self._entries.index(entry)
                    start = index
                    while start and not self._entries[start - 1].protected:
                        start -= 1
                    end = index + 1
                    while end < len(self._entries) and not self._entries[end].protected:
                        end += 1
                    if not entry.protected:
                        ready = [
                            old
                            for old in self._entries[start:end]
                            if old.state == "ready" and not old.protected and old.priority == entry.priority
                        ]
                        newest = max(ready, key=lambda old: old.sequence)
                        for old in ready:
                            if old is not newest:
                                self._drop_locked(old)
                                dropped.append(old)
                self._condition.notify_all()
            for old in dropped:
                self._emit("drop reason=replaced", old)

    def _candidate_locked(self) -> _Entry | None:
        if self._canceling:
            return None
        explicit = [
            e for e in self._entries if getattr(e.action, "priority", None) is not None and e.state != "playing"
        ]
        if explicit:
            top = max(explicit, key=lambda e: (e.priority, -e.sequence))
            if top.state == "ready":
                return top
            if top.priority >= 20:
                return None
        for entry in self._entries:
            if entry.state == "ready":
                return entry
            if entry.protected:
                return None
        return None

    def _play_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._candidate_locked() is not None)
                if self._closed:
                    return
                entry = self._candidate_locked()
            if not self._current(entry):
                continue
            with self._condition:
                if self._closed or not entry.valid or self._candidate_locked() is not entry:
                    continue
                entry.state = "playing"  # Atomic claim: later terminals cannot preempt it.
                self._playing = entry
                audio = entry.audio
            success = False
            try:
                self._emit("playback claimed", entry)
                success = self._play(entry.action, audio)
                self._emit(f"playback returned success={bool(success)}", entry)
                if success and self._current(entry):
                    with self._completion_gate:
                        with self._condition:
                            admitted = not self._closed and entry.valid
                        if admitted:
                            self._completed(entry.action)
            except Exception as error:
                self._emit(f"playback/completion failed: {error!r}", entry)
            finally:
                with self._condition:
                    self._playing = None
                    self._drop_locked(entry)

    def preempt(self, action: Any, timeout: float = 5.0, *, _validated: bool = False) -> None:
        """Stop lower-priority audio before admitting a higher-priority action.

        The cancellation gate spans every stop and old playback completion: no
        stale stop can hit a newly claimed audio. Equal priorities never interrupt.
        Call on the transport worker, never an execution or ROS callback.
        """
        priority = getattr(action, "priority", None)
        if priority is None:
            return
        if not _validated and not self._is_current(action):
            return
        deadline = time.monotonic() + timeout
        with self._cancel_gate:
            if not _validated and not self._is_current(action):
                return
            with self._completion_gate, self._condition:
                was_canceling = self._canceling
                old_play = self._playing
                interrupt = old_play is not None and old_play.priority < priority

                def same_request(entry):
                    return getattr(entry.action, "request_id", "") == getattr(action, "request_id", "")

                victims = [e for e in self._entries if e.priority < priority and (priority >= 30 or same_request(e))]
                interrupt = interrupt and (priority >= 30 or same_request(old_play))
                if interrupt:
                    self._canceling = True
                for entry in victims:
                    self._drop_locked(entry)
            try:
                for entry in victims:
                    self._emit("drop reason=priority", entry)
                while interrupt:
                    self._stop()
                    with self._condition:
                        if self._playing is not old_play:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("priority cancellation timed out")
                        self._condition.wait(timeout=min(0.05, remaining))
            finally:
                with self._condition:
                    self._canceling = was_canceling
                    self._condition.notify_all()

    def cancel(self, stop: Callable[[], None] = lambda: None, timeout: float = 5.0) -> None:
        """Invalidate admitted speech and interrupt playback, keeping both workers alive.

        Submissions after invalidation survive. New playback waits until stop has
        returned and the old play callback has exited. Stop is retried while an
        old callback is active, covering a play request sent just after the first
        stop. Synthesis is invalidated, not forcibly interrupted. Stop must return
        promptly; a stuck play callback raises TimeoutError. Do not call from a
        speech worker or completion callback.
        """
        if threading.current_thread() in self._workers:
            raise RuntimeError("cancel must be called outside speech workers")
        deadline = time.monotonic() + max(0, timeout)
        with self._cancel_gate:
            with self._completion_gate, self._condition:
                self._canceling = True
                old_play = self._playing
                dropped = list(self._entries)
                for entry in dropped:
                    self._drop_locked(entry)
            try:
                for entry in dropped:
                    self._emit("drop reason=canceled", entry)
                while True:
                    stop()
                    with self._condition:
                        if old_play is None or self._playing is not old_play:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("speech cancellation timed out: speech-playback")
                        self._condition.wait_for(lambda: self._playing is not old_play, timeout=min(0.05, remaining))
            finally:
                with self._condition:
                    self._canceling = False
                    self._condition.notify_all()

    def close(self, stop: Callable[[], None] = lambda: None, timeout: float = 5) -> None:
        """Stop admission, unblock adapter waits, join workers or raise TimeoutError.

        Repeated calls retry joins and invoke stop again. A callback already
        running is allowed to finish before close linearizes. Timeout covers
        completion-gate acquisition and both worker joins; stop must not block.
        """
        if threading.current_thread() in self._workers:
            raise RuntimeError("close must be called outside speech workers")
        deadline = time.monotonic() + max(0, timeout)
        acquired = self._completion_gate.acquire(timeout=max(0, deadline - time.monotonic()))
        with self._condition:
            self._closed = True
            dropped = list(self._entries)
            for entry in dropped:
                self._drop_locked(entry)
            self._condition.notify_all()
        if acquired:
            self._completion_gate.release()
        for entry in dropped:
            self._emit("drop reason=closed", entry)
        stop_error = None
        try:
            stop()
        except Exception as error:
            stop_error = error
        for worker in self._workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        alive = [worker.name for worker in self._workers if worker.is_alive()]
        if alive or not acquired:
            raise TimeoutError(f"speech shutdown timed out: {alive or ['completion callback']}")
        if stop_error is not None:
            raise stop_error
