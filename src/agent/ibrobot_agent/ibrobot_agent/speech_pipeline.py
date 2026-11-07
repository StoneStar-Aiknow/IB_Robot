"""ROS-free, bounded single-synthesizer/single-player speech scheduling.

A claimed playback is not preempted. ``then`` entries partition latest-ready
selection; terminals are protected too. Callbacks run outside the state lock.
``close`` linearizes against completion callbacks: a callback already admitted
finishes before closing starts, otherwise it is suppressed. Consequently a
completion callback must not wait for a concurrent close (nor call close itself).
"""

from __future__ import annotations

import threading
import time
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
    def barrier(self) -> bool:
        return bool(self.action.then)

    @property
    def protected(self) -> bool:
        return (
            self.barrier
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
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._synthesize = synthesize
        self._play = play
        self._is_current = is_current
        self._completed = completed
        self._log = log
        self._capacity = capacity
        self._condition = threading.Condition()
        self._completion_gate = threading.Lock()
        self._entries: list[_Entry] = []
        self._synthesizing: _Entry | None = None
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
        dropped: list[tuple[_Entry, str]] = []
        with self._condition:
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
                entry = next(entry for entry in self._entries if entry.state == "pending")
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
                        ready = [old for old in self._entries[start:end] if old.state == "ready" and not old.protected]
                        newest = max(ready, key=lambda old: old.sequence)
                        for old in ready:
                            if old is not newest:
                                self._drop_locked(old)
                                dropped.append(old)
                self._condition.notify_all()
            for old in dropped:
                self._emit("drop reason=replaced", old)

    def _candidate_locked(self) -> _Entry | None:
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
                    self._drop_locked(entry)

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
