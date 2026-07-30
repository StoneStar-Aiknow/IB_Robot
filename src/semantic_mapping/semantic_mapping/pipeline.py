"""Bounded mapping work queue and serialized fusion commit helpers."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class QueuedFrame(Generic[T]):
    sequence: int
    payload: T
    enqueued_ns: int


@dataclass
class PipelineDiagnostics:
    enqueued: int = 0
    dequeued: int = 0
    dropped_oldest: int = 0
    dropped_newest: int = 0
    backpressure_waits: int = 0
    committed: int = 0
    commit_failures: int = 0
    queue_wait_ns: list[int] = field(default_factory=list)
    commit_time_ns: list[int] = field(default_factory=list)


class BoundedFrameQueue(Generic[T]):
    """Thread-safe FIFO with explicit live-drop or offline-backpressure policy."""

    POLICIES = {"drop_oldest", "drop_newest", "backpressure"}

    def __init__(self, capacity: int, policy: str = "drop_oldest"):
        if capacity <= 0:
            raise ValueError("frame queue capacity must be positive")
        if policy not in self.POLICIES:
            raise ValueError(f"queue policy must be one of {sorted(self.POLICIES)}")
        self.capacity = capacity
        self.policy = policy
        self.diagnostics = PipelineDiagnostics()
        self._queue: deque[QueuedFrame[T]] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._sequence = 0

    def put(self, payload: T, timeout: float | None = None) -> bool:
        with self._condition:
            if self._closed:
                raise RuntimeError("frame queue is closed")
            if len(self._queue) >= self.capacity:
                if self.policy == "drop_oldest":
                    self._queue.popleft()
                    self.diagnostics.dropped_oldest += 1
                elif self.policy == "drop_newest":
                    self.diagnostics.dropped_newest += 1
                    return False
                else:
                    self.diagnostics.backpressure_waits += 1
                    deadline = None if timeout is None else time.monotonic() + timeout
                    while len(self._queue) >= self.capacity and not self._closed:
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0.0:
                            return False
                        self._condition.wait(remaining)
                    if self._closed:
                        raise RuntimeError("frame queue is closed")
            frame = QueuedFrame(self._sequence, payload, time.monotonic_ns())
            self._sequence += 1
            self._queue.append(frame)
            self.diagnostics.enqueued += 1
            self._condition.notify_all()
            return True

    def get(self, timeout: float | None = None) -> QueuedFrame[T] | None:
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._queue and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if not self._queue:
                return None
            frame = self._queue.popleft()
            self.diagnostics.dequeued += 1
            self.diagnostics.queue_wait_ns.append(max(0, time.monotonic_ns() - frame.enqueued_ns))
            self._condition.notify_all()
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._queue)


class SerializedCommitter:
    """Keep tracker/database mutation single-threaded without locking inference."""

    def __init__(self, diagnostics: PipelineDiagnostics | None = None):
        self.diagnostics = diagnostics or PipelineDiagnostics()
        self._lock = threading.Lock()

    def commit(self, callback, *args, **kwargs):
        started = time.monotonic_ns()
        with self._lock:
            try:
                result = callback(*args, **kwargs)
            except Exception:
                self.diagnostics.commit_failures += 1
                raise
            finally:
                self.diagnostics.commit_time_ns.append(max(0, time.monotonic_ns() - started))
        self.diagnostics.committed += 1
        return result
