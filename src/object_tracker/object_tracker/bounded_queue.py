"""Thread-safe latest-item queue for sensor callback handoff."""

from collections import deque
from threading import Lock
from typing import Generic, TypeVar

Item = TypeVar("Item")


class LatestQueue(Generic[Item]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("queue capacity must be positive")
        self._items: deque[Item] = deque(maxlen=capacity)
        self._lock = Lock()
        self.dropped = 0

    def push(self, item: Item) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self.dropped += 1
            self._items.append(item)

    def pop_latest(self) -> Item | None:
        with self._lock:
            if not self._items:
                return None
            latest = self._items[-1]
            self._items.clear()
            return latest

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
