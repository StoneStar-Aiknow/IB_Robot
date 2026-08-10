"""Small bounded ingress pool shared by scheduled ROS action servers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

GOAL_CONTEXTS_PER_ENDPOINT = 2


class GoalSlotPool:
    """Reserve one execution context per accepted action goal."""

    def __init__(
        self,
        endpoints: Iterable[str],
        *,
        capacity: int = GOAL_CONTEXTS_PER_ENDPOINT,
        protected_capacity: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("goal slot capacity must be positive")
        if protected_capacity < 0 or protected_capacity >= capacity:
            raise ValueError("protected goal slot capacity must be in [0, capacity)")
        self._slots = {endpoint: threading.BoundedSemaphore(capacity) for endpoint in endpoints}
        general_capacity = capacity - protected_capacity
        self._general_slots = {endpoint: threading.BoundedSemaphore(general_capacity) for endpoint in endpoints}

    def try_acquire(self, endpoint: str, *, protected: bool = False) -> bool:
        """Acquire a bounded execution context.

        Unprotected work must acquire both a general slot and a total slot.
        Protected work only consumes the total pool, so unprotected requests can
        never exhaust the capacity reserved for protected traffic.
        """

        general_acquired = False
        if not protected:
            general_acquired = self._general_slots[endpoint].acquire(blocking=False)
            if not general_acquired:
                return False
        if self._slots[endpoint].acquire(blocking=False):
            return True
        if general_acquired:
            self._general_slots[endpoint].release()
        return False

    def run(
        self,
        endpoint: str,
        callback: Callable[[Any], Any],
        goal_handle: Any,
        *,
        protected: bool = False,
    ) -> Any:
        try:
            return callback(goal_handle)
        finally:
            self._slots[endpoint].release()
            if not protected:
                self._general_slots[endpoint].release()


__all__ = ["GOAL_CONTEXTS_PER_ENDPOINT", "GoalSlotPool"]
