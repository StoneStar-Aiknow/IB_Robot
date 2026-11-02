"""Load-time rollback for native runtime resources."""

from __future__ import annotations

import threading
from collections.abc import Callable

from .lifecycle import check_lifecycle, lifecycle_lock


class LoadRollback:
    """Release in reverse order, retaining failed cleanup and its dependencies."""

    def __init__(self) -> None:
        self._callbacks: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []
        self._finished = False
        self._lock = threading.Lock()

    def defer(self, callback: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        with self._lock:
            if self._finished:
                raise RuntimeError("cannot register rollback after it has finished")
            self._callbacks.append((callback, args, kwargs))

    def commit(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._callbacks.clear()

    def rollback(self) -> tuple[Exception, ...]:
        try:
            with lifecycle_lock(self._lock):
                if self._finished:
                    return ()
                while self._callbacks:
                    callback, args, kwargs = self._callbacks[-1]
                    check_lifecycle()
                    callback(*args, **kwargs)
                    self._callbacks.pop()
                self._finished = True
                return ()
        except Exception as exc:
            # Earlier callbacks may release a lease used by this resource.
            return (exc,)

    def close(self) -> None:
        """Expose retained cleanup through the same owner interface as handles."""
        errors = self.rollback()
        if errors:
            raise errors[0]


__all__ = ["LoadRollback"]
