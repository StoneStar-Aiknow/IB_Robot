"""Load-time rollback for native runtime resources."""

from __future__ import annotations

import threading
from collections.abc import Callable


class LoadRollback:
    """Run registered partial-load cleanup callbacks once in reverse order."""

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
        with self._lock:
            if self._finished:
                return ()
            self._finished = True
            callbacks = tuple(reversed(self._callbacks))
            self._callbacks.clear()
        errors: list[Exception] = []
        for callback, args, kwargs in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
        return tuple(errors)


__all__ = ["LoadRollback"]
