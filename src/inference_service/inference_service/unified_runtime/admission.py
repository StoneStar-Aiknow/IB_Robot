"""Runtime-owned process resource admission."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


def _acquire(lock, deadline: object | None, message: str) -> None:
    remaining = None
    if deadline is not None:
        getter = getattr(deadline, "remaining_seconds", None)
        remaining = getter() if callable(getter) else None
        if remaining is not None and remaining <= 0:
            raise RuntimeError(message)
    if remaining is None:
        lock.acquire()
    elif not lock.acquire(timeout=remaining):
        raise RuntimeError(message)


class NativeAdmission:
    """One handle-level instance/domain admission lease."""

    def __init__(self, provider: object, name: str, capabilities: object) -> None:
        self._provider = provider
        self._name = name
        self._instance = provider.register_instance(
            name, bool(getattr(capabilities, "supports_multiple_instances", False))
        )
        self._instance_gate = threading.BoundedSemaphore(int(getattr(capabilities, "max_in_flight_per_instance", 1)))
        self._instance_exclusive = threading.Lock()
        self._domain = None
        domain = getattr(capabilities, "resource_domain", None)
        if domain is not None:
            self._domain = provider.register(domain, int(getattr(capabilities, "resource_domain_limit", 1)))
        self._closed = False
        self._close_lock = threading.Lock()

    @contextmanager
    def admit(self, deadline: object | None = None) -> Iterator[None]:
        self._assert_open()
        _acquire(self._instance_gate, deadline, "runtime request admission deadline expired")
        domain_acquired = False
        try:
            if self._domain is not None:
                self._domain.acquire(getattr(deadline, "deadline_at", None) if deadline else None)
                domain_acquired = True
            yield
        finally:
            if domain_acquired:
                self._domain.release()
            self._instance_gate.release()

    @contextmanager
    def exclusive(self, deadline: object | None = None) -> Iterator[None]:
        self._assert_open()
        _acquire(self._instance_exclusive, deadline, "runtime control admission deadline expired")
        try:
            _acquire(self._instance_gate, deadline, "runtime control admission deadline expired")
            try:
                domain = self._domain
                if domain is None:
                    yield
                else:
                    with domain.exclusive(getattr(deadline, "deadline_at", None) if deadline else None):
                        yield
            finally:
                self._instance_gate.release()
        finally:
            self._instance_exclusive.release()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._domain is not None:
            self._domain.close()
        self._instance.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime admission is closed")


__all__ = ["NativeAdmission"]
