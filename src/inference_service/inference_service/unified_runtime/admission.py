"""Runtime-owned process resource admission."""

from __future__ import annotations

import threading
from _thread import LockType
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .contracts import Deadline, ExecutionContext
from .errors import RecoveryAction, RecoveryScope
from .lifecycle import POLL_INTERVAL_SECONDS


def acquire_admission(
    lock: LockType | threading.BoundedSemaphore,
    deadline: Deadline | None,
    message: str,
    context: ExecutionContext | None = None,
    check: Callable[[], None] | None = None,
) -> None:
    """Wait for instance or shared-domain capacity without consuming it on failure.

    With a context, its deadline and cancellation govern each bounded wait;
    ``check`` also runs before each attempt to reject revoked admission.
    Without a context, use the explicit deadline and raise RuntimeError(message)
    on expiry. Callers recheck admission after acquisition and own release.
    """
    if context is not None:
        while True:
            context.check("admission")
            if check is not None:
                check()
            remaining = context.deadline.remaining_seconds()
            if lock.acquire(
                timeout=POLL_INTERVAL_SECONDS if remaining is None else min(POLL_INTERVAL_SECONDS, remaining)
            ):
                return
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
        self._stateful = bool(getattr(capabilities, "stateful", False))
        self._instance_gate = threading.BoundedSemaphore(int(getattr(capabilities, "max_in_flight_per_instance", 1)))
        self._instance_exclusive = threading.Lock()
        self._domain = None
        domain = getattr(capabilities, "resource_domain", None)
        self._instance = provider.register_instance(
            name, bool(getattr(capabilities, "supports_multiple_instances", False))
        )
        try:
            if domain is not None:
                self._domain = provider.register(domain, int(getattr(capabilities, "resource_domain_limit", 1)))
        except Exception:
            self._instance.close()
            raise
        self._closed = False
        self._close_lock = threading.Lock()
        self._retained: list[tuple[object, bool]] = []

    @contextmanager
    def admit(
        self, deadline: object | None = None, *, context=None, managed: bool = False, recovery_key=None, check=None
    ) -> Iterator[None]:
        self._assert_open()
        if managed:
            if not self._instance_gate.acquire(blocking=False):
                raise RuntimeError("scheduler/runtime instance capacity mismatch")
        else:
            acquire_admission(
                self._instance_gate, deadline, "runtime request admission deadline expired", context, check
            )
        domain_acquired = False
        retained = False
        entered = False
        try:
            if self._domain is not None and not managed:
                kwargs = {"context": context} if context is not None else {}
                if check is not None:
                    kwargs["check"] = check
                self._domain.acquire(getattr(deadline, "deadline_at", None) if deadline else None, **kwargs)
                domain_acquired = True
            if context is not None:
                context.check("admission")
            if check is not None:
                check()
            entered = True
            yield
        except BaseException as exc:
            evidence = getattr(exc, "evidence", None)
            known = getattr(evidence, "outcome_known", getattr(exc, "outcome_known", not self._stateful))
            if entered and not known:
                # Python unwinding does not prove native completion. Recovery
                # or successful close must return these exact capacity tokens.
                with self._close_lock:
                    recovery = getattr(exc, "recovery", None)
                    runtime_fault = (
                        recovery is not None
                        and recovery.scope is not RecoveryScope.STREAM
                        and (recovery.action in {RecoveryAction.RESET_RUNTIME, RecoveryAction.RELOAD})
                    )
                    key = None if runtime_fault else recovery_key
                    self._retained.append((key, domain_acquired))
                retained = True
            raise
        finally:
            if not retained:
                if domain_acquired:
                    self._domain.release()
                self._instance_gate.release()

    @property
    def has_retained(self) -> bool:
        """Whether recovery still owes proof for an occupied capacity token."""
        with self._close_lock:
            return bool(self._retained)

    def recovered(self, recovery_key=None) -> None:
        """Return held slots only after their owning state bank has recovered."""
        with self._close_lock:
            self._release_retained(recovery_key)

    def _release_retained(self, recovery_key) -> None:
        remaining = []
        for key, domain_acquired in self._retained:
            if recovery_key is not None and key != recovery_key:
                remaining.append((key, domain_acquired))
                continue
            if domain_acquired:
                self._domain.release()
            self._instance_gate.release()
        self._retained = remaining

    @contextmanager
    def exclusive(self, deadline: object | None = None) -> Iterator[None]:
        self._assert_open()
        acquire_admission(self._instance_exclusive, deadline, "runtime control admission deadline expired")
        try:
            acquire_admission(self._instance_gate, deadline, "runtime control admission deadline expired")
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
            self._release_retained(None)
            if self._domain is not None:
                self._domain.close()
            self._instance.close()
            self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime admission is closed")


__all__ = ["NativeAdmission", "acquire_admission"]
