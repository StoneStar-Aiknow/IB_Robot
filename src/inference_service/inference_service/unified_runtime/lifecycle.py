"""An inherited deadline for synchronous, nested lifecycle cleanup."""

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from .contracts import Deadline

_deadline = ContextVar("runtime_lifecycle_deadline", default=None)
# Signals stop the entire host, including executor workers that do not inherit
# ContextVars. Per-call deadlines and cancellation remain context-local.
_stop = None
_context = ContextVar("runtime_lifecycle_context", default=None)
POLL_INTERVAL_SECONDS = 0.05


def check_startup():
    check_lifecycle(phase="startup")
    stopping = _stop
    if stopping is not None and stopping.is_set():
        raise RuntimeError("model host stopped during initialization")


def lifecycle_context():
    """Return the enclosing control context, including its cancellation token."""
    return _context.get()


def lifecycle_deadline(deadline=None, *additional_deadlines):
    candidates = [
        value
        for value in (deadline, *additional_deadlines, _deadline.get(), getattr(_stop, "deadline", None))
        if value is not None and value.bounded
    ]
    return min(candidates, key=lambda value: value.remaining_seconds()) if candidates else Deadline.unbounded()


def lifecycle_remaining(timeout):
    remaining = lifecycle_deadline().remaining_seconds()
    return timeout if remaining is None else min(timeout, max(0.0, remaining))


@contextmanager
def lifecycle_lock(lock, context=None):
    if context is None:
        context = lifecycle_context()
    while True:
        deadline = lifecycle_deadline(context.deadline if context is not None else None)
        if context is None and deadline.expired:
            raise TimeoutError("lifecycle owner is still busy")
        check_lifecycle(context, phase="control_lock")
        remaining = deadline.remaining_seconds()
        if lock.acquire(timeout=POLL_INTERVAL_SECONDS if remaining is None else min(POLL_INTERVAL_SECONDS, remaining)):
            break
    try:
        yield
    finally:
        lock.release()


def check_lifecycle(context=None, *, phase="native_cleanup"):
    """Check between native calls; an in-flight noncancellable SDK call must finish."""
    context = context if context is not None else _context.get()
    lifecycle_deadline(context.deadline if context is not None else None).check(phase)
    if context is not None:
        context.check(phase)


@contextmanager
def lifecycle_scope(context):
    """Propagate an existing control deadline/token without renewing its budget."""
    deadline_token = _deadline.set(lifecycle_deadline(context.deadline))
    context_token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(context_token)
        _deadline.reset(deadline_token)


@contextmanager
def lifecycle_budget(timeout):
    """Nested owners share one budget; a child can shorten but never renew it."""
    deadline = lifecycle_deadline(Deadline.after(timeout))
    token = _deadline.set(deadline)
    try:
        yield deadline
    finally:
        _deadline.reset(token)


class LifecycleStop(threading.Event):
    """A process stop request starts the cleanup budget, including startup rollback."""

    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout
        self.deadline = None

    def set(self):
        if self.deadline is None:
            self.deadline = Deadline.after(self.timeout)
        super().set()


@contextmanager
def lifecycle_stop_scope(stopping):
    """Install a process stop source from the signal-owning main thread."""
    global _stop
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("process lifecycle stop must be installed on the main thread")
    previous, _stop = _stop, stopping
    try:
        yield
    finally:
        _stop = previous


def wait_for_startup(wait, timeout):
    """Poll interruptible discovery without cancelling any admitted SDK work."""
    until = time.monotonic() + timeout
    while True:
        check_startup()
        remaining = until - time.monotonic()
        if remaining <= 0:
            return False
        if wait(min(POLL_INTERVAL_SECONDS, remaining)):
            check_startup()
            return True
