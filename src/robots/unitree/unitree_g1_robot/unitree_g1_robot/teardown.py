"""Bounded executor teardown shared by the runtime and the vendor mock.

``rclpy``'s ``Executor.shutdown(timeout_sec)`` stops dispatching and waits at
most ``timeout_sec`` for callbacks that are already executing. It returns
``False`` when that wait expires and does not cancel the callback, so closing
the SDK or destroying the node from the timed-out branch would race a callback
that may still be using them.

The process cannot wait the callback out either. ``MultiThreadedExecutor`` owns
a ``concurrent.futures.ThreadPoolExecutor`` whose workers are joined by
``concurrent.futures.thread._python_exit`` at interpreter exit, so a callback
that never returns keeps the process alive after ``main()`` returns.

Teardown therefore has four outcomes, and only one of them is success:

* ``executor timeout`` - the bounded wait expired; a callback may still run;
* ``SDK close failed`` - the channels could not be closed, so the node is left
  alive as well (its handlers may still be reached by the SDK reader threads);
* ``node destroy failed`` - the SDK closed but the node could not be destroyed;
* success - both steps ran and the caller may report a completed cleanup.

Every failure is returned to the caller with a reason and detail; nothing is
swallowed. The caller emits one diagnostic naming the failed step and exits
with ``TEARDOWN_FAILURE_EXIT_CODE`` through ``os._exit``. That bounded exit is
only possible while Python can still run (the GIL is available); a vendor call
that holds the GIL inside the library (Issue #207) can only be bounded by an
external supervisor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rclpy.executors import Executor

#: Bound on the post-spin wait for callbacks that are already executing.
TEARDOWN_TIMEOUT_S = 2.0

#: Exit status for any failed teardown (timeout, SDK close failure or node
#: destroy failure). It is distinct from 0 (clean teardown) and 1 (startup
#: failure); the diagnostic line names which step failed.
TEARDOWN_FAILURE_EXIT_CODE = 3

#: Failure reasons carried by :class:`TeardownResult`.
TEARDOWN_TIMEOUT = "executor timeout"
TEARDOWN_CLOSE_FAILED = "SDK close failed"
TEARDOWN_DESTROY_FAILED = "node destroy failed"

#: Stable diagnostic prefix of the timed-out branch (tests and operators grep it).
TEARDOWN_TIMEOUT_MARKER = "executor did not quiesce within"


@dataclass(frozen=True)
class TeardownResult:
    """Outcome of one bounded teardown attempt."""

    ok: bool
    reason: str = ""
    detail: str = ""


def bounded_executor_teardown(
    executor: Executor,
    node: Any,
    close: Callable[[], Any],
    *,
    timeout_s: float = TEARDOWN_TIMEOUT_S,
) -> TeardownResult:
    """Wait for in-flight callbacks, then close the SDK and destroy the node.

    Success requires all three steps. A failed step stops the sequence: an SDK
    close failure does not destroy the node (the reader threads may still reach
    its handlers), and a destroy failure is reported instead of being treated
    as a completed cleanup.
    """
    if not executor.shutdown(timeout_sec=timeout_s):
        return TeardownResult(False, TEARDOWN_TIMEOUT)
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - report the failed step, keep the node alive
        return TeardownResult(False, TEARDOWN_CLOSE_FAILED, f"{type(exc).__name__}: {exc}")
    try:
        node.destroy_node()
    except Exception as exc:  # noqa: BLE001 - report the failed step, do not claim cleanup
        return TeardownResult(False, TEARDOWN_DESTROY_FAILED, f"{type(exc).__name__}: {exc}")
    return TeardownResult(True)


def teardown_failure_message(
    label: str,
    result: TeardownResult,
    *,
    timeout_s: float = TEARDOWN_TIMEOUT_S,
) -> str:
    """One diagnostic line that names the failed teardown step."""
    detail = f" ({result.detail})" if result.detail else ""
    if result.reason == TEARDOWN_TIMEOUT:
        step = f"{TEARDOWN_TIMEOUT_MARKER} {timeout_s:.1f}s; a callback may still be using the SDK/node"
    elif result.reason == TEARDOWN_CLOSE_FAILED:
        step = f"{TEARDOWN_CLOSE_FAILED}{detail}; the node was not destroyed"
    elif result.reason == TEARDOWN_DESTROY_FAILED:
        step = f"{TEARDOWN_DESTROY_FAILED}{detail}"
    else:
        step = f"{result.reason}{detail}"
    return f"{label}: executor teardown failed: {step}; the process exits {TEARDOWN_FAILURE_EXIT_CODE}"
