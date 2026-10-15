"""Bind the CLI's existing stop event to its Hermes execution owner."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, nullcontext, suppress

from robot_skill_cli.hermes_stop_state import StopStore


def record_workflow_event(event: str, data: dict) -> None:
    """Persist identity/results before stdout, even if the Hermes parent exits."""
    root = os.environ.get("IBROBOT_STOP_ROOT", "")
    if not root:
        return
    store = StopStore(root)
    session = os.environ.get("IBROBOT_STOP_SESSION", "")
    token = os.environ.get("IBROBOT_STOP_TOKEN", "")
    if event == "workflow_presentation":
        store.update(session, token, presentation=data)
    elif event == "workflow_terminal":
        from robot_skill_cli.hermes_robot_runner import terminal_state

        owner = store.snapshot(session)["owner"]
        state = terminal_state(owner.get("presentation") if owner else None, data)
        store.update(session, token, terminal=data, state=state, finished_at=time.time())


def wait_for_presentation() -> None:
    """Do not submit motion before the owning Hermes surface displays the plan."""
    if os.environ.get("IBROBOT_STOP_PRESENTATION_ACK") != "1":
        return
    store = StopStore(os.environ["IBROBOT_STOP_ROOT"])
    session, token = os.environ["IBROBOT_STOP_SESSION"], os.environ["IBROBOT_STOP_TOKEN"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        owner = store.snapshot(session)["owner"]
        if not owner or owner["token"] != token:
            break
        if owner.get("presentation_flushed"):
            return
        if owner.get("presentation_failed"):
            break
        time.sleep(0.01)
    raise RuntimeError("Hermes did not confirm plan presentation")


def record_admission_error(code: str, message: str) -> None:
    """Record only errors with explicit evidence that no goal could be submitted."""
    root = os.environ.get("IBROBOT_STOP_ROOT", "")
    if not root:
        return
    store = StopStore(root)
    session, token = os.environ.get("IBROBOT_STOP_SESSION", ""), os.environ.get("IBROBOT_STOP_TOKEN", "")
    with store.locked(session) as record:
        owner = record["owner"]
        if (
            owner
            and owner["token"] == token
            and (owner["presentation"] is None or owner.get("fresh_submission_pending") is True)
        ):
            owner.update(state="not_submitted", admission_error={"code": code, "message": message})


@contextmanager
def managed_stop(stop_event: threading.Event, *, fresh_request: bool = False):
    root = os.environ.get("IBROBOT_STOP_ROOT", "")
    if not root:
        if os.environ.get("IBROBOT_HERMES_MANAGED_REQUIRED") == "1":
            raise ValueError("Hermes robot motion requires the managed workflow adapter; rerun hermes-robot-configure")
        yield nullcontext
        return
    store = StopStore(root)
    session = os.environ.get("IBROBOT_STOP_SESSION", "")
    token = os.environ.get("IBROBOT_STOP_TOKEN", "")
    if not token:
        raise ValueError("missing managed workflow identity")
    # Explicit request IDs may refer to an already-running idempotent task.
    # Only the fresh, internally generated request path can prove no prior goal.
    if fresh_request:
        store.update(session, token, fresh_submission_pending=True)
    closed = threading.Event()

    def check() -> bool:
        try:
            stopped = store.stopped(session, token)
        except (OSError, ValueError, RuntimeError):
            stopped = True
        if stopped:
            stop_event.set()
            return True
        return False

    def watch():
        while not closed.wait(0.02):
            if check():
                with suppress(OSError, ValueError, RuntimeError):
                    store.update(session, token, stop_observed_at=time.time())
                return

    check()  # Preserve a stop that arrived before this process initialized.
    thread = threading.Thread(target=watch, name="hermes-workflow-stop", daemon=True)
    thread.start()
    try:

        @contextmanager
        def guard():
            # Persist before entering code that can send a goal, including when
            # it raises or the client dies before the lock context exits.
            store.update(session, token, fresh_submission_pending=False)
            with store.submission_guard(session, token, stop_event):
                yield

        yield guard
    finally:
        closed.set()
        thread.join(timeout=1.0)
