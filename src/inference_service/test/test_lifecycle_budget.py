"""Lifecycle budgets remain bounded across nested cleanup and loading."""

import threading
from types import SimpleNamespace

import pytest

from inference_service.unified_runtime import Deadline, ModelRuntimeHandle
from inference_service.unified_runtime.lifecycle import (
    lifecycle_budget,
    lifecycle_deadline,
    lifecycle_lock,
    lifecycle_remaining,
)


def test_nested_budget_never_renews_parent_deadline_and_restores_scope():
    assert lifecycle_deadline().is_unbounded
    with lifecycle_budget(1) as parent:
        with lifecycle_budget(10) as child:
            assert child is parent
            assert 0 < lifecycle_remaining(10) <= 1
        with lifecycle_budget(0.01) as shorter:
            assert shorter.remaining_seconds() < parent.remaining_seconds()
        assert lifecycle_deadline() is parent
    assert lifecycle_deadline().is_unbounded


@pytest.mark.parametrize("shortest", ["parent", "context", "explicit"])
def test_control_context_keeps_shortest_deadline_and_selected_cancellation_token(shortest):
    from inference_service.unified_runtime import ExecutionContext
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    deadlines = {name: Deadline.after(1 if name == shortest else 10) for name in ("parent", "context", "explicit")}
    parent = ExecutionContext("parent", deadlines["parent"])
    supplied = ExecutionContext("child", deadlines["context"])
    handle = ModelRuntimeHandle(object())
    with lifecycle_scope(parent):
        control = handle._control_context(supplied, deadlines["explicit"])
        assert control.deadline is deadlines[shortest]
        assert control.token is supplied.token
        inherited = handle._control_context(None, deadlines["explicit"])
        assert inherited.token is parent.token
        assert inherited.deadline is deadlines["explicit" if shortest == "explicit" else "parent"]


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_nested_handle_control_inherits_parent_cancellation_while_waiting(operation):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    calls = []
    handle = ModelRuntimeHandle(
        SimpleNamespace(reset=lambda: calls.append("reset"), close=lambda: calls.append("close")), resettable=True
    )
    handle.load()
    parent = ExecutionContext("parent")
    entered = threading.Event()
    handle._control_lock.acquire()

    def control():
        with lifecycle_scope(parent):
            entered.set()
            getattr(handle, operation)()

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(control)
        try:
            assert entered.wait(1)
            parent.token.cancel("stop nested cleanup")
            with pytest.raises(ExecutionFailure, match="request_canceled"):
                pending.result(timeout=0.5)
            assert not calls
        finally:
            handle._control_lock.release()
    handle.close()
    assert calls == ["close"]


@pytest.mark.parametrize("kind", ["rollback", "providers", "lease", "acl_manager", "pipeline_manager"])
def test_cleanup_owner_locks_obey_the_waiting_callers_budget(kind):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
    from inference_service.pipeline.manager import InferencePipelineManager
    from inference_service.unified_runtime import LoadRollback, RuntimeProviders

    entered, release = threading.Event(), threading.Event()

    def blocking(*_):
        entered.set()
        assert release.wait(2)
        return 0

    if kind == "rollback":
        owner = LoadRollback()
        owner.defer(blocking)
        first_call, second_call = owner.close, owner.close
    elif kind == "providers":
        owner = RuntimeProviders(SimpleNamespace(close=blocking), object())
        first_call, second_call = owner.close, owner.close
    elif kind == "pipeline_manager":
        owner = InferencePipelineManager((SimpleNamespace(pipeline_id="p", close=blocking),))
        first_call, second_call = owner.close, owner.close
    else:
        acl = SimpleNamespace(
            init=blocking if kind == "acl_manager" else lambda: 0,
            finalize=blocking if kind == "lease" else lambda: 0,
            rt=SimpleNamespace(
                set_device=lambda _: 0,
                create_context=lambda _: ("context", 0),
                set_context=lambda _: 0,
                destroy_context=lambda _: 0,
                reset_device=lambda _: 0,
            ),
        )
        manager = AclRuntimeManager(lambda: acl)
        if kind == "lease":
            owner = manager.acquire(0)
            first_call, second_call = owner.close, owner.close
        else:
            owner = manager
            first_call, second_call = lambda: owner.acquire(0), owner.close

    def waiting_close():
        with lifecycle_budget(0.01):
            with pytest.raises(Exception) as error:
                second_call()
            assert "busy" in str(error.value) or "deadline" in str(error.value)

    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(first_call)
        try:
            assert entered.wait(1)
            workers.submit(waiting_close).result(timeout=0.5)
            assert not first.done()
        finally:
            release.set()
        result = first.result(timeout=1)
    if kind == "acl_manager":
        result.close()
    owner.close()


def test_cleanup_budget_expires_while_waiting_for_lifecycle_owner():
    lock = threading.Lock()
    lock.acquire()
    try:
        with lifecycle_budget(0), pytest.raises(TimeoutError, match="still busy"), lifecycle_lock(lock):
            pytest.fail("cleanup must not enter while another owner holds the lock")
    finally:
        lock.release()
    with lifecycle_lock(lock):
        assert lock.locked()
    assert not lock.locked()


def test_expired_inherited_budget_prevents_loading():
    from inference_service.unified_runtime.errors import DeadlineExceeded

    calls = []
    handle = ModelRuntimeHandle(SimpleNamespace(load=lambda: calls.append("load")))
    with lifecycle_budget(0):
        with pytest.raises(DeadlineExceeded):
            handle.load()
        assert not calls and not handle.ready
    handle.load()
    assert calls == ["load"] and handle.ready
    handle.close()


@pytest.mark.parametrize("operation", ["close", "reset", "close_stream", "reset_stream"])
def test_control_lock_wait_obeys_shared_deadline_and_retains_owner(operation):
    from types import SimpleNamespace

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRuntimeHandle

    calls = []
    runtime = SimpleNamespace(
        open_stream=lambda _: object(),
        step=lambda *_: {},
        close_stream=lambda *_: calls.append("stream"),
        reset_stream=lambda *_: None,
        reset=lambda *_: None,
        close=lambda: calls.append("runtime"),
    )
    handle = ModelRuntimeHandle(runtime, streaming_runtime=runtime, state_scope="stream", resettable=True)
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))
    lock = handle._streams[stream.stream_id].control_lock if "stream" in operation else handle._control_lock
    lock.acquire()
    try:
        with lifecycle_budget(0.01), pytest.raises(ExecutionFailure):
            getattr(handle, operation)(stream, ExecutionContext("control")) if "stream" in operation else getattr(
                handle, operation
            )()
        assert calls == [] and not handle.closed
        if operation == "close":
            assert not handle.ready
    finally:
        lock.release()
        handle.close()
    assert handle.closed and calls == ["stream", "runtime"]


def test_expired_close_does_not_start_native_cleanup_even_when_quiescent():
    from types import SimpleNamespace

    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle

    calls = []
    handle = ModelRuntimeHandle(SimpleNamespace(close=lambda: calls.append("close")))
    handle.load()
    with lifecycle_budget(0), pytest.raises(ExecutionFailure) as failure:
        handle.close()
    assert failure.value.close_pending and not calls and not handle.ready
    handle.close()
    assert calls == ["close"]


def test_load_rollback_keeps_dependencies_when_cleanup_consumes_budget():
    from datetime import datetime, timedelta, timezone

    from inference_service.unified_runtime import ExecutionContext, LoadRollback
    from inference_service.unified_runtime.errors import DeadlineExceeded
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    clock = [datetime(2026, 10, 29, tzinfo=timezone.utc)]
    deadline = Deadline.after(1, now=clock[0], clock=lambda: clock[0])
    calls = []
    rollback = LoadRollback()
    rollback.defer(calls.append, "lease")

    def close_model():
        calls.append("model")
        clock[0] += timedelta(seconds=2)

    rollback.defer(close_model)
    with lifecycle_scope(ExecutionContext("cleanup", deadline)):
        errors = rollback.rollback()
    assert len(errors) == 1 and isinstance(errors[0], DeadlineExceeded)
    assert calls == ["model"]
    assert rollback.rollback() == ()
    assert calls == ["model", "lease"]


def test_nested_budget_bounds_handle_control_without_extending_explicit_deadline():
    from types import SimpleNamespace

    from inference_service.unified_runtime import ExecutionContext, ModelRuntimeHandle

    handle = ModelRuntimeHandle(SimpleNamespace())
    handle.load()
    explicit = ExecutionContext("short", Deadline.after(0.01))
    with lifecycle_budget(1), lifecycle_budget(10):
        bounded = handle._control_context(None, None)
        assert bounded.deadline.remaining_seconds() <= 1
        assert handle._control_context(explicit, None).deadline == explicit.deadline
    handle.close()


def test_handle_close_budget_retains_active_work_until_retry():
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    entered, release = threading.Event(), threading.Event()
    closed = []

    class Model:
        def execute(self, request, context):
            entered.set()
            assert release.wait(3)
            return {}

        def close(self):
            closed.append(True)

    handle = ModelRuntimeHandle(Model())
    handle.load()
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(handle.execute, ModelRequest(), ExecutionContext("held"))
        try:
            assert entered.wait(1)
            with lifecycle_budget(0.03), pytest.raises(ExecutionFailure) as failure:
                handle.close()
            assert failure.value.close_pending
            assert not closed and not handle.closed and not handle.ready
            assert lifecycle_deadline().is_unbounded
        finally:
            release.set()
        running.result(timeout=2)
    handle.close()
    assert closed == [True] and handle.closed
