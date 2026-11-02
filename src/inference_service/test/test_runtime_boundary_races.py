"""Deterministic races at the public execution and lifecycle boundaries."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
from inference_service.backends.errors import BackendAdmissionError, BackendInferenceError
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeState,
    RuntimeAssembly,
    RuntimeProviders,
    StreamState,
)


def streaming(*, bank="per_stream", **operations):
    native = SimpleNamespace(
        open_stream=lambda _: object(),
        step=lambda *_: {},
        reset_stream=lambda *_: None,
        close_stream=lambda *_: None,
        close=lambda: None,
    )
    native.__dict__.update(operations)
    owner = ModelRuntimeHandle(
        native, streaming_runtime=native, state_scope="stream", state_bank_mode=bank, stateful=True, resettable=True
    )
    owner.load()
    return owner


@pytest.mark.parametrize("entry", ["execute", "frame", "stream"])
@pytest.mark.parametrize("resettable", [False, True])
@pytest.mark.parametrize("cleanup_pending", [False, True])
def test_public_failure_matches_final_recovery_and_diagnostic_snapshot(entry, resettable, cleanup_pending):
    from inference_service.pipeline.runtime import InferencePipeline

    def fail(*_args, **_kwargs):
        error = BackendInferenceError(
            "cleanup pending" if cleanup_pending else "unknown", operation_started=True, outcome_known=cleanup_pending
        )
        error.close_pending = cleanup_pending
        raise error

    def submit(*args, **kwargs):
        future = Future()
        try:
            fail()
        except Exception as exc:
            future.set_exception(exc)
        return future

    native = SimpleNamespace(
        execute=fail,
        submit_frame=submit,
        open_stream=lambda _: object(),
        step=fail,
        reset_stream=lambda *_: None,
        close_stream=lambda *_: None,
        close=lambda: None,
    )
    owner = ModelRuntimeHandle(
        native,
        resettable=resettable,
        streaming_runtime=native,
        state_scope="stream" if entry == "stream" else "request",
        state_bank_mode="runtime_exclusive" if entry == "stream" else None,
    )
    owner.load()
    try:
        stream = owner.open_stream(ExecutionContext("open")) if entry == "stream" else None
        with pytest.raises(ExecutionFailure) as caught:
            if entry == "stream":
                owner.step(stream, ModelRequest(), ExecutionContext("bad"))
            elif entry == "frame":
                owner.submit_frame(ModelRequest(), ExecutionContext("bad")).result(1)
            else:
                owner.execute(ModelRequest(), ExecutionContext("bad"))
        failure = caught.value
        assert failure.recovery == owner._recovery_requirement == owner._last_failure.recovery
        assert failure.recoverable == owner._recovery_available == owner._last_failure.recoverable
        assert failure.recovery.scope.value == "runtime"
        assert failure.recovery.action.value == ("reset_runtime" if resettable else "reload")
        assert owner.state is (LifecycleState.RESET_REQUIRED if resettable else LifecycleState.FAILED)
        with pytest.raises(Exception) as projected:
            InferencePipeline._raise_execution_failure(failure)
        assert projected.value.recovery == failure.recovery
        assert projected.value.recoverable == failure.recoverable
    finally:
        owner.close()


@pytest.mark.parametrize("entry", ["runtime", "session", "stream", "exclusive_stream", "runtime_stream"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_reset_completion_survives_late_control_and_releases_capacity(entry, stop):
    from datetime import datetime, timedelta, timezone

    from inference_service.model_sessions import ModelSession
    from inference_service.unified_runtime import RecoveryAction

    now = [datetime(2026, 11, 3, tzinfo=timezone.utc)]
    context = ExecutionContext("reset", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    reset_calls = []

    def reset(*_):
        reset_calls.append("reset")
        if stop == "cancel":
            context.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    def execute(*_):
        raise BackendInferenceError("unknown", operation_started=True, outcome_known=False)

    class Session(ModelSession):
        supports_reset = True

        def __init__(self):
            super().__init__("test", BackendCapabilities())

        def _load(self, *_):
            pass

        def _execute(self, *_):
            return {}

        def _reset(self):
            reset()

        def _close(self):
            pass

    native = SimpleNamespace(
        execute=execute,
        reset=reset,
        close=lambda: None,
        open_stream=lambda _: object(),
        step=execute,
        reset_stream=reset,
        close_stream=lambda *_: None,
    )
    if entry == "session":
        native.reset = Session().reset
    if entry in {"stream", "exclusive_stream", "runtime_stream"}:
        del native.reset
    providers = RuntimeProviders(object(), ResourceDomainAdmissions())
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            native,
            resettable=True,
            streaming_runtime=native,
            state_scope="stream" if "stream" in entry else "request",
            capabilities=BackendCapabilities(resettable=True),
            providers=providers,
            state_bank_mode=("per_stream" if entry == "stream" else "runtime_exclusive") if "stream" in entry else None,
        )
    )
    owner.load()
    try:
        stream = owner.open_stream(ExecutionContext("open")) if "stream" in entry else None
        if entry in {"runtime", "session", "runtime_stream", "stream"}:
            with pytest.raises(ExecutionFailure):
                if stream:
                    owner.step(stream, ModelRequest(), ExecutionContext("bad"))
                else:
                    owner.execute(ModelRequest(), ExecutionContext("bad"))
            assert owner._admission._retained
        with pytest.raises(ExecutionFailure) as caught:
            if entry in {"stream", "exclusive_stream"}:
                owner.reset_stream(stream, context)
            else:
                owner.reset(context)
        assert caught.value.code == ("request_canceled" if stop == "cancel" else "deadline_exceeded")
        assert caught.value.evidence.state is OutcomeState.COMPLETED and caught.value.evidence.outcome_known
        assert caught.value.recovery.action is RecoveryAction.NONE
        assert reset_calls == ["reset"] and owner.ready and not owner._admission._retained
        assert owner._recovery_requirement is None
        if stream:
            assert stream.state is StreamState.OPEN
        native.execute = lambda *_: {}
        native.step = lambda *_: {}
        if stream:
            owner.step(stream, ModelRequest(), ExecutionContext("valid", Deadline.after(1)))
        else:
            owner.execute(ModelRequest(), ExecutionContext("valid", Deadline.after(1)))
    finally:
        owner.close()


@pytest.mark.parametrize("reset_times_out", [False, True])
@pytest.mark.parametrize("reload_required", [False, True])
@pytest.mark.parametrize("stream_entry", [False, True])
def test_failure_during_reset_drain_cannot_restore_stale_ready(
    monkeypatch, reset_times_out, reload_required, stream_entry
):
    from inference_service.unified_runtime import OutcomeEvidence, RecoveryAction, RecoveryRequirement, RecoveryScope

    entered = {name: threading.Event() for name in ("unknown", "slow")}
    release_unknown, release_slow, draining = (threading.Event() for _ in range(3))
    resets = []

    def execute(_request, context):
        entered[context.request_id].set()
        assert (release_unknown if context.request_id == "unknown" else release_slow).wait(2)
        if context.request_id == "unknown":
            if reload_required or stream_entry:
                raise ExecutionFailure(
                    "native_fault",
                    "reload required",
                    evidence=OutcomeEvidence.started(outcome_known=False),
                    recovery=RecoveryRequirement(
                        RecoveryScope.RUNTIME,
                        RecoveryAction.RELOAD if reload_required else RecoveryAction.RESET_RUNTIME,
                    ),
                )
            raise BackendInferenceError("unknown", operation_started=True, outcome_known=False)
        return {}

    if stream_entry:
        native = SimpleNamespace(
            open_stream=lambda _: object(),
            step=lambda _stream, request, context: execute(request, context),
            reset_stream=lambda *_: resets.append("reset"),
            close_stream=lambda *_: None,
            close=lambda: None,
        )
        owner = ModelRuntimeHandle(
            native,
            streaming_runtime=native,
            state_scope="stream",
            state_bank_mode="per_stream",
            max_open_streams=2,
            max_active_executions=2,
            stateful=True,
            resettable=True,
        )
    else:
        owner = ModelRuntimeHandle(
            SimpleNamespace(execute=execute, reset=lambda *_: resets.append("reset")),
            max_active_executions=2,
            resettable=True,
        )
    owner.load()
    streams = {name: owner.open_stream(ExecutionContext(f"open-{name}")) for name in entered} if stream_entry else {}

    def run(name):
        if stream_entry:
            return owner.step(streams[name], ModelRequest(), ExecutionContext(name))
        return owner.execute(ModelRequest(), ExecutionContext(name))

    wait = owner._wait_quiescent_locked

    def observe(context, **kwargs):
        draining.set()
        wait(context, **kwargs)

    monkeypatch.setattr(owner, "_wait_quiescent_locked", observe)
    try:
        with ThreadPoolExecutor(3) as pool:
            unknown = pool.submit(run, "unknown")
            slow = pool.submit(run, "slow")
            assert all(event.wait(1) for event in entered.values())
            reset = pool.submit(owner.reset, ExecutionContext("reset", Deadline.after(0.1 if reset_times_out else 1)))
            assert draining.wait(1)
            release_unknown.set()
            with pytest.raises(ExecutionFailure):
                unknown.result(1)
            if not reset_times_out:
                release_slow.set()
            if reset_times_out or reload_required:
                with pytest.raises(ExecutionFailure):
                    reset.result(1)
                assert not owner.ready and not resets
            else:
                reset.result(1)
                assert owner.ready and resets == ["reset"] * (2 if stream_entry else 1)
            release_slow.set()
            slow.result(1)
        if reset_times_out:
            assert owner.state is (LifecycleState.FAILED if reload_required else LifecycleState.RESET_REQUIRED)
    finally:
        release_unknown.set()
        release_slow.set()
        owner.close()


def test_nonblocking_cancel_observer_leaves_completion_to_executor_hook():
    entered, release = threading.Event(), threading.Event()
    observations = []

    def execute(*_):
        entered.set()
        assert release.wait(2)
        return {}

    def cancel(*_args, **_kwargs):
        observations.append("executor")
        release.set()

    owner = ModelRuntimeHandle(SimpleNamespace(execute=execute, cancel=cancel))
    owner.load()
    context = ExecutionContext("cancel-callback")
    try:
        with ThreadPoolExecutor(1) as pool:
            work = pool.submit(owner.execute, ModelRequest(), context)
            assert entered.wait(1)

            def on_cancel():
                assert context.token.cancelled
                observations.append("observer")

            context.cancellation_token.add_callback(on_cancel)
            owner.cancel(context.request_id)
            with pytest.raises(ExecutionFailure) as caught:
                work.result(1)
            assert caught.value.evidence.state is OutcomeState.COMPLETED
            assert observations == ["observer", "executor"]
            assert owner.active_executions == 0
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize("whole_runtime", [False, True])
def test_step_revalidates_after_context_check_before_taking_execution_slot(monkeypatch, whole_runtime):
    calls = []
    owner = streaming(step=lambda *_: calls.append("step"))
    stream = owner.open_stream(ExecutionContext("open"))
    check = ExecutionContext.check

    def check_and_close(context, phase="request_boundary"):
        check(context, phase)
        if phase == "stream_step_admission":
            owner.close() if whole_runtime else owner.close_stream(stream)

    monkeypatch.setattr(ExecutionContext, "check", check_and_close)
    with pytest.raises(ExecutionFailure):
        owner.step(stream, ModelRequest(), ExecutionContext("step"))
    assert stream.closed and not calls
    owner.close()


def test_reset_revalidates_record_after_resolving_it(monkeypatch):
    calls = []
    owner = streaming(reset_stream=lambda *_: calls.append("reset"))
    stream = owner.open_stream(ExecutionContext("open"))
    resolve = owner._resolve_stream
    first = True

    def resolve_and_close(handle, operation, **kwargs):
        nonlocal first
        record = resolve(handle, operation, **kwargs)
        if operation == "reset_stream" and first:
            first = False
            owner.close_stream(stream)
        return record

    monkeypatch.setattr(owner, "_resolve_stream", resolve_and_close)
    with pytest.raises(ExecutionFailure):
        owner.reset_stream(stream, ExecutionContext("reset"))
    assert stream.closed and not calls
    owner.close()


@pytest.mark.parametrize("runtime_close", [False, True])
@pytest.mark.parametrize("reset_fails", [False, True])
def test_close_wins_over_an_already_running_stream_reset(runtime_close, reset_fails):
    entered, release = threading.Event(), threading.Event()

    def reset(*_):
        entered.set()
        assert release.wait(2)
        if reset_fails:
            raise RuntimeError("native reset failed")

    owner = streaming(reset_stream=reset, bank="runtime_exclusive" if runtime_close else "per_stream")
    stream = owner.open_stream(ExecutionContext("open"))
    if runtime_close:
        owner.mark_reset_required()
    with ThreadPoolExecutor(max_workers=1) as pool:
        resetting = (
            pool.submit(owner.reset, ExecutionContext("reset"))
            if runtime_close
            else pool.submit(owner.reset_stream, stream, ExecutionContext("reset"))
        )
        try:
            assert entered.wait(1)
            with pytest.raises(ExecutionFailure):
                owner.close(deadline=Deadline.after(0)) if runtime_close else owner.close_stream(
                    stream, deadline=Deadline.after(0)
                )
        finally:
            release.set()
        if reset_fails:
            with pytest.raises(ExecutionFailure, match="native reset failed"):
                resetting.result(1)
        else:
            resetting.result(1)
    if runtime_close:
        assert owner.state is LifecycleState.CLOSING
    with pytest.raises(ExecutionFailure):
        owner.step(stream, ModelRequest(), ExecutionContext("late-step"))
    owner.close()
    assert owner.closed


def test_cancel_reaches_active_stream_context_and_preserves_completion_evidence():
    entered, release = threading.Event(), threading.Event()

    def step(*_):
        entered.set()
        assert release.wait(2)
        return {}

    owner = streaming(step=step)
    stream = owner.open_stream(ExecutionContext("open"))
    context = ExecutionContext("step")
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(owner.step, stream, ModelRequest(), context)
        try:
            assert entered.wait(1)
            owner.cancel("step")
            assert context.cancellation_token.cancelled
        finally:
            release.set()
        with pytest.raises(ExecutionFailure) as failure:
            running.result(1)
    assert failure.value.evidence.state is OutcomeState.COMPLETED
    # The caller discarded this result and cannot advance matching host state.
    # Device completion frees capacity, but the advanced stream still needs reset.
    assert failure.value.evidence.outcome_known and failure.value.evidence.state_mutated
    assert stream.state is StreamState.RESET_REQUIRED and owner.ready
    assert owner._execution_contexts == {}
    with pytest.raises(ExecutionFailure, match="recovery_required"):
        owner.step(stream, ModelRequest(), ExecutionContext("blocked"))
    owner.reset_stream(stream, ExecutionContext("recover"))
    assert owner.step(stream, ModelRequest(), ExecutionContext("recovered")).evidence.outcome_known
    owner.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_frame_delivery_failure_preserves_successful_future_completion(cancelled):
    future = Future()
    owner = ModelRuntimeHandle(SimpleNamespace(submit_frame=lambda *_, **__: future), resettable=True)
    owner.load()
    context = ExecutionContext("frame", Deadline.after(0) if not cancelled else Deadline.unbounded())
    # Expire only at the delivery boundary, after successful registration.
    if not cancelled:
        from datetime import datetime, timedelta, timezone

        now = [datetime.now(timezone.utc)]
        context = ExecutionContext("frame", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    result = owner.submit_frame(ModelRequest(), context)
    if cancelled:
        owner.cancel("frame")
    else:
        now[0] += timedelta(seconds=2)
    future.set_result({})
    with pytest.raises(ExecutionFailure) as failure:
        result.result(1)
    assert failure.value.evidence.state is OutcomeState.COMPLETED
    assert failure.value.evidence.outcome_known and owner.ready
    owner.close()


def test_late_open_retains_native_stream_until_cleanup_can_run():
    closed = []

    def open_native(context):
        context.cancellation_token.cancel("caller left")
        return "native"

    owner = streaming(open_stream=open_native, close_stream=lambda native, _: closed.append(native))
    with pytest.raises(ExecutionFailure) as failure:
        owner.open_stream(ExecutionContext("open"))
    assert failure.value.evidence.state is OutcomeState.COMPLETED
    assert failure.value.close_pending and owner.open_stream_count == 1 and not closed
    owner.close()
    assert closed == ["native"]


def test_reentrant_close_rejects_before_waiting_for_another_closers_lock(monkeypatch):
    executing, closing = threading.Event(), threading.Event()
    failures = []

    def step(*_):
        executing.set()
        assert closing.wait(1)
        try:
            owner.close_stream(stream, deadline=Deadline.after(0.2))
        except ExecutionFailure as exc:
            failures.append(exc.code)
        return {}

    owner = streaming(step=step)
    stream = owner.open_stream(ExecutionContext("open"))
    wait = owner._wait_for_stream_quiescence_locked

    def drain(record, context):
        closing.set()
        return wait(record, context)

    monkeypatch.setattr(owner, "_wait_for_stream_quiescence_locked", drain)
    with ThreadPoolExecutor(2) as pool:
        work = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("step"))
        assert executing.wait(1)
        closer = pool.submit(owner.close_stream, stream, deadline=Deadline.after(1))
        work.result(timeout=2)
        closer.result(timeout=2)
    assert failures == ["stream_reentrant"] and stream.closed
    owner.close()


def test_public_cancel_reaches_open_stream_and_retains_late_native_result():
    entered, release = threading.Event(), threading.Event()
    closed = []

    def open_native(context):
        entered.set()
        assert release.wait(2)
        return "native"

    owner = streaming(open_stream=open_native, close_stream=lambda native, _: closed.append(native))
    context = ExecutionContext("opening")
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(owner.open_stream, context)
        try:
            assert entered.wait(1)
            owner.cancel(context.request_id)
            assert context.token.cancelled
        finally:
            release.set()
        with pytest.raises(ExecutionFailure) as caught:
            pending.result(timeout=1)
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.close_pending and owner.open_stream_count == 1
    assert not owner._execution_contexts
    owner.close()
    assert closed == ["native"]


def test_open_racing_runtime_close_keeps_failed_native_cleanup_owned():
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def open_native(_):
        entered.set()
        assert release.wait(2)
        return "native"

    def close_native(native, _):
        attempts.append(native)
        if len(attempts) == 1:
            raise RuntimeError("stream busy")

    owner = streaming(open_stream=open_native, close_stream=close_native)
    with ThreadPoolExecutor(max_workers=1) as pool:
        opening = pool.submit(owner.open_stream, ExecutionContext("open"))
        try:
            assert entered.wait(1)
            with pytest.raises(ExecutionFailure):
                owner.close(deadline=Deadline.after(0))
        finally:
            release.set()
        with pytest.raises(ExecutionFailure) as failure:
            opening.result(1)
        assert failure.value.close_pending and owner.open_stream_count == 1
    owner.close()
    assert attempts == ["native", "native"] and owner.closed


def test_native_domain_wait_is_cancellable_without_starting_backend(monkeypatch):
    provider = ResourceDomainAdmissions()
    blocker = provider.register("device", 1)
    blocker.acquire()
    waiting = threading.Event()
    gate = provider._domains["device"].semaphore
    acquire = gate.acquire

    def observe(**kwargs):
        waiting.set()
        return acquire(**kwargs)

    monkeypatch.setattr(gate, "acquire", observe)
    calls = []
    session = SimpleNamespace(
        name="session",
        capabilities=BackendCapabilities(resource_domain="device"),
        execute=lambda *_: calls.append("execute"),
    )
    providers = RuntimeProviders(object(), provider)
    owner = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=session, session=session, providers=providers))
    owner.load()
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(owner.execute, ModelRequest(), ExecutionContext("waiting"))
        try:
            assert waiting.wait(1)
            owner.cancel("waiting")
            with pytest.raises(ExecutionFailure) as failure:
                running.result(1)
            assert failure.value.evidence.state is OutcomeState.NOT_STARTED and owner.ready
            assert not calls
        finally:
            blocker.release()
    owner.close()
    blocker.close()
    providers.close()


def test_cleanup_pending_with_known_outcome_still_disables_handle_readiness():
    def execute(*_):
        error = BackendInferenceError("dataset cleanup pending", operation_started=True, outcome_known=True)
        error.close_pending = True
        raise error

    owner = ModelRuntimeHandle(SimpleNamespace(execute=execute))
    owner.load()
    with pytest.raises(ExecutionFailure) as failure:
        owner.execute(ModelRequest(), ExecutionContext("request"))
    assert failure.value.evidence.outcome_known and failure.value.close_pending
    assert not owner.ready and owner.state is LifecycleState.FAILED
    owner.close()


def test_standard_backend_admission_rejection_is_not_a_model_failure():
    def execute(*_):
        raise BackendAdmissionError("capacity full")

    owner = ModelRuntimeHandle(SimpleNamespace(execute=execute))
    owner.load()
    with pytest.raises(ExecutionFailure):
        owner.execute(ModelRequest(), ExecutionContext("request"))
    assert owner.ready and owner.health.failure_count == 0
    owner.close()


@pytest.mark.parametrize("with_stream", [False, True])
def test_cancellation_after_drain_wait_started_releases_waiter_before_work_finishes(monkeypatch, with_stream):
    entered, release, draining = threading.Event(), threading.Event(), threading.Event()

    def execute(*_):
        entered.set()
        assert release.wait(3)
        return {}

    owner = streaming(step=execute) if with_stream else ModelRuntimeHandle(SimpleNamespace(execute=execute))
    if not with_stream:
        owner.load()
    stream = owner.open_stream(ExecutionContext("open")) if with_stream else None
    wait = owner._condition.wait

    def observed_wait(timeout=None):
        draining.set()
        return wait(timeout)

    monkeypatch.setattr(owner._condition, "wait", observed_wait)
    stopping = ExecutionContext("shutdown")

    def close():
        owner.close(stopping)

    with ThreadPoolExecutor(max_workers=2) as pool:
        work = (
            pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("work"))
            if with_stream
            else pool.submit(owner.execute, ModelRequest(), ExecutionContext("work"))
        )
        try:
            assert entered.wait(1)
            cleanup = pool.submit(close)
            assert draining.wait(1)
            stopping.cancellation_token.cancel()
            with pytest.raises(ExecutionFailure):
                cleanup.result(1)
            assert not owner.ready and not owner.closed
        finally:
            release.set()
        work.result(1)
    owner.close()


def test_managed_drain_requires_authority_and_excludes_execute_and_close():
    entered, release = threading.Event(), threading.Event()

    def drain():
        entered.set()
        assert release.wait(2)

    owner = ModelRuntimeHandle(SimpleNamespace(drain_uncertain_operations=drain))
    owner.load()
    authority = owner.bind_execution_policy(
        SimpleNamespace(
            ready=True,
            execute=lambda *_: None,
            reset=lambda *_: None,
            close=lambda *_: None,
            cancel=lambda *_args, **_kwargs: None,
        )
    )
    with pytest.raises(ExecutionFailure, match="dispatch"):
        owner.drain_managed(ExecutionContext("unauthorized"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        draining = pool.submit(owner.drain_managed, ExecutionContext("drain"), _dispatch_key=authority)
        try:
            assert entered.wait(1)
            with pytest.raises(ExecutionFailure):
                owner._execute_local(ModelRequest(), ExecutionContext("execute"), _dispatch_key=authority)
            with pytest.raises(ExecutionFailure):
                owner._close_local(deadline=Deadline.after(0), _dispatch_key=authority)
        finally:
            release.set()
        draining.result(1)
    owner._close_local(_dispatch_key=authority)
    assert owner.closed


@pytest.mark.parametrize("all_supported", [False, True])
def test_managed_drain_requires_completion_from_every_native_session(all_supported):
    calls = []
    caps = BackendCapabilities()
    acl = SimpleNamespace(name="acl", capabilities=caps, drain_uncertain_operations=lambda: calls.append("acl"))
    other = SimpleNamespace(execute_role=lambda *_: {})
    if all_supported:
        other.drain_uncertain_operations = lambda: calls.append("other")

    def unknown(*_):
        raise BackendInferenceError("native completion unknown", operation_started=True, outcome_known=False)

    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=unknown),
            session=acl,
            role_assemblies={"acl": acl, "other": other},
            providers=RuntimeProviders(object(), ResourceDomainAdmissions()),
            resettable=True,
        )
    )
    owner.load()
    policy = SimpleNamespace(ready=True, cancel=lambda *_args, **_kwargs: None)
    for name in ("execute", "reset", "close"):
        setattr(policy, name, lambda *args, name=name: getattr(owner, f"_{name}_local")(*args, _dispatch_key=authority))
    authority = owner.bind_execution_policy(policy)
    try:
        with pytest.raises(ExecutionFailure):
            owner.execute(ModelRequest(), ExecutionContext("unknown"))
        if all_supported:
            owner.drain_managed(ExecutionContext("drain"), _dispatch_key=authority)
            with owner._admission.admit(Deadline.after(0.02)):
                pass
            assert calls == ["acl", "other"]
        else:
            with pytest.raises(ExecutionFailure, match="backend_drain_unsupported"):
                owner.drain_managed(ExecutionContext("drain"), _dispatch_key=authority)
            with pytest.raises(RuntimeError, match="admission deadline"), owner._admission.admit(Deadline.after(0.02)):
                pytest.fail("incomplete drain returned retained capacity")
            assert not calls
        assert not owner.ready
    finally:
        owner.close()


def test_policy_binding_rejects_an_active_stream_step():
    entered, release = threading.Event(), threading.Event()

    def step(*_):
        entered.set()
        assert release.wait(2)
        return {}

    owner = streaming(step=step)
    stream = owner.open_stream(ExecutionContext("open"))
    policy = SimpleNamespace(
        ready=True,
        **{
            name: lambda *_: None
            for name in (
                "execute",
                "cancel",
                "reset",
                "close",
                "open_stream",
                "step",
                "reset_stream",
                "close_stream",
            )
        },
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("work"))
        try:
            assert entered.wait(1)
            with pytest.raises(RuntimeError, match="binding"):
                owner.bind_execution_policy(policy)
        finally:
            release.set()
        work.result(1)
    owner.close()


def test_admission_constructor_conflict_does_not_leave_an_instance_registered():
    from inference_service.unified_runtime import NativeAdmission

    provider = ResourceDomainAdmissions()
    existing = provider.register("domain", 1)
    caps = SimpleNamespace(resource_domain="domain", resource_domain_limit=2)
    with pytest.raises(BackendAdmissionError):
        NativeAdmission(provider, "instance", caps)
    caps.resource_domain_limit = 1
    admission = NativeAdmission(provider, "instance", caps)
    admission.close()
    existing.close()
    assert not provider._domains and not provider._instances


def test_handle_invalid_configuration_cannot_register_admission():
    provider = ResourceDomainAdmissions()
    session = SimpleNamespace(name="instance", capabilities=BackendCapabilities())
    assembly = RuntimeAssembly(
        runtime_executor=session, session=session, stateful=True, providers=RuntimeProviders(object(), provider)
    )
    with pytest.raises(ValueError, match="stateful request"):
        ModelRuntimeHandle(assembly)
    assert not provider._instances and not assembly.ownership_transferred


def test_second_handle_claim_failure_releases_only_its_own_admission():
    provider = ResourceDomainAdmissions()
    caps = SimpleNamespace(supports_multiple_instances=True)
    session = SimpleNamespace(name="instance", capabilities=caps)
    assembly = RuntimeAssembly(
        runtime_executor=session, session=session, providers=RuntimeProviders(object(), provider)
    )
    owner = ModelRuntimeHandle(assembly)
    with pytest.raises(RuntimeError, match="already been transferred"):
        ModelRuntimeHandle(assembly)
    assert provider._instances["instance"].references == 1
    owner.close()
    assert not provider._instances


@pytest.mark.parametrize("proof", ["admission", "completed", "unknown"])
def test_open_stream_respects_native_evidence_before_recovery(proof):
    from inference_service.unified_runtime import OutcomeEvidence

    blocked = [True]
    error = BackendAdmissionError("capacity busy") if proof == "admission" else RuntimeError("open failed")
    if proof != "admission":
        error.evidence = (
            OutcomeEvidence.completed("open")
            if proof == "completed"
            else OutcomeEvidence.started("open", outcome_known=False)
        )

    def open_stream(_):
        if blocked[0]:
            raise error
        return object()

    owner = streaming(open_stream=open_stream)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.open_stream(ExecutionContext("open-failed"))
        expected = {
            "admission": OutcomeState.NOT_STARTED,
            "completed": OutcomeState.COMPLETED,
            "unknown": OutcomeState.STARTED,
        }[proof]
        assert caught.value.evidence.state is expected
        assert caught.value.evidence.outcome_known is (proof != "unknown")
        assert owner.ready is (proof != "unknown")
        blocked[0] = False
        if proof == "unknown":
            with pytest.raises(ExecutionFailure):
                owner.open_stream(ExecutionContext("after-failure"))
        else:
            owner.open_stream(ExecutionContext("after-rejection"))
    finally:
        owner.close()


def test_public_reset_stream_cannot_bypass_runtime_control():
    owner = streaming()
    try:
        stream = owner.open_stream(ExecutionContext("open"))
        with pytest.raises(TypeError, match="_from_runtime"):
            owner.reset_stream(stream, ExecutionContext("reset"), _from_runtime=True)
        owner.reset_stream(stream, ExecutionContext("public-reset"))
        assert owner.ready
    finally:
        owner.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
@pytest.mark.parametrize("stream_entry", [False, True])
@pytest.mark.parametrize("policy_bound", [False, True])
def test_control_from_active_execution_fails_before_wait_or_state_change(operation, stream_entry, policy_bound):
    def execute(*_):
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(deadline=Deadline.after(0.1))
        assert caught.value.code == "runtime_reentrant"
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert owner.local_ready
        return {}

    owner = (
        streaming(step=execute)
        if stream_entry
        else ModelRuntimeHandle(SimpleNamespace(execute=execute), resettable=True)
    )
    if not stream_entry:
        owner.load()
    if policy_bound:
        policy = SimpleNamespace(ready=True, cancel=lambda *_args, **_kwargs: None)
        for name in ("execute", "reset", "close", "open_stream", "step", "reset_stream", "close_stream"):
            setattr(
                policy, name, lambda *args, name=name: getattr(owner, f"_{name}_local")(*args, _dispatch_key=authority)
            )
        authority = owner.bind_execution_policy(policy)
    try:
        if stream_entry:
            stream = owner.open_stream(ExecutionContext("open"))
            owner.step(stream, ModelRequest(), ExecutionContext("step"))
        else:
            owner.execute(ModelRequest(), ExecutionContext("execute"))
        assert owner.ready and owner.active_executions == 0
    finally:
        owner.close()
