from __future__ import annotations

import threading
import weakref
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from inference_service.unified_runtime import (
    CancellationToken,
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ExecutionFailureFactory,
    LifecycleState,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OutcomeEvidence,
    OutcomeState,
    RecoveryAction,
    RecoveryRequirement,
    RecoveryScope,
    ResultAdapter,
    RuntimeAssembly,
    StreamHandle,
    StreamState,
)


def context(request_id: str = "request-1", *, token: CancellationToken | None = None) -> ExecutionContext:
    return ExecutionContext(request_id, Deadline.unbounded(), token or CancellationToken())


class RecordingComponent:
    def __init__(self, name: str, events: list[str], *, fail_load: bool = False, fail_reset: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail_load = fail_load
        self.fail_reset = fail_reset
        self.close_count = 0
        self.reset_count = 0

    def load(self, _context: object) -> None:
        self.events.append(f"load:{self.name}")
        if self.fail_load:
            raise RuntimeError(f"load:{self.name}")

    def reset(self, _context: ExecutionContext) -> None:
        self.events.append(f"reset:{self.name}")
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError(f"reset:{self.name}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        self.close_count += 1


class EchoExecutor(RecordingComponent):
    def __init__(self, events: list[str]) -> None:
        super().__init__("executor", events)
        self.contexts: list[ExecutionContext] = []

    def execute(self, request: ModelRequest, execution_context: ExecutionContext) -> object:
        self.contexts.append(execution_context)
        return {"outputs": {"value": request.inputs.get("value")}}


@pytest.mark.parametrize("retry_failed_reset", [False, True])
def test_reset_failure_retry_requires_explicit_opt_in(retry_failed_reset):
    executor = EchoExecutor([])
    executor.fail_reset = True
    assembly = RuntimeAssembly(runtime_executor=executor, resettable=True)
    if retry_failed_reset:
        assembly.retry_failed_reset = True
    handle = ModelRuntimeHandle(assembly)
    handle.load()
    try:
        with pytest.raises(ExecutionFailure):
            handle.reset()
        assert handle.state is (LifecycleState.RESET_REQUIRED if retry_failed_reset else LifecycleState.FAILED)
        with pytest.raises(ExecutionFailure):
            handle.execute(ModelRequest({"value": 1}), context())
        executor.fail_reset = False
        if retry_failed_reset:
            handle.reset()
            assert handle.health.ready
        else:
            with pytest.raises(ExecutionFailure):
                handle.reset()
    finally:
        handle.close()


def test_reset_completion_runs_inside_admission_barrier():
    events = []
    executor = EchoExecutor(events)

    def reset_complete():
        assert events[-1] == "reset:executor"
        assert handle.health.state is LifecycleState.RESETTING
        with pytest.raises(ExecutionFailure):
            handle.execute(ModelRequest({"value": 1}), context())
        events.append("reset_complete")

    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            resettable=True,
            reset_complete=reset_complete,
        )
    )
    handle.load()
    try:
        handle.reset()
        assert handle.health.ready
        assert events[-1] == "reset_complete"
    finally:
        handle.close()


@pytest.mark.parametrize("retry_failed_reset", [False, True])
def test_explicit_reload_from_reset_cannot_be_retried(retry_failed_reset):
    executor = EchoExecutor([])
    calls = []

    def reset(_):
        calls.append("reset")
        if len(calls) == 1:
            raise ExecutionFailure(
                "reset_failed",
                "native state requires replacement",
                recovery=RecoveryRequirement("runtime", "reload"),
                evidence=OutcomeEvidence.started("reset", outcome_known=False),
            )

    executor.reset = reset
    owner = ModelRuntimeHandle(
        RuntimeAssembly(runtime_executor=executor, resettable=True, retry_failed_reset=retry_failed_reset)
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure):
            owner.reset()
        assert owner.state is LifecycleState.FAILED
        with pytest.raises(ExecutionFailure):
            owner.reset()
        assert calls == ["reset"] and not owner.ready
        assert owner.health.recovery_requirement.action is RecoveryAction.RELOAD
    finally:
        owner.close()


def test_failed_load_retains_pending_rollback_resources_for_close_retry():
    from inference_service.backends.errors import BackendLifecycleError

    events = []
    provider = RecordingComponent("provider", events)
    resource = RecordingComponent("device", events)
    failing = RecordingComponent("failing", events, fail_load=True)
    unattempted = RecordingComponent("unattempted", events)
    original_close = resource.close

    def pending_close():
        if resource.close_count == 0:
            resource.close_count += 1
            raise BackendLifecycleError("drain pending", close_pending=True)
        original_close()

    resource.close = pending_close
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=EchoExecutor(events),
            owned_components=(provider, resource, failing, unattempted),
        )
    )
    with pytest.raises(ExecutionFailure, match="failed to load"):
        handle.load()
    assert provider.close_count == 0
    handle.close()
    handle.close()
    assert unattempted.close_count == 1
    assert resource.close_count == 2
    assert provider.close_count == 1
    assert failing.close_count == 1


def test_non_pending_cleanup_error_releases_dependencies_once():
    from types import SimpleNamespace

    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.pipeline.errors import PipelineManagerError
    from inference_service.pipeline.manager import InferencePipelineManager

    events = []
    executor = EchoExecutor(events)
    provider = RecordingComponent("provider", events)

    def close():
        executor.close_count += 1
        raise BackendLifecycleError("cleanup failed after release")

    executor.close = close
    handle = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=executor, owned_components=(provider, executor)))
    pipeline = SimpleNamespace(pipeline_id="policy", load=handle.load, close=handle.close)
    manager = InferencePipelineManager((pipeline,))
    manager.start()
    with pytest.raises(PipelineManagerError):
        manager.close()
    assert handle.health.state is LifecycleState.CLOSED
    manager.close()
    assert executor.close_count == provider.close_count == 1


def test_manager_retries_pending_close():
    from types import SimpleNamespace

    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.pipeline.errors import PipelineManagerError
    from inference_service.pipeline.manager import InferencePipelineManager

    attempts = []

    def close():
        attempts.append("close")
        if len(attempts) == 1:
            raise BackendLifecycleError("drain pending", close_pending=True)

    pipeline = SimpleNamespace(pipeline_id="policy", load=lambda: None, close=close)
    manager = InferencePipelineManager((pipeline,))
    manager.start()
    with pytest.raises(PipelineManagerError):
        manager.close()
    manager.close()
    manager.close()
    assert len(attempts) == 2


@pytest.mark.parametrize("scheduled", [False, True])
def test_stateless_component_reset_skip_requires_opt_in(scheduled):
    from types import SimpleNamespace

    events = []
    executor = EchoExecutor(events)
    component = RecordingComponent("stateless", events, fail_reset=True)
    component.capabilities = SimpleNamespace(stateful=False, resettable=False)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            owned_components=(component, executor),
            resettable=True,
            retry_failed_reset=scheduled,
        )
    )
    handle.load()
    try:
        if scheduled:
            handle.reset()
            assert component.reset_count == 0
            assert handle.health.ready
        else:
            with pytest.raises(ExecutionFailure):
                handle.reset()
            assert component.reset_count == 1
    finally:
        handle.close()


def test_typed_values_are_read_only_and_deadline_is_absolute() -> None:
    inputs = {"value": 4}
    metadata = {"nested": {"source": "test"}}
    request = ModelRequest(inputs, metadata)
    inputs["other"] = 5
    metadata["nested"]["changed"] = True

    assert request.inputs == {"value": 4}
    assert request.metadata["nested"] == {"source": "test"}
    with pytest.raises(TypeError):
        request.inputs["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        request.metadata["new"] = 1  # type: ignore[index]

    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deadline = Deadline.after(2, now=origin)
    assert deadline.remaining_seconds(now=origin + timedelta(seconds=1)) == pytest.approx(1.0)
    assert not deadline.is_expired(now=origin + timedelta(seconds=1))
    assert deadline.is_expired(now=origin + timedelta(seconds=2))

    token = CancellationToken()
    execution_context = ExecutionContext("request-1", deadline, token)
    assert execution_context.cancellation_token is token
    assert execution_context.deadline is deadline


def test_recovery_matrix_and_serialization_omit_local_cause() -> None:
    factory = ExecutionFailureFactory()
    with pytest.raises(ValueError, match="invalid recovery requirement"):
        factory.create(
            "bad",
            "bad recovery",
            scope=RecoveryScope.STREAM,
            action=RecoveryAction.RESET_RUNTIME,
        )

    with pytest.raises(ValueError, match="invalid recovery requirement"):
        RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RESET_RUNTIME)

    cause = RuntimeError("local-only")
    failure = factory.create(
        "backend_async_failure",
        "device outcome is unknown",
        evidence=OutcomeEvidence.started("acl_async", outcome_known=False, state_mutated=True),
        recovery=RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RESET_STREAM),
        recoverable=True,
        cause=cause,
        details={"vendor_code": 17},
    )
    assert isinstance(failure, ExecutionFailure)
    assert failure.cause is cause
    serialized = failure.to_dict()
    assert serialized["recovery"] == {"scope": "stream", "action": "reset_stream"}
    assert "cause" not in serialized
    assert serialized["evidence"]["state"] == "started"


def test_result_adapter_is_success_only_and_validates_outputs() -> None:
    adapter = ResultAdapter(required_outputs=("value",))
    result = adapter.adapt(
        {"outputs": {"value": 3}},
        evidence=OutcomeEvidence.completed("backend"),
        latency=2.5,
    )
    assert isinstance(result, ModelResult)
    assert result.outputs["value"] == 3
    assert result.latency_ms == pytest.approx(2.5)
    assert not hasattr(adapter, "adapt_error")

    with pytest.raises(ValueError, match="missing declared outputs"):
        adapter.adapt({"outputs": {"other": 1}})

    class BadOutputExecutor:
        def execute(self, _request: ModelRequest, _context: ExecutionContext) -> object:
            return {"outputs": {"other": 1}}

        def close(self) -> None:
            return None

    handle = ModelRuntimeHandle(BadOutputExecutor(), result_adapter=adapter)
    handle.load()
    with pytest.raises(ExecutionFailure) as validation_failure:
        handle.execute(ModelRequest({"value": 1}), context("validation"))
    assert validation_failure.value.code == "output_validation_failed"
    assert validation_failure.value.evidence.phase == "output_validation"
    handle.close()


def test_load_rolls_back_in_reverse_order_and_close_is_idempotent() -> None:
    events: list[str] = []
    first = RecordingComponent("first", events)
    second = RecordingComponent("second", events)
    failing = RecordingComponent("failing", events, fail_load=True)
    assembly = RuntimeAssembly(
        runtime_executor=first,
        owned_components=(first, second, failing),
        runtime_id="rollback-runtime",
    )
    handle = ModelRuntimeHandle(assembly)

    with pytest.raises(ExecutionFailure) as error:
        handle.load(context("load"))
    assert error.value.code == "runtime_load_failed"
    assert handle.state is LifecycleState.FAILED
    assert events[-2:] == ["close:second", "close:first"]

    handle.close()
    handle.close()
    assert first.close_count == 1
    assert second.close_count == 1
    assert failing.close_count == 1
    assert handle.state is LifecycleState.CLOSED


def test_lifecycle_barrier_reset_matrix_and_shared_context() -> None:
    events: list[str] = []
    executor = EchoExecutor(events)
    handle = ModelRuntimeHandle(executor, runtime_id="request-runtime")
    with pytest.raises(ExecutionFailure) as not_ready:
        handle.execute(ModelRequest({"value": 1}), context())
    assert not_ready.value.code == "runtime_not_ready"

    handle.load(context("load"))
    token = CancellationToken()
    execution_context = context("request-42", token=token)
    result = handle.execute(ModelRequest({"value": 8}), execution_context)
    assert result.outputs["value"] == 8
    assert executor.contexts == [execution_context]
    assert result.evidence.state is OutcomeState.COMPLETED

    handle.reset()
    assert handle.state is LifecycleState.READY
    assert executor.reset_count == 0
    handle.close()

    with pytest.raises(ValueError, match="stateful request"):
        ModelRuntimeHandle(EchoExecutor([]), stateful=True, state_scope="request")


def test_cancellation_and_expired_deadline_fail_before_backend_start() -> None:
    events: list[str] = []
    executor = EchoExecutor(events)
    handle = ModelRuntimeHandle(executor)
    handle.load()

    token = CancellationToken()
    token.cancel("caller")
    with pytest.raises(ExecutionFailure) as canceled:
        handle.execute(ModelRequest({"value": 1}), context("cancelled", token=token))
    assert canceled.value.code == "request_canceled"
    assert canceled.value.evidence.state is OutcomeState.NOT_STARTED
    assert not executor.contexts

    expired = ExecutionContext("expired", Deadline.at(datetime.now(timezone.utc) - timedelta(seconds=1)))
    with pytest.raises(ExecutionFailure) as timed_out:
        handle.execute(ModelRequest({"value": 1}), expired)
    assert timed_out.value.code == "deadline_exceeded"
    assert timed_out.value.evidence.state is OutcomeState.NOT_STARTED
    handle.close()


def test_control_operations_drain_active_requests_and_stop_admission() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingExecutor(EchoExecutor):
        def execute(self, request: ModelRequest, execution_context: ExecutionContext) -> object:
            self.contexts.append(execution_context)
            entered.set()
            release.wait(2)
            return {"outputs": {"value": request.inputs["value"]}}

    executor = BlockingExecutor([])
    handle = ModelRuntimeHandle(executor, resettable=True)
    handle.load()
    result: list[ModelResult] = []

    def run() -> None:
        result.append(handle.execute(ModelRequest({"value": 9}), context("active")))

    execution_thread = threading.Thread(target=run)
    execution_thread.start()
    assert entered.wait(1)

    reset_done = threading.Event()

    def reset() -> None:
        handle.reset()
        reset_done.set()

    reset_thread = threading.Thread(target=reset)
    reset_thread.start()
    assert not reset_done.wait(0.05)
    with pytest.raises(ExecutionFailure) as blocked:
        handle.execute(ModelRequest({"value": 1}), context("blocked-during-reset"))
    assert blocked.value.code in {"runtime_not_ready", "admission_rejected"}

    release.set()
    execution_thread.join(timeout=2)
    reset_thread.join(timeout=2)
    assert result[0].outputs["value"] == 9
    assert reset_done.is_set()
    assert handle.state is LifecycleState.READY
    handle.close()


def test_policy_requests_reject_duplicate_ids_without_overwriting_cancellation() -> None:
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()

    class Executor(EchoExecutor):
        def execute(self, request, execution_context):
            entered.set()
            assert release.wait(2)
            return super().execute(request, execution_context)

    handle = ModelRuntimeHandle(Executor([]), max_active_executions=2)
    handle.load()
    authority = handle.bind_execution_policy(
        SimpleNamespace(
            ready=True,
            cancel=lambda *_args, **_kwargs: None,
            execute=lambda request, context: handle._execute_local(request, context, _dispatch_key=authority),
            reset=lambda context: handle._reset_local(context, _dispatch_key=authority),
            close=lambda context: handle._close_local(context, _dispatch_key=authority),
        )
    )
    original = context("same")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(handle.execute, ModelRequest({"value": 1}), original)
            try:
                assert entered.wait(1)
                with pytest.raises(ExecutionFailure) as rejected:
                    handle.execute(ModelRequest({"value": 2}), context("same"))
                assert rejected.value.code == "request_conflict"
                handle.cancel("same")
                assert original.cancellation_token.is_cancelled
            finally:
                release.set()
            with pytest.raises(ExecutionFailure):
                first.result(timeout=1)
        assert handle._active_executions == 0
    finally:
        handle.close()


@pytest.mark.parametrize("cancel_remaining", [False, True])
def test_parallel_direct_requests_keep_duplicate_ids_and_all_cancel_owners(cancel_remaining):
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Barrier(3)
    release = [threading.Event(), threading.Event()]
    contexts = [context("same"), context("same")]

    def execute(request, execution_context):
        entered.wait(timeout=2)
        assert release[request.inputs["index"]].wait(2)
        return request.inputs

    handle = ModelRuntimeHandle(SimpleNamespace(execute=execute), max_active_executions=2)
    handle.load()
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(handle.execute, ModelRequest({"index": i}), contexts[i]) for i in range(2)]
            try:
                entered.wait(timeout=2)
                release[0].set()
                assert futures[0].result(timeout=1).outputs == {"index": 0}
                if cancel_remaining:
                    handle.cancel("same")
                    assert contexts[1].cancellation_token.is_cancelled
                    assert not contexts[0].cancellation_token.is_cancelled
                release[1].set()
                if cancel_remaining:
                    with pytest.raises(ExecutionFailure) as failure:
                        futures[1].result(timeout=1)
                    assert failure.value.evidence.state is OutcomeState.COMPLETED
                else:
                    assert futures[1].result(timeout=1).outputs == {"index": 1}
                assert handle.active_executions == 0 and handle.ready
            finally:
                for event in release:
                    event.set()
    finally:
        handle.close()


class FakeStreaming:
    def __init__(self) -> None:
        self.next_id = 0
        self.states: dict[str, int] = {}
        self.step_calls: list[str] = []
        self.reset_calls: list[str] = []
        self.close_calls: list[str] = []
        self.block_started = threading.Event()
        self.allow_step = threading.Event()
        self.fail_unknown = False

    def load(self, _context: object) -> None:
        return None

    def open_stream(self, _context: ExecutionContext) -> StreamHandle:
        self.next_id += 1
        handle = StreamHandle(f"stream-{self.next_id}")
        self.states[handle.stream_id] = 0
        return handle

    def step(self, stream_handle: StreamHandle, _request: ModelRequest, _context: ExecutionContext) -> object:
        self.step_calls.append(stream_handle.stream_id)
        self.block_started.set()
        if self.allow_step.is_set():
            self.allow_step.wait(1)
        if self.fail_unknown:
            error = RuntimeError("async device failure")
            error.operation_started = True  # type: ignore[attr-defined]
            error.outcome_known = False  # type: ignore[attr-defined]
            error.state_mutated = True  # type: ignore[attr-defined]
            error.phase = "acl_async"  # type: ignore[attr-defined]
            raise error
        self.states[stream_handle.stream_id] += 1
        return {"outputs": {"count": self.states[stream_handle.stream_id]}}

    def reset_stream(self, stream_handle: StreamHandle, _context: ExecutionContext) -> None:
        self.reset_calls.append(stream_handle.stream_id)
        self.states[stream_handle.stream_id] = 0

    def close_stream(self, stream_handle: StreamHandle, _context: ExecutionContext) -> None:
        self.close_calls.append(stream_handle.stream_id)
        self.states.pop(stream_handle.stream_id, None)

    def close(self) -> None:
        return None


def stream_handle(
    *, mode: str = "per_stream", limit: int = 2, resettable: bool = True
) -> tuple[ModelRuntimeHandle, FakeStreaming]:
    streaming = FakeStreaming()
    handle = ModelRuntimeHandle(
        EchoExecutor([]),
        streaming_runtime=streaming,
        state_scope="stream",
        state_bank_mode=mode,
        max_open_streams=limit,
        stateful=True,
        resettable=resettable,
        runtime_id="stream-runtime",
    )
    handle.load()
    return handle, streaming


def test_stream_limits_identity_close_and_reset_are_scoped() -> None:
    handle, streaming = stream_handle()
    first = handle.open_stream(context("open-1"))
    second = handle.open_stream(context("open-2"))
    assert handle.diagnostics().open_streams == 2

    with pytest.raises(ExecutionFailure) as capacity:
        handle.open_stream(context("open-3"))
    assert capacity.value.code == "stream_capacity_exhausted"

    assert handle.step(first, ModelRequest({"x": 1}), context("step-1")).outputs["count"] == 1
    assert handle.step(second, ModelRequest({"x": 1}), context("step-2")).outputs["count"] == 1
    handle.reset_stream(first, context("reset-1"))
    assert handle.step(first, ModelRequest({"x": 1}), context("step-3")).outputs["count"] == 1
    assert streaming.reset_calls == [first.stream_id]

    handle.reset()
    assert streaming.reset_calls == [first.stream_id, first.stream_id, second.stream_id]
    assert handle.state is LifecycleState.READY
    assert handle.step(first, ModelRequest({"x": 1}), context("step-after-runtime-reset")).outputs["count"] == 1

    handle.close_stream(first, context("close-1"))
    handle.close_stream(first, context("close-1-again"))
    assert streaming.close_calls == [first.stream_id]
    with pytest.raises(ExecutionFailure) as closed:
        handle.step(first, ModelRequest({"x": 1}), context("step-closed"))
    assert closed.value.code == "stream_closed"
    with pytest.raises(ExecutionFailure) as unknown:
        handle.step(StreamHandle("missing"), ModelRequest({"x": 1}), context("step-missing"))
    assert unknown.value.code == "stream_not_found"
    handle.close()


def test_closed_stream_history_releases_native_state_and_never_reuses_public_ids() -> None:
    class NativeStream:
        stream_id = "recycled-native-id"

    handle, streaming = stream_handle()
    native_refs = []

    def open_native(_context):
        native = NativeStream()
        native_refs.append(weakref.ref(native))
        streaming.states[native.stream_id] = 0
        return native

    streaming.open_stream = open_native
    public_handles = []
    try:
        for index in range(300):
            stream = handle.open_stream(context(f"open-{index}"))
            public_handles.append(stream)
            assert handle.step(stream, ModelRequest(), context()).outputs["count"] == 1
            handle.close_stream(stream)
            assert native_refs[-1]() is None

        assert len({stream.stream_id for stream in public_handles}) == 300
        assert len(handle.stream_diagnostics()) == 128
        assert handle.open_stream_count == 0
        oldest = public_handles[0]
        handle.close_stream(oldest)
        with pytest.raises(ExecutionFailure, match="stream_closed"):
            handle.step(oldest, ModelRequest(), context())
        with pytest.raises(ExecutionFailure, match="stream_closed"):
            handle.reset_stream(oldest, context())
        with pytest.raises(ExecutionFailure, match="stream_not_found"):
            handle.close_stream(oldest.stream_id)
        with pytest.raises(ExecutionFailure, match="stream_not_found"):
            handle.close_stream(StreamHandle(public_handles[-1].stream_id))
        assert len(streaming.close_calls) == 300
    finally:
        handle.close()


def test_closed_history_eviction_keeps_failed_stream_owned_until_close_confirmation() -> None:
    handle, streaming = stream_handle(limit=2)
    pending = handle.open_stream(context())
    close = streaming.close_stream

    def fail_pending(stream, execution_context):
        if stream.stream_id == pending.stream_id:
            raise RuntimeError("native close outcome unknown")
        close(stream, execution_context)

    streaming.close_stream = fail_pending
    try:
        with pytest.raises(ExecutionFailure) as failed:
            handle.close_stream(pending)
        assert failed.value.close_pending
        for _ in range(300):
            stream = handle.open_stream(context())
            handle.close_stream(stream)
        diagnostics = handle.stream_diagnostics()
        assert len(diagnostics) == 129
        assert any(item.stream_id == pending.stream_id and item.state is StreamState.FAILED for item in diagnostics)
        assert pending.stream_id in streaming.states
        assert handle.open_stream_count == 1
        streaming.close_stream = close
        handle.close_stream(pending)
        assert pending.closed and handle.open_stream_count == 0
        assert not streaming.states
    finally:
        streaming.close_stream = close
        handle.close()


@pytest.mark.parametrize("failure_kind", ["canceled", "deadline", "backend"])
def test_failed_stream_close_retains_resources_until_successful_retry(failure_kind) -> None:
    handle, streaming = stream_handle(limit=1)
    stream = handle.open_stream(context("open"))
    close_context = context("close")
    original_close = streaming.close_stream
    if failure_kind == "canceled":
        close_context.cancellation_token.cancel("stop")
    elif failure_kind == "deadline":
        close_context = ExecutionContext("close", Deadline.after(0))
    else:

        def fail_close(*_args):
            raise RuntimeError("native close is pending")

        streaming.close_stream = fail_close
    with pytest.raises(ExecutionFailure) as failed:
        handle.close_stream(stream, close_context)
    assert failed.value.close_pending
    assert failed.value.evidence.outcome_known is (failure_kind != "backend")
    assert handle.open_stream_count == 1
    assert stream.stream_id in streaming.states
    with pytest.raises(ExecutionFailure):
        handle.step(stream, ModelRequest(), context("after-close"))
    with pytest.raises(ExecutionFailure):
        handle.reset_stream(stream, context("reset-after-close"))
    with pytest.raises(ExecutionFailure):
        handle.open_stream(context("replacement"))
    streaming.close_stream = original_close
    handle.close_stream(stream, context("retry-close"))
    assert handle.open_stream_count == 0
    assert stream.stream_id not in streaming.states
    handle.close()


def test_runtime_close_does_not_release_model_before_failed_stream_cleanup() -> None:
    handle, streaming = stream_handle(limit=1)
    stream = handle.open_stream(context("open"))
    original_close = streaming.close_stream
    released = []
    streaming.close = lambda: released.append("model")

    def fail_close(*_args):
        raise RuntimeError("native close is pending")

    streaming.close_stream = fail_close
    with pytest.raises(ExecutionFailure) as failed:
        handle.close()
    assert failed.value.close_pending
    assert not handle.closed and not released
    assert stream.stream_id in streaming.states
    streaming.close_stream = original_close
    handle.close()
    assert handle.closed and released == ["model"]
    assert not streaming.states


def test_same_stream_reentrancy_does_not_block_other_isolated_stream() -> None:
    handle, streaming = stream_handle(limit=2)
    first = handle.open_stream(context("open-1"))
    second = handle.open_stream(context("open-2"))
    entered = threading.Event()
    release = threading.Event()

    original_step = streaming.step

    def blocking_step(stream: StreamHandle, request: ModelRequest, execution_context: ExecutionContext) -> object:
        entered.set()
        release.wait(1)
        return original_step(stream, request, execution_context)

    streaming.step = blocking_step  # type: ignore[method-assign]
    first_error: list[BaseException] = []

    def run_first() -> None:
        try:
            handle.step(first, ModelRequest({"x": 1}), context("first"))
        except BaseException as exc:  # pragma: no cover - diagnostic if the test fails
            first_error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(1)
    with pytest.raises(ExecutionFailure) as reentrant:
        handle.step(first, ModelRequest({"x": 1}), context("reentrant"))
    assert reentrant.value.code == "stream_reentrant"

    release.set()
    thread.join(timeout=2)
    assert not first_error
    assert handle.step(second, ModelRequest({"x": 1}), context("second")).outputs["count"] == 1
    handle.close()


def test_started_unknown_failure_isolated_to_stream_or_runtime_bank() -> None:
    handle, streaming = stream_handle(mode="per_stream", limit=2)
    first = handle.open_stream(context("open-1"))
    second = handle.open_stream(context("open-2"))
    streaming.fail_unknown = True
    with pytest.raises(ExecutionFailure) as unknown:
        handle.step(first, ModelRequest({"x": 1}), context("unknown-1"))
    assert unknown.value.evidence.state is OutcomeState.STARTED
    assert unknown.value.evidence.outcome_known is False
    assert unknown.value.recovery == RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RESET_STREAM)
    assert handle.state is LifecycleState.READY
    assert handle.stream_diagnostics(first).state is StreamState.RESET_REQUIRED
    assert handle.diagnostics().to_dict()["streams"][0]["recovery_requirement"] == {
        "scope": "stream",
        "action": "reset_stream",
    }

    streaming.fail_unknown = False
    assert handle.step(second, ModelRequest({"x": 1}), context("safe-stream")).outputs["count"] == 1
    handle.reset_stream(first, context("recover-1"))
    assert handle.step(first, ModelRequest({"x": 1}), context("recovered-1")).outputs["count"] == 1
    handle.close()

    exclusive, exclusive_streaming = stream_handle(mode="runtime_exclusive", limit=1)
    only = exclusive.open_stream(context("open-exclusive"))
    exclusive_streaming.fail_unknown = True
    with pytest.raises(ExecutionFailure) as exclusive_error:
        exclusive.step(only, ModelRequest({"x": 1}), context("unknown-exclusive"))
    assert exclusive.state is LifecycleState.RESET_REQUIRED
    assert exclusive_error.value.recovery == RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RESET_RUNTIME)
    with pytest.raises(ExecutionFailure) as blocked:
        exclusive.open_stream(context("blocked"))
    assert blocked.value.code == "recovery_required"
    exclusive_streaming.fail_unknown = False
    with pytest.raises(ExecutionFailure, match="recovery_required"):
        exclusive.reset_stream(only, context("recover-exclusive"))
    exclusive.reset(context("recover-runtime"))
    assert exclusive.state is LifecycleState.READY
    assert exclusive.step(only, ModelRequest({"x": 1}), context("after-exclusive-reset")).outputs["count"] == 1
    exclusive.close()


def test_stateful_nonresettable_stream_fails_explicitly_without_claiming_reset() -> None:
    handle, _streaming = stream_handle(resettable=False)
    stream = handle.open_stream(context("open"))
    with pytest.raises(ExecutionFailure) as unsupported:
        handle.reset_stream(stream, context("reset"))
    assert unsupported.value.code == "reset_unsupported"
    assert handle.stream_diagnostics(stream).state is StreamState.OPEN
    handle.close_stream(stream, context("close"))
    handle.close()
