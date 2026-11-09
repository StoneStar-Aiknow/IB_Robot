import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.ascend.model import AclModel
from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("invalid", [None, "input_shape", "output_shape", "output_dtype", "missing_output"])
def test_role_execution_uses_shared_validation_and_capture(isolated, invalid):
    from inference_manifest import CompiledDeployment
    from inference_service.backends.types import BackendState
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.unified_runtime import ExecutionContext, ModelRequest

    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
            "execution": ["encoder"],
            "artifacts": {"encoder": {"path": "encoder.om", "format": "om"}},
            "bindings": {"encoder": _bindings()},
        }
    )
    captured = []
    inputs = {"input": np.ones((1, 1) if invalid == "input_shape" else (1,), dtype=np.float32)}
    output = np.ones(
        (1, 1) if invalid == "output_shape" else (1,), dtype=np.float64 if invalid == "output_dtype" else np.float32
    )
    session = AscendOmModelSession(
        runtime_manager=object(), diagnostic_capture=lambda name, value: captured.append(name)
    )
    session._context = SimpleNamespace(deployment=deployment)
    session._state = BackendState.READY
    session._models = {
        "encoder": SimpleNamespace(
            execute=lambda *args, **kwargs: {} if invalid == "missing_output" else {0: output}, close=lambda: None
        )
    }
    if isolated:
        # Exercise the actual isolated submission branch, including its Future.
        def submit(*args, **kwargs):
            future = Future()
            future.set_result({} if invalid == "missing_output" else {0: output})
            return SimpleNamespace(completion=future)

        session._models["encoder"].submit_async_isolated = submit
        session._request_stream = lambda *args: "stream"
    try:
        if invalid:
            with pytest.raises(BackendInferenceError, match="shape|dtype|semantics"):
                session.execute_role(
                    "encoder", inputs, ModelRequest(inputs), ExecutionContext("role"), isolated=isolated
                )
        else:
            result = session.execute_role(
                "encoder", inputs, ModelRequest(inputs), ExecutionContext("role"), isolated=isolated
            )
            np.testing.assert_array_equal(result["output"], output)
            assert captured == ["encoder_in_input", "encoder_out_output"]
    finally:
        session.close()


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_iterative_role_reuses_key_before_future_callbacks_finish(monkeypatch, cleanup_error):
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.scheduler.operations import OperationIdentity, OperationKind, OperationRegistry
    from inference_service.unified_runtime import ExecutionContext

    registry = OperationRegistry(max_records=4, max_waiters_per_operation=1)
    identity = OperationIdentity(str(uuid4()), 1, str(uuid4()), 1, str(uuid4()))
    context = ExecutionContext("iterative-request")
    release_callbacks = threading.Event()
    callback_started = threading.Event()
    caller_thread = threading.current_thread()
    finish = registry.finish
    threads = []
    operation_ids = []

    def delayed_finish(*args, **kwargs):
        if threading.current_thread() is not caller_thread:
            callback_started.set()
            assert release_callbacks.wait(5)
        return finish(*args, **kwargs)

    monkeypatch.setattr(registry, "finish", delayed_finish)

    def operation_factory(role):
        operation, created = registry.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=(1, context.request_id, role),
            identity=identity,
            deadline_mono_ns=10**30,
            waiter_id=context.request_id,
        )
        assert created
        operation_ids.append(operation.operation_id)
        return operation, registry

    class Completion(Future):
        def add_done_callback(self, fn):
            super().add_done_callback(fn)
            if cleanup_error:
                error = BackendInferenceError("cleanup pending", operation_started=True, outcome_known=True)
                error.close_pending = True
                worker = threading.Thread(target=self.set_exception, args=(error,))
            else:
                worker = threading.Thread(target=self.set_result, args=({0: "actions"},))
            threads.append(worker)
            worker.start()

    session = AscendOmModelSession(runtime_manager=object())
    session._models = {
        "action_expert": SimpleNamespace(
            submit_async_isolated=lambda *args, **kwargs: SimpleNamespace(completion=Completion())
        )
    }
    try:
        for _ in range(2):
            callback_started.clear()

            def run():
                return session._submit_isolated_role(
                    "action_expert",
                    {},
                    context,
                    read_outputs=True,
                    stream="stream",
                    operation_factory=operation_factory,
                )

            if cleanup_error:
                with pytest.raises(BackendInferenceError, match="cleanup pending"):
                    run()
            else:
                assert run() == {0: "actions"}
            assert callback_started.wait(5)
            assert registry.find((1, context.request_id, "action_expert")) is None
        assert len(set(operation_ids)) == 2
    finally:
        release_callbacks.set()
        for worker in threads:
            worker.join(5)
            assert not worker.is_alive()
    assert all(registry.get(operation_id) is None for operation_id in operation_ids)


def _bindings():
    return ArtifactBindings(
        inputs=(TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,)),),
        outputs=(TensorBinding(semantic="output", index=0, dtype="float32", shape=(1,)),),
    )


def test_completion_callback_close_does_not_wait_for_queued_completion(model_probe):
    model, state, _ = model_probe
    entered, release, callback_done = (threading.Event() for _ in range(3))
    errors = []

    def synchronize(_):
        entered.set()
        assert release.wait(2)
        return 0

    model._acl.rt.synchronize_stream = synchronize
    first = model.submit_async_isolated({}, stream="first")
    assert entered.wait(1)
    second = model.submit_async_isolated({}, stream="second")

    def close(_):
        try:
            model.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            callback_done.set()

    first.completion.add_done_callback(close)
    release.set()
    assert callback_done.wait(1)
    assert len(errors) == 1 and errors[0].close_pending
    assert not model._closed
    assert second.completion.result(1) == {0: "result"}
    model.close()
    assert model._closed


def test_normal_output_copy_does_not_quarantine_another_isolated_submit(model_probe, monkeypatch):
    from inference_service.backends.ascend.model import AclTensorDescriptor, _DatasetBuffer

    model, state, _ = model_probe
    entered, release = threading.Event(), threading.Event()
    frees = []
    model.output_descriptors = (AclTensorDescriptor(0, "output", np.dtype("float32"), (1,), 4),)

    def copy(*_):
        entered.set()
        assert release.wait(2)
        return 0

    model._acl.rt.malloc_host = lambda _: (object(), 0)
    model._acl.rt.memcpy = copy
    model._acl.rt.free_host = lambda ptr: (frees.append(ptr), 0)[1]
    model._acl.util = SimpleNamespace(ptr_to_bytes=lambda *_: np.ones(1, np.float32).tobytes())
    monkeypatch.setattr(model, "_read_dataset_outputs", AclModel._read_dataset_outputs.__get__(model))
    monkeypatch.setattr(model, "_create_dataset", lambda *_: (object(), [_DatasetBuffer(object(), None, 4, False)]))
    first = model.submit_async_isolated({}, stream="first")
    try:
        assert entered.wait(1)
        second = model.submit_async_isolated({}, stream="second")
        assert state.submitted == 2
    finally:
        release.set()
    first.completion.result(1)
    second.completion.result(1)
    assert len(frees) == 2 and not model._quarantined_host_buffers


@pytest.fixture
def model_probe(monkeypatch):
    state = SimpleNamespace(submit=0, sync=0, destroyed=[], submitted=0)

    def submit(*_args):
        state.submitted += 1
        return state.submit

    acl = SimpleNamespace(
        mdl=SimpleNamespace(execute_async=submit),
        rt=SimpleNamespace(synchronize_stream=lambda _stream: state.sync),
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None)
    model = AclModel(lease, "encoder", Path("fake.om"), _bindings(), async_execution=True)
    created = []

    def create(*_args):
        dataset = object()
        created.append(dataset)
        return dataset, []

    monkeypatch.setattr(model, "_create_dataset", create)
    monkeypatch.setattr(
        model, "_destroy_dataset", lambda dataset, buffers: state.destroyed.append(dataset) if dataset else None
    )
    monkeypatch.setattr(model, "_prepare_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_read_dataset_outputs", lambda *args: {0: "result"})
    yield model, state, created
    state.sync = 0
    model.close()


@pytest.mark.parametrize("failure_at", ["output_dataset", "input_copy"])
def test_pre_submit_cleanup_failure_quarantines_handle_with_known_outcome(model_probe, monkeypatch, failure_at):
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    model, state, created = model_probe
    create, destroy = model._create_dataset, model._destroy_dataset
    failed = [False]

    def create_dataset(*args):
        if failure_at == "output_dataset" and created:
            raise RuntimeError("output allocation failed")
        return create(*args)

    def prepare(*args, **kwargs):
        raise RuntimeError("input copy failed")

    def destroy_dataset(*args):
        if not failed[0]:
            failed[0] = True
            raise RuntimeError("buffer release failed")
        return destroy(*args)

    monkeypatch.setattr(model, "_create_dataset", create_dataset)
    monkeypatch.setattr(model, "_prepare_inputs", prepare)
    monkeypatch.setattr(model, "_destroy_dataset", destroy_dataset)
    owner = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: model.submit_async_isolated({}, stream="stream"),
            close=model.close,
        )
    )
    owner.load()
    with pytest.raises(ExecutionFailure) as failure:
        owner.execute(ModelRequest(), ExecutionContext("first"))
    assert failure.value.evidence.state is OutcomeState.NOT_STARTED
    assert failure.value.evidence.outcome_known and failure.value.close_pending
    assert not owner.ready and state.submitted == 0
    with pytest.raises(ExecutionFailure):
        owner.execute(ModelRequest(), ExecutionContext("second"))
    owner.close()
    assert owner.closed and set(state.destroyed) == set(created)


@pytest.mark.parametrize("failure_at", ["input_dataset", "output_dataset"])
@pytest.mark.parametrize("failure_kind", ["malloc", "create_dataset"])
def test_pre_submit_allocation_failure_with_clean_rollback_allows_retry(
    model_probe, monkeypatch, failure_at, failure_kind
):
    from inference_service.backends.ascend.model import AclTensorDescriptor
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    model, state, _ = model_probe
    descriptor = AclTensorDescriptor(index=0, name="value", dtype="float32", shape=(1,), size=4)
    model.input_descriptors = model.output_descriptors = (descriptor,)
    model._create_dataset = AclModel._create_dataset.__get__(model)
    model._destroy_dataset = AclModel._destroy_dataset.__get__(model)
    allocated, released = [], []
    datasets = []
    failing = [True]

    def create():
        dataset = object()
        datasets.append(dataset)
        target = len(datasets) == (1 if failure_at == "input_dataset" else 2)
        return None if failing[0] and target and failure_kind == "create_dataset" else dataset

    def malloc(*_):
        target = len(datasets) == (1 if failure_at == "input_dataset" else 2)
        if failing[0] and target and failure_kind == "malloc":
            return None, 1
        pointer = object()
        allocated.append(pointer)
        return pointer, 0

    model._acl.mdl.create_dataset = create
    model._acl.mdl.add_dataset_buffer = lambda *_: 0
    model._acl.mdl.destroy_dataset = lambda dataset: (released.append(dataset), 0)[1]
    model._acl.rt.malloc = malloc
    model._acl.rt.free = lambda pointer: (released.append(pointer), 0)[1]
    model._acl.create_data_buffer = lambda *_: object()
    model._acl.destroy_data_buffer = lambda *_: 0
    owner = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: model.submit_async_isolated({}, stream="s").completion.result(1),
            close=model.close,
        )
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("allocation"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED and caught.value.evidence.outcome_known
        assert not caught.value.close_pending and state.submitted == 0
        assert owner.ready and not model._quarantined_datasets
        assert all(pointer in released for pointer in allocated)
        expected_datasets = datasets[:-1] if failure_kind == "create_dataset" else datasets
        assert all(dataset in released for dataset in expected_datasets)
        failing[0] = False
        owner.execute(ModelRequest(), ExecutionContext("valid", Deadline.after(1)))
        assert owner.ready and state.submitted == 1
    finally:
        owner.close()


@pytest.mark.parametrize("failure", ["submit", "sync"])
def test_uncertain_execution_retains_datasets_until_trusted_drain(model_probe, failure):
    model, state, created = model_probe
    setattr(state, failure, 1)
    with pytest.raises(BackendInferenceError) as error:
        model.submit_async_isolated({}, stream="stream").completion.result(timeout=2)
    assert error.value.operation_started and not error.value.outcome_known
    assert state.destroyed == []
    assert len(created) == 2
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.submit_async_isolated({}, stream="stream")
    assert state.submitted == 1
    state.sync = 1
    with pytest.raises(RuntimeError, match="synchronize_stream"):
        model.close()
    assert state.destroyed == []
    assert not model._closed
    state.sync = 0
    model.close()
    assert set(state.destroyed) == set(created)
    assert model._closed


def test_output_read_failure_is_known_and_releases_datasets(model_probe, monkeypatch):
    model, state, created = model_probe

    def fail(*args):
        raise ValueError("invalid output")

    monkeypatch.setattr(model, "_read_dataset_outputs", fail)
    with pytest.raises(BackendInferenceError) as error:
        model.submit_async_isolated({}, stream="stream").completion.result(timeout=2)
    assert error.value.operation_started and error.value.outcome_known
    assert set(state.destroyed) == set(created)


def test_queued_acl_completion_cannot_be_cancelled_after_device_submission(model_probe, monkeypatch):
    model, state, created = model_probe
    entered, release = threading.Event(), threading.Event()
    synced = []

    def occupy_worker():
        entered.set()
        assert release.wait(2)

    blocker = model._completion_executor.submit(occupy_worker)
    assert entered.wait(1)
    monkeypatch.setattr(model._acl.rt, "synchronize_stream", lambda stream: synced.append(stream) or 0)
    try:
        completion = model.submit_async_isolated({}, stream="queued").completion
        assert state.submitted == 1
        assert completion.cancel() is False
        assert not completion.done() and not state.destroyed
    finally:
        release.set()
    blocker.result(timeout=1)
    assert completion.result(timeout=1) == {0: "result"}
    model.close()
    assert synced == ["queued"] and set(state.destroyed) == set(created)


def test_failed_cleanup_diagnostics_do_not_retain_undeliverable_host_output(model_probe, monkeypatch):
    import gc
    import weakref

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    model, _, _ = model_probe
    references = []
    destroy = model._destroy_dataset

    def outputs(*_):
        value = np.ones(1024, np.float32)
        references.append(weakref.ref(value))
        return {0: value}

    def fail_cleanup(*_):
        raise RuntimeError("cleanup pending")

    monkeypatch.setattr(model, "_read_dataset_outputs", outputs)
    monkeypatch.setattr(model, "_destroy_dataset", fail_cleanup)
    owner = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: model.submit_async_isolated({}, stream="s").completion.result(1),
            reset=lambda: model.drain_quarantined(),
            close=model.close,
        ),
        resettable=True,
    )
    owner.load()
    with pytest.raises(ExecutionFailure) as failure:
        owner.execute(ModelRequest(), ExecutionContext("output"))
    monkeypatch.setattr(model, "_destroy_dataset", destroy)
    owner.reset()
    gc.collect()
    assert references and references[0]() is None
    assert owner.last_failure.to_dict() == failure.value.to_dict() and owner.ready
    owner.close()


def test_acl_close_lock_wait_obeys_lifecycle_budget(model_probe):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    model, _, _ = model_probe
    model._execution_lock.acquire()

    def close():
        with lifecycle_budget(0.02):
            model.close()

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(close)
        try:
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.5)
            assert not model._closed
        finally:
            model._execution_lock.release()
    model.close()
    assert model._closed


@pytest.mark.parametrize("output_fails", [False, True])
def test_completed_request_retains_failed_cleanup_without_double_release(model_probe, monkeypatch, output_fails):
    model, state, created = model_probe
    destroy = model._destroy_dataset
    failed = False

    def release(dataset, buffers):
        nonlocal failed
        if dataset is created[0] and not failed:
            failed = True
            raise RuntimeError("input cleanup pending")
        destroy(dataset, buffers)

    monkeypatch.setattr(model, "_destroy_dataset", release)
    if output_fails:

        def fail_output(*_):
            raise ValueError("output decode failed")

        monkeypatch.setattr(model, "_read_dataset_outputs", fail_output)
    with pytest.raises(BackendInferenceError) as failure:
        model.submit_async_isolated({}, stream="stream").completion.result(timeout=2)
    assert failure.value.operation_started and failure.value.outcome_known and failure.value.close_pending
    assert failure.value.code == ("async_output_failed" if output_fails else "async_cleanup_pending")
    assert failure.value.details["cleanup_error"] == "input cleanup pending"
    assert state.destroyed == [created[1]]
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.submit_async_isolated({}, stream="stream")
    model.close()
    assert state.destroyed == [created[1], created[0]]


def test_acl_known_completion_cleanup_failure_reaches_handle_health(model_probe, monkeypatch):
    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    model, _state, _created = model_probe
    destroy = model._destroy_dataset

    def fail(*_):
        raise RuntimeError("dataset still owned")

    monkeypatch.setattr(model, "_destroy_dataset", fail)
    executor = SimpleNamespace(
        execute=lambda *_: model.submit_async_isolated({}, stream="stream").completion.result(1), close=model.close
    )
    owner = ModelRuntimeHandle(executor)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.execute(ModelRequest(), ExecutionContext("cleanup"))
        assert failure.value.evidence.outcome_known and failure.value.close_pending
        assert not owner.ready
    finally:
        monkeypatch.setattr(model, "_destroy_dataset", destroy)
        owner.close()


def test_acl_completion_wait_consumes_close_budget_without_releasing_active_datasets(model_probe, monkeypatch):
    from inference_service.unified_runtime.errors import DeadlineExceeded
    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    model, state, created = model_probe
    entered, release = threading.Event(), threading.Event()

    def read(*_):
        entered.set()
        assert release.wait(2)
        return {0: "complete"}

    monkeypatch.setattr(model, "_read_dataset_outputs", read)
    future = model.submit_async_isolated({}, stream="stream").completion
    try:
        assert entered.wait(1)
        with lifecycle_budget(0.01), pytest.raises(DeadlineExceeded):
            model.close()
        assert not model._closed and not state.destroyed
    finally:
        release.set()
    assert future.result(timeout=1) == {0: "complete"}
    model.close()
    assert model._closed and set(state.destroyed) == set(created)


@pytest.mark.parametrize("failure_at", ["create_buffer", "add_buffer"])
def test_partial_dataset_creation_retains_failed_cleanup(monkeypatch, failure_at):
    from inference_service.backends.ascend.model import AclTensorDescriptor

    released = []
    free_attempts = []

    def free(pointer):
        free_attempts.append(pointer)
        if len(free_attempts) == 1:
            return 1
        released.append(("device", pointer))
        return 0

    acl = SimpleNamespace(
        rt=SimpleNamespace(malloc=lambda *args: (10, 0), free=free),
        create_data_buffer=lambda *args: None if failure_at == "create_buffer" else 20,
        destroy_data_buffer=lambda value: (released.append(("buffer", value)), 0)[1],
        mdl=SimpleNamespace(
            create_dataset=lambda: 30,
            add_dataset_buffer=lambda *args: 1,
            destroy_dataset=lambda value: (released.append(("dataset", value)), 0)[1],
        ),
    )
    model = AclModel(SimpleNamespace(acl=acl, bind_current_thread=lambda: None), "test", Path("x.om"), _bindings())
    descriptor = AclTensorDescriptor(index=0, name="input", dtype="float32", shape=(1,), size=4)
    with pytest.raises(BackendInferenceError, match="acl.rt.free") as failure:
        model._create_dataset((descriptor,), {})
    assert failure.value.code == "dataset_cleanup_pending"
    assert failure.value.close_pending and failure.value.outcome_known
    model.close()
    model.close()
    expected = [("device", 10), ("dataset", 30)]
    if failure_at == "add_buffer":
        expected.insert(0, ("buffer", 20))
    assert released == expected
    assert free_attempts == [10, 10]


def test_legacy_model_has_no_async_worker_or_lock():
    model = AclModel(SimpleNamespace(acl=object()), "legacy", Path("fake.om"), _bindings())
    assert model._execution_lock is None
    assert model._completion_executor is None
    with pytest.raises(BackendInferenceError, match="disabled"):
        model.submit_async_isolated({}, stream="stream")


@pytest.mark.parametrize("failure", ["submit", "sync"])
def test_shared_execution_quarantines_before_reuse_and_drains_before_free(model_probe, monkeypatch, failure):
    model, state, _ = model_probe
    model.model_id = 1
    model.input_dataset, model.output_dataset = object(), object()
    datasets = {model.input_dataset, model.output_dataset}
    unloaded = []
    monkeypatch.setattr(model._acl.mdl, "unload", lambda _model: (unloaded.append(True), 0)[1], raising=False)
    setattr(state, failure, 1)
    with pytest.raises(BackendInferenceError) as error:
        model.execute({}, stream="stream")
    assert error.value.operation_started and not error.value.outcome_known
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.execute({}, stream="stream")
    with pytest.raises(BackendInferenceError, match="quarantined"):
        model.submit_async_isolated({}, stream="stream")
    assert state.submitted == 1
    state.sync = 1
    with pytest.raises(RuntimeError, match="synchronize_stream"):
        model.close()
    assert state.destroyed == [] and unloaded == []
    state.sync = 0
    model.close()
    model.close()
    assert set(state.destroyed) == datasets and len(state.destroyed) == 2
    assert unloaded == [True]


@pytest.mark.parametrize("scheduled", [False, True])
def test_manager_retries_real_session_drain_without_releasing_dependencies(model_probe, scheduled):
    from inference_service.backends.types import BackendState, RuntimeContext
    from inference_service.model_sessions.ascend import AscendOmModelSession
    from inference_service.pipeline.errors import PipelineManagerError
    from inference_service.pipeline.manager import InferencePipelineManager
    from inference_service.pipeline.runtime import InferencePipeline
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        LifecycleState,
        ModelRequest,
        ModelRuntimeHandle,
        RuntimeAssembly,
    )

    model, state, _ = model_probe
    session = AscendOmModelSession(runtime_manager=object(), priority_scheduling=scheduled)
    session._load = lambda *_args: None
    closed = []
    session._models = {
        "producer": SimpleNamespace(close=lambda: closed.append("producer")),
        "consumer": model,
    }
    session._priority_streams = SimpleNamespace(close=lambda: closed.append("streams"))
    session._lease = SimpleNamespace(close=lambda: closed.append("lease"))
    provider = SimpleNamespace(close=lambda: closed.append("provider"))
    executor = SimpleNamespace(close=lambda: closed.append("executor"))
    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            owned_components=(provider, session, executor),
            load_context=context,
        )
    )
    pipeline = object.__new__(InferencePipeline)
    pipeline._pipeline_id = "policy"
    pipeline._context = context
    pipeline._unified_handle = handle
    # Cleanup ownership is mandatory in both execution modes; priority only
    # selects the Session's execution behavior, never whether cleanup can retry.
    manager = InferencePipelineManager((pipeline,))
    manager.start()
    state.submit = state.sync = 1
    with pytest.raises(BackendInferenceError):
        model.submit_async_isolated({}, stream="stream")
    with pytest.raises(PipelineManagerError, match="shutdown"):
        manager.close()
    assert handle.health.state is LifecycleState.CLOSING
    assert session.health().state is BackendState.FAILED
    assert closed == ["executor"]
    with pytest.raises(ExecutionFailure):
        handle.execute(ModelRequest({}), ExecutionContext("after-close"))
    with pytest.raises(PipelineManagerError, match="closed"):
        manager.start()
    state.sync = 0
    manager.close()
    manager.close()
    assert closed == ["executor", "producer", "streams", "lease", "provider"]
    assert handle.health.state is LifecycleState.CLOSED
    assert session.health().state is BackendState.CLOSED


def test_scheduled_session_close_keeps_streams_and_lease_until_models_drain(model_probe):
    from inference_service.backends.types import BackendState
    from inference_service.model_sessions.ascend import AscendOmModelSession

    model, state, _ = model_probe
    session = AscendOmModelSession(runtime_manager=object(), priority_scheduling=True)
    closed = []
    session._models = {"encoder": model}
    session._priority_streams = SimpleNamespace(close=lambda: closed.append("streams"))
    session._lease = SimpleNamespace(close=lambda: closed.append("lease"))
    session._context = object()
    state.submit = state.sync = 1
    with pytest.raises(BackendInferenceError):
        model.submit_async_isolated({}, stream="stream")
    with pytest.raises(BackendLifecycleError, match="close pending"):
        session.close()
    assert session.health().state is BackendState.FAILED
    assert session._context is not None
    assert closed == []
    state.sync = 0
    session.close()
    assert closed == ["streams", "lease"]
    assert session.health().state is BackendState.CLOSED


@pytest.mark.parametrize("scheduled", [False, True])
def test_ascend_session_factory_enables_async_only_when_scheduled(tmp_path, monkeypatch, scheduled):
    from inference_manifest import CompiledDeployment
    from inference_service.backends import RuntimeContext
    from inference_service.model_sessions import ascend

    path = tmp_path / "model.om"
    path.write_bytes(b"fake")
    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
            "execution": ["model"],
            "bindings": {"model": _bindings()},
            "artifacts": {"model": {"path": "model.om", "format": "om"}},
        }
    )
    options = []
    lease = SimpleNamespace(close=lambda: None)

    def factory(_lease, role, model_path, bindings, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(load_descriptor=lambda: None, prepare_datasets=lambda **kwargs: None, close=lambda: None)

    monkeypatch.setattr(
        ascend.AclPriorityStreamPool,
        "create",
        lambda _lease, *, cleanup: SimpleNamespace(level_count=8, close=lambda: None),
    )
    session = ascend.AscendOmModelSession(
        runtime_manager=SimpleNamespace(acquire=lambda device: lease),
        model_factory=factory,
        priority_scheduling=scheduled,
    )
    context = RuntimeContext(
        SimpleNamespace(deployment=deployment, bundle_root=tmp_path), priority_scheduling=scheduled
    )
    try:
        session.load(context)
        assert options == [{"async_execution": True} if scheduled else {}]
    finally:
        session.close()


def test_acl_primary_and_cleanup_errors_survive_public_serialization(model_probe, monkeypatch):
    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle
    from inference_service.unified_runtime.errors import serialize_execution_failure

    model, _, _ = model_probe
    destroy = model._destroy_dataset

    def read(*_):
        raise ValueError("output decode failed")

    def release(*_):
        raise RuntimeError("dataset release failed")

    monkeypatch.setattr(model, "_read_dataset_outputs", read)
    monkeypatch.setattr(model, "_destroy_dataset", release)
    owner = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: model.submit_async_isolated({}, stream="s").completion.result(1),
            close=model.close,
        )
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("two-errors"))
        diagnostic = serialize_execution_failure(caught.value)
        assert diagnostic["code"] == "async_output_failed"
        assert "output decode failed" in diagnostic["message"]
        assert diagnostic["details"]["cleanup_error"] == "dataset release failed"
        assert diagnostic["details"]["close_pending"]
    finally:
        monkeypatch.setattr(model, "_destroy_dataset", destroy)
        owner.close()


def test_real_output_copy_free_failure_releases_numpy_after_recovery(model_probe, monkeypatch):
    import gc
    import weakref

    from inference_service.backends.ascend.model import AclTensorDescriptor, _DatasetBuffer
    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    model, _, _ = model_probe
    references = []
    blocked = [True]
    frombuffer = np.frombuffer

    class Tracked(np.ndarray):
        def __array_finalize__(self, _):
            references.append(weakref.ref(self))

    monkeypatch.setattr(np, "frombuffer", lambda *args, **kwargs: frombuffer(*args, **kwargs).view(Tracked))
    model.output_descriptors = (AclTensorDescriptor(0, "output", np.dtype("float32"), (1024,), 4096),)
    model._acl.rt.malloc_host = lambda _: (object(), 0)
    model._acl.rt.memcpy = lambda *_: 0
    model._acl.rt.free_host = lambda _: int(blocked[0])
    model._acl.util = SimpleNamespace(ptr_to_bytes=lambda *_: bytes(4096))
    monkeypatch.setattr(model, "_read_dataset_outputs", AclModel._read_dataset_outputs.__get__(model))
    monkeypatch.setattr(model, "_create_dataset", lambda *_: (object(), [_DatasetBuffer(object(), None, 4096, False)]))
    owner = ModelRuntimeHandle(
        SimpleNamespace(
            execute=lambda *_: model.submit_async_isolated({}, stream="s").completion.result(1),
            reset=model.drain_quarantined,
            close=model.close,
        ),
        resettable=True,
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("free-host"))
        assert caught.value.close_pending and "free_host" in caught.value.message
        blocked[0] = False
        owner.reset()
        gc.collect()
        assert references and all(reference() is None for reference in references)
        assert owner.last_failure.to_dict() == caught.value.to_dict() and owner.ready
        assert not model._quarantined_host_buffers
    finally:
        blocked[0] = False
        owner.close()


@pytest.mark.parametrize("failure_at", ["copy", "decode"])
def test_real_output_failure_preserves_primary_error_when_free_host_also_fails(model_probe, monkeypatch, failure_at):
    from inference_service.backends.ascend.model import AclTensorDescriptor, _DatasetBuffer
    from inference_service.unified_runtime import ExecutionFailureFactory

    model, _, _ = model_probe
    blocked = [True]
    model.output_descriptors = (AclTensorDescriptor(0, "output", np.dtype("float32"), (1,), 4),)
    model._acl.rt.malloc_host = lambda _: (object(), 0)
    model._acl.rt.memcpy = lambda *_: int(failure_at == "copy")
    model._acl.rt.free_host = lambda _: int(blocked[0])
    model._acl.util = SimpleNamespace(ptr_to_bytes=lambda *_: b"invalid")
    monkeypatch.setattr(model, "_read_dataset_outputs", AclModel._read_dataset_outputs.__get__(model))
    monkeypatch.setattr(model, "_create_dataset", lambda *_: (object(), [_DatasetBuffer(object(), None, 4, False)]))
    try:
        with pytest.raises(BackendInferenceError) as caught:
            model.submit_async_isolated({}, stream="s").completion.result(1)
        diagnostic = ExecutionFailureFactory().from_exception(caught.value).to_dict()
        assert diagnostic["code"] == "async_output_failed"
        assert ("D2H" if failure_at == "copy" else "buffer size") in diagnostic["message"]
        assert "free_host" in diagnostic["details"]["cleanup_error"]
        assert diagnostic["details"]["close_pending"] and diagnostic["evidence"]["outcome_known"]
        assert model._quarantined_host_buffers
    finally:
        blocked[0] = False
        model.drain_quarantined()
    assert not model._quarantined_host_buffers


@pytest.mark.parametrize("stream", [None, "shared-stream", "bank"])
@pytest.mark.parametrize("failure_at", ["allocate", "d2h", "decode", "copy", "reshape"])
def test_shared_outputs_keep_completion_and_release_domain(model_probe, monkeypatch, stream, failure_at):
    from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
    from inference_service.backends.ascend.model import AclTensorDescriptor, _DatasetBuffer
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
        RuntimeAssembly,
        RuntimeProviders,
    )

    model, state, _ = model_probe
    model.model_id, model.input_dataset, model.output_dataset = 1, object(), object()
    model._acl.mdl.execute = lambda *_: 0
    model._acl.mdl.unload = lambda _: 0
    model.output_descriptors = (
        AclTensorDescriptor(0, "output", np.dtype("float32"), (2,) if failure_at == "reshape" else (1,), 4),
    )
    model.output_buffers = [_DatasetBuffer(object(), None, 4, False)]
    model.output_host_buffers = [None]
    model._acl.rt.malloc_host = lambda _: (object(), int(failure_at == "allocate"))
    model._acl.rt.free_host = lambda _: 0
    model._acl.rt.memcpy = lambda *_: int(failure_at == "d2h")
    model._acl.util = SimpleNamespace(ptr_to_bytes=lambda *_: b"bad" if failure_at == "decode" else bytes(4))
    if failure_at == "copy":

        class Uncopyable:
            def copy(self):
                raise MemoryError("host output allocation failed")

        monkeypatch.setattr(np, "frombuffer", lambda *_, **__: Uncopyable())
    if stream == "bank":
        from inference_service.backends.ascend.model import _DatasetResources

        model._dataset_banks = [_DatasetResources(model.input_dataset, [], model.output_dataset, model.output_buffers)]
        model._dataset_bank_host_buffers = [[None if failure_at == "allocate" else object()]]
        model._host_output_indices = {0}
    domains = ResourceDomainAdmissions()
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(
                execute=lambda *_: model.execute_bank(0, {}) if stream == "bank" else model.execute({}, stream=stream)
            ),
            session=SimpleNamespace(name="acl-outputs", capabilities=BackendCapabilities(resource_domain="ascend:0")),
            providers=RuntimeProviders(object(), domains),
        )
    )
    owner.load()
    peer = domains.register("ascend:0", 1)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("output"))
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known and owner.ready
        assert state.submitted == int(stream == "shared-stream")
        peer.acquire(context=ExecutionContext("peer", Deadline.after(0.1)))
        peer.release()
    finally:
        peer.close()
        owner.close()


@pytest.mark.parametrize("failure_at", ["semantic", "missing", "size", "native", "d2h", "reshape", "mapping"])
def test_stateful_bank_failure_preserves_completion_and_recovery(model_probe, monkeypatch, failure_at):
    from inference_service.backends import BackendCapabilities, BackendState, ResourceDomainAdmissions
    from inference_service.backends.ascend.model import AclTensorDescriptor, _DatasetBuffer, _DatasetResources
    from inference_service.model_sessions.ascend_stateful import StatefulAscendOmModelSession
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
        RecoveryAction,
        RuntimeAssembly,
        RuntimeProviders,
    )

    model, _, _ = model_probe
    calls = []
    failing = [True]
    model.model_id = 1
    model.input_descriptors = (AclTensorDescriptor(0, "input", np.dtype("float32"), (1,), 4),)
    shape = (2,) if failure_at == "reshape" else (1,)
    model.output_descriptors = (AclTensorDescriptor(0, "output", np.dtype("float32"), shape, 4),)
    model._dataset_banks = [
        _DatasetResources(
            object(), [_DatasetBuffer(object(), None, 4, True)], object(), [_DatasetBuffer(object(), None, 4, False)]
        )
        for _ in range(2)
    ]
    model._dataset_bank_host_buffers = [[object()], [object()]]
    model._host_output_indices = {0}
    model._acl.mdl.execute = lambda *_: calls.append("native") or int(failing[0] and failure_at == "native")
    model._acl.mdl.unload = lambda *_: 0
    model._acl.rt.free_host = lambda *_: 0
    model._acl.rt.memcpy = lambda *args: int(failing[0] and failure_at == "d2h" and args[-1] == 2)
    model._acl.util = SimpleNamespace(bytes_to_ptr=lambda _: object(), ptr_to_bytes=lambda *_: bytes(4))
    monkeypatch.setattr(model, "_prepare_inputs", AclModel._prepare_inputs.__get__(model))
    session = StatefulAscendOmModelSession(runtime_manager=object())
    session._state = BackendState.LOADING
    session._loading = True
    session._update_loaded_capabilities(stateful=True, resettable=True)
    session._loading = False
    session._state = BackendState.READY
    session._models = {"encoder": model}
    session._state_indices = {"encoder": ()}
    session._state_banks = {"encoder": 0}
    session._loaded_deployment = lambda: SimpleNamespace(bindings={"encoder": _bindings()})
    session._capture_role_inputs = session._capture_role_outputs = lambda *_: None
    session._validate_role_values = lambda *_, **__: None
    execute_bank = model.execute_bank

    def run_bank(bank, inputs):
        if failing[0] and failure_at in {"missing", "size"}:
            inputs = {} if failure_at == "missing" else {0: np.zeros(2, np.float32)}
        result = execute_bank(bank, inputs)
        return {} if failing[0] and failure_at == "mapping" else result

    model.execute_bank = run_bank
    domains = ResourceDomainAdmissions()
    capabilities = BackendCapabilities(resource_domain="bank-test", stateful=True, resettable=True)
    runtime = SimpleNamespace(
        open_stream=lambda *_: object(),
        close_stream=lambda *_: None,
        reset_stream=lambda *_: None,
        step=lambda _, request, context: session.execute_role("encoder", request.inputs, request, context),
    )
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=runtime,
            streaming_runtime=runtime,
            session=SimpleNamespace(name="bank", capabilities=capabilities),
            providers=RuntimeProviders(object(), domains),
            state_scope="stream",
            stateful=True,
            resettable=True,
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
        )
    )
    owner.load()
    stream = owner.open_stream(ExecutionContext("open"))
    peer = domains.register("bank-test", 1)
    request = ModelRequest({"input": np.zeros(1, np.float32)})
    unstarted = failure_at in {"semantic", "missing", "size"}
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.step(stream, ModelRequest() if failure_at == "semantic" else request, ExecutionContext("bad"))
        evidence = caught.value.evidence
        assert evidence.state is (
            OutcomeState.NOT_STARTED
            if unstarted
            else OutcomeState.STARTED
            if failure_at == "native"
            else OutcomeState.COMPLETED
        )
        assert evidence.outcome_known is (failure_at != "native")
        assert evidence.state_mutated is not unstarted
        assert owner.ready is unstarted
        assert len(calls) == int(not unstarted)
        if failure_at == "native":
            from inference_service.unified_runtime.errors import DeadlineExceeded

            with pytest.raises(DeadlineExceeded):
                peer.acquire(context=ExecutionContext("blocked", Deadline.after(0.02)))
        else:
            peer.acquire(context=ExecutionContext("peer", Deadline.after(0.1)))
            peer.release()
        if unstarted:
            assert session._state_banks["encoder"] == 0
            failing[0] = False
            assert "output" in owner.step(stream, request, ExecutionContext("good")).outputs
            assert session._state_banks["encoder"] == 1
        else:
            assert caught.value.recovery.action is RecoveryAction.RESET_RUNTIME
    finally:
        peer.close()
        owner.close()
