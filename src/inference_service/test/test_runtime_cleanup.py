"""Runtime owners retain failed cleanup and release dependencies only after success."""

from types import SimpleNamespace

import pytest

from inference_service.backends import BackendCapabilities, RuntimeContext
from inference_service.model_sessions import ModelSession
from inference_service.unified_runtime import (
    ExecutionFailure,
    LoadRollback,
    ModelRuntimeHandle,
    RuntimeAssembly,
    RuntimeProviders,
)


def test_rollback_retries_failed_resource_before_releasing_dependencies():
    events = []
    rollback = LoadRollback()
    rollback.defer(lambda: events.append("dependency"))

    def close():
        events.append("resource")
        if events.count("resource") == 1:
            raise RuntimeError("cleanup pending")

    rollback.defer(close)
    assert len(rollback.rollback()) == 1
    assert events == ["resource"]
    assert rollback.rollback() == ()
    assert rollback.rollback() == ()
    assert events == ["resource", "resource", "dependency"]


def test_provider_cleanup_waits_for_retained_owner_to_settle():
    events = []

    def environment_close():
        events.append("environment")
        if events.count("environment") == 1:
            raise RuntimeError("runtime cleanup pending")

    providers = RuntimeProviders.create(
        SimpleNamespace(close=lambda: events.append("acl")),
        SimpleNamespace(close=lambda: events.append("admission")),
    )
    providers.retain_cleanup(SimpleNamespace(close=environment_close))
    with pytest.raises(RuntimeError, match="cleanup pending") as failure:
        providers.close()
    assert failure.value.close_pending
    assert failure.value.__cause__ is not None
    assert not providers.closed and events == ["environment"]
    providers.close()
    providers.close()
    assert providers.closed and events == ["environment", "environment", "admission", "acl"]


@pytest.mark.parametrize("with_provider", [False, True])
def test_pipeline_factory_transfers_failed_startup_cleanup_owner(monkeypatch, with_provider):
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.pipeline import factory
    from inference_service.pipeline.errors import PipelineManagerError

    events = []
    available = [False]

    def load():
        raise RuntimeError("original load failure")

    def close():
        events.append("pipeline")
        if not available[0]:
            raise BackendLifecycleError("release pending", close_pending=True)

    pipeline = SimpleNamespace(pipeline_id="policy", load=load, close=close)
    monkeypatch.setattr(factory, "create_inference_pipeline", lambda *_args, **_kwargs: pipeline)
    providers = (
        RuntimeProviders(SimpleNamespace(close=lambda: events.append("acl")), object()) if with_provider else None
    )
    with pytest.raises(PipelineManagerError, match="original load failure") as failure:
        factory.create_pipeline_manager("policy", object(), providers=providers)
    assert failure.value.close_pending and failure.value.cleanup_owner is not None
    assert str(failure.value.__cause__) == "original load failure"
    available[0] = True
    if providers is not None:
        providers.close()
        assert events == ["pipeline", "pipeline", "acl"]
    else:
        failure.value.cleanup_owner.close()
        assert events == ["pipeline", "pipeline"]


@pytest.mark.parametrize("cleanup_failures", [1, 2, 3])
def test_legacy_session_retains_partial_load_rollback_until_close(cleanup_failures):
    from inference_service.backends import BackendCapabilities, BackendState, RuntimeContext
    from inference_service.backends.errors import BackendLifecycleError, BackendLoadError
    from inference_service.model_sessions import ModelSession

    events = []

    class PartialSession(ModelSession):
        def __init__(self):
            super().__init__("partial", BackendCapabilities())

        def _load(self, context, rollback):
            rollback.defer(lambda: events.append("lease released"))
            rollback.defer(self.release_model)
            raise RuntimeError("partial SDK load")

        def release_model(self):
            events.append("model cleanup")
            if events.count("model cleanup") <= cleanup_failures:
                raise RuntimeError("device cleanup pending")

        def _execute(self, request, context):
            pytest.fail("failed session must not execute")

        def _close(self):
            pytest.fail("partial load must use its retained rollback")

    session = PartialSession()
    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace()))
    with pytest.raises(BackendLoadError, match="partial SDK load") as failure:
        session.load(context)
    assert not failure.value.rollback_clean
    assert events == ["model cleanup"]
    for _ in range(cleanup_failures - 1):
        with pytest.raises(BackendLifecycleError) as failure:
            session.close()
        assert failure.value.close_pending
        assert session.health().state is BackendState.FAILED
        assert "lease released" not in events
    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED
    assert events == ["model cleanup"] * (cleanup_failures + 1) + ["lease released"]


@pytest.mark.parametrize("cleanup_failures", [0, 1, 2, 3])
def test_partial_session_load_retains_failed_cleanup_through_handle_close(cleanup_failures):
    events = []

    class PartialSession(ModelSession):
        def __init__(self):
            super().__init__("partial", BackendCapabilities())

        def _load(self, context, rollback):
            rollback.defer(lambda: events.append("lease released"))
            rollback.defer(self.release_model)
            raise RuntimeError("partial SDK load")

        def release_model(self):
            events.append("model cleanup")
            if events.count("model cleanup") <= cleanup_failures:
                raise RuntimeError("device cleanup pending")

        def _execute(self, request, context):
            pytest.fail("failed session must not execute")

        def _close(self):
            pytest.fail("partial load must use its retained rollback")

    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace()))
    session = PartialSession()
    handle = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=session, session=session, load_context=context))
    with pytest.raises(ExecutionFailure, match="partial SDK load") as failure:
        handle.load()
    clean = cleanup_failures < 2
    assert failure.value.details["cleanup_complete"] is clean
    assert not handle.ready
    assert events.count("lease released") == int(clean)
    assert events.count("model cleanup") == min(cleanup_failures + 1, 2)
    for _ in range(max(0, cleanup_failures - 2)):
        with pytest.raises(ExecutionFailure) as failure:
            handle.close()
        assert failure.value.close_pending and not handle.closed
        assert "lease released" not in events
    handle.close()
    handle.close()
    assert handle.closed
    assert events == ["model cleanup"] * (cleanup_failures + 1) + ["lease released"]


def test_successful_close_retry_does_not_retain_failed_call_payload():
    import gc
    import weakref

    import numpy as np

    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle

    references = []

    def close():
        if not references:
            payload = np.ones(1024 * 1024, dtype=np.float32)
            references.append(weakref.ref(payload))
            raise BackendLifecycleError("native close pending", close_pending=True)

    owner = ModelRuntimeHandle(SimpleNamespace(close=close))
    owner.load()

    def failed_call():
        request = np.ones(1024 * 1024, dtype=np.float32)
        with pytest.raises(ExecutionFailure):
            owner.close()
        references.append(weakref.ref(request))

    failed_call()
    owner.close()
    gc.collect()
    assert owner.closed and all(reference() is None for reference in references)
