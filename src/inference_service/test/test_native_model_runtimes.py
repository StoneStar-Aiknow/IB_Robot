from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import TorchRuntimeProfile
from inference_service.backends import BackendState, RuntimeContext
from inference_service.model_sessions import TorchModelSession
from inference_service.pipeline import SequentialModelExecutor
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    LoadRollback,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OutcomeEvidence,
    RuntimeAssembly,
    RuntimeLatency,
)


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    __version__ = "fake-torch-1.0"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def as_tensor(value) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def is_tensor(value) -> bool:
        return isinstance(value, _FakeTensor)

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext

        return nullcontext()


class _FakeModule:
    def __init__(self) -> None:
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, inputs):
        self.calls += 1
        return {"scores": _FakeTensor(inputs["features"].value + 1.0)}


def _context(tmp_path) -> RuntimeContext:
    manifest = SimpleNamespace(
        model=SimpleNamespace(
            interface="tensor_model",
            model_type="fake_torch",
            operation="infer",
            inputs=(),
            outputs=(SimpleNamespace(semantic="scores", dtype="float32", shape=(1, 2)),),
        )
    )
    validated = SimpleNamespace(
        manifest=manifest,
        deployment=SimpleNamespace(backend="torch"),
        deployment_name="torch_cpu",
        runtime_profile=TorchRuntimeProfile(device="cpu"),
        bundle_root=tmp_path,
        deployment_fingerprint="sha256:test",
        top_level_identity=SimpleNamespace(interface="tensor_model", model_type="fake_torch", operation="infer"),
    )
    return RuntimeContext(validated, runtime_profile=validated.runtime_profile)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("bind_before_load", [False, True])
def test_policy_requires_the_native_session_managed_completion_contract(tmp_path, managed, bind_before_load):
    from dataclasses import replace

    context = replace(_context(tmp_path), managed_execution=managed)
    session = TorchModelSession(lambda _: _FakeModule(), torch_loader=_FakeTorch)
    owner = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=session, session=session, load_context=context))
    policy = SimpleNamespace(ready=True, cancel=lambda *_: None)
    for operation in ("execute", "reset", "close"):
        setattr(
            policy,
            operation,
            lambda *args, operation=operation: getattr(owner, f"_{operation}_local")(*args, _dispatch_key=authority),
        )
    authority = None
    try:
        if bind_before_load:
            authority = owner.bind_execution_policy(policy)
            if managed:
                owner.load()
            else:
                with pytest.raises(ExecutionFailure, match="managed_execution"):
                    owner.load()
                return
        else:
            owner.load()
            if not managed:
                with pytest.raises(ValueError, match="managed_execution"):
                    owner.bind_execution_policy(policy)
                assert owner.ready and not owner.execution_policy_bound
                return
            authority = owner.bind_execution_policy(policy)
        result = owner.execute(ModelRequest({"features": np.zeros((1, 2), np.float32)}), ExecutionContext("managed"))
        np.testing.assert_array_equal(result.outputs["scores"], np.ones((1, 2), np.float32))
    finally:
        owner.close()


def test_retained_execution_failure_does_not_own_torch_resources_after_close(tmp_path):
    import gc
    import traceback
    import weakref

    references = []

    class Module(_FakeModule):
        def __call__(self, inputs):
            temporary = _FakeTensor([3])
            references.extend(weakref.ref(value) for value in (self, inputs["features"], temporary))
            raise RuntimeError("kernel failed")

    session = TorchModelSession(lambda _: Module(), torch_loader=_FakeTorch)
    owner = ModelRuntimeHandle(
        RuntimeAssembly(runtime_executor=session, session=session, load_context=_context(tmp_path))
    )
    owner.load()
    failures = []
    stacks = []
    for attempt in range(2):
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(
                ModelRequest({"features": np.zeros((1, 2), np.float32)}), ExecutionContext(f"failed-{attempt}")
            )
        failures.append(caught.value)
        stacks.append(traceback.extract_tb(caught.value.cause.__traceback__))
    owner.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    for failure, stack in zip(failures, stacks, strict=True):
        assert "kernel failed" in str(failure.cause)
        assert traceback.extract_tb(failure.cause.__traceback__) == stack


def test_torch_model_resource_uses_native_request_and_context(tmp_path) -> None:
    module = _FakeModule()
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)
    context = _context(tmp_path)
    session.load(context)

    outputs = session.execute(
        ModelRequest({"features": np.ones((1, 2), dtype=np.float32)}),
        ExecutionContext("torch-1"),
    )

    np.testing.assert_array_equal(outputs["scores"], np.full((1, 2), 2.0, dtype=np.float32))
    assert module.calls == 1
    assert session.health().state is BackendState.READY
    session.close()


@pytest.mark.parametrize("failure", ["loader", "validation"])
def test_torch_constructor_registers_completion_before_model_loader(tmp_path, failure):
    from inference_service.backends.errors import BackendLoadError
    from inference_service.unified_runtime import LifecycleState

    calls = []
    release_allowed = [False]
    torch = _FakeTorch()

    def synchronize(_):
        calls.append("sync")
        if not release_allowed[0]:
            raise RuntimeError("device completion pending")

    torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=synchronize, current_device=lambda: 0)

    def loader(_):
        calls.append("load")
        if failure == "loader":
            raise BackendLoadError("loader failed after acquiring device")
        return object()  # Acquired object fails the callable contract after return.

    context = _context(tmp_path)
    context = RuntimeContext(context.validated_manifest, runtime_profile=TorchRuntimeProfile(device="cuda"))
    session = TorchModelSession(loader, torch_loader=lambda: torch)
    owner = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=session, session=session, load_context=context))
    with pytest.raises(ExecutionFailure) as error:
        owner.load()
    assert not error.value.details["cleanup_complete"]
    assert calls == ["load", "sync", "sync"]
    assert session.health().state is BackendState.FAILED and owner.state is LifecycleState.FAILED
    release_allowed[0] = True
    owner.close()
    assert calls[-1] == "sync" and owner.closed and session.health().state is BackendState.CLOSED


def test_native_session_is_owned_by_runtime_handle(tmp_path) -> None:
    session = TorchModelSession(lambda _context: _FakeModule(), torch_loader=_FakeTorch)
    context = _context(tmp_path)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=session,
            session=session,
            execution_contract=ExecutionContract(),
            load_context=context,
        )
    )
    handle.load(context)
    result = handle.execute(
        ModelRequest({"features": np.zeros((1, 2), dtype=np.float32)}),
        ExecutionContext("handle-1"),
    )

    assert isinstance(result, ModelResult)
    assert result.successful
    handle.close()
    assert session.health().state is BackendState.CLOSED


def test_native_executor_does_not_own_model_resource() -> None:
    resource = SimpleNamespace(load=lambda _context: None, close=lambda: None)

    class Stage:
        def execute(self, frame, *, deadline):
            del deadline
            frame.values["output"] = frame.values["input"]

    class Adapter:
        def adapt(self, frame):
            return ModelResult(
                outputs={"output": frame.values["output"]},
                latency=RuntimeLatency(0.1, 0.1),
                evidence=OutcomeEvidence.completed("adaptation"),
            )

    executor = SequentialModelExecutor((Stage(),), Adapter(), components=(resource,))
    executor.load(SimpleNamespace())
    executor.close()
    assert resource.load is not None


def test_load_rollback_is_native_and_reverse_order() -> None:
    events: list[str] = []
    rollback = LoadRollback()
    rollback.defer(lambda: events.append("first"))
    rollback.defer(lambda: events.append("second"))
    assert rollback.rollback() == ()
    assert events == ["second", "first"]


def test_native_runtime_rejects_expired_request_before_resource_execution() -> None:
    class Runtime:
        def execute(self, _request, _context):
            raise AssertionError("backend must not execute")

    handle = ModelRuntimeHandle(Runtime(), execution_contract="request-direct")
    handle.load()
    with pytest.raises(ExecutionFailure) as error:
        handle.execute(ModelRequest({"x": 1}), ExecutionContext.create("expired", timeout=0))
    assert error.value.code == "deadline_exceeded"
    handle.close()


def test_torch_device_resolution_error_has_device_reason(tmp_path):
    from types import SimpleNamespace

    from inference_service.backends.errors import BackendLoadError
    from inference_service.model_sessions.torch import TorchModelSession

    calls = []
    torch = SimpleNamespace(device=lambda _: (_ for _ in ()).throw(RuntimeError("device lookup failed")))
    session = TorchModelSession(lambda _: calls.append("load"), torch_loader=lambda: torch)
    with pytest.raises(BackendLoadError) as caught:
        session.load(_context(tmp_path))
    assert caught.value.code == "device_unavailable"
    assert "resolve Torch device" in str(caught.value)
    assert calls == []
