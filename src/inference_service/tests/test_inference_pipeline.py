from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import (
    BackendAdmissionError,
    BackendAdmissionEvidence,
    BackendCancellationError,
    BackendCapabilities,
    BackendInferenceError,
    BackendResult,
    BackendState,
    InferenceRequest,
    LifecycleBackend,
    PartialLoadRollback,
    ResourceDomainAdmissions,
    RuntimeContext,
)
from inference_service.codecs import BindingPolicyCodec, BoundInputs, ExecutionPlan
from inference_service.pipeline import (
    InferencePipeline,
    InferencePipelineManager,
    PipelineLifecycleError,
    PipelineManagerError,
    PipelineNotFoundError,
    PipelineNotReadyError,
    PipelineState,
    PipelineStateMachine,
    PipelineTimeoutError,
    PipelineTransitionError,
    PipelineValidationError,
)
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest

_MULTI_INSTANCE_EVIDENCE = BackendAdmissionEvidence(
    sdk_initialization=True,
    multi_instance_execution=True,
    failure_isolation=True,
    independent_close=True,
)


def _context(root: Path, *, compiled: bool = False) -> RuntimeContext:
    root.mkdir(parents=True)
    bundle_paths = create_policy_bundle(root, include_weights=not compiled)
    deployment_name = "compiled" if compiled else "cpu"
    manifest = make_manifest(
        root,
        bundle_paths,
        deployment_name=deployment_name,
        compiled=compiled,
        backend="rknn" if compiled else "torch",
    )
    write_manifest(root, manifest)
    return RuntimeContext(load_inference_manifest(root, deployment_name))


def _request(
    request_id: str = "request-1",
    *,
    prompt: str | None = None,
    deadline: datetime | None = None,
    value: float = 1.0,
    priority: int = 0,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        inputs={"observation.state": np.full((1, 6), value, dtype=np.float32)},
        prompt=prompt,
        deadline=deadline,
        priority=priority,
    )


def _thread_call(callback: Callable[[], object]) -> tuple[threading.Thread, list[object], list[BaseException]]:
    results: list[object] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(callback())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results, errors


def _unsafe_backend_result(action: object, actual_chunk_size: int) -> BackendResult:
    result = object.__new__(BackendResult)
    object.__setattr__(result, "action", action)
    object.__setattr__(result, "actual_chunk_size", actual_chunk_size)
    object.__setattr__(result, "backend_latency_ms", 0.2)
    object.__setattr__(result, "metadata", {})
    return result


class RecordingProcessor:
    def __init__(self, name: str, trace: list[str] | None = None) -> None:
        self.name = name
        self.trace = trace
        self.load_calls = 0
        self.close_calls = 0
        self.reset_calls = 0
        self.calls: list[dict[str, object]] = []

    def load(self, context: RuntimeContext) -> None:
        self.load_calls += 1
        if self.trace is not None:
            self.trace.append(f"load:{self.name}")

    def __call__(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        recorded = dict(inputs)
        self.calls.append(recorded)
        return recorded

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1
        if self.trace is not None:
            self.trace.append(f"close:{self.name}")


class RecordingPostprocessor:
    def __init__(self, name: str, trace: list[str] | None = None, offset: float = 0.0) -> None:
        self.name = name
        self.trace = trace
        self.offset = offset
        self.load_calls = 0
        self.close_calls = 0
        self.reset_calls = 0
        self.calls: list[object] = []

    def load(self, context: RuntimeContext) -> None:
        self.load_calls += 1
        if self.trace is not None:
            self.trace.append(f"load:{self.name}")

    def __call__(self, action: object) -> object:
        self.calls.append(action)
        return np.asarray(action) + self.offset

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1
        if self.trace is not None:
            self.trace.append(f"close:{self.name}")


class MockBackend(LifecycleBackend):
    def __init__(
        self,
        name: str,
        *,
        capabilities: BackendCapabilities | None = None,
        domains: ResourceDomainAdmissions | None = None,
        trace: list[str] | None = None,
    ) -> None:
        super().__init__(name, capabilities or BackendCapabilities(), domains=domains)
        self.trace = trace
        self.fail_load = False
        self.load_calls = 0
        self.close_calls = 0
        self.reset_calls = 0
        self.cancel_calls: list[str] = []
        self.requests: list[InferenceRequest] = []
        self.result_factory: Callable[[InferenceRequest], BackendResult] = lambda request: BackendResult(
            action=np.full((1, 2, 6), float(np.asarray(request.inputs["observation.state"])[0, 0])),
            actual_chunk_size=2,
            backend_latency_ms=0.2,
            metadata={"request_id": request.request_id},
        )
        self.infer_release = threading.Event()
        self.infer_release.set()
        self.reset_started = threading.Event()
        self.reset_release = threading.Event()
        self.reset_release.set()
        self._metrics = threading.Condition()
        self.infer_entries = 0
        self.active_infer = 0
        self.max_active_infer = 0

    def wait_for_infer_entries(self, count: int, timeout: float = 2.0) -> bool:
        with self._metrics:
            return self._metrics.wait_for(lambda: self.infer_entries >= count, timeout=timeout)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        self.load_calls += 1
        if self.trace is not None:
            self.trace.append("load:backend")
        if self.fail_load:
            raise RuntimeError("mock backend load failed")

    def _infer(self, request: InferenceRequest) -> BackendResult:
        with self._metrics:
            self.requests.append(request)
            self.infer_entries += 1
            self.active_infer += 1
            self.max_active_infer = max(self.max_active_infer, self.active_infer)
            self._metrics.notify_all()
        try:
            if not self.infer_release.wait(timeout=2):
                raise RuntimeError("mock inference release timed out")
            return self.result_factory(request)
        finally:
            with self._metrics:
                self.active_infer -= 1
                self._metrics.notify_all()

    def _reset(self) -> None:
        self.reset_calls += 1
        self.reset_started.set()
        if not self.reset_release.wait(timeout=2):
            raise RuntimeError("mock reset release timed out")

    def _cancel(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)

    def _close(self) -> None:
        self.close_calls += 1
        if self.trace is not None:
            self.trace.append("close:backend")


def _native_pipeline(
    root: Path,
    pipeline_id: str,
    *,
    backend: MockBackend | None = None,
    domains: ResourceDomainAdmissions | None = None,
    preprocessor: RecordingProcessor | None = None,
    postprocessor: RecordingPostprocessor | None = None,
    request_timeout: float | None = None,
    default_task: str | None = None,
) -> tuple[InferencePipeline, MockBackend]:
    runtime_backend = backend or MockBackend(f"mock-{pipeline_id}", domains=domains)
    pipeline = InferencePipeline(
        pipeline_id,
        _context(root),
        executor=runtime_backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        request_timeout=request_timeout,
        default_task=default_task,
    )
    return pipeline, runtime_backend


def test_native_pipeline_loads_hooks_passes_canonical_inputs_and_standardizes_result(tmp_path):
    trace: list[str] = []
    preprocessor = RecordingProcessor("preprocessor", trace)
    postprocessor = RecordingPostprocessor("postprocessor", trace, offset=1.0)
    backend = MockBackend("mock-native", domains=ResourceDomainAdmissions(), trace=trace)
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        default_task="pick default",
    )

    pipeline.load()
    result = pipeline.infer(_request())

    assert trace[:3] == ["load:preprocessor", "load:postprocessor", "load:backend"]
    assert backend.requests[0].inputs["observation.state"].shape == (1, 6)
    assert backend.requests[0].inputs["task"] == "pick default"
    assert backend.requests[0].prompt == "pick default"
    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), 2.0))
    assert result.pipeline_id == "policy"
    assert result.bundle == "test-cpu"
    assert result.deployment == "cpu"
    assert result.backend == "mock-native"
    assert result.state is PipelineState.READY
    assert result.actual_chunk_size == 2
    assert result.raw_action is None
    assert result.metadata["pipeline_id"] == "policy"
    assert result.metadata["deployment_fingerprint"] == result.deployment_fingerprint
    assert set(result.metadata["latency_ms"]) == {"total", "preprocess", "backend", "postprocess"}

    diagnostics = pipeline.diagnostics()
    assert diagnostics.ready is True
    assert diagnostics.metadata["bundle"] == "test-cpu"
    pipeline.close()
    assert trace[-3:] == ["close:backend", "close:postprocessor", "close:preprocessor"]


def test_pipeline_preserves_request_priority_for_backend(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()

    pipeline.infer(_request(priority=7))

    assert backend.requests[0].priority == 7
    pipeline.close()


def test_pipeline_merges_control_inputs_after_preprocessing_and_captures_raw_action(tmp_path):
    class DroppingProcessor(RecordingProcessor):
        def __call__(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
            recorded = dict(inputs)
            self.calls.append(recorded)
            return {key: value for key, value in recorded.items() if key != "noise"}

    class MutatingPostprocessor(RecordingPostprocessor):
        def __call__(self, action: object) -> object:
            self.calls.append(action)
            value = np.asarray(action)
            value += 5.0
            return value

    preprocessor = DroppingProcessor("preprocessor")
    postprocessor = MutatingPostprocessor("postprocessor")
    pipeline, backend = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    pipeline.load()
    noise = np.full((1, 2, 6), 7.0, dtype=np.float32)

    result = pipeline.infer(
        _request(),
        control_inputs={"noise": noise},
        capture_raw_action=True,
    )

    assert "noise" not in preprocessor.calls[0]
    np.testing.assert_array_equal(backend.requests[0].inputs["noise"], noise)
    np.testing.assert_array_equal(result.raw_action, np.ones((1, 2, 6)))
    np.testing.assert_array_equal(result.action, np.full((1, 2, 6), 6.0))
    pipeline.close()


def test_pipeline_rejects_control_input_collisions(tmp_path):
    pipeline, backend = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        preprocessor=RecordingProcessor("preprocessor"),
    )
    pipeline.load()

    with pytest.raises(PipelineValidationError, match="control inputs conflict") as error:
        pipeline.infer(
            _request(),
            control_inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        )

    assert error.value.details["conflicting_inputs"] == ("observation.state",)
    assert backend.infer_entries == 0
    pipeline.close()


def test_compiled_pipeline_uses_selected_bindings_codec_and_execution_plan(tmp_path):
    context = _context(tmp_path / "compiled-bundle", compiled=True)
    backend = MockBackend("mock-compiled", domains=ResourceDomainAdmissions())

    def compiled_result(request: InferenceRequest) -> BackendResult:
        plan = request.inputs["execution_plan"]
        role_inputs = request.inputs["role_inputs"]
        assert isinstance(plan, ExecutionPlan)
        assert plan.role_names == ("policy",)
        assert isinstance(role_inputs["policy"], BoundInputs)
        assert list(role_inputs["policy"].by_runtime_name) == ["state", "image"]
        return BackendResult(
            action={"actions": np.full((1, 4, 6), 3.0, dtype=np.float32)},
            actual_chunk_size=4,
            backend_latency_ms=0.5,
        )

    backend.result_factory = compiled_result
    pipeline = InferencePipeline(
        "compiled_policy",
        context,
        executor=backend,
        codec=BindingPolicyCodec(),
    )
    pipeline.load()
    request = InferenceRequest(
        request_id="compiled-request",
        inputs={
            "observation.state": np.zeros((1, 6), dtype=np.float32),
            "observation.images.top": np.zeros((1, 3, 16, 24), dtype=np.float32),
        },
    )

    result = pipeline.infer(request)

    np.testing.assert_array_equal(result.action, np.full((1, 4, 6), 3.0))
    assert result.actual_chunk_size == 4
    assert result.deployment == "compiled"
    pipeline.close()


def test_manager_routes_two_independent_pipelines_and_rejects_unknown_id(tmp_path):
    first, first_backend = _native_pipeline(tmp_path / "first", "first")
    second, second_backend = _native_pipeline(tmp_path / "second", "second")
    manager = InferencePipelineManager((first, second))
    manager.start()

    first_result = manager.infer("first", _request("first", value=1.0))
    second_result = manager.infer("second", _request("second", value=2.0))

    assert float(np.asarray(first_result.action)[0, 0, 0]) == 1.0
    assert float(np.asarray(second_result.action)[0, 0, 0]) == 2.0
    assert first_backend.infer_entries == 1
    assert second_backend.infer_entries == 1
    with pytest.raises(PipelineNotFoundError) as error:
        manager.infer("missing", _request())
    assert error.value.code == "pipeline_not_found"
    assert error.value.details["available_pipelines"] == ("first", "second")
    manager.close()


def test_manager_rejects_duplicate_ids(tmp_path):
    first_backend = MockBackend("first-duplicate", domains=ResourceDomainAdmissions())
    second_backend = MockBackend("second-duplicate", domains=ResourceDomainAdmissions())
    first, _ = _native_pipeline(tmp_path / "first", "duplicate", backend=first_backend)
    second, _ = _native_pipeline(tmp_path / "second", "duplicate", backend=second_backend)

    with pytest.raises(PipelineManagerError) as error:
        InferencePipelineManager((first, second))

    assert error.value.code == "duplicate_pipeline_id"
    first.close()
    second.close()


def test_same_pipeline_requests_serialize_through_backend_admission(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()
    backend.infer_release.clear()

    first, _, first_errors = _thread_call(lambda: pipeline.infer(_request("first")))
    assert backend.wait_for_infer_entries(1)
    second, _, second_errors = _thread_call(lambda: pipeline.infer(_request("second")))
    time.sleep(0.05)
    assert backend.infer_entries == 1

    backend.infer_release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert backend.infer_entries == 2
    assert backend.max_active_infer == 1
    pipeline.close()


def test_different_pipeline_backends_can_overlap_without_shared_domain(tmp_path):
    domains = ResourceDomainAdmissions()
    first, first_backend = _native_pipeline(tmp_path / "first", "first", domains=domains)
    second, second_backend = _native_pipeline(tmp_path / "second", "second", domains=domains)
    manager = InferencePipelineManager((first, second))
    manager.start()
    first_backend.infer_release.clear()
    second_backend.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: manager.infer("first", _request("first")))
    second_thread, _, second_errors = _thread_call(lambda: manager.infer("second", _request("second")))
    assert first_backend.wait_for_infer_entries(1)
    assert second_backend.wait_for_infer_entries(1)

    first_backend.infer_release.set()
    second_backend.infer_release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    manager.close()


def test_shared_backend_resource_domain_serializes_different_pipelines(tmp_path):
    domains = ResourceDomainAdmissions()
    capabilities = BackendCapabilities(
        supports_multiple_instances=True,
        resource_domain="shared-accelerator",
        admission_evidence=_MULTI_INSTANCE_EVIDENCE,
    )
    first_backend = MockBackend("shared", capabilities=capabilities, domains=domains)
    second_backend = MockBackend("shared", capabilities=capabilities, domains=domains)
    first, _ = _native_pipeline(tmp_path / "first", "first", backend=first_backend)
    second, _ = _native_pipeline(tmp_path / "second", "second", backend=second_backend)
    manager = InferencePipelineManager((first, second))
    manager.start()
    first_backend.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: manager.infer("first", _request("first")))
    assert first_backend.wait_for_infer_entries(1)
    second_thread, _, second_errors = _thread_call(lambda: manager.infer("second", _request("second")))
    time.sleep(0.05)
    assert second_backend.infer_entries == 0

    first_backend.infer_release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert second_backend.infer_entries == 1
    manager.close()


def test_targeted_reset_only_resets_selected_pipeline(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)
    first_backend = MockBackend("reset-first", capabilities=capabilities, domains=ResourceDomainAdmissions())
    second_backend = MockBackend("reset-second", capabilities=capabilities, domains=ResourceDomainAdmissions())
    first, _ = _native_pipeline(tmp_path / "first", "first", backend=first_backend)
    second, _ = _native_pipeline(tmp_path / "second", "second", backend=second_backend)
    manager = InferencePipelineManager((first, second))
    manager.start()

    manager.reset("first")

    assert first_backend.reset_calls == 1
    assert second_backend.reset_calls == 0
    assert manager.health("first").state is PipelineState.READY
    assert manager.health("second").state is PipelineState.READY
    manager.close()


def test_pipeline_reset_resets_backend_and_processors(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = MockBackend("reset-policy", capabilities=capabilities, domains=ResourceDomainAdmissions())
    preprocessor = RecordingProcessor("preprocessor")
    postprocessor = RecordingPostprocessor("postprocessor")
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    pipeline.load()

    pipeline.reset()

    assert backend.reset_calls == 1
    assert preprocessor.reset_calls == 1
    assert postprocessor.reset_calls == 1
    pipeline.close()


def test_processor_reset_failure_leaves_pipeline_failed(tmp_path):
    class FailingResetProcessor(RecordingProcessor):
        def reset(self) -> None:
            raise RuntimeError("processor reset failed")

    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = MockBackend("reset-policy", capabilities=capabilities, domains=ResourceDomainAdmissions())
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=FailingResetProcessor("preprocessor"),
        postprocessor=RecordingPostprocessor("postprocessor"),
    )
    pipeline.load()

    with pytest.raises(RuntimeError, match="processor reset failed"):
        pipeline.reset()

    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_stateful_non_resettable_backend_rejects_reset_without_mutating_processors(tmp_path):
    capabilities = BackendCapabilities(resettable=False, stateful=True)
    backend = MockBackend("stateful", capabilities=capabilities, domains=ResourceDomainAdmissions())
    preprocessor = RecordingProcessor("preprocessor")
    postprocessor = RecordingPostprocessor("postprocessor")
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    pipeline.load()

    with pytest.raises(PipelineLifecycleError) as error:
        pipeline.reset()

    assert error.value.code == "reset_unsupported"
    assert backend.reset_calls == 0
    assert preprocessor.reset_calls == 0
    assert postprocessor.reset_calls == 0
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_backend_reset_admission_timeout_preserves_ready_state_and_processors(tmp_path):
    class AdmissionTimeoutBackend(MockBackend):
        def reset(self, deadline=None) -> None:
            raise BackendAdmissionError("reset admission timed out", code="deadline_exceeded")

    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = AdmissionTimeoutBackend("reset-policy", capabilities=capabilities, domains=ResourceDomainAdmissions())
    preprocessor = RecordingProcessor("preprocessor")
    postprocessor = RecordingPostprocessor("postprocessor")
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    pipeline.load()

    with pytest.raises(BackendAdmissionError, match="reset admission timed out"):
        pipeline.reset()

    assert pipeline.state is PipelineState.READY
    assert preprocessor.reset_calls == 0
    assert postprocessor.reset_calls == 0
    pipeline.close()


def test_backend_reset_expired_before_execution_does_not_fail_backend(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = MockBackend("reset-policy", capabilities=capabilities, domains=ResourceDomainAdmissions())
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendAdmissionError):
        backend.reset(deadline=datetime.now(timezone.utc) - timedelta(seconds=1))

    assert backend.health().state is BackendState.READY
    assert backend.reset_calls == 0
    pipeline.close()


def test_deadline_after_backend_reset_before_processor_reset_fails_pipeline(tmp_path):
    class LateReturningBackend(MockBackend):
        def reset(self, deadline=None) -> None:
            self.reset_calls += 1
            time.sleep(0.02)

    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = LateReturningBackend("reset-policy", capabilities=capabilities, domains=ResourceDomainAdmissions())
    preprocessor = RecordingProcessor("preprocessor")
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        preprocessor=preprocessor,
    )
    pipeline.load()

    with pytest.raises(PipelineTimeoutError):
        pipeline.reset(deadline=datetime.now(timezone.utc) + timedelta(milliseconds=5))

    assert backend.health().ready
    assert preprocessor.reset_calls == 0
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_resetting_pipeline_does_not_change_other_pipeline_state_or_admission(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)
    first_backend = MockBackend("reset-first", capabilities=capabilities, domains=ResourceDomainAdmissions())
    second_backend = MockBackend("reset-second", capabilities=capabilities, domains=ResourceDomainAdmissions())
    first, _ = _native_pipeline(tmp_path / "first", "first", backend=first_backend)
    second, _ = _native_pipeline(tmp_path / "second", "second", backend=second_backend)
    manager = InferencePipelineManager((first, second))
    manager.start()
    first_backend.reset_release.clear()

    reset_thread, _, reset_errors = _thread_call(lambda: manager.reset("first"))
    assert first_backend.reset_started.wait(timeout=2)
    assert manager.health("first").state is PipelineState.RESETTING
    assert manager.health("second").state is PipelineState.READY

    result = manager.infer("second", _request("second"))
    assert result.pipeline_id == "second"
    assert second_backend.infer_entries == 1

    first_backend.reset_release.set()
    reset_thread.join(timeout=2)
    assert reset_errors == []
    assert manager.health("first").state is PipelineState.READY
    manager.close()


@pytest.mark.parametrize(
    ("action", "actual_chunk_size", "message"),
    [
        (np.full((1, 2, 6), np.nan), 2, "non-finite"),
        (np.zeros((1, 2, 5)), 2, "action dimension must be 6"),
        (np.zeros((1, 2, 6)), 1, "reported actual_chunk_size 1"),
        (np.zeros((2, 2, 6)), 2, "batch dimension must be one"),
        (np.zeros((1, 2, 6)), 0, "invalid actual_chunk_size"),
    ],
)
def test_pipeline_rejects_invalid_backend_outputs(tmp_path, action, actual_chunk_size, message):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    backend.result_factory = lambda request: _unsafe_backend_result(action, actual_chunk_size)
    pipeline.load()

    with pytest.raises(PipelineValidationError, match=message):
        pipeline.infer(_request())

    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_pipeline_revalidates_postprocessed_action(tmp_path):
    postprocessor = RecordingPostprocessor("postprocessor")

    def wrong_dimension(action: object) -> object:
        return np.zeros((1, 2, 5), dtype=np.float32)

    postprocessor.__call__ = wrong_dimension  # type: ignore[method-assign]
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        postprocessor=wrong_dimension,  # type: ignore[arg-type]
    )
    pipeline.load()

    with pytest.raises(PipelineValidationError, match="postprocessor action dimension must be 6"):
        pipeline.infer(_request())

    pipeline.close()


@pytest.mark.parametrize("failure_stage", ["backend_output", "postprocessor", "postprocessor_output"])
def test_stateful_backend_completion_followed_by_processing_failure_fails_pipeline(tmp_path, failure_stage):
    backend = MockBackend(
        "stateful-processing-failure",
        capabilities=BackendCapabilities(resettable=True, stateful=True),
    )
    postprocessor = None
    if failure_stage == "backend_output":
        backend.result_factory = lambda request: _unsafe_backend_result(np.zeros((1, 2, 5)), 2)
    elif failure_stage == "postprocessor":

        def postprocessor(action: object) -> object:
            raise RuntimeError("postprocessor failed")
    else:

        def postprocessor(action: object) -> object:
            return np.zeros((1, 2, 5), dtype=np.float32)

    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        postprocessor=postprocessor,  # type: ignore[arg-type]
    )
    pipeline.load()

    with pytest.raises((PipelineValidationError, RuntimeError)):
        pipeline.infer(_request())

    assert pipeline.state is PipelineState.FAILED
    with pytest.raises(PipelineNotReadyError):
        pipeline.infer(_request("after-processing-failure"))
    pipeline.close()


@pytest.mark.parametrize("failure_stage", ["decode", "raw_snapshot"])
def test_stateful_backend_completion_followed_by_internal_processing_failure_fails_pipeline(
    tmp_path, monkeypatch, failure_stage
):
    backend = MockBackend(
        "stateful-internal-failure",
        capabilities=BackendCapabilities(resettable=True, stateful=True),
    )
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()
    if failure_stage == "decode":
        monkeypatch.setattr(
            pipeline,
            "_decode_backend_action",
            lambda result: (_ for _ in ()).throw(RuntimeError("decode failed")),
        )
        capture_raw_action = False
    else:
        monkeypatch.setattr(
            "inference_service.pipeline.runtime._snapshot_action",
            lambda action: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        capture_raw_action = True

    expected_message = "decode failed" if failure_stage == "decode" else "snapshot failed"
    with pytest.raises(RuntimeError, match=expected_message):
        pipeline.infer(_request(), capture_raw_action=capture_raw_action)

    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_concurrent_result_is_discarded_after_another_request_fails_pipeline(tmp_path):
    second_in_postprocessor = threading.Event()
    release_second = threading.Event()

    def postprocessor(action: object) -> object:
        value = float(np.asarray(action).reshape(-1)[0])
        if value == 2.0:
            second_in_postprocessor.set()
            if not release_second.wait(timeout=2):
                raise RuntimeError("second postprocessor release timed out")
            return action
        if not second_in_postprocessor.wait(timeout=2):
            raise RuntimeError("second request did not reach postprocessor")
        raise RuntimeError("first postprocessor failed")

    backend = MockBackend(
        "stateful-concurrent-processing",
        capabilities=BackendCapabilities(resettable=True, stateful=True),
        domains=ResourceDomainAdmissions(),
    )
    pipeline, _ = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        postprocessor=postprocessor,  # type: ignore[arg-type]
    )
    pipeline.load()

    second_thread, second_results, second_errors = _thread_call(lambda: pipeline.infer(_request("second", value=2.0)))
    assert second_in_postprocessor.wait(timeout=2)
    first_thread, _, first_errors = _thread_call(lambda: pipeline.infer(_request("first", value=1.0)))
    first_thread.join(timeout=2)
    assert isinstance(first_errors[0], RuntimeError)
    assert pipeline.state is PipelineState.FAILED

    release_second.set()
    second_thread.join(timeout=2)

    assert second_results == []
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], PipelineNotReadyError)
    pipeline.close()


def test_stateful_protocol_backend_exception_fails_closed_even_if_health_stays_ready(tmp_path):
    class MisreportingBackend:
        name = "misreporting"
        capabilities = BackendCapabilities(resettable=False, stateful=True)

        def load(self, context: RuntimeContext) -> None:
            return None

        def infer(self, request: InferenceRequest) -> BackendResult:
            raise RuntimeError("state mutated before failure")

        def health(self):
            from inference_service.backends import BackendHealth

            return BackendHealth(state=BackendState.READY, ready=True)

        def close(self) -> None:
            return None

    pipeline = InferencePipeline("policy", _context(tmp_path / "bundle"), executor=MisreportingBackend())
    pipeline.load()

    with pytest.raises(RuntimeError, match="state mutated before failure"):
        pipeline.infer(_request())

    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_request_prompt_overrides_pipeline_default_task(tmp_path):
    preprocessor = RecordingProcessor("preprocessor")
    pipeline, backend = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        preprocessor=preprocessor,
        default_task="configured default",
    )
    pipeline.load()

    pipeline.infer(_request("default"))
    pipeline.infer(_request("override", prompt="request prompt"))

    assert preprocessor.calls[0]["task"] == "configured default"
    assert preprocessor.calls[1]["task"] == "request prompt"
    assert backend.requests[0].prompt == "configured default"
    assert backend.requests[1].prompt == "request prompt"
    pipeline.close()


def test_uncancellable_backend_overrun_finishes_synchronously_and_discards_late_result(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy", request_timeout=0.01)

    def slow_result(request: InferenceRequest) -> BackendResult:
        time.sleep(0.03)
        return BackendResult(
            action=np.zeros((1, 2, 6), dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=30.0,
        )

    backend.result_factory = slow_result
    pipeline.load()
    start = time.perf_counter()

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.infer(_request())

    elapsed = time.perf_counter() - start
    assert elapsed >= 0.03
    assert error.value.details["phase"] == "backend"
    assert error.value.details["backend_completed"] is True
    assert error.value.details["cancellation_supported"] is False
    assert error.value.details["timeout_mode"] == "cooperative_deadline_no_detached_threads"
    assert backend.active_infer == 0
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_stateful_backend_overrun_fails_pipeline_after_discarding_result(tmp_path):
    backend = MockBackend(
        "mock-policy",
        capabilities=BackendCapabilities(resettable=True, stateful=True),
    )
    pipeline, backend = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        backend=backend,
        request_timeout=0.01,
    )

    def slow_result(request: InferenceRequest) -> BackendResult:
        time.sleep(0.03)
        return BackendResult(
            action=np.zeros((1, 2, 6), dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=30.0,
        )

    backend.result_factory = slow_result
    pipeline.load()

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.infer(_request())

    assert error.value.details["backend_completed"] is True
    assert pipeline.state is PipelineState.FAILED
    with pytest.raises(PipelineNotReadyError):
        pipeline.infer(_request("after-timeout"))
    pipeline.close()


def test_timeout_while_waiting_for_backend_admission_never_invokes_second_call(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()
    backend.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: pipeline.infer(_request("first")))
    assert backend.wait_for_infer_entries(1)

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.infer(
            _request(
                "second",
                deadline=datetime.now(timezone.utc) + timedelta(seconds=0.05),
            )
        )

    assert error.value.details["phase"] == "backend_admission"
    assert error.value.details["backend_completed"] is False
    assert backend.infer_entries == 1
    backend.infer_release.set()
    first_thread.join(timeout=2)
    assert first_errors == []
    pipeline.close()


def test_request_deadline_takes_precedence_over_longer_pipeline_timeout(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy", request_timeout=1.0)

    def slow_result(request: InferenceRequest) -> BackendResult:
        time.sleep(0.03)
        return BackendResult(
            action=np.zeros((1, 2, 6), dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=30.0,
        )

    backend.result_factory = slow_result
    pipeline.load()
    deadline = datetime.now(timezone.utc) + timedelta(seconds=0.01)

    with pytest.raises(PipelineTimeoutError):
        pipeline.infer(_request(deadline=deadline))

    pipeline.close()


def test_partial_manager_startup_failure_closes_every_created_pipeline_once(tmp_path):
    first, first_backend = _native_pipeline(tmp_path / "first", "first")
    second, second_backend = _native_pipeline(tmp_path / "second", "second")
    second_backend.fail_load = True
    manager = InferencePipelineManager((first, second))

    with pytest.raises(PipelineManagerError) as error:
        manager.start()

    assert error.value.code == "startup_failed"
    assert error.value.pipeline_id == "second"
    assert first.state is PipelineState.CLOSED
    assert second.state is PipelineState.CLOSED
    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1
    with pytest.raises(PipelineManagerError) as route_error:
        manager.infer("first", _request())
    assert route_error.value.code == "manager_not_ready"
    manager.close()
    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1


def test_manager_shutdown_closes_all_backends_exactly_once(tmp_path):
    first, first_backend = _native_pipeline(tmp_path / "first", "first")
    second, second_backend = _native_pipeline(tmp_path / "second", "second")
    manager = InferencePipelineManager((first, second))
    manager.start()

    manager.close()
    manager.close()

    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1
    assert first.state is PipelineState.CLOSED
    assert second.state is PipelineState.CLOSED


def test_manager_can_close_one_pipeline_by_id_without_closing_another(tmp_path):
    first, first_backend = _native_pipeline(tmp_path / "first", "first")
    second, second_backend = _native_pipeline(tmp_path / "second", "second")
    manager = InferencePipelineManager((first, second))
    manager.start()

    manager.close("first")

    assert first.state is PipelineState.CLOSED
    assert second.state is PipelineState.READY
    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 0
    manager.infer("second", _request())
    manager.close()
    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1


def test_pipeline_state_machine_rejects_illegal_transitions():
    machine = PipelineStateMachine()

    with pytest.raises(PipelineTransitionError) as error:
        machine.transition(PipelineState.READY)

    assert error.value.details == {"source": "created", "target": "ready"}
    machine.transition(PipelineState.CLOSING)
    with pytest.raises(PipelineTransitionError):
        machine.transition(PipelineState.READY)
    machine.transition(PipelineState.CLOSED)
    with pytest.raises(PipelineTransitionError):
        machine.transition(PipelineState.LOADING)


def test_reset_state_blocks_new_admission_until_backend_health_is_ready(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)
    backend = MockBackend("resettable", capabilities=capabilities, domains=ResourceDomainAdmissions())
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()
    backend.reset_release.clear()

    reset_thread, _, reset_errors = _thread_call(pipeline.reset)
    assert backend.reset_started.wait(timeout=2)
    assert pipeline.state is PipelineState.RESETTING

    with pytest.raises(PipelineNotReadyError) as error:
        pipeline.infer(_request())

    assert error.value.details["state"] == "resetting"
    assert backend.infer_entries == 0
    backend.reset_release.set()
    reset_thread.join(timeout=2)
    assert reset_errors == []
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_reset_does_not_return_ready_when_backend_health_failed(tmp_path):
    capabilities = BackendCapabilities(resettable=True, stateful=True)

    class FailingResetBackend(MockBackend):
        def _reset(self) -> None:
            self.reset_calls += 1
            self.report_runtime_failure(BackendInferenceError("device lost", code="device_lost"))

    backend = FailingResetBackend("reset-failure", capabilities=capabilities, domains=ResourceDomainAdmissions())
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(PipelineNotReadyError, match="not ready after reset"):
        pipeline.reset()

    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_backend_health_failure_propagates_out_of_pipeline_ready(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()
    backend.report_runtime_failure(BackendInferenceError("worker exited", code="worker_exited", recoverable=True))

    diagnostics = pipeline.diagnostics()

    assert diagnostics.state is PipelineState.DEGRADED
    assert diagnostics.backend_health.state is BackendState.DEGRADED
    with pytest.raises(PipelineNotReadyError) as error:
        pipeline.infer(_request())
    assert error.value.details["state"] == "degraded"
    pipeline.close()


def test_backend_inference_failure_moves_pipeline_to_failed(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")

    def fail(request: InferenceRequest) -> BackendResult:
        raise BackendInferenceError("device lost", code="device_lost")

    backend.result_factory = fail
    pipeline.load()

    with pytest.raises(BackendInferenceError, match="device lost") as error:
        pipeline.infer(_request())

    assert error.value.operation_started is True
    assert error.value.outcome_known is True
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_preprocessor_failure_is_reported_before_backend_execution(tmp_path):
    class FailingProcessor(RecordingProcessor):
        def __call__(self, inputs):
            raise ValueError("invalid observation")

    pipeline, _backend = _native_pipeline(
        tmp_path / "bundle",
        "policy",
        preprocessor=FailingProcessor("preprocessor"),
    )
    pipeline.load()

    with pytest.raises(ValueError, match="invalid observation") as error:
        pipeline.infer(_request())

    assert error.value.operation_started is False
    assert error.value.outcome_known is True
    pipeline.close()


def test_backend_admission_rejection_preserves_nonrecoverable_certainty(tmp_path):
    class RejectingBackend(MockBackend):
        def _validate_request(self, request: InferenceRequest) -> None:
            raise BackendAdmissionError("priority unsupported", code="unsupported_priority")

    backend = RejectingBackend("rejecting")
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendAdmissionError, match="priority unsupported") as error:
        pipeline.infer(_request(priority=8))

    assert error.value.operation_started is False
    assert error.value.outcome_known is True
    assert error.value.recoverable is False
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_late_stateful_cancel_moves_pipeline_to_failed(tmp_path):
    class LateCancelBackend(MockBackend):
        def _cancel(self, request_id: str) -> None:
            super()._cancel(request_id)
            time.sleep(0.02)

    backend = LateCancelBackend(
        "late-cancel",
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
        domains=ResourceDomainAdmissions(),
    )
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendCancellationError, match="deadline expired during execution") as error:
        pipeline.cancel("request", deadline=datetime.now(timezone.utc) + timedelta(milliseconds=5))

    assert error.value.operation_started is True
    assert backend.cancel_calls == ["request"]
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_stateful_cancel_exception_moves_backend_and_pipeline_to_failed(tmp_path):
    class FailingCancelBackend(MockBackend):
        def _cancel(self, request_id: str) -> None:
            super()._cancel(request_id)
            raise RuntimeError("cancel acknowledgement lost")

    backend = FailingCancelBackend(
        "failing-cancel",
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
    )
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendCancellationError, match="outcome is unknown") as error:
        pipeline.cancel("request")

    assert error.value.operation_started is True
    assert error.value.outcome_known is False
    assert backend.health().state is BackendState.FAILED
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_stateful_cancel_expired_before_execution_keeps_pipeline_ready(tmp_path):
    backend = MockBackend(
        "expired-cancel",
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
        domains=ResourceDomainAdmissions(),
    )
    pipeline, _ = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendAdmissionError, match="deadline expired before execution") as error:
        pipeline.cancel("request", deadline=datetime.now(timezone.utc) - timedelta(seconds=1))

    assert error.value.operation_started is False
    assert backend.cancel_calls == []
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_close_is_terminal_idempotent_and_prevents_load_or_infer(tmp_path):
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.close()
    pipeline.close()

    assert pipeline.state is PipelineState.CLOSED
    assert backend.close_calls == 1
    with pytest.raises(PipelineLifecycleError) as load_error:
        pipeline.load()
    assert load_error.value.code == "invalid_load_state"
    with pytest.raises(PipelineNotReadyError) as infer_error:
        pipeline.infer(_request())
    assert infer_error.value.details["state"] == "closed"


# ---------------------------------------------------------------------------
# Compatibility facade architecture tests (task 2.6)
# ---------------------------------------------------------------------------


def test_facade_delegates_lifecycle_to_generic_pipeline_without_independent_state(tmp_path):
    """InferencePipeline wraps GenericModelPipeline and delegates all lifecycle state."""
    from inference_service.pipeline import GenericModelPipeline

    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    assert isinstance(pipeline._pipeline, GenericModelPipeline)
    assert not hasattr(pipeline, "_state_machine")
    assert not hasattr(pipeline, "_condition")
    assert not hasattr(pipeline, "_active_requests")

    pipeline.load()
    assert pipeline.state is pipeline._pipeline.state
    assert pipeline.state is PipelineState.READY

    pipeline.close()
    assert pipeline.state is PipelineState.CLOSED
    assert backend.close_calls == 1


def test_facade_diagnostics_reflect_unified_runtime_state(tmp_path):
    """Diagnostics are mapped from PipelineRuntimeDiagnostics, not independent state."""
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()

    runtime_diag = pipeline._pipeline.diagnostics()
    pipeline_diag = pipeline.diagnostics()

    assert pipeline_diag.pipeline_id == runtime_diag.pipeline_id
    assert pipeline_diag.state is runtime_diag.state
    assert pipeline_diag.backend_health == runtime_diag.executor_health
    assert pipeline_diag.active_requests == runtime_diag.active_requests
    assert pipeline_diag.deployment_fingerprint == runtime_diag.deployment.deployment_fingerprint
    assert pipeline_diag.backend == runtime_diag.deployment.backend
    assert pipeline_diag.default_task_configured is False
    pipeline.close()


def test_facade_cancel_delegates_to_unified_request_control_path(tmp_path):
    """cancel() marks the core control, delegates backend cancel, and discards the late result as request_canceled."""
    backend = MockBackend(
        "cancellable",
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
        domains=ResourceDomainAdmissions(),
    )
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    infer_started = threading.Event()
    infer_release = threading.Event()

    def slow_result(request: InferenceRequest) -> BackendResult:
        infer_started.set()
        infer_release.wait(timeout=2.0)
        return BackendResult(
            action=np.zeros((1, 2, 6), dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=10.0,
        )

    backend.result_factory = slow_result

    errors: list[BaseException] = []

    def run_infer():
        try:
            pipeline.infer(_request("cancel-target"))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_infer)
    thread.start()
    assert infer_started.wait(timeout=2.0)

    core = pipeline._pipeline._core
    assert "cancel-target" in core._request_controls

    pipeline.cancel("cancel-target")
    infer_release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "cancel-target" not in core._request_controls
    assert backend.cancel_calls == ["cancel-target"]
    assert len(errors) == 1
    assert getattr(errors[0], "code", None) == "request_canceled"
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_facade_cancel_non_cancellable_backend_discards_late_result_without_backend_cancel(tmp_path):
    """Non-cancellable backends are not interrupted; the late result is discarded as request_canceled."""
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()
    assert backend.capabilities.supports_cancellation is False

    infer_started = threading.Event()
    infer_release = threading.Event()

    def slow_result(request: InferenceRequest) -> BackendResult:
        infer_started.set()
        infer_release.wait(timeout=2.0)
        return BackendResult(
            action=np.zeros((1, 2, 6), dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=10.0,
        )

    backend.result_factory = slow_result

    errors: list[BaseException] = []

    def run_infer():
        try:
            pipeline.infer(_request("non-cancellable-target"))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_infer)
    thread.start()
    assert infer_started.wait(timeout=2.0)

    core = pipeline._pipeline._core
    assert "non-cancellable-target" in core._request_controls

    pipeline.cancel("non-cancellable-target")
    infer_release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "non-cancellable-target" not in core._request_controls
    assert backend.cancel_calls == []
    assert len(errors) == 1
    assert getattr(errors[0], "code", None) == "request_canceled"
    assert pipeline.state is PipelineState.READY
    pipeline.close()


def test_facade_preserves_deployment_identity_in_result_metadata(tmp_path):
    """Result deployment identity matches RuntimeContext, not an independent copy."""
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy")
    pipeline.load()

    result = pipeline.infer(_request())
    context = pipeline.runtime_context

    assert result.bundle == context.validated_manifest.manifest.bundle.name
    assert result.bundle_uuid == context.validated_manifest.manifest.bundle.uuid
    assert result.deployment == context.deployment_name
    assert result.deployment_uuid == context.deployment.uuid
    assert result.deployment_fingerprint == context.deployment_fingerprint
    assert result.backend == backend.name
    assert result.metadata["deployment_fingerprint"] == context.deployment_fingerprint
    assert result.metadata["backend"] == backend.name
    pipeline.close()


def test_facade_adapt_error_preserves_structured_policy_exception(tmp_path):
    """Policy executor adapt_error re-raises the original exception, not a fabricated result."""
    backend = MockBackend("failing", domains=ResourceDomainAdmissions())

    def fail(request: InferenceRequest) -> BackendResult:
        raise BackendInferenceError("device lost", code="device_lost")

    backend.result_factory = fail
    pipeline, backend = _native_pipeline(tmp_path / "bundle", "policy", backend=backend)
    pipeline.load()

    with pytest.raises(BackendInferenceError, match="device lost") as exc_info:
        pipeline.infer(_request())

    assert exc_info.value.code == "device_lost"
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()
