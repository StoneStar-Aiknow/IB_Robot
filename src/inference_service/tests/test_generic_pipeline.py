from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import (
    BackendCapabilities,
    BackendHealth,
    BackendInferenceError,
    BackendLoadError,
    BackendState,
    PartialLoadRollback,
    RuntimeContext,
)
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import ModelSession, ModelSessionBuilderRegistry
from inference_service.pipeline import (
    ExecutionControl,
    ExecutionError,
    GenericModelPipeline,
    InferenceStage,
    IterationStep,
    IterativeStage,
    PipelineLifecycleError,
    PipelineState,
    PipelineTimeoutError,
    PreprocessStage,
    ResultAdapter,
    SequentialModelExecutor,
    StageFrame,
)
from tests.manifest_fixtures import create_non_policy_bundle, make_non_policy_manifest, write_manifest


def _context(tmp_path: Path) -> RuntimeContext:
    bundle_files = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, bundle_files)
    write_manifest(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(tmp_path, "ascend"))


def _request(request_id: str = "request", *, deadline: datetime | None = None) -> NamedTensorRequest:
    return NamedTensorRequest(
        request_id,
        {"observation.image": np.zeros((1, 3, 384, 384), dtype=np.float32)},
        deadline=deadline,
    )


class _FakeSession(ModelSession):
    _next_id = 0

    def __init__(self, *, fail_load: bool = False, delay: float = 0.0) -> None:
        type(self)._next_id += 1
        super().__init__(
            f"fake-session-{type(self).__name__}-{type(self)._next_id}",
            BackendCapabilities(resettable=True, max_in_flight_per_instance=1),
        )
        self.fail_load = fail_load
        self.delay = delay
        self.execute_calls = 0
        self.reset_calls = 0
        self.close_calls = 0
        self.role_calls: list[str] = []
        self.loaded_resource = False

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        self.loaded_resource = True
        rollback.defer(setattr, self, "loaded_resource", False)
        if self.fail_load:
            raise BackendLoadError("fake load failed", code="fake_load_failed")

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        self.execute_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"tag_logits": np.zeros((1, 4585), dtype=np.float32)}

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        self.role_calls.append(role)
        assert request.request_id
        return {"tag_logits": np.zeros((1, 4585), dtype=np.float32)}

    def _reset(self) -> None:
        self.reset_calls += 1

    def _close(self) -> None:
        self.close_calls += 1
        self.loaded_resource = False


class _SessionExecutor:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.deadlines: list[datetime | None] = []

    def load(self, context: RuntimeContext) -> None:
        self.session.load(context)

    def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> object:
        assert isinstance(request, NamedTensorRequest)
        control.raise_if_canceled("fake-session")
        self.deadlines.append(deadline)
        return self.session.infer(replace(request, deadline=deadline))

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        del request_id, deadline

    @staticmethod
    def adapt_error(error: ExecutionError) -> object:
        if error.cause is not None:
            raise error.cause
        raise RuntimeError(error.message)

    def health(self) -> BackendHealth:
        return self.session.health()

    def reset(self, deadline: datetime | None = None) -> None:
        self.session.reset(deadline)

    def close(self) -> None:
        self.session.close()


def test_generic_pipeline_lifecycle_identity_reset_and_repeated_close(tmp_path) -> None:
    session = _FakeSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))

    pipeline.load()
    result = pipeline.execute(_request())

    assert result.deployment.bundle == "test-ram-plus"
    assert result.deployment.deployment == "ascend"
    assert result.metadata["pipeline_id"] == "generic"
    assert result.latency.total_ms >= result.latency.backend_ms
    diagnostics = pipeline.diagnostics()
    assert diagnostics.ready
    assert diagnostics.state is PipelineState.READY
    assert diagnostics.active_requests == 0
    assert diagnostics.deployment.deployment_fingerprint

    pipeline.reset()
    assert session.reset_calls == 1
    pipeline.close()
    pipeline.close()
    assert session.close_calls == 1
    assert pipeline.state is PipelineState.CLOSED


def test_generic_pipeline_rolls_back_failed_executor_load(tmp_path) -> None:
    session = _FakeSession(fail_load=True)
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))

    with pytest.raises(BackendLoadError, match="fake load failed"):
        pipeline.load()

    assert not session.loaded_resource
    assert session.close_calls == 1
    assert pipeline.state is PipelineState.FAILED
    pipeline.close()


def test_generic_pipeline_propagates_one_absolute_deadline(tmp_path) -> None:
    session = _FakeSession()
    executor = _SessionExecutor(session)
    pipeline = GenericModelPipeline("generic", _context(tmp_path), executor, request_timeout=1.0)
    pipeline.load()
    request_deadline = datetime.now(timezone.utc) + timedelta(seconds=10)

    pipeline.execute(_request(deadline=request_deadline))

    assert executor.deadlines[0] is not None
    assert executor.deadlines[0] < request_deadline
    pipeline.close()


def test_generic_pipeline_rejects_expired_deadline_before_session_execution(tmp_path) -> None:
    session = _FakeSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))
    pipeline.load()

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.execute(_request(deadline=datetime.now(timezone.utc) - timedelta(seconds=1)))

    assert error.value.code == "deadline_exceeded"
    assert error.value.details["cancellation_supported"] is False
    assert session.execute_calls == 0
    assert pipeline.health().ready
    pipeline.close()


def test_generic_pipeline_reports_injected_cancellation_capability(tmp_path) -> None:
    session = _FakeSession()
    pipeline = GenericModelPipeline(
        "generic",
        _context(tmp_path),
        _SessionExecutor(session),
        supports_cancellation=True,
    )
    pipeline.load()

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.execute(_request(deadline=datetime.now(timezone.utc) - timedelta(seconds=1)))

    assert error.value.details["cancellation_supported"] is True
    pipeline.close()


def test_generic_pipeline_discards_late_result_without_failing_session(tmp_path) -> None:
    session = _FakeSession(delay=0.02)
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))
    pipeline.load()

    with pytest.raises(PipelineTimeoutError) as error:
        pipeline.execute(_request(deadline=datetime.now(timezone.utc) + timedelta(milliseconds=5)))

    assert error.value.details["backend_completed"]
    assert session.execute_calls == 1
    assert session.health().ready
    pipeline.close()


def test_generic_pipeline_reflects_executor_failure_health(tmp_path) -> None:
    class FailingSession(_FakeSession):
        def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
            raise BackendInferenceError("runtime lost", code="runtime_lost")

    session = FailingSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))
    pipeline.load()

    with pytest.raises(BackendInferenceError, match="runtime lost"):
        pipeline.execute(_request())

    diagnostics = pipeline.health()
    assert diagnostics.state is PipelineState.FAILED
    assert diagnostics.executor_health.state is BackendState.FAILED
    pipeline.close()


def test_model_session_execution_validates_role_abi_and_deadline(tmp_path) -> None:
    session = _FakeSession()
    session.load(_context(tmp_path))
    request = _request()

    with session.execution(request) as execution:
        outputs = execution.invoke("model", request.inputs)

    assert outputs["tag_logits"].shape == (1, 4585)
    assert session.role_calls == ["model"]
    assert session.health().last_successful_inference_time is not None

    expired = replace(request, request_id="expired", deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(Exception) as error, session.execution(expired):
        pass
    assert getattr(error.value, "code", None) == "deadline_exceeded"
    assert session.role_calls == ["model"]
    session.close()


def test_generic_pipeline_close_reports_error_after_state_is_closed(tmp_path) -> None:
    class FailingCloseSession(_FakeSession):
        def _close(self) -> None:
            super()._close()
            raise RuntimeError("close failed")

    session = FailingCloseSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))
    pipeline.load()

    with pytest.raises(PipelineLifecycleError, match="close failed"):
        pipeline.close()

    assert pipeline.state is PipelineState.CLOSED
    pipeline.close()


def test_generic_pipeline_pipeline_admission_does_not_bypass_session_serialization(tmp_path) -> None:
    class SerialSession(_FakeSession):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
            with self.lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                time.sleep(0.02)
                return super()._execute(request)
            finally:
                with self.lock:
                    type(self).active -= 1

    session = SerialSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), _SessionExecutor(session))
    pipeline.load()
    errors: list[Exception] = []

    def execute(index: int) -> None:
        try:
            pipeline.execute(_request(f"request-{index}"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=execute, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert SerialSession.max_active == 1
    pipeline.close()


def test_sequential_executor_runs_concrete_stages_and_adapts_result(tmp_path) -> None:
    class Adapter:
        def adapt(self, frame: StageFrame) -> object:
            return frame.values["result"]

        def adapt_error(self, error: ExecutionError) -> object:
            if error.cause is not None:
                raise error.cause
            raise RuntimeError(error.message)

    class ResultStage:
        def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
            del deadline
            frame.values["result"] = frame.values["input"] + 1

    assert isinstance(PreprocessStage(lambda values: {"input": values["input"]}), InferenceStage)
    assert isinstance(Adapter(), ResultAdapter)
    executor = SequentialModelExecutor(
        (PreprocessStage(lambda values: {"input": values["input"]}), ResultStage()),
        Adapter(),
    )
    pipeline = GenericModelPipeline("generic", _context(tmp_path), executor)
    pipeline.load()
    frame_result = executor.execute(
        NamedTensorRequest("stage", {"input": np.float32(2.0)}),
        deadline=None,
        control=ExecutionControl("stage"),
    )
    assert frame_result == np.float32(3.0)
    pipeline.close()


def test_direct_model_stage_does_not_nest_model_session_admission(tmp_path) -> None:
    from inference_service.pipeline import ModelResultAdapter, ModelStage

    session = _FakeSession()
    executor = SequentialModelExecutor(
        (ModelStage("model", session),),
        ModelResultAdapter(),
        components=(session,),
    )
    pipeline = GenericModelPipeline("direct", _context(tmp_path), executor)

    pipeline.load()
    result = pipeline.execute(_request("direct"))

    assert session.execute_calls == 1
    assert result.outputs["tag_logits"].shape == (1, 4585)
    pipeline.close()


def test_model_session_builder_registry_validates_and_selects_manifest_key(tmp_path) -> None:
    context = _context(tmp_path)
    registry = ModelSessionBuilderRegistry()
    registry.register("perception", "ram_plus", "", "ascend", lambda _context: _FakeSession())

    session = registry.create(context)

    assert isinstance(session, _FakeSession)


def test_model_session_builder_registry_fails_closed_for_missing_builder(tmp_path) -> None:
    registry = ModelSessionBuilderRegistry()

    with pytest.raises(BackendLoadError) as error:
        registry.create(_context(tmp_path))

    assert error.value.code == "session_builder_unavailable"


def test_sequential_executor_holds_one_session_execution_scope_across_stages() -> None:
    events: list[str] = []

    class Session:
        @contextmanager
        def execution(self, request: NamedTensorRequest):
            events.append(f"enter:{request.request_id}")
            try:
                yield object()
            finally:
                events.append(f"exit:{request.request_id}")

    class Stage:
        def __init__(self, name: str) -> None:
            self.name = name

        def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
            del frame, deadline
            events.append(self.name)

    class Adapter:
        @staticmethod
        def adapt(frame: StageFrame) -> object:
            del frame
            return "result"

        @staticmethod
        def adapt_error(error: ExecutionError) -> object:
            raise RuntimeError(error.message)

    session = Session()
    executor = SequentialModelExecutor((Stage("first"), Stage("second")), Adapter(), components=(session,))

    assert executor.execute(_request("scoped"), deadline=None, control=ExecutionControl("scoped")) == "result"
    assert events == ["enter:scoped", "first", "second", "exit:scoped"]


def test_session_execution_uses_request_inputs_instead_of_internal_frame_values() -> None:
    entered_inputs: list[Mapping[str, object]] = []

    class Session:
        @contextmanager
        def execution(self, request: NamedTensorRequest):
            entered_inputs.append(request.inputs)
            yield object()

    request = NamedTensorRequest("scoped", {"observation.state": np.zeros((1, 6), dtype=np.float32)})
    frame = StageFrame(request, values={**request.inputs, "_backend_started": True})

    frame.open_session_execution(Session(), request, None)
    frame.close()

    assert list(entered_inputs[0]) == ["observation.state"]


def test_iterative_stage_preserves_step_order_and_updates(tmp_path) -> None:
    events: list[object] = []

    class State:
        def initialize(self, frame: StageFrame) -> None:
            events.append("initialize")

        def prepare_step(self, frame: StageFrame, step: IterationStep) -> None:
            events.append(("prepare", step.index))

        def update(self, frame: StageFrame, step: IterationStep) -> None:
            events.append(("update", step.index))

        def finalize(self, frame: StageFrame) -> None:
            events.append("finalize")

    class Body:
        def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
            del deadline
            events.append("body")

    stage = IterativeStage((IterationStep(0, 1.0, 0.5), IterationStep(1, 0.5, 0.5)), (Body(),), State())
    frame = StageFrame(NamedTensorRequest("iterative", {"input": np.float32(1.0)}))
    stage.execute(frame, deadline=None)
    assert events == [
        "initialize",
        ("prepare", 0),
        "body",
        ("update", 0),
        ("prepare", 1),
        "body",
        ("update", 1),
        "finalize",
    ]
    frame.close()


def test_iterative_stage_stops_between_steps_when_canceled() -> None:
    control = ExecutionControl("iterative-cancel")
    events: list[object] = []

    class State:
        def initialize(self, frame: StageFrame) -> None:
            events.append("initialize")

        def prepare_step(self, frame: StageFrame, step: IterationStep) -> None:
            events.append(("prepare", step.index))

        def update(self, frame: StageFrame, step: IterationStep) -> None:
            events.append(("update", step.index))

        def finalize(self, frame: StageFrame) -> None:
            events.append("finalize")

    class Body:
        def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
            del deadline
            events.append("body")
            frame.control.cancel()

    stage = IterativeStage((IterationStep(0, 1.0, 0.5), IterationStep(1, 0.5, 0.5)), (Body(),), State())
    frame = StageFrame(
        NamedTensorRequest("iterative-cancel", {"input": np.float32(1.0)}),
        control=control,
    )

    with pytest.raises(Exception) as error:
        stage.execute(frame, deadline=None)

    assert getattr(error.value, "code", None) == "request_canceled"
    assert events == ["initialize", ("prepare", 0), "body", ("update", 0)]
    frame.close()


def test_generic_pipeline_cancel_marks_active_control_and_adapts_canceled_error(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class CancellableExecutor(_SessionExecutor):
        def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> object:
            started.set()
            release.wait(timeout=1.0)
            control.raise_if_canceled("test")
            return super().execute(request, deadline=deadline, control=control)

        def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
            del request_id, deadline
            release.set()

    session = _FakeSession()
    pipeline = GenericModelPipeline("generic", _context(tmp_path), CancellableExecutor(session))
    pipeline.load()
    errors: list[Exception] = []

    def execute() -> None:
        try:
            pipeline.execute(_request("cancel-me"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    assert started.wait(timeout=1.0)
    pipeline.cancel("cancel-me")
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert getattr(errors[0], "code", None) == "request_canceled"
    assert pipeline.health().ready
    pipeline.close()


def test_sequential_executor_cancel_delegates_only_to_supported_component() -> None:
    calls: list[tuple[str, datetime | None]] = []

    class Component:
        capabilities = replace(BackendCapabilities(), supports_cancellation=True)

        @staticmethod
        def cancel(request_id: str, deadline: datetime | None = None) -> None:
            calls.append((request_id, deadline))

    class Adapter:
        @staticmethod
        def adapt(frame: StageFrame) -> object:
            return frame.values

        @staticmethod
        def adapt_error(error: ExecutionError) -> object:
            raise RuntimeError(error.message)

    class Stage:
        @staticmethod
        def execute(frame: StageFrame, *, deadline: datetime | None) -> None:
            del frame, deadline

    deadline = datetime.now(timezone.utc) + timedelta(seconds=1)
    executor = SequentialModelExecutor((Stage(),), Adapter(), components=(Component(),))
    executor.cancel("request", deadline)
    assert calls == [("request", deadline)]
