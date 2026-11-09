"""Real session builders preserve native owners across failed load and Close."""

from types import SimpleNamespace

import pytest

from inference_manifest import CompiledDeployment
from inference_service.backends import BackendState, RuntimeContext
from inference_service.backends.errors import BackendLifecycleError, BackendLoadError
from inference_service.model_sessions.ascend import AscendOmModelSession
from inference_service.model_sessions.hmm import HMMModelSession
from inference_service.model_sessions.onnx import OnnxRuntimeModelSession
from inference_service.model_sessions.rknn import RKNNModelSession


def compiled_context(tmp_path, backend, *, priority=False):
    runtime = {"ascend": "acl", "hmm": "tcim", "rknn": "rknn", "onnx": "onnx"}[backend]
    extension = "om" if backend == "ascend" else backend
    artifact = f"model.{extension}"
    (tmp_path / artifact).write_bytes(b"fake SDK artifact")
    deployment = CompiledDeployment.model_validate(
        {
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {
                "backend": backend,
                "target": {"runtime": runtime},
                "profile": ({"device_id": 0} if backend == "ascend" else {}),
            },
            "execution": ["model"],
            "artifacts": {"model": {"path": artifact, "format": extension}},
            "bindings": {
                "model": {
                    "inputs": [{"semantic": "input", "index": 0, "dtype": "float32", "shape": [1]}],
                    "outputs": [{"semantic": "output", "index": 0, "dtype": "float32", "shape": [1]}],
                }
            },
        }
    )
    return RuntimeContext(SimpleNamespace(deployment=deployment, bundle_root=tmp_path), priority_scheduling=priority)


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_stateful_ascend_entry_after_base_validation_proves_not_started(tmp_path, monkeypatch, stop):
    from datetime import datetime, timedelta, timezone

    from inference_service.model_sessions.ascend_stateful import StatefulAscendOmModelSession
    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        OutcomeState,
    )

    session = StatefulAscendOmModelSession(runtime_manager=object())
    session._state = BackendState.READY
    session._context = compiled_context(tmp_path, "ascend")
    session._models = {"model": SimpleNamespace(close=lambda: None)}
    session._context.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=()))
    now = [datetime(2026, 11, 3, tzinfo=timezone.utc)]
    context = ExecutionContext("entry", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    calls = []

    def validate(_request, ctx):
        if stop == "cancel":
            ctx.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    monkeypatch.setattr(session, "_validate_request", validate)
    monkeypatch.setattr(session, "_execute_role", lambda *_: (calls.append("native"), {})[1])
    with pytest.raises(ExecutionFailure) as caught:
        session.execute(ModelRequest(), context)
    assert caught.value.evidence.state is OutcomeState.NOT_STARTED
    assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated and not calls
    monkeypatch.setattr(session, "_validate_request", lambda *_: None)
    session.execute(ModelRequest(), ExecutionContext("valid"))
    assert calls == ["native"]
    session.close()


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("first_role", [True, False])
def test_role_control_evidence_uses_actual_execution_progress(tmp_path, monkeypatch, backend, stop, first_role):
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    context = compiled_context(tmp_path, backend)
    binding = context.deployment.bindings["model"]
    context.validated_manifest.deployment = context.deployment.model_copy(
        update={"execution": ("first", "second"), "bindings": {"first": binding, "second": binding}}
    )
    context.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=binding.outputs))
    session = HMMModelSession() if backend == "hmm" else RKNNModelSession()
    session._context, session._state = context, BackendState.READY
    calls = []

    def native(role):
        def run(*_, **__):
            calls.append(role)
            return {0: np.ones(1, np.float32)}

        return SimpleNamespace(execute=run, infer=run)

    setattr(session, "_modules" if backend == "hmm" else "_sessions", {r: native(r) for r in ("first", "second")})
    now = [datetime(2026, 10, 30, tzinfo=timezone.utc)]
    execution = ExecutionContext("controlled", Deadline.after(1, now=now[0], clock=lambda: now[0]))

    def stop_context():
        if stop == "cancel":
            execution.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    check = session._check_completed_context

    def between_roles(ctx, phase):
        check(ctx, phase)
        if phase == "model.first":
            stop_context()

    monkeypatch.setattr(session, "_check_completed_context", between_roles)
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=lambda: None))
    owner.load()
    request = ModelRequest({"input": np.ones(1, np.float32)})
    try:
        if first_role:
            from dataclasses import replace

            session._capabilities = replace(session.capabilities, stateful=True)
            stop_context()
            with pytest.raises(ExecutionFailure) as caught:
                session.execute_role("first", request.inputs, request, execution)
        else:
            with pytest.raises(ExecutionFailure) as caught:
                owner.execute(request, execution)
        assert caught.value.evidence.state is (OutcomeState.NOT_STARTED if first_role else OutcomeState.COMPLETED)
        assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated
        assert calls == ([] if first_role else ["first"])
        assert owner.ready
        monkeypatch.setattr(session, "_check_completed_context", check)
        assert owner.execute(request, ExecutionContext("next")).evidence.outcome_known
        assert calls[-2:] == ["first", "second"]
    finally:
        owner.close()


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
@pytest.mark.parametrize("completion_fails", [False, True])
def test_public_session_execute_keeps_execution_context_and_unknown_completion_quarantined(
    tmp_path, backend, completion_fails
):
    import numpy as np

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    calls = []

    class SDK:
        NPU_CORE_ALL = 7

        def load_rknn(self, _path):
            return 0

        def init_runtime(self, **kwargs):
            return 0

        def get_num_inputs(self):
            return 1

        def get_num_outputs(self):
            return 1

        def get_input_name(self, _index):
            return "input"

        def get_output_name(self, _index):
            return "output"

        def set_input(self, _name, value):
            self.value = value

        def run(self):
            calls.append("run")

        def sync(self):
            if completion_fails:
                raise RuntimeError("device sync failed")

        def get_output(self, _name):
            return self.value + 1

        def inference(self, inputs, **kwargs):
            calls.append("run")
            if completion_fails:
                raise RuntimeError("device inference failed")
            return [inputs[0] + 1]

        def release(self):
            calls.append("close")

    runtime = SimpleNamespace(load=lambda *_: SDK(), WeightManager=lambda **_: SimpleNamespace(release=lambda: None))
    session = (
        HMMModelSession(runtime_loader=lambda: runtime)
        if backend == "hmm"
        else RKNNModelSession(rknn_loader=lambda: SDK)
    )
    context = compiled_context(tmp_path, backend)
    context.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(outputs=context.deployment.bindings["model"].outputs)
    )
    session.load(context)
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=session.close))
    owner.load()
    request = ModelRequest({"input": np.ones(1, np.float32)})
    try:
        if completion_fails:
            with pytest.raises(ExecutionFailure) as failure:
                owner.execute(request, ExecutionContext("first"))
            assert not failure.value.evidence.outcome_known and not owner.ready
            with pytest.raises(ExecutionFailure):
                owner.execute(request, ExecutionContext("second"))
            assert calls == ["run"]
        else:
            np.testing.assert_array_equal(session.execute(request, ExecutionContext("direct"))["output"], [2])
            np.testing.assert_array_equal(owner.execute(request, ExecutionContext("handle")).outputs["output"], [2])
    finally:
        owner.close()


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_session_entry_control_rejects_before_native_call_without_retaining_admission(tmp_path, monkeypatch, stop):
    from inference_service.backends import ResourceDomainAdmissions
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
    from inference_service.unified_runtime.errors import DeadlineExceeded

    session = RKNNModelSession()
    session._context = compiled_context(tmp_path, "rknn")
    calls = []
    monkeypatch.setattr(session, "_load", lambda *_: None)
    monkeypatch.setattr(session, "_execute", lambda *_: calls.append("native") or {})
    monkeypatch.setattr(session, "_validate_request", lambda *_: None)
    monkeypatch.setattr(session, "_validate_values", lambda *_: None)
    session._context.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=()))
    provider = ResourceDomainAdmissions()
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=session,
            session=session,
            providers=RuntimeProviders(object(), provider),
            load_context=session._context,
        )
    )
    owner.load()
    original = ExecutionContext.check

    def check(context, phase="request_boundary"):
        if context.request_id == "first" and phase == "backend":
            if stop == "deadline":
                raise DeadlineExceeded(phase=phase)
            context.token.cancel()
        original(context, phase)

    monkeypatch.setattr(ExecutionContext, "check", check)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("first"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known and owner.ready and not calls
        owner.execute(ModelRequest(), ExecutionContext("next", Deadline.after(0.1)))
        assert calls == ["native"]
    finally:
        owner.close()


@pytest.mark.parametrize("priority", [False, True])
def test_standalone_ascend_and_context_inherit_same_priority_default(tmp_path, priority):
    from inference_service.backends import runtime_context_defaults

    streams = []
    acl = SimpleNamespace(
        rt=SimpleNamespace(
            device_get_stream_priority_range=lambda: (7, 0, 0),
            create_stream_with_config=lambda value, _: (streams.append(value) or value, 0),
            synchronize_stream=lambda _: 0,
            destroy_stream=lambda _: 0,
        )
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None, close=lambda: None)
    model = SimpleNamespace(load_descriptor=lambda: None, prepare_datasets=lambda **_: None, close=lambda: None)
    validated = compiled_context(tmp_path, "ascend").validated_manifest
    with runtime_context_defaults(priority_scheduling=priority):
        session = AscendOmModelSession(
            runtime_manager=SimpleNamespace(acquire=lambda _: lease), model_factory=lambda *_, **__: model
        )
        context = RuntimeContext(validated)
    session.load(context)
    assert session.health().state is BackendState.READY
    assert bool(streams) is priority
    session.close()


@pytest.mark.parametrize("backend,expire_at", [("hmm", "load"), ("hmm", "option"), ("rknn", "load")])
def test_expired_startup_budget_prevents_later_native_initialization(tmp_path, backend, expire_at):
    from datetime import datetime, timedelta, timezone

    from inference_service.unified_runtime import Deadline, ExecutionContext
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    now = [datetime(2026, 10, 30, tzinfo=timezone.utc)]
    deadline = Deadline.after(1, now=now[0], clock=lambda: now[0])
    context = compiled_context(tmp_path, backend)
    deployment = context.deployment
    context.validated_manifest.deployment = deployment.model_copy(
        update={
            "execution": ("first", "second"),
            "artifacts": {role: deployment.artifacts["model"] for role in ("first", "second")},
            "bindings": {role: deployment.bindings["model"] for role in ("first", "second")},
        }
    )
    context = RuntimeContext(context.validated_manifest)
    events = []

    class SDK:
        NPU_CORE_ALL = 7

        def load_rknn(self, *_):
            events.append("load")
            now[0] += timedelta(seconds=2)
            return 0

        def init_runtime(self, **_):
            events.append("init")
            return 0

        def get_num_inputs(self):
            return 1

        def get_num_outputs(self):
            return 1

        def get_input_name(self, _):
            return "input"

        def get_output_name(self, _):
            return "output"

        def release(self):
            events.append("release")

    def load(*_):
        native = SDK()
        native.load_rknn()
        return native

    runtime = SimpleNamespace(
        load=load, WeightManager=lambda **_: SimpleNamespace(release=lambda: events.append("weights"))
    )
    if expire_at == "option":

        def option(_):
            events.append("option")
            now[0] += timedelta(seconds=2)
            return object()

        runtime.Option = option
    session = (
        HMMModelSession(runtime_loader=lambda: runtime)
        if backend == "hmm"
        else RKNNModelSession(rknn_loader=lambda: SDK)
    )
    with lifecycle_scope(ExecutionContext("startup", deadline)), pytest.raises(BackendLoadError):
        session.load(context)
    assert events == [expire_at]
    session.close()
    assert events == [expire_at] + (["release"] if expire_at == "load" else []) + (
        ["weights"] if backend == "hmm" else []
    )


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
@pytest.mark.parametrize("partial_load", [False, True])
@pytest.mark.parametrize("not_started", [False, True])
def test_real_session_builder_retries_native_release_before_dependencies(tmp_path, backend, partial_load, not_started):
    events = []

    class SDK:
        NPU_CORE_ALL = 7

        def load_rknn(self, _path):
            return 0

        def init_runtime(self, **_kwargs):
            if partial_load:
                raise RuntimeError("initialization failed")
            return 0

        def get_num_inputs(self):
            if partial_load:
                raise RuntimeError("initialization failed")
            return 1

        def get_num_outputs(self):
            return 1

        def get_input_name(self, _index):
            return "input"

        def get_output_name(self, _index):
            return "output"

        def release(self):
            events.append("model")
            if events.count("model") <= 2:
                error = RuntimeError("release pending")
                if not_started:
                    from inference_service.unified_runtime import OutcomeEvidence

                    error.evidence = OutcomeEvidence.not_started("release")
                raise error

    def release_weights():
        assert events.count("model") == 3
        assert "weights" not in events
        events.append("weights")

    runtime = SimpleNamespace(load=lambda *_: SDK(), WeightManager=lambda **_: SimpleNamespace(release=release_weights))
    session = (
        HMMModelSession(runtime_loader=lambda: runtime)
        if backend == "hmm"
        else RKNNModelSession(rknn_loader=lambda: SDK)
    )
    context = compiled_context(tmp_path, backend)
    if partial_load:
        with pytest.raises(BackendLoadError) as failure:
            session.load(context)
        assert not failure.value.rollback_clean
    else:
        session.load(context)
        assert session.health().ready
        with pytest.raises(BackendLifecycleError) as failure:
            session.close()
        assert failure.value.close_pending
    assert events == ["model"]
    with pytest.raises(BackendLifecycleError) as failure:
        session.close()
    assert failure.value.close_pending and session.health().state is BackendState.FAILED
    if not not_started:
        assert events == ["model"]
        with pytest.raises(BackendLifecycleError) as pending:
            session.close()
        assert pending.value.close_pending and events == ["model"]
        return
    assert events == ["model", "model"]
    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED
    assert events == ["model"] * 3 + (["weights"] if backend == "hmm" else [])


@pytest.mark.parametrize("priority", [False, True])
@pytest.mark.parametrize("partial_load", [False, True])
def test_real_ascend_session_retries_model_then_streams_then_lease(tmp_path, priority, partial_load):
    events = []
    acl = SimpleNamespace(
        rt=SimpleNamespace(
            device_get_stream_priority_range=lambda: (7, 0, 0),
            create_stream_with_config=lambda priority, _: (priority, 0),
            synchronize_stream=lambda _: 0,
            destroy_stream=lambda stream: (events.append(f"stream:{stream}"), 0)[1],
        )
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None, close=lambda: events.append("lease"))

    class Model:
        def __init__(self, *_args, **_kwargs):
            pass

        def load_descriptor(self):
            if partial_load:
                raise RuntimeError("descriptor failed")

        def prepare_datasets(self, **_kwargs):
            pass

        def close(self):
            events.append("model")
            if events.count("model") <= 2:
                raise RuntimeError("model cleanup pending")

    session = AscendOmModelSession(
        runtime_manager=SimpleNamespace(acquire=lambda _: lease), model_factory=Model, priority_scheduling=priority
    )
    if partial_load:
        with pytest.raises(BackendLoadError) as failure:
            session.load(compiled_context(tmp_path, "ascend", priority=priority))
        assert not failure.value.rollback_clean
    else:
        session.load(compiled_context(tmp_path, "ascend", priority=priority))
        with pytest.raises(BackendLifecycleError) as failure:
            session.close()
        assert failure.value.close_pending
    with pytest.raises(BackendLifecycleError) as failure:
        session.close()
    assert failure.value.close_pending and events == ["model", "model"]
    session.close()
    session.close()
    assert events == ["model"] * 3 + ([f"stream:{i}" for i in reversed(range(8))] if priority else []) + ["lease"]
    assert session.health().state is BackendState.CLOSED


def test_partial_priority_pool_is_owned_before_first_stream_allocation(tmp_path):
    events = []
    attempts = 0

    def synchronize(_):
        nonlocal attempts
        attempts += 1
        return 1 if attempts <= 2 else 0

    acl = SimpleNamespace(
        rt=SimpleNamespace(
            device_get_stream_priority_range=lambda: (7, 0, 0),
            create_stream_with_config=lambda priority, _: (priority, int(priority == 1)),
            synchronize_stream=synchronize,
            destroy_stream=lambda stream: (events.append(f"stream:{stream}"), 0)[1],
        )
    )
    lease = SimpleNamespace(acl=acl, bind_current_thread=lambda: None, close=lambda: events.append("lease"))
    session = AscendOmModelSession(runtime_manager=SimpleNamespace(acquire=lambda _: lease), priority_scheduling=True)
    with pytest.raises(BackendLoadError) as failure:
        session.load(compiled_context(tmp_path, "ascend", priority=True))
    assert not failure.value.rollback_clean and not events
    with pytest.raises(BackendLifecycleError) as failure:
        session.close()
    assert failure.value.close_pending and not events
    session.close()
    assert events == ["stream:0", "lease"]


def test_hisilicon_session_keeps_worker_until_exit_is_confirmed():
    from inference_service.model_sessions.hisilicon import HisiliconModelSession

    events = []

    class Worker:
        def close(self):
            events.append("close")
            if len(events) == 1:
                raise RuntimeError("worker did not exit")

    session = HisiliconModelSession()
    worker = Worker()
    session._protocol = worker
    with pytest.raises(BackendLifecycleError) as failure:
        session.close()
    assert failure.value.close_pending and session._protocol is worker
    session.close()
    session.close()
    assert events == ["close", "close"] and session.health().state is BackendState.CLOSED


@pytest.mark.parametrize("backend", ["hmm", "rknn", "ascend", "onnx"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_control_between_base_check_and_first_native_role(tmp_path, monkeypatch, backend, stop):
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from inference_service.unified_runtime import (
        Deadline,
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    session = (
        AscendOmModelSession(runtime_manager=SimpleNamespace())
        if backend == "ascend"
        else {"hmm": HMMModelSession, "rknn": RKNNModelSession, "onnx": OnnxRuntimeModelSession}[backend]()
    )
    session._context = compiled_context(tmp_path, backend)
    session._state = BackendState.READY
    calls = []
    native = SimpleNamespace(
        execute=lambda *a, **kw: calls.append("native") or {}, infer=lambda *a: calls.append("native") or {}
    )
    setattr(
        session,
        {"hmm": "_modules", "rknn": "_sessions", "ascend": "_models", "onnx": "_sessions"}[backend],
        {"model": native},
    )
    now = [datetime(2026, 10, 31, tzinfo=timezone.utc)]
    execution = ExecutionContext("between", Deadline.after(1, now=now[0], clock=lambda: now[0]))

    def after_base_check(*_):
        if stop == "cancel":
            execution.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    monkeypatch.setattr(session, "_validate_request", after_base_check)
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), execution)
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated
        assert owner.ready and calls == []
    finally:
        owner.close()


@pytest.mark.parametrize("role_entry", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "empty", "native_failure"])
@pytest.mark.parametrize("backend", ["rknn", "hmm", "ascend"])
def test_input_rejection_does_not_quarantine_healthy_native_resources(
    tmp_path, monkeypatch, role_entry, invalid, backend
):
    import numpy as np

    from inference_service.backends import ResourceDomainAdmissions
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.codecs import build_execution_plan
    from inference_service.pipeline import ModelStage, SequentialModelExecutor
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

    runtime = compiled_context(tmp_path, backend)
    binding = runtime.deployment.bindings["model"]
    runtime.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=binding.outputs))
    calls, fail = [], [invalid == "native_failure"]

    def infer(inputs):
        calls.append(inputs)
        if fail[0]:
            raise BackendInferenceError("SDK completion unknown", operation_started=True, outcome_known=False)
        return {0: np.ones(1, np.float32)}

    if backend == "rknn":
        session = RKNNModelSession()
        session._sessions = {"model": SimpleNamespace(infer=infer)}
    elif backend == "hmm":
        from inference_service.backends.hmm.backend import HMMModule

        native = SimpleNamespace(
            get_num_inputs=lambda: 1,
            get_num_outputs=lambda: 1,
            get_input_name=lambda _: "input",
            get_output_name=lambda _: "output",
            set_input=lambda *_: None,
            run=lambda: infer({}),
            get_output=lambda _: np.ones(1, np.float32),
            release=lambda: None,
        )
        module = HMMModule(
            SimpleNamespace(load=lambda *_: native), "model", tmp_path / "model.hmm", binding, option=None
        )
        session = HMMModelSession()
        session._modules = {"model": module}
    else:
        from inference_service.backends.ascend.model import AclModel, AclTensorDescriptor, _DatasetBuffer

        acl = SimpleNamespace(
            mdl=SimpleNamespace(
                execute=lambda *_: (infer({}), 0)[1],
                destroy_dataset=lambda _: 0,
                unload=lambda _: 0,
            ),
            rt=SimpleNamespace(
                memcpy=lambda *_: 0, malloc_host=lambda _: (1, 0), free_host=lambda _: 0, free=lambda _: 0
            ),
            util=SimpleNamespace(bytes_to_ptr=lambda _: 1, ptr_to_bytes=lambda *_: np.ones(1, np.float32).tobytes()),
        )
        model = AclModel(
            SimpleNamespace(acl=acl, bind_current_thread=lambda: None), "model", tmp_path / "model.om", binding
        )
        model.model_id, model.input_dataset, model.output_dataset = 1, 2, 3
        model.input_descriptors = (AclTensorDescriptor(0, "input", np.dtype("float32"), (1,), 4),)
        model.output_descriptors = (AclTensorDescriptor(0, "output", np.dtype("float32"), (1,), 4),)
        model.input_buffers = [_DatasetBuffer(4, None, 4, True)]
        model.output_buffers = [_DatasetBuffer(5, None, 4, True)]
        model.output_host_buffers = [None]
        session = AscendOmModelSession(runtime_manager=object())
        session._models = {"model": model}
    monkeypatch.setattr(session, "_load", lambda *_: None)
    executor = SequentialModelExecutor(
        (ModelStage("model", session),),
        SimpleNamespace(adapt=lambda frame: {"output": frame.values["output"]}),
        execution_plan=build_execution_plan(("model",), {"model": binding}) if role_entry else None,
    )
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            session=session,
            load_context=runtime,
            providers=RuntimeProviders(object(), ResourceDomainAdmissions()),
        )
    )
    owner.load()
    inputs = {} if invalid == "missing" else {"input": np.ones(0 if invalid == "empty" else 1, np.float32)}
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(inputs), ExecutionContext("invalid"))
        if invalid == "native_failure":
            assert caught.value.evidence.state is OutcomeState.STARTED and not caught.value.evidence.outcome_known
            assert not owner.ready and len(calls) == 1
        else:
            assert caught.value.evidence.state is OutcomeState.NOT_STARTED and caught.value.evidence.outcome_known
            assert owner.ready and not calls
            if backend == "ascend" and invalid == "empty":
                assert "role 'model' input 0" in caught.value.message
                assert "got 0 bytes, expected 4 bytes" in caught.value.message
                assert caught.value.code == "input_size_mismatch"
            owner.execute(
                ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("valid", Deadline.after(0.1))
            )
            assert len(calls) == 1
    finally:
        owner.close()


@pytest.mark.parametrize("stateful", [False, True])
def test_onnx_later_role_cancel_preserves_prior_execution(tmp_path, stateful):
    import numpy as np

    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, OutcomeState

    runtime = compiled_context(tmp_path, "onnx")
    binding = runtime.deployment.bindings["model"]
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": ("first", "second"), "bindings": {"first": binding, "second": binding}}
    )
    runtime.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(inputs=binding.inputs, outputs=binding.outputs)
    )
    context = ExecutionContext("between-onnx-roles")
    calls = []

    def run(*_):
        calls.append("native")
        context.token.cancel()
        return [np.ones(1, np.float32)]

    session = OnnxRuntimeModelSession(stateful=stateful)
    session._context, session._state = runtime, BackendState.READY
    session._sessions = {
        r: SimpleNamespace(run=run, get_outputs=lambda: [SimpleNamespace(name="output")]) for r in ("first", "second")
    }
    session._input_names = {r: {"input": "input"} for r in ("first", "second")}
    session._output_names = {r: {"output": "output"} for r in ("first", "second")}
    try:
        with pytest.raises(ExecutionFailure) as caught:
            session.execute(ModelRequest({"input": np.ones(1, np.float32)}), context)
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known and caught.value.evidence.state_mutated is stateful
        assert calls == ["native"]
    finally:
        session.close()


@pytest.mark.parametrize("backend", ["rknn", "ascend", "onnx"])
def test_later_role_input_error_preserves_prior_execution(tmp_path, backend):
    import numpy as np

    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, OutcomeState

    runtime = compiled_context(tmp_path, backend)
    first = runtime.deployment.bindings["model"]
    second = first.model_copy(update={"inputs": (first.inputs[0].model_copy(update={"semantic": "later_input"}),)})
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": ("first", "second"), "bindings": {"first": first, "second": second}}
    )
    runtime.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(inputs=first.inputs, outputs=first.outputs)
    )
    calls = []
    session = {
        "rknn": RKNNModelSession,
        "ascend": lambda: AscendOmModelSession(runtime_manager=object()),
        "onnx": OnnxRuntimeModelSession,
    }[backend]()
    session._context, session._state = runtime, BackendState.READY
    attribute = "_models" if backend == "ascend" else "_sessions"
    if backend == "onnx":
        session._sessions = {
            role: SimpleNamespace(
                run=lambda *_: calls.append("native") or [np.ones(1, np.float32)],
                get_outputs=lambda: [SimpleNamespace(name="output")],
            )
            for role in ("first", "second")
        }
        session._input_names = {"first": {"input": "input"}, "second": {"later_input": "input"}}
        session._output_names = {role: {"output": "output"} for role in ("first", "second")}
    else:
        resources = {
            role: SimpleNamespace(
                **{
                    "infer" if backend == "rknn" else "execute": lambda *_, **__: (
                        calls.append("native") or {0: np.ones(1, np.float32)}
                    )
                }
            )
            for role in ("first", "second")
        }
        setattr(session, attribute, resources)
    try:
        with pytest.raises(BackendInferenceError if backend == "onnx" else ExecutionFailure) as caught:
            session.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("later-invalid"))
        assert caught.value.evidence.state is OutcomeState.COMPLETED and caught.value.evidence.outcome_known
        assert calls == ["native"]
    finally:
        setattr(session, attribute, {})
        session.close()


def test_hisilicon_worker_exit_preserves_code_and_requires_runtime_reload():
    import numpy as np

    from inference_manifest import ArtifactBindings, TensorBinding
    from inference_service.backends.hisilicon.sd3403_protocol import SD3403WorkerExitedError
    from inference_service.model_sessions.hisilicon import HisiliconModelSession
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
        RecoveryAction,
        RecoveryScope,
    )

    fault = SD3403WorkerExitedError("worker exited unexpectedly (returncode=1)")

    def execute(_):
        raise fault

    session = HisiliconModelSession()
    binding = TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,))
    session._context = SimpleNamespace(
        deployment=SimpleNamespace(bindings={"policy": ArtifactBindings(inputs=(binding,), outputs=(binding,))})
    )
    session._state = BackendState.READY
    session._action_binding = binding
    session._protocol = SimpleNamespace(execute=execute)
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("worker-exit"))
        failure = caught.value
        assert failure.code == "worker_exited" and failure.cause.__cause__ is fault
        assert failure.recovery.scope is RecoveryScope.RUNTIME and failure.recovery.action is RecoveryAction.RELOAD
        assert not failure.recoverable and failure.evidence.state is OutcomeState.STARTED
        assert not failure.evidence.outcome_known and not owner.ready
    finally:
        owner.close()


@pytest.mark.parametrize("roles", [("model",), ("first", "second")])
@pytest.mark.parametrize("invalid", ["empty", "missing", "native_failure"])
def test_rknn_last_role_output_failure_keeps_native_completion(tmp_path, roles, invalid):
    import numpy as np

    from inference_service.backends.errors import BackendInferenceError
    from inference_service.pipeline import ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest, ModelRuntimeHandle

    runtime = compiled_context(tmp_path, "rknn")
    binding = runtime.deployment.bindings["model"]
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": roles, "bindings": dict.fromkeys(roles, binding)}
    )
    runtime.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(inputs=binding.inputs, outputs=binding.outputs)
    )
    calls = []

    def infer(role):
        calls.append(role)
        if role == roles[-1]:
            if invalid == "native_failure":
                raise BackendInferenceError("completion unknown", operation_started=True, outcome_known=False)
            return {} if invalid == "missing" else {0: np.ones(0, np.float32)}
        return {0: np.ones(1, np.float32)}

    session = RKNNModelSession()
    session._context, session._state = runtime, BackendState.READY
    session._sessions = {role: SimpleNamespace(infer=lambda _, role=role: infer(role)) for role in roles}
    owner = ModelRuntimeHandle(
        SequentialModelExecutor((ModelStage("model", session),), SimpleNamespace(adapt=lambda _: {}))
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("outputs"))
        assert failure.value.evidence.outcome_known is (invalid != "native_failure")
        assert owner.ready is (invalid != "native_failure")
        assert calls == list(roles)
    finally:
        owner.close()
        session.close()


@pytest.mark.parametrize("role_entry", [False, True])
def test_hisilicon_success_preserves_request_and_target_metadata(tmp_path, role_entry):
    import numpy as np

    from inference_service.backends.hisilicon.sd3403_protocol import SD3403Response
    from inference_service.model_sessions.hisilicon import HisiliconModelSession
    from inference_service.unified_runtime import ExecutionContext, ModelRequest

    runtime = compiled_context(tmp_path, "rknn")
    binding = runtime.deployment.bindings["model"]
    binding = binding.model_copy(update={"outputs": (binding.outputs[0].model_copy(update={"semantic": "action"}),)})
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": ("policy",), "bindings": {"policy": binding}}
    )
    runtime.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(inputs=binding.inputs, outputs=binding.outputs)
    )
    session = HisiliconModelSession()
    session._context, session._state = runtime, BackendState.READY
    session._action_binding = binding.outputs[0]
    session._options = {"perf_enabled": False}
    session._protocol = SimpleNamespace(
        execute=lambda _: SD3403Response({0: np.ones(1, np.float32)}, 1200, 42),
        model_load_ms=3.0,
        close=lambda: None,
    )
    request, context = ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("business-request")
    try:
        outputs = (
            session.execute_role("policy", request.inputs, request, context)
            if role_entry
            else session.execute(request, context)
        )
        np.testing.assert_array_equal(outputs["action"], np.ones(1, np.float32))
        metadata = session.execution_metadata(context.request_id)
        assert metadata["request_id"] == context.request_id and metadata["worker_request_id"] == 42
        assert metadata["worker_latency_ms"] == 1.2 and metadata["target_soc"] == runtime.target.soc
    finally:
        session.close()


@pytest.mark.parametrize("invalid", ["empty", "none", "conversion", "native"])
def test_rknn_sdk_return_boundary_preserves_completion(tmp_path, invalid):
    import numpy as np

    from inference_service.pipeline import ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    calls = []

    class InvalidArray:
        def __array__(self, *args, **kwargs):
            raise ValueError("invalid SDK array")

    class SDK:
        NPU_CORE_ALL = 7

        def load_rknn(self, _):
            return 0

        def init_runtime(self, **kwargs):
            return 0

        def inference(self, **kwargs):
            calls.append("inference")
            if invalid == "native":
                raise RuntimeError("native work may remain")
            return {"empty": [], "none": None, "conversion": [InvalidArray()]}[invalid]

        def release(self):
            calls.append("release")

    context = compiled_context(tmp_path, "rknn")
    context.validated_manifest.manifest = SimpleNamespace(
        model=SimpleNamespace(outputs=context.deployment.bindings["model"].outputs)
    )
    session = RKNNModelSession(rknn_loader=lambda: SDK)
    session.load(context)
    owner = ModelRuntimeHandle(
        SequentialModelExecutor((ModelStage("model", session),), SimpleNamespace(adapt=lambda _: {}))
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("invalid"))
        assert failure.value.evidence.state is (OutcomeState.STARTED if invalid == "native" else OutcomeState.COMPLETED)
        assert failure.value.evidence.outcome_known is (invalid != "native")
        assert owner.ready is (invalid != "native")
        assert calls == ["inference"]
    finally:
        owner.close()
        session.close()


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
@pytest.mark.parametrize("entry", ["execute", "execute_role"])
def test_public_session_validation_after_native_return_keeps_completion(tmp_path, backend, entry):
    import numpy as np

    from inference_service.pipeline import ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import (
        ExecutionContext,
        ExecutionFailure,
        ModelRequest,
        ModelRuntimeHandle,
        OutcomeState,
    )

    calls = []

    class SDK:
        NPU_CORE_ALL = 7

        def load_rknn(self, _):
            return 0

        def init_runtime(self, **kwargs):
            return 0

        def get_num_inputs(self):
            return 1

        def get_num_outputs(self):
            return 1

        def get_input_name(self, _):
            return "input"

        def get_output_name(self, _):
            return "output"

        def set_input(self, name, value):
            self.value = value

        def run(self):
            calls.append("run")

        def sync(self):
            calls.append("sync")

        def get_output(self, _):
            return self.value

        def inference(self, inputs, **kwargs):
            calls.append("run")
            return inputs

        def release(self):
            pass

    context = compiled_context(tmp_path, backend)
    outputs = context.deployment.bindings["model"].outputs
    if entry == "execute":
        outputs = (outputs[0].model_copy(update={"shape": (2,)}),)
    context.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=outputs))
    session = (
        RKNNModelSession(rknn_loader=lambda: SDK)
        if backend == "rknn"
        else HMMModelSession(
            runtime_loader=lambda: SimpleNamespace(load=lambda *_: SDK(), WeightManager=lambda **_: None)
        )
    )
    session.load(context)

    def execute(request, execution):
        inputs = {"input": request.inputs["input"]}
        return session.execute_role("model", inputs, ModelRequest(inputs), execution)

    component = session if entry == "execute" else SimpleNamespace(execute=execute, capabilities=session.capabilities)
    owner = ModelRuntimeHandle(
        SequentialModelExecutor((ModelStage("model", component),), SimpleNamespace(adapt=lambda _: {}))
    )
    owner.load()
    try:
        values = np.ones(1, np.float64 if entry == "execute_role" else np.float32)
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": values}), ExecutionContext("validate"))
        assert caught.value.code == (
            "output_shape_mismatch" if entry == "execute" else "role_model_input_dtype_mismatch"
        )
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known and owner.ready
        assert calls == (["run", "sync"] if backend == "hmm" else ["run"])
    finally:
        owner.close()
        session.close()
