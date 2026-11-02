"""Native input and host output failures preserve actual execution progress."""

from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends import ResourceDomainAdmissions
from inference_service.backends.hisilicon.sd3403_protocol import SD3403Response
from inference_service.backends.hmm.backend import HMMModule
from inference_service.model_sessions.ascend import AscendOmModelSession
from inference_service.model_sessions.hisilicon import HisiliconModelSession
from inference_service.model_sessions.hmm import HMMModelSession
from inference_service.model_sessions.onnx import OnnxRuntimeModelSession
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeState,
    RecoveryAction,
    RecoveryRequirement,
    RuntimeAssembly,
    RuntimeProviders,
)
from test.test_session_cleanup_ownership import compiled_context


def _session(tmp_path, monkeypatch, backend, failure_at=None):
    runtime = compiled_context(tmp_path, "rknn" if backend == "hisilicon" else backend)
    role = "policy" if backend == "hisilicon" else "model"
    binding = runtime.deployment.bindings["model"]
    if backend == "hisilicon":
        binding = binding.model_copy(
            update={"outputs": (binding.outputs[0].model_copy(update={"semantic": "action"}),)}
        )
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": (role,), "bindings": {role: binding}}
    )
    runtime.validated_manifest.manifest = SimpleNamespace(model=SimpleNamespace(outputs=binding.outputs))
    calls = []
    error = RuntimeError("host output processing failed")

    def fail(*_, **__):
        raise error

    def run(*_, **__):
        calls.append("run")
        return [np.ones(1, np.float32)]

    if backend == "hisilicon":
        session = HisiliconModelSession()
        session._action_binding = binding.outputs[0]
        session._options = {"perf_enabled": False}
        session._protocol = SimpleNamespace(
            execute=lambda *_: SD3403Response({} if failure_at else {0: run()[0]}, 100, 1),
            model_load_ms=1.0,
            close=lambda: None,
        )
        if failure_at:
            session._protocol.execute = lambda *_: (run(), SD3403Response({}, 100, 1))[1]
    elif backend == "onnx":
        session = OnnxRuntimeModelSession(stateful=True)
        session._sessions = {
            role: SimpleNamespace(
                run=run,
                get_outputs=fail if failure_at == "mapping" else lambda: [SimpleNamespace(name="output")],
            )
        }
        session._input_names = {role: {"input": "input"}}
        session._output_names = {role: {"output": "output"}}
        if failure_at == "index":
            session._sessions[role].run = lambda *_: (run(), [])[1]
        elif failure_at == "state":
            session._state_outputs = {
                role: {"state": binding.outputs[0].model_copy(update={"semantic": "missing_state"})}
            }
    elif backend == "ascend":
        session = AscendOmModelSession(runtime_manager=object())
        session._models = {role: SimpleNamespace(execute=lambda *_, **__: {0: run()[0]}, close=lambda: None)}
        if failure_at == "mapping":
            monkeypatch.setattr(session, "_bound_output", fail)
        elif failure_at == "capture":
            monkeypatch.setattr(session, "_capture_role_outputs", fail)
    else:
        native = SimpleNamespace(
            get_num_inputs=lambda: 1,
            get_num_outputs=lambda: 1,
            get_input_name=lambda _: "input",
            get_output_name=lambda _: "output",
            set_input=lambda *_: calls.append("set_input"),
            run=run,
            sync=lambda: calls.append("sync"),
            get_output=fail
            if failure_at == "get"
            else lambda _: SimpleNamespace(numpy=fail if failure_at else lambda: np.ones(1, np.float32)),
            release=lambda: None,
        )
        session = HMMModelSession()
        session._modules = {
            role: HMMModule(
                SimpleNamespace(load=lambda *_: native),
                role,
                tmp_path / "model.hmm",
                binding,
                option=None,
            )
        }
    monkeypatch.setattr(session, "_load", lambda *_: None)
    return session, runtime, calls, error, role


def test_hmm_overflow_rejects_before_sdk_and_allows_next_request(tmp_path, monkeypatch):
    session, runtime, calls, _, _ = _session(tmp_path, monkeypatch, "hmm")
    providers = RuntimeProviders(object(), ResourceDomainAdmissions())
    owner = ModelRuntimeHandle(RuntimeAssembly(session, session=session, load_context=runtime, providers=providers))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": [10**1000]}), ExecutionContext("overflow"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known
        assert calls == [] and owner.ready and not owner._admission._retained
        owner.execute(ModelRequest({"input": [1.0]}), ExecutionContext("valid", Deadline.after(0.1)))
        assert calls == ["set_input", "run", "sync"] and owner.ready
    finally:
        owner.close()
        providers.close()


@pytest.mark.parametrize("backend", ["hisilicon", "onnx"])
@pytest.mark.parametrize("role_entry", [False, True])
def test_missing_input_keeps_native_session_and_domain_available(tmp_path, monkeypatch, backend, role_entry):
    session, runtime, calls, _, role = _session(tmp_path, monkeypatch, backend)
    domains = ResourceDomainAdmissions()

    def execute(request, context):
        if role_entry:
            return session.execute_role(role, request.inputs, request, context)
        return session.execute(request, context)

    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=execute),
            session=session,
            load_context=runtime,
            providers=RuntimeProviders(object(), domains),
        )
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("missing"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated
        assert calls == [] and owner.ready
        owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("valid", Deadline.after(0.1)))
        assert calls == ["run"] and owner.ready
    finally:
        owner.close()


@pytest.mark.parametrize(
    "backend,failure_at",
    [
        ("ascend", "mapping"),
        ("ascend", "capture"),
        ("hmm", "get"),
        ("hmm", "numpy"),
        ("hisilicon", "missing_action"),
        ("onnx", "mapping"),
        ("onnx", "index"),
        ("onnx", "state"),
    ],
)
@pytest.mark.parametrize("role_entry", [False, True])
def test_native_completion_survives_session_output_processing(tmp_path, monkeypatch, backend, failure_at, role_entry):
    session, runtime, calls, _, role = _session(tmp_path, monkeypatch, backend, failure_at)
    domains = ResourceDomainAdmissions()

    def execute(request, context):
        if role_entry:
            return session.execute_role(role, request.inputs, request, context)
        return session.execute(request, context)

    stateful = session.capabilities.stateful
    streaming = (
        SimpleNamespace(
            open_stream=lambda _: None,
            step=lambda _, request, context: execute(request, context),
            reset_stream=lambda *_: session.reset(),
            close_stream=lambda *_: None,
        )
        if stateful
        else None
    )
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=execute),
            session=session,
            load_context=runtime,
            providers=RuntimeProviders(object(), domains),
            streaming_runtime=streaming,
            stateful=stateful,
            resettable=stateful,
            state_scope="stream" if stateful else "request",
            state_bank_mode="runtime_exclusive" if stateful else None,
        )
    )
    owner.load()
    stream = owner.open_stream(ExecutionContext("open")) if stateful else None
    peer = domains.register(session.capabilities.resource_domain, 1) if session.capabilities.resource_domain else None
    try:
        with pytest.raises(ExecutionFailure) as caught:
            request = ModelRequest({"input": np.ones(1, np.float32)})
            if stateful:
                owner.step(stream, request, ExecutionContext("output"))
            else:
                owner.execute(request, ExecutionContext("output"))
        assert calls == (["set_input", "run", "sync"] if backend == "hmm" else ["run"])
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known
        assert caught.value.evidence.state_mutated is session.capabilities.stateful
        assert owner.ready is not stateful
        if stateful:
            assert caught.value.recovery.action is RecoveryAction.RESET_RUNTIME
            with pytest.raises(ExecutionFailure, match="recovery_required"):
                owner.step(stream, request, ExecutionContext("blocked"))
        assert not owner._admission._retained
        assert owner._admission._instance_gate.acquire(blocking=False)
        owner._admission._instance_gate.release()
        if peer:
            peer.acquire(context=ExecutionContext("peer", Deadline.after(0.1)))
            peer.release()
    finally:
        if peer:
            peer.close()
        owner.close()


def test_completed_output_error_keeps_explicit_recovery_requirement(tmp_path, monkeypatch):
    session, runtime, _, error, _ = _session(tmp_path, monkeypatch, "ascend", "capture")
    error.recovery = RecoveryRequirement("runtime", "reload")
    session.load(runtime)
    try:
        with pytest.raises(RuntimeError) as caught:
            session.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("capture"))
        assert caught.value is error and error.evidence.outcome_known
        assert error.recovery == RecoveryRequirement("runtime", "reload")
    finally:
        session.close()


@pytest.mark.parametrize("failed_role", ["first", "second"])
def test_ascend_input_capture_failure_preserves_progress_and_shared_capacity(tmp_path, monkeypatch, failed_role):
    session, runtime, calls, _, _ = _session(tmp_path, monkeypatch, "ascend")
    binding = runtime.deployment.bindings["model"]
    runtime.validated_manifest.deployment = runtime.deployment.model_copy(
        update={"execution": ("first", "second"), "bindings": {"first": binding, "second": binding}}
    )

    def execute(role):
        calls.append(role)
        return {0: np.ones(1, np.float32)}

    session._models = {
        role: SimpleNamespace(execute=lambda *_, role=role, **__: execute(role), close=lambda: None)
        for role in ("first", "second")
    }
    blocked = [True]

    def capture(name, value):
        if blocked[0] and name == f"{failed_role}_in_input":
            np.save(tmp_path / "missing-directory" / "input.npy", value)

    session._diagnostic_capture = capture
    domains = ResourceDomainAdmissions()
    providers = RuntimeProviders(object(), domains)
    owner = ModelRuntimeHandle(RuntimeAssembly(session, session=session, load_context=runtime, providers=providers))
    owner.load()
    peer = domains.register(session.capabilities.resource_domain, 1)
    try:
        request = ModelRequest({"input": np.ones(1, np.float32)})
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(request, ExecutionContext("capture-fails"))
        assert caught.value.evidence.state is (
            OutcomeState.NOT_STARTED if failed_role == "first" else OutcomeState.COMPLETED
        )
        assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated
        assert calls == ([] if failed_role == "first" else ["first"])
        assert owner.ready and not owner._admission._retained
        peer.acquire(context=ExecutionContext("peer", Deadline.after(0.1)))
        peer.release()
        blocked[0] = False
        calls.clear()
        owner.execute(request, ExecutionContext("valid", Deadline.after(0.1)))
        assert calls == ["first", "second"] and owner.ready
    finally:
        peer.close()
        owner.close()
        providers.close()


@pytest.mark.parametrize("value", [["bad"], [[1], [1, 2]], np.ones(2, np.float32)])
@pytest.mark.parametrize("role_entry", [False, True])
def test_hisilicon_input_validation_precedes_worker_submission(tmp_path, monkeypatch, value, role_entry):
    session, runtime, calls, _, role = _session(tmp_path, monkeypatch, "hisilicon")
    executor = (
        SimpleNamespace(execute=lambda request, context: session.execute_role(role, request.inputs, request, context))
        if role_entry
        else session
    )
    owner = ModelRuntimeHandle(RuntimeAssembly(executor, session=session, load_context=runtime))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": value}), ExecutionContext("invalid"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert owner.ready and calls == []
        owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), ExecutionContext("valid"))
        assert calls == ["run"]
    finally:
        owner.close()


def test_hmm_set_input_failure_is_not_started_and_allows_retry(tmp_path, monkeypatch):
    session, runtime, calls, _, _ = _session(tmp_path, monkeypatch, "hmm")
    # Exercise the actual HMMModule, not only the Session's boundary.
    module = next(iter(session._modules.values()))
    native = module._module
    original = native.set_input
    native.set_input = lambda *_: (_ for _ in ()).throw(ValueError("invalid native input"))
    owner = ModelRuntimeHandle(RuntimeAssembly(session, session=session, load_context=runtime))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": [1.0]}), ExecutionContext("invalid"))
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert owner.ready and calls == []
        native.set_input = original
        owner.execute(ModelRequest({"input": [1.0]}), ExecutionContext("valid"))
        assert calls == ["set_input", "run", "sync"]
    finally:
        owner.close()


def test_ascend_legacy_bind_failure_precedes_native_submission():
    from inference_service.backends.ascend.model import AclModel

    model = object.__new__(AclModel)
    model._execution_lock = None
    model.role = "model"
    model.input_dataset = model.output_dataset = model.model_id = object()
    model._lease = SimpleNamespace(bind_current_thread=lambda: (_ for _ in ()).throw(RuntimeError("bind failed")))
    with pytest.raises(RuntimeError, match="bind failed") as caught:
        model.execute({})
    assert caught.value.evidence.state is OutcomeState.NOT_STARTED


@pytest.mark.parametrize("failure", ["return_code", "exception"])
def test_direct_acl_execution_failure_keeps_explicit_unknown_evidence(failure):
    from inference_service.backends.ascend.model import AclModel

    calls = []
    model = object.__new__(AclModel)
    model._execution_lock = None
    model.role = "model"
    model.input_dataset = model.output_dataset = model.model_id = object()
    model.input_buffers = []
    model._lease = SimpleNamespace(bind_current_thread=lambda: None)
    model._prepare_inputs = lambda *_args, **_kwargs: None

    def execute(*_):
        calls.append("execute")
        if failure == "exception":
            raise RuntimeError("native submit failed")
        return 1

    model._acl = SimpleNamespace(mdl=SimpleNamespace(execute=execute))
    owner = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: model.execute({})), resettable=True)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("acl"))
        assert calls == ["execute"]
        assert not caught.value.evidence.outcome_known and owner.reset_required
    finally:
        owner.close()
