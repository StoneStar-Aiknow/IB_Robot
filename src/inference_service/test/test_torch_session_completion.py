"""Completion evidence must come from production Torch Session calls."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends import BackendState
from inference_service.backends.errors import BackendLifecycleError
from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession
from inference_service.model_sessions.torch import TorchModelSession
from inference_service.unified_runtime import ExecutionContext, ModelRequest


def session_probe(kind, managed, events, fail):
    def synchronize(_):
        events.append("sync")
        if fail[0] == "sync":
            raise RuntimeError("completion unknown")

    def run(*_, **__):
        events.append("execute")
        if fail[0] in {"execute", "sync"}:
            raise ValueError("output transfer failed")
        return {} if kind == "tensor" else np.ones((1, 2), dtype=np.float32)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(device=lambda _: nullcontext(), synchronize=synchronize), inference_mode=nullcontext
    )
    torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: torch.cuda.synchronize(device))
    session = TorchModelSession(lambda _: None) if kind == "tensor" else LeRobotTorchModelSession("cuda")
    session._torch = torch
    session._device = "cuda:0"
    if kind == "tensor":
        session._module = run
        session._device_name = "cuda"
    else:
        session._policy = SimpleNamespace(predict_action_chunk=run)
    session._state = BackendState.READY
    session._context = SimpleNamespace(
        managed_execution=managed,
        model_type="act",
        validated_manifest=SimpleNamespace(manifest=SimpleNamespace(model=SimpleNamespace(inputs=(), outputs=()))),
    )
    return session


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("sync_fails", [False, True])
def test_callable_cancel_after_submission_uses_real_session_completion(kind, managed, sync_fails):
    from inference_service.unified_runtime import OutcomeState
    from inference_service.unified_runtime.errors import CancellationRequested

    events, fail = [], ["sync" if sync_fails else None]
    session = session_probe(kind, managed, events, fail)
    context = ExecutionContext("canceled_callback")

    def callback(*_, **__):
        events.append("submitted")
        context.token.cancel()
        context.check("callback")

    if kind == "tensor":
        session._module = callback
    else:
        session._policy.predict_action_chunk = callback
    try:
        with pytest.raises(CancellationRequested) as caught:
            session.execute(ModelRequest(), context)
        assert events == ["submitted", "sync"]
        assert caught.value.evidence.state is (OutcomeState.STARTED if sync_fails else OutcomeState.COMPLETED)
        assert caught.value.evidence.outcome_known is not sync_fails
    finally:
        fail[0] = None
        session.close()


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("failure", [None, "execute", "sync"])
def test_real_session_execution_waits_for_failure_evidence_and_scopes_success_sync(kind, managed, failure):
    events, fail = [], [failure]
    session = session_probe(kind, managed, events, fail)
    if failure is not None:
        with pytest.raises(ValueError) as error:
            session.execute(ModelRequest(), ExecutionContext("torch"))
        assert error.value.operation_started and error.value.outcome_known is (failure != "sync")
    else:
        session.execute(ModelRequest(), ExecutionContext("torch"))
    assert events == ["execute"] + (["sync"] if managed or failure else [])
    fail[0] = None
    session.close()
    assert events[-1] == "sync"
    assert session.health().state is BackendState.CLOSED


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("managed", [False, True])
def test_real_session_close_keeps_models_until_device_completion_is_proven(kind, managed):
    events, fail = [], ["sync"]
    session = session_probe(kind, managed, events, fail)
    native = session._module if kind == "tensor" else session._policy
    with pytest.raises(BackendLifecycleError) as error:
        session.close()
    assert error.value.close_pending and session.health().state is BackendState.FAILED
    assert (session._module if kind == "tensor" else session._policy) is native
    fail[0] = None
    session.close()
    assert session.health().state is BackendState.CLOSED and events == ["sync", "sync"]


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
def test_real_session_close_terminates_after_bounded_drain_failures(kind):
    from inference_service.model_sessions.torch_completion import RELEASE_DRAIN_ATTEMPTS

    events, fail = [], ["sync"]
    session = session_probe(kind, False, events, fail)
    for _ in range(RELEASE_DRAIN_ATTEMPTS):
        with pytest.raises(BackendLifecycleError) as error:
            session.close()
        assert error.value.close_pending and session.health().state is BackendState.FAILED
        assert (session._module if kind == "tensor" else session._policy) is not None
    session.close()
    assert session.health().state is BackendState.CLOSED
    assert (session._module if kind == "tensor" else session._policy) is None
    assert events == ["sync"] * RELEASE_DRAIN_ATTEMPTS


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
def test_stateful_session_control_failure_evidence_marks_state_mutated(kind):
    from inference_service.unified_runtime import OutcomeState
    from inference_service.unified_runtime.errors import CancellationRequested

    events, fail = [], [None]
    session = session_probe(kind, False, events, fail)
    session._capabilities = replace(session.capabilities, stateful=True)
    context = ExecutionContext("stateful-cancel")

    def callback(*_, **__):
        context.token.cancel()
        context.check("callback")

    if kind == "tensor":
        session._module = callback
    else:
        session._policy.predict_action_chunk = callback
    try:
        with pytest.raises(CancellationRequested) as caught:
            session.execute(ModelRequest(), context)
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known
        assert caught.value.evidence.state_mutated is True
    finally:
        session.close()


def test_lerobot_reset_waits_for_device_before_mutating_policy_state():
    events, fail = [], ["sync"]
    session = session_probe("lerobot", False, events, fail)
    session._capabilities = replace(session.capabilities, resettable=True)
    session._policy.reset = lambda: events.append("reset")
    with pytest.raises(RuntimeError, match="completion unknown"):
        session.reset()
    assert events == ["sync"]
    fail[0] = None
    session.reset()
    assert events == ["sync", "sync", "reset"]
    session.close()


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("sync_fails", [False, True])
def test_public_session_validation_discards_direct_torch_result_only_after_completion(managed, sync_fails):
    from inference_manifest import SemanticTensor
    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle, OutcomeState

    events, fail = [], [None]
    session = session_probe("tensor", managed, events, fail)

    def output(_):
        events.append("execute")
        return {"result": np.ones(1, np.float64)}

    def sync(_):
        events.append("sync")
        if sync_fails:
            raise RuntimeError("completion unknown")

    session._module = output
    session._torch.is_tensor = lambda _: False
    session._torch.cuda.synchronize = sync
    session._context.validated_manifest.manifest.model.outputs = (
        SemanticTensor(semantic="result", dtype="float32", shape=(1,)),
    )
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("invalid-output"))
        assert caught.value.evidence.outcome_known is not sync_fails
        assert caught.value.evidence.state is (OutcomeState.STARTED if sync_fails else OutcomeState.COMPLETED)
        assert owner.ready is not sync_fails
        assert events == ["execute", "sync"]
    finally:
        owner.close()
        session._torch.cuda.synchronize = lambda _: None
        session.close()


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_native_completion_then_context_expiry_remains_completed_at_handle(kind, stop):
    from datetime import datetime, timedelta, timezone

    from inference_service.unified_runtime import Deadline, ExecutionFailure, ModelRuntimeHandle, OutcomeState

    events, fail = [], [None]
    now = [datetime(2026, 10, 30, tzinfo=timezone.utc)]
    context = ExecutionContext("late", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    session = session_probe(kind, True, events, fail)

    def complete(_):
        events.append("sync")
        if stop == "deadline":
            now[0] += timedelta(seconds=2)
        else:
            context.cancellation_token.cancel("cancel after device completion")

    session._torch.cuda.synchronize = complete
    # The real Session is already loaded by session_probe; the callable adapter
    # lets Handle exercise its execution boundary without loading it twice.
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=session.close))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.execute(ModelRequest(), context)
        assert failure.value.evidence.state is OutcomeState.COMPLETED
        assert failure.value.evidence.outcome_known and owner.ready
        assert events == ["execute", "sync"]
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_direct_late_result_requires_device_fence(kind, stop, fence_fails):
    from datetime import datetime, timedelta, timezone

    from inference_service.unified_runtime import Deadline, ExecutionFailure, ModelRuntimeHandle, OutcomeState

    events, fail = [], [None]
    now = [datetime(2026, 10, 30, tzinfo=timezone.utc)]
    context = ExecutionContext("late", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    session = session_probe(kind, False, events, fail)
    run = session._module if kind == "tensor" else session._policy.predict_action_chunk

    def late(*args, **kwargs):
        result = run(*args, **kwargs)
        if stop == "deadline":
            now[0] += timedelta(seconds=2)
        else:
            context.cancellation_token.cancel()
        return result

    if kind == "tensor":
        session._module = late
    else:
        session._policy.predict_action_chunk = late

    def synchronize(_):
        events.append("sync")
        if fence_fails:
            raise RuntimeError("device still busy")

    session._torch.cuda.synchronize = synchronize
    owner = ModelRuntimeHandle(SimpleNamespace(execute=session.execute, close=session.close))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.execute(ModelRequest(), context)
        assert failure.value.code == ("deadline_exceeded" if stop == "deadline" else "request_canceled")
        assert failure.value.evidence.state is (OutcomeState.STARTED if fence_fails else OutcomeState.COMPLETED)
        assert failure.value.evidence.outcome_known is not fence_fails
        assert owner.ready is not fence_fails
        assert events == ["execute", "sync"]
    finally:
        fence_fails = False
        owner.close()


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("fails", [False, True])
def test_loading_pins_cuda_device_for_execution_and_close(tmp_path, monkeypatch, kind, fails):
    from contextlib import contextmanager

    import torch as real_torch

    from inference_manifest import TorchRuntimeProfile
    from inference_service.backends import RuntimeContext

    current, synced, placements = [0], [], []

    @contextmanager
    def device_scope(device):
        previous = current[0]
        current[0] = real_torch.device(device).index if real_torch.device(device).index is not None else current[0]
        try:
            yield
        finally:
            current[0] = previous

    def synchronize(device):
        index = real_torch.device(device).index
        synced.append(current[0] if index is None else index)

    torch = SimpleNamespace(
        device=real_torch.device,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: current[0],
            device=device_scope,
            synchronize=synchronize,
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: synchronize(device)),
            empty_cache=lambda: None,
        ),
    )

    class Module:
        def to(self, device):
            placements.append(real_torch.device(device))
            return self

        def eval(self):
            return self

        def __call__(self, *_args, **_kwargs):
            assert current[0] == 0
            if fails:
                raise ValueError("inference failed on device zero")
            return {} if kind == "tensor" else np.ones((1, 2), np.float32)

        predict_action_chunk = __call__

    module = Module()
    model_type = "act" if kind == "lerobot" else "tensor"
    identity = SimpleNamespace(
        interface="policy" if kind == "lerobot" else "tensor_model",
        model_type=model_type,
        operation="predict" if kind == "lerobot" else "infer",
    )
    validated = SimpleNamespace(
        deployment=SimpleNamespace(backend="torch", extra_parameters={}),
        bundle_root=tmp_path,
        manifest=SimpleNamespace(model=SimpleNamespace(**vars(identity), inputs=(), outputs=())),
        top_level_identity=identity,
    )
    context = RuntimeContext(validated, runtime_profile=TorchRuntimeProfile(device="cuda"), managed_execution=True)
    if kind == "tensor":
        session = TorchModelSession(lambda _: module, torch_loader=lambda: torch)
    else:
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "policy_preprocessor.json").write_text("{}")
        modules = {
            "torch": torch,
            "lerobot.configs.policies": SimpleNamespace(
                PreTrainedConfig=type(
                    "Config", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: SimpleNamespace())}
                )
            ),
            "lerobot.policies.factory": SimpleNamespace(
                get_policy_class=lambda _: SimpleNamespace(from_pretrained=lambda *args, **kwargs: module),
                make_pre_post_processors=lambda **kwargs: (lambda x: x, lambda x: x),
            ),
            "torch_models.policy_provider": SimpleNamespace(resolve_policy_provider=lambda *_: None),
        }
        monkeypatch.setattr(LeRobotTorchModelSession, "_import_required", staticmethod(lambda name, _: modules[name]))
        session = LeRobotTorchModelSession("cuda")
    session.load(context)
    current[0] = 1
    try:
        if fails:
            with pytest.raises(ValueError, match="device zero") as caught:
                session.execute(ModelRequest(), ExecutionContext("execute"))
            assert caught.value.outcome_known
        else:
            session.execute(ModelRequest(), ExecutionContext("execute"))
        assert placements == [real_torch.device("cuda:0")]
        assert synced == [0] and current[0] == 1
    finally:
        session.close()
    assert synced == [0, 0]


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_pipeline_discards_direct_result_only_after_device_fence(kind, stop, fence_fails):
    from datetime import datetime, timedelta, timezone

    from inference_service.pipeline import HostComputeStage, ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import Deadline, ExecutionFailure, ModelRuntimeHandle, OutcomeState

    events, fail = [], [None]
    session = session_probe(kind, False, events, fail)
    now = [datetime.now(timezone.utc)]
    context = ExecutionContext("after-session", Deadline.after(5, now=now[0], clock=lambda: now[0]))
    model = ModelStage("model", session)

    class AfterSession:
        def execute(self, frame, *, deadline):
            frame.values["_model_request"] = frame.request
            model.execute(frame, deadline=deadline)
            if stop == "cancel":
                context.token.cancel()
            else:
                now[0] += timedelta(seconds=10)

    def synchronize(_):
        events.append("sync")
        if fence_fails:
            raise RuntimeError("device still busy")

    session._torch.cuda.synchronize = synchronize
    executor = SequentialModelExecutor(
        (AfterSession(), HostComputeStage(lambda _: {})), SimpleNamespace(adapt=lambda _: {}), components=(session,)
    )
    owner = ModelRuntimeHandle(executor)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), context)
        assert caught.value.evidence.state is (OutcomeState.STARTED if fence_fails else OutcomeState.COMPLETED)
        assert caught.value.evidence.outcome_known is not fence_fails
        assert owner.ready is not fence_fails and events == ["execute", "sync"]
        if fence_fails:
            with pytest.raises(ExecutionFailure):
                owner.execute(ModelRequest(), ExecutionContext("blocked"))
    finally:
        fence_fails = False
        owner.close()
        session.close()


@pytest.mark.parametrize("stage_name", ["preprocess", "postprocess"])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("boundary", ["processor_error", "processor_cancel", "next_checkpoint"])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_device_processors_have_completion_evidence(stage_name, managed, boundary, fence_fails):
    from inference_service.model_sessions.lerobot_torch import LeRobotSessionPostprocessor, LeRobotSessionPreprocessor
    from inference_service.pipeline import HostComputeStage, SequentialModelExecutor
    from inference_service.pipeline.runtime import _PolicyPostprocessStage, _PolicyPreprocessStage
    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle, OutcomeState

    events, fail = [], [None]
    session = session_probe("lerobot", managed, events, fail)
    context = ExecutionContext("processor")

    def processor(value):
        events.append("processor-enqueue")
        if boundary == "processor_error":
            raise ValueError("processor failed after enqueue")
        context.token.cancel()
        if boundary == "processor_cancel":
            context.check("processor")
        return value

    def synchronize(_):
        events.append("sync")
        if fence_fails:
            raise RuntimeError("processor completion unknown")

    session._torch.cuda.synchronize = synchronize
    session._preprocessor = session._postprocessor = processor
    facade = SimpleNamespace(
        pipeline_id="processor",
        _default_task=None,
        _instrument_processors=False,
        _stage_policy="sequential",
        _preprocessor=LeRobotSessionPreprocessor(session),
        _postprocessor=LeRobotSessionPostprocessor(session),
        _action_dimension=2,
        _raise_if_expired=lambda *_args, **_kwargs: None,
    )
    if stage_name == "preprocess":
        stage = _PolicyPreprocessStage(facade)
        stages = (stage, HostComputeStage(lambda _: {}))
    else:
        stages = (
            HostComputeStage(
                lambda _: {
                    "_semantic_action": np.ones((1, 2), np.float32),
                    "_backend_result": SimpleNamespace(actual_chunk_size=1),
                }
            ),
            _PolicyPostprocessStage(facade),
            HostComputeStage(lambda _: {}),
        )
    executor = SequentialModelExecutor(stages, SimpleNamespace(adapt=lambda _: {}), components=(session,))
    owner = ModelRuntimeHandle(executor)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), context)
        assert caught.value.evidence.state is not OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known is not fence_fails
        assert owner.ready is not fence_fails
        assert events[0] == "processor-enqueue" and "sync" in events
    finally:
        fence_fails = False
        owner.close()
        session.close()


@pytest.mark.parametrize("unknown_before_fence", [False, True])
def test_pipeline_device_fences_cannot_erase_another_unknown_operation(unknown_before_fence):
    from inference_service.pipeline import HostComputeStage, ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle, OutcomeEvidence

    events, first_fail = [], [None]
    first = session_probe("tensor", False, events, first_fail)
    second = session_probe("tensor", False, events, [None])
    context = ExecutionContext("multiple-devices")
    model = ModelStage("model", first)

    class AfterSession:
        def execute(self, frame, *, deadline):
            frame.values["_model_request"] = frame.request
            model.execute(frame, deadline=deadline)
            if unknown_before_fence:
                raise ExecutionFailure(
                    "native_unknown", "another operation is pending", evidence=OutcomeEvidence.started()
                )
            ModelStage("other", second).execute(frame, deadline=deadline)
            first_fail[0] = "sync"
            context.token.cancel()

    executor = SequentialModelExecutor(
        (AfterSession(), HostComputeStage(lambda _: {})),
        SimpleNamespace(adapt=lambda _: {}),
        components=(first, second),
    )
    owner = ModelRuntimeHandle(executor)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), context)
        assert not caught.value.evidence.outcome_known and not owner.ready
        assert events == (["execute"] if unknown_before_fence else ["execute", "execute", "sync", "sync"])
    finally:
        first_fail[0] = None
        owner.close()
        first.close()
        second.close()


def test_npu_load_execution_and_close_keep_the_loading_device(tmp_path, monkeypatch):
    import sys
    from contextlib import contextmanager

    from inference_manifest import TorchRuntimeProfile
    from inference_service.backends import RuntimeContext

    current, placements, waits = [0], [], []

    class Device:
        def __init__(self, name):
            self.type, _, index = name.partition(":")
            self.index = int(index) if index else None

    @contextmanager
    def device_scope(device):
        previous = current[0]
        current[0] = device.index if device.index is not None else current[0]
        try:
            yield
        finally:
            current[0] = previous

    def synchronize(device):
        waits.append(current[0] if device.index is None else device.index)

    class Module:
        def to(self, device):
            placements.append(device.index)
            return self

        def eval(self):
            return self

        def __call__(self, inputs):
            assert current[0] == 0
            return {}

    torch = SimpleNamespace(
        device=Device,
        inference_mode=nullcontext,
        npu=SimpleNamespace(
            current_device=lambda: current[0],
            is_available=lambda: True,
            device=device_scope,
            synchronize=synchronize,
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: synchronize(device)),
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    identity = SimpleNamespace(interface="tensor_model", model_type="tensor", operation="infer")
    validated = SimpleNamespace(
        top_level_identity=identity,
        bundle_root=tmp_path,
        deployment=SimpleNamespace(backend="torch", extra_parameters={}),
        manifest=SimpleNamespace(model=SimpleNamespace(**vars(identity), inputs=(), outputs=())),
    )
    context = RuntimeContext(validated, runtime_profile=TorchRuntimeProfile(device="npu"), managed_execution=True)
    session = TorchModelSession(lambda _: Module(), torch_loader=lambda: torch)
    session.load(context)
    current[0] = 1
    session.execute(ModelRequest(), ExecutionContext("other-thread-device"))
    session.close()
    assert placements == [0] and waits == [0, 0] and current[0] == 1


@pytest.mark.parametrize("kind", ["tensor", "lerobot"])
@pytest.mark.parametrize("failure_at", ["cancel", "deadline", "execute"])
def test_pipeline_reuses_session_fence_and_skips_idle_components(kind, failure_at):
    from datetime import datetime, timedelta, timezone

    from inference_service.pipeline import HostComputeStage, ModelStage, SequentialModelExecutor
    from inference_service.unified_runtime import Deadline, ExecutionFailure, ModelRuntimeHandle

    events, idle_events = [], []
    session = session_probe(kind, False, events, ["execute" if failure_at == "execute" else None])
    idle = session_probe(kind, False, idle_events, [None])
    now = [datetime.now(timezone.utc)]
    context = ExecutionContext("direct", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    run = session._module if kind == "tensor" else session._policy.predict_action_chunk

    def stop(*args, **kwargs):
        value = run(*args, **kwargs)
        if failure_at == "cancel":
            context.token.cancel()
        elif failure_at == "deadline":
            now[0] += timedelta(seconds=2)
        return value

    if kind == "tensor":
        session._module = stop
    else:
        session._policy.predict_action_chunk = stop
    owner = ModelRuntimeHandle(
        SequentialModelExecutor(
            (HostComputeStage(lambda _: {"_model_request": ModelRequest()}), ModelStage("model", session)),
            SimpleNamespace(adapt=lambda _: {}),
            components=(session, idle),
        )
    )
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), context)
        assert caught.value.evidence.outcome_known and owner.ready
        assert events == ["execute", "sync"] and idle_events == []
    finally:
        owner.close()
        session.close()
        idle.close()
