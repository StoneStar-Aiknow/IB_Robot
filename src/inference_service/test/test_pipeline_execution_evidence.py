"""Sequential and iterative pipelines report progress across native roles."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends import BackendCapabilities, BackendState
from inference_service.backends.errors import BackendAdmissionError, BackendInferenceError
from inference_service.codecs import build_execution_plan
from inference_service.model_sessions.base import ModelSession
from inference_service.pipeline import IterationStep, IterativeStage, ModelStage, SequentialModelExecutor
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeState,
)


@pytest.mark.parametrize("failure_at", ["preprocess", "host", "iteration", "adapter"])
@pytest.mark.parametrize("error_type", [ValueError, KeyError])
def test_plain_host_errors_preserve_frame_progress_and_allow_next_request(failure_at, error_type):
    from inference_service.pipeline import HostComputeStage, PreprocessStage

    failing = [True]
    calls = []

    def fail(*_):
        if failing[0]:
            raise error_type("host rejected input or output")
        return {}

    model = ModelStage("model", SimpleNamespace(execute=lambda *_: calls.append("model") or {"output": 1}))
    stages = (model,)

    def adapt(frame):
        return frame.values["output"]

    if failure_at == "preprocess":
        stages = (PreprocessStage(fail), model)
    elif failure_at == "host":
        stages += (HostComputeStage(fail),)
    elif failure_at == "iteration":
        stages = (
            IterativeStage(
                (IterationStep(0, 0, 1),),
                stages,
                SimpleNamespace(
                    initialize=lambda *_: None, prepare_step=lambda *_: None, update=fail, finalize=lambda *_: None
                ),
            ),
        )
    else:
        adapt = fail
    owner = ModelRuntimeHandle(SequentialModelExecutor(stages, SimpleNamespace(adapt=adapt)))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("bad"))
        evidence = caught.value.evidence
        assert evidence.state is (OutcomeState.NOT_STARTED if failure_at == "preprocess" else OutcomeState.COMPLETED)
        assert evidence.outcome_known and owner.ready
        assert len(calls) == (0 if failure_at == "preprocess" else 1)
        failing[0] = False
        owner.execute(ModelRequest(), ExecutionContext("good"))
    finally:
        owner.close()


@pytest.mark.parametrize("mode", ["first", "second", "iteration"])
@pytest.mark.parametrize("stop", ["cancel", "deadline", "reject", "unknown"])
def test_pipeline_merges_completed_roles_with_later_entry_evidence(mode, stop):
    calls, recorded = [], []
    now = [datetime(2026, 10, 31, tzinfo=timezone.utc)]
    context = ExecutionContext("pipeline", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    fail_at = 1 if mode == "first" else 2

    class Session(ModelSession):
        def __init__(self):
            super().__init__("evidence", BackendCapabilities())
            self._state = BackendState.READY
            self.entries = 0

        def _require_ready(self):
            super()._require_ready()
            self.entries += 1
            if self.entries == fail_at:
                if stop == "cancel":
                    context.token.cancel()
                elif stop == "deadline":
                    now[0] += timedelta(seconds=2)
                elif stop == "reject":
                    raise BackendAdmissionError("role rejected before native submission")

        def _execute_role(self, role, inputs, request, execution):
            calls.append(role)
            if self.entries == fail_at and stop == "unknown":
                raise BackendInferenceError("completion unknown", operation_started=True, outcome_known=False)
            return {"output": np.ones(1, np.float32)}

        def _validate_role_values(self, *args, **kwargs):
            pass

        def _execute(self, request, context):
            raise AssertionError("must use role entry")

        def _load(self, context, rollback):
            pass

        def _close(self):
            pass

    session = Session()
    roles = ("a",) if mode == "iteration" else ("a", "b")
    binding = ArtifactBindings(
        inputs=(TensorBinding(semantic="input", index=0, dtype="float32", shape=(1,)),),
        outputs=(TensorBinding(semantic="output", index=0, dtype="float32", shape=(1,)),),
    )
    plan = build_execution_plan(roles, dict.fromkeys(roles, binding))
    stages = tuple(ModelStage(role, session) for role in roles)
    if mode == "iteration":
        adapter = SimpleNamespace(
            initialize=lambda *_: None, prepare_step=lambda *_: None, update=lambda *_: None, finalize=lambda *_: None
        )
        stages = (IterativeStage((IterationStep(0, 0, 1), IterationStep(1, 1, 1)), stages, adapter, roles),)
    executor = SequentialModelExecutor(
        stages,
        SimpleNamespace(adapt=lambda frame: frame.values["output"]),
        execution_plan=plan,
        error_handler=lambda exc, started: recorded.append(started),
    )
    owner = ModelRuntimeHandle(executor)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest({"input": np.ones(1, np.float32)}), context)
        evidence = caught.value.evidence
        expected = (
            OutcomeState.STARTED
            if stop == "unknown"
            else (OutcomeState.NOT_STARTED if mode == "first" else OutcomeState.COMPLETED)
        )
        assert evidence.state is expected and evidence.outcome_known is (stop != "unknown")
        assert len(calls) == fail_at - (stop != "unknown")
        assert recorded == [expected is not OutcomeState.NOT_STARTED]
        assert owner.ready is (stop != "unknown")
    finally:
        owner.close()


@pytest.mark.parametrize("iterative", [False, True])
def test_cancellation_between_completed_stages_keeps_progress(iterative):
    from inference_service.pipeline import HostComputeStage

    context = ExecutionContext("between")
    calls = []
    session = SimpleNamespace(execute=lambda *_: calls.append("model") or {"output": 1})
    stages = (ModelStage("model", session),)
    if iterative:
        adapter = SimpleNamespace(
            initialize=lambda *_: None,
            prepare_step=lambda *_: None,
            update=lambda *_: context.token.cancel(),
            finalize=lambda *_: None,
        )
        stages = (IterativeStage((IterationStep(0, 0, 1), IterationStep(1, 1, 1)), stages, adapter),)
    else:

        def cancel(_):
            context.token.cancel()
            return {}

        stages += (HostComputeStage(cancel), HostComputeStage(lambda _: {}))
    owner = ModelRuntimeHandle(SequentialModelExecutor(stages, SimpleNamespace(adapt=lambda _: {})))
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), context)
        assert caught.value.code == "request_canceled"
        assert caught.value.evidence.state is OutcomeState.COMPLETED and caught.value.evidence.outcome_known
        assert calls == ["model"] and owner.ready
    finally:
        owner.close()


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_request_direct_aggregates_completed_predecessor_before_second_session_entry(stateful, stop):
    from inference_service.unified_runtime import RequestDirectAssembler

    calls = []
    now = [datetime(2026, 10, 31, tzinfo=timezone.utc)]
    execution = ExecutionContext("roles", Deadline.after(1, now=now[0], clock=lambda: now[0]))

    class Session(ModelSession):
        def __init__(self, name):
            super().__init__(name, BackendCapabilities(stateful=stateful))
            self._state = BackendState.READY
            self._context = SimpleNamespace(
                validated_manifest=SimpleNamespace(manifest=SimpleNamespace(model=SimpleNamespace(outputs=())))
            )

        def _require_ready(self):
            super()._require_ready()
            if self.name == "B":
                if stop == "cancel":
                    execution.token.cancel()
                else:
                    now[0] += timedelta(seconds=2)

        def _execute(self, request, context):
            calls.append(self.name)
            return {}

        def _load(self, context, rollback):
            pass

        def _close(self):
            pass

    a, b = Session("A"), Session("B")
    # Use the same nested per-role RequestDirectExecutor shape as Factory.
    assembly = RequestDirectAssembler().assemble(
        role_assemblies={"a": RequestDirectAssembler().assemble(a), "b": RequestDirectAssembler().assemble(b)}
    )
    owner = ModelRuntimeHandle(assembly)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), execution)
        assert caught.value.evidence.state is OutcomeState.COMPLETED
        assert caught.value.evidence.outcome_known and caught.value.evidence.state_mutated is stateful
        assert calls == ["A"] and owner.ready
    finally:
        owner.close()
