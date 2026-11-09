"""Host loading defaults stay isolated from plugin and concurrent host state."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from inference_service.backends import RuntimeContext, runtime_context_defaults


def context(**kwargs):
    return RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(), role_runtime_profiles={}), **kwargs)


def test_runtime_defaults_are_scoped_and_explicit_values_override_them():
    assert not context().priority_scheduling
    assert not context().managed_execution
    with runtime_context_defaults(priority_scheduling=True, managed_execution=True):
        assert context().priority_scheduling
        assert context().managed_execution
        assert not context(priority_scheduling=False).priority_scheduling
        with runtime_context_defaults(priority_scheduling=False):
            assert not context().priority_scheduling
            assert not context().managed_execution
        assert context().priority_scheduling
    assert not context().priority_scheduling
    assert not context().managed_execution


def test_failed_load_restores_defaults_and_other_threads_do_not_inherit_them():
    with pytest.raises(RuntimeError, match="model load failed"), runtime_context_defaults(priority_scheduling=True):
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert not worker.submit(context).result().priority_scheduling
        assert context().priority_scheduling
        raise RuntimeError("model load failed")
    assert not context().priority_scheduling


@pytest.mark.parametrize("factory_name", ["create_inference_pipeline", "create_pipeline_manager"])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_pipeline_factories_preserve_scoped_priority_unless_explicit(monkeypatch, factory_name, scoped, explicit):
    from inference_service.pipeline import factory

    seen = []

    def validate(runtime):
        seen.append(runtime)
        raise RuntimeError("stop after context validation")

    monkeypatch.setattr(
        factory, "_resolve_pipeline_dependencies", lambda **_: (SimpleNamespace(validate=validate), None, None, None)
    )
    with (
        runtime_context_defaults(priority_scheduling=scoped, managed_execution=True),
        pytest.raises(RuntimeError, match="context validation"),
    ):
        getattr(factory, factory_name)(
            "policy", context().validated_manifest, **({} if explicit is None else {"priority_scheduling": explicit})
        )
    assert len(seen) == 1
    assert seen[0].priority_scheduling is (scoped if explicit is None else explicit)
    assert seen[0].managed_execution


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize(
    "builder", ["_build_session_context", "_build_hmm_session_context", "_build_rknn_session_context"]
)
def test_derived_runtime_context_preserves_explicit_mode_outside_default_scope(managed, builder):
    from inference_service.pipeline import factory

    original = context(managed_execution=managed)
    with runtime_context_defaults(priority_scheduling=False, managed_execution=not managed):
        derived = getattr(factory, builder)(original, {"device_id": 0, "target": "rk3588", "core_mask": 7})
    assert derived.managed_execution is managed
    assert derived.priority_scheduling == original.priority_scheduling


@pytest.mark.parametrize("independent", [False, True])
def test_stage_ticket_identity_survives_multi_role_executor_and_model_stage(independent):
    from inference_service.pipeline.runtime_core import StageFrame
    from inference_service.pipeline.stages import ModelStage
    from inference_service.unified_runtime import ExecutionContext, ModelRequest, RequestDirectExecutor
    from inference_service.unified_runtime.stage_ticket import StageTicket

    seen = []
    ticket = StageTicket(1, "snapshot", 1)
    request = ticket.attach(ModelRequest({"input": 1}))
    context = ExecutionContext("request")

    def execute(request, context):
        seen.append(request)
        return {}

    executor = RequestDirectExecutor({"a": execute, "b": execute})
    executor.execute(request, context)
    frame = StageFrame(request, values={"_execution_context": context, "_independent_stage": independent})
    # The role branch only needs this manifest-frame protocol; actual ModelStage
    # request reconstruction and its session call remain in the test path.
    frame.execution_frame = SimpleNamespace(
        begin_role=lambda _: {"input": 1},
        role_host_inputs=lambda _: {"input": 1},
        plan=SimpleNamespace(
            role=lambda _: SimpleNamespace(bindings=SimpleNamespace(inputs=())), device_links_for_consumer=lambda _: ()
        ),
        finish_role=lambda *_: None,
    )
    frame.execution_plan = frame.execution_frame.plan
    session = SimpleNamespace(execute_role=lambda role, inputs, request, context, **kwargs: execute(request, context))
    ModelStage("a", session).execute(frame, deadline=None)
    assert len(seen) == 3
    assert all(StageTicket.from_request(item) == ticket for item in seen)


def test_policy_request_preparation_preserves_ticket():
    from inference_service.pipeline.runtime import _PolicyModelRequestStage
    from inference_service.pipeline.runtime_core import StageFrame
    from inference_service.unified_runtime import ModelRequest
    from inference_service.unified_runtime.stage_ticket import StageTicket

    ticket = StageTicket(1, "camera-frame", 2)
    request = ticket.attach(ModelRequest({"input": 1}))
    facade = SimpleNamespace(_execution_plan=None, _stage_policy="independent")
    frame = StageFrame(request, values={"_canonical_inputs": {"image": 3}})
    _PolicyModelRequestStage(facade).execute(frame, deadline=None)
    prepared = frame.values["_model_request"]
    assert prepared.inputs == {"image": 3}
    assert StageTicket.from_request(prepared) == ticket
