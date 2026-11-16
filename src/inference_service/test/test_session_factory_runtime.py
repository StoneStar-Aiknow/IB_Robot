"""Native session factory behavior against the existing manifest/ORT fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry
from inference_service.model_sessions import build_onnx_model_session
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeFactory,
    RuntimeFactoryError,
    RuntimeProviders,
)
from inference_service.unified_runtime.registry import SessionBuilderRegistry

from .test_onnx_model_session import _context, _fake_ort, _FakeGraph, _stateful_manifest, _stateless_manifest


@pytest.mark.parametrize("model_type", ["act", "diffusion"])
def test_factory_accepts_existing_policy_builder_registry(tmp_path, model_type):
    from inference_manifest import load_inference_manifest
    from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession
    from inference_service.runtime_composition import build_policy_runtime_dependencies

    from .manifest_fixtures import create_policy_bundle, make_manifest, write_json

    files = create_policy_bundle(tmp_path, policy_type=model_type)
    manifest = make_manifest(tmp_path, files)
    if model_type == "diffusion":
        manifest["deployments"]["cpu"]["execution_contract"].update(
            execution_structure="iterative", orchestration_visibility="session"
        )
    write_json(tmp_path / "inference_manifest.json", manifest)

    dependencies = build_policy_runtime_dependencies()
    handle = None
    try:
        handle = ModelRuntimeFactory.create_session(
            load_inference_manifest(tmp_path, "cpu"),
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
            runtime_options={"model_dtype": "fp32"},
        )
        assert isinstance(handle.assembly.session, LeRobotTorchModelSession)
        assert handle.assembly.load_context.runtime_options == {"model_dtype": "fp32"}
    finally:
        if handle is not None:
            handle.close()
        dependencies.providers.close()


@pytest.mark.parametrize("load_entry", ["create_session", "load_session"])
@pytest.mark.parametrize("role_execution", [False, True])
def test_session_factory_rejects_executor_iteration_before_building(tmp_path, load_entry, role_execution):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_composition import build_policy_runtime_dependencies

    from .manifest_fixtures import create_policy_bundle, make_manifest, write_json

    files = create_policy_bundle(tmp_path, policy_type="pi05", include_weights=False)
    manifest = make_manifest(tmp_path, files, deployment_name="ascend", compiled=True, backend="ascend")
    deployment = manifest["deployments"]["ascend"]
    deployment["runtime_profile"]["target"]["soc"] = "Ascend310P3"
    deployment["artifacts"]["policy"]["format"] = "om"
    deployment["execution_contract"].update(execution_structure="iterative", orchestration_visibility="executor")
    write_json(tmp_path / "inference_manifest.json", manifest)
    validated = load_inference_manifest(tmp_path, "ascend")
    dependencies = build_policy_runtime_dependencies()
    try:
        with pytest.raises(RuntimeFactoryError) as failure:
            getattr(ModelRuntimeFactory, load_entry)(
                validated,
                registry_set=dependencies.registry_set,
                providers=dependencies.providers,
                role_execution=role_execution,
                override=lambda *_: pytest.fail("executor iteration reached a bare Session builder"),
            )
        assert failure.value.code == "executor_orchestration_required"
    finally:
        dependencies.providers.close()


@pytest.mark.parametrize("role_execution", [False, True])
@pytest.mark.parametrize("load_entry", ["explicit", "factory"])
def test_factory_rejects_lerobot_state_discovered_during_load(tmp_path, monkeypatch, role_execution, load_entry):
    import torch

    from inference_manifest import load_inference_manifest
    from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession
    from inference_service.runtime_composition import build_policy_runtime_dependencies

    from .test_native_object_runtime import _create_bundle

    # Use the real Session load path, including its post-load capability update.
    # The small local policy replaces weight loading, not lifecycle behavior.
    policy = torch.nn.Linear(1, 1)
    policy.reset = lambda: None
    modules = {
        "torch": torch,
        "lerobot.configs.policies": SimpleNamespace(
            PreTrainedConfig=type(
                "Config",
                (),
                {
                    "from_pretrained": staticmethod(lambda *_args, **_kwargs: SimpleNamespace()),
                },
            ),
        ),
        "lerobot.policies.factory": SimpleNamespace(
            get_policy_class=lambda _: SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: policy),
            make_pre_post_processors=lambda **_: (lambda x: x, lambda x: x),
        ),
        "torch_models.policy_provider": SimpleNamespace(resolve_policy_provider=lambda *_: None),
    }
    monkeypatch.setattr(LeRobotTorchModelSession, "_import_required", staticmethod(lambda name, _: modules[name]))
    dependencies = build_policy_runtime_dependencies()
    handle = None
    options = dict(
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
        role_execution=role_execution,
    )
    spec = load_inference_manifest(_create_bundle(tmp_path), "cpu")
    try:
        if role_execution:
            # Native deployments have no manifest roles. Reject before loading;
            # the direct case below still exercises load-time state discovery.
            with pytest.raises(RuntimeFactoryError) as failed:
                ModelRuntimeFactory.create_session(spec, **options)
            assert failed.value.code == "role_execution_missing"
            return
        with pytest.raises(ExecutionFailure, match="stateful request") as failed:
            if load_entry == "factory":
                handle = ModelRuntimeFactory.load_session(spec, **options)
            else:
                handle = ModelRuntimeFactory.create_session(spec, **options)
                assert not handle.assembly.session.capabilities.stateful
                handle.load()
        assert failed.value.details["cleanup_complete"]
        if handle is not None:
            assert handle.assembly.session.capabilities.stateful
            assert handle.assembly.session.policy is None
            assert not handle.ready
            with pytest.raises(ExecutionFailure):
                handle.execute(ModelRequest(metadata={"action_method": "select_action"}), ExecutionContext("blocked"))
    finally:
        if handle is not None:
            handle.close()
        dependencies.providers.close()


def test_role_factory_rejects_device_links_before_creating_session(tmp_path):
    from inference_manifest import load_inference_manifest

    from .test_inference_manifest import _host_orchestrated_manifest, write_manifest

    manifest, _ = _host_orchestrated_manifest(tmp_path)
    write_manifest(tmp_path, manifest)
    validated = load_inference_manifest(tmp_path, "ascend")
    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    try:
        with pytest.raises(RuntimeFactoryError, match="device links") as failure:
            ModelRuntimeFactory.create_session(
                validated,
                registry_set=registries,
                providers=providers,
                role_execution=True,
                override=lambda *_: pytest.fail("unsafe role execution reached session creation"),
            )
        assert failure.value.code == "role_device_links_unsupported"
    finally:
        providers.close()


@pytest.mark.parametrize("entry", ["create_session", "load_session"])
@pytest.mark.parametrize("unsupported", ["empty_roles", "missing_method"])
def test_role_factory_rejects_unexecutable_resource_and_releases_it(tmp_path, entry, unsupported):
    from contextlib import ExitStack

    from inference_manifest import load_inference_manifest
    from inference_service.model_sessions.native_torch import NativeTorchResource

    from .test_native_object_runtime import _create_bundle, _registries

    spec = (
        load_inference_manifest(_create_bundle(tmp_path), "cpu")
        if unsupported == "empty_roles"
        else _context(tmp_path, _stateless_manifest()).validated_manifest
    )
    events = []

    def builder(context):
        events.append("build")
        if unsupported == "empty_roles":
            resource = NativeTorchResource(context.validated_manifest, lambda: events.append("sdk_load"), lambda *_: {})
            close = resource.close

            def release():
                events.append("close")
                close()

            resource.close = release
            return resource
        return SimpleNamespace(
            capabilities=SimpleNamespace(stateful=False, resettable=False),
            load=lambda *_: events.append("load"),
            close=lambda: events.append("close"),
            execute=lambda *_: {},
        )

    with ExitStack() as cleanup:
        providers = RuntimeProviders.create(object(), object())
        cleanup.callback(providers.close)
        with pytest.raises(RuntimeFactoryError) as failed:
            handle = getattr(ModelRuntimeFactory, entry)(
                spec, registry_set=_registries(spec), providers=providers, role_execution=True, override=builder
            )
            cleanup.callback(handle.close)
        assert failed.value.code == (
            "role_execution_missing" if unsupported == "empty_roles" else "role_execution_unsupported"
        )
        assert events == ([] if unsupported == "empty_roles" else ["build", "close"])


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("load_entry", ["explicit", "factory"])
def test_factory_owns_role_session_and_recurrent_stream(tmp_path, stateful, load_entry):
    if stateful:
        graph = _FakeGraph(
            ("input", "state"),
            ("prob", "state_out"),
            lambda feed: [
                np.asarray([np.sum(feed["input"])], np.float32),
                feed["state"] + 1,
            ],
        )
    else:
        graph = _FakeGraph(("input",), ("prob",), lambda feed: [np.asarray([np.sum(feed["input"])], np.float32)])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    context = _context(tmp_path, _stateful_manifest() if stateful else _stateless_manifest())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    providers = RuntimeProviders.create(object(), object())
    create = ModelRuntimeFactory.load_session if load_entry == "factory" else ModelRuntimeFactory.create_session
    handle = create(
        context.validated_manifest,
        registry_set=registries,
        providers=providers,
        role_execution=True,
        override=build_onnx_model_session,
        builder_options={"ort_loader": lambda: ort},
    )
    if load_entry == "explicit":
        handle.load()
    request = ModelRequest({"host.silero.audio": np.ones(4, np.float32)}, {"role": "model"})
    try:
        if stateful:
            stream = handle.open_stream(ExecutionContext("open"))
            with pytest.raises(ExecutionFailure):
                handle.open_stream(ExecutionContext("second-stream"))
            for index in range(2):
                handle.step(stream, request, ExecutionContext(f"step-{index}"))
            assert [feed["state"].tolist() for feed in graph.feeds] == [[0, 0], [1, 1]]
            handle.reset_stream(stream, ExecutionContext("reset"))
            result = handle.step(stream, request, ExecutionContext("after-reset"))
            assert graph.feeds[-1]["state"].tolist() == [0, 0]
            handle.close_stream(stream, ExecutionContext("stream-close"))
            stream = handle.open_stream(ExecutionContext("reopen"))
            result = handle.step(stream, request, ExecutionContext("first-reopened-frame"))
            assert graph.feeds[-1]["state"].tolist() == [0, 0]
            handle.close_stream(stream, ExecutionContext("reopened-close"))
        else:
            result = handle.execute(request, ExecutionContext("execute"))
        assert result.outputs["host.silero.prob"].tolist() == [4]
    finally:
        handle.close()
        providers.close()
    assert handle.closed and providers.closed


def test_role_adapter_rejects_per_stream_bank_before_loading(tmp_path):
    manifest = _stateful_manifest()
    contract = manifest["deployments"]["onnx_cpu"]["execution_contract"]
    contract["state_bank_mode"] = "per_stream"
    contract["max_open_streams"] = 2
    for link in contract["state_links"]:
        link["scope"] = "stream"
    context = _context(tmp_path, manifest)
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    providers = RuntimeProviders.create(object(), object())
    try:
        with pytest.raises(RuntimeFactoryError, match="runtime_exclusive"):
            ModelRuntimeFactory.create_session(
                context.validated_manifest,
                registry_set=registries,
                providers=providers,
                role_execution=True,
                override=build_onnx_model_session,
                builder_options={"ort_loader": lambda: pytest.fail("unsupported bank reached backend load")},
            )
    finally:
        providers.close()


@pytest.mark.parametrize("role", [None, "", 123, "missing_role"])
@pytest.mark.parametrize("managed", [False, True])
def test_role_validation_proves_not_started(role, managed):
    from inference_service.scheduler.runtime_controller import RuntimeExecutionController
    from inference_service.unified_runtime import ModelRuntimeHandle, OutcomeState
    from inference_service.unified_runtime.factory import SessionRoleExecutor

    from .runtime_scheduler_helpers import target

    session = SimpleNamespace(execute_role=lambda *_: pytest.fail("invalid role reached Session"))
    handle = ModelRuntimeHandle(executor=SessionRoleExecutor(session, ("model",)))
    handle.load()
    host = RuntimeExecutionController(target(), handle) if managed else None
    request, context = ModelRequest(metadata={"role": role}), ExecutionContext("invalid-role")
    try:
        if host is not None:
            host.accept_epoch("test")
            sub = host.prepare(request, context)
            host.execute(sub, "test")
            host.completion(sub).result(timeout=1)
            assert host.query(sub, "test")["state"] == "not_started"
        with pytest.raises(ExecutionFailure) as failure:
            host.result(sub) if host is not None else handle.execute(request, context)
        assert failure.value.evidence.state is OutcomeState.NOT_STARTED
        assert handle.local_ready
    finally:
        if host is not None:
            host.shutdown()
        handle.close()


def test_factory_retains_failed_session_cleanup(tmp_path):
    context = _context(tmp_path, _stateless_manifest())
    calls = []
    allow_close = [False]

    class Session:
        capabilities = SimpleNamespace(stateful=False)  # Missing reset capability fails assembly.

        def close(self):
            calls.append("close")
            if not allow_close[0]:
                raise RuntimeError("cleanup pending")

    # The malformed executor fails assembly after the factory acquired it.
    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    with pytest.raises(AttributeError, match="resettable") as failure:
        ModelRuntimeFactory.create_session(
            context.validated_manifest,
            registry_set=registries,
            providers=providers,
            override=lambda *_: Session(),
        )
    assert failure.value.cleanup_owner is not None and calls == ["close"]
    allow_close[0] = True
    providers.close()
    assert calls == ["close", "close"] and providers.closed


def test_factory_rejects_contract_override_before_creating_session(tmp_path):
    context = _context(tmp_path, _stateless_manifest())
    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    try:
        with pytest.raises(RuntimeFactoryError, match="must match the validated deployment"):
            ModelRuntimeFactory.create_session(
                context.validated_manifest,
                registry_set=registries,
                providers=providers,
                execution_contract="stream-direct",
                override=lambda *_: pytest.fail("invalid contract reached the builder"),
            )
    finally:
        providers.close()


def test_factory_rejects_stateful_resource_under_request_contract_and_releases_it(tmp_path):
    context = _context(tmp_path, _stateless_manifest())
    calls = []

    class Session:
        capabilities = SimpleNamespace(stateful=True, resettable=True)

        def close(self):
            calls.append("close")

    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    try:
        with pytest.raises(RuntimeFactoryError) as failure:
            ModelRuntimeFactory.create_session(
                context.validated_manifest,
                registry_set=registries,
                providers=providers,
                override=lambda *_: Session(),
            )
        assert failure.value.code == "stateful_request_runtime"
        assert calls == ["close"]
    finally:
        providers.close()


def test_runtime_reset_of_open_role_stream_resets_native_session_once(tmp_path, monkeypatch):
    context = _context(tmp_path, _stateful_manifest())
    graph = _FakeGraph(("input", "state"), ("prob", "state_out"), lambda feed: [np.zeros(1, np.float32), feed["state"]])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    handle = ModelRuntimeFactory.load_session(
        context.validated_manifest,
        registry_set=registries,
        providers=providers,
        role_execution=True,
        override=build_onnx_model_session,
        builder_options={"ort_loader": lambda: ort},
    )
    session = handle.assembly.session
    resets = []
    reset = session.reset

    def counted(context):
        resets.append(context.request_id)
        return reset(context)

    monkeypatch.setattr(session, "reset", counted)
    try:
        stream = handle.open_stream(ExecutionContext("open"))
        resets.clear()
        handle.reset(ExecutionContext("runtime-reset"))
        assert resets == ["runtime-reset"]
        handle.reset_stream(stream, ExecutionContext("stream-reset"))
        assert resets == ["runtime-reset", "stream-reset"]
    finally:
        handle.close()
        providers.close()


@pytest.mark.parametrize("role", [None, "first"])
def test_create_session_rejects_composite_without_implicit_role_selection(role):
    providers = RuntimeProviders.create(object(), object())
    registries = SimpleNamespace(
        session_builder_registry=SessionBuilderRegistry(), backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    )
    composite = SimpleNamespace(runtime_profile=None, role_runtime_profiles={"first": object(), "second": object()})
    try:
        with pytest.raises(RuntimeFactoryError) as failed:
            ModelRuntimeFactory.create_session(
                composite,
                registry_set=registries,
                providers=providers,
                role=role,
                override=lambda *_: pytest.fail("composite reached a single-session builder"),
            )
        assert failed.value.code == "composite_session_unsupported"
    finally:
        providers.close()
