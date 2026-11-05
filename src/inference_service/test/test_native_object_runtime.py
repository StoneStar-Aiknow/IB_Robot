"""Domain resources use the registered Session factory and real CPU Torch."""

import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, RuntimeContext
from inference_service.backends.errors import BackendCompatibilityError
from inference_service.model_sessions import native_torch
from inference_service.model_sessions.native_torch import NativeTorchResource, build_native_torch_resource
from inference_service.unified_runtime import (
    CancellationToken,
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeFactory,
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeProviders,
    SessionBuilderKey,
    SessionBuilderRegistry,
)
from inference_service.unified_runtime.errors import DeadlineExceeded
from inference_service.unified_runtime.registry import RuntimeRegistryError

from .manifest_fixtures import create_policy_bundle, make_manifest, write_json


def _create_bundle(root):
    files = create_policy_bundle(root)
    write_json(root / "inference_manifest.json", make_manifest(root, files))
    return root


def _registries(spec, builder=build_native_torch_resource):
    sessions = SessionBuilderRegistry()
    if builder is not None:
        sessions.register(SessionBuilderKey.from_context(RuntimeContext(spec)), builder)
    return RegistrySet(BackendRegistry(STATIC_BACKEND_DESCRIPTORS), sessions, RuntimeAssemblerRegistry())


def test_native_domain_api_load_execute_close(tmp_path):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    events = []

    class Model:
        def close(self):
            events.append("close")

    def factory():
        events.append("load")
        return Model()

    providers = RuntimeProviders.create(object(), object())
    runtime = ModelRuntimeFactory.load_session(
        spec,
        registry_set=_registries(spec),
        allowed_deployments=frozenset({"cpu"}),
        builder_options={"factory": factory, "execute": lambda model, values: {"label": values["label"]}},
        providers=providers,
    )
    try:
        result = runtime.execute(ModelRequest({"label": "cup"}), ExecutionContext("detect-cup"))
        assert result.outputs["value"] == {"label": "cup"}
        runtime.close()
        runtime.close()
        assert events == ["load", "close"]
        with pytest.raises(ExecutionFailure, match="closed"):
            runtime.execute(ModelRequest({"label": "bottle"}), ExecutionContext("after-close"))
    finally:
        runtime.close()
        providers.close()


@pytest.mark.parametrize("restriction", ["allowlist", "native_allowlist", "target", "identity", "evidence"])
def test_session_factory_validates_backend_before_calling_builder(tmp_path, restriction):
    from dataclasses import replace

    spec = load_inference_manifest(_create_bundle(tmp_path), "cpu")
    registries = _registries(spec)
    options = {}
    expected = "adapter_deployment_mismatch"
    if restriction in {"allowlist", "native_allowlist"}:
        options["allowed_deployments"] = frozenset({"another_deployment"})
    else:
        descriptor = registries.backend_registry.descriptor("torch")
        if restriction == "target":
            descriptor = replace(descriptor, target_validator=lambda _: "unsupported test target")
            expected = "incompatible_backend_target"
        elif restriction == "identity":
            identity = ("policy", "diffusion", "predict")
            descriptor = replace(
                descriptor,
                supported_identities=frozenset({identity}),
                conformance_evidence=tuple(e for e in descriptor.conformance_evidence if e.identity == identity),
            )
            expected = "unsupported_model_backend_pair"
        else:
            descriptor = replace(
                descriptor,
                conformance_evidence=tuple(
                    replace(e, devices=frozenset({"cuda"})) for e in descriptor.conformance_evidence
                ),
            )
            expected = "missing_conformance_evidence"
        options["backend_registry"] = BackendRegistry({"torch": descriptor})
    providers = RuntimeProviders.create(object(), object())
    if restriction == "native_allowlist":
        options["builder_options"] = {
            "factory": lambda: pytest.fail("rejected deployment reached resource loading"),
            "execute": lambda *_: None,
        }
    else:
        options["override"] = lambda *_: pytest.fail("rejected deployment reached session construction")
    try:
        with pytest.raises(BackendCompatibilityError) as failure:
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=registries,
                providers=providers,
                **options,
            )
        assert failure.value.code == expected
    finally:
        providers.close()


@pytest.mark.parametrize("managed", [False, True])
def test_native_module_executes_without_autograd_in_worker(tmp_path, managed):
    import threading

    import torch

    from inference_service.backends.types import runtime_context_defaults
    from inference_service.scheduler.runtime_controller import RuntimeExecutionController

    from .runtime_scheduler_helpers import target

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    observed = []

    def execute(model, inputs):
        observed.append((threading.get_ident(), torch.is_grad_enabled()))
        return model(inputs["x"])

    with runtime_context_defaults(priority_scheduling=False, managed_execution=managed):
        runtime = ModelRuntimeFactory.load_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": lambda: torch.nn.Linear(64, 64).eval(), "execute": execute},
        )
    host = RuntimeExecutionController(target(resource="cpu"), runtime) if managed else None
    try:
        request = ModelRequest({"x": torch.ones(1024, 64)})
        context = ExecutionContext("linear", Deadline.after(5))
        # The scheduler worker must establish its own context.
        with torch.no_grad() if managed else torch.enable_grad():
            if host is None:
                result = runtime.execute(request, context)
            else:
                host.accept_epoch("test")
                sub = host.prepare(request, context)
                host.execute(sub, "test")
                host.completion(sub).result(timeout=5)
                result = host.result(sub)
        assert observed[0][1] is False
        assert (observed[0][0] != threading.get_ident()) is managed
        assert result.outputs["value"].shape == (1024, 64)
        assert result.outputs["value"].grad_fn is None
    finally:
        if host is not None:
            host.shutdown()
            runtime._close_local(ExecutionContext("cleanup"), _dispatch_key=host._authority)
        else:
            runtime.close()
        providers.close()


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("managed", [False, True])
def test_domain_callback_transfers_snapshot_of_reused_sdk_buffer(tmp_path, kind, managed):
    import numpy as np
    import torch

    from inference_service.backends.types import runtime_context_defaults
    from inference_service.scheduler.runtime_controller import RuntimeExecutionController

    from .runtime_scheduler_helpers import target

    class SDK:
        def __init__(self):
            self.buffer = np.zeros(4) if kind == "numpy" else torch.zeros(4)

        def infer(self, value):
            self.buffer[:] = value
            return self.buffer

        def close(self):
            self.buffer[:] = -1

    def execute(model, inputs):
        borrowed = model.infer(inputs["value"])
        # The domain callback knows the SDK owns and reuses this storage.
        # Transfer an independent result before leaving the execution scope.
        return borrowed.copy() if kind == "numpy" else borrowed.detach().clone()

    spec = load_inference_manifest(_create_bundle(tmp_path), "cpu")
    providers = RuntimeProviders.create(object(), object())
    with runtime_context_defaults(priority_scheduling=False, managed_execution=managed):
        runtime = ModelRuntimeFactory.load_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": SDK, "execute": execute},
        )
    host = RuntimeExecutionController(target(resource="cpu"), runtime) if managed else None
    if host is not None:
        host.accept_epoch("test")
    try:
        pending = []
        for value in (1, 2):
            request = ModelRequest({"value": value})
            context = ExecutionContext(f"request-{value}", Deadline.after(5))
            if host is None:
                pending.append(runtime.execute(request, context))
            else:
                sub = host.prepare(request, context)
                host.execute(sub, "test")
                host.completion(sub).result(timeout=5)
                pending.append(sub)
        # Request 2 reuses the SDK buffer before request 1 is consumed.
        results = [host.result(sub) for sub in pending] if host is not None else pending
        assert [result.outputs["value"].tolist() for result in results] == [[1] * 4, [2] * 4]
    finally:
        if host is not None:
            host.shutdown()
            runtime._close_local(ExecutionContext("cleanup"), _dispatch_key=host._authority)
        else:
            runtime.close()
        providers.close()
    # Close mutates SDK state after the caller has consumed both results.
    assert [result.outputs["value"].tolist() for result in results] == [[1] * 4, [2] * 4]


def test_native_load_drain_failure_keeps_cleanup_owned(tmp_path, monkeypatch):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    calls = []
    drained = [False]

    class Model:
        def close(self):
            calls.append("close")

    def drain(*_):
        if not drained[0]:
            raise RuntimeError("completion pending")

    monkeypatch.setattr(native_torch, "drain_torch_device", drain)
    providers = RuntimeProviders.create(object(), object())
    try:
        with pytest.raises(ExecutionFailure, match="completion pending"):
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec),
                builder_options={"factory": Model, "execute": lambda *_: None},
                providers=providers,
            )
        assert not calls
    finally:
        drained[0] = True
        providers.close()
    assert calls == ["close"]


@pytest.mark.parametrize("multiple_instances", [False, True])
def test_clean_failed_load_releases_instance_lease_before_retry(tmp_path, multiple_instances):
    from dataclasses import replace

    from inference_service.backends.admission import ResourceDomainAdmissions
    from inference_service.backends.types import BackendAdmissionEvidence

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    admissions = ResourceDomainAdmissions()
    providers = RuntimeProviders.create(object(), admissions)
    attempts = []

    def factory():
        attempts.append(None)
        raise RuntimeError("clean load failure")

    def builder(context):
        resource = NativeTorchResource(spec, factory, lambda *_: None)
        resource.capabilities = replace(
            resource.capabilities,
            supports_multiple_instances=multiple_instances,
            admission_evidence=BackendAdmissionEvidence(
                sdk_initialization=True, multi_instance_execution=True, failure_isolation=True, independent_close=True
            ),
        )
        return resource

    try:
        initial_callbacks = len(providers._state.cleanup._callbacks)
        for _ in range(3):
            with pytest.raises(ExecutionFailure, match="clean load failure"):
                ModelRuntimeFactory.load_session(
                    spec, registry_set=_registries(spec, builder), providers=providers, runtime_id="retry-native"
                )
            assert not admissions._instances
            assert len(providers._state.cleanup._callbacks) == initial_callbacks
        assert len(attempts) == 3
    finally:
        providers.close()


def test_native_expired_startup_never_calls_builder_or_sdk(tmp_path):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    try:
        with pytest.raises(DeadlineExceeded):
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec, lambda *_: pytest.fail("expired startup called the builder")),
                providers=providers,
                timeout=0,
            )
    finally:
        providers.close()


def test_native_startup_expiring_during_validation_does_not_construct_sdk(tmp_path, monkeypatch):
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("startup", Deadline.after(1, now=clock[0], clock=lambda: clock[0]))
    load_manifest = native_torch.load_inference_manifest
    calls = []

    def validate(*args):
        current = load_manifest(*args)
        clock[0] += timedelta(seconds=2)
        return current

    monkeypatch.setattr(native_torch, "load_inference_manifest", validate)
    providers = RuntimeProviders.create(object(), object())
    try:
        with lifecycle_scope(context), pytest.raises(ExecutionFailure) as failure:
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec),
                providers=providers,
                builder_options={"factory": lambda: calls.append("load"), "execute": lambda *_: None},
            )
        assert isinstance(failure.value.cause, DeadlineExceeded)
        assert calls == []
    finally:
        providers.close()


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_native_entry_interruption_proves_not_started_and_keeps_runtime_ready(
    tmp_path, monkeypatch, managed, interruption
):
    from inference_service.backends.types import runtime_context_defaults
    from inference_service.scheduler.runtime_controller import RuntimeExecutionController

    from .runtime_scheduler_helpers import target

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    calls = []
    with runtime_context_defaults(priority_scheduling=False, managed_execution=managed):
        runtime = ModelRuntimeFactory.load_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": object, "execute": lambda *_: calls.append("execute")},
        )
    import time

    boot_clock = [time.monotonic_ns()]
    host = (
        RuntimeExecutionController(target(resource="cpu"), runtime, now_ns=lambda: boot_clock[0]) if managed else None
    )
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("entry", Deadline.after(10, now=clock[0], clock=lambda: clock[0]))
    check = ExecutionContext.check

    def interrupt_at_entry(self, phase):
        if phase == "native_model" and self.request_id == "entry":
            if interruption == "cancel":
                self.cancellation_token.cancel("entry race")
            else:
                clock[0] += timedelta(seconds=20)
                boot_clock[0] += 20_000_000_000
        return check(self, phase)

    monkeypatch.setattr(ExecutionContext, "check", interrupt_at_entry)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            if host is None:
                runtime.execute(ModelRequest(), context)
            else:
                host.accept_epoch("test")
                sub = host.prepare(ModelRequest(), context)
                host.execute(sub, "test")
                host.completion(sub).result(timeout=2)
                assert host.query(sub, "test")["state"] == "not_started"
                host.result(sub)
        assert failure.value.evidence.state.value == "not_started"
        assert failure.value.evidence.outcome_known
        assert calls == [] and runtime.local_ready
        if host is None:
            runtime.execute(ModelRequest(), ExecutionContext("healthy-next"))
        else:
            assert host.ready
            sub = host.prepare(ModelRequest(), ExecutionContext("healthy-next"))
            host.execute(sub, "test")
            host.completion(sub).result(timeout=2)
            host.result(sub)
        assert calls == ["execute"]
    finally:
        if host is not None:
            host.shutdown()
        runtime.close()
        providers.close()


@pytest.mark.parametrize("interruption", ["cancel", "deadline", "adaptation"])
@pytest.mark.parametrize("completion_fails", [False, True])
def test_native_direct_result_discard_waits_for_device_completion(
    tmp_path, monkeypatch, interruption, completion_fails
):
    from inference_service.unified_runtime.adapters import ResultAdapter
    from inference_service.unified_runtime.errors import OutputValidationError

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    calls = []
    runtime = ModelRuntimeFactory.load_session(
        spec,
        registry_set=_registries(spec),
        providers=providers,
        builder_options={"factory": object, "execute": lambda *_: calls.append("execute")},
    )
    resource = runtime.assembly.session
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("discard", Deadline.after(1, now=clock[0], clock=lambda: clock[0]))
    check = ExecutionContext.check
    fail_drain = [completion_fails]

    def drain(*_):
        calls.append("drain")
        if fail_drain[0]:
            raise RuntimeError("device completion failed")

    def interrupt_at_result(self, phase):
        if phase == "result":
            if interruption == "cancel":
                self.cancellation_token.cancel("delivery race")
            else:
                clock[0] += timedelta(seconds=2)
        return check(self, phase)

    resource._torch = SimpleNamespace(
        no_grad=nullcontext, cuda=SimpleNamespace(device=lambda _: nullcontext(), synchronize=drain)
    )
    resource._device_name, resource._device = "cuda", 0
    monkeypatch.setattr(ExecutionContext, "check", interrupt_at_result)
    if interruption == "adaptation":

        def reject_result(*args, **kwargs):
            raise OutputValidationError("domain output validation failed")

        monkeypatch.setattr(ResultAdapter, "adapt", reject_result)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            runtime.execute(ModelRequest(), context)
        assert calls == ["execute", "drain"]
        assert failure.value.evidence.outcome_known is not completion_fails
        assert runtime.local_ready is not completion_fails
        if completion_fails:
            assert failure.value.evidence.state.value == "started"
            assert failure.value.details["completion_error"] == "device completion failed"
    finally:
        fail_drain[0] = False
        runtime.close()
        providers.close()


def test_native_requires_registered_model_identity(tmp_path):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    try:
        with pytest.raises(RuntimeRegistryError) as failure:
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec, builder=None),
                builder_options={"factory": lambda: pytest.fail("unregistered SDK loaded"), "execute": lambda *_: None},
                providers=providers,
            )
        assert failure.value.code == "session_builder_unavailable"
    finally:
        providers.close()


def test_native_adapter_works_through_session_factory_and_preserves_context(tmp_path):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    seen = []

    class RecordingResource(NativeTorchResource):
        def execute(self, request, context):
            seen.append(context)
            return super().execute(request, context)

    def builder(context):
        return RecordingResource(context.validated_manifest, object, lambda _, inputs: inputs["number"] + 1)

    providers = RuntimeProviders.create(object(), object())
    handle = ModelRuntimeFactory.load_session(spec, registry_set=_registries(spec, builder), providers=providers)
    context = ExecutionContext("outer-request", deadline=Deadline.after(5), cancellation_token=CancellationToken())
    try:
        assert handle.execute(ModelRequest({"number": 3}), context).outputs["value"] == 4
        assert seen == [context] and seen[0] is context
        assert handle.assembly.identity == spec.manifest.model
        assert handle.assembly.deployment_fingerprint == spec.fingerprint
        assert handle.assembly.execution_contract == spec.deployment.execution_contract
        assert handle.assembly.providers is providers
    finally:
        handle.close()
        providers.close()


@pytest.mark.parametrize("blocked_by", ["deadline", "cancellation"])
def test_native_request_context_blocks_sdk_execution(tmp_path, blocked_by):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    runtime = ModelRuntimeFactory.load_session(
        spec,
        registry_set=_registries(spec),
        providers=providers,
        builder_options={"factory": object, "execute": lambda *_: pytest.fail("blocked request reached the SDK")},
    )
    context = ExecutionContext("outer-request")
    if blocked_by == "deadline":
        context = ExecutionContext("outer-request", deadline=Deadline.after(0))
    else:
        context.cancellation_token.cancel("caller cancelled")
    try:
        with pytest.raises(ExecutionFailure):
            runtime.execute(ModelRequest({"number": 1}), context)
    finally:
        runtime.close()
        providers.close()


def test_native_changed_deployment_rejected_before_sdk_load(tmp_path):
    bundle = _create_bundle(tmp_path / "native")
    spec = load_inference_manifest(bundle, "cpu")
    path = bundle / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deployments"]["cpu"]["revision"] += 1
    path.write_text(json.dumps(manifest))
    providers = RuntimeProviders.create(object(), object())
    try:
        with pytest.raises(ExecutionFailure, match="deployment changed"):
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec),
                providers=providers,
                builder_options={
                    "factory": lambda: pytest.fail("changed deployment loaded"),
                    "execute": lambda *_: None,
                },
            )
    finally:
        providers.close()


def test_native_resource_rejects_unbundled_identity():
    with pytest.raises(TypeError, match="validated bundle"):
        NativeTorchResource(SimpleNamespace(), object, lambda *_: None)


def test_native_resource_can_bind_before_managed_load(tmp_path):
    from inference_service.backends.types import runtime_context_defaults
    from inference_service.scheduler.runtime_controller import RuntimeExecutionController

    from .runtime_scheduler_helpers import target

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    providers = RuntimeProviders.create(object(), object())
    with runtime_context_defaults(priority_scheduling=False, managed_execution=True):
        handle = ModelRuntimeFactory.create_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": object, "execute": lambda _, values: values["x"]},
        )
    host = None
    try:
        assert not handle.local_ready
        host = RuntimeExecutionController(target(resource="cpu"), handle)
        handle.load()
        host.accept_epoch("test")
        sub = host.prepare(ModelRequest({"x": 42}), ExecutionContext("bound-before-load"))
        host.execute(sub, "test")
        host.completion(sub).result(timeout=5)
        assert host.result(sub).outputs["value"] == 42
    finally:
        if host is not None:
            host.shutdown()
            handle._close_local(ExecutionContext("cleanup"), _dispatch_key=host._authority)
        else:
            handle.close()
        providers.close()


@pytest.mark.parametrize("granularity", ["checkpoint", "stage"])
def test_native_resource_rejects_cancellation_it_cannot_deliver(tmp_path, granularity):
    from dataclasses import replace

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    contract = spec.deployment.execution_contract.model_copy(update={"cancellation_granularity": granularity})
    spec = replace(spec, deployment=spec.deployment.model_copy(update={"execution_contract": contract}))
    with pytest.raises(ValueError, match="request_boundary"):
        NativeTorchResource(spec, object, lambda *_: None)


@pytest.mark.parametrize("managed", [False, True])
def test_native_builder_obeys_runtime_context_completion_mode(tmp_path, monkeypatch, managed):
    from contextlib import contextmanager

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    modes = []

    @contextmanager
    def scope(*args, **kwargs):
        modes.append(kwargs["complete_on_success"])
        yield

    monkeypatch.setattr(native_torch, "torch_execution_scope", scope)
    context = RuntimeContext(spec, managed_execution=managed)
    resource = build_native_torch_resource(context, factory=object, execute=lambda *_: 42)
    try:
        resource.load(context)
        assert resource.execute(ModelRequest(), ExecutionContext("mode")) == {"value": 42}
        assert resource.managed_execution is managed
        assert modes == [managed]
    finally:
        resource.close()


def test_native_close_keeps_sdk_owner_until_final_drain(tmp_path, monkeypatch):
    import gc
    import weakref

    from inference_service.backends.errors import BackendLifecycleError

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    calls, references = [], []

    class Model:
        def __init__(self):
            references.append(weakref.ref(self))

        def close(self):
            calls.append("close")

    context = RuntimeContext(spec)
    resource = build_native_torch_resource(context, factory=Model, execute=lambda *_: None)
    resource.load(context)
    drains = []

    def drain(*_):
        drains.append(None)
        if len(drains) == 2:
            raise RuntimeError("final completion pending")

    monkeypatch.setattr(native_torch, "drain_torch_device", drain)
    try:
        with pytest.raises(BackendLifecycleError) as failure:
            resource.close()
        assert failure.value.close_pending
        gc.collect()
        assert references[0]() is resource.value and resource.value is not None
        del failure
        resource.close()
        gc.collect()
        assert resource.value is None and references[0]() is None
        assert calls == ["close"]
    finally:
        resource.close()


@pytest.mark.parametrize("not_started", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_native_sdk_close_retries_only_with_nonstart_evidence(tmp_path, not_started, error_type):
    from inference_service.unified_runtime.contracts import OutcomeEvidence

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    calls = []

    class Model:
        def close(self):
            calls.append("close")
            if len(calls) == 1:
                error = error_type("SDK close failed after partial release")
                if not_started:
                    error.evidence = OutcomeEvidence.not_started("sdk_close")
                raise error

    providers = RuntimeProviders.create(object(), object())
    runtime = ModelRuntimeFactory.load_session(
        spec,
        registry_set=_registries(spec),
        providers=providers,
        builder_options={"factory": Model, "execute": lambda *_: None},
    )
    resource = runtime.assembly.session
    owner = resource.value
    with pytest.raises(ExecutionFailure) as failure:
        runtime.close()
    assert failure.value.close_pending and not runtime.closed
    if not_started:
        runtime.close()
        assert runtime.closed and resource.value is None
        assert calls == ["close", "close"]
        providers.close()
    else:
        for _ in range(2):
            with pytest.raises(ExecutionFailure) as failure:
                runtime.close()
            assert failure.value.close_pending and not runtime.closed
            assert resource.value is owner
        # Process-owner cleanup must obey the same quarantine as Handle.close.
        providers.retain_cleanup(runtime)
        with pytest.raises(ExecutionFailure):
            providers.close()
        assert not providers.closed
        assert calls == ["close"]


@pytest.mark.parametrize("drain_number", [1, 2])
def test_native_close_interrupted_drain_keeps_handle_retryable(tmp_path, monkeypatch, drain_number):
    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    calls = []
    model = SimpleNamespace(close=lambda: calls.append("close"))
    providers = RuntimeProviders.create(object(), object())
    runtime = ModelRuntimeFactory.load_session(
        spec,
        registry_set=_registries(spec),
        providers=providers,
        builder_options={"factory": lambda: model, "execute": lambda *_: None},
    )
    drains = []

    def drain(*_):
        drains.append(None)
        if len(drains) == drain_number:
            raise KeyboardInterrupt("device synchronization interrupted")

    monkeypatch.setattr(native_torch, "drain_torch_device", drain)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            runtime.close()
        assert failure.value.close_pending and not failure.value.evidence.outcome_known
        assert not runtime.closed and not runtime.ready
        assert runtime.assembly.session.value is model
        runtime.close()
        assert runtime.closed and runtime.assembly.session.value is None
        assert calls == ["close"]
    finally:
        runtime.close()
        providers.close()


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_ros_adapter_requires_loaded_native_handle_before_binding(tmp_path, monkeypatch, device):
    from inference_service.backends.types import runtime_context_defaults
    from inference_service.scheduler.runtime_protocol import ControlError
    from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter

    from .runtime_scheduler_helpers import FakeNode, target

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    bundle = _create_bundle(tmp_path / "native")
    path = bundle / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deployments"]["cpu"]["runtime_profile"]["profile"]["device"] = device
    path.write_text(json.dumps(manifest))
    spec = load_inference_manifest(bundle, "cpu")
    providers = RuntimeProviders.create(object(), object())
    with runtime_context_defaults(priority_scheduling=False, managed_execution=True):
        handle = ModelRuntimeFactory.create_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": object, "execute": lambda *_: None},
        )
    node = FakeNode({})
    try:
        from dataclasses import replace

        t = replace(
            target(resource="cpu"),
            deployment_fingerprint=spec.fingerprint,
            physical_device=f"local/{device}" + (":0" if device != "cpu" else ""),
        )
        with pytest.raises(ControlError, match="RUNTIME_LOAD_REQUIRED"):
            SchedulerRuntimeAdapter(node, t, handle)
        assert not handle.execution_policy_bound and not node.bus
    finally:
        handle.close()
        providers.close()


@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
@pytest.mark.parametrize("phase", ["first_drain", "sdk_close"])
def test_native_close_checks_budget_between_native_calls(tmp_path, monkeypatch, interruption, phase):
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    spec = load_inference_manifest(_create_bundle(tmp_path / "native"), "cpu")
    clock = [datetime.now(timezone.utc)]
    context = ExecutionContext("close", Deadline.after(1, now=clock[0], clock=lambda: clock[0]))
    calls = []

    def interrupt():
        if interruption == "deadline":
            clock[0] += timedelta(seconds=2)
        else:
            context.cancellation_token.cancel("stop cleanup")

    class Model:
        def close(self):
            calls.append("close")
            if phase == "sdk_close":
                interrupt()

    resource = NativeTorchResource(spec, Model, lambda *_: None)
    resource.load(RuntimeContext(spec))
    owner = resource.value

    def drain(*_):
        calls.append("drain")
        if phase == "first_drain":
            interrupt()

    monkeypatch.setattr(native_torch, "drain_torch_device", drain)
    try:
        with lifecycle_scope(context), pytest.raises(BackendLifecycleError) as failure:
            resource.close()
        assert failure.value.close_pending
        assert calls == (["drain"] if phase == "first_drain" else ["drain", "close"])
        assert resource.value is owner
        resource.close()
        assert calls.count("close") == 1
        assert resource.value is None
    finally:
        resource.close()
