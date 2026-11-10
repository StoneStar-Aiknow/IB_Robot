"""Library model creation uses real Global transport without a business host."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from inference_manifest import TorchRuntimeProfile
from inference_service.runtime_creation import runtime_creation
from inference_service.scheduler.runtime_environment import RuntimeEnvironment
from inference_service.scheduler.runtime_protocol import NotStartedControlError
from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    RuntimeAssembly,
    RuntimeProviders,
)

from .runtime_scheduler_helpers import StreamFixture
from .runtime_scheduler_helpers import initialization_target as target


def manifest(interface="tensor_model"):
    return SimpleNamespace(
        fingerprint="public-runtime-test",
        role_runtime_profiles={},
        bundle_root=Path("."),
        deployment=SimpleNamespace(
            backend="torch",
            artifacts={},
            role_runtime_profiles={},
            runtime_profile=SimpleNamespace(backend="torch"),
        ),
        runtime_profile=SimpleNamespace(backend="torch", profile=TorchRuntimeProfile(device="cpu")),
        manifest=SimpleNamespace(model=SimpleNamespace(interface=interface, model_type="test", operation="infer")),
    )


def test_stream_recovery_preserves_other_stream_state(global_runtime):
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime import StreamHandle

    _, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    banks, resets, drains = {}, [], []

    class Model(StreamFixture):
        def open_stream(self, context):
            stream = StreamHandle(str(uuid4()))
            banks[stream.stream_id] = 0
            return stream

        def step(self, stream, request, context):
            banks[stream.stream_id] += 1
            if request.inputs.get("fail"):
                raise BackendInferenceError("completion unknown", operation_started=True, outcome_known=False)
            return {"count": banks[stream.stream_id]}

        def reset_stream(self, stream, context):
            resets.append(stream.stream_id)
            banks[stream.stream_id] = 0

        def drain_uncertain_operations(self):
            drains.append(True)

    model = Model()
    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(
                executor=model,
                streaming_runtime=model,
                state_scope="stream",
                stateful=True,
                resettable=True,
                state_bank_mode="per_stream",
                max_open_streams=2,
            )
            handle.load()
        a = handle.open_stream(ExecutionContext("a", Deadline.after(5)))
        b = handle.open_stream(ExecutionContext("b", Deadline.after(5)))
        assert handle.step(b, ModelRequest(), ExecutionContext("b1", Deadline.after(5))).outputs == {"count": 1}
        with pytest.raises(ExecutionFailure):
            handle.step(a, ModelRequest({"fail": True}), ExecutionContext("a1", Deadline.after(5)))
        handle.reset_stream(a, ExecutionContext("reset-a", Deadline.after(5)))
        assert drains and len(resets) == 1
        assert handle.step(b, ModelRequest(), ExecutionContext("b2", Deadline.after(5))).outputs == {"count": 2}
        assert handle.step(a, ModelRequest(), ExecutionContext("a2", Deadline.after(5))).outputs == {"count": 1}
    finally:
        providers.close()


@pytest.mark.parametrize("purpose", ["online", "local", "simulation", "benchmark", "isolated_offline", "distributed"])
@pytest.mark.parametrize("failure", ["arguments", "deployment", "builder", "load"])
def test_local_node_releases_repeated_clean_factory_failures(global_runtime, tmp_path, purpose, failure):
    import gc
    import weakref

    from inference_manifest import load_inference_manifest
    from inference_service.backends.errors import BackendCompatibilityError
    from inference_service.host_lifecycle import close_node_runtimes
    from inference_service.unified_runtime import ModelRuntimeFactory
    from inference_service.unified_runtime.registry import RuntimeRegistryError

    from .test_native_object_runtime import _create_bundle, _registries

    node = Node("local_failed_creation")
    providers = RuntimeProviders.create(object(), object())
    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    registries = _registries(spec, None) if failure == "builder" else _registries(spec)
    expected = {
        "arguments": TypeError,
        "deployment": BackendCompatibilityError,
        "builder": RuntimeRegistryError,
        "load": ExecutionFailure,
    }[failure]
    refs = []

    def attempt():
        # A validation traceback can retain unrelated caller payloads as well
        # as the failed transaction. Check actual reachability after GC.
        class Payload:
            pass

        payload = Payload()
        refs.append(weakref.ref(payload))

        def fail_load():
            raise ValueError("native model load failed")

        options = {"builder_options": {"factory": fail_load, "execute": lambda *_: None}}
        if failure == "arguments":
            options["unknown_assembly_option"] = True
        elif failure == "deployment":
            options["allowed_deployments"] = frozenset({"not-this-deployment"})
        with pytest.raises(expected), runtime_creation(node=node, purpose=purpose) as construction:
            refs.append(weakref.ref(construction))
            ModelRuntimeFactory.load_session(spec, registry_set=registries, providers=providers, **options)

    try:
        for _ in range(5):
            attempt()
            assert not node._runtime_constructions
        gc.collect()
        assert all(ref() is None for ref in refs), "clean failure retained a transaction or caller payload"
    finally:
        close_node_runtimes(node)
        providers.close()
        node.destroy_node()


@pytest.mark.parametrize("failed_load", [False, True])
def test_local_node_keeps_ownership_through_creation_and_failed_cleanup(global_runtime, failed_load):
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.host_lifecycle import close_node_runtimes

    node = Node("local_creation_owner")
    closed = []
    pending = [failed_load]

    class Model:
        def load(self):
            if failed_load:
                raise ValueError("SDK load failed")

        def execute(self, *_):
            return {}

        def close(self):
            if pending[0]:
                pending[0] = False
                raise BackendLifecycleError("cleanup pending", close_pending=True)
            closed.append(True)

    try:
        try:
            with runtime_creation(node=node):
                first = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
                first.load()
                first.close()  # This scope can still attach another local model.
                second = ModelRuntimeHandle(Model())
                second.load()
        except ExecutionFailure:
            assert failed_load
        assert node._runtime_constructions and not second.closed
        close_node_runtimes(node)
        assert first.closed and second.closed and closed == [True]
        assert not node._runtime_constructions
    finally:
        close_node_runtimes(node)
        node.destroy_node()


@pytest.mark.parametrize("enabled", [False, True])
def test_creation_inherits_node_policy_before_freezing_providers(global_runtime, enabled):
    from rclpy.parameter import Parameter

    _, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(scheduler_enabled=not enabled)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    node = Node(
        "creation_configuration",
        parameter_overrides=[
            Parameter("runtime_scheduler_enabled", value=enabled),
            Parameter("runtime_scheduler_endpoint", value=endpoint),
        ],
    )
    try:
        with runtime_creation(providers=providers, node=node, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"ok": True}))
            handle.load()
        assert environment.scheduler_enabled is enabled
        assert (handle._execution_policy is not None) is enabled
        assert handle.execute(ModelRequest(), ExecutionContext("configured", Deadline.after(5))).outputs == {"ok": True}
        node.set_parameters([Parameter("runtime_scheduler_enabled", value=not enabled)])
        with (
            pytest.raises(ValueError, match="after runtime creation starts"),
            runtime_creation(providers=providers, node=node, validated_manifest=manifest()),
        ):
            pytest.fail("frozen process configuration changed")
    finally:
        providers.close()
        node.destroy_node()


def test_automatic_control_transport_inherits_explicit_host_domain(monkeypatch):
    import os

    from rclpy.context import Context
    from rclpy.parameter import Parameter

    # The isolated test domain belongs to this test; the environment default is
    # deliberately different, without creating any nodes in that default domain.
    domain = int(os.environ["IBROBOT_TEST_ROS_DOMAIN_ID"])
    monkeypatch.setenv("ROS_DOMAIN_ID", str((domain + 1) % 200))
    context = Context()
    context.init(domain_id=domain)
    node = Node(
        "explicit_domain",
        context=context,
        parameter_overrides=[
            Parameter("runtime_scheduler_enabled", value=True),
            Parameter("runtime_scheduler_endpoint", value="/explicit_domain/global"),
        ],
    )
    GlobalRuntimeEndpoint(
        node, [], endpoint="/explicit_domain/global", resources={"cpu": {"physical_device": "local/cpu"}}
    )
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    providers = RuntimeProviders.create(object(), object())
    try:
        with runtime_creation(providers=providers, node=node, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"domain": domain}))
            handle.load()
        assert providers.runtime_environment._context.get_domain_id() == domain
        assert handle.execute(ModelRequest(), ExecutionContext("domain", Deadline.after(5))).outputs == {
            "domain": domain
        }
    finally:
        providers.close()
        executor.shutdown(timeout_sec=5)
        spin.join(5)
        assert not spin.is_alive()
        node.destroy_node()
        context.shutdown()


@pytest.mark.parametrize("with_providers", [False, True])
def test_explicit_creation_consumes_projected_connection_and_publishes_before_return(global_runtime, with_providers):
    from rclpy.parameter import Parameter

    from inference_service.host_lifecycle import close_node_runtimes
    from robot_config.launch_builders.model_runtime import model_runtime_connection_parameters

    core, endpoint, _ = global_runtime
    parameters = model_runtime_connection_parameters(
        {"runtime_scheduler": {"enable": True, "runtime_endpoint": endpoint, "startup_readiness_timeout": 4.0}}
    )
    node = Node(
        "explicit_unspun_host",
        parameter_overrides=[Parameter(name, value=value) for name, value in parameters.items()],
    )
    providers = RuntimeProviders.create(object(), object()) if with_providers else None
    try:
        # No endpoint/timeout duplication and no serving executor at creation.
        with runtime_creation(node=node, providers=providers, targets=[target("projected").to_dict()]):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"ready": True}))
            handle.load()
        assert node.executor is None
        assert handle.ready
        assert handle._execution_policy._open_stream_timeout == 4.0
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        try:
            # The public call itself must be usable; no private registration waits.
            assert handle.execute(ModelRequest(), ExecutionContext("immediate", Deadline.after(3))).outputs == {
                "ready": True
            }
        finally:
            close_node_runtimes(node)
            executor.shutdown(timeout_sec=5)
            spin.join(5)
            assert not spin.is_alive()
        assert handle.closed and not core.shutdown_pending()
    finally:
        close_node_runtimes(node)
        if providers is not None:
            providers.close()
        node.destroy_node()


def test_local_creation_then_real_ros_host_reuses_disabled_providers(global_runtime):
    providers = RuntimeProviders.create(object(), object())
    node = Node("local_then_ros")
    handles = []
    try:
        for owner in (None, node):
            with runtime_creation(providers=providers, node=owner):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"local": True}))
                handle.load()
                handles.append(handle)
            assert handle.execute(ModelRequest(), ExecutionContext("local")).outputs == {"local": True}
        assert providers.runtime_environment._node is None
    finally:
        providers.close()
        node.destroy_node()
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_host_without_providers_freezes_configuration_across_creations(global_runtime, enabled):
    from rclpy.parameter import Parameter

    from inference_service.host_lifecycle import close_node_runtimes

    _, endpoint, _ = global_runtime
    node = Node(
        "explicit_frozen_configuration",
        parameter_overrides=[
            Parameter("runtime_scheduler_enabled", value=enabled),
            Parameter("runtime_scheduler_endpoint", value=endpoint),
        ],
    )
    try:
        with runtime_creation(node=node, targets=[target("first").to_dict()], timeout=5):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
            handle.load()
        assert handle.ready
        node.set_parameters([Parameter("runtime_scheduler_enabled", value=not enabled)])
        with (
            pytest.raises(ValueError, match="after runtime creation starts"),
            runtime_creation(node=node, targets=[target("second").to_dict()], timeout=5),
        ):
            pytest.fail("host changed scheduling policy while its first model was alive")
    finally:
        close_node_runtimes(node)
        node.destroy_node()


@pytest.mark.parametrize("failure", ["expired", "locked"])
def test_failed_retain_releases_admission_before_local_factory_retry(tmp_path, monkeypatch, failure):
    from inference_manifest import load_inference_manifest
    from inference_service.backends import ResourceDomainAdmissions
    from inference_service.unified_runtime import ModelRuntimeFactory

    from .test_native_object_runtime import _create_bundle, _registries

    environment = RuntimeEnvironment()
    admission = ResourceDomainAdmissions()
    providers = RuntimeProviders.create(object(), admission, runtime_environment=environment)
    spec = load_inference_manifest(_create_bundle(tmp_path / "retain"), "cpu")
    locked, release = threading.Event(), threading.Event()
    original_retain = environment.retain
    workers = []

    def hold_membership():
        with environment._lock:
            locked.set()
            assert release.wait(3)

    def fail_retain(handle):
        assert admission._instances, "failure must happen after admission acquisition"
        if failure == "expired":
            time.sleep(0.04)
        else:
            blocker = threading.Thread(target=hold_membership, daemon=True)
            workers.append(blocker)
            blocker.start()
            assert locked.wait(1)
        original_retain(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(environment, "retain", fail_retain)
            with pytest.raises(TimeoutError), runtime_creation(providers=providers, timeout=0.03):
                ModelRuntimeFactory.load_session(
                    spec,
                    registry_set=_registries(spec),
                    providers=providers,
                    builder_options={"factory": object, "execute": lambda *_: {}},
                )
        assert not admission._instances and not admission._domains
        assert not environment._handles
    finally:
        release.set()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
    with runtime_creation(providers=providers):
        handle = ModelRuntimeFactory.load_session(
            spec,
            registry_set=_registries(spec),
            providers=providers,
            builder_options={"factory": object, "execute": lambda *_: {}},
        )
    providers.close()
    assert handle.closed and not admission._instances and not admission._domains


@pytest.mark.parametrize("explicit", [False, True])
def test_completed_construction_rollback_releases_provider_callback(global_runtime, tmp_path, monkeypatch, explicit):
    import gc
    import weakref
    from dataclasses import replace

    from inference_manifest import load_inference_manifest
    from inference_manifest.runtime_identity import runtime_composition_fingerprint
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.unified_runtime import ModelRuntimeFactory

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, transport = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    spec = load_inference_manifest(_create_bundle(tmp_path / "rollback"), "cpu")
    selected = replace(
        target("rollback"),
        deployment_fingerprint=spec.fingerprint,
        composition_fingerprint=runtime_composition_fingerprint("model_runtime", {}),
    )
    supplied = {"node": transport.node, "targets": [selected.to_dict()]} if explicit else {}
    references, closes = [], []
    original = ModelRuntimeFactory.create_session

    def create(*args, **kwargs):
        handle = original(*args, **kwargs)
        references.extend(
            weakref.ref(value) for value in (handle, handle._assembly, handle._assembly.session, handle._construction)
        )

        def load():
            raise RuntimeError("load rejected")

        close = handle.close

        def pending_close(*args, **kwargs):
            closes.append(True)
            if not handle.closed:
                raise BackendLifecycleError("cleanup pending", close_pending=True)
            return close(*args, **kwargs)

        handle.load = load
        # Only Factory's first public Close fails. Construction uses its existing
        # unpublished rollback path and must then release all long-lived owners.
        handle.close = pending_close
        return handle

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ModelRuntimeFactory, "create_session", create)
            with (
                pytest.raises(RuntimeError, match="load rejected"),
                runtime_creation(validated_manifest=spec, providers=providers, timeout=5, **supplied),
            ):
                ModelRuntimeFactory.load_session(
                    spec,
                    providers=providers,
                    registry_set=_registries(spec),
                    builder_options={"factory": object, "execute": lambda *_: {}},
                )
        assert closes == [True] and not environment._handles and not core._initializations
        gc.collect()
        assert all(reference() is None for reference in references)
        assert not providers.closed
        if not explicit:
            assert environment._spin.is_alive()
    finally:
        providers.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_dtype_must_match_admitted_composition_before_session_builder(global_runtime, tmp_path, explicit):
    from dataclasses import replace

    from inference_manifest import load_inference_manifest
    from inference_manifest.runtime_identity import runtime_composition_fingerprint
    from inference_service.host_lifecycle import close_node_runtimes
    from inference_service.unified_runtime import ModelRuntimeFactory

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, transport = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "dtype"), "cpu")
    declared = {"model_dtype": "fp16"}
    selected = replace(
        target("dtype"),
        deployment_fingerprint=spec.fingerprint,
        composition_fingerprint=runtime_composition_fingerprint("model_runtime", declared),
    )
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    supplied = {"node": transport.node, "targets": [selected.to_dict()]} if explicit else {}
    calls = []
    try:
        with (
            pytest.raises(ValueError, match="composition mismatch"),
            runtime_creation(providers=providers, validated_manifest=spec, runtime_options=declared, **supplied),
        ):
            ModelRuntimeFactory.load_session(
                spec,
                providers=providers,
                registry_set=_registries(spec, lambda *_: calls.append("constructed")),
                runtime_options={"model_dtype": "fp32"},
            )
        assert not calls and not core._initializations and not core.shutdown_pending()
    finally:
        close_node_runtimes(transport.node)
        providers.close()


@pytest.fixture
def multi_role_onnx_bundle(tmp_path):
    import hashlib
    import json

    from inference_manifest import load_inference_manifest

    from .test_onnx_model_session import _multi_role_manifest, _write_onnx_bundle

    data = _multi_role_manifest()
    _write_onnx_bundle(tmp_path, data)
    for artifact in data["deployments"]["onnx_cpu"]["artifacts"].values():
        artifact["sha256"] = hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest()
    (tmp_path / "inference_manifest.json").write_text(json.dumps(data))
    return load_inference_manifest(tmp_path, "onnx_cpu")


def test_public_role_and_full_sessions_cannot_share_composition(global_runtime, tmp_path, multi_role_onnx_bundle):
    from dataclasses import replace

    import numpy as np

    from inference_service.model_sessions import OnnxRuntimeModelSession
    from inference_service.runtime_creation import load_model_runtime
    from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile

    from .test_native_object_runtime import _registries
    from .test_onnx_model_session import _fake_ort, _FakeGraph

    _, endpoint, _ = global_runtime
    spec = multi_role_onnx_bundle
    first = _FakeGraph(("spectrum",), ("enhanced",), lambda feed: [feed["spectrum"] * 2])
    second = _FakeGraph(("audio",), ("prob",), lambda feed: [np.asarray([np.mean(feed["audio"])], np.float32)])
    ort = _fake_ort(
        {str(tmp_path / "artifacts/fullsubnet.onnx"): first, str(tmp_path / "artifacts/silero_vad.onnx"): second}
    )
    registries = _registries(spec, lambda *_args, **_kwargs: OnnxRuntimeModelSession(ort_loader=lambda: ort))
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    request = ModelRequest({"host.fullsubnet.spectrum": np.ones(4, np.float32)}, metadata={"role": "fullsubnet"})
    targets = []
    try:
        for role_execution in (True, False):
            handle = load_model_runtime(
                validated_manifest=spec,
                providers=providers,
                registry_set=registries,
                role_execution=role_execution,
                timeout=5,
            )
            targets.append(handle._execution_policy.target)
            result = handle.execute(request, ExecutionContext("role" if role_execution else "full", Deadline.after(5)))
            assert len(first.feeds) == (1 if role_execution else 2)
            assert len(second.feeds) == (0 if role_execution else 1)
            key = "internal.enhanced" if role_execution else "host.silero.prob"
            np.testing.assert_allclose(result.outputs[key], 2)
            handle.close(deadline=Deadline.after(5))
        assert targets[0].composition_fingerprint != targets[1].composition_fingerprint
        profile = tmp_path.parent / "models" / "_work" / tmp_path.name / "profiling" / "role.json"
        write_runtime_profile(
            targets[0],
            [
                {
                    "operation_id": str(i),
                    "target_id": targets[0].target_id,
                    "identity": RuntimeProfileRegistry.identity(targets[0]),
                    "work": "execute",
                    "input_bucket": "same-request",
                    "occupancy_ns": 100,
                    "measured_at": time.time(),
                }
                for i in range(20)
            ],
            profile,
            min_samples=20,
        )
        RuntimeProfileRegistry(replace(targets[0], profile_path=str(profile)), min_samples=20)
        with pytest.raises(ValueError, match="profile identity mismatch"):
            RuntimeProfileRegistry(replace(targets[1], profile_path=str(profile)), min_samples=20)
    finally:
        providers.close()


def test_explicit_full_session_scope_rejects_role_assembly_before_builder(global_runtime, multi_role_onnx_bundle):
    from inference_service.model_sessions import OnnxRuntimeModelSession
    from inference_service.unified_runtime import ModelRuntimeFactory

    from .test_native_object_runtime import _registries

    core, endpoint, _ = global_runtime
    spec = multi_role_onnx_bundle
    calls = []

    def builder(*args, **kwargs):
        calls.append(True)
        return OnnxRuntimeModelSession()

    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    try:
        with (
            pytest.raises(ValueError, match="composition mismatch"),
            runtime_creation(providers=providers, validated_manifest=spec, timeout=5),
        ):
            ModelRuntimeFactory.create_session(
                spec, providers=providers, registry_set=_registries(spec, builder), role_execution=True
            )
        assert not calls and not core._initializations and not core._fenced
    finally:
        providers.close()


@pytest.mark.parametrize("purpose", ["online", "local", "simulation", "benchmark", "isolated_offline", "distributed"])
def test_isolated_creation_overrides_outer_session_scheduling_defaults(purpose):
    from inference_service.backends import RuntimeContext, runtime_context_defaults

    spec = manifest()
    spec.top_level_identity = spec.manifest.model
    with runtime_context_defaults(priority_scheduling=True, managed_execution=True):
        with runtime_creation(purpose=purpose, scheduler_enabled=False):
            context = RuntimeContext(spec, runtime_profile=TorchRuntimeProfile(device="cpu"))
            assert not context.priority_scheduling and not context.managed_execution
        restored = RuntimeContext(spec, runtime_profile=TorchRuntimeProfile(device="cpu"))
        assert restored.priority_scheduling and restored.managed_execution


@pytest.mark.parametrize("reason", ["GLOBAL_UNAVAILABLE", "INITIALIZATION_ADMISSION_TIMEOUT", "RESOURCE_BUSY"])
def test_explicit_host_enter_failure_does_not_retain_empty_construction(monkeypatch, reason):
    from contextlib import contextmanager

    from inference_service.scheduler import runtime_initialization

    @contextmanager
    def fail(*args, **kwargs):
        raise NotStartedControlError(reason)
        yield

    monkeypatch.setattr(runtime_initialization, "runtime_initialization", fail)
    node = SimpleNamespace(
        has_parameter=lambda name: False,
        declare_parameter=lambda name, value, *args: SimpleNamespace(value=value),
    )
    for _ in range(3):
        with (
            pytest.raises(NotStartedControlError, match=reason),
            runtime_creation(node=node, targets=[target("enter_failure").to_dict()], scheduler_enabled=True),
        ):
            pytest.fail("failed initialization entered construction")
        assert node._runtime_constructions == []


@pytest.mark.parametrize(
    "global_runtime",
    [{"resources": {"device": {"physical_device": "local/cuda:2"}}}],
    indirect=True,
)
@pytest.mark.parametrize("fail_load", [False, True])
def test_torch_initialization_owns_device_until_completion_or_proven_rollback(global_runtime, monkeypatch, fail_load):
    from contextlib import nullcontext
    from dataclasses import replace

    from inference_manifest.runtime_identity import runtime_composition_fingerprint
    from inference_service.backends import RuntimeContext
    from inference_service.host_lifecycle import close_node_runtimes
    from inference_service.model_sessions.torch import TorchModelSession
    from inference_service.scheduler.runtime_initialization import settle_initializations

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    core, endpoint, transport = global_runtime
    entered, release = threading.Event(), threading.Event()
    failing = [fail_load]
    spec = manifest()
    spec.top_level_identity = spec.manifest.model
    spec.manifest.model.inputs = spec.manifest.model.outputs = ()
    spec.runtime_profile.profile = TorchRuntimeProfile(device="cuda")
    selected = replace(
        target("torch_initialization"),
        hardware_resource_id="device",
        physical_device="local/cuda:2",
        deployment_fingerprint=spec.fingerprint,
        composition_fingerprint=runtime_composition_fingerprint("model_runtime", {"torch_device_index": 2}),
    )
    handles = []

    def synchronize(device):
        assert device.index == 2
        entered.set()
        if failing[0]:
            raise RuntimeError("device completion still unknown")
        assert release.wait(3)

    torch = SimpleNamespace(
        device=lambda _: SimpleNamespace(index=2),
        cuda=SimpleNamespace(
            is_available=lambda: True, device=lambda _: nullcontext(), synchronize=synchronize, empty_cache=lambda: None
        ),
    )

    def create():
        with runtime_creation(
            node=transport.node,
            scheduler_enabled=True,
            targets=[selected.to_dict()],
            endpoint=endpoint,
            validated_manifest=spec,
            runtime_options={"torch_device_index": 2},
            timeout=5,
        ):
            context = RuntimeContext(
                spec, runtime_profile=TorchRuntimeProfile(device="cuda"), runtime_options={"torch_device_index": 2}
            )
            session = TorchModelSession(lambda _: lambda inputs: {}, torch_loader=lambda: torch)
            handle = ModelRuntimeHandle(
                RuntimeAssembly(runtime_executor=session, session=session, load_context=context)
            )
            handles.append(handle)
            handle.load()
        return handle

    try:
        with ThreadPoolExecutor(1) as workers:
            creating = workers.submit(create)
            assert entered.wait(2)
            if fail_load:
                with pytest.raises(ExecutionFailure):
                    creating.result(timeout=2)
                for _ in range(5):
                    with pytest.raises(ExecutionFailure):
                        settle_initializations(transport.node)
                    assert not handles[0].closed and "device" in core._initializations
                    assert not core.resource_ready("device")
                failing[0] = False
                release.set()
                settle_initializations(transport.node)
                assert handles[0].closed and not core._initializations
            else:
                assert not creating.done() and not handles[0].local_ready
                assert "device" in core._initializations and not core.resource_ready("device")
                release.set()
                assert creating.result(timeout=3).local_ready
                assert not core._initializations
    finally:
        failing[0] = False
        release.set()
        close_node_runtimes(transport.node)


def test_same_detached_host_concurrent_model_close_keeps_control_executor(global_runtime):
    from inference_service.host_lifecycle import collect_node_runtime_transport

    core, endpoint, _ = global_runtime
    node = Node("shared_detached_close")
    handles, calls = [], []
    barrier = threading.Barrier(2)
    serving = MultiThreadedExecutor(num_threads=4)
    serving.add_node(node)
    spin = threading.Thread(target=serving.spin, daemon=True)
    spin.start()
    try:
        for i in range(2):
            with runtime_creation(
                node=node,
                targets=[target(f"shared_{i}").to_dict()],
                endpoint=endpoint,
                timeout=5,
                scheduler_enabled=True,
            ):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: calls.append("close")))
                handle.load()
            handles.append(handle)
            handle._execution_policy.adapter.wait_for_registration(ExecutionContext("registered", Deadline.after(3)))
        expires = time.monotonic() + 3
        while not all(handle._execution_policy.ready for handle in handles):
            assert time.monotonic() < expires
            time.sleep(0.01)
        serving.shutdown(timeout_sec=3)
        spin.join(3)
        serving.remove_node(node)
        assert node not in serving.get_nodes()

        def close(handle):
            barrier.wait(timeout=2)
            handle.close(deadline=Deadline.after(5))

        with ThreadPoolExecutor(2) as workers:
            pending = [workers.submit(close, handle) for handle in handles]
            for future in pending:
                future.result(timeout=6)
        assert calls == ["close", "close"]
        assert all(handle.closed for handle in handles)
        assert node.executor is None and not node._runtime_constructions
        assert not core.shutdown_pending()
        collect_node_runtime_transport(node)
        assert not node._runtime_retired_adapters
    finally:
        for handle in handles:
            handle.close(deadline=Deadline.after(5))
        serving.shutdown(timeout_sec=3)
        spin.join(3)
        node.destroy_node()


@pytest.mark.parametrize("interrupt_close", [False, True])
def test_retirement_and_peer_close_keep_shared_control_node(global_runtime, monkeypatch, interrupt_close):
    _, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    retiring, retired, closing = threading.Event(), threading.Event(), threading.Event()
    allow_retirement, allow_close = threading.Event(), threading.Event()
    handles = []
    workers = ThreadPoolExecutor(1)
    try:
        for _ in range(3):
            with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"answer": 1}))
                handle.load()
            handles.append(handle)
        first = handles[0]._execution_policy.adapter
        second = handles[1]._execution_policy.adapter
        destroy, wait_for_registration = first.destroy_transport, second.wait_for_registration

        def paused_destroy(**kwargs):
            retiring.set()
            assert allow_retirement.wait(3)
            destroy(**kwargs)
            retired.set()

        def paused_close(context):
            closing.set()  # closing_control_transport has acquired its executor.
            assert allow_close.wait(3)
            if interrupt_close:
                raise RuntimeError("close interrupted before retirement")
            return wait_for_registration(context)

        monkeypatch.setattr(first, "destroy_transport", paused_destroy)
        monkeypatch.setattr(second, "wait_for_registration", paused_close)
        handles[0].close(deadline=Deadline.after(5))
        assert retiring.wait(2)
        pending = workers.submit(handles[1].close, deadline=Deadline.after(5))
        assert closing.wait(2)
        allow_retirement.set()
        assert retired.wait(2)
        allow_close.set()
        if interrupt_close:
            with pytest.raises(ExecutionFailure, match="close interrupted"):
                pending.result(timeout=3)
        else:
            pending.result(timeout=3)
        assert environment._node.executor is environment._executor
        assert environment._node in environment._executor.get_nodes()
        assert handles[2].execute(ModelRequest(), ExecutionContext("survivor", Deadline.after(2))).outputs == {
            "answer": 1
        }
    finally:
        allow_retirement.set()
        allow_close.set()
        workers.shutdown(wait=True)
        monkeypatch.undo()
        providers.close()


@pytest.mark.parametrize("purpose", ["online", "simulation", "benchmark", "isolated_offline"])
@pytest.mark.parametrize("fail_load", [False, True])
def test_isolated_models_remain_owned_until_provider_shutdown(purpose, fail_load):
    events = []
    environment = RuntimeEnvironment(endpoint="/must_not_contact_global")
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)

    class Model:
        def load(self):
            events.append("load")
            if fail_load:
                raise ValueError("failed load")

        def execute(self, request, context):
            return {"isolated": True}

        def close(self):
            events.append("close")

    try:
        try:
            with runtime_creation(purpose=purpose, providers=providers):
                handle = ModelRuntimeHandle(Model())
                handle.load()
        except Exception:
            if not fail_load:
                raise
        else:
            assert not fail_load
            assert handle.execute(ModelRequest(), ExecutionContext("isolated")).outputs == {"isolated": True}
        assert environment._node is None  # No Global connection for isolated use.
    finally:
        providers.close()
    assert handle.closed
    assert events == ["load", "close"]
    providers.close()
    assert events.count("close") == 1


@pytest.mark.parametrize("purpose", ["online", "simulation"])
def test_public_loader_reuses_registered_domain_session(global_runtime, tmp_path, purpose):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime

    from .test_native_object_runtime import _create_bundle, _registries

    _, endpoint, _ = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    events = []

    class Domain:
        def close(self):
            events.append("closed")

    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    try:
        handle = load_model_runtime(
            validated_manifest=spec,
            registry_set=_registries(spec),
            providers=providers,
            purpose=purpose,
            timeout=5,
            builder_options={"factory": Domain, "execute": lambda model, inputs: inputs["label"]},
        )
        result = handle.execute(ModelRequest({"label": "cup"}), ExecutionContext("domain", Deadline.after(5)))
        assert result.outputs["value"] == "cup"
        assert handle.execution_policy_bound is (purpose == "online")
        handle.close(deadline=Deadline.after(5))
        assert events == ["closed"]
    finally:
        providers.close()


@pytest.mark.parametrize("purpose", ["online", "local", "simulation", "benchmark", "isolated_offline"])
def test_ownerless_scope_keeps_failed_factory_cleanup_with_providers(tmp_path, monkeypatch, purpose):
    from inference_manifest import load_inference_manifest
    from inference_service.backends import ResourceDomainAdmissions
    from inference_service.model_sessions import native_torch
    from inference_service.unified_runtime import ModelRuntimeFactory

    from .test_native_object_runtime import _create_bundle, _registries

    spec = load_inference_manifest(_create_bundle(tmp_path / "pending_load"), "cpu")
    admissions = ResourceDomainAdmissions()
    providers = RuntimeProviders.create(object(), admissions)
    closed = []
    pending = [True]

    class Model:
        def close(self):
            closed.append(True)

    def drain(*_):
        if pending[0]:
            raise RuntimeError("device completion pending")

    monkeypatch.setattr(native_torch, "drain_torch_device", drain)
    try:
        with pytest.raises(ExecutionFailure, match="device completion pending"), runtime_creation(purpose=purpose):
            ModelRuntimeFactory.load_session(
                spec,
                registry_set=_registries(spec),
                providers=providers,
                builder_options={"factory": Model, "execute": lambda *_: None},
            )
        assert not closed and admissions._instances
        pending[0] = False
        providers.close()
        assert closed == [True]
        assert not admissions._instances and not admissions._domains
    finally:
        pending[0] = False
        providers.close()


def test_public_stream_loader_owns_session_and_all_stream_operations(tmp_path):
    from inference_service.backends import BackendCapabilities
    from inference_service.runtime_creation import load_streaming_model_runtime
    from inference_service.unified_runtime import StreamHandle

    from .test_native_object_runtime import _registries
    from .test_onnx_model_session import _context, _stateful_manifest

    spec = _context(tmp_path, _stateful_manifest()).validated_manifest
    events = []
    session = SimpleNamespace(
        capabilities=BackendCapabilities(stateful=True, resettable=True),
        load=lambda context: events.append("load"),
        close=lambda: events.append("close"),
        execute=lambda *_: {},
        open_stream=lambda context: StreamHandle("native-stream"),
        step=lambda stream, request, context: {"value": request.inputs["value"] + 1},
        reset_stream=lambda stream, context: events.append("reset_stream"),
        close_stream=lambda stream, context: events.append("close_stream"),
    )
    providers = RuntimeProviders.create(object(), object(), runtime_environment=RuntimeEnvironment())
    try:
        handle = load_streaming_model_runtime(
            validated_manifest=spec,
            registry_set=_registries(spec, lambda *_: session),
            providers=providers,
            purpose="simulation",
        )
        stream = handle.open_stream(ExecutionContext("open"))
        assert handle.step(stream, ModelRequest({"value": 2}), ExecutionContext("step")).outputs["value"] == 3
        handle.reset_stream(stream, ExecutionContext("reset"))
        handle.close_stream(stream, ExecutionContext("close-stream"))
        handle.close()
        assert events == ["load", "reset_stream", "close_stream", "close"]
        assert handle.assembly.session is session
    finally:
        providers.close()


@pytest.fixture
def global_runtime(request, tmp_path):
    rclpy.init()
    node = Node(f"public_runtime_global_{uuid4().hex}")
    endpoint = f"/public_runtime_{uuid4().hex}/global"
    settings = getattr(request, "param", {})
    if settings.get("scoped_instances"):
        from inference_manifest.runtime_identity import runtime_composition_fingerprint
        from inference_service.scheduler.runtime_protocol import RuntimeExecutionTarget

        settings["targets"] = [
            RuntimeExecutionTarget(
                target_id=f"voice/{name}",
                endpoint=f"{endpoint}/{name}",
                lifecycle_group=f"voice/{name}",
                template_only=True,
                hardware_resource_id="cpu",
                physical_device="local/cpu",
                deployment_fingerprint=manifest().fingerprint,
                composition_fingerprint=runtime_composition_fingerprint("model_runtime", {}),
                configuration_generation="test",
                interface="tensor_model",
                model_type="test",
                operation="infer",
                request_timeout_ms=budget,
            )
            for name, budget in (("vad", 1000), ("direction_vad", 2000))
        ]
    if settings.get("prediction"):
        from inference_manifest.runtime_identity import runtime_composition_fingerprint
        from inference_service.scheduler.runtime_controller import freeze_inputs
        from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry, write_runtime_profile
        from inference_service.scheduler.runtime_protocol import RuntimeExecutionTarget

        profile = tmp_path / "models/_work/test/profiling/runtime.json"
        target = RuntimeExecutionTarget(
            target_id="local/test",
            template_only=True,
            endpoint=f"{endpoint}/host",
            lifecycle_group="local/test",
            hardware_resource_id="cpu",
            physical_device="local/cpu",
            deployment_fingerprint=manifest().fingerprint,
            composition_fingerprint=runtime_composition_fingerprint("model_runtime", {}),
            configuration_generation="test",
            interface="tensor_model",
            model_type="test",
            operation="infer",
            predictive_admission=True,
            hardware_profile_fingerprint="cpu-test",
            runtime_version="test-v1",
            profile_path=str(profile),
        )
        bucket = freeze_inputs(ModelRequest(), 4096)[3]
        write_runtime_profile(
            target,
            [
                {
                    "operation_id": str(uuid4()),
                    "target_id": target.target_id,
                    "identity": RuntimeProfileRegistry.identity(target),
                    "work": "execute",
                    "input_bucket": bucket,
                    "occupancy_ns": 100000000,
                    "measured_at": time.time(),
                }
                for _ in range(10000)
            ],
            profile,
        )
        settings = {
            "targets": [] if settings["prediction"] == "missing" else [target],
            "resources": {
                "cpu": {
                    "physical_device": "local/cpu",
                    "predictive_admission": True,
                    "hardware_profile_fingerprint": "cpu-test",
                    "managed_coverage": True,
                }
            },
        }
    transport = GlobalRuntimeEndpoint(
        node,
        settings.get("targets", []),
        endpoint=endpoint,
        resources=settings.get("resources", {"cpu": {"physical_device": "local/cpu"}}),
    )
    core = transport.core
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield core, endpoint, transport
    finally:
        executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.try_shutdown()
        assert not thread.is_alive()


@pytest.mark.parametrize("global_runtime", [{"prediction": "valid"}], indirect=True)
def test_library_model_automatically_joins_predictive_admission(global_runtime):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )

    class Model:
        runtime_version = "test-v1"

        def execute(self, request, context):
            return {"scheduled": True}

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = Model()
            handle = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=model, session=model))
            handle.load()
        participants = set(core.targets) - {"local/test"}
        assert len(participants) == 1
        assert core.targets[next(iter(participants))].profile_path == core.targets["local/test"].profile_path
        result = handle.execute(ModelRequest(), ExecutionContext("predictive", Deadline.after(5)))
        assert result.outputs == {"scheduled": True}
        handle.close(deadline=Deadline.after(5))
        assert not core._fenced
    finally:
        providers.close()


@pytest.mark.parametrize("global_runtime", [{"prediction": "missing"}], indirect=True)
def test_prediction_without_profile_rejects_before_model_construction(global_runtime):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    constructed = []
    try:
        with (
            pytest.raises(NotStartedControlError, match="RUNTIME_PROFILE_REQUIRED.*offline profile"),
            runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5),
        ):
            constructed.append(True)
        assert not constructed
        assert not core._initializations
    finally:
        providers.close()


@pytest.mark.parametrize("interface", ["tensor_model", "policy"])
def test_library_models_share_global_ownership_from_load_to_concurrent_close(global_runtime, interface):
    core, endpoint, transport = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    events = []
    handles = []

    class Model:
        def __init__(self, index):
            assert "cpu" in core._initializations
            self.index = index
            events.append((index, "construct"))

        def load(self):
            assert "cpu" in core._initializations
            events.append((self.index, "load"))

        def execute(self, request, context):
            assert not core._initializations
            events.append((self.index, "execute"))
            return request.inputs

        def close(self):
            events.append((self.index, "close"))

    try:
        for index in range(2):
            with runtime_creation(providers=providers, validated_manifest=manifest(interface), timeout=5):
                handle = ModelRuntimeHandle(Model(index))
                handle.load()
                assert not handle.ready
            handles.append(handle)
            assert handle.ready, (
                handle._execution_policy.adapter._resource_ready,
                handle.local_ready,
                handle._execution_policy.controller.ready,
                core._hosts,
                core._fenced,
            )
            result = handle.execute(ModelRequest({"value": index}), ExecutionContext(str(index), Deadline.after(5)))
            assert result.outputs == {"value": index}
        assert len(core.targets) == 2
        assert {t.hardware_resource_id for t in core.targets.values()} == {"cpu"}
        assert not hasattr(transport, "pipeline_registration_handler")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(handle.close, deadline=Deadline.after(5)) for handle in handles]
            for future in futures:
                future.result(timeout=8)
        assert all(handle.closed for handle in handles)
        assert all(events.count((index, "close")) == 1 for index in range(2))
        assert not core._fenced
        assert not core._initializations
    finally:
        providers.close()


def test_missing_global_rejects_before_any_backend_construction():
    environment = RuntimeEnvironment(endpoint=f"/absent_{uuid4().hex}", scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    constructed = []
    try:
        with (
            pytest.raises(NotStartedControlError, match="GLOBAL_UNAVAILABLE"),
            runtime_creation(
                providers=providers,
                validated_manifest=manifest(),
                timeout=0.1,
            ),
        ):
            constructed.append(True)
        assert constructed == []
    finally:
        providers.close()


def test_unknown_device_rejects_before_backend_construction(global_runtime):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    validated = manifest()
    validated.runtime_profile.profile = TorchRuntimeProfile(device="mps")
    constructed = []
    try:
        with (
            pytest.raises(Exception, match="UNKNOWN_RESOURCE"),
            runtime_creation(
                providers=providers,
                validated_manifest=validated,
                timeout=5,
            ),
        ):
            constructed.append(True)
        assert constructed == []
        assert not core._initializations
    finally:
        providers.close()


def test_stream_open_waits_for_the_other_models_device_ownership(global_runtime):
    from inference_service.unified_runtime import StreamHandle

    _, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    executing, release, opened = threading.Event(), threading.Event(), threading.Event()

    class BusyModel:
        def execute(self, request, context):
            executing.set()
            assert release.wait(5)
            return {}

    class StreamModel(StreamFixture):
        def open_stream(self, context):
            opened.set()  # Represents device reset performed by stream Open.
            return StreamHandle("stream")

        def close_stream(self, stream, context):
            pass

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            busy = ModelRuntimeHandle(BusyModel())
            busy.load()
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5, request_timeout_ms=50):
            streaming = ModelRuntimeHandle(
                executor=StreamModel(),
                streaming_runtime=StreamModel(),
                state_scope="stream",
                stateful=True,
            )
            streaming.load()
        with ThreadPoolExecutor(max_workers=2) as workers:
            run = workers.submit(busy.execute, ModelRequest(), ExecutionContext("busy", Deadline.after(5)))
            assert executing.wait(2)
            opening = workers.submit(streaming.open_stream, ExecutionContext("open"))
            try:
                assert not opened.wait(0.15), "stream Open bypassed device admission"
            finally:
                release.set()
            run.result(timeout=5)
            stream = opening.result(timeout=5)
            assert opened.is_set()
            streaming.close_stream(stream, ExecutionContext("stream-close", Deadline.after(5)))
        streaming.close(deadline=Deadline.after(5))
        busy.close(deadline=Deadline.after(5))
    finally:
        release.set()
        providers.close()


def test_stream_close_retries_unfinished_cleanup_through_global_recovery(global_runtime):
    from inference_service.unified_runtime import StreamHandle

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    events = []

    class StreamModel(StreamFixture):
        failures = 2

        def open_stream(self, context):
            return StreamHandle("stream")

        def close_stream(self, stream, context):
            events.append("close_stream")
            if self.failures:
                self.failures -= 1
                raise RuntimeError("native stream cleanup pending")

        def drain_uncertain_operations(self):
            events.append("drain")

        def close(self):
            events.append("close_model")

    model = StreamModel()
    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(
                executor=model, streaming_runtime=model, state_scope="stream", stateful=True, resettable=True
            )
            handle.load()
        stream = handle.open_stream(ExecutionContext("open", Deadline.after(5)))
        for attempt in range(2):
            with pytest.raises(ExecutionFailure):
                handle.close_stream(stream, ExecutionContext(f"close-{attempt}", Deadline.after(5)))
            assert handle.open_stream_count == 1
            assert "close_model" not in events
        handle.close_stream(stream, ExecutionContext("close-success", Deadline.after(5)))
        assert handle.open_stream_count == 0
        assert events.count("close_stream") == 3
        assert "drain" in events
        assert not core._fenced
        handle.close(deadline=Deadline.after(5))
        assert events[-1] == "close_model"
    finally:
        providers.close()


def test_repeated_runtime_close_reclaims_entities_while_environment_stays_alive(global_runtime):
    _, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    try:
        environment.prepare(manifest(), {}, 5, 5000)

        def entity_counts():
            return tuple(
                len(tuple(getattr(environment._node, field))) for field in ("services", "clients", "timers", "guards")
            )

        baseline = entity_counts()
        for index in range(6):
            with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
                handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda request, context: {}))
                handle.load()
            handle.execute(ModelRequest(), ExecutionContext(f"cycle-{index}", Deadline.after(5)))
            handle.close(deadline=Deadline.after(5))
            assert not environment._handles
            expires = time.monotonic() + 3
            while entity_counts() != baseline:
                assert environment._spin.is_alive()
                assert time.monotonic() < expires, "retired host transport was not reclaimed"
                time.sleep(0.01)
            assert environment._spin.is_alive()
    finally:
        providers.close()


def test_close_timeout_preserves_environment_owner_and_transport_until_retry(global_runtime):
    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    entered, release = threading.Event(), threading.Event()
    closes = []

    def execute(request, context):
        entered.set()
        assert release.wait(5)
        return {}

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            busy = ModelRuntimeHandle(executor=SimpleNamespace(execute=execute))
            busy.load()
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            retiring = ModelRuntimeHandle(
                executor=SimpleNamespace(execute=lambda *_: {}, close=lambda: closes.append(True))
            )
            retiring.load()
        adapter = retiring._execution_policy.adapter
        with ThreadPoolExecutor(max_workers=1) as workers:
            running = workers.submit(busy.execute, ModelRequest(), ExecutionContext("busy", Deadline.after(5)))
            try:
                assert entered.wait(2)
                with pytest.raises(ExecutionFailure):
                    retiring.close(deadline=Deadline.after(0.05))
                assert retiring in environment._handles
                assert not closes
                assert adapter._service in tuple(environment._node.services)
                assert adapter._client in tuple(environment._node.clients)
                assert environment._spin.is_alive()
            finally:
                release.set()
            running.result(timeout=5)
        retiring.close(deadline=Deadline.after(5))
        assert retiring not in environment._handles
        assert closes == [True]
        expires = time.monotonic() + 5
        while not (core.resource_ready("cpu") and busy.ready):
            assert environment._spin.is_alive()
            assert time.monotonic() < expires, "peer readiness did not recover after retirement"
            time.sleep(0.01)
        assert busy.execute(ModelRequest(), ExecutionContext("peer", Deadline.after(5))).outputs == {}
        busy.close(deadline=Deadline.after(5))
    finally:
        release.set()
        providers.close()


def test_repeated_failed_load_retires_clean_handles_without_closing_environment(global_runtime):
    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    releases = []

    class PartialModel:
        def load(self):
            raise RuntimeError("load rejected after acquisition")

        def close(self):
            releases.append(True)

    try:
        for _ in range(4):
            with (
                pytest.raises(ExecutionFailure, match="load rejected"),
                runtime_creation(
                    providers=providers,
                    validated_manifest=manifest(),
                    timeout=5,
                ),
            ):
                handle = ModelRuntimeHandle(PartialModel())
                handle.load()
            assert handle.closed
            assert not environment._handles
            assert not core._initializations
            assert not core._fenced
        assert len(releases) == 4
        assert environment._spin.is_alive()
    finally:
        providers.close()


def test_failed_publication_closes_unpublished_models_through_global(global_runtime, monkeypatch):
    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    released = []

    def failed_publication(*args):
        assert not core._initializations
        raise TimeoutError("registration confirmation lost")

    monkeypatch.setattr(environment, "publish", failed_publication)
    try:
        for _ in range(3):
            with (
                pytest.raises(TimeoutError, match="confirmation"),
                runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5),
            ):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: released.append(True)))
                handle.load()
            assert handle.closed
            assert not environment._handles
            assert not core._initializations
            assert not core._fenced
        assert len(released) == 3
    finally:
        providers.close()


@pytest.mark.parametrize("raises", [False, True])
def test_incomplete_binding_reclaims_host_transport_after_construction_rollback(global_runtime, monkeypatch, raises):
    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    releases = []

    # Inject an incomplete policy installation after real host transport was
    # created. Construction must reject it and retire the unpublished host.
    def fail_binding(*args, **kwargs):
        if raises:
            raise RuntimeError("policy installation failed")
        return None

    monkeypatch.setattr(ModelRuntimeHandle, "bind_execution_policy", fail_binding)

    def entity_counts():
        return tuple(
            len(tuple(getattr(environment._node, field))) for field in ("services", "clients", "timers", "guards")
        )

    try:
        environment.prepare(manifest(), {}, 5, 5000)
        baseline = entity_counts()
        for _ in range(3):
            with (
                pytest.raises(
                    Exception, match="policy installation failed" if raises else "did not install an execution policy"
                ),
                runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5),
            ):
                handle = ModelRuntimeHandle(
                    executor=SimpleNamespace(execute=lambda *_: {}, close=lambda: releases.append(True))
                )
                handle.load()
            assert handle.closed
            assert not environment._handles
            assert not core._initializations
            assert not core._fenced
            expires = time.monotonic() + 3
            while entity_counts() != baseline:
                assert environment._spin.is_alive()
                assert time.monotonic() < expires, "rolled-back host transport was not reclaimed"
                time.sleep(0.01)
        assert len(releases) == 3
    finally:
        providers.close()


def test_registry_validation_failure_releases_initialization_before_retry(global_runtime):
    from inference_service.unified_runtime.registry import (
        RuntimeRegistryError,
        SessionBuilderKey,
        SessionBuilderRegistry,
    )

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    try:
        for _ in range(2):
            with (
                pytest.raises(RuntimeRegistryError, match="unavailable"),
                runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5),
            ):
                SessionBuilderRegistry().create(SessionBuilderKey("tensor_model", "missing", "infer", "torch"))
            assert not core._initializations
            assert not core._fenced
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
            handle.load()
        handle.execute(ModelRequest(), ExecutionContext("after-validation", Deadline.after(5)))
        handle.close(deadline=Deadline.after(5))
    finally:
        providers.close()


@pytest.mark.parametrize("custom", [False, True])
def test_default_process_dependencies_inherit_owning_node_endpoint(global_runtime, custom, monkeypatch):
    import sys

    from rclpy.parameter import Parameter

    from inference_service.runtime_composition import (
        build_model_service_runtime_dependencies,
        require_runtime_dependencies,
    )

    core, endpoint, _ = global_runtime
    # The process default disagrees with the owning node's explicit override.
    monkeypatch.setattr(sys, "argv", ["test", "--ros-args", "-p", "runtime_scheduler_endpoint:=/wrong_global"])
    owner = Node(
        "configured_model_owner",
        parameter_overrides=[
            Parameter("runtime_scheduler_enabled", value=True),
            Parameter("runtime_scheduler_endpoint", value=endpoint),
        ],
    )
    dependencies = build_model_service_runtime_dependencies(
        **({"providers": RuntimeProviders.create(object(), object())} if custom else {})
    )
    try:
        _, providers = require_runtime_dependencies(
            dependencies.registry_set, dependencies.providers, owner="test", node=owner
        )
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {}))
            handle.load()
        assert providers.runtime_environment.endpoint == endpoint
        assert len(core.targets) == 1
        handle.execute(ModelRequest(), ExecutionContext("configured-endpoint", Deadline.after(5)))
        providers.close()
        assert handle.closed
    finally:
        dependencies.providers.close()
        owner.destroy_node()


@pytest.mark.parametrize("global_runtime", [{"prediction": "valid"}], indirect=True)
def test_two_instances_share_profile_without_sharing_lifecycle(global_runtime):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    handles = []

    class Model:
        runtime_version = "test-v1"

        def execute(self, request, context):
            return {"scheduled": True}

    try:
        for _ in range(2):
            with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
                model = Model()
                handle = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=model, session=model))
                handle.load()
            handles.append(handle)
        assert handles[0]._execution_policy.target.target_id != handles[1]._execution_policy.target.target_id
        handles[0].close(deadline=Deadline.after(5))
        handles[1]._execution_policy.adapter.refresh_registration(
            ExecutionContext("peer-ready", Deadline.after(5)), require_ready=True
        )
        assert handles[1].execute(ModelRequest(), ExecutionContext("survivor", Deadline.after(5))).outputs["scheduled"]
        assert not core._fenced
    finally:
        providers.close()


def test_finish_timeout_retains_receipt_until_close(global_runtime, monkeypatch):
    from inference_service.scheduler.runtime_protocol import ControlError

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    original = core.finish_initialization
    tokens = []

    def unavailable(resource, token, **kwargs):
        tokens.append(token)
        raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")

    # Drop Finish before application for the entire initial wait window.
    monkeypatch.setattr(core, "finish_initialization", unavailable)
    handle = None
    try:
        with (
            pytest.raises(ControlError),
            runtime_creation(providers=providers, validated_manifest=manifest(), timeout=1),
        ):
            handle = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *args: {}))
            handle.load()
        assert core._initializations
        receipts = providers.runtime_environment._node._runtime_initialization_receipts
        assert len(receipts) == 1
        monkeypatch.setattr(core, "finish_initialization", original)
        handle.close(deadline=Deadline.after(5))
        assert not core._initializations and not receipts
        assert handle.closed
        assert len(set(tokens)) == 1
    finally:
        monkeypatch.setattr(core, "finish_initialization", original)
        providers.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_close_inside_creation_retires_initialization_without_publishing(global_runtime, explicit):
    from inference_service.host_lifecycle import close_node_runtimes

    core, endpoint, transport = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    options = (
        {
            "node": transport.node,
            "targets": [target("closed_during_load").to_dict()],
            "endpoint": endpoint,
            "scheduler_enabled": True,
        }
        if explicit
        else {"providers": providers, "validated_manifest": manifest()}
    )
    released = []
    try:
        with runtime_creation(**options, timeout=5):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: released.append(True)))
            handle.load()
            handle.close()
        assert handle.closed and released == [True]
        assert not core._initializations and not core.shutdown_pending()
        assert not environment._handles
    finally:
        close_node_runtimes(transport.node)
        providers.close()


@pytest.mark.parametrize(
    "global_runtime",
    [{"resources": {"cpu": {"physical_device": "local/cpu"}, "other": {"physical_device": "local/cuda:0"}}}],
    indirect=True,
)
@pytest.mark.parametrize("environment_close", [False, True])
def test_failed_peer_initialization_cannot_block_independent_close(global_runtime, environment_close, monkeypatch):
    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    released, pending = [], [True]

    class FailingModel:
        def load(self):
            raise RuntimeError("partial SDK load")

        def execute(self, *_):
            pytest.fail("failed model must not execute")

        def close(self):
            if pending[0]:
                error = RuntimeError("device cleanup pending")
                error.close_pending = True
                raise error

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            healthy = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: released.append(True)))
            healthy.load()
        spec = manifest()
        spec.runtime_profile.profile = TorchRuntimeProfile(device="cuda")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with (
            pytest.raises(ExecutionFailure),
            runtime_creation(
                providers=providers, validated_manifest=spec, runtime_options={"torch_device_index": 0}, timeout=5
            ),
        ):
            failed = ModelRuntimeHandle(FailingModel())
            failed.load()
        if environment_close:
            with pytest.raises(ExecutionFailure):
                providers.close()
        else:
            healthy.close(deadline=Deadline.after(5))
        assert healthy.closed and released == [True]
        assert not failed.closed and failed in environment._handles
        assert "other" in core._initializations
    finally:
        pending[0] = False
        providers.close()


def test_retired_adapter_is_released_while_control_thread_remains_alive():
    import gc
    import weakref

    released = threading.Event()
    environment = RuntimeEnvironment()
    environment._node = SimpleNamespace(executor=None)
    environment._executor = SimpleNamespace(
        spin_once=lambda **_: environment._stopping.wait(0.01),
        _cb_iter=None,
        remove_node=lambda _: None,
        add_node=lambda _: None,
        shutdown=lambda: None,
    )

    class Adapter:
        def destroy_transport(self, *, _quiescent=False):
            assert _quiescent
            pass

    adapter = Adapter()
    reference = weakref.ref(adapter, lambda _: released.set())
    environment._retired.put(adapter)
    del adapter
    worker = threading.Thread(target=environment._spin_control)
    worker.start()
    try:
        until = time.monotonic() + 1
        while environment._retired.unfinished_tasks and time.monotonic() < until:
            time.sleep(0.01)
        assert environment._retired.unfinished_tasks == 0
        gc.collect()
        assert released.wait(0.5) and reference() is None
        assert worker.is_alive()
    finally:
        environment._stopping.set()
        worker.join(2)


@pytest.mark.parametrize("lost_reply", ["finish", "ack"])
def test_lost_finish_reply_can_settle_after_retention_window(global_runtime, monkeypatch, lost_reply):
    from inference_service.scheduler.runtime_protocol import ControlError

    core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    original = core.finish_initialization
    receipt = []

    def lose_reply(*args, **kwargs):
        result = original(*args, **kwargs)
        receipt.append(result)
        if kwargs.get("acknowledged", False) == (lost_reply == "ack"):
            raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")
        return result

    monkeypatch.setattr(core, "finish_initialization", lose_reply)
    try:
        with (
            pytest.raises(ControlError),
            runtime_creation(providers=providers, validated_manifest=manifest(), timeout=2),
        ):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
            handle.load()
        assert receipt and not core._initializations
        # Advance only collection; subsequent ROS statuses still use the real clock.
        with monkeypatch.context() as patch:
            patch.setattr(core, "now_ns", lambda: time.monotonic_ns() + 301_000_000_000)
            core.collect()
        monkeypatch.setattr(core, "finish_initialization", original)
        handle.close(deadline=Deadline.after(5))
        assert handle.closed and not environment._handles
        assert not environment._node._runtime_initialization_receipts
    finally:
        monkeypatch.setattr(core, "finish_initialization", original)
        providers.close()


@pytest.mark.parametrize(
    "global_runtime",
    [{"resources": {"cpu": {"physical_device": "local/cpu"}, "other": {"physical_device": "local/cuda:0"}}}],
    indirect=True,
)
def test_explicit_node_close_preserves_failed_initializer_and_closes_peer(global_runtime):
    from dataclasses import replace

    from inference_service.host_lifecycle import close_node_runtimes

    core, endpoint, transport = global_runtime
    pending, released = [True], []

    def cleanup():
        if pending[0]:
            error = RuntimeError("cleanup pending")
            error.close_pending = True
            raise error

    def fail_load():
        raise RuntimeError("partial load")

    # Fail first: its retained owner must not abort the node's later Close loop.
    failed_target = replace(target("failed"), hardware_resource_id="other", physical_device="local/cuda:0")
    try:
        with (
            pytest.raises(ExecutionFailure),
            runtime_creation(
                node=transport.node,
                targets=[failed_target.to_dict()],
                endpoint=endpoint,
                timeout=5,
                scheduler_enabled=True,
            ),
        ):
            failed = ModelRuntimeHandle(SimpleNamespace(load=fail_load, close=cleanup, execute=lambda *_: {}))
            failed.load()
        with runtime_creation(
            node=transport.node,
            targets=[target("healthy").to_dict()],
            endpoint=endpoint,
            timeout=5,
            scheduler_enabled=True,
        ):
            healthy = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: released.append(True)))
            healthy.load()
        with pytest.raises(ExecutionFailure):
            close_node_runtimes(transport.node)
        assert healthy.closed and released == [True]
        assert not failed.closed and "other" in core._initializations
        assert len(transport.node._runtime_constructions) == 1
    finally:
        pending[0] = False
        close_node_runtimes(transport.node)


@pytest.mark.parametrize("lost_responses", ["first", "all"])
def test_resource_discovery_retries_lost_responses_within_creation_budget(global_runtime, monkeypatch, lost_responses):
    from rclpy.service import Service

    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    core, endpoint, transport = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    send_response = Service.send_response
    replies = []

    def lose_response(service, response, header):
        if service is transport.service:
            replies.append(response.payload)
            if lost_responses == "all" or len(replies) == 1:
                return
        return send_response(service, response, header)

    try:
        # Establish DDS discovery independently of the injected response loss.
        environment.prepare(manifest(), {}, 5, 5000)
        monkeypatch.setattr(Service, "send_response", lose_response)
        started = time.monotonic()
        with lifecycle_budget(1):
            if lost_responses == "all":
                with pytest.raises(NotStartedControlError, match="RESOURCE_DISCOVERY_TIMEOUT"):
                    environment.prepare(manifest(), {}, 5, 5000)
            else:
                _, selected = environment.prepare(manifest(), {}, 5, 5000)
                assert selected["physical_device"] == "local/cpu"
        assert time.monotonic() - started < 2, "retries renewed the inherited creation deadline"
        assert len(replies) >= 2 and len(set(replies)) == 1
        assert not environment._client._pending_requests
        assert not core._initializations
    finally:
        environment.close()


def test_lost_begin_and_abort_are_replayed_after_transport_recovers(global_runtime, monkeypatch):
    from inference_service.scheduler import runtime_initialization
    from inference_service.scheduler.runtime_protocol import ControlError
    from inference_service.unified_runtime.errors import DeadlineExceeded

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    begin, abort = core.begin_initialization, core.abort_initialization
    tokens, aborts = [], []
    check_startup = runtime_initialization.check_startup

    def lost_begin(statuses, token, **kwargs):
        tokens.append(token)
        begin(statuses, token, **kwargs)
        raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")

    def lost_abort(*args, **kwargs):
        aborts.append((args, kwargs))
        raise ControlError("INITIALIZATION_CONTROL_TIMEOUT")

    def expire_after_begin_replay():
        check_startup()
        if len(tokens) >= 2:
            raise DeadlineExceeded(phase="startup")

    monkeypatch.setattr(core, "begin_initialization", lost_begin)
    monkeypatch.setattr(core, "abort_initialization", lost_abort)
    monkeypatch.setattr(runtime_initialization, "check_startup", expire_after_begin_replay)
    try:
        providers.runtime_environment.prepare(manifest(), {}, 5, 5000)
        with (
            pytest.raises(ControlError),
            # Trigger expiry only after two accepted Begins, independently of
            # cold DDS discovery. Abort loss must retain that original owner.
            runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5),
        ):
            pytest.fail("SDK construction must not start without a Begin reply")
        assert len(tokens) >= 2 and aborts, "the lost-reply injections were never reached"
        assert core._initializations
        assert providers.runtime_environment._node._runtime_initialization_receipts
        monkeypatch.setattr(core, "abort_initialization", abort)
        providers.close()
        assert not core._initializations
        assert len(set(tokens)) == 1
    finally:
        monkeypatch.setattr(core, "begin_initialization", begin)
        monkeypatch.setattr(core, "abort_initialization", abort)
        providers.close()


def test_failed_unbound_cleanup_can_retry_under_original_initialization(global_runtime):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    pending = True
    calls = []

    class Model:
        def load(self):
            raise RuntimeError("partial SDK load")

        def execute(self, request, context):
            pytest.fail("failed initialization must never execute")

        def close(self):
            calls.append("close")
            assert core._initializations
            if pending:
                error = RuntimeError("device cleanup pending")
                error.close_pending = True
                raise error

    try:
        with (
            pytest.raises(ExecutionFailure),
            runtime_creation(providers=providers, validated_manifest=manifest(), timeout=2),
        ):
            handle = ModelRuntimeHandle(Model())
            handle.load()
        assert core._initializations and not handle.closed
        with pytest.raises(ExecutionFailure, match="close_drain_failed"):
            providers.close()
        original_epoch = core.epoch
        core.epoch = str(uuid4())
        attempts = len(calls)
        try:
            with pytest.raises(Exception, match="EPOCH"):
                providers.close()
            assert len(calls) == attempts
            assert not handle.closed
        finally:
            core.epoch = original_epoch
        pending = False
        providers.close()
        assert handle.closed and not core._initializations
        assert not core._fenced
        assert len(calls) >= 3
    finally:
        pending = False
        providers.close()


@pytest.mark.parametrize("global_runtime", [{"scoped_instances": True}], indirect=True)
def test_process_configuration_selects_same_model_instance_policy_over_ros(global_runtime):
    from rclpy.parameter import Parameter

    core, endpoint, _ = global_runtime
    for name, budget in (("vad", 1000), ("direction_vad", 2000)):
        node = Node(
            "consumer_" + name,
            parameter_overrides=[
                Parameter("runtime_scheduler_endpoint", value=endpoint),
                Parameter("configured_target_ids", value=[f"voice/{name}"]),
            ],
        )
        environment = RuntimeEnvironment(scheduler_enabled=True)
        environment.configure_from_node(node)
        providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
        try:
            previous = set(core.targets)
            with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda request, context: {}))
                handle.load()
            dynamic = [core.targets[key] for key in set(core.targets) - previous]
            assert len(dynamic) == 1 and dynamic[0].request_timeout_ms == budget
            handle.execute(ModelRequest(), ExecutionContext("request"))
        finally:
            providers.close()
            node.destroy_node()


def test_existing_stream_waits_for_peer_initialization_without_restarting(global_runtime):
    from inference_service.unified_runtime import StreamHandle

    _, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    loading, release = threading.Event(), threading.Event()
    executions = []

    class StreamModel(StreamFixture):
        def open_stream(self, context):
            return StreamHandle("live")

        def step(self, stream, request, context):
            executions.append(request.inputs["value"])
            return {"value": executions[-1]}

        def close_stream(self, stream, context):
            pass

    def add_model():
        class Peer:
            def load(self):
                loading.set()
                assert release.wait(5)

        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(executor=Peer(), session=Peer())
            handle.load()
        return handle

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = StreamModel()
            handle = ModelRuntimeHandle(
                executor=model, streaming_runtime=model, state_scope="stream", stateful=True, resettable=True
            )
            handle.load()
        stream = handle.open_stream(ExecutionContext("open", Deadline.after(5)))
        with ThreadPoolExecutor(2) as workers:
            peer = workers.submit(add_model)
            assert loading.wait(3)
            result = workers.submit(
                handle.step, stream, ModelRequest({"value": 7}), ExecutionContext("step", Deadline.after(5))
            )
            try:
                time.sleep(0.15)
                assert not result.done() and not executions
            finally:
                release.set()
            peer.result(timeout=5)
            assert result.result(timeout=5).outputs == {"value": 7}
        assert executions == [7]
        assert handle.step(stream, ModelRequest({"value": 8}), ExecutionContext("next", Deadline.after(5))).outputs == {
            "value": 8
        }
    finally:
        release.set()
        providers.close()


@pytest.mark.parametrize("override,expected", [(None, 125.0), (3.0, 3.0)])
def test_process_initialization_budget_reaches_discovery(global_runtime, monkeypatch, override, expected):
    from rclpy.parameter import Parameter

    _, endpoint, _ = global_runtime
    node = Node(
        "budget_consumer",
        parameter_overrides=[
            Parameter("runtime_scheduler_endpoint", value=endpoint),
            Parameter("runtime_initialization_timeout", value=125.0),
        ],
    )
    environment = RuntimeEnvironment(scheduler_enabled=True)
    environment.configure_from_node(node)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    prepare = environment.prepare
    budgets = []

    def discover(validated, options, timeout, request_timeout_ms, **kwargs):
        budgets.append(timeout)
        return prepare(validated, options, timeout, request_timeout_ms, **kwargs)

    monkeypatch.setattr(environment, "prepare", discover)
    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=override):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda request, context: {"value": 1}))
            handle.load()
        assert handle.execute(ModelRequest(), ExecutionContext("budget")).outputs == {"value": 1}
        assert budgets == [expected]
    finally:
        providers.close()
        node.destroy_node()


@pytest.mark.parametrize("purpose", ["online", "simulation", "benchmark", "isolated_offline"])
@pytest.mark.parametrize("explicit_environment", [False, True])
def test_local_creation_rejects_closed_process_and_releases_closed_handles(purpose, explicit_environment):
    providers = RuntimeProviders.create(
        object(), object(), **({"runtime_environment": RuntimeEnvironment()} if explicit_environment else {})
    )
    environment = providers.runtime_environment
    for _ in range(3):
        with runtime_creation(purpose=purpose, providers=providers):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
            handle.load()
        handle.close()
        assert not environment._handles
    providers.close()
    with pytest.raises(RuntimeError, match="closed"), runtime_creation(purpose=purpose, providers=providers):
        pytest.fail("a closed process admitted a model constructor")


def test_creation_and_shutdown_share_ownership_without_serializing_constructors():
    environment = RuntimeEnvironment()
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    entered, finish, shutdown_started = threading.Event(), threading.Event(), threading.Event()
    errors, closed = [], []

    def constructor():
        try:
            with runtime_creation(providers=providers):
                entered.set()
                assert finish.wait(3)
                # This is the failed Factory's ownership transfer while shutdown
                # waits for its construction. It must not need shutdown's lock.
                providers.retain_cleanup(SimpleNamespace(close=lambda: closed.append("rollback")))
                handle = ModelRuntimeHandle(
                    SimpleNamespace(execute=lambda *_: {}, close=lambda: closed.append("model"))
                )
                handle.load()
        except BaseException as exc:
            errors.append(exc)

    def shutdown():
        shutdown_started.set()
        try:
            providers.close()
        except BaseException as exc:
            errors.append(exc)

    loading = threading.Thread(target=constructor, daemon=True)
    loading.start()
    assert entered.wait(1)
    try:
        # A different local model can finish construction and retirement while
        # the first constructor is blocked in native work.
        with runtime_creation(providers=providers, timeout=0.1):
            peer = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
            peer.load()
        peer.close(deadline=Deadline.after(0.1))
        stopping = threading.Thread(target=shutdown, daemon=True)
        stopping.start()
        assert shutdown_started.wait(1)
        finish.set()
        loading.join(3)
        stopping.join(3)
        assert not loading.is_alive() and not stopping.is_alive()
        assert not errors
        assert providers.closed and sorted(closed) == ["model", "rollback"]
    finally:
        finish.set()
        loading.join(3)


@pytest.mark.parametrize("configured,override", [(True, False), (False, True), (True, 1)])
def test_online_scheduler_override_cannot_change_process_policy(configured, override):
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(scheduler_enabled=configured)
    )
    with pytest.raises(ValueError), runtime_creation(providers=providers, scheduler_enabled=override):
        pytest.fail("conflicting online policy admitted a model")
    providers.close()


def test_targets_do_not_enable_a_disabled_process():
    providers = RuntimeProviders.create(object(), object(), runtime_environment=RuntimeEnvironment())
    with runtime_creation(providers=providers, targets=[{}]) as construction:
        assert construction.purpose.value == "local"
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"local": True}))
        handle.load()
    assert handle.execute(ModelRequest(), ExecutionContext("local")).outputs == {"local": True}
    providers.close()
    assert handle.closed


@pytest.mark.parametrize("configured", [None, False])
def test_explicit_targets_do_not_enable_a_disabled_host(global_runtime, configured):
    from rclpy.parameter import Parameter

    core, endpoint, _ = global_runtime
    node = Node(
        "disabled_explicit_host",
        parameter_overrides=[] if configured is None else [Parameter("runtime_scheduler_enabled", value=configured)],
    )
    try:
        from inference_service.host_lifecycle import close_node_runtimes

        closed = []
        with runtime_creation(node=node, targets=[target().to_dict()], endpoint=endpoint) as construction:
            assert construction.purpose.value == "local"
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}, close=lambda: closed.append(True)))
            handle.load()
        handle.execute(ModelRequest(), ExecutionContext("local"))
        assert not core.targets and not core._initializations
        with pytest.raises(ValueError, match="host configuration"), runtime_creation(node=node, scheduler_enabled=True):
            pytest.fail("explicit override bypassed disabled host configuration")
        close_node_runtimes(node)
        assert handle.closed and closed == [True]
        assert not node._runtime_constructions
    finally:
        node.destroy_node()


@pytest.mark.parametrize("blocked_lock", [False, True])
def test_close_budget_covers_initialization_settlement(blocked_lock):
    from inference_service.scheduler.runtime_initialization import InitializationSettlement
    from inference_service.scheduler.runtime_protocol import ControlError
    from inference_service.unified_runtime.lifecycle import lifecycle_remaining

    budgets = []
    with runtime_creation(purpose="simulation") as construction:
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()

    def replay(payload, until):
        budgets.append((until - time.monotonic(), lifecycle_remaining(60)))

    receipt = InitializationSettlement(None, {}, replay, lambda: None, 60)
    construction.settle_initialization = receipt.settle
    if blocked_lock:
        receipt._lock.acquire()
    try:
        started = time.monotonic()
        if blocked_lock:
            with pytest.raises(ControlError, match="INITIALIZATION_CONTROL_TIMEOUT"):
                handle.close(deadline=Deadline.after(0.01))
            assert not budgets and not handle.closed
        else:
            handle.close(deadline=Deadline.after(0.01))
            assert budgets and all(0 < budget <= 0.01 for budget in budgets[0])
        assert time.monotonic() - started < 0.2
    finally:
        if blocked_lock:
            receipt._lock.release()
        handle.close()


@pytest.mark.parametrize("failure", ["arguments", "constructor", "empty_session", "assembly_cleanup"])
def test_pre_handle_failure_settles_original_initialization(global_runtime, tmp_path, failure):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime
    from inference_service.scheduler.runtime_initialization import settle_initializations

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, _ = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    pending, attempts = [True], []

    def builder(context, *, required):
        raise ValueError("constructor rejected configuration")

    def cleanup():
        attempts.append("close")
        if pending[0]:
            raise RuntimeError("cleanup pending")

    options = {"builder_options": {"required": True}} if failure == "constructor" else {}
    if failure == "empty_session":

        def builder(context):
            return None

    if failure == "assembly_cleanup":
        # An invalid stream assembly fails only after a Session exists.
        def builder(context):
            return SimpleNamespace(capabilities=SimpleNamespace(stateful=False), close=cleanup)

        options["streaming_execution"] = True
    try:
        with pytest.raises((ValueError, TypeError)):
            load_model_runtime(
                validated_manifest=spec,
                providers=providers,
                registry_set=_registries(spec, builder),
                timeout=5,
                **options,
            )
        if failure == "assembly_cleanup":
            assert core._initializations and attempts == ["close"]
            pending[0] = False
            settle_initializations(environment._node)
            assert attempts == ["close", "close"]
        assert not core._initializations and not core.shutdown_pending()
        # A subsequent model on the same device must be admitted.
        handle = load_model_runtime(
            validated_manifest=spec,
            providers=providers,
            registry_set=_registries(spec),
            timeout=5,
            builder_options={"factory": object, "execute": lambda *_: 1},
        )
        handle.close(deadline=Deadline.after(5))
    finally:
        pending[0] = False
        providers.close()


@pytest.mark.parametrize("pending_cleanup", [False, True])
def test_caught_assembly_failure_preserves_cleanup_evidence(global_runtime, tmp_path, pending_cleanup):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime
    from inference_service.scheduler.runtime_initialization import settle_initializations
    from inference_service.unified_runtime import ModelRuntimeFactory
    from inference_service.unified_runtime.registry import RuntimeDependencyError

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, _ = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "caught"), "cpu")
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    pending, closes = [pending_cleanup], []

    def close():
        closes.append(True)
        if pending[0]:
            raise RuntimeError("cleanup pending")

    def builder(context):
        return SimpleNamespace(capabilities=SimpleNamespace(stateful=False), close=close)

    try:
        with (
            pytest.raises(RuntimeDependencyError, match="cleanup is still pending|must produce an owned"),
            runtime_creation(
                providers=providers, validated_manifest=spec, session_execution="stream", timeout=5
            ) as construction,
            pytest.raises(ValueError, match="streaming requires"),
        ):
            ModelRuntimeFactory.load_session(
                spec, providers=providers, registry_set=_registries(spec, builder), streaming_execution=True
            )
        assert bool(core._initializations) is pending_cleanup
        assert construction.rollback_clean is not pending_cleanup
        assert closes == [True]
        if pending_cleanup:
            pending[0] = False
            settle_initializations(environment._node)
        handle = load_model_runtime(
            validated_manifest=spec,
            providers=providers,
            registry_set=_registries(spec),
            timeout=5,
            builder_options={"factory": object, "execute": lambda *_: "ok"},
        )
        assert (
            handle.execute(ModelRequest(), ExecutionContext("after-cleanup", Deadline.after(3))).outputs["value"]
            == "ok"
        )
        handle.close(deadline=Deadline.after(5))
    finally:
        pending[0] = False
        providers.close()


def test_default_process_initialization_budget_reaches_session_loading(tmp_path):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime
    from inference_service.unified_runtime.lifecycle import lifecycle_remaining

    from .test_native_object_runtime import _create_bundle, _registries

    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    environment = RuntimeEnvironment()
    environment.initialization_timeout = 0.5
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    budgets = []

    def factory():
        budgets.append(lifecycle_remaining(60))
        return object()

    try:
        handle = load_model_runtime(
            validated_manifest=spec,
            providers=providers,
            registry_set=_registries(spec),
            builder_options={"factory": factory, "execute": lambda *_: 1},
        )
        assert 0 < budgets[0] <= 0.5
        handle.close()
    finally:
        providers.close()


@pytest.mark.parametrize("as_id", [False, True])
def test_stream_close_resumes_remote_activation_after_local_completion(global_runtime, monkeypatch, as_id):
    from inference_service.scheduler.runtime_protocol import ControlError
    from inference_service.unified_runtime import StreamHandle

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    closed = []

    class Model(StreamFixture):
        def open_stream(self, context):
            return StreamHandle("native")

        def close_stream(self, stream, context):
            closed.append(stream)

    try:
        model = Model()
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(
                executor=model, streaming_runtime=model, state_scope="stream", stateful=True, resettable=True
            )
            handle.load()
        stream = handle.open_stream(ExecutionContext("open", Deadline.after(5)))
        adapter = handle._execution_policy.adapter
        core.fence_resource("cpu")
        recover = adapter.controller.recover
        block = [True]

        def delayed_activation(epoch, operation_id, command, *args, **kwargs):
            if command == "activate" and block[0]:
                raise ControlError("RECOVERY_EVIDENCE_PENDING")
            return recover(epoch, operation_id, command, *args, **kwargs)

        monkeypatch.setattr(adapter.controller, "recover", delayed_activation)
        with pytest.raises(ExecutionFailure):
            handle.close_stream(stream, ExecutionContext("short-close", Deadline.after(0.2)))
        assert stream.closed and len(closed) == 1 and core._fenced
        block[0] = False
        handle.close_stream(stream.stream_id if as_id else stream, ExecutionContext("resume", Deadline.after(5)))
        assert not core._fenced and not adapter.has_recovery(("close_stream", stream.stream_id))
        assert len(closed) == 1
        handle.close(deadline=Deadline.after(5))
    finally:
        providers.close()


@pytest.mark.parametrize("phase", ["executor", "client", "thread"])
def test_partial_environment_start_releases_ros_resources_and_can_retry(global_runtime, monkeypatch, phase):
    import rclpy.executors
    import rclpy.node

    _core, endpoint, _ = global_runtime
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    created = []
    node_type = rclpy.node.Node

    class TrackedNode(node_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    def fail(*args, **kwargs):
        raise RuntimeError("transport startup failed")

    with monkeypatch.context() as patch:
        patch.setattr(rclpy.node, "Node", TrackedNode)
        if phase == "executor":
            patch.setattr(rclpy.executors.SingleThreadedExecutor, "add_node", fail)
        elif phase == "client":
            patch.setattr(node_type, "create_client", fail)
        else:
            start_thread = threading.Thread.start

            def fail_control_thread(thread):
                if thread.name == "model-runtime-control":
                    fail()
                return start_thread(thread)

            patch.setattr(threading.Thread, "start", fail_control_thread)
        with pytest.raises(RuntimeError, match="transport startup failed"):
            environment.prepare(manifest(), {}, 1, 5000)
    assert len(created) == 1 and not created[0].context.ok()
    try:
        node, selected = environment.prepare(manifest(), {}, 5, 5000)
        assert selected["physical_device"] == "local/cpu" and node.context.ok()
        assert environment._spin.is_alive()
    finally:
        environment.close()


def test_public_close_retries_after_worker_claim_expires_without_reopening(global_runtime, monkeypatch):
    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    closed, queued = [], []

    class Model:
        def execute(self, request, context):
            return {}

        def close(self):
            closed.append(True)

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            handle = ModelRuntimeHandle(Model())
            handle.load()
        controller = handle._execution_policy.controller
        with monkeypatch.context() as patch:
            patch.setattr(controller._workers, "submit", lambda fn, record: queued.append((fn, record)))
            with pytest.raises(ExecutionFailure):
                handle.close(deadline=Deadline.after(0.2))
            assert len(queued) == 1 and not closed
            run, record = queued.pop()
            run(record)
            assert record.state == "not_started"
            assert controller._gate == "closing" and not handle.ready
        handle.close(deadline=Deadline.after(5))
        assert handle.closed and closed == [True]
        assert not core.shutdown_pending() and not core._fenced
    finally:
        for run, record in queued:
            run(record)
        providers.close()


@pytest.mark.parametrize("phase", ["executor", "service", "client"])
def test_partial_initializer_creation_destroys_temporary_node(global_runtime, monkeypatch, phase):
    from rclpy._rclpy_pybind11 import InvalidHandle

    from inference_service.scheduler import runtime_initialization as initialization_module

    _core, endpoint, transport = global_runtime
    owner = transport.node
    created = []
    node_type = initialization_module.Node

    class TrackedNode(node_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    def fail(*args, **kwargs):
        raise RuntimeError("initializer startup failed")

    with monkeypatch.context() as patch:
        patch.setattr(initialization_module, "Node", TrackedNode)
        if phase == "executor":
            patch.setattr(initialization_module.SingleThreadedExecutor, "add_node", fail)
        elif phase == "service":
            patch.setattr(initialization_module, "_service_type", fail)
        else:
            patch.setattr(node_type, "create_client", fail)
        with (
            pytest.raises(RuntimeError, match="initializer startup failed"),
            initialization_module.runtime_initialization(owner, [target().to_dict()], endpoint, boot_id=str(uuid4())),
        ):
            pytest.fail("initializer setup failure reached the SDK")
    assert len(created) == 1
    with pytest.raises(InvalidHandle), created[0].handle:
        pytest.fail("temporary initializer node was not destroyed")
    assert owner.context.ok()


@pytest.mark.parametrize("operation", ["construct", "retire"])
def test_environment_membership_lock_obeys_caller_budget(operation):
    environment = RuntimeEnvironment()
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    with runtime_creation(providers=providers):
        handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
        handle.load()
    locked, release = threading.Event(), threading.Event()

    def hold_membership():
        with environment._lock:
            locked.set()
            assert release.wait(3)

    blocker = threading.Thread(target=hold_membership, daemon=True)
    blocker.start()
    assert locked.wait(1)
    try:
        start = time.monotonic()
        if operation == "construct":
            with pytest.raises(TimeoutError), runtime_creation(providers=providers, timeout=0.01):
                pytest.fail("constructor entered after its lock budget expired")
        else:
            with pytest.raises(Exception, match="busy|deadline"):
                handle.close(deadline=Deadline.after(0.01))
            assert handle.closed and handle in environment._handles
        assert time.monotonic() - start < 0.2
    finally:
        release.set()
        blocker.join(3)
        handle.close()
        assert not environment._handles
        providers.close()


@pytest.mark.parametrize("from_node", [False, True])
def test_default_creation_budget_includes_configuration_lock_wait(from_node):
    environment = RuntimeEnvironment(scheduler_enabled=from_node)
    environment.initialization_timeout = 60.0 if from_node else 0.02
    parameters = {"runtime_scheduler_enabled": True, "runtime_initialization_timeout": 0.02}
    node = (
        SimpleNamespace(
            has_parameter=lambda name: name in parameters,
            get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
            declare_parameter=lambda name, value, *args: SimpleNamespace(value=value),
        )
        if from_node
        else None
    )
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    locked, release = threading.Event(), threading.Event()

    def hold_membership():
        with environment._lock:
            locked.set()
            release.wait(0.4)

    blocker = threading.Thread(target=hold_membership, daemon=True)
    blocker.start()
    assert locked.wait(1)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError), runtime_creation(providers=providers, node=node):
            pytest.fail("constructor entered after default budget expired")
        assert time.monotonic() - started < 0.2
        assert environment._initializing == 0
    finally:
        release.set()
        blocker.join(1)
        providers.close()


def test_creation_exit_does_not_wait_for_transport_membership_lock():
    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    environment = RuntimeEnvironment()
    locked, release = threading.Event(), threading.Event()

    def hold_membership():
        with environment._lock:
            locked.set()
            release.wait(0.4)

    blocker = threading.Thread(target=hold_membership, daemon=True)
    try:
        with lifecycle_budget(0.02), environment.initialization():
            blocker.start()
            assert locked.wait(1)
            started = time.monotonic()
        assert time.monotonic() - started < 0.2
        assert environment._initializing == 0
    finally:
        release.set()
        blocker.join(1)
        environment.close()


@pytest.mark.parametrize("providers_present", [False, True])
@pytest.mark.parametrize("explicit_targets", [False, True])
def test_online_endpoint_override_cannot_change_configured_authority(monkeypatch, providers_present, explicit_targets):
    from inference_service.scheduler import runtime_initialization as initialization_module

    parameters = {"runtime_scheduler_enabled": True, "runtime_scheduler_endpoint": "/authority_a"}
    node = SimpleNamespace(
        has_parameter=lambda name: name in parameters,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        declare_parameter=lambda name, value, *args: SimpleNamespace(value=value),
    )
    environment = RuntimeEnvironment(endpoint="/authority_a", scheduler_enabled=True)
    providers = (
        RuntimeProviders.create(object(), object(), runtime_environment=environment) if providers_present else None
    )
    monkeypatch.setattr(environment, "prepare", lambda *args, **kwargs: pytest.fail("conflict reached discovery"))
    monkeypatch.setattr(
        initialization_module,
        "runtime_initialization",
        lambda *args, **kwargs: pytest.fail("conflict reached initialization"),
    )
    try:
        with (
            pytest.raises(ValueError, match="authority"),
            runtime_creation(
                providers=providers,
                node=node,
                endpoint="/authority_b",
                targets=[target().to_dict()] if explicit_targets else None,
                validated_manifest=None if explicit_targets else manifest(),
            ),
        ):
            pytest.fail("conflicting scheduler authority entered construction")
        assert not environment._handles and environment._initializing == 0
    finally:
        if providers is not None:
            providers.close()


@pytest.mark.parametrize("failure", ["factory_validation", "load"])
def test_explicit_host_failed_creation_retires_its_initialization(global_runtime, failure):
    from inference_service.host_lifecycle import close_node_runtimes
    from inference_service.unified_runtime import ModelRuntimeFactory, RuntimeFactoryError

    core, endpoint, transport = global_runtime
    selected = target("explicit_failure")
    node = transport.node
    with (
        pytest.raises(RuntimeFactoryError if failure == "factory_validation" else ExecutionFailure),
        runtime_creation(
            node=node,
            scheduler_enabled=True,
            targets=[selected.to_dict()],
            endpoint=endpoint,
            timeout=2,
        ),
    ):
        if failure == "factory_validation":
            ModelRuntimeFactory.create(object(), None, None)
        else:

            def fail_load():
                raise RuntimeError("load failed before binding")

            handle = ModelRuntimeHandle(SimpleNamespace(load=fail_load, execute=lambda *_: {}))
            handle.load()
    close_node_runtimes(node)
    assert not core._initializations and not core.shutdown_pending()
    assert selected.target_id in core._retired


@pytest.mark.parametrize(
    "global_runtime,accelerator",
    [
        ({"resources": {"accelerator": {"physical_device": "local/cuda:2"}}}, "cuda"),
        ({"resources": {"accelerator": {"physical_device": "local/npu:2"}}}, "npu"),
    ],
    indirect=["global_runtime"],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_managed_public_creation_propagates_explicit_torch_index(global_runtime, monkeypatch, accelerator, explicit):
    from dataclasses import replace

    from inference_manifest.runtime_identity import runtime_composition_fingerprint

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    core, endpoint, transport = global_runtime
    spec = manifest()
    spec.runtime_profile.profile = TorchRuntimeProfile(device=accelerator)
    options = {"torch_device_index": 2}
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    selected = replace(
        target("indexed"),
        hardware_resource_id="accelerator",
        physical_device=f"local/{accelerator}:2",
        deployment_fingerprint=spec.fingerprint,
        composition_fingerprint=runtime_composition_fingerprint("model_runtime", options),
    )
    supplied = {"node": transport.node, "targets": [selected.to_dict()], "endpoint": endpoint} if explicit else {}
    try:
        with runtime_creation(
            providers=providers, validated_manifest=spec, runtime_options=options, timeout=3, **supplied
        ):
            handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
            handle.load()
            # The adapter checks the actual loaded Session's index separately
            # from the configuration used to request initialization ownership.
            handle.assembly.load_context = SimpleNamespace(validated_manifest=spec, runtime_options=options)
            handle.assembly.session = SimpleNamespace(runtime_device_index=2)
        assert handle._execution_policy.target.physical_device == f"local/{accelerator}:2"
        handle.execute(ModelRequest(), ExecutionContext("indexed", Deadline.after(2)))
        handle.close(deadline=Deadline.after(3))
        assert not core.shutdown_pending()
    finally:
        providers.close()


@pytest.mark.parametrize("invalid", ["missing_registry", "unknown_option"])
def test_public_loader_rejects_argument_binding_before_device_reservation(global_runtime, tmp_path, invalid):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, _ = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    environment = RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    providers = RuntimeProviders.create(object(), object(), runtime_environment=environment)
    calls = []
    options = {
        "validated_manifest": spec,
        "providers": providers,
        "timeout": 5,
        "registry_set": _registries(spec),
        "builder_options": {"factory": lambda: calls.append("constructed") or object(), "execute": lambda *_: "ok"},
    }
    bad = dict(options)
    if invalid == "missing_registry":
        del bad["registry_set"]
    else:
        bad["unknown_assembly_option"] = True
    try:
        with pytest.raises(TypeError):
            load_model_runtime(**bad)
        assert not calls and not core._initializations and not environment._handles
        assert environment._node is None
        handle = load_model_runtime(**options)
        assert calls == ["constructed"]
        assert handle.execute(ModelRequest(), ExecutionContext("valid", Deadline.after(3))).outputs["value"] == "ok"
        handle.close(deadline=Deadline.after(5))
    finally:
        providers.close()


@pytest.mark.parametrize("invalid", ["unknown_option", "timeout"])
def test_explicit_factory_validation_settles_initialization_before_retry(global_runtime, tmp_path, invalid):
    from inference_manifest import load_inference_manifest
    from inference_service.runtime_creation import load_model_runtime
    from inference_service.unified_runtime.factory import ModelRuntimeFactory

    from .test_native_object_runtime import _create_bundle, _registries

    core, endpoint, _ = global_runtime
    spec = load_inference_manifest(_create_bundle(tmp_path / "domain"), "cpu")
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    options = {
        "registry_set": _registries(spec),
        "providers": providers,
        "builder_options": {"factory": object, "execute": lambda *_: "ok"},
    }
    bad = {"unknown_assembly_option": True} if invalid == "unknown_option" else {"timeout": "invalid"}
    try:
        with (
            pytest.raises((TypeError, ValueError)),
            runtime_creation(providers=providers, validated_manifest=spec, timeout=5) as construction,
        ):
            ModelRuntimeFactory.load_session(spec, **options, **bad)
        assert construction.rollback_clean
        assert not core._initializations and not core.shutdown_pending()
        handle = load_model_runtime(validated_manifest=spec, timeout=5, **options)
        assert handle.execute(ModelRequest(), ExecutionContext("valid", Deadline.after(3))).outputs["value"] == "ok"
        handle.close(deadline=Deadline.after(5))
    finally:
        providers.close()


def test_explicit_host_reclaims_closed_models_at_spin_boundary(global_runtime):
    import gc
    import weakref
    from dataclasses import replace

    from inference_service.host_lifecycle import close_node_runtimes, collect_node_runtime_transport

    _, endpoint, _ = global_runtime
    node = Node("explicit_retirement")
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    stopping = threading.Event()
    unrelated_entered, unrelated_release = threading.Event(), threading.Event()

    def unrelated_callback():
        timer.cancel()
        unrelated_entered.set()
        assert unrelated_release.wait(30)

    timer = node.create_timer(0.01, unrelated_callback)
    errors = []

    def spin():
        try:
            while not stopping.is_set():
                executor.spin_once(timeout_sec=0.02)
                collect_node_runtime_transport(node, executor)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    assert unrelated_entered.wait(3)
    baseline = tuple(len(tuple(getattr(node, field))) for field in ("services", "clients", "timers", "guards"))
    refs = []
    try:
        for _index in range(6):
            selected = replace(target("explicit"), endpoint="/runtime/explicit_reused")
            with runtime_creation(
                node=node, targets=[selected.to_dict()], endpoint=endpoint, timeout=5, scheduler_enabled=True
            ):
                handle = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {}))
                handle.load()
            refs.append(weakref.ref(handle))
            handle._execution_policy.adapter.wait_for_registration(ExecutionContext("registration", Deadline.after(5)))
            handle.execute(ModelRequest(), ExecutionContext("explicit", Deadline.after(5)))
            handle.close(deadline=Deadline.after(5))
            assert not node._runtime_constructions
            expires = time.monotonic() + 3
            while (
                tuple(len(tuple(getattr(node, field))) for field in ("services", "clients", "timers", "guards"))
                != baseline
            ):
                assert time.monotonic() < expires and not errors
                time.sleep(0.01)
            del handle
        expires = time.monotonic() + 3
        while any(ref() is not None for ref in refs):
            gc.collect()
            assert time.monotonic() < expires, "executor retained a closed model"
            time.sleep(0.01)
        assert not errors
    finally:
        unrelated_release.set()
        stopping.set()
        executor.wake()
        worker.join(5)
        executor.remove_node(node)
        executor.shutdown(timeout_sec=5)
        close_node_runtimes(node)
        node.destroy_node()


def test_managed_open_delivery_racing_close_never_cleans_under_peer_ownership(global_runtime, monkeypatch):
    from inference_service.unified_runtime import LifecycleState, StreamHandle

    _, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    delivered, release_delivery, busy, release_busy, stream_closed = [threading.Event() for _ in range(5)]
    overlaps = []

    class StreamModel(StreamFixture):
        def open_stream(self, context):
            return StreamHandle("native")

        def close_stream(self, stream, context):
            overlaps.append(busy.is_set() and not release_busy.is_set())
            stream_closed.set()

    class BusyModel:
        def execute(self, request, context):
            busy.set()
            assert release_busy.wait(5)
            return {}

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = StreamModel()
            streaming = ModelRuntimeHandle(model, streaming_runtime=model, state_scope="stream", stateful=True)
            streaming.load()
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            peer = ModelRuntimeHandle(BusyModel())
            peer.load()
        adapter = streaming._execution_policy.adapter
        execute = adapter.execute

        def delay(request, context, **kwargs):
            result = execute(request, context, **kwargs)
            if request.metadata.get("runtime_handle_operation") == "open_stream":
                delivered.set()
                assert release_delivery.wait(5)
            return result

        monkeypatch.setattr(adapter, "execute", delay)
        with ThreadPoolExecutor(3) as workers:
            opening = workers.submit(streaming.open_stream, ExecutionContext("open", Deadline.after(5)))
            assert delivered.wait(3)
            running = workers.submit(peer.execute, ModelRequest(), ExecutionContext("busy", Deadline.after(5)))
            assert busy.wait(3)
            closing = workers.submit(streaming.close, ExecutionContext("close", Deadline.after(5)))
            expires = time.monotonic() + 2
            while streaming.state is not LifecycleState.CLOSING:
                assert time.monotonic() < expires
                time.sleep(0.001)
            release_delivery.set()
            with pytest.raises(ExecutionFailure):
                opening.result(timeout=2)
            assert not stream_closed.is_set(), "delivery compensation bypassed Global ownership"
            release_busy.set()
            running.result(timeout=3)
            closing.result(timeout=3)
        assert overlaps == [False]
    finally:
        release_delivery.set()
        release_busy.set()
        providers.close()


@pytest.mark.parametrize("expired_delivery", [False, True])
def test_pending_stream_close_retry_does_not_close_open_awaiting_delivery(
    global_runtime, monkeypatch, expired_delivery
):
    from inference_service.unified_runtime import StreamHandle

    _, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    opened, release_delivery = threading.Event(), threading.Event()
    close_started, release_close = threading.Event(), threading.Event()
    closed = []

    class Model(StreamFixture):
        def open_stream(self, context):
            return StreamHandle(str(uuid4()))

        def close_stream(self, stream, context):
            closed.append(stream.stream_id)
            if expired_delivery and len(closed) == 2:
                close_started.set()
                assert release_close.wait(5)

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = Model()
            handle = ModelRuntimeHandle(
                model,
                streaming_runtime=model,
                state_scope="stream",
                stateful=True,
                state_bank_mode="per_stream",
                max_open_streams=2,
            )
            handle.load()
        first = handle.open_stream(ExecutionContext("first", Deadline.after(3)))
        # A real pending Close coexists with the new Open's delivery window.
        with pytest.raises(ExecutionFailure):
            handle.close_stream(first, ExecutionContext("expired-close", Deadline.after(0)))
        adapter = handle._execution_policy.adapter
        execute = adapter.execute

        def delay(request, context, **kwargs):
            result = execute(request, context, **kwargs)
            if request.metadata.get("runtime_handle_operation") == "open_stream":
                opened.set()
                assert release_delivery.wait(5)
            return result

        monkeypatch.setattr(adapter, "execute", delay)
        if expired_delivery:
            handle._execution_policy._open_stream_timeout = 0.3
        with ThreadPoolExecutor(2) as pool:
            # An unbounded caller still has the binding's bounded Open window.
            opening = pool.submit(handle.open_stream, ExecutionContext("new-open"))
            try:
                assert opened.wait(3)
                if expired_delivery:
                    time.sleep(0.31)
                    retry = pool.submit(
                        handle.retry_pending_stream_closes, ExecutionContext("retry-expired-open", Deadline.after(3))
                    )
                    assert close_started.wait(2)
                    release_delivery.set()
                    # Once compensation owns the stream, late delivery must
                    # not hand its still-open-but-closing handle to the caller.
                    with pytest.raises(ExecutionFailure):
                        opening.result(timeout=2)
                    release_close.set()
                    retry.result(timeout=2)
                    assert handle.open_stream_count == 0 and len(closed) == 2
                else:
                    handle.retry_pending_stream_closes(ExecutionContext("retry-old-close", Deadline.after(3)))
                    assert first.closed and handle.open_stream_count == 1 and len(closed) == 1
                    release_delivery.set()
                    second = opening.result(timeout=2)
                    assert not second.closed
                    assert (
                        handle.step(second, ModelRequest(), ExecutionContext("step", Deadline.after(3))).outputs == {}
                    )
            finally:
                release_delivery.set()
                release_close.set()
    finally:
        release_delivery.set()
        release_close.set()
        providers.close()


def test_failed_stream_delivery_retains_pending_close_and_releases_capacity(global_runtime, monkeypatch):
    from concurrent.futures import Future

    from inference_service.scheduler.runtime_protocol import decode_control
    from inference_service.unified_runtime import StreamHandle

    core, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )
    closed = []

    class Model(StreamFixture):
        def open_stream(self, context):
            return StreamHandle("native")

        def close_stream(self, stream, context):
            closed.append(stream)

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = Model()
            handle = ModelRuntimeHandle(model, streaming_runtime=model, state_scope="stream", stateful=True)
            handle.load()
        adapter = handle._execution_policy.adapter
        call = adapter._client.call_async

        def lose_terminal(request):
            if decode_control(request.payload)["verb"] == "terminal_report":
                return Future()
            return call(request)

        with monkeypatch.context() as patch:
            patch.setattr(adapter._client, "call_async", lose_terminal)
            patch.setattr(core, "pending_reports", lambda: [])
            with pytest.raises(ExecutionFailure):
                handle.open_stream(ExecutionContext("lost-open", Deadline.after(0.1)))
        assert handle.open_stream_count == 1
        handle.retry_pending_stream_closes(ExecutionContext("compensate", Deadline.after(5)))
        assert handle.open_stream_count == 0 and len(closed) == 1
        stream = handle.open_stream(ExecutionContext("next-open", Deadline.after(3)))
        handle.close_stream(stream, ExecutionContext("next-close", Deadline.after(3)))
        assert len(closed) == 2
    finally:
        providers.close()


def test_managed_closed_stream_object_survives_history_eviction(global_runtime):
    from inference_service.unified_runtime import StreamHandle

    _, endpoint, _ = global_runtime
    providers = RuntimeProviders.create(
        object(), object(), runtime_environment=RuntimeEnvironment(endpoint=endpoint, scheduler_enabled=True)
    )

    class Model(StreamFixture):
        def open_stream(self, context):
            return StreamHandle("native")

    try:
        with runtime_creation(providers=providers, validated_manifest=manifest(), timeout=5):
            model = Model()
            handle = ModelRuntimeHandle(model, streaming_runtime=model, state_scope="stream", stateful=True)
            handle.load()
        streams = []
        for index in range(130):
            context = ExecutionContext(f"stream-{index}", Deadline.after(3))
            stream = handle.open_stream(context)
            handle.close_stream(stream, context)
            streams.append(stream)
        assert len(handle._streams) == 128
        handle.close_stream(streams[0], ExecutionContext("replay", Deadline.after(3)))
        with pytest.raises(ExecutionFailure, match="not found"):
            handle.close_stream(streams[0].stream_id, ExecutionContext("string", Deadline.after(3)))
    finally:
        providers.close()
