from __future__ import annotations

import importlib
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendAdmissionError,
    BackendAdmissionEvidence,
    BackendCancellationError,
    BackendCapabilities,
    BackendCapabilityError,
    BackendCompatibilityError,
    BackendDescriptor,
    BackendInferenceError,
    BackendNotReadyError,
    BackendPriorityMapping,
    BackendRegistry,
    BackendRegistryError,
    BackendResult,
    BackendState,
    ConformanceEvidence,
    InferenceRequest,
    LifecycleBackend,
    PartialLoadRollback,
    ResourceDomainAdmissions,
    RuntimeContext,
)
from tests.manifest_fixtures import (
    create_non_policy_bundle,
    create_policy_bundle,
    make_manifest,
    make_non_policy_manifest,
    write_manifest,
)

_MULTI_INSTANCE_EVIDENCE = BackendAdmissionEvidence(
    sdk_initialization=True,
    multi_instance_execution=True,
    failure_isolation=True,
    independent_close=True,
)


def test_backend_priority_mapping_is_explicit_and_backend_owned() -> None:
    mapping = BackendPriorityMapping((7, 3, 0))

    assert mapping.generic_level_count == 3
    assert mapping.map_generic(0) == 7
    assert mapping.map_generic(2) == 0
    with pytest.raises(ValueError, match="generic priority"):
        mapping.map_generic(3)


def test_backend_hardware_identity_is_absent_from_legacy_capabilities() -> None:
    capabilities = BackendCapabilities()

    assert capabilities.hardware_resource_id is None
    assert capabilities.priority_mapping is None


def test_backend_capabilities_preserves_legacy_positional_field_order() -> None:
    capabilities = BackendCapabilities(False, False, False, 1, False, "npu", 1, True, True, None)

    assert capabilities.resource_domain == "npu"
    assert capabilities.max_in_flight_per_resource_domain == 1
    assert capabilities.supports_attention
    assert capabilities.supports_cancellation
    assert capabilities.hardware_resource_id is None
    assert capabilities.priority_mapping is None


class FakeBackend(LifecycleBackend):
    def __init__(
        self,
        name: str = "fake",
        capabilities: BackendCapabilities | None = None,
        *,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        super().__init__(name, capabilities or BackendCapabilities(), domains=domains)
        self.load_started = threading.Event()
        self.load_release = threading.Event()
        self.load_release.set()
        self.infer_release = threading.Event()
        self.infer_release.set()
        self.recovery_started = threading.Event()
        self.recovery_release = threading.Event()
        self.recovery_release.set()
        self.cancel_started = threading.Event()
        self.cancel_release = threading.Event()
        self.cancel_release.set()
        self.cancel_errors: list[Exception] = []
        self.fail_load = False
        self.fail_recovery = False
        self.infer_errors: list[Exception] = []
        self.resources: list[str] = []
        self.rollback_order: list[str] = []
        self.reset_calls = 0
        self.close_calls = 0
        self.recovery_calls = 0
        self._metrics = threading.Condition()
        self._active_infer = 0
        self.max_active_infer = 0
        self.infer_entries = 0

    def wait_for_infer_entries(self, count: int, timeout: float = 2.0) -> bool:
        with self._metrics:
            return self._metrics.wait_for(lambda: self.infer_entries >= count, timeout=timeout)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        self.load_started.set()
        if not self.load_release.wait(timeout=2):
            raise RuntimeError("timed out waiting to continue fake load")
        self._allocate("first", rollback)
        self._allocate("second", rollback)
        if self.fail_load:
            raise RuntimeError("partial load failure")

    def _allocate(self, resource: str, rollback: PartialLoadRollback) -> None:
        self.resources.append(resource)
        rollback.defer(self._rollback_resource, resource)

    def _rollback_resource(self, resource: str) -> None:
        self.resources.remove(resource)
        self.rollback_order.append(resource)

    def _infer(self, request: InferenceRequest) -> BackendResult:
        with self._metrics:
            self._active_infer += 1
            self.infer_entries += 1
            self.max_active_infer = max(self.max_active_infer, self._active_infer)
            self._metrics.notify_all()
        try:
            if not self.infer_release.wait(timeout=2):
                raise RuntimeError("timed out waiting to continue fake inference")
            if self.infer_errors:
                raise self.infer_errors.pop(0)
            return BackendResult(
                action=[request.inputs["value"]],
                actual_chunk_size=1,
                backend_latency_ms=0.25,
                metadata={"request_id": request.request_id},
            )
        finally:
            with self._metrics:
                self._active_infer -= 1
                self._metrics.notify_all()

    def _reset(self) -> None:
        self.reset_calls += 1

    def _recover(self) -> None:
        self.recovery_calls += 1
        self.recovery_started.set()
        if not self.recovery_release.wait(timeout=2):
            raise RuntimeError("timed out waiting to continue fake recovery")
        if self.fail_recovery:
            raise RuntimeError("fake recovery failure")

    def _cancel(self, request_id: str) -> None:
        self.cancel_started.set()
        if not self.cancel_release.wait(timeout=2):
            raise RuntimeError("timed out waiting to continue fake cancellation")
        if self.cancel_errors:
            raise self.cancel_errors.pop(0)

    def _close(self) -> None:
        self.close_calls += 1
        self.resources.clear()


def _make_context(
    root: Path,
    *,
    policy_type: str = "act",
    backend: str = "torch",
    target_soc: str | None = None,
    target_runtime: str | None = None,
    artifact_format: str | None = None,
) -> RuntimeContext:
    root.mkdir()
    compiled = backend != "torch"
    bundle_paths = create_policy_bundle(root, policy_type=policy_type, include_weights=not compiled)
    manifest = make_manifest(
        root,
        bundle_paths,
        deployment_name=backend,
        compiled=compiled,
        backend=backend,
        policy_type=policy_type,
    )
    if compiled:
        defaults = {
            "ascend": ("ascend310", "acl", "om"),
            "hisilicon": ("sd3403", "hisilicon-worker", "om"),
            "rknn": ("rk3588", "rknn-lite", "rknn"),
            "hmm": ("lq50", "tcim", "hmm"),
        }
        soc, runtime, file_format = defaults[backend]
        deployment = manifest["deployments"][backend]
        deployment["target"] = {
            "soc": target_soc or soc,
            "runtime": target_runtime or runtime,
        }
        deployment["artifacts"]["policy"]["format"] = artifact_format or file_format
        if backend == "hisilicon":
            worker = root / "artifacts" / "worker"
            worker.write_bytes(b"worker")
            worker.chmod(0o755)
            deployment["artifacts"]["worker"] = {
                "path": "artifacts/worker",
                "format": "executable",
            }
            if len(deployment["bindings"]["policy"]["outputs"]) == 1:
                deployment["bindings"]["policy"]["outputs"][0]["index"] = 1
    write_manifest(root, manifest)
    return RuntimeContext(
        validated_manifest=load_inference_manifest(root, backend),
        runtime_options={"device_id": 0},
    )


def _request(request_id: str = "request-1") -> InferenceRequest:
    return InferenceRequest(request_id=request_id, inputs={"value": 1.0}, metadata={"source": "test"})


def _make_non_policy_context(root: Path, *, family: str = "ram_plus") -> RuntimeContext:
    root.mkdir()
    bundle_paths = create_non_policy_bundle(root)
    write_manifest(root, make_non_policy_manifest(root, bundle_paths, family=family))
    return RuntimeContext(load_inference_manifest(root, "ascend"))


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


def _close_all(*backends: FakeBackend) -> None:
    for backend in backends:
        backend.close()


def test_generic_request_priority_is_not_limited_to_ascend_range():
    request = InferenceRequest(request_id="generic-priority", inputs={"value": 1.0}, priority=1024)

    assert request.priority == 1024


def test_inference_request_preserves_legacy_positional_metadata_field() -> None:
    metadata = {"source": "legacy-positional"}

    request = InferenceRequest("legacy-request", {"value": 1.0}, None, None, metadata)

    assert request.metadata == metadata
    assert request.priority == 0


def test_load_returns_only_after_ready_and_infer_updates_health(tmp_path):
    context = _make_context(tmp_path / "bundle")
    domains = ResourceDomainAdmissions()
    backend = FakeBackend(domains=domains)
    backend.load_release.clear()

    assert backend.health().state is BackendState.CREATED
    with pytest.raises(BackendNotReadyError, match="state is created"):
        backend.infer(_request())

    thread, _, errors = _thread_call(lambda: backend.load(context))
    assert backend.load_started.wait(timeout=2)
    assert backend.health().state is BackendState.LOADING
    assert thread.is_alive()

    backend.load_release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []
    health = backend.health()
    assert health.state is BackendState.READY
    assert health.ready is True

    result = backend.infer(_request())
    assert result.action == [1.0]
    assert result.actual_chunk_size == 1
    assert result.metadata == {"request_id": "request-1"}
    assert backend.health().last_successful_inference_time is not None
    backend.close()


def test_runtime_failure_leaves_ready_and_rejects_later_requests(tmp_path):
    backend = FakeBackend(domains=ResourceDomainAdmissions())
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_errors.append(BackendInferenceError("device lost", code="device_lost"))

    with pytest.raises(BackendInferenceError, match="device lost"):
        backend.infer(_request())

    health = backend.health()
    assert health.state is BackendState.FAILED
    assert health.ready is False
    assert health.reason_code == "device_lost"
    assert health.message == "device lost"
    assert health.failure_count == 1
    with pytest.raises(BackendNotReadyError, match="state is failed"):
        backend.infer(_request("request-2"))
    backend.close()


def test_recoverable_runtime_failure_exposes_recovering_then_ready(tmp_path):
    backend = FakeBackend(domains=ResourceDomainAdmissions())
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_errors.append(BackendInferenceError("worker exited", code="worker_exited", recoverable=True))

    with pytest.raises(BackendInferenceError, match="worker exited"):
        backend.infer(_request())
    degraded = backend.health()
    assert degraded.state is BackendState.DEGRADED
    assert degraded.ready is False
    assert degraded.reason_code == "worker_exited"

    backend.recovery_release.clear()
    thread, _, errors = _thread_call(backend.recover)
    assert backend.recovery_started.wait(timeout=2)
    recovering = backend.health()
    assert recovering.state is BackendState.RECOVERING
    assert recovering.ready is False
    assert recovering.reason_code == "worker_exited"

    backend.recovery_release.set()
    thread.join(timeout=2)
    assert errors == []
    recovered = backend.health()
    assert recovered.state is BackendState.READY
    assert recovered.reason_code == "recovered"
    assert recovered.failure_count == 1
    assert backend.recovery_calls == 1
    backend.close()


def test_close_remains_terminal_when_recovery_fails_concurrently(tmp_path):
    backend = FakeBackend(domains=ResourceDomainAdmissions())
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_errors.append(BackendInferenceError("worker exited", code="worker_exited", recoverable=True))

    with pytest.raises(BackendInferenceError, match="worker exited"):
        backend.infer(_request())
    assert backend.health().state is BackendState.DEGRADED

    backend.fail_recovery = True
    backend.recovery_release.clear()

    recovery_thread, _, recovery_errors = _thread_call(backend.recover)
    assert backend.recovery_started.wait(timeout=2)
    close_thread, _, close_errors = _thread_call(backend.close)
    assert backend.health().state is BackendState.CLOSING

    backend.recovery_release.set()
    recovery_thread.join(timeout=2)
    close_thread.join(timeout=2)
    assert len(recovery_errors) == 1
    assert isinstance(recovery_errors[0], BackendInferenceError)
    assert close_errors == []
    assert backend.close_calls == 1
    assert backend.health().state is BackendState.CLOSED


def test_reset_is_capability_gated_and_supported_reset_runs(tmp_path):
    context = _make_context(tmp_path / "bundle")
    unsupported = FakeBackend(name="unsupported-reset", domains=ResourceDomainAdmissions())
    unsupported.load(context)
    with pytest.raises(BackendCapabilityError) as error:
        unsupported.reset()
    assert error.value.capability == "reset"
    unsupported.close()

    supported = FakeBackend(
        name="supported-reset",
        capabilities=BackendCapabilities(resettable=True, stateful=True),
        domains=ResourceDomainAdmissions(),
    )
    supported.load(context)
    supported.reset()
    assert supported.reset_calls == 1
    supported.close()


def test_cancel_can_overlap_inference_but_serializes_with_reset(tmp_path):
    class BlockingControlBackend(FakeBackend):
        def __init__(self):
            super().__init__(
                capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
                domains=ResourceDomainAdmissions(),
            )
            self.reset_started = threading.Event()

        def _reset(self) -> None:
            self.reset_started.set()
            super()._reset()

    backend = BlockingControlBackend()
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_release.clear()
    backend.cancel_release.clear()

    infer_thread, _, infer_errors = _thread_call(lambda: backend.infer(_request("target")))
    assert backend.wait_for_infer_entries(1)
    cancel_thread, _, cancel_errors = _thread_call(lambda: backend.cancel("target"))
    assert backend.cancel_started.wait(timeout=2)
    reset_thread, _, reset_errors = _thread_call(backend.reset)
    time.sleep(0.05)
    assert not backend.reset_started.is_set()

    backend.cancel_release.set()
    cancel_thread.join(timeout=2)
    backend.infer_release.set()
    infer_thread.join(timeout=2)
    assert backend.reset_started.wait(timeout=2)
    reset_thread.join(timeout=2)

    assert infer_errors == []
    assert cancel_errors == []
    assert reset_errors == []
    backend.close()


def test_close_waits_for_active_cancel_before_releasing_resources(tmp_path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
        domains=ResourceDomainAdmissions(),
    )
    backend.load(_make_context(tmp_path / "bundle"))
    backend.cancel_release.clear()

    cancel_thread, _, cancel_errors = _thread_call(lambda: backend.cancel("target"))
    assert backend.cancel_started.wait(timeout=2)
    close_thread, _, close_errors = _thread_call(backend.close)
    time.sleep(0.05)

    assert close_thread.is_alive()
    assert backend.close_calls == 0
    backend.cancel_release.set()
    cancel_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert cancel_errors == []
    assert close_errors == []
    assert backend.close_calls == 1
    assert backend.health().state is BackendState.CLOSED


def test_cancel_waiting_for_reset_respects_deadline_without_starting(tmp_path):
    class BlockingResetBackend(FakeBackend):
        def __init__(self):
            super().__init__(
                capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
                domains=ResourceDomainAdmissions(),
            )
            self.reset_started = threading.Event()
            self.reset_release = threading.Event()

        def _reset(self) -> None:
            self.reset_started.set()
            if not self.reset_release.wait(timeout=2):
                raise RuntimeError("reset release timed out")

    backend = BlockingResetBackend()
    backend.load(_make_context(tmp_path / "bundle"))
    reset_thread, _, reset_errors = _thread_call(backend.reset)
    assert backend.reset_started.wait(timeout=2)

    with pytest.raises(BackendAdmissionError, match="waiting for control admission") as error:
        backend.cancel("target", deadline=datetime.now(timezone.utc) + timedelta(milliseconds=20))

    assert error.value.operation_started is False
    assert not backend.cancel_started.is_set()
    assert backend.health().state is BackendState.READY
    backend.reset_release.set()
    reset_thread.join(timeout=2)
    assert reset_errors == []
    backend.close()


def test_started_cancel_admission_error_fails_stateful_backend_closed(tmp_path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(resettable=True, stateful=True, supports_cancellation=True),
        domains=ResourceDomainAdmissions(),
    )
    backend.load(_make_context(tmp_path / "bundle"))
    backend.cancel_errors.append(
        BackendAdmissionError(
            "cancel acknowledgement timed out",
            code="deadline_exceeded",
            operation_started=True,
        )
    )

    with pytest.raises(BackendCancellationError, match="cancel acknowledgement timed out") as error:
        backend.cancel("target")

    assert error.value.code == "deadline_exceeded"
    assert error.value.operation_started is True
    assert error.value.outcome_known is False
    assert backend.health().state is BackendState.FAILED
    backend.close()


def test_partial_load_rolls_back_in_reverse_order_and_close_is_idempotent(tmp_path):
    backend = FakeBackend(domains=ResourceDomainAdmissions())
    backend.fail_load = True

    with pytest.raises(Exception, match="partial load failure"):
        backend.load(_make_context(tmp_path / "bundle"))

    assert backend.resources == []
    assert backend.rollback_order == ["second", "first"]
    assert backend.health().state is BackendState.FAILED
    backend.close()
    backend.close()
    assert backend.close_calls == 1
    assert backend.health().state is BackendState.CLOSED


def test_registry_validation_is_lazy_and_fake_factory_is_loaded_only_on_create(monkeypatch, tmp_path):
    context = _make_context(tmp_path / "bundle")
    attempted_imports: list[str] = []
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None):
        attempted_imports.append(name)
        if name.split(".", maxsplit=1)[0] in {"acl", "rknn", "rknnlite", "tcim", "torch_npu"}:
            raise AssertionError(f"optional SDK import attempted: {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    descriptor = BACKEND_REGISTRY.validate(context)
    assert descriptor.name == "torch"
    assert attempted_imports == []

    factory_module = ModuleType("tests.fake_backend_factory")
    created: list[FakeBackend] = []

    def create_backend(_context: RuntimeContext) -> FakeBackend:
        backend = FakeBackend(name="torch", domains=ResourceDomainAdmissions())
        created.append(backend)
        return backend

    factory_module.create_backend = create_backend
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    registry = BackendRegistry(
        {
            "torch": BackendDescriptor(
                name="torch",
                factory="tests.fake_backend_factory:create_backend",
                supported_policy_families=frozenset({"act"}),
                conformance_evidence=frozenset({ConformanceEvidence("policy", "act")}),
                target_validator=lambda deployment: None,
            )
        }
    )

    backend = registry.create(context)
    assert attempted_imports == ["tests.fake_backend_factory"]
    assert backend is created[0]
    backend.close()


def test_registry_does_not_instantiate_unavailable_backend_during_validation(tmp_path):
    context = _make_context(tmp_path / "bundle", backend="rknn")
    registry = BackendRegistry(
        {
            "rknn": BackendDescriptor(
                name="rknn",
                factory="tests.unavailable_backend_factory:create_backend",
                supported_policy_families=frozenset({"act"}),
                conformance_evidence=frozenset({ConformanceEvidence("policy", "act")}),
                target_validator=lambda deployment: None,
            )
        }
    )

    assert registry.validate(context).name == "rknn"
    with pytest.raises(BackendRegistryError, match="factory module.*unavailable") as error:
        registry.create(context)
    assert error.value.code == "factory_unavailable"


@pytest.mark.parametrize(
    ("policy_type", "backend", "supported"),
    [
        ("act", "torch", True),
        ("act", "ascend", True),
        ("act", "hisilicon", True),
        ("act", "rknn", True),
        ("act", "hmm", False),
        ("diffusion", "torch", True),
        ("diffusion", "ascend", False),
        ("diffusion", "hisilicon", False),
        ("diffusion", "rknn", False),
        ("diffusion", "hmm", False),
        ("pi05", "torch", True),
        ("pi05", "ascend", True),
        ("pi05", "hisilicon", False),
        ("pi05", "rknn", False),
        ("pi05", "hmm", True),
        ("smolvla", "torch", True),
        ("smolvla", "ascend", False),
        ("smolvla", "hisilicon", False),
        ("smolvla", "rknn", True),
        ("smolvla", "hmm", True),
    ],
)
def test_registry_enforces_exact_policy_support_matrix(tmp_path, policy_type, backend, supported):
    context = _make_context(tmp_path / "bundle", policy_type=policy_type, backend=backend)

    if supported:
        assert BACKEND_REGISTRY.validate(context).name == backend
    else:
        with pytest.raises(BackendCompatibilityError, match="does not support") as error:
            BACKEND_REGISTRY.validate(context)
        assert error.value.code == "unsupported_policy_backend_pair"


def test_registry_accepts_non_policy_model_family_support(tmp_path):
    context = _make_non_policy_context(tmp_path / "bundle")
    registry = BackendRegistry(
        {
            "ascend": BackendDescriptor(
                name="ascend",
                factory="tests.fake_backend_factory:create_backend",
                target_validator=lambda deployment: None,
                supported_model_families=frozenset({"ram_plus"}),
                conformance_evidence=frozenset({ConformanceEvidence("perception", "ram_plus")}),
            )
        }
    )

    assert registry.validate(context).name == "ascend"
    assert context.model.family == "ram_plus"
    with pytest.raises(ValueError, match=r"RuntimeContext\.policy is unavailable.*ram_plus"):
        _ = context.policy


def test_registry_accepts_non_policy_model_kind_support(tmp_path):
    context = _make_non_policy_context(tmp_path / "bundle")
    registry = BackendRegistry(
        {
            "ascend": BackendDescriptor(
                name="ascend",
                factory="tests.fake_backend_factory:create_backend",
                target_validator=lambda deployment: None,
                supported_model_kinds=frozenset({"perception"}),
                supported_model_families=frozenset({"ram_plus"}),
                conformance_evidence=frozenset({ConformanceEvidence("perception", "ram_plus")}),
            )
        }
    )

    assert registry.validate(context).name == "ascend"


def test_registry_accepts_declared_non_policy_model_support(tmp_path):
    context = _make_non_policy_context(tmp_path / "bundle", family="siglip2")

    descriptor = BACKEND_REGISTRY.validate(context)

    assert descriptor.name == "ascend"


def test_descriptor_can_declare_model_support_without_policy_support():
    descriptor = BackendDescriptor(
        name="ascend",
        factory="tests.fake_backend_factory:create_backend",
        target_validator=lambda deployment: None,
        supported_model_families=frozenset({"ram_plus"}),
    )

    descriptor.validate_definition()


def test_descriptor_rejects_unknown_perception_family():
    with pytest.raises(BackendRegistryError, match="unknown model families") as error:
        BackendDescriptor(
            name="ascend",
            factory="tests.fake_backend_factory:create_backend",
            target_validator=lambda deployment: None,
            supported_model_families=frozenset({"unknown_perception"}),
        ).validate_definition()

    assert error.value.code == "unknown_model_family"


def _make_perception_context(
    root: Path,
    *,
    backend: str = "ascend",
    family: str = "ram_plus",
    deployment_name: str = "ascend",
) -> RuntimeContext:
    root.mkdir()
    bundle_paths = create_non_policy_bundle(root)
    manifest = make_non_policy_manifest(root, bundle_paths, family=family)
    if backend == "torch":
        manifest["deployments"][deployment_name] = {
            "uuid": "123e4567-e89b-42d3-a456-426614174001",
            "revision": 1,
            "backend": "torch",
            "device": "cpu",
        }
    else:
        manifest["deployments"][deployment_name] = manifest["deployments"]["ascend"]
    write_manifest(root, manifest)
    return RuntimeContext(load_inference_manifest(root, deployment_name))


@pytest.mark.parametrize("backend", ["torch", "ascend", "hisilicon", "rknn", "hmm"])
def test_static_registry_evidence_matches_declared_policy_matrix(backend):
    descriptor = BACKEND_REGISTRY.descriptor(backend)
    declared = descriptor.supported_policy_families
    evidenced = {family for kind, family in descriptor.evidence_pairs if kind == "policy"}
    assert declared == evidenced


@pytest.mark.parametrize(
    ("family", "backend"),
    [
        ("ram_plus", "torch"),
        ("sam2", "torch"),
        ("siglip2", "torch"),
        ("grounding_dino", "torch"),
        ("dummy_echo", "torch"),
        ("ram_plus", "ascend"),
        ("graspgen", "ascend"),
        ("sam2", "ascend"),
        ("grounding_dino", "ascend"),
        ("fullsubnet_cumulative_stateful", "ascend"),
    ],
)
def test_registry_supports_declared_perception_deployments(tmp_path, family, backend):
    context = _make_perception_context(tmp_path / "bundle", backend=backend, family=family)

    assert BACKEND_REGISTRY.validate(context).name == backend


@pytest.mark.parametrize("family", ["sam2", "siglip2", "grounding_dino"])
def test_registry_accepts_perception_family_with_compiled_ascend_support(tmp_path, family):
    context = _make_perception_context(tmp_path / "bundle", backend="ascend", family=family)

    assert BACKEND_REGISTRY.validate(context).name == "ascend"


def test_registry_validate_fails_closed_when_evidence_absent(tmp_path):
    context = _make_perception_context(tmp_path / "bundle", backend="ascend", family="ram_plus")
    registry = BackendRegistry(
        {
            "ascend": BackendDescriptor(
                name="ascend",
                factory="tests.fake_backend_factory:create_backend",
                target_validator=lambda deployment: None,
                supported_model_families=frozenset({"ram_plus"}),
            )
        }
    )

    with pytest.raises(BackendCompatibilityError, match="lacks conformance evidence") as error:
        registry.validate(context)
    assert error.value.code == "missing_conformance_evidence"


def test_registry_validate_rejects_adapter_deployment_mismatch(tmp_path):
    context = _make_perception_context(tmp_path / "bundle", backend="ascend", family="ram_plus")

    with pytest.raises(BackendCompatibilityError, match="not in the adapter supported deployments") as error:
        BACKEND_REGISTRY.validate(context, allowed_deployments=frozenset({"torch_cpu"}))
    assert error.value.code == "adapter_deployment_mismatch"


def test_descriptor_definition_rejects_evidence_overclaiming_support():
    with pytest.raises(BackendRegistryError, match="claims undeclared policy family") as error:
        BackendDescriptor(
            name="ascend",
            factory="tests.fake_backend_factory:create_backend",
            target_validator=lambda deployment: None,
            supported_policy_families=frozenset({"act"}),
            conformance_evidence=frozenset({ConformanceEvidence("policy", "pi05")}),
        ).validate_definition()
    assert error.value.code == "evidence_overclaims_support"


def test_descriptor_definition_rejects_non_policy_evidence_without_kind_or_family():
    with pytest.raises(BackendRegistryError, match="claims undeclared model") as error:
        BackendDescriptor(
            name="ascend",
            factory="tests.fake_backend_factory:create_backend",
            target_validator=lambda deployment: None,
            supported_model_families=frozenset({"ram_plus"}),
            conformance_evidence=frozenset({ConformanceEvidence("perception", "sam2")}),
        ).validate_definition()
    assert error.value.code == "evidence_overclaims_support"


def test_evidence_coherence_accepts_kind_backed_non_policy_evidence():
    BackendDescriptor(
        name="ascend",
        factory="tests.fake_backend_factory:create_backend",
        target_validator=lambda deployment: None,
        supported_model_kinds=frozenset({"perception"}),
        conformance_evidence=frozenset({ConformanceEvidence("perception", "ram_plus")}),
    ).validate_definition()


@pytest.mark.parametrize("alias", ["ascend_om", "ascend_om_3403", "3403", "om"])
def test_registry_rejects_noncanonical_backend_names_without_aliases(alias):
    with pytest.raises(BackendRegistryError, match="not registered"):
        BACKEND_REGISTRY.descriptor(alias)


@pytest.mark.parametrize(
    ("backend", "target_soc", "target_runtime", "artifact_format", "message"),
    [
        ("hisilicon", "rk3588", "hisilicon-worker", "om", "target.soc must be 'sd3403'"),
        ("hisilicon", "sd3403", "sd3403-worker", "om", "target.runtime must be 'hisilicon-worker'"),
        ("rknn", "rk3568", "rknn-lite", "rknn", "not in the RK3588 family"),
        ("rknn", "rk3588", "acl", "rknn", "is not an RKNN runtime"),
        ("ascend", "ascend310", "unknown", "om", "Ascend ACL runtime family"),
        ("ascend", "ascend310", "acl", "bin", "must use format 'om'"),
    ],
)
def test_registry_validates_explicit_backend_targets(
    tmp_path,
    backend,
    target_soc,
    target_runtime,
    artifact_format,
    message,
):
    context = _make_context(
        tmp_path / "bundle",
        backend=backend,
        target_soc=target_soc,
        target_runtime=target_runtime,
        artifact_format=artifact_format,
    )

    with pytest.raises(BackendCompatibilityError, match=message) as error:
        BACKEND_REGISTRY.validate(context)
    assert error.value.code == "incompatible_backend_target"


def test_same_backend_instance_serializes_inference(tmp_path):
    backend = FakeBackend(domains=ResourceDomainAdmissions())
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_release.clear()

    first, _, first_errors = _thread_call(lambda: backend.infer(_request("first")))
    assert backend.wait_for_infer_entries(1)
    second, _, second_errors = _thread_call(lambda: backend.infer(_request("second")))
    time.sleep(0.05)
    assert backend.infer_entries == 1

    backend.infer_release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert backend.infer_entries == 2
    assert backend.max_active_infer == 1
    backend.close()


def test_multiple_instances_are_rejected_without_evidence(tmp_path):
    domains = ResourceDomainAdmissions()
    first = FakeBackend(name="single-instance", domains=domains)
    with pytest.raises(BackendAdmissionError, match="does not support multiple live instances"):
        FakeBackend(name="single-instance", domains=domains)
    first.close()


def test_shared_resource_domain_serializes_independent_instances(tmp_path):
    domains = ResourceDomainAdmissions()
    capabilities = BackendCapabilities(
        supports_multiple_instances=True,
        resource_domain="shared-npu",
        admission_evidence=_MULTI_INSTANCE_EVIDENCE,
    )
    first = FakeBackend(name="multi", capabilities=capabilities, domains=domains)
    second = FakeBackend(name="multi", capabilities=capabilities, domains=domains)
    first.load(_make_context(tmp_path / "first"))
    second.load(_make_context(tmp_path / "second"))
    first.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: first.infer(_request("first")))
    assert first.wait_for_infer_entries(1)
    second_thread, _, second_errors = _thread_call(lambda: second.infer(_request("second")))
    time.sleep(0.05)
    assert second.infer_entries == 0

    first.infer_release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert second.infer_entries == 1
    _close_all(first, second)


def test_evidenced_per_instance_limit_allows_bounded_overlap(tmp_path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(
            thread_safe=True,
            max_in_flight_per_instance=2,
            admission_evidence=BackendAdmissionEvidence(
                overlapping_calls=True,
                output_isolation=True,
                failure_isolation=True,
                deterministic_cleanup=True,
            ),
        ),
        domains=ResourceDomainAdmissions(),
    )
    backend.load(_make_context(tmp_path / "bundle"))
    backend.infer_release.clear()

    first, _, first_errors = _thread_call(lambda: backend.infer(_request("first")))
    second, _, second_errors = _thread_call(lambda: backend.infer(_request("second")))
    assert backend.wait_for_infer_entries(2)
    third, _, third_errors = _thread_call(lambda: backend.infer(_request("third")))
    time.sleep(0.05)
    assert backend.infer_entries == 2
    assert backend.max_active_infer == 2

    backend.infer_release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    third.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert third_errors == []
    assert backend.infer_entries == 3
    backend.close()


def test_evidenced_resource_domain_limit_allows_bounded_overlap(tmp_path):
    domains = ResourceDomainAdmissions()
    capabilities = BackendCapabilities(
        supports_multiple_instances=True,
        resource_domain="shared-npu",
        max_in_flight_per_resource_domain=2,
        admission_evidence=_MULTI_INSTANCE_EVIDENCE,
    )
    first = FakeBackend(name="multi", capabilities=capabilities, domains=domains)
    second = FakeBackend(name="multi", capabilities=capabilities, domains=domains)
    third = FakeBackend(name="multi", capabilities=capabilities, domains=domains)
    first.load(_make_context(tmp_path / "first"))
    second.load(_make_context(tmp_path / "second"))
    third.load(_make_context(tmp_path / "third"))
    first.infer_release.clear()
    second.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: first.infer(_request("first")))
    second_thread, _, second_errors = _thread_call(lambda: second.infer(_request("second")))
    assert first.wait_for_infer_entries(1)
    assert second.wait_for_infer_entries(1)
    third_thread, _, third_errors = _thread_call(lambda: third.infer(_request("third")))
    time.sleep(0.05)
    assert third.infer_entries == 0

    first.infer_release.set()
    assert third.wait_for_infer_entries(1)
    second.infer_release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    third_thread.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    assert third_errors == []

    first.close()
    assert second.health().state is BackendState.READY
    assert second.infer(_request("after-independent-close")).action == [1.0]
    _close_all(second, third)


def test_independent_resource_domains_allow_physical_overlap(tmp_path):
    domains = ResourceDomainAdmissions()
    first = FakeBackend(
        name="multi",
        capabilities=BackendCapabilities(
            supports_multiple_instances=True,
            resource_domain="npu-0",
            admission_evidence=_MULTI_INSTANCE_EVIDENCE,
        ),
        domains=domains,
    )
    second = FakeBackend(
        name="multi",
        capabilities=BackendCapabilities(
            supports_multiple_instances=True,
            resource_domain="npu-1",
            admission_evidence=_MULTI_INSTANCE_EVIDENCE,
        ),
        domains=domains,
    )
    first.load(_make_context(tmp_path / "first"))
    second.load(_make_context(tmp_path / "second"))
    first.infer_release.clear()
    second.infer_release.clear()

    first_thread, _, first_errors = _thread_call(lambda: first.infer(_request("first")))
    second_thread, _, second_errors = _thread_call(lambda: second.infer(_request("second")))
    assert first.wait_for_infer_entries(1)
    assert second.wait_for_infer_entries(1)

    first.infer_release.set()
    second.infer_release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert first_errors == []
    assert second_errors == []
    _close_all(first, second)


def test_admission_limits_above_one_require_explicit_conformance_evidence():
    with pytest.raises(ValueError, match="requires conformance evidence"):
        BackendCapabilities(thread_safe=True, max_in_flight_per_instance=2)
    with pytest.raises(ValueError, match="requires conformance evidence"):
        BackendCapabilities(supports_multiple_instances=True)
    with pytest.raises(ValueError, match="requires supports_multiple_instances"):
        BackendCapabilities(resource_domain="npu", max_in_flight_per_resource_domain=2)

    capabilities = BackendCapabilities(
        thread_safe=True,
        max_in_flight_per_instance=2,
        supports_multiple_instances=True,
        resource_domain="npu",
        max_in_flight_per_resource_domain=2,
        admission_evidence=BackendAdmissionEvidence(
            overlapping_calls=True,
            output_isolation=True,
            failure_isolation=True,
            deterministic_cleanup=True,
            sdk_initialization=True,
            multi_instance_execution=True,
            independent_close=True,
        ),
    )
    assert capabilities.max_in_flight_per_instance == 2
    assert capabilities.resource_domain_limit == 2
