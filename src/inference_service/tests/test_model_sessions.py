from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import (
    BackendCapabilityError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendNotReadyError,
    BackendState,
    ResourceDomainAdmissions,
    RuntimeContext,
)
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import AscendOmModelSession, TorchModelSession
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_non_policy_bundle,
    make_manifest,
    make_non_policy_manifest,
    write_manifest,
)


def _torch_context(tmp_path: Path, *, device: str = "cpu") -> RuntimeContext:
    bundle_files = create_non_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, bundle_files, deployment_name=f"torch_{device}")
    manifest["model"] = {
        "kind": "generic",
        "family": "fake_torch",
        "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1, 2]}],
        "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1, 2]}],
    }
    manifest["deployments"][f"torch_{device}"]["device"] = device
    write_manifest(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(tmp_path, f"torch_{device}"))


def _ascend_context(tmp_path: Path) -> RuntimeContext:
    bundle_files = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, bundle_files)
    write_manifest(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(tmp_path, "ascend"))


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = _FakeCuda(cuda_available)

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def as_tensor(value) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def is_tensor(value) -> bool:
        return isinstance(value, _FakeTensor)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeModule:
    def __init__(self, *, fail_eval: bool = False) -> None:
        self.fail_eval = fail_eval
        self.device = None
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        if self.fail_eval:
            raise RuntimeError("eval failed")
        return self

    def __call__(self, inputs):
        self.calls += 1
        features = inputs["features"]
        return {"scores": _FakeTensor(features.value + np.float32(1.0))}


class _FailingModule(_FakeModule):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def __call__(self, inputs):
        self.calls += 1
        raise self.error


class _SlowModule(_FakeModule):
    def __call__(self, inputs):
        time.sleep(0.02)
        return super().__call__(inputs)


def test_torch_session_readiness_named_execution_and_repeated_close(tmp_path) -> None:
    context = _torch_context(tmp_path)
    module = _FakeModule()
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)

    with pytest.raises(BackendNotReadyError):
        session.infer(NamedTensorRequest("before-load", {"features": np.zeros((1, 2), dtype=np.float32)}))

    session.load(context)
    assert session.health().ready
    result = session.infer(NamedTensorRequest("request-1", {"features": np.ones((1, 2), dtype=np.float32)}))

    np.testing.assert_array_equal(result.outputs["scores"], np.full((1, 2), 2.0, dtype=np.float32))
    assert result.deployment.deployment == "torch_cpu"
    assert result.deployment.deployment_fingerprint == context.deployment_fingerprint
    assert result.metadata["model_family"] == "fake_torch"
    assert module.calls == 1
    health = session.health()
    assert health.state is BackendState.READY
    assert health.ready
    assert health.reason_code is None
    assert health.failure_count == 0
    assert health.last_successful_inference_time is not None

    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED


def test_torch_session_fails_clear_when_sdk_or_cuda_is_unavailable(tmp_path) -> None:
    context = _torch_context(tmp_path)
    missing = TorchModelSession(
        lambda _context: _FakeModule(),
        torch_loader=lambda: (_ for _ in ()).throw(BackendLoadError("torch missing", code="missing_dependency")),
    )
    with pytest.raises(BackendLoadError, match="torch missing") as missing_error:
        missing.load(context)
    assert missing_error.value.code == "missing_dependency"
    assert missing.health().reason_code == "missing_dependency"
    missing.close()

    cuda_root = tmp_path / "cuda"
    cuda_root.mkdir()
    cuda = TorchModelSession(lambda _context: _FakeModule(), torch_loader=lambda: _FakeTorch(cuda_available=False))
    with pytest.raises(BackendLoadError, match="CUDA device is unavailable") as cuda_error:
        cuda.load(_torch_context(cuda_root, device="cuda"))
    assert cuda_error.value.code == "device_unavailable"
    cuda.close()


def test_torch_session_partial_load_rolls_back_module(tmp_path) -> None:
    context = _torch_context(tmp_path)
    torch_module = _FakeTorch()
    session = TorchModelSession(lambda _context: _FakeModule(fail_eval=True), torch_loader=lambda: torch_module)

    with pytest.raises(BackendLoadError, match="eval failed"):
        session.load(context)

    assert session.health().state is BackendState.FAILED
    session.close()


class _FakeLease:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeRuntimeManager:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.lease = _FakeLease()
        self.acquire_calls = []

    def acquire(self, device_id: int, config_path: str | None = None):
        self.acquire_calls.append((device_id, config_path))
        if self.error is not None:
            raise self.error
        return self.lease


class _FakeAclModel:
    instances = []
    fail_role = None

    def __init__(self, _lease, role, path, bindings) -> None:
        self.role = role
        self.path = path
        self.bindings = bindings
        self.close_calls = 0
        self.execute_calls = 0
        self.__class__.instances.append(self)

    def load_descriptor(self) -> None:
        if self.role == self.fail_role:
            raise BackendLoadError("runtime name mismatch", code="runtime_name_mismatch")

    def prepare_datasets(self) -> None:
        pass

    def execute(self, inputs):
        self.execute_calls += 1
        assert sorted(inputs) == list(range(len(self.bindings.inputs)))
        return {
            int(binding.index): np.full(binding.shape, 0.25, dtype=np.dtype(binding.dtype))
            for binding in self.bindings.outputs
            if binding.index is not None
        }

    def close(self) -> None:
        self.close_calls += 1


def test_ascend_session_validates_abi_executes_named_outputs_and_closes_once(tmp_path) -> None:
    _FakeAclModel.instances = []
    _FakeAclModel.fail_role = None
    manager = _FakeRuntimeManager()
    session = AscendOmModelSession(runtime_manager=manager, model_factory=_FakeAclModel)
    context = _ascend_context(tmp_path)

    session.load(context)
    result = session.infer(
        NamedTensorRequest("acl-1", {"observation.image": np.zeros((1, 3, 384, 384), dtype=np.float32)})
    )

    assert result.outputs["tag_logits"].shape == (1, 4585)
    assert result.deployment.backend == "ascend"
    assert result.deployment.deployment_fingerprint == context.deployment_fingerprint
    assert _FakeAclModel.instances[0].execute_calls == 1

    session.close()
    session.close()
    assert _FakeAclModel.instances[0].close_calls == 1
    assert manager.lease.close_calls == 1


def test_ascend_close_reports_structured_lifecycle_error_after_releasing_all_resources(tmp_path) -> None:
    class FailingLease(_FakeLease):
        def close(self) -> None:
            super().close()
            raise RuntimeError("lease close failed")

    class FailingManager(_FakeRuntimeManager):
        def __init__(self) -> None:
            super().__init__()
            self.lease = FailingLease()

    class FailingModel(_FakeAclModel):
        def close(self) -> None:
            super().close()
            raise RuntimeError("model close failed")

    FailingModel.instances = []
    FailingModel.fail_role = None
    manager = FailingManager()
    session = AscendOmModelSession(runtime_manager=manager, model_factory=FailingModel)
    session.load(_ascend_context(tmp_path))

    with pytest.raises(BackendLifecycleError, match="model close failed.*lease close failed") as error:
        session.close()

    assert error.value.code == "close_failed"
    assert FailingModel.instances[0].close_calls == 1
    assert manager.lease.close_calls == 1
    assert session.health().state is BackendState.CLOSED
    assert session.health().reason_code == "close_failed"
    session.close()


def test_ascend_session_rolls_back_lease_on_abi_failure(tmp_path) -> None:
    _FakeAclModel.instances = []
    _FakeAclModel.fail_role = "model"
    manager = _FakeRuntimeManager()
    session = AscendOmModelSession(runtime_manager=manager, model_factory=_FakeAclModel)

    with pytest.raises(BackendLoadError, match="runtime name mismatch") as error:
        session.load(_ascend_context(tmp_path))

    assert error.value.code == "runtime_name_mismatch"
    assert _FakeAclModel.instances[0].close_calls == 1
    assert manager.lease.close_calls == 1
    assert session.health().state is BackendState.FAILED
    session.close()


@pytest.mark.parametrize(
    ("message", "code"),
    [("ACL SDK unavailable", "missing_dependency"), ("Ascend device 0 unavailable", "device_unavailable")],
)
def test_ascend_session_fails_clear_when_sdk_or_device_is_unavailable(tmp_path, message, code) -> None:
    manager = _FakeRuntimeManager(error=BackendLoadError(message, code=code))
    session = AscendOmModelSession(runtime_manager=manager, model_factory=_FakeAclModel)

    with pytest.raises(BackendLoadError, match=message) as error:
        session.load(_ascend_context(tmp_path))

    assert error.value.code == code
    assert session.health().reason_code == code
    session.close()


def test_session_rejects_expired_deadline_before_execution(tmp_path) -> None:
    module = _FakeModule()
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)
    session.load(_torch_context(tmp_path))

    with pytest.raises(Exception) as deadline_error:
        session.infer(
            NamedTensorRequest(
                "expired",
                {"features": np.zeros((1, 2), dtype=np.float32)},
                deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )
    assert getattr(deadline_error.value, "code", None) == "deadline_exceeded"
    assert module.calls == 0
    assert session.health().state is BackendState.READY
    assert session.health().failure_count == 0
    session.close()


def test_session_deadline_expiring_during_execution_does_not_fail_runtime(tmp_path) -> None:
    module = _SlowModule()
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)
    session.load(_torch_context(tmp_path))

    with pytest.raises(Exception) as deadline_error:
        session.infer(
            NamedTensorRequest(
                "expired-during-execution",
                {"features": np.zeros((1, 2), dtype=np.float32)},
                deadline=datetime.now(timezone.utc) + timedelta(milliseconds=5),
            )
        )

    assert getattr(deadline_error.value, "code", None) == "deadline_exceeded"
    assert module.calls == 1
    assert session.health().state is BackendState.READY
    assert session.health().failure_count == 0
    session.close()


def test_torch_session_rejects_unsupported_controls_without_claiming_capabilities(tmp_path) -> None:
    session = TorchModelSession(lambda _context: _FakeModule(), torch_loader=_FakeTorch)
    session.load(_torch_context(tmp_path))

    assert not session.capabilities.resettable
    assert not session.capabilities.supports_cancellation
    for operation, capability in (
        (session.reset, "reset"),
        (lambda: session.cancel("request"), "cancellation"),
        (session.recover, "recovery"),
    ):
        with pytest.raises(BackendCapabilityError) as error:
            operation()
        assert error.value.code == "unsupported_capability"
        assert error.value.capability == capability
    assert session.health().state is BackendState.READY
    assert session.health().failure_count == 0
    session.close()


def test_ascend_session_rejects_unsupported_controls_without_claiming_capabilities(tmp_path) -> None:
    _FakeAclModel.fail_role = None
    session = AscendOmModelSession(runtime_manager=_FakeRuntimeManager(), model_factory=_FakeAclModel)
    session.load(_ascend_context(tmp_path))

    assert not session.capabilities.resettable
    assert not session.capabilities.supports_cancellation
    for operation, capability in (
        (session.reset, "reset"),
        (lambda: session.cancel("request"), "cancellation"),
        (session.recover, "recovery"),
    ):
        with pytest.raises(BackendCapabilityError) as error:
            operation()
        assert error.value.code == "unsupported_capability"
        assert error.value.capability == capability
    assert session.health().state is BackendState.READY
    session.close()


def test_runtime_failure_is_terminal_without_concrete_recovery_support(tmp_path) -> None:
    runtime_error = BackendInferenceError("device context lost", code="runtime_lost", recoverable=True)
    module = _FailingModule(runtime_error)
    session = TorchModelSession(lambda _context: module, torch_loader=_FakeTorch)
    session.load(_torch_context(tmp_path))

    with pytest.raises(BackendInferenceError, match="device context lost") as error:
        session.infer(NamedTensorRequest("failure", {"features": np.zeros((1, 2), dtype=np.float32)}))

    assert error.value is runtime_error
    health = session.health()
    assert health.state is BackendState.FAILED
    assert not health.ready
    assert not health.recoverable
    assert health.reason_code == "runtime_lost"
    assert health.failure_count == 1
    with pytest.raises(BackendCapabilityError) as recovery_error:
        session.recover()
    assert recovery_error.value.capability == "recovery"
    with pytest.raises(BackendNotReadyError):
        session.infer(NamedTensorRequest("after-failure", {"features": np.zeros((1, 2), dtype=np.float32)}))

    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED


def test_ascend_resource_domain_serializes_sessions(tmp_path) -> None:
    class SerialModel(_FakeAclModel):
        fail_role = None
        active = 0
        max_active = 0
        lock = threading.Lock()

        def execute(self, inputs):
            with self.lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                time.sleep(0.05)
                return super().execute(inputs)
            finally:
                with self.lock:
                    type(self).active -= 1

    class LeaseManager:
        def acquire(self, _device_id, _config_path=None):
            return _FakeLease()

    domains = ResourceDomainAdmissions()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    sessions = [
        AscendOmModelSession(runtime_manager=LeaseManager(), model_factory=SerialModel, domains=domains),
        AscendOmModelSession(runtime_manager=LeaseManager(), model_factory=SerialModel, domains=domains),
    ]
    sessions[0].load(_ascend_context(first_root))
    sessions[1].load(_ascend_context(second_root))
    request = NamedTensorRequest("serialized", {"observation.image": np.zeros((1, 3, 384, 384), dtype=np.float32)})
    threads = [threading.Thread(target=session.infer, args=(request,)) for session in sessions]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert SerialModel.max_active == 1
    for session in sessions:
        session.close()


def test_ascend_partial_load_closes_prior_models_in_reverse_order(tmp_path) -> None:
    root = tmp_path
    bundle_paths = create_non_policy_bundle(root)
    for name in ("first", "second"):
        artifact = root / "artifacts" / f"{name}.om"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(name.encode())
    entries = [BundleFile(path=path) for path in bundle_paths]
    manifest = {
        "schema_version": 2,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": "two-role-model",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "two-role-model", entries),
            },
        },
        "model": {
            "kind": "generic",
            "family": "two_role",
            "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1, 2]}],
            "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1, 2]}],
        },
        "deployments": {
            "ascend": {
                "uuid": TEST_DEPLOYMENT_UUID,
                "revision": 1,
                "backend": "ascend",
                "target": {"soc": "fake", "runtime": "acl"},
                "artifacts": {
                    "first": {"path": "artifacts/first.om", "format": "om"},
                    "second": {"path": "artifacts/second.om", "format": "om"},
                },
                "execution": ["first", "second"],
                "bindings": {
                    "first": {
                        "inputs": [
                            {
                                "semantic": "features",
                                "runtime_name": "features",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 2],
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": "internal.hidden",
                                "runtime_name": "hidden",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 2],
                            }
                        ],
                    },
                    "second": {
                        "inputs": [
                            {
                                "semantic": "internal.hidden",
                                "runtime_name": "hidden",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 2],
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": "scores",
                                "runtime_name": "scores",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 2],
                            }
                        ],
                    },
                },
            }
        },
    }
    write_manifest(root, manifest)
    context = RuntimeContext(load_inference_manifest(root, "ascend"))
    _FakeAclModel.instances = []
    _FakeAclModel.fail_role = "second"
    manager = _FakeRuntimeManager()
    session = AscendOmModelSession(runtime_manager=manager, model_factory=_FakeAclModel)

    with pytest.raises(BackendLoadError, match="runtime name mismatch"):
        session.load(context)

    first = next(model for model in _FakeAclModel.instances if model.role == "first")
    assert first.close_calls == 1
    assert manager.lease.close_calls == 1
    session.close()
