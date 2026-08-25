from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import (
    STATIC_BACKEND_DESCRIPTORS,
    BackendCapabilityError,
    BackendInferenceError,
    BackendLoadError,
    BackendRegistry,
    BackendRegistryError,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.hisilicon.sd3403_protocol import SD3403Response, SD3403WorkerExitedError
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import HisiliconModelSession
from inference_service.pipeline import PipelineValidationError, create_inference_pipeline
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_policy_bundle,
    policy_model,
    v3_runtime_deployment,
    write_manifest,
)

_STATIC_BACKEND_REGISTRY = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)


class FakeProtocol:
    instances: list[FakeProtocol] = []

    def __init__(self, worker_path: Path, model_path: Path, **options) -> None:
        self.worker_path = worker_path
        self.model_path = model_path
        self.options = options
        self.started = False
        self.closed = False
        self.close_count = 0
        self.inputs: tuple[np.ndarray, ...] | None = None
        self.outputs: dict[int, np.ndarray] = {1: np.ones((1, 4, 6), dtype=np.float32)}
        self.error: Exception | None = None
        self.start_error: Exception | None = None
        self.model_load_ms = 12.5
        type(self).instances.append(self)

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def execute(self, inputs: tuple[np.ndarray, ...]) -> SD3403Response:
        self.inputs = inputs
        if self.error is not None:
            raise self.error
        return SD3403Response(outputs=self.outputs, worker_latency_us=2500, request_id=11)

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


def hisilicon_context(tmp_path, *, executable: bool = True, runtime_options=None) -> RuntimeContext:
    bundle_paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    model = tmp_path / "artifacts" / "model.om"
    worker = tmp_path / "artifacts" / "worker"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"om")
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755 if executable else 0o644)
    entries = [BundleFile(path=path) for path in bundle_paths]
    manifest = {
        "schema_version": 3,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": "hisilicon-act",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "hisilicon-act", entries),
            },
        },
        "deployments": {
            "hisilicon": v3_runtime_deployment(
                {
                    "uuid": TEST_DEPLOYMENT_UUID,
                    "revision": 1,
                    "backend": "hisilicon",
                    "target": {"soc": "sd3403", "runtime": "hisilicon-worker"},
                    "artifacts": {
                        "policy": {"path": "artifacts/model.om", "format": "om"},
                        "worker": {
                            "path": "artifacts/worker",
                            "format": "executable",
                        },
                    },
                    "execution": ["policy"],
                    "bindings": {
                        "policy": {
                            "inputs": [
                                {
                                    "semantic": "observation.images.top",
                                    "runtime_name": "camera",
                                    "index": 1,
                                    "dtype": "float32",
                                    "shape": [1, 3, 16, 24],
                                    "layout": "NCHW",
                                },
                                {
                                    "semantic": "observation.state",
                                    "runtime_name": "state",
                                    "index": 0,
                                    "dtype": "float32",
                                    "shape": [1, 6],
                                },
                            ],
                            "outputs": [
                                {
                                    "semantic": "action",
                                    "runtime_name": "action",
                                    "index": 1,
                                    "dtype": "float32",
                                    "shape": [1, 4, 6],
                                }
                            ],
                        }
                    },
                }
            )
        },
        "model": policy_model("act"),
    }
    write_manifest(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(tmp_path, "hisilicon"), runtime_options=runtime_options or {})


@pytest.fixture(autouse=True)
def clear_protocol_instances():
    FakeProtocol.instances.clear()


def patch_identity_processors(monkeypatch):
    from inference_service import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.factory,
        "create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, lambda action: action),
    )


def execute_policy_role(session, request):
    with session.execution(request) as execution:
        return execution.invoke("policy", request.inputs)


def test_hisilicon_session_loads_exact_manifest_artifacts_and_pipeline_uses_bindings(tmp_path, monkeypatch):
    patch_identity_processors(monkeypatch)
    context = hisilicon_context(
        tmp_path,
        runtime_options={
            "perf_enabled": True,
            "perf_log_every": 2,
            "graceful_close_timeout": 1.25,
            "force_close": False,
        },
    )
    pipeline = create_inference_pipeline(
        "policy",
        context.validated_manifest,
        runtime_options=context.runtime_options,
        model_session_factory=lambda ctx, options: HisiliconModelSession(protocol_factory=FakeProtocol),
    )
    pipeline.load()

    result = pipeline.infer(
        InferenceRequest(
            request_id="request-1",
            inputs={
                "observation.state": np.arange(6, dtype=np.float64),
                "observation.images.top": np.ones((3, 8, 12), dtype=np.float32),
            },
        )
    )

    protocol = FakeProtocol.instances[0]
    assert protocol.started is True
    assert protocol.worker_path == context.resolved_artifacts["worker"]
    assert protocol.model_path == context.resolved_artifacts["policy"]
    assert protocol.options == {"graceful_close_timeout": 1.25, "force_close": False}
    assert protocol.inputs is not None
    assert [value.shape for value in protocol.inputs] == [(1, 6), (1, 3, 16, 24)]
    assert [value.dtype for value in protocol.inputs] == [np.dtype("float32"), np.dtype("float32")]
    np.testing.assert_array_equal(result.action, np.ones((1, 4, 6), dtype=np.float32))
    assert result.actual_chunk_size == 4
    assert result.metadata["protocol"] == "sd3403-v1"
    assert result.metadata["worker_request_id"] == 11
    assert result.metadata["worker_latency_ms"] == 2.5
    pipeline.close()
    assert protocol.closed is True


def test_hisilicon_session_worker_exit_is_recoverable_without_request_replay(tmp_path):
    context = hisilicon_context(tmp_path)
    session = HisiliconModelSession(protocol_factory=FakeProtocol)
    session.load(context)
    first = FakeProtocol.instances[0]
    first.error = SD3403WorkerExitedError("worker exited unexpectedly")

    with pytest.raises(BackendInferenceError, match="worker exited unexpectedly") as error:
        execute_policy_role(
            session,
            NamedTensorRequest(
                request_id="failed",
                inputs=bound_policy_inputs(),
            ),
        )

    assert error.value.code == "worker_exited"
    assert error.value.recoverable is True
    assert session.health().state is BackendState.DEGRADED
    session.recover()
    assert first.closed is True
    assert len(FakeProtocol.instances) == 2
    assert FakeProtocol.instances[1].started is True
    assert session.health().state is BackendState.READY
    session.close()


def test_hisilicon_session_load_failure_rolls_back_protocol(tmp_path):
    context = hisilicon_context(tmp_path)

    def failing_factory(worker_path, model_path, **options):
        protocol = FakeProtocol(worker_path, model_path, **options)
        protocol.start_error = RuntimeError("worker startup failed")
        return protocol

    session = HisiliconModelSession(protocol_factory=failing_factory)

    with pytest.raises(BackendLoadError, match="worker startup failed"):
        session.load(context)

    protocol = FakeProtocol.instances[0]
    assert protocol.close_count == 1
    assert session.health().state is BackendState.FAILED
    session.close()
    assert protocol.close_count == 1


def test_hisilicon_session_repeated_close_is_idempotent_and_reset_is_unsupported(tmp_path):
    context = hisilicon_context(tmp_path)
    session = HisiliconModelSession(protocol_factory=FakeProtocol)
    session.load(context)

    assert session.capabilities.max_in_flight_per_instance == 1
    assert session.capabilities.resettable is False
    with pytest.raises(BackendCapabilityError) as error:
        session.reset()
    assert error.value.code == "unsupported_capability"

    session.close()
    session.close()
    assert FakeProtocol.instances[0].close_count == 1
    assert session.health().state is BackendState.CLOSED


def test_hisilicon_session_serializes_requests_to_one_worker(tmp_path):
    class BlockingProtocol(FakeProtocol):
        entered = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        active = 0
        max_active = 0

        def execute(self, inputs):
            with self.lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            self.entered.set()
            assert self.release.wait(timeout=1.0)
            try:
                return super().execute(inputs)
            finally:
                with self.lock:
                    type(self).active -= 1

    context = hisilicon_context(tmp_path)
    session = HisiliconModelSession(protocol_factory=BlockingProtocol)
    session.load(context)
    request = NamedTensorRequest(
        request_id="serialized",
        inputs=bound_policy_inputs(),
    )
    errors: list[Exception] = []

    def infer() -> None:
        try:
            execute_policy_role(session, request)
        except Exception as exc:  # pragma: no cover - asserted through errors below
            errors.append(exc)

    first = threading.Thread(target=infer)
    second = threading.Thread(target=infer)
    first.start()
    assert BlockingProtocol.entered.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)

    assert BlockingProtocol.max_active == 1
    BlockingProtocol.release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert BlockingProtocol.max_active == 1
    session.close()


def test_hisilicon_pipeline_rejects_non_finite_action_output(tmp_path, monkeypatch):
    patch_identity_processors(monkeypatch)
    context = hisilicon_context(tmp_path)
    session = HisiliconModelSession(protocol_factory=FakeProtocol)
    pipeline = create_inference_pipeline(
        "policy",
        context.validated_manifest,
        model_session_factory=lambda ctx, options: session,
    )
    pipeline.load()
    FakeProtocol.instances[0].outputs = {1: np.full((1, 4, 6), np.nan, dtype=np.float32)}

    with pytest.raises(PipelineValidationError, match="non-finite"):
        pipeline.infer(
            InferenceRequest(
                request_id="non-finite",
                inputs={
                    "observation.state": np.zeros(6, dtype=np.float32),
                    "observation.images.top": np.zeros((3, 16, 24), dtype=np.float32),
                },
            )
        )

    assert session.health().state is BackendState.READY
    pipeline.close()


def bound_policy_inputs():
    return {
        "observation.state": np.zeros((1, 6), dtype=np.float32),
        "observation.images.top": np.zeros((1, 3, 16, 24), dtype=np.float32),
    }


def test_hisilicon_session_rejects_non_executable_worker_before_protocol_creation(tmp_path):
    context = hisilicon_context(tmp_path, executable=False)
    session = HisiliconModelSession(protocol_factory=FakeProtocol)

    with pytest.raises(BackendLoadError, match="not executable") as error:
        session.load(context)

    assert error.value.code == "worker_not_executable"
    assert FakeProtocol.instances == []
    session.close()


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"action_output_index": 1},
        {"perf_enabled": "true"},
        {"perf_log_every": 0},
        {"graceful_close_timeout": -1},
    ],
)
def test_hisilicon_session_rejects_invalid_operational_options(tmp_path, runtime_options):
    context = hisilicon_context(tmp_path, runtime_options=runtime_options)
    session = HisiliconModelSession(protocol_factory=FakeProtocol)

    with pytest.raises(BackendLoadError) as error:
        session.load(context)

    assert error.value.code == "invalid_runtime_options"
    session.close()


def test_registry_marks_hisilicon_as_session_only(tmp_path):
    context = hisilicon_context(tmp_path)

    descriptor = _STATIC_BACKEND_REGISTRY.validate(context)
    assert descriptor.factory is None
    with pytest.raises(BackendRegistryError) as error:
        _STATIC_BACKEND_REGISTRY._create_legacy_backend(context)
    assert error.value.code == "legacy_backend_unavailable"
