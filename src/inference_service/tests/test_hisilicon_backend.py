from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest, sha256_file
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendCapabilityError,
    BackendInferenceError,
    BackendLoadError,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.backends.hisilicon.backend import HisiliconBackend
from inference_service.backends.hisilicon.sd3403_protocol import SD3403Response, SD3403WorkerExitedError
from inference_service.codecs import create_policy_codec
from inference_service.pipeline import InferencePipeline, PipelineValidationError
from tests.manifest_fixtures import create_policy_bundle, write_manifest


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
    entries = [BundleFile(path=path, sha256=sha256_file(tmp_path / path)) for path in bundle_paths]
    manifest = {
        "schema_version": 1,
        "bundle": {
            "name": "hisilicon-act",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {"algorithm": "sha256", "value": canonical_bundle_digest(entries)},
        },
        "deployments": {
            "hisilicon": {
                "backend": "hisilicon",
                "target": {"soc": "sd3403", "runtime": "hisilicon-worker"},
                "artifacts": {
                    "policy": {"path": "artifacts/model.om", "format": "om", "sha256": sha256_file(model)},
                    "worker": {
                        "path": "artifacts/worker",
                        "format": "executable",
                        "sha256": sha256_file(worker),
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
        },
    }
    write_manifest(tmp_path, manifest)
    return RuntimeContext(load_inference_manifest(tmp_path, "hisilicon"), runtime_options=runtime_options or {})


@pytest.fixture(autouse=True)
def clear_protocol_instances():
    FakeProtocol.instances.clear()


def test_hisilicon_backend_loads_exact_manifest_artifacts_and_pipeline_uses_bindings(tmp_path):
    context = hisilicon_context(
        tmp_path,
        runtime_options={
            "perf_enabled": True,
            "perf_log_every": 2,
            "graceful_close_timeout": 1.25,
            "force_close": False,
        },
    )
    backend = HisiliconBackend(protocol_factory=FakeProtocol)
    pipeline = InferencePipeline("policy", context, backend, codec=create_policy_codec(context.policy))
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


def test_hisilicon_backend_worker_exit_is_recoverable_without_request_replay(tmp_path):
    context = hisilicon_context(tmp_path)
    backend = HisiliconBackend(protocol_factory=FakeProtocol)
    backend.load(context)
    first = FakeProtocol.instances[0]
    first.error = SD3403WorkerExitedError("worker exited unexpectedly")

    with pytest.raises(BackendInferenceError, match="worker exited unexpectedly") as error:
        backend.infer(
            InferenceRequest(
                request_id="failed",
                inputs={
                    "execution_plan": create_policy_execution_plan(context),
                    "role_inputs": {"policy": bound_policy_inputs()},
                },
            )
        )

    assert error.value.code == "worker_exited"
    assert error.value.recoverable is True
    assert backend.health().state is BackendState.DEGRADED
    backend.recover()
    assert first.closed is True
    assert len(FakeProtocol.instances) == 2
    assert FakeProtocol.instances[1].started is True
    assert backend.health().state is BackendState.READY
    backend.close()


def test_hisilicon_backend_load_failure_rolls_back_protocol(tmp_path):
    context = hisilicon_context(tmp_path)

    def failing_factory(worker_path, model_path, **options):
        protocol = FakeProtocol(worker_path, model_path, **options)
        protocol.start_error = RuntimeError("worker startup failed")
        return protocol

    backend = HisiliconBackend(protocol_factory=failing_factory)

    with pytest.raises(BackendLoadError, match="worker startup failed"):
        backend.load(context)

    protocol = FakeProtocol.instances[0]
    assert protocol.close_count == 1
    assert backend.health().state is BackendState.FAILED
    backend.close()
    assert protocol.close_count == 1


def test_hisilicon_backend_repeated_close_is_idempotent_and_reset_is_unsupported(tmp_path):
    context = hisilicon_context(tmp_path)
    backend = HisiliconBackend(protocol_factory=FakeProtocol)
    backend.load(context)

    assert backend.capabilities.max_in_flight_per_instance == 1
    assert backend.capabilities.resettable is False
    with pytest.raises(BackendCapabilityError) as error:
        backend.reset()
    assert error.value.code == "unsupported_capability"

    backend.close()
    backend.close()
    assert FakeProtocol.instances[0].close_count == 1
    assert backend.health().state is BackendState.CLOSED


def test_hisilicon_backend_serializes_requests_to_one_worker(tmp_path):
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
    backend = HisiliconBackend(protocol_factory=BlockingProtocol)
    backend.load(context)
    request = InferenceRequest(
        request_id="serialized",
        inputs={
            "execution_plan": create_policy_execution_plan(context),
            "role_inputs": {"policy": bound_policy_inputs()},
        },
    )
    errors: list[Exception] = []

    def infer() -> None:
        try:
            backend.infer(request)
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
    backend.close()


def test_hisilicon_pipeline_rejects_non_finite_action_output(tmp_path):
    context = hisilicon_context(tmp_path)
    backend = HisiliconBackend(protocol_factory=FakeProtocol)
    pipeline = InferencePipeline("policy", context, backend, codec=create_policy_codec(context.policy))
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

    assert backend.health().state is BackendState.READY
    pipeline.close()


def create_policy_execution_plan(context: RuntimeContext):
    from inference_service.codecs import build_execution_plan

    deployment = context.deployment
    return build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)


def bound_policy_inputs():
    from inference_service.codecs import BoundInputs, BoundTensor

    return BoundInputs(
        (
            BoundTensor("observation.state", "state", 0, np.zeros((1, 6), dtype=np.float32)),
            BoundTensor("observation.images.top", "camera", 1, np.zeros((1, 3, 16, 24), dtype=np.float32)),
        )
    )


def test_hisilicon_backend_rejects_non_executable_worker_before_protocol_creation(tmp_path):
    context = hisilicon_context(tmp_path, executable=False)
    backend = HisiliconBackend(protocol_factory=FakeProtocol)

    with pytest.raises(BackendLoadError, match="not executable") as error:
        backend.load(context)

    assert error.value.code == "worker_not_executable"
    assert FakeProtocol.instances == []
    backend.close()


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"action_output_index": 1},
        {"perf_enabled": "true"},
        {"perf_log_every": 0},
        {"graceful_close_timeout": -1},
    ],
)
def test_hisilicon_backend_rejects_invalid_operational_options(tmp_path, runtime_options):
    context = hisilicon_context(tmp_path, runtime_options=runtime_options)
    backend = HisiliconBackend(protocol_factory=FakeProtocol)

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == "invalid_runtime_options"
    backend.close()


def test_registry_creates_canonical_hisilicon_backend_lazily(tmp_path):
    context = hisilicon_context(tmp_path)

    backend = BACKEND_REGISTRY.create(context)

    assert isinstance(backend, HisiliconBackend)
    assert backend.name == "hisilicon"
    backend.close()
