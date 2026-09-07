"""Dependency-light single-pipeline facade over the unified inference runtime."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from inference_manifest import PolicyMetadata, load_inference_manifest
from inference_service.backends import BackendCapabilities, BackendRegistry, InferenceRequest
from inference_service.pipeline import InferencePipelineManager, create_pipeline_manager
from inference_service.runtime_composition import require_runtime_dependencies
from inference_service.unified_runtime import ModelRuntimeHandle, RegistrySet, RuntimeProviders


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a Torch placement device without interpreting backend identities."""

    normalized = device.lower().strip().replace("-", "_")

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        try:
            import torch_npu

            if torch_npu.npu.is_available():
                return torch.device("npu:0")
        except ImportError:
            pass
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device_index = normalized[4:].lstrip(":") or "0"
        return torch.device(f"cuda:{device_index}")
    if normalized in {"mps", "metal"}:
        if not mps_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("npu"):
        try:
            import torch_npu

            if torch_npu.npu.is_available():
                device_index = normalized[3:].lstrip(":") or "0"
                return torch.device(f"npu:{device_index}")
        except ImportError:
            pass
        raise RuntimeError("NPU requested but torch_npu not available")
    raise ValueError(f"Unknown Torch device: {device}")


@dataclass(frozen=True)
class InferenceResult:
    action: object
    chunk_size: int
    latency_ms: float
    policy_type: str
    backend_type: str
    raw_action: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        candidate = self.action
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        return np.asarray(candidate)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.to_numpy()).shape)


class PureInferenceEngine:
    """Own one validated unified pipeline with no ROS dependencies."""

    def __init__(
        self,
        model_path: str | Path,
        deployment: str,
        *,
        pipeline_id: str = "pure",
        request_timeout: float | None = None,
        default_task: str | None = None,
        runtime_options: Mapping[str, object] | None = None,
        priority_scheduling: bool = False,
        registry: BackendRegistry | None = None,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
        model_session_factory=None,
        pi05_diagnostic_schedule=None,
        pi05_diagnostic_schedule_source: str | None = None,
    ) -> None:
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._providers = providers
        validated_manifest = load_inference_manifest(model_path, deployment)
        self._pipeline_id = pipeline_id
        self._context = validated_manifest
        self._chunk_size: int | None = None
        self._manager: InferencePipelineManager = create_pipeline_manager(
            pipeline_id,
            validated_manifest,
            request_timeout=request_timeout,
            default_task=default_task,
            runtime_options=runtime_options,
            priority_scheduling=priority_scheduling,
            registry=registry,
            registry_set=registry_set,
            providers=providers,
            model_session_factory=model_session_factory,
            pi05_diagnostic_schedule=pi05_diagnostic_schedule,
            pi05_diagnostic_schedule_source=pi05_diagnostic_schedule_source,
        )

    def __call__(
        self,
        batch: dict[str, Tensor | np.ndarray | object],
        *,
        prompt: str | None = None,
        request_id: str | None = None,
        control_inputs: Mapping[str, object] | None = None,
        capture_raw_action: bool = False,
    ) -> InferenceResult:
        result = self._manager.infer(
            self._pipeline_id,
            InferenceRequest(
                request_id=request_id or uuid.uuid4().hex,
                inputs=batch,
                prompt=prompt,
            ),
            control_inputs=control_inputs,
            capture_raw_action=capture_raw_action,
        )
        self._chunk_size = result.actual_chunk_size
        return InferenceResult(
            action=result.action,
            chunk_size=result.actual_chunk_size,
            latency_ms=result.total_latency_ms,
            policy_type=self._context.policy.policy_type,
            backend_type=result.backend,
            raw_action=result.raw_action,
            metadata=result.metadata,
        )

    def reset(self) -> None:
        self._manager.reset(self._pipeline_id)

    def close(self) -> None:
        self._manager.close()

    @property
    def policy_type(self) -> str:
        return self._context.policy.policy_type

    @property
    def backend_type(self) -> str:
        return self._context.deployment.backend

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._manager.capabilities(self._pipeline_id)

    @property
    def runtime_handle(self) -> ModelRuntimeHandle | None:
        """Expose the unified handle for diagnostics and controlled recovery."""

        return self._manager.runtime_handle(self._pipeline_id)

    @property
    def policy_metadata(self) -> PolicyMetadata:
        return self._context.policy

    @property
    def nominal_chunk_size(self) -> int | None:
        return self._context.policy.nominal_chunk_size

    @property
    def max_action_dimension(self) -> int | None:
        return self._context.policy.max_action_dimension

    @property
    def chunk_size(self) -> int | None:
        return self._chunk_size
