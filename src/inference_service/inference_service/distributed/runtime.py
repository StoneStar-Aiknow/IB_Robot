"""Split edge processor and cloud backend runtimes for distributed pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np
import torch

from inference_manifest import CompiledDeployment, ValidatedManifest
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendCapabilities,
    BackendRegistry,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import create_policy_codec
from inference_service.pipeline import InferencePipeline, InferencePipelineManager
from inference_service.pipeline.processors import create_lerobot_processor_views
from inference_service.pipeline.validation import validate_action_output


def _identity_preprocessor(inputs: Mapping[str, object]) -> Mapping[str, object]:
    return inputs


def _identity_postprocessor(action: object) -> object:
    return action


class EdgeProcessorRuntime:
    """Preserve the edge raw-observation and robot-unit boundary."""

    def __init__(
        self,
        pipeline_id: str,
        validated_manifest: ValidatedManifest,
        *,
        default_task: str | None = None,
    ) -> None:
        self.pipeline_id = pipeline_id
        self._manifest = validated_manifest
        self._default_task = default_task
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

    def preprocess(self, inputs: Mapping[str, object], *, prompt: str | None = None) -> Mapping[str, object]:
        self._require_ready()
        return dict(inputs)

    def postprocess(self, action: object, *, actual_chunk_size: int) -> object:
        self._require_ready()
        validate_action_output(
            action,
            actual_chunk_size=actual_chunk_size,
            action_dimension=self._manifest.policy.output_features["action"].shape[-1],
            pipeline_id=self.pipeline_id,
            phase="postprocessor",
        )
        return action

    def reset(self, deadline: datetime | None = None) -> None:
        self._require_ready()
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("edge processor reset deadline expired")

    def close(self) -> None:
        self._loaded = False

    def _require_ready(self) -> None:
        if not self._loaded:
            raise RuntimeError("edge processors are not loaded")


class CloudBackendRuntime:
    """Execute canonical preprocessed tensors without loading LeRobot processors."""

    def __init__(
        self,
        pipeline_id: str,
        validated_manifest: ValidatedManifest,
        *,
        request_timeout: float | None = None,
        runtime_options: Mapping[str, object] | None = None,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        context = RuntimeContext(validated_manifest, runtime_options=runtime_options or {})
        backend = registry.create(context)
        codec = create_policy_codec(context.policy) if isinstance(context.deployment, CompiledDeployment) else None
        preprocessor, postprocessor = create_lerobot_processor_views()
        pipeline = InferencePipeline(
            pipeline_id,
            context,
            backend,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            codec=codec,
            request_timeout=request_timeout,
        )
        self.pipeline_id = pipeline_id
        self._manager = InferencePipelineManager((pipeline,))
        self._manager.start()

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._manager.capabilities(self.pipeline_id)

    def infer(
        self,
        request_id: str,
        inputs: Mapping[str, object],
        *,
        prompt: str | None = None,
        deadline: datetime | None = None,
    ):
        processor_inputs = {
            key: torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()
        }
        return self._manager.infer(
            self.pipeline_id,
            InferenceRequest(request_id=request_id, inputs=processor_inputs, prompt=prompt, deadline=deadline),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        self._manager.reset(self.pipeline_id, deadline)

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        self._manager.cancel(self.pipeline_id, request_id, deadline)

    def health(self):
        return self._manager.health(self.pipeline_id)

    def close(self) -> None:
        self._manager.close()
