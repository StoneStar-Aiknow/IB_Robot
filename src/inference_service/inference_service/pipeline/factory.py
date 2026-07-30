"""Unified construction of validated backend and pipeline instances."""

from __future__ import annotations

from collections.abc import Mapping

from inference_manifest import CompiledDeployment, ValidatedManifest
from inference_service.backends import BACKEND_REGISTRY, BackendRegistry, RuntimeContext
from inference_service.codecs import create_policy_codec
from inference_service.pipeline.manager import InferencePipelineManager
from inference_service.pipeline.processors import create_lerobot_processor_views
from inference_service.pipeline.runtime import InferencePipeline


def create_inference_pipeline(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry = BACKEND_REGISTRY,
) -> InferencePipeline:
    """Create one pipeline exclusively from a validated manifest and registry."""

    context = RuntimeContext(
        validated_manifest,
        runtime_options=runtime_options or {},
        priority_scheduling=priority_scheduling,
    )
    backend = registry.create(context)
    preprocessor = None
    postprocessor = None
    codec = None
    try:
        if isinstance(context.deployment, CompiledDeployment):
            preprocessor, postprocessor = create_lerobot_processor_views()
            codec = create_policy_codec(context.policy)
        return InferencePipeline(
            pipeline_id,
            context,
            backend,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            codec=codec,
            request_timeout=request_timeout,
            default_task=default_task,
            execution_mode=execution_mode,
        )
    except Exception:
        backend.close()
        raise


def create_pipeline_manager(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry = BACKEND_REGISTRY,
) -> InferencePipelineManager:
    pipeline = create_inference_pipeline(
        pipeline_id,
        validated_manifest,
        request_timeout=request_timeout,
        default_task=default_task,
        execution_mode=execution_mode,
        runtime_options=runtime_options,
        priority_scheduling=priority_scheduling,
        registry=registry,
    )
    manager = InferencePipelineManager((pipeline,))
    manager.start()
    return manager
