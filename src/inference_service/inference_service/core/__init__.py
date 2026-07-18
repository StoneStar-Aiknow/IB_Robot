"""Dependency-light public facade for unified inference pipelines."""

from inference_service.core.pure_inference_engine import (
    InferenceResult,
    PureInferenceEngine,
    resolve_device,
)

__all__ = [
    "InferenceResult",
    "PureInferenceEngine",
    "resolve_device",
]
