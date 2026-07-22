"""Backend action validation shared by native and compiled pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference_service.pipeline.errors import PipelineValidationError


@dataclass(frozen=True)
class ActionValidation:
    shape: tuple[int, ...]
    action_dimension: int
    chunk_size: int


def _as_numpy(action: object, pipeline_id: str, phase: str) -> np.ndarray:
    candidate = action
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} {phase} action cannot be converted to an array",
            pipeline_id=pipeline_id,
            details={"phase": phase},
        ) from exc


def validate_action_output(
    action: object,
    *,
    actual_chunk_size: int,
    action_dimension: int,
    pipeline_id: str,
    phase: str = "backend",
) -> ActionValidation:
    """Validate finite action values and the supported batch/chunk layouts."""

    if type(actual_chunk_size) is not int or actual_chunk_size < 1:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} reported invalid actual_chunk_size {actual_chunk_size!r}; "
            "expected a positive integer",
            pipeline_id=pipeline_id,
            details={"phase": phase, "actual_chunk_size": actual_chunk_size},
        )
    if action_dimension < 1:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} has invalid configured action dimension {action_dimension}",
            pipeline_id=pipeline_id,
            details={"phase": phase, "action_dimension": action_dimension},
        )

    array = _as_numpy(action, pipeline_id, phase)
    shape = tuple(array.shape)
    if array.ndim == 1:
        chunk_size = 1
    elif array.ndim == 2:
        chunk_size = shape[0]
    elif array.ndim == 3:
        if shape[0] != 1:
            raise PipelineValidationError(
                f"pipeline {pipeline_id!r} {phase} action batch dimension must be one, got shape {shape}",
                pipeline_id=pipeline_id,
                details={"phase": phase, "shape": shape},
            )
        chunk_size = shape[1]
    else:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} {phase} action rank must be 1, 2, or 3, got shape {shape}",
            pipeline_id=pipeline_id,
            details={"phase": phase, "shape": shape},
        )

    if shape[-1] != action_dimension:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} {phase} action dimension must be {action_dimension}, got shape {shape}",
            pipeline_id=pipeline_id,
            details={"phase": phase, "shape": shape, "action_dimension": action_dimension},
        )
    if chunk_size < 1 or actual_chunk_size != chunk_size:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} reported actual_chunk_size {actual_chunk_size}, "
            f"but {phase} action shape {shape} contains chunk size {chunk_size}",
            pipeline_id=pipeline_id,
            details={
                "phase": phase,
                "shape": shape,
                "actual_chunk_size": actual_chunk_size,
                "returned_chunk_size": chunk_size,
            },
        )
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} {phase} action must contain numeric values",
            pipeline_id=pipeline_id,
            details={"phase": phase, "shape": shape},
        ) from exc
    if not finite:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} {phase} action contains non-finite values",
            pipeline_id=pipeline_id,
            details={"phase": phase, "shape": shape},
        )
    return ActionValidation(shape=shape, action_dimension=action_dimension, chunk_size=chunk_size)
