"""Success-only result adaptation for the unified runtime boundary."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from .contracts import ExecutionContext, ModelResult, OutcomeEvidence, OutcomeState, RuntimeLatency
from .errors import ExecutionFailure, OutputValidationError


@runtime_checkable
class ResultAdapterProtocol(Protocol):
    """Success-only adapter contract for custom assemblers."""

    def adapt(self, frame: object, **kwargs: object) -> ModelResult: ...


def _call_validator(validator: Callable[..., object], outputs: object, context: ExecutionContext | None) -> object:
    try:
        signature = inspect.signature(validator)
    except (TypeError, ValueError):
        return validator(outputs)
    for args, kwargs in (
        ((outputs,), {}),
        ((outputs, context), {}),
        ((outputs,), {"context": context}),
    ):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return validator(*args, **kwargs)
    return validator(outputs)


class ResultAdapter:
    """Convert a successful internal frame into exactly one ``ModelResult``.

    There is intentionally no error adaptation method.  Exceptions are
    normalized by :class:`ExecutionFailureFactory` at the handle boundary.
    """

    def __init__(
        self,
        validator: Callable[..., object] | None = None,
        *,
        output_validator: Callable[..., object] | None = None,
        required_outputs: tuple[str, ...] = (),
    ) -> None:
        if validator is not None and output_validator is not None:
            raise ValueError("provide validator or output_validator, not both")
        self._validator = validator or output_validator
        self._required_outputs = tuple(required_outputs)
        if any(not isinstance(name, str) or not name for name in self._required_outputs):
            raise ValueError("required output names must be non-empty strings")

    def adapt(
        self,
        frame: object,
        *,
        context: ExecutionContext | None = None,
        evidence: OutcomeEvidence | None = None,
        latency: float | RuntimeLatency | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        if isinstance(frame, ExecutionFailure):
            raise frame
        if isinstance(frame, ModelResult):
            if frame.evidence.state is not OutcomeState.COMPLETED or not frame.evidence.outcome_known:
                raise OutputValidationError("cannot publish a result with incomplete outcome evidence")
            return frame

        outputs = self._outputs_from_frame(frame)
        if self._required_outputs:
            if not isinstance(outputs, Mapping):
                raise OutputValidationError("declared outputs must be a mapping")
            missing = [name for name in self._required_outputs if name not in outputs]
            if missing:
                raise OutputValidationError(f"missing declared outputs: {', '.join(missing)}")
        if self._validator is not None:
            valid = _call_validator(self._validator, outputs, context)
            if valid is False:
                raise OutputValidationError("result output validation returned false")

        frame_metadata = getattr(frame, "metadata", None)
        if isinstance(frame, Mapping) and "metadata" in frame:
            frame_metadata = frame.get("metadata")
        merged_metadata: dict[str, object] = {}
        if isinstance(frame_metadata, Mapping):
            merged_metadata.update(frame_metadata)
        if metadata:
            merged_metadata.update(metadata)

        selected_latency = latency
        if selected_latency is None:
            frame_latency = getattr(frame, "latency", None)
            if isinstance(frame, Mapping) and "latency" in frame:
                frame_latency = frame.get("latency")
            selected_latency = frame_latency if frame_latency is not None else (time.perf_counter() - started) * 1000.0
        selected_evidence = evidence or getattr(frame, "evidence", None)
        if not isinstance(selected_evidence, OutcomeEvidence):
            selected_evidence = OutcomeEvidence.completed("adaptation")
        elif selected_evidence.state is not OutcomeState.COMPLETED or not selected_evidence.outcome_known:
            raise OutputValidationError("cannot adapt a result with incomplete outcome evidence")
        return ModelResult(
            outputs=outputs,
            latency=selected_latency,
            evidence=selected_evidence,
            metadata=merged_metadata,
        )

    @staticmethod
    def _outputs_from_frame(frame: object) -> object:
        if isinstance(frame, Mapping) and "outputs" in frame:
            return frame["outputs"]
        outputs = getattr(frame, "outputs", None)
        if outputs is not None:
            return outputs
        return frame

    __call__ = adapt


SuccessResultAdapter = ResultAdapter
ModelResultAdapter = ResultAdapter


__all__ = ["ModelResultAdapter", "ResultAdapter", "ResultAdapterProtocol", "SuccessResultAdapter"]
