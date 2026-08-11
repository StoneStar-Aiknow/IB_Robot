"""Concrete ordered executor built from manifest-aware runtime stages."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from inference_service.backends.types import BackendHealth, BackendState, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pipeline.runtime_core import ExecutionControl, ExecutionError, ModelExecutor, StageFrame
from inference_service.pipeline.stages import InferenceStage, ModelStage, ResultAdapter


class SequentialModelExecutor(ModelExecutor):
    """Run a fixed stage sequence while delegating resources to owned components."""

    def __init__(
        self,
        stages: Iterable[InferenceStage],
        result_adapter: ResultAdapter,
        *,
        components: Iterable[object] = (),
        execution_plan=None,
    ) -> None:
        self._stages = tuple(stages)
        if not self._stages:
            raise ValueError("sequential executor requires at least one stage")
        self._result_adapter = result_adapter
        self._components = self._unique(tuple(components))
        self._execution_plan = execution_plan
        self._context: RuntimeContext | None = None

    def load(self, context: RuntimeContext) -> None:
        self._context = context
        loaded: list[object] = []
        try:
            for component in self._components:
                load = getattr(component, "load", None)
                if callable(load):
                    load(context)
                    loaded.append(component)
        except Exception:
            for component in reversed(loaded):
                close = getattr(component, "close", None)
                if callable(close):
                    close()
            raise

    def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> object:
        if not isinstance(request, NamedTensorRequest):
            raise TypeError("SequentialModelExecutor requires a NamedTensorRequest")
        frame = StageFrame(request, execution_plan=self._execution_plan, values=request.inputs, control=control)
        try:
            first_model = next(
                (index for index, stage in enumerate(self._stages) if isinstance(stage, ModelStage)),
                None,
            )
            if first_model is None:
                self._open_session_executions(frame, request, deadline)
                for stage in self._stages:
                    stage.execute(frame, deadline=deadline)
            else:
                for stage in self._stages[:first_model]:
                    stage.execute(frame, deadline=deadline)
                if frame.execution_frame is not None:
                    self._open_session_executions(frame, request, deadline)
                for stage in self._stages[first_model:]:
                    stage.execute(frame, deadline=deadline)
            return self._result_adapter.adapt(frame)
        finally:
            frame.close()

    def _open_session_executions(
        self, frame: StageFrame, request: NamedTensorRequest, deadline: datetime | None
    ) -> None:
        for component in self._components:
            frame.open_session_execution(component, request, deadline)

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        supported = False
        for component in self._components:
            cancel = getattr(component, "cancel", None)
            capabilities = getattr(component, "capabilities", None)
            if not callable(cancel) or not getattr(capabilities, "supports_cancellation", False):
                continue
            supported = True
            cancel(request_id, deadline=deadline)
        if not supported:
            from inference_service.backends import BackendCapabilityError

            raise BackendCapabilityError("executor does not support cancellation", capability="cancellation")

    def adapt_error(self, error: ExecutionError) -> object:
        return self._result_adapter.adapt_error(error)

    def health(self) -> BackendHealth:
        healths = [component.health() for component in self._components if callable(getattr(component, "health", None))]
        if not healths:
            return BackendHealth(state=BackendState.READY, ready=True)
        failed = next((health for health in healths if not health.ready), None)
        if failed is not None:
            return BackendHealth(
                state=failed.state,
                ready=False,
                reason_code=failed.reason_code,
                message=failed.message,
                recoverable=failed.recoverable,
                last_successful_inference_time=failed.last_successful_inference_time,
                failure_count=sum(health.failure_count for health in healths),
            )
        latest = max(
            (health.last_successful_inference_time for health in healths if health.last_successful_inference_time),
            default=None,
        )
        return BackendHealth(
            state=BackendState.READY,
            ready=True,
            last_successful_inference_time=latest,
            failure_count=sum(health.failure_count for health in healths),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        for component in self._components:
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset(deadline=deadline)

    def close(self) -> None:
        errors: list[Exception] = []
        for component in reversed(self._components):
            close = getattr(component, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    @staticmethod
    def _unique(components: tuple[object, ...]) -> tuple[object, ...]:
        unique: list[object] = []
        seen: set[int] = set()
        for component in components:
            if id(component) not in seen:
                seen.add(id(component))
                unique.append(component)
        return tuple(unique)
