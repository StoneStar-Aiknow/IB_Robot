"""Cloud request execution with immediate structured distributed results."""

from __future__ import annotations

from dataclasses import replace

from inference_service.distributed.runtime import CloudBackendRuntime
from inference_service.distributed.session import CloudSession, DistributedProtocolError
from inference_service.distributed.types import (
    DistributedRequest,
    DistributedResult,
    Operation,
    PipelineIdentity,
    PipelineStatus,
    StructuredError,
    structured_error_from_exception,
)


class DistributedCloudService:
    def __init__(
        self,
        identity: PipelineIdentity,
        runtime: CloudBackendRuntime | None,
        *,
        startup_error: StructuredError | None = None,
    ) -> None:
        if runtime is None and startup_error is None:
            raise ValueError("startup_error is required when the cloud runtime is unavailable")
        self.identity = identity
        self.runtime = runtime
        self.startup_error = startup_error
        self.session = CloudSession(identity)

    def observe_edge(self, status: PipelineStatus) -> PipelineStatus:
        if self.runtime is None:
            self.session.observe_edge(status, backend_ready=False)
            return self.status()
        health = self.runtime.health()
        self.session.observe_edge(status, backend_ready=health.ready)
        return self.status()

    def status(self) -> PipelineStatus:
        if self.runtime is None:
            status = self.session.status(
                backend_ready=False,
                backend_state="failed",
                reset_supported=False,
                cancellation_supported=False,
            )
            return replace(status, error=self.startup_error)
        health = self.runtime.health()
        capabilities = self.runtime.capabilities
        return self.session.status(
            backend_ready=health.ready,
            backend_state=health.state.value,
            reset_supported=capabilities.resettable,
            cancellation_supported=capabilities.supports_cancellation,
        )

    def handle(self, request: DistributedRequest) -> DistributedResult:
        if self.runtime is None:
            return DistributedResult(
                operation=request.operation,
                pipeline_id=self.identity.pipeline_id,
                request_id=request.request_id,
                session_id=request.session_id,
                session_generation=request.session_generation,
                deployment_fingerprint=self.identity.deployment_fingerprint,
                success=False,
                backend_ready=False,
                backend_state="failed",
                target_request_id=request.target_request_id,
                error=self.startup_error,
            )
        try:
            self.session.validate_request(request)
            if request.operation is Operation.INFER:
                pipeline_result = self.runtime.infer(
                    request.request_id,
                    request.inputs,
                    prompt=request.prompt,
                    deadline=request.deadline,
                )
                action = pipeline_result.action
                chunk_size = pipeline_result.actual_chunk_size
                latency_ms = pipeline_result.backend_latency_ms
            elif request.operation is Operation.RESET:
                self.runtime.reset()
                action = None
                chunk_size = 0
                latency_ms = 0.0
            elif request.operation is Operation.CANCEL:
                self.runtime.cancel(request.target_request_id)
                action = None
                chunk_size = 0
                latency_ms = 0.0
            else:
                raise ValueError(f"unsupported distributed operation {request.operation!r}")
        except Exception as exc:
            stage = "routing" if isinstance(exc, DistributedProtocolError) else self._operation_stage(request.operation)
            error = (
                exc.error if isinstance(exc, DistributedProtocolError) else structured_error_from_exception(exc, stage)
            )
            health = self.runtime.health()
            return DistributedResult(
                operation=request.operation,
                pipeline_id=self.identity.pipeline_id,
                request_id=request.request_id,
                session_id=request.session_id,
                session_generation=request.session_generation,
                deployment_fingerprint=self.identity.deployment_fingerprint,
                success=False,
                backend_ready=health.ready,
                backend_state=health.state.value,
                target_request_id=request.target_request_id,
                error=error,
            )

        health = self.runtime.health()
        return DistributedResult(
            operation=request.operation,
            pipeline_id=self.identity.pipeline_id,
            request_id=request.request_id,
            session_id=request.session_id,
            session_generation=request.session_generation,
            deployment_fingerprint=self.identity.deployment_fingerprint,
            success=True,
            action=action,
            actual_chunk_size=chunk_size,
            backend_latency_ms=latency_ms,
            backend_ready=health.ready,
            backend_state=health.state.value,
            target_request_id=request.target_request_id,
        )

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()

    @staticmethod
    def _operation_stage(operation: Operation) -> str:
        return {
            Operation.INFER: "backend",
            Operation.RESET: "reset",
            Operation.CANCEL: "cancel",
        }[operation]
