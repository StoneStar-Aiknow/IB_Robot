"""Dependency-free echo reference for generic model-service conformance."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_service.backends import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.generic_runtime import NamedTensorRequest, NamedTensorResult
from inference_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import ModelSession
from inference_service.pipeline import (
    ExecutionError,
    GenericModelPipeline,
    ModelStage,
    PreprocessStage,
    SequentialModelExecutor,
    StageFrame,
)

from .perception_adapter import AdapterIdentity, PerceptionAdapter


class EchoAdapter(PerceptionAdapter):
    """Translate a typed ``value`` field to and from an identity tensor graph."""

    identity = AdapterIdentity(
        family="dummy_echo",
        preprocessing="identity-float32-v1",
        postprocessing="identity-float32-v1",
        supported_deployments=frozenset({"cpu"}),
    )

    def preprocess(self, value: object) -> Mapping[str, np.ndarray]:
        tensor = np.asarray(value, dtype=np.float32)
        if tensor.shape != (2,) or not np.isfinite(tensor).all():
            raise ValueError("dummy echo input must contain two finite values")
        return {"echo.input": tensor}

    def postprocess(self, result: NamedTensorResult, **options) -> list[float]:
        del options
        output = np.asarray(result.outputs["echo.output"], dtype=np.float32)
        if output.shape != (2,):
            raise RuntimeError(f"dummy echo output must have shape (2,), got {output.shape}")
        return output.astype(float).tolist()


class _EchoSession(ModelSession):
    def __init__(self) -> None:
        super().__init__(
            "model-session:dummy-echo",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        del rollback
        if context.model.family != EchoAdapter.identity.family:
            raise ValueError(f"dummy echo requires family {EchoAdapter.identity.family!r}")

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        return {"echo.output": np.asarray(request.inputs["echo.input"], dtype=np.float32).copy()}

    def _close(self) -> None:
        return None


class EchoServicePlugin(ModelServicePlugin):
    """Reference typed-service plugin; its service class is supplied by tests."""

    service_type = "ibrobot_msgs/srv/EchoModel"

    def __init__(self, host, validated, options) -> None:
        del host
        if options:
            raise ValueError(f"dummy echo does not accept runtime options: {sorted(options)}")
        self.adapter = EchoAdapter()
        self.adapter.validate_deployment(validated.deployment_name)
        self.session = _EchoSession()
        context = RuntimeContext(validated)

        class _EchoResultAdapter:
            def adapt(_self, frame: StageFrame) -> list[float]:
                result = frame.values["_model_result"]
                if not isinstance(result, NamedTensorResult):
                    raise TypeError("dummy echo model stage did not return a NamedTensorResult")
                return self.adapter.postprocess(result)

            def adapt_error(_self, error: ExecutionError) -> object:
                if error.cause is not None:
                    raise error.cause
                raise RuntimeError(error.message)

        def preprocess(values):
            result = self.adapter.preprocess(values["value"])
            values.clear()
            return result

        executor = SequentialModelExecutor(
            (
                PreprocessStage(preprocess),
                ModelStage("model", self.session),
            ),
            _EchoResultAdapter(),
            components=(self.session,),
        )
        self.pipeline = GenericModelPipeline("dummy-echo", context, executor)
        self.pipeline.load()

    def handle(self, request, response) -> str:
        response.value = self.pipeline.execute(
            NamedTensorRequest(
                request_id=str(request.request_id),
                inputs={"value": request.value},
            )
        )
        return "echoed 2 values"

    def runtime_status(self) -> PluginRuntimeStatus:
        diagnostics = self.pipeline.diagnostics()
        return PluginRuntimeStatus(
            state=diagnostics.state.value,
            ready=diagnostics.ready,
            failure_reason=diagnostics.executor_health.message or "",
            metadata={
                "runtime_version": "dummy-echo-v1",
                "pipeline_id": diagnostics.pipeline_id,
                "bundle": diagnostics.deployment.bundle,
                "deployment": diagnostics.deployment.deployment,
                "backend": diagnostics.deployment.backend,
                "deployment_fingerprint": diagnostics.deployment.deployment_fingerprint,
            },
        )

    def close(self) -> None:
        self.pipeline.close()


__all__ = ["EchoAdapter", "EchoServicePlugin"]
