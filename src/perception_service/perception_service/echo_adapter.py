"""Dependency-free echo reference for generic model-service conformance."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_service.backends import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import ModelSession
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    LoadRollback,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    RegistrySet,
    RuntimeAssembly,
    RuntimeProviders,
)

from .perception_adapter import AdapterIdentity, PerceptionAdapter


class EchoAdapter(PerceptionAdapter):
    """Translate a typed ``value`` field to and from an identity tensor graph."""

    identity = AdapterIdentity(
        model_type="dummy_echo",
        preprocessing="identity-float32-v1",
        postprocessing="identity-float32-v1",
        supported_deployments=frozenset({"cpu"}),
        operation="echo",
    )

    def preprocess(self, value: object) -> Mapping[str, np.ndarray]:
        tensor = np.asarray(value, dtype=np.float32)
        if tensor.shape != (2,) or not np.isfinite(tensor).all():
            raise ValueError("dummy echo input must contain two finite values")
        return {"echo.input": tensor}

    def postprocess(self, result: ModelResult, **options) -> list[float]:
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

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        del rollback
        if (context.interface, context.model_type, context.operation) != (
            "tensor_model",
            EchoAdapter.identity.model_type,
            EchoAdapter.identity.operation,
        ):
            raise ValueError("dummy echo requires tensor_model/dummy_echo/echo")

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        return {"echo.output": np.asarray(request.inputs["echo.input"], dtype=np.float32).copy()}

    def _close(self) -> None:
        return None


class EchoServicePlugin(ModelServicePlugin):
    """Reference typed-service plugin; its service class is supplied by tests."""

    service_type = "ibrobot_msgs/srv/EchoModel"

    def __init__(
        self,
        host,
        validated,
        options,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        del host
        if registry_set is None or providers is None:
            raise ValueError("dummy echo requires explicitly injected runtime dependencies")
        if options:
            raise ValueError(f"dummy echo does not accept runtime options: {sorted(options)}")
        self.adapter = EchoAdapter()
        self.validated = validated
        self.adapter.validate_deployment(validated.deployment_name)
        self.session = _EchoSession()
        context = RuntimeContext(validated, runtime_profile=validated.runtime_profile)
        self._runtime_handle = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=self.session,
                session=self.session,
                execution_contract=ExecutionContract(),
                runtime_id="dummy-echo",
                load_context=context,
            )
        )
        self._runtime_handle.load(context)

    def handle(self, request, response) -> str:
        model_inputs = self.adapter.preprocess(request.value)
        result = self._runtime_handle.execute(
            ModelRequest(model_inputs, metadata={"service_type": self.service_type}),
            ExecutionContext(str(request.request_id)),
        )
        response.value = self.adapter.postprocess(result)
        return "echoed 2 values"

    def runtime_status(self) -> PluginRuntimeStatus:
        diagnostics = self._runtime_handle.diagnostics()
        return PluginRuntimeStatus(
            state=diagnostics.state.value,
            ready=diagnostics.health.ready,
            failure_reason=diagnostics.health.message or "",
            metadata={
                "runtime_version": "dummy-echo-v1",
                "pipeline_id": "dummy-echo",
                "bundle": self.validated.manifest.bundle.name,
                "deployment": self.validated.deployment_name,
                "backend": self.validated.deployment.backend,
                "deployment_fingerprint": self.validated.deployment_fingerprint,
            },
        )

    def close(self) -> None:
        self._runtime_handle.close()


__all__ = ["EchoAdapter", "EchoServicePlugin"]
