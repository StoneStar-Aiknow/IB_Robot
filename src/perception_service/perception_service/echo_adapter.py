"""Dependency-free echo reference for generic model-service conformance."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_service.backends import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.generic_runtime import NamedTensorRequest, NamedTensorResult
from inference_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import ModelSession

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
        self.session.load(RuntimeContext(validated))

    def handle(self, request, response) -> str:
        result = self.session.infer(
            NamedTensorRequest(
                request_id=str(request.request_id),
                inputs=self.adapter.preprocess(request.value),
            )
        )
        response.value = self.adapter.postprocess(result)
        return "echoed 2 values"

    def runtime_status(self) -> PluginRuntimeStatus:
        health = self.session.health()
        return PluginRuntimeStatus(
            state=health.state.value,
            ready=health.ready,
            failure_reason=health.message or "",
            metadata={"runtime_version": "dummy-echo-v1"},
        )

    def close(self) -> None:
        self.session.close()


__all__ = ["EchoAdapter", "EchoServicePlugin"]
