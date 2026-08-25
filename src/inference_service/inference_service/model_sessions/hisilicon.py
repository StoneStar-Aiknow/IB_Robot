"""Manifest-bound Hisilicon SD3403 worker model session."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.hisilicon.sd3403_protocol import (
    DEFAULT_GRACEFUL_CLOSE_TIMEOUT,
    SD3403Protocol,
    SD3403ProtocolError,
    SD3403WorkerError,
    SD3403WorkerExitedError,
)
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession

LOGGER = logging.getLogger(__name__)
_ALLOWED_RUNTIME_OPTIONS = frozenset({"perf_enabled", "perf_log_every", "graceful_close_timeout", "force_close"})
ProtocolFactory = Callable[..., SD3403Protocol]


def validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
    """Validate Hisilicon worker options before constructing a session."""

    unknown = sorted(set(options) - _ALLOWED_RUNTIME_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown Hisilicon runtime options: {unknown}", code="invalid_runtime_options")
    perf_enabled = options.get("perf_enabled", False)
    force_close = options.get("force_close", True)
    if type(perf_enabled) is not bool or type(force_close) is not bool:
        raise BackendLoadError(
            "Hisilicon perf_enabled and force_close runtime options must be booleans",
            code="invalid_runtime_options",
        )
    perf_log_every = options.get("perf_log_every", 1)
    if type(perf_log_every) is not int or perf_log_every < 1:
        raise BackendLoadError(
            "Hisilicon perf_log_every runtime option must be a positive integer",
            code="invalid_runtime_options",
        )
    graceful_close_timeout = options.get("graceful_close_timeout", DEFAULT_GRACEFUL_CLOSE_TIMEOUT)
    if type(graceful_close_timeout) not in {int, float} or graceful_close_timeout < 0:
        raise BackendLoadError(
            "Hisilicon graceful_close_timeout runtime option must be a non-negative number",
            code="invalid_runtime_options",
        )
    return {
        "perf_enabled": perf_enabled,
        "perf_log_every": perf_log_every,
        "graceful_close_timeout": float(graceful_close_timeout),
        "force_close": force_close,
    }


class HisiliconModelSession(ModelSession):
    """Execute the manifest policy role through one SD3403 worker process."""

    def __init__(
        self,
        *,
        protocol_factory: ProtocolFactory = SD3403Protocol,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        super().__init__(
            "model-session:hisilicon",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                resource_domain="hisilicon",
                max_in_flight_per_resource_domain=1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
            domains=domains,
        )
        self._protocol_factory = protocol_factory
        self._protocol: SD3403Protocol | None = None
        self._model_path: Path | None = None
        self._worker_path: Path | None = None
        self._action_binding: TensorBinding | None = None
        self._options: dict[str, object] = {}
        self._inference_count = 0
        self._last_metadata: dict[str, object] = {}

    def execution_metadata(self, request_id: str) -> Mapping[str, object]:
        if self._last_metadata.get("request_id") != request_id:
            return {}
        return dict(self._last_metadata)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or context.backend != "hisilicon":
            raise BackendLoadError(
                "HisiliconModelSession requires a compiled hisilicon deployment", code="invalid_deployment"
            )
        if context.interface != "policy" or context.model_type != "act" or context.operation != "predict":
            raise BackendLoadError(
                "HisiliconModelSession requires policy/act/predict",
                code="unsupported_policy_backend_pair",
            )
        if deployment.target.soc != "sd3403" or deployment.target.runtime != "hisilicon-worker":
            raise BackendLoadError(
                "HisiliconModelSession requires target.soc 'sd3403' and target.runtime 'hisilicon-worker'",
                code="incompatible_backend_target",
            )
        if deployment.execution != ("policy",):
            raise BackendLoadError(
                f"HisiliconModelSession requires execution ['policy'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        self._validate_artifacts(deployment)
        options = validate_runtime_options(context.runtime_options)
        model_path = self._require_artifact(context, "policy")
        worker_path = self._require_artifact(context, "worker")
        if not os.access(worker_path, os.X_OK):
            raise BackendLoadError(
                f"Hisilicon worker artifact is not executable: {worker_path}", code="worker_not_executable"
            )

        bindings = deployment.bindings["policy"]
        input_indices = [binding.index for binding in bindings.inputs]
        if any(index is None for index in input_indices) or sorted(input_indices) != list(range(len(input_indices))):
            raise BackendLoadError(
                "Hisilicon worker input bindings require explicit contiguous indices",
                code="invalid_input_bindings",
            )
        unsupported_dtypes = sorted({binding.dtype for binding in bindings.inputs if binding.dtype != "float32"})
        if unsupported_dtypes:
            raise BackendLoadError(
                f"Hisilicon worker input bindings must use float32, got {unsupported_dtypes}",
                code="invalid_input_bindings",
            )
        if any(binding.index is None for binding in bindings.outputs):
            raise BackendLoadError(
                "Hisilicon worker output bindings require explicit runtime indices",
                code="invalid_output_bindings",
            )
        action_bindings = [binding for binding in bindings.outputs if binding.semantic == "action"]
        if len(action_bindings) != 1 or action_bindings[0].index is None:
            raise BackendLoadError(
                "Hisilicon worker requires exactly one action output binding with an explicit runtime index",
                code="invalid_output_bindings",
            )

        protocol = self._create_protocol(worker_path, model_path, options)
        rollback.defer(protocol.close)
        protocol.start()
        self._protocol = protocol
        self._model_path = model_path
        self._worker_path = worker_path
        self._action_binding = action_bindings[0]
        self._options = options

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        return self._execute_role("policy", request.inputs, request)

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        protocol = self._protocol
        action_binding = self._action_binding
        if protocol is None or action_binding is None:
            raise BackendInferenceError("Hisilicon model session is not loaded", code="runtime_not_loaded")
        if role != "policy":
            raise BackendInferenceError(f"unknown Hisilicon execution role {role!r}", code="unknown_execution_role")
        deployment = self._require_context().deployment
        ordered_inputs = tuple(
            inputs[binding.semantic]
            for binding in sorted(deployment.bindings[role].inputs, key=lambda item: item.index)
        )
        started = time.perf_counter()
        try:
            response = protocol.execute(ordered_inputs)
        except SD3403WorkerExitedError as exc:
            raise BackendInferenceError(str(exc), code="worker_exited", recoverable=True) from exc
        except SD3403WorkerError as exc:
            raise BackendInferenceError(str(exc), code="worker_inference_failed") from exc
        except SD3403ProtocolError as exc:
            raise BackendInferenceError(str(exc), code="worker_protocol_error") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            action = response.outputs[int(action_binding.index)]
        except KeyError as exc:
            raise BackendInferenceError(
                f"worker response is missing manifest action output index {action_binding.index}; "
                f"available indices: {sorted(response.outputs)}",
                code="missing_action_output",
            ) from exc
        self._inference_count += 1
        if bool(self._options["perf_enabled"]) and self._inference_count % int(self._options["perf_log_every"]) == 0:
            LOGGER.info(
                "Hisilicon inference request_id=%s worker_request_id=%s e2e_ms=%.3f worker_ms=%.3f",
                request.request_id,
                response.request_id,
                latency_ms,
                response.worker_latency_us / 1000.0,
            )
        context = self._require_context()
        self._last_metadata = {
            "request_id": request.request_id,
            "worker_request_id": response.request_id,
            "worker_latency_ms": response.worker_latency_us / 1000.0,
            "worker_model_load_ms": protocol.model_load_ms,
            "protocol": "sd3403-v1",
            "target_soc": context.target.soc if context.target is not None else None,
        }
        return {"action": action}

    def _recover(self) -> None:
        if self._worker_path is None or self._model_path is None:
            raise BackendInferenceError("Hisilicon model session has no worker to recover", code="runtime_not_loaded")
        previous = self._protocol
        self._protocol = None
        if previous is not None:
            previous.close()
        replacement = self._create_protocol(self._worker_path, self._model_path, self._options)
        try:
            replacement.start()
        except Exception:
            replacement.close()
            raise
        self._protocol = replacement

    def _close(self) -> None:
        protocol = self._protocol
        self._protocol = None
        self._model_path = None
        self._worker_path = None
        self._action_binding = None
        self._options = {}
        self._last_metadata = {}
        if protocol is not None:
            protocol.close()

    def _create_protocol(self, worker_path: Path, model_path: Path, options: Mapping[str, object]) -> SD3403Protocol:
        return self._protocol_factory(
            worker_path,
            model_path,
            graceful_close_timeout=float(options["graceful_close_timeout"]),
            force_close=bool(options["force_close"]),
        )

    @staticmethod
    def _validate_artifacts(deployment: CompiledDeployment) -> None:
        missing = sorted({"policy", "worker"} - set(deployment.artifacts))
        if missing:
            raise BackendLoadError(
                f"Hisilicon deployment is missing required artifact roles: {missing}", code="missing_artifact_role"
            )
        if deployment.artifacts["policy"].format != "om":
            raise BackendLoadError("Hisilicon policy artifact format must be 'om'", code="invalid_artifact_format")
        if deployment.artifacts["worker"].format != "executable":
            raise BackendLoadError(
                "Hisilicon worker artifact format must be 'executable'", code="invalid_artifact_format"
            )

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        path = context.resolved_artifacts.get(role)
        if path is None:
            raise BackendLoadError(
                f"Hisilicon deployment is missing artifact role {role!r}", code="missing_artifact_role"
            )
        if not path.is_file():
            raise BackendLoadError(
                f"Hisilicon artifact {role!r} is not a regular file: {path}", code="invalid_artifact"
            )
        return path
