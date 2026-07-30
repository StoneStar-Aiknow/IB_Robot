"""Manifest-driven Hisilicon backend for worker-executed OM artifacts."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.hisilicon.sd3403_protocol import (
    DEFAULT_GRACEFUL_CLOSE_TIMEOUT,
    SD3403Protocol,
    SD3403ProtocolError,
    SD3403WorkerError,
    SD3403WorkerExitedError,
)
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendResult, InferenceRequest, RuntimeContext
from inference_service.codecs import BoundInputs, ExecutionPlan

LOGGER = logging.getLogger(__name__)
_ALLOWED_RUNTIME_OPTIONS = frozenset({"perf_enabled", "perf_log_every", "graceful_close_timeout", "force_close"})
ProtocolFactory = Callable[..., SD3403Protocol]


class HisiliconBackend(LifecycleBackend):
    """Execute an ACT OM through the manifest-declared Hisilicon worker."""

    def __init__(
        self,
        *,
        protocol_factory: ProtocolFactory = SD3403Protocol,
        expose_hardware_identity: bool = False,
    ) -> None:
        super().__init__(
            "hisilicon",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                hardware_resource_id="hisilicon:0" if expose_hardware_identity else None,
            ),
        )
        self._protocol_factory = protocol_factory
        self._protocol: SD3403Protocol | None = None
        self._context: RuntimeContext | None = None
        self._model_path: Path | None = None
        self._worker_path: Path | None = None
        self._action_binding: TensorBinding | None = None
        self._options: dict[str, object] = {}
        self._inference_count = 0

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hisilicon":
            raise BackendLoadError(
                "HisiliconBackend requires a compiled hisilicon deployment", code="invalid_deployment"
            )
        if context.policy.policy_type != "act":
            raise BackendLoadError(
                f"HisiliconBackend does not support policy family {context.policy.policy_type!r}",
                code="unsupported_policy_backend_pair",
            )
        if deployment.target.soc != "sd3403" or deployment.target.runtime != "hisilicon-worker":
            raise BackendLoadError(
                "HisiliconBackend requires target.soc 'sd3403' and target.runtime 'hisilicon-worker'",
                code="incompatible_backend_target",
            )
        if deployment.execution != ("policy",):
            raise BackendLoadError(
                f"HisiliconBackend requires execution ['policy'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        self._validate_artifacts(deployment)
        options = self._validate_runtime_options(context.runtime_options)
        model_path = self._require_artifact(context, "policy")
        worker_path = self._require_artifact(context, "worker")
        if not os.access(worker_path, os.X_OK):
            raise BackendLoadError(
                f"Hisilicon worker artifact is not executable: {worker_path}",
                code="worker_not_executable",
            )

        bindings = deployment.bindings["policy"]
        if any(binding.index is None for binding in bindings.inputs):
            raise BackendLoadError(
                "Hisilicon worker input bindings require explicit contiguous indices",
                code="invalid_input_bindings",
            )
        unsupported_input_dtypes = sorted({binding.dtype for binding in bindings.inputs if binding.dtype != "float32"})
        if unsupported_input_dtypes:
            raise BackendLoadError(
                f"Hisilicon worker input bindings must use float32, got {unsupported_input_dtypes}",
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
        self._context = context
        self._model_path = model_path
        self._worker_path = worker_path
        self._action_binding = action_bindings[0]
        self._options = options

    def _infer(self, request: InferenceRequest) -> BackendResult:
        protocol = self._protocol
        context = self._context
        action_binding = self._action_binding
        if protocol is None or context is None or action_binding is None:
            raise BackendInferenceError("HisiliconBackend is not fully loaded", code="runtime_not_loaded")
        inputs = self._bound_inputs(request)
        started = time.perf_counter()
        try:
            response = protocol.execute(inputs.ordered_values)
        except SD3403WorkerExitedError as exc:
            raise BackendInferenceError(str(exc), code="worker_exited", recoverable=True) from exc
        except SD3403WorkerError as exc:
            raise BackendInferenceError(str(exc), code="worker_inference_failed") from exc
        except SD3403ProtocolError as exc:
            raise BackendInferenceError(str(exc), code="worker_protocol_error") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            raw_action = response.outputs[int(action_binding.index)]
        except KeyError as exc:
            raise BackendInferenceError(
                f"worker response is missing manifest action output index {action_binding.index}; "
                f"available indices: {sorted(response.outputs)}",
                code="missing_action_output",
            ) from exc
        actual_chunk_size = self._chunk_size(
            raw_action, action_binding, context.policy.output_features["action"].shape[-1]
        )
        self._inference_count += 1
        if bool(self._options["perf_enabled"]) and self._inference_count % int(self._options["perf_log_every"]) == 0:
            LOGGER.info(
                "Hisilicon inference request_id=%s worker_request_id=%s e2e_ms=%.3f worker_ms=%.3f",
                request.request_id,
                response.request_id,
                latency_ms,
                response.worker_latency_us / 1000.0,
            )
        return BackendResult(
            action=response.outputs,
            actual_chunk_size=actual_chunk_size,
            backend_latency_ms=latency_ms,
            metadata={
                "request_id": request.request_id,
                "worker_request_id": response.request_id,
                "worker_latency_ms": response.worker_latency_us / 1000.0,
                "worker_model_load_ms": protocol.model_load_ms,
                "protocol": "sd3403-v1",
                "target_soc": context.target.soc if context.target is not None else None,
                "deployment_name": context.deployment_name,
                "deployment_fingerprint": context.deployment_fingerprint,
            },
        )

    def _recover(self) -> None:
        if self._worker_path is None or self._model_path is None:
            raise BackendInferenceError("HisiliconBackend has no loaded worker to recover", code="runtime_not_loaded")
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
        self._context = None
        self._model_path = None
        self._worker_path = None
        self._action_binding = None
        self._options = {}
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
        required = {"policy", "worker"}
        missing = sorted(required - set(deployment.artifacts))
        if missing:
            raise BackendLoadError(
                f"Hisilicon deployment is missing required artifact roles: {missing}",
                code="missing_artifact_role",
            )
        if deployment.artifacts["policy"].format != "om":
            raise BackendLoadError("Hisilicon policy artifact format must be 'om'", code="invalid_artifact_format")
        if deployment.artifacts["worker"].format != "executable":
            raise BackendLoadError(
                "Hisilicon worker artifact format must be 'executable'",
                code="invalid_artifact_format",
            )

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        try:
            path = context.resolved_artifacts[role]
        except KeyError as exc:
            raise BackendLoadError(
                f"Hisilicon deployment is missing artifact role {role!r}", code="missing_artifact_role"
            ) from exc
        if not path.is_file():
            raise BackendLoadError(
                f"Hisilicon artifact {role!r} is not a regular file: {path}", code="invalid_artifact"
            )
        return path

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
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

    @staticmethod
    def _bound_inputs(request: InferenceRequest) -> BoundInputs:
        plan = request.inputs.get("execution_plan")
        role_inputs = request.inputs.get("role_inputs")
        if not isinstance(plan, ExecutionPlan) or plan.role_names != ("policy",):
            raise BackendInferenceError(
                "HisiliconBackend requires a single policy execution plan", code="invalid_request"
            )
        if not isinstance(role_inputs, Mapping):
            raise BackendInferenceError("HisiliconBackend request is missing role_inputs", code="invalid_request")
        bound = role_inputs.get("policy")
        if not isinstance(bound, BoundInputs):
            raise BackendInferenceError("HisiliconBackend policy inputs are not bound tensors", code="invalid_request")
        return bound

    @staticmethod
    def _chunk_size(action: object, binding: TensorBinding, action_dimension: int) -> int:
        array = np.asarray(action)
        if array.ndim == 1:
            if all(dimension > 0 for dimension in binding.shape) and int(np.prod(binding.shape)) == array.size:
                array = array.reshape(binding.shape)
            elif array.size % action_dimension == 0:
                return array.size // action_dimension
        if array.ndim < 2 or array.shape[-1] != action_dimension or array.shape[-2] < 1:
            raise BackendInferenceError(
                f"worker action output shape {array.shape} is incompatible with action dimension {action_dimension}",
                code="invalid_action_shape",
            )
        return int(array.shape[-2])


def create_backend(context: RuntimeContext) -> HisiliconBackend:
    """Lazy registry factory for the canonical Hisilicon backend."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hisilicon":
        raise BackendLoadError("HisiliconBackend requires a compiled hisilicon deployment", code="invalid_deployment")
    return HisiliconBackend(expose_hardware_identity=context.priority_scheduling)
