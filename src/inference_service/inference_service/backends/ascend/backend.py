"""Manifest-driven Ascend ACL backend for ACT OM deployments.

PI0.5 Ascend execution now runs through ``AscendOmModelSession`` and the shared
``IterativeStage``; this backend retains only the ACT (single policy role)
control flow so the compiled ACL resource boundary is not duplicated.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.ascend.acl_runtime import (
    ACL_RUNTIME_MANAGER,
    AclPriorityStreamPool,
    AclRuntimeLease,
    AclRuntimeManager,
)
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel
from inference_service.backends.errors import BackendAdmissionError, BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendPriorityMapping,
    BackendResult,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import BoundInputs, ExecutionPlan

_ALLOWED_RUNTIME_OPTIONS = frozenset({"device_id", "acl_config_path"})


class AscendBackend(LifecycleBackend):
    """Own ACL runtime state and execute manifest-declared ACT OM roles."""

    def __init__(
        self,
        device_id: int,
        *,
        priority_scheduling: bool = False,
        runtime_manager: AclRuntimeManager = ACL_RUNTIME_MANAGER,
    ) -> None:
        if not isinstance(priority_scheduling, bool):
            raise TypeError("priority_scheduling must be a bool")
        super().__init__(
            "ascend",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                hardware_resource_id=f"ascend:{device_id}" if priority_scheduling else None,
                resource_domain=None if priority_scheduling else f"ascend:{device_id}",
                max_in_flight_per_resource_domain=None if priority_scheduling else 1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._device_id = device_id
        self._priority_scheduling = priority_scheduling
        self._runtime_manager = runtime_manager
        self._lease: AclRuntimeLease | None = None
        self._priority_streams: AclPriorityStreamPool | None = None
        self._models: dict[str, AclModel] = {}
        self._shared_buffers: list[AclDeviceBuffer] = []
        self._context: RuntimeContext | None = None
        self._options: dict[str, object] = {}

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        if context.priority_scheduling != self._priority_scheduling:
            raise BackendLoadError(
                "Ascend backend priority mode differs from its runtime context",
                code="deployment_context_mismatch",
            )
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise BackendLoadError("AscendBackend requires a compiled ascend deployment", code="invalid_deployment")
        if context.policy.policy_type != "act":
            raise BackendLoadError(
                f"AscendBackend no longer hosts {context.policy.policy_type!r}; "
                "PI0.5 runs through AscendOmModelSession",
                code="unsupported_policy_backend_pair",
            )
        options = self._validate_runtime_options(context.runtime_options)
        if int(options["device_id"]) != self._device_id:
            raise BackendLoadError(
                "Ascend backend device_id changed after construction", code="deployment_context_mismatch"
            )
        if any(deployment.artifacts[role].format != "om" for role in deployment.execution):
            raise BackendLoadError("Ascend execution artifacts must use format 'om'", code="invalid_artifact_format")
        if not (deployment.target.runtime.startswith("acl") or deployment.target.runtime.startswith("ascend")):
            raise BackendLoadError(
                f"Ascend target.runtime {deployment.target.runtime!r} is not in the ACL runtime family",
                code="incompatible_backend_target",
            )

        if deployment.execution != ("policy",):
            raise BackendLoadError(
                f"Ascend ACT requires execution ['policy'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        if any(link.producer_binding == "input" for link in deployment.device_links):
            raise BackendLoadError(
                "Ascend does not support input-sourced device links",
                code="unsupported_device_link_source",
            )

        lease = self._runtime_manager.acquire(self._device_id, options["acl_config_path"])
        rollback.defer(lease.close)
        priority_streams = AclPriorityStreamPool.create(lease) if self._priority_scheduling else None
        if priority_streams is not None:
            rollback.defer(priority_streams.close)
        models: dict[str, AclModel] = {}
        shared_buffers: list[AclDeviceBuffer] = []

        def cleanup_loaded_resources() -> None:
            errors: list[Exception] = []
            for model in reversed(tuple(models.values())):
                try:
                    model.close()
                except Exception as exc:
                    errors.append(exc)
            for shared in reversed(shared_buffers):
                try:
                    lease.acl.rt.free(shared.pointer)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        rollback.defer(cleanup_loaded_resources)
        for role in deployment.execution:
            model = AclModel(lease, role, self._require_artifact(context, role), deployment.bindings[role])
            model.load_descriptor()
            models[role] = model

        input_overrides: dict[str, dict[int, AclDeviceBuffer]] = {role: {} for role in deployment.execution}
        output_overrides: dict[str, dict[int, AclDeviceBuffer]] = {role: {} for role in deployment.execution}
        shared_outputs: dict[tuple[str, int], AclDeviceBuffer] = {}
        for link in deployment.device_links:
            producer_binding = self._binding_for_semantic(
                deployment.bindings[link.producer].outputs, link.semantic, "producer output"
            )
            consumer_binding = self._binding_for_semantic(
                deployment.bindings[link.consumer].inputs, link.semantic, "consumer input"
            )
            if producer_binding.index is None or consumer_binding.index is None:
                raise BackendLoadError(
                    f"Ascend device link {link.semantic!r} requires explicit runtime indices",
                    code="invalid_device_link",
                )
            producer_desc = models[link.producer].output_descriptors[int(producer_binding.index)]
            consumer_desc = models[link.consumer].input_descriptors[int(consumer_binding.index)]
            if producer_desc.size != consumer_desc.size:
                raise BackendLoadError(
                    f"Ascend device link {link.semantic!r} size differs between producer "
                    f"({producer_desc.size}) and consumer ({consumer_desc.size})",
                    code="device_link_size_mismatch",
                )
            producer_key = (link.producer, int(producer_binding.index))
            shared = shared_outputs.get(producer_key)
            if shared is None:
                pointer, ret = lease.acl.rt.malloc(producer_desc.size, 0)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.malloc(device link {link.semantic}) failed with ACL error code {ret}")
                shared = AclDeviceBuffer(pointer=pointer, size=producer_desc.size)
                shared_outputs[producer_key] = shared
                shared_buffers.append(shared)
            output_overrides[link.producer][int(producer_binding.index)] = shared
            input_overrides[link.consumer][int(consumer_binding.index)] = shared
        for role in deployment.execution:
            models[role].prepare_datasets(
                input_overrides=input_overrides[role],
                output_overrides=output_overrides[role],
            )

        self._lease = lease
        self._priority_streams = priority_streams
        self._models = models
        self._shared_buffers = shared_buffers
        self._context = context
        self._options = options
        self._update_loaded_capabilities(
            priority_mapping=(
                BackendPriorityMapping(tuple(range(priority_streams.level_count)))
                if priority_streams is not None
                else None
            )
        )

    def _infer(self, request: InferenceRequest) -> BackendResult:
        context = self._context
        if context is None:
            raise BackendInferenceError("AscendBackend is not fully loaded", code="runtime_not_loaded")
        plan, role_inputs = self._request_execution(request)
        stream, native_priority = self._request_stream(request)
        started = time.perf_counter()
        outputs = self._infer_act(plan, role_inputs, stream=stream)
        latency_ms = (time.perf_counter() - started) * 1000.0
        action = self._raw_action(outputs, plan)
        metadata = {
            "request_id": request.request_id,
            "device_id": self._device_id,
            "target_soc": context.target.soc if context.target is not None else None,
            "deployment_name": context.deployment_name,
            "deployment_fingerprint": context.deployment_fingerprint,
        }
        if native_priority is not None:
            metadata["hardware_priority"] = native_priority
        return BackendResult(
            action=outputs,
            actual_chunk_size=self._chunk_size(action),
            backend_latency_ms=latency_ms,
            metadata=metadata,
        )

    def _close(self) -> None:
        lease = self._lease
        priority_streams = self._priority_streams
        models = self._models
        shared_buffers = self._shared_buffers
        self._lease = None
        self._priority_streams = None
        self._models = {}
        self._shared_buffers = []
        self._context = None
        self._options = {}
        errors: list[Exception] = []
        if priority_streams is not None:
            try:
                priority_streams.close()
            except Exception as exc:
                errors.append(exc)
        for model in reversed(tuple(models.values())):
            try:
                model.close()
            except Exception as exc:
                errors.append(exc)
        if lease is not None:
            for shared in reversed(shared_buffers):
                try:
                    lease.acl.rt.free(shared.pointer)
                except Exception as exc:
                    errors.append(exc)
        if lease is not None:
            try:
                lease.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _infer_act(
        self,
        plan: ExecutionPlan,
        role_inputs: Mapping[str, BoundInputs],
        *,
        stream: object | None,
    ) -> dict[int, np.ndarray]:
        if plan.role_names != ("policy",):
            raise BackendInferenceError("Ascend ACT requires one policy role", code="invalid_request")
        return self._models["policy"].execute(role_inputs["policy"], stream=stream)

    def _request_stream(self, request: InferenceRequest) -> tuple[object | None, int | None]:
        streams = self._priority_streams
        if streams is None:
            if request.priority != 0:
                raise BackendAdmissionError(
                    "non-zero Ascend priority requires scheduler-enabled priority streams",
                    code="hardware_priority_unavailable",
                )
            return None, None
        mapping = self.capabilities.priority_mapping
        if mapping is None:
            raise BackendAdmissionError("Ascend priority mapping is unavailable", code="hardware_priority_unavailable")
        try:
            native_priority = mapping.map_generic(request.priority)
        except ValueError as exc:
            raise BackendAdmissionError(str(exc), code="unsupported_priority") from exc
        if not streams.supports(native_priority):
            raise BackendAdmissionError(
                f"Ascend hardware does not expose native priority {native_priority}",
                code="hardware_priority_unavailable",
            )
        return streams.select(native_priority), native_priority

    @staticmethod
    def _request_execution(request: InferenceRequest) -> tuple[ExecutionPlan, Mapping[str, BoundInputs]]:
        plan = request.inputs.get("execution_plan")
        role_inputs = request.inputs.get("role_inputs")
        if not isinstance(plan, ExecutionPlan):
            raise BackendInferenceError("AscendBackend request is missing execution_plan", code="invalid_request")
        if not isinstance(role_inputs, Mapping):
            raise BackendInferenceError("AscendBackend request is missing role_inputs", code="invalid_request")
        for role in plan.role_names:
            if not isinstance(role_inputs.get(role), BoundInputs):
                raise BackendInferenceError(
                    f"AscendBackend role {role!r} inputs are not bound tensors",
                    code="invalid_request",
                )
        return plan, role_inputs

    @staticmethod
    def _raw_action(outputs: object, plan: ExecutionPlan) -> np.ndarray:
        action_role = next(
            role for role in plan.roles if any(binding.semantic == "action" for binding in role.bindings.outputs)
        )
        binding = next(binding for binding in action_role.bindings.outputs if binding.semantic == "action")
        role_outputs = (
            outputs[action_role.name] if isinstance(outputs, Mapping) and action_role.name in outputs else outputs
        )
        if not isinstance(role_outputs, Mapping) or binding.index not in role_outputs:
            raise BackendInferenceError(
                "Ascend runtime did not return the bound action output", code="missing_action_output"
            )
        action = np.asarray(role_outputs[binding.index])
        if action.ndim == 1 and all(dimension > 0 for dimension in binding.shape):
            expected_size = int(np.prod(binding.shape, dtype=np.int64))
            if action.size == expected_size:
                action = action.reshape(binding.shape)
        return action

    @staticmethod
    def _chunk_size(action: np.ndarray) -> int:
        if action.ndim < 2 or action.shape[-2] < 1:
            raise BackendInferenceError(
                f"Ascend action output has invalid shape {action.shape}",
                code="invalid_action_shape",
            )
        return int(action.shape[-2])

    @staticmethod
    def _binding_for_semantic(bindings: tuple[TensorBinding, ...], semantic: str, description: str) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1:
            raise BackendLoadError(
                f"Ascend deployment requires exactly one {description} binding for {semantic!r}",
                code="invalid_bindings",
            )
        return matches[0]

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        try:
            path = context.resolved_artifacts[role]
        except KeyError as exc:
            raise BackendLoadError(
                f"Ascend deployment is missing artifact role {role!r}", code="missing_artifact_role"
            ) from exc
        if not path.is_file():
            raise BackendLoadError(f"Ascend artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
        return path

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
        unknown = sorted(set(options) - _ALLOWED_RUNTIME_OPTIONS)
        if unknown:
            raise BackendLoadError(f"unknown Ascend runtime options: {unknown}", code="invalid_runtime_options")
        device_id = options.get("device_id", 0)
        if type(device_id) is not int or device_id < 0:
            raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
        acl_config_path = options.get("acl_config_path")
        if acl_config_path is not None and (type(acl_config_path) is not str or not acl_config_path.strip()):
            raise BackendLoadError("Ascend acl_config_path must be a non-empty string", code="invalid_runtime_options")
        return {
            "device_id": device_id,
            "acl_config_path": acl_config_path,
        }


def create_backend(context: RuntimeContext) -> AscendBackend:
    """Lazy registry factory for Ascend ACL execution."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
        raise BackendLoadError("AscendBackend requires a compiled ascend deployment", code="invalid_deployment")
    options = AscendBackend._validate_runtime_options(context.runtime_options)
    return AscendBackend(
        int(options["device_id"]),
        priority_scheduling=context.priority_scheduling,
    )
