"""Generic manifest-bound Ascend OM model session."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import INTERNAL_SEMANTIC_PREFIX, CompiledDeployment, TensorBinding
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.ascend.acl_runtime import (
    ACL_RUNTIME_MANAGER,
    AclPriorityStreamPool,
    AclRuntimeLease,
    AclRuntimeManager,
)
from inference_service.backends.ascend.model import AclModel
from inference_service.backends.errors import (
    BackendAdmissionError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
)
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendPriorityMapping,
    RuntimeContext,
)
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession


class AscendOmModelSession(ModelSession):
    """Execute manifest-declared OM roles using shared ACL leases and model resources."""

    # Subclasses that need extra knobs (a denoising seed, say) widen this rather than
    # reimplementing _load; anything not listed is still rejected at load time.
    allowed_runtime_options: frozenset[str] = frozenset({"device_id", "acl_config_path"})

    def __init__(
        self,
        device_id: int = 0,
        *,
        runtime_manager: AclRuntimeManager = ACL_RUNTIME_MANAGER,
        model_factory=AclModel,
        domains: ResourceDomainAdmissions | None = None,
        priority_scheduling: bool = False,
    ) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        if not isinstance(priority_scheduling, bool):
            raise TypeError("priority_scheduling must be a bool")
        super().__init__(
            "model-session:ascend",
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
            domains=domains,
        )
        self._device_id = device_id
        self._priority_scheduling = priority_scheduling
        self._runtime_manager = runtime_manager
        self._model_factory = model_factory
        self._lease: AclRuntimeLease | None = None
        self._priority_streams: AclPriorityStreamPool | None = None
        self._models: dict[str, AclModel] = {}
        self._linked_inputs: frozenset[tuple[str, str]] = frozenset()

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(None if self._lease is None else self._lease.acl)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        if context.priority_scheduling != self._priority_scheduling:
            raise BackendLoadError(
                "Ascend model-session priority mode differs from its runtime context",
                code="deployment_context_mismatch",
            )
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise BackendLoadError(
                "AscendOmModelSession requires a compiled Ascend deployment", code="invalid_deployment"
            )
        unknown_options = sorted(set(context.runtime_options) - self.allowed_runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown Ascend model-session options: {unknown_options}", code="invalid_runtime_options"
            )
        device_id = context.runtime_options.get("device_id", self._device_id)
        if type(device_id) is not int or device_id != self._device_id:
            raise BackendLoadError("Ascend device_id does not match the session", code="deployment_context_mismatch")
        if any(deployment.artifacts[role].format != "om" for role in deployment.execution):
            raise BackendLoadError("Ascend execution artifacts must use format 'om'", code="invalid_artifact_format")
        if not (deployment.target.runtime.startswith("acl") or deployment.target.runtime.startswith("ascend")):
            raise BackendLoadError(
                f"Ascend target runtime {deployment.target.runtime!r} is not ACL-compatible",
                code="incompatible_backend_target",
            )
        if any(link.producer_binding == "input" for link in deployment.device_links):
            raise BackendLoadError(
                "Ascend model sessions do not support input-sourced device links",
                code="unsupported_device_link_source",
            )
        config_path = context.runtime_options.get("acl_config_path")
        if config_path is not None and (not isinstance(config_path, str) or not config_path.strip()):
            raise BackendLoadError("acl_config_path must be a non-empty string", code="invalid_runtime_options")
        lease = self._runtime_manager.acquire(self._device_id, config_path)
        rollback.defer(lease.close)
        priority_streams = AclPriorityStreamPool.create(lease) if self._priority_scheduling else None
        if priority_streams is not None:
            rollback.defer(priority_streams.close)
        models: dict[str, AclModel] = {}

        def close_models() -> None:
            for model in reversed(tuple(models.values())):
                model.close()

        rollback.defer(close_models)
        for role in deployment.execution:
            path = context.resolved_artifacts.get(role)
            if path is None or not path.is_file():
                raise BackendLoadError(f"Ascend artifact {role!r} is unavailable: {path}", code="invalid_artifact")
            model = self._model_factory(lease, role, path, deployment.bindings[role])
            models[role] = model
            model.load_descriptor()
        linked_inputs = {(link.consumer, link.semantic) for link in deployment.device_links}
        for role in deployment.execution:
            input_overrides = {}
            for link in deployment.device_links:
                if link.consumer != role:
                    continue
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
                producer_buffer = models[link.producer].output_buffer(int(producer_binding.index))
                consumer_descriptor = models[role].input_descriptors[int(consumer_binding.index)]
                if producer_buffer.size != consumer_descriptor.size:
                    raise BackendLoadError(
                        f"Ascend device link {link.semantic!r} size differs between producer "
                        f"({producer_buffer.size}) and consumer ({consumer_descriptor.size})",
                        code="device_link_size_mismatch",
                    )
                input_overrides[int(consumer_binding.index)] = producer_buffer
            models[role].prepare_datasets(input_overrides=input_overrides)
        self._lease = lease
        self._priority_streams = priority_streams
        self._models = models
        self._linked_inputs = frozenset(linked_inputs)
        self._update_loaded_capabilities(
            priority_mapping=(
                BackendPriorityMapping(tuple(range(priority_streams.level_count)))
                if priority_streams is not None
                else None
            )
        )

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        deployment = self._loaded_deployment()
        values = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role_index, role in enumerate(deployment.execution):
            for semantic, output in self._run_role(role_index, role, values, request=request).items():
                if not semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                    public_outputs[semantic] = output
        return public_outputs

    def _loaded_deployment(self) -> CompiledDeployment:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment) or not self._models:
            raise BackendInferenceError("Ascend OM model session is not loaded", code="runtime_not_loaded")
        return deployment

    def _run_role(
        self,
        role_index: int,
        role: str,
        values: dict[str, object],
        *,
        request: NamedTensorRequest,
    ) -> dict[str, np.ndarray]:
        """Execute one manifest role and write its outputs back into ``values``.

        Sessions whose roles are joined by the compiled graph alone walk ``execution``
        straight through, but a host-orchestrated model computes tensors between roles and
        drives them one at a time. Both need identical binding, device-link and
        read-selection behaviour per role, so that behaviour lives here rather than in the
        loop that happens to call it.
        """
        runtime_outputs = self._models[role].execute(
            self._role_inputs(role, values),
            read_outputs=self._role_read_indices(role_index, role),
            stream=self._request_stream(request),
        )
        outputs: dict[str, np.ndarray] = {}
        for binding in self._loaded_deployment().bindings[role].outputs:
            if binding.index is not None and int(binding.index) not in runtime_outputs:
                continue
            output = self._bound_output(role, binding, runtime_outputs)
            values[binding.semantic] = output
            outputs[binding.semantic] = output
        return outputs

    def _role_inputs(self, role: str, values: Mapping[str, object]) -> dict[int, np.ndarray]:
        """Gather one role's runtime inputs, skipping the slots a device link already fills."""
        indexed_inputs: dict[int, np.ndarray] = {}
        for binding in self._loaded_deployment().bindings[role].inputs:
            if binding.index is None:
                raise BackendInferenceError(
                    f"Ascend input {binding.semantic!r} has no runtime index", code="invalid_input_bindings"
                )
            if (role, binding.semantic) in self._linked_inputs:
                continue
            try:
                indexed_inputs[int(binding.index)] = np.asarray(values[binding.semantic])
            except KeyError as exc:
                raise BackendInferenceError(
                    f"Ascend role {role!r} is missing semantic input {binding.semantic!r}",
                    code="missing_semantic_input",
                ) from exc
        return indexed_inputs

    def _role_read_indices(self, role_index: int, role: str) -> set[int]:
        """Select the output slots worth copying back to the host for one role.

        Everything that leaves the compiled graph is read, plus any internal tensor a later
        role consumes over something other than a device link - those still have to travel
        through host memory.
        """
        deployment = self._loaded_deployment()
        bindings = deployment.bindings[role]
        public_indices = {
            int(binding.index)
            for binding in bindings.outputs
            if binding.index is not None and not binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX)
        }
        host_internal_indices = {
            int(binding.index)
            for binding in bindings.outputs
            if binding.index is not None
            and binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX)
            and any(
                later_input.semantic == binding.semantic
                for later_role in deployment.execution[role_index + 1 :]
                for later_input in deployment.bindings[later_role].inputs
                if not any(
                    link.producer == role and link.consumer == later_role and link.semantic == binding.semantic
                    for link in deployment.device_links
                )
            )
        }
        return public_indices | host_internal_indices

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        deployment = self._loaded_deployment()
        try:
            role_index = deployment.execution.index(role)
        except ValueError as exc:
            raise BackendInferenceError(
                f"unknown or unloaded Ascend role {role!r}", code="unknown_execution_role"
            ) from exc
        return self._run_role(role_index, role, dict(inputs), request=request)

    def _request_stream(self, request: NamedTensorRequest) -> object | None:
        streams = self._priority_streams
        if streams is None:
            if request.priority != 0:
                raise BackendAdmissionError(
                    "non-zero Ascend priority requires scheduler-enabled priority streams",
                    code="hardware_priority_unavailable",
                )
            return None
        mapping = self.capabilities.priority_mapping
        if mapping is None:
            raise BackendAdmissionError(
                "Ascend priority mapping is unavailable",
                code="hardware_priority_unavailable",
            )
        try:
            native_priority = mapping.map_generic(request.priority)
        except ValueError as exc:
            raise BackendAdmissionError(str(exc), code="unsupported_priority") from exc
        if not streams.supports(native_priority):
            raise BackendAdmissionError(
                f"Ascend hardware does not expose native priority {native_priority}",
                code="hardware_priority_unavailable",
            )
        return streams.select(native_priority)

    def _close(self) -> None:
        models = self._models
        lease = self._lease
        priority_streams = self._priority_streams
        self._models = {}
        self._lease = None
        self._priority_streams = None
        self._linked_inputs = frozenset()
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
            try:
                lease.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise BackendLifecycleError(
                f"Ascend model session close failed: {'; '.join(str(error) for error in errors)}",
                code="close_failed",
            )

    @staticmethod
    def _bound_output(role: str, binding: TensorBinding, outputs: Mapping[int, np.ndarray]) -> np.ndarray:
        if binding.index is None or int(binding.index) not in outputs:
            raise BackendInferenceError(
                f"Ascend role {role!r} did not return output {binding.semantic!r}", code="missing_runtime_output"
            )
        return outputs[int(binding.index)]

    @staticmethod
    def _binding_for_semantic(bindings, semantic: str, description: str) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1:
            raise BackendLoadError(
                f"Ascend device link {semantic!r} requires exactly one {description} binding",
                code="invalid_device_link",
            )
        return matches[0]
