"""Generic manifest-bound Ascend OM model session."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.ascend.acl_runtime import ACL_RUNTIME_MANAGER, AclRuntimeLease, AclRuntimeManager
from inference_service.backends.ascend.model import AclModel
from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError, BackendLoadError
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession


class AscendOmModelSession(ModelSession):
    """Execute manifest-declared OM roles using shared ACL leases and model resources."""

    def __init__(
        self,
        device_id: int = 0,
        *,
        runtime_manager: AclRuntimeManager = ACL_RUNTIME_MANAGER,
        model_factory=AclModel,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        super().__init__(
            "model-session:ascend",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                resource_domain=f"ascend:{device_id}",
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
        self._device_id = device_id
        self._runtime_manager = runtime_manager
        self._model_factory = model_factory
        self._lease: AclRuntimeLease | None = None
        self._models: dict[str, AclModel] = {}

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise BackendLoadError(
                "AscendOmModelSession requires a compiled Ascend deployment", code="invalid_deployment"
            )
        unknown_options = sorted(set(context.runtime_options) - {"device_id", "acl_config_path"})
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
        if deployment.device_links:
            raise BackendLoadError(
                "generic Ascend OM sessions currently require host-routed execution without device links",
                code="unsupported_execution_graph",
            )

        config_path = context.runtime_options.get("acl_config_path")
        if config_path is not None and (not isinstance(config_path, str) or not config_path.strip()):
            raise BackendLoadError("acl_config_path must be a non-empty string", code="invalid_runtime_options")
        lease = self._runtime_manager.acquire(self._device_id, config_path)
        rollback.defer(lease.close)
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
            model.prepare_datasets()
        self._lease = lease
        self._models = models

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or not self._models:
            raise BackendInferenceError("Ascend OM model session is not loaded", code="runtime_not_loaded")
        values = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role in deployment.execution:
            bindings = deployment.bindings[role]
            indexed_inputs: dict[int, np.ndarray] = {}
            for binding in bindings.inputs:
                if binding.index is None:
                    raise BackendInferenceError(
                        f"Ascend input {binding.semantic!r} has no runtime index", code="invalid_input_bindings"
                    )
                try:
                    indexed_inputs[int(binding.index)] = np.asarray(values[binding.semantic])
                except KeyError as exc:
                    raise BackendInferenceError(
                        f"Ascend role {role!r} is missing semantic input {binding.semantic!r}",
                        code="missing_semantic_input",
                    ) from exc
            runtime_outputs = self._models[role].execute(indexed_inputs)
            for binding in bindings.outputs:
                output = self._bound_output(role, binding, runtime_outputs)
                values[binding.semantic] = output
                if not binding.semantic.startswith("internal."):
                    public_outputs[binding.semantic] = output
        return public_outputs

    def _close(self) -> None:
        models = self._models
        lease = self._lease
        self._models = {}
        self._lease = None
        errors: list[Exception] = []
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
