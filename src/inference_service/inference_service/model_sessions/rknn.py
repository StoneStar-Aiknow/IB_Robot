"""Generic manifest-bound RKNNLite model session."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError, BackendLoadError
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.rknn.backend import RKNNBackend, RKNNSession
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession

_HOST_ARTIFACT_FORMATS = frozenset({"pt", "pytorch"})


class RKNNModelSession(ModelSession):
    """Execute manifest-declared RKNNLite roles through shared RKNN sessions.

    The session owns only RKNN runtime resource lifecycle (RKNNLite module load,
    init_runtime, release), admission, partial-load rollback, health, manifest
    device-link rejection, runtime option/target/core-mask validation, host-role
    skipping, semantic-to-index role invocation, and close. It deliberately does
    not implement PI0.5/SmolVLA timestep loops, Euler/state updates, embedding
    construction, time embedding, noise sampling, or any policy-family control
    flow; those responsibilities belong to executor-owned stages above this
    session that drive it through ``execution().invoke(role, ...)``.

    Runtime specifics (per-role ``data_format`` from image binding layouts,
    ``share_group`` session reuse, ``core_mask`` resolution, ``target``/``init_runtime``
    behavior, dtype conversion, device-link rejection) are reused verbatim from
    :class:`RKNNBackend` so this session stays numerically and operationally
    identical to the existing backend path. Only the low-level
    :class:`RKNNSession` is instantiated for device execution.
    """

    def __init__(
        self,
        *,
        rknn_loader: Callable[[], type] | None = None,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        super().__init__(
            "model-session:rknn",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                resource_domain="rknn",
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
        self._rknn_loader = rknn_loader or RKNNBackend._import_rknn_type
        self._rknn_type: type | None = None
        self._sessions: dict[str, RKNNSession] = {}
        self._owned_sessions: tuple[RKNNSession, ...] = ()
        self._host_roles: frozenset[str] = frozenset()
        self._target: str | None = None
        self._core_mask: int = 0

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._rknn_type)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "rknn":
            raise BackendLoadError("RKNNModelSession requires a compiled rknn deployment", code="invalid_deployment")
        if deployment.device_links:
            raise BackendLoadError(
                "RKNNLite does not support manifest device-pointer links; declare host-visible internal bindings",
                code="unsupported_device_links",
            )
        options = RKNNBackend._validate_runtime_options(context.runtime_options)

        host_roles: list[str] = []
        for role in deployment.execution:
            artifact = deployment.artifacts[role]
            if artifact.format in _HOST_ARTIFACT_FORMATS:
                host_roles.append(role)
                continue
            if artifact.format != "rknn":
                raise BackendLoadError(
                    f"RKNN role {role!r} artifact format must be 'rknn', 'pt', or 'pytorch'",
                    code="invalid_artifact_format",
                )

        rknn_type = self._rknn_loader()
        core_mask = RKNNBackend._resolve_core_mask(rknn_type, options["core_mask"])
        target = options["target"]
        sessions: dict[str, RKNNSession] = {}
        owned_sessions: list[RKNNSession] = []
        shared_sessions: dict[tuple[object, ...], RKNNSession] = {}

        def close_sessions() -> None:
            errors: list[Exception] = []
            for session in reversed(owned_sessions):
                try:
                    session.close()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        rollback.defer(close_sessions)
        for role in deployment.execution:
            if role in host_roles:
                continue
            cache_key = RKNNBackend._session_cache_key(deployment, role)
            existing = shared_sessions.get(cache_key)
            if existing is not None:
                sessions[role] = existing
                continue
            path = RKNNBackend._require_artifact(context, role)
            data_format = RKNNBackend._runtime_data_format(deployment.bindings[role])
            try:
                session = RKNNSession(
                    rknn_type,
                    role,
                    path,
                    target=target,
                    core_mask=core_mask,
                    data_format=data_format,
                )
            except BackendLoadError:
                raise
            except Exception as exc:
                raise BackendLoadError(
                    f"RKNN role {role!r} failed to load from manifest artifact: {exc}",
                    code="runtime_load_failed",
                ) from exc
            sessions[role] = session
            shared_sessions[cache_key] = session
            owned_sessions.append(session)

        self._rknn_type = rknn_type
        self._sessions = sessions
        self._owned_sessions = tuple(owned_sessions)
        self._host_roles = frozenset(host_roles)
        self._target = target
        self._core_mask = core_mask

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or not self._sessions:
            raise BackendInferenceError("RKNN model session is not loaded", code="runtime_not_loaded")
        values: dict[str, object] = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role in deployment.execution:
            if role in self._host_roles:
                continue
            outputs = self._execute_role(role, values, request)
            values.update(outputs)
            for semantic, value in outputs.items():
                if not semantic.startswith("internal."):
                    public_outputs[semantic] = value
        return public_outputs

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        del request
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or role not in self._sessions:
            if role in self._host_roles:
                raise BackendInferenceError(
                    f"RKNN role {role!r} is a host role and is not executable by the model session",
                    code="host_role_not_executable",
                )
            raise BackendInferenceError(f"unknown or unloaded RKNN role {role!r}", code="unknown_execution_role")
        bindings = deployment.bindings[role]
        indexed_inputs = self._indexed_inputs(role, bindings.inputs, inputs)
        runtime_outputs = self._sessions[role].infer(indexed_inputs)
        return self._semantic_outputs(bindings, role, runtime_outputs)

    def _close(self) -> None:
        sessions = self._owned_sessions
        self._rknn_type = None
        self._sessions = {}
        self._owned_sessions = ()
        self._host_roles = frozenset()
        self._target = None
        self._core_mask = 0
        errors: list[Exception] = []
        for session in reversed(sessions):
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise BackendLifecycleError(
                f"RKNN model session close failed: {'; '.join(str(error) for error in errors)}",
                code="close_failed",
            )

    @staticmethod
    def _indexed_inputs(
        role: str,
        input_bindings: tuple[TensorBinding, ...],
        values: Mapping[str, object],
    ) -> dict[int, np.ndarray]:
        indexed: dict[int, np.ndarray] = {}
        for binding in input_bindings:
            if binding.index is None:
                raise BackendInferenceError(
                    f"RKNN role {role!r} input {binding.semantic!r} has no runtime index",
                    code="invalid_input_bindings",
                )
            try:
                value = values[binding.semantic]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"RKNN role {role!r} is missing semantic input {binding.semantic!r}",
                    code="missing_semantic_input",
                ) from exc
            indexed[int(binding.index)] = RKNNBackend._convert_runtime_value(
                binding, value, role=role, direction="input"
            )
        return indexed

    @staticmethod
    def _semantic_outputs(
        bindings: ArtifactBindings,
        role: str,
        runtime_outputs: Mapping[int, np.ndarray],
    ) -> dict[str, np.ndarray]:
        expected = {int(binding.index) for binding in bindings.outputs if binding.index is not None}
        missing = sorted(expected - set(runtime_outputs))
        unexpected = sorted(set(runtime_outputs) - expected)
        if missing or unexpected:
            raise BackendInferenceError(
                f"RKNN role {role!r} runtime outputs do not match manifest (missing={missing}, unexpected={unexpected})",
                code="invalid_runtime_outputs",
            )
        return {
            binding.semantic: RKNNBackend._convert_runtime_value(
                binding,
                runtime_outputs[int(binding.index)],
                role=role,
                direction="output",
            )
            for binding in bindings.outputs
            if binding.index is not None
        }
