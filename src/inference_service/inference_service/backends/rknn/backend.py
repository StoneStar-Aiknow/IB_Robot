"""Manifest-driven RKNNLite backend for ACT deployments.

SmolVLA has migrated to ``RKNNModelSession`` and the shared SmolVLA family
executor; this backend executes ACT only and fails closed for every other
policy family. RKNN-specific runtime helpers (target/core-mask resolution,
per-role ``data_format``, ``share_group`` reuse, dtype conversion, device-link
rejection) are reused by :class:`RKNNModelSession`.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendResult, InferenceRequest, RuntimeContext
from inference_service.codecs import BoundInputs, ExecutionPlan

_ALLOWED_RUNTIME_OPTIONS = frozenset({"target", "core_mask", "random_seed"})


class RKNNSession:
    """One RKNNLite module initialized from one manifest execution role."""

    def __init__(
        self,
        rknn_type: type,
        role: str,
        path: Path,
        *,
        target: str | None,
        core_mask: int,
        data_format: str | None,
    ) -> None:
        self.role = role
        self._runtime = rknn_type()
        self._data_format = data_format
        self._closed = False
        try:
            ret = self._runtime.load_rknn(str(path))
            if ret != 0:
                raise RuntimeError(f"load_rknn returned {ret}")
            ret = self._runtime.init_runtime(target=target, core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"init_runtime returned {ret}")
        except Exception:
            self.close()
            raise

    def infer(self, inputs: BoundInputs | Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        if self._closed:
            raise BackendInferenceError(f"RKNN role {self.role!r} is closed", code="runtime_not_loaded")
        if isinstance(inputs, BoundInputs):
            ordered = inputs.ordered_values
        else:
            ordered = tuple(inputs[index] for index in sorted(inputs))
        outputs = self._runtime.inference(inputs=list(ordered), data_format=self._data_format)
        if outputs is None or len(outputs) == 0:
            raise BackendInferenceError(f"RKNN role {self.role!r} returned no outputs", code="missing_runtime_output")
        return {index: np.asarray(output) for index, output in enumerate(outputs)}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        release = getattr(self._runtime, "release", None)
        if callable(release):
            release()


class RKNNBackend(LifecycleBackend):
    """Execute ACT through manifest-declared RKNN modules.

    SmolVLA has migrated to ``RKNNModelSession`` and the shared SmolVLA family
    executor; this backend fails closed for SmolVLA (and any non-ACT family).
    """

    def __init__(
        self,
        *,
        rknn_loader: Callable[[], type] | None = None,
        expose_hardware_identity: bool = False,
    ) -> None:
        super().__init__(
            "rknn",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                hardware_resource_id="rknn:0" if expose_hardware_identity else None,
            ),
        )
        self._rknn_loader = rknn_loader or self._import_rknn_type
        self._sessions: dict[str, RKNNSession] = {}
        self._owned_sessions: tuple[RKNNSession, ...] = ()
        self._context: RuntimeContext | None = None
        self._options: dict[str, object] = {}

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "rknn":
            raise BackendLoadError("RKNNBackend requires a compiled rknn deployment", code="invalid_deployment")
        if context.policy.policy_type == "smolvla":
            raise BackendLoadError(
                "RKNNBackend no longer hosts SmolVLA; SmolVLA runs through RKNNModelSession",
                code="unsupported_policy_backend_pair",
            )
        if context.policy.policy_type != "act":
            raise BackendLoadError(
                f"RKNNBackend does not support policy family {context.policy.policy_type!r}",
                code="unsupported_policy_backend_pair",
            )
        if deployment.execution != ("policy",):
            raise BackendLoadError(
                f"RKNN ACT requires ['policy'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        options = self._validate_runtime_options(context.runtime_options)
        for role in deployment.execution:
            artifact = deployment.artifacts[role]
            if artifact.format != "rknn":
                raise BackendLoadError(
                    f"RKNN role {role!r} artifact format must be 'rknn'",
                    code="invalid_artifact_format",
                )
            self._require_artifact(context, role)
        self._validate_act_plan(deployment)

        rknn_type = self._rknn_loader()
        core_mask = self._resolve_core_mask(rknn_type, options["core_mask"])
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
        for role in self._session_load_order(deployment, context.policy.policy_type):
            cache_key = self._session_cache_key(deployment, role)
            if cache_key in shared_sessions:
                sessions[role] = shared_sessions[cache_key]
                continue
            try:
                session = RKNNSession(
                    rknn_type,
                    role,
                    self._require_artifact(context, role),
                    target=options["target"],
                    core_mask=core_mask,
                    data_format=self._runtime_data_format(deployment.bindings[role]),
                )
                sessions[role] = session
                shared_sessions[cache_key] = session
                owned_sessions.append(session)
            except Exception as exc:
                raise BackendLoadError(
                    f"RKNN role {role!r} failed to load from manifest artifact: {exc}",
                    code="runtime_load_failed",
                ) from exc

        self._sessions = sessions
        self._owned_sessions = tuple(owned_sessions)
        self._context = context
        self._options = {**options, "core_mask": core_mask}

    def _infer(self, request: InferenceRequest) -> BackendResult:
        context = self._context
        if context is None:
            raise BackendInferenceError("RKNNBackend is not fully loaded", code="runtime_not_loaded")
        plan, role_inputs = self._request_execution(request)
        if plan.role_names != context.deployment.execution:
            raise BackendInferenceError(
                f"RKNN request execution {list(plan.role_names)} does not match deployment execution "
                f"{list(context.deployment.execution)}",
                code="invalid_request",
            )
        started = time.perf_counter()
        outputs = self._infer_act(plan, role_inputs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        action = self._raw_action(outputs, plan)
        return BackendResult(
            action=outputs,
            actual_chunk_size=self._chunk_size(action),
            backend_latency_ms=latency_ms,
            metadata={
                "request_id": request.request_id,
                "target_soc": context.target.soc if context.target is not None else None,
                "runtime_target": self._options.get("target"),
                "core_mask": self._options.get("core_mask"),
                "deployment_name": context.deployment_name,
                "deployment_fingerprint": context.deployment_fingerprint,
            },
        )

    def _close(self) -> None:
        sessions = self._owned_sessions
        self._sessions = {}
        self._owned_sessions = ()
        self._context = None
        self._options = {}
        errors: list[Exception] = []
        for session in reversed(sessions):
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _infer_act(self, plan: ExecutionPlan, role_inputs: Mapping[str, BoundInputs]) -> dict[int, np.ndarray]:
        if plan.role_names != ("policy",):
            raise BackendInferenceError("RKNN ACT requires one policy role", code="invalid_request")
        return self._infer_role(plan, "policy", role_inputs["policy"])

    def _infer_role(
        self,
        plan: ExecutionPlan,
        role: str,
        inputs: BoundInputs | Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        return self._infer_role_bindings(plan.role(role).bindings, role, inputs)

    def _infer_role_bindings(
        self,
        bindings: ArtifactBindings,
        role: str,
        inputs: BoundInputs | Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        indexed_inputs = self._indexed_inputs(bindings, role, inputs)
        outputs = self._sessions[role].infer(indexed_inputs)
        return self._validate_runtime_outputs(bindings, role, outputs)

    def _indexed_inputs(
        self,
        bindings: ArtifactBindings,
        role: str,
        inputs: BoundInputs | Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        if isinstance(inputs, BoundInputs):
            values = {int(tensor.index): tensor.value for tensor in inputs.tensors if tensor.index is not None}
        else:
            values = {int(index): value for index, value in inputs.items()}
        expected = {int(binding.index) for binding in bindings.inputs if binding.index is not None}
        if set(values) != expected:
            raise BackendInferenceError(
                f"RKNN role {role!r} input indices {sorted(values)} do not match manifest indices {sorted(expected)}",
                code="invalid_runtime_inputs",
            )
        return {
            int(binding.index): self._convert_runtime_value(
                binding,
                values[int(binding.index)],
                role=role,
                direction="input",
            )
            for binding in bindings.inputs
            if binding.index is not None
        }

    def _validate_runtime_outputs(
        self,
        bindings: ArtifactBindings,
        role: str,
        outputs: Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        expected = {int(binding.index) for binding in bindings.outputs if binding.index is not None}
        missing = sorted(expected - set(outputs))
        unexpected = sorted(set(outputs) - expected)
        if missing or unexpected:
            raise BackendInferenceError(
                f"RKNN role {role!r} runtime outputs do not match manifest (missing={missing}, unexpected={unexpected})",
                code="invalid_runtime_outputs",
            )
        return {
            int(binding.index): self._convert_runtime_value(
                binding,
                outputs[int(binding.index)],
                role=role,
                direction="output",
            )
            for binding in bindings.outputs
            if binding.index is not None
        }

    @classmethod
    def _convert_runtime_value(
        cls,
        binding: TensorBinding,
        value: object,
        *,
        role: str,
        direction: str,
    ) -> np.ndarray:
        try:
            converted = np.ascontiguousarray(np.asarray(value, dtype=cls._numpy_dtype(binding.dtype)))
        except (TypeError, ValueError) as exc:
            raise BackendInferenceError(
                f"RKNN role {role!r} {direction} {binding.semantic!r} cannot convert to {binding.dtype}",
                code=f"runtime_{direction}_dtype_mismatch",
            ) from exc
        if (
            direction == "output"
            and binding.semantic == "action"
            and converted.ndim == 1
            and all(dimension > 0 for dimension in binding.shape)
            and converted.size == int(np.prod(binding.shape, dtype=np.int64))
        ):
            converted = converted.reshape(binding.shape)
        if converted.ndim != len(binding.shape) or any(
            expected != -1 and expected != actual
            for expected, actual in zip(binding.shape, converted.shape, strict=True)
        ):
            raise BackendInferenceError(
                f"RKNN role {role!r} {direction} {binding.semantic!r} shape {converted.shape} "
                f"does not match manifest shape {binding.shape}",
                code=f"runtime_{direction}_shape_mismatch",
            )
        return converted

    @staticmethod
    def _request_execution(request: InferenceRequest) -> tuple[ExecutionPlan, Mapping[str, BoundInputs]]:
        plan = request.inputs.get("execution_plan")
        role_inputs = request.inputs.get("role_inputs")
        if not isinstance(plan, ExecutionPlan):
            raise BackendInferenceError("RKNNBackend request is missing execution_plan", code="invalid_request")
        if not isinstance(role_inputs, Mapping):
            raise BackendInferenceError("RKNNBackend request is missing role_inputs", code="invalid_request")
        for role in plan.role_names:
            if not isinstance(role_inputs.get(role), BoundInputs):
                raise BackendInferenceError(
                    f"RKNNBackend role {role!r} inputs are not bound tensors",
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
                "RKNN runtime did not return the bound action output", code="missing_action_output"
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
                f"RKNN action output has invalid shape {action.shape}", code="invalid_action_shape"
            )
        return int(action.shape[-2])

    @staticmethod
    def _validate_act_plan(deployment: CompiledDeployment) -> None:
        if deployment.device_links:
            raise BackendLoadError("RKNN ACT does not support device links", code="unsupported_device_links")
        bindings = deployment.bindings["policy"]
        if any(binding.index is None for binding in (*bindings.inputs, *bindings.outputs)):
            raise BackendLoadError("RKNN ACT bindings require explicit runtime indices", code="invalid_bindings")

    @staticmethod
    def _session_load_order(deployment: CompiledDeployment, policy_type: str) -> tuple[str, ...]:
        del policy_type
        return tuple(role for role in deployment.execution if deployment.artifacts[role].format != "pt")

    @staticmethod
    def _session_cache_key(deployment: CompiledDeployment, role: str) -> tuple[object, ...]:
        artifact = deployment.artifacts[role]
        if artifact.share_group is None:
            return ("role", role)
        bindings = deployment.bindings[role]
        return (
            "share_group",
            artifact.share_group,
            artifact.path,
            tuple(
                (binding.runtime_name, binding.index, binding.dtype, binding.shape, binding.layout)
                for binding in bindings.inputs
            ),
            tuple(
                (binding.runtime_name, binding.index, binding.dtype, binding.shape, binding.layout)
                for binding in bindings.outputs
            ),
        )

    @staticmethod
    def _runtime_data_format(bindings: ArtifactBindings) -> str | None:
        layouts = {
            binding.layout.lower()
            for binding in bindings.inputs
            if binding.semantic.startswith(("observation.image", "observation.images.")) and binding.layout is not None
        }
        if len(layouts) > 1:
            raise BackendLoadError(
                f"RKNN role uses mixed image layouts {sorted(layouts)}; RKNNLite accepts one data_format per call",
                code="invalid_bindings",
            )
        return next(iter(layouts), None)

    @staticmethod
    def _binding_for_semantics(
        bindings: tuple[TensorBinding, ...], semantics: set[str] | frozenset[str], description: str
    ) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic in semantics]
        if len(matches) != 1 or matches[0].index is None:
            raise BackendLoadError(
                f"RKNN deployment requires exactly one indexed {description} binding",
                code="invalid_bindings",
            )
        return matches[0]

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        try:
            path = context.resolved_artifacts[role]
        except KeyError as exc:
            raise BackendLoadError(
                f"RKNN deployment is missing artifact role {role!r}", code="missing_artifact_role"
            ) from exc
        if not path.is_file():
            raise BackendLoadError(f"RKNN artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
        return path

    @staticmethod
    def _to_numpy_weight(value: object, path: Path, name: str) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        cast_float = getattr(value, "float", None)
        if callable(cast_float):
            value = cast_float()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
        try:
            return np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        except (TypeError, ValueError) as exc:
            raise BackendLoadError(
                f"RKNN embedding artifact {path} contains invalid tensor {name!r}",
                code="invalid_embedding",
            ) from exc

    @staticmethod
    def _is_image_semantic(semantic: str) -> bool:
        return (
            semantic == "observation.image"
            or semantic.startswith("observation.image.")
            or semantic.startswith("observation.images.")
        )

    @staticmethod
    def _numpy_dtype(dtype: str) -> np.dtype:
        if dtype != "bfloat16":
            return np.dtype(dtype)
        try:
            return np.dtype(dtype)
        except TypeError:
            try:
                extension = importlib.import_module("ml_dtypes")
            except ImportError as exc:
                raise BackendLoadError(
                    "RKNN bfloat16 bindings require NumPy bfloat16 support or ml_dtypes",
                    code="unsupported_runtime_dtype",
                ) from exc
            return np.dtype(extension.bfloat16)

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
        unknown = sorted(set(options) - _ALLOWED_RUNTIME_OPTIONS)
        if unknown:
            raise BackendLoadError(f"unknown RKNN runtime options: {unknown}", code="invalid_runtime_options")
        target = options.get("target")
        if target is not None and (type(target) is not str or not target.strip()):
            raise BackendLoadError("RKNN target must be a non-empty string or null", code="invalid_runtime_options")
        core_mask = options.get("core_mask", "all")
        if type(core_mask) not in {str, int} or (type(core_mask) is int and core_mask < 0):
            raise BackendLoadError(
                "RKNN core_mask must be a non-negative integer or supported string name",
                code="invalid_runtime_options",
            )
        if type(core_mask) is str and core_mask.lower() not in {"all", "auto", "0", "1", "2"}:
            raise BackendLoadError(f"unsupported RKNN core_mask {core_mask!r}", code="invalid_runtime_options")
        random_seed = options.get("random_seed")
        if random_seed is not None and type(random_seed) is not int:
            raise BackendLoadError("RKNN random_seed must be an integer or null", code="invalid_runtime_options")
        return {"target": target, "core_mask": core_mask, "random_seed": random_seed}

    @staticmethod
    def _resolve_core_mask(rknn_type: type, value: object) -> int:
        if type(value) is int:
            return value
        names = {
            "all": "NPU_CORE_ALL",
            "auto": "NPU_CORE_AUTO",
            "0": "NPU_CORE_0",
            "1": "NPU_CORE_1",
            "2": "NPU_CORE_2",
        }
        try:
            attribute = names[str(value).lower()]
        except KeyError as exc:
            raise BackendLoadError(f"unsupported RKNN core_mask {value!r}", code="invalid_runtime_options") from exc
        try:
            return int(getattr(rknn_type, attribute))
        except AttributeError as exc:
            raise BackendLoadError(
                f"installed RKNNLite does not expose {attribute}",
                code="incompatible_dependency",
            ) from exc

    @staticmethod
    def _import_rknn_type() -> type:
        try:
            module = importlib.import_module("rknnlite.api")
            return module.RKNNLite
        except (ImportError, OSError, AttributeError) as exc:
            raise BackendLoadError(
                f"RKNNLite dependency 'rknnlite.api.RKNNLite' is unavailable: {exc}",
                code="missing_dependency",
            ) from exc


def create_backend(context: RuntimeContext) -> RKNNBackend:
    """Lazy registry factory for RKNNLite execution."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "rknn":
        raise BackendLoadError("RKNNBackend requires a compiled rknn deployment", code="invalid_deployment")
    RKNNBackend._validate_runtime_options(context.runtime_options)
    return RKNNBackend(expose_hardware_identity=context.priority_scheduling)
