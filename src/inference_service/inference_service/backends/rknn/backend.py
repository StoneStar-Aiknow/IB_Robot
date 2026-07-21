"""Manifest-driven RKNNLite backend for ACT and SmolVLA deployments."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_manifest.json_utils import load_json_strict
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendResult, InferenceRequest, RuntimeContext
from inference_service.codecs import BoundInputs, ExecutionFrame, ExecutionPlan

_ALLOWED_RUNTIME_OPTIONS = frozenset({"target", "core_mask", "random_seed"})
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})


@dataclass(frozen=True)
class _SmolVLAEmbedding:
    token_weight: np.ndarray
    state_weight: np.ndarray
    state_bias: np.ndarray


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
    """Execute ACT and SmolVLA through manifest-declared RKNN modules."""

    def __init__(self, *, rknn_loader: Callable[[], type] | None = None) -> None:
        super().__init__("rknn", BackendCapabilities(max_in_flight_per_instance=1))
        self._rknn_loader = rknn_loader or self._import_rknn_type
        self._sessions: dict[str, RKNNSession] = {}
        self._owned_sessions: tuple[RKNNSession, ...] = ()
        self._embedding: _SmolVLAEmbedding | None = None
        self._context: RuntimeContext | None = None
        self._policy_config: dict[str, object] = {}
        self._options: dict[str, object] = {}
        self._random: np.random.Generator | None = None

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "rknn":
            raise BackendLoadError("RKNNBackend requires a compiled rknn deployment", code="invalid_deployment")
        if context.policy.policy_type not in {"act", "smolvla"}:
            raise BackendLoadError(
                f"RKNNBackend does not support policy family {context.policy.policy_type!r}",
                code="unsupported_policy_backend_pair",
            )
        if context.policy.policy_type == "act":
            valid_execution = deployment.execution == ("policy",)
            expected_description = "['policy']"
        else:
            valid_execution = (
                len(deployment.execution) >= 4
                and deployment.execution[-3:] == ("embedding", "prefill", "action")
                and all(role == "vision" or role.startswith("vision_") for role in deployment.execution[:-3])
            )
            expected_description = "one or more vision roles followed by ['embedding', 'prefill', 'action']"
        if not valid_execution:
            raise BackendLoadError(
                f"RKNN {context.policy.policy_type} requires {expected_description}, got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        options = self._validate_runtime_options(context.runtime_options)
        for role in deployment.execution:
            artifact = deployment.artifacts[role]
            if role == "embedding":
                if artifact.format not in {"pt", "pytorch"}:
                    raise BackendLoadError(
                        "RKNN SmolVLA embedding artifact must use format 'pt' or 'pytorch'",
                        code="invalid_artifact_format",
                    )
            elif artifact.format != "rknn":
                raise BackendLoadError(
                    f"RKNN role {role!r} artifact format must be 'rknn'",
                    code="invalid_artifact_format",
                )
            self._require_artifact(context, role)

        policy_config = self._load_policy_config(context.validated_manifest.bundle_root / "config.json")
        embedding = None
        if context.policy.policy_type == "act":
            self._validate_act_plan(deployment)
        else:
            try:
                state_projection = deployment.artifacts["state_projection"]
            except KeyError as exc:
                raise BackendLoadError(
                    "RKNN SmolVLA requires a manifest-declared state_projection artifact",
                    code="missing_artifact_role",
                ) from exc
            if state_projection.format not in {"pt", "pytorch"}:
                raise BackendLoadError(
                    "RKNN SmolVLA state_projection artifact must use format 'pt' or 'pytorch'",
                    code="invalid_artifact_format",
                )
            embedding = self._load_embedding(
                self._require_artifact(context, "embedding"),
                self._require_artifact(context, "state_projection"),
            )
            self._validate_smolvla_plan(deployment, embedding, policy_config)

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
        self._embedding = embedding
        self._context = context
        self._policy_config = policy_config
        self._options = {**options, "core_mask": core_mask}
        self._random = np.random.default_rng(options["random_seed"])

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
        if context.policy.policy_type == "act":
            outputs = self._infer_act(plan, role_inputs)
        else:
            outputs = self._infer_smolvla(plan, role_inputs)
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
        self._embedding = None
        self._context = None
        self._policy_config = {}
        self._options = {}
        self._random = None
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

    def _infer_smolvla(
        self, plan: ExecutionPlan, role_inputs: Mapping[str, BoundInputs]
    ) -> dict[str, dict[int, np.ndarray]]:
        vision_roles = plan.role_names[:-3]
        if (
            len(vision_roles) < 1
            or plan.role_names[-3:] != ("embedding", "prefill", "action")
            or any(role != "vision" and not role.startswith("vision_") for role in vision_roles)
        ):
            raise BackendInferenceError(
                "RKNN SmolVLA requires vision role(s), embedding, prefill, and action",
                code="invalid_request",
            )
        frame = ExecutionFrame(plan)
        try:
            for role in vision_roles:
                frame.begin_role(role)
                raw_outputs = self._infer_role(plan, role, role_inputs[role])
                frame.finish_role(role, self._semantic_outputs(plan, role, raw_outputs))

            embedding_host_inputs = frame.begin_role("embedding")
            embedding_outputs = self._execute_embedding(
                plan.role("embedding").bindings,
                role_inputs["embedding"],
                embedding_host_inputs,
            )
            frame.finish_role("embedding", embedding_outputs)

            prefill_host_inputs = frame.begin_role("prefill")
            prefill_bound = self._inputs_from_semantics(
                plan.role("prefill").bindings,
                role_inputs["prefill"],
                prefill_host_inputs,
            )
            prefill_outputs = self._infer_role(plan, "prefill", prefill_bound)
            frame.finish_role("prefill", self._semantic_outputs(plan, "prefill", prefill_outputs))

            action_host_inputs = frame.begin_role("action")
            action_outputs = self._execute_action(
                plan.role("action").bindings,
                role_inputs["action"],
                action_host_inputs,
            )
            frame.finish_role("action")
            return {"action": action_outputs}
        finally:
            frame.close()

    def _execute_embedding(
        self,
        bindings: ArtifactBindings,
        external_inputs: BoundInputs,
        host_inputs: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if self._embedding is None:
            raise BackendInferenceError("RKNN SmolVLA embedding is not loaded", code="runtime_not_loaded")
        values = {tensor.semantic: tensor.value for tensor in external_inputs.tensors}
        values.update(host_inputs)
        token_binding = self._binding_for_semantics(bindings.inputs, {"observation.language.tokens"}, "language tokens")
        mask_binding = self._binding_for_semantics(
            bindings.inputs, {"observation.language.attention_mask"}, "language mask"
        )
        tokens = np.asarray(values[token_binding.semantic], dtype=np.int64)
        if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= self._embedding.token_weight.shape[0]:
            raise BackendInferenceError("SmolVLA token id is outside the embedding table", code="invalid_token_id")
        language = np.ascontiguousarray(self._embedding.token_weight[tokens])
        hidden_size = language.shape[-1]

        state_binding = self._binding_for_semantics(bindings.inputs, {"observation.state"}, "state")
        state = np.asarray(values[state_binding.semantic], dtype=np.float32)
        if state.ndim != 2 or state.shape[-1] != self._embedding.state_weight.shape[1]:
            raise BackendInferenceError(
                f"SmolVLA state shape {state.shape} is incompatible with projection input "
                f"{self._embedding.state_weight.shape[1]}",
                code="invalid_state_shape",
            )
        state_embedding = state @ self._embedding.state_weight.T + self._embedding.state_bias
        state_embedding = np.ascontiguousarray(state_embedding[:, None, :], dtype=language.dtype)

        image_semantics = [
            binding.semantic for binding in bindings.inputs if binding.semantic.startswith("internal.image_embedding.")
        ]
        image_embeddings = [np.asarray(values[semantic], dtype=language.dtype) for semantic in image_semantics]
        if not image_embeddings or any(image.shape[-1] != hidden_size for image in image_embeddings):
            raise BackendInferenceError(
                "SmolVLA image and language embeddings must share one hidden dimension",
                code="invalid_embedding_shape",
            )
        scaled_images = [image * math.sqrt(hidden_size) for image in image_embeddings]
        language = language * math.sqrt(hidden_size)
        prefix_embeddings = np.concatenate((*scaled_images, language, state_embedding), axis=1)
        language_mask = np.asarray(values[mask_binding.semantic], dtype=bool)
        image_masks = [np.ones(image.shape[:2], dtype=bool) for image in image_embeddings]
        state_mask = np.ones(state_embedding.shape[:2], dtype=bool)
        prefix_pad_masks = np.concatenate((*image_masks, language_mask, state_mask), axis=1)
        attention_markers = np.concatenate(
            (
                *(np.zeros(image.shape[:2], dtype=np.int64) for image in image_embeddings),
                np.zeros(language_mask.shape, dtype=np.int64),
                np.ones(state_mask.shape, dtype=np.int64),
            ),
            axis=1,
        )

        output_embeddings = self._binding_for_semantics(
            bindings.outputs, {"internal.prefix_embeddings"}, "prefix embeddings"
        )
        output_pad_masks = self._binding_for_semantics(
            bindings.outputs, {"internal.prefix_pad_masks"}, "prefix pad masks"
        )
        prefix_embeddings = self._pad_prefix(prefix_embeddings, output_embeddings.shape, np.float32(0.0))
        prefix_pad_masks = self._pad_prefix(prefix_pad_masks, output_pad_masks.shape, False)
        semantic_outputs = {
            output_embeddings.semantic: np.ascontiguousarray(
                prefix_embeddings,
                dtype=self._numpy_dtype(output_embeddings.dtype),
            ),
            output_pad_masks.semantic: np.ascontiguousarray(
                prefix_pad_masks,
                dtype=self._numpy_dtype(output_pad_masks.dtype),
            ),
        }
        attention_bindings = [binding for binding in bindings.outputs if binding.semantic == "internal.attention_mask"]
        position_bindings = [binding for binding in bindings.outputs if binding.semantic == "internal.position_ids"]
        if len(attention_bindings) != 1 or len(position_bindings) != 1:
            raise BackendInferenceError(
                "RKNN SmolVLA embedding role requires attention_mask and position_ids outputs",
                code="invalid_execution_plan",
            )
        attention_binding = attention_bindings[0]
        position_binding = position_bindings[0]
        attention_markers = self._pad_prefix(attention_markers, output_pad_masks.shape, 0)
        cumulative_markers = np.cumsum(attention_markers, axis=1)
        attention_mask = cumulative_markers[:, None, :] <= cumulative_markers[:, :, None]
        attention_mask &= prefix_pad_masks[:, None, :] & prefix_pad_masks[:, :, None]
        position_ids = np.cumsum(prefix_pad_masks.astype(np.int64), axis=1) - 1
        position_ids = np.where(prefix_pad_masks, position_ids, 0)
        semantic_outputs[attention_binding.semantic] = np.ascontiguousarray(
            attention_mask,
            dtype=self._numpy_dtype(attention_binding.dtype),
        )
        semantic_outputs[position_binding.semantic] = np.ascontiguousarray(
            position_ids,
            dtype=self._numpy_dtype(position_binding.dtype),
        )
        return semantic_outputs

    def _execute_action(
        self,
        bindings: ArtifactBindings,
        external_inputs: BoundInputs,
        host_inputs: Mapping[str, np.ndarray],
    ) -> dict[int, np.ndarray]:
        initial = self._inputs_from_semantics(bindings, external_inputs, host_inputs)
        noise_binding = self._binding_for_semantics(bindings.inputs, _NOISE_SEMANTICS, "noise")
        time_binding = self._binding_for_semantics(bindings.inputs, _TIME_SEMANTICS, "time")
        action_binding = self._binding_for_semantics(bindings.outputs, {"action"}, "action output")
        noise = initial.get(int(noise_binding.index))
        if noise is None:
            noise = self._sample_noise(noise_binding)
        steps = self._positive_config_int("num_steps")
        dt = -1.0 / steps
        outputs: dict[int, np.ndarray] = {}
        for step in range(steps):
            inputs = dict(initial)
            inputs[int(noise_binding.index)] = np.ascontiguousarray(
                noise,
                dtype=self._numpy_dtype(noise_binding.dtype),
            )
            inputs[int(time_binding.index)] = np.full(
                self._static_shape(time_binding),
                1.0 - step / steps,
                dtype=self._numpy_dtype(time_binding.dtype),
            )
            outputs = self._infer_role_bindings(bindings, "action", inputs)
            try:
                velocity = np.asarray(outputs[int(action_binding.index)])
            except KeyError as exc:
                raise BackendInferenceError(
                    f"RKNN action output index {action_binding.index} is missing",
                    code="missing_runtime_output",
                ) from exc
            noise = np.asarray(noise, dtype=np.float32) + dt * velocity.astype(np.float32)
        return {
            int(action_binding.index): self._convert_runtime_value(
                action_binding,
                noise,
                role="action",
                direction="output",
            )
        }

    def _inputs_from_semantics(
        self,
        bindings: ArtifactBindings,
        external_inputs: BoundInputs,
        host_inputs: Mapping[str, np.ndarray],
    ) -> dict[int, np.ndarray]:
        values = {tensor.semantic: tensor.value for tensor in external_inputs.tensors}
        values.update(host_inputs)
        indexed: dict[int, np.ndarray] = {}
        for binding in bindings.inputs:
            if (
                binding.semantic in _NOISE_SEMANTICS or binding.semantic in _TIME_SEMANTICS
            ) and binding.semantic not in values:
                continue
            if binding.semantic not in values:
                raise BackendInferenceError(
                    f"RKNN role input {binding.semantic!r} is missing",
                    code="missing_runtime_input",
                )
            if binding.index is None:
                raise BackendInferenceError(
                    f"RKNN role input {binding.semantic!r} requires an explicit index",
                    code="invalid_input_bindings",
                )
            indexed[int(binding.index)] = self._convert_runtime_value(
                binding,
                values[binding.semantic],
                role="runtime",
                direction="input",
            )
        return indexed

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
    def _semantic_outputs(
        cls,
        plan: ExecutionPlan,
        role: str,
        outputs: Mapping[int, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            binding.semantic: outputs[int(binding.index)]
            for binding in plan.role(role).bindings.outputs
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

    def _sample_noise(self, binding: TensorBinding) -> np.ndarray:
        if self._random is None:
            raise BackendInferenceError("RKNN random generator is unavailable", code="runtime_not_loaded")
        return np.ascontiguousarray(
            self._random.standard_normal(self._static_shape(binding)).astype(self._numpy_dtype(binding.dtype))
        )

    def _positive_config_int(self, key: str) -> int:
        value = self._policy_config.get(key)
        if type(value) is not int or value < 1:
            raise BackendInferenceError(
                f"RKNN SmolVLA requires positive integer {key!r} in LeRobot config",
                code="invalid_policy_config",
            )
        return value

    @staticmethod
    def _pad_prefix(value: np.ndarray, shape: tuple[int, ...], pad_value: object) -> np.ndarray:
        if len(shape) != value.ndim or shape[0] not in {-1, value.shape[0]}:
            raise BackendInferenceError(
                f"RKNN prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
                code="invalid_prefix_shape",
            )
        target_length = shape[1]
        if target_length < 1 or value.shape[1] > target_length:
            raise BackendInferenceError(
                f"RKNN prefix length {value.shape[1]} exceeds manifest length {target_length}",
                code="invalid_prefix_shape",
            )
        if value.shape[1] == target_length:
            if any(
                expected != -1 and expected != actual
                for expected, actual in zip(shape[2:], value.shape[2:], strict=True)
            ):
                raise BackendInferenceError(
                    f"RKNN prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
                    code="invalid_prefix_shape",
                )
            return value
        if any(
            expected != -1 and expected != actual for expected, actual in zip(shape[2:], value.shape[2:], strict=True)
        ):
            raise BackendInferenceError(
                f"RKNN prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
                code="invalid_prefix_shape",
            )
        pad_shape = list(value.shape)
        pad_shape[1] = target_length - value.shape[1]
        padding = np.full(pad_shape, pad_value, dtype=value.dtype)
        return np.concatenate((value, padding), axis=1)

    @classmethod
    def _validate_smolvla_plan(
        cls,
        deployment: CompiledDeployment,
        embedding: _SmolVLAEmbedding,
        policy_config: Mapping[str, object],
    ) -> None:
        if deployment.device_links:
            raise BackendLoadError(
                "RKNNLite does not support manifest device-pointer links; declare host-visible internal bindings",
                code="unsupported_device_links",
            )
        for key in ("chunk_size", "max_action_dim", "num_steps"):
            value = policy_config.get(key)
            if type(value) is not int or value < 1:
                raise BackendLoadError(
                    f"RKNN SmolVLA requires positive integer {key!r} in LeRobot config",
                    code="invalid_policy_config",
                )
        if policy_config.get("add_image_special_tokens", False) is not False:
            raise BackendLoadError(
                "RKNN SmolVLA does not support add_image_special_tokens=true",
                code="unsupported_policy_config",
            )
        if policy_config.get("empty_cameras", 0) != 0:
            raise BackendLoadError(
                "RKNN SmolVLA requires empty_cameras=0",
                code="unsupported_policy_config",
            )

        vision_roles = deployment.execution[:-3]
        embedding_bindings = deployment.bindings["embedding"]
        image_inputs = [
            binding for binding in embedding_bindings.inputs if binding.semantic.startswith("internal.image_embedding.")
        ]
        if len(image_inputs) != len(vision_roles):
            raise BackendLoadError(
                "RKNN SmolVLA embedding role must consume one internal.image_embedding.* tensor per vision role",
                code="invalid_bindings",
            )
        image_semantics: list[str] = []
        for role in vision_roles:
            role_bindings = deployment.bindings[role]
            if len(role_bindings.inputs) != 1 or len(role_bindings.outputs) != 1:
                raise BackendLoadError(
                    f"RKNN SmolVLA vision role {role!r} requires exactly one input and one output binding",
                    code="invalid_bindings",
                )
            image_input = role_bindings.inputs[0]
            image_output = role_bindings.outputs[0]
            if not cls._is_image_semantic(image_input.semantic) or not image_output.semantic.startswith(
                "internal.image_embedding."
            ):
                raise BackendLoadError(
                    f"RKNN SmolVLA vision role {role!r} must map one image semantic to one image embedding",
                    code="invalid_bindings",
                )
            image_semantics.append(image_output.semantic)
        if tuple(image_semantics) != tuple(binding.semantic for binding in image_inputs):
            raise BackendLoadError(
                "RKNN SmolVLA image embedding input order must match manifest vision execution order",
                code="invalid_bindings",
            )

        cls._binding_for_semantics(
            embedding_bindings.inputs,
            {"observation.language.tokens"},
            "language tokens",
        )
        cls._binding_for_semantics(
            embedding_bindings.inputs,
            {"observation.language.attention_mask"},
            "language mask",
        )
        state_binding = cls._binding_for_semantics(
            embedding_bindings.inputs,
            {"observation.state"},
            "state",
        )
        prefix_embeddings = cls._binding_for_semantics(
            embedding_bindings.outputs,
            {"internal.prefix_embeddings"},
            "prefix embeddings",
        )
        prefix_pad_masks = cls._binding_for_semantics(
            embedding_bindings.outputs,
            {"internal.prefix_pad_masks"},
            "prefix pad masks",
        )
        attention_mask = cls._binding_for_semantics(
            embedding_bindings.outputs,
            {"internal.attention_mask"},
            "attention mask",
        )
        position_ids = cls._binding_for_semantics(
            embedding_bindings.outputs,
            {"internal.position_ids"},
            "position ids",
        )
        if embedding.token_weight.ndim != 2:
            raise BackendLoadError(
                f"RKNN SmolVLA token embedding weight must be rank 2, got {embedding.token_weight.shape}",
                code="invalid_embedding",
            )
        hidden_size = embedding.token_weight.shape[1]
        state_dim = state_binding.shape[-1]
        if state_dim < 1 or embedding.state_weight.shape != (hidden_size, state_dim):
            raise BackendLoadError(
                f"RKNN SmolVLA state projection shape {embedding.state_weight.shape} must be "
                f"({hidden_size}, {state_dim})",
                code="invalid_embedding",
            )
        if embedding.state_bias.shape != (hidden_size,):
            raise BackendLoadError(
                f"RKNN SmolVLA state projection bias shape {embedding.state_bias.shape} must be ({hidden_size},)",
                code="invalid_embedding",
            )
        if len(prefix_embeddings.shape) != 3 or prefix_embeddings.shape[-1] != hidden_size:
            raise BackendLoadError(
                "RKNN SmolVLA prefix embedding binding is incompatible with the embedding hidden size",
                code="invalid_bindings",
            )
        prefix_length = prefix_embeddings.shape[1]
        if (
            prefix_length < 1
            or prefix_pad_masks.shape != (prefix_embeddings.shape[0], prefix_length)
            or attention_mask.shape != (prefix_embeddings.shape[0], prefix_length, prefix_length)
            or position_ids.shape != (prefix_embeddings.shape[0], prefix_length)
        ):
            raise BackendLoadError(
                "RKNN SmolVLA prefix mask and position bindings must use one consistent static prefix length",
                code="invalid_bindings",
            )

        prefill_bindings = deployment.bindings["prefill"]
        required_prefill_inputs = (
            "internal.prefix_embeddings",
            "internal.attention_mask",
            "internal.position_ids",
        )
        if tuple(binding.semantic for binding in prefill_bindings.inputs) != required_prefill_inputs:
            raise BackendLoadError(
                f"RKNN SmolVLA prefill inputs must be ordered as {list(required_prefill_inputs)}",
                code="invalid_bindings",
            )
        cache_semantics = tuple(binding.semantic for binding in prefill_bindings.outputs)
        if not cache_semantics or any(not semantic.startswith("internal.past_") for semantic in cache_semantics):
            raise BackendLoadError(
                "RKNN SmolVLA prefill outputs must declare internal.past_* cache semantics",
                code="invalid_bindings",
            )

        action_bindings = deployment.bindings["action"]
        noise_binding = cls._binding_for_semantics(action_bindings.inputs, _NOISE_SEMANTICS, "noise")
        cls._binding_for_semantics(action_bindings.inputs, _TIME_SEMANTICS, "time")
        cls._binding_for_semantics(
            action_bindings.inputs,
            {"internal.prefix_pad_masks"},
            "prefix pad masks",
        )
        action_binding = cls._binding_for_semantics(action_bindings.outputs, {"action"}, "action output")
        action_internal_semantics = tuple(
            binding.semantic
            for binding in action_bindings.inputs
            if binding.semantic.startswith("internal.") and binding.semantic != "internal.prefix_pad_masks"
        )
        if action_internal_semantics != cache_semantics:
            raise BackendLoadError(
                "RKNN SmolVLA action cache inputs must match prefill outputs in order",
                code="invalid_bindings",
            )
        chunk_size = int(policy_config["chunk_size"])
        max_action_dim = int(policy_config["max_action_dim"])
        expected_action_shape = (1, chunk_size, max_action_dim)
        if noise_binding.shape != expected_action_shape or action_binding.shape != expected_action_shape:
            raise BackendLoadError(
                f"RKNN SmolVLA noise and action bindings must use shape {expected_action_shape}",
                code="invalid_bindings",
            )

    @staticmethod
    def _validate_act_plan(deployment: CompiledDeployment) -> None:
        if deployment.device_links:
            raise BackendLoadError("RKNN ACT does not support device links", code="unsupported_device_links")
        bindings = deployment.bindings["policy"]
        if any(binding.index is None for binding in (*bindings.inputs, *bindings.outputs)):
            raise BackendLoadError("RKNN ACT bindings require explicit runtime indices", code="invalid_bindings")

    @staticmethod
    def _session_load_order(deployment: CompiledDeployment, policy_type: str) -> tuple[str, ...]:
        roles = tuple(role for role in deployment.execution if role != "embedding")
        if policy_type != "smolvla":
            return roles
        vision_roles = tuple(role for role in roles if role == "vision" or role.startswith("vision_"))
        return ("prefill", *vision_roles, "action")

    @staticmethod
    def _session_cache_key(deployment: CompiledDeployment, role: str) -> tuple[object, ...]:
        artifact = deployment.artifacts[role]
        bindings = deployment.bindings[role]
        return (
            artifact.sha256,
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
    def _static_shape(binding: TensorBinding) -> tuple[int, ...]:
        if any(dimension < 1 for dimension in binding.shape):
            raise BackendInferenceError(
                f"RKNN runtime-generated input {binding.semantic!r} requires a static shape, got {binding.shape}",
                code="dynamic_runtime_input",
            )
        return binding.shape

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
    def _load_policy_config(path: Path) -> dict[str, object]:
        try:
            value = load_json_strict(path)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to read LeRobot config {path}: {exc}", code="invalid_policy_config"
            ) from exc
        if not isinstance(value, dict):
            raise BackendLoadError(f"LeRobot config must be an object: {path}", code="invalid_policy_config")
        return value

    @staticmethod
    def _load_embedding(embedding_path: Path, state_projection_path: Path) -> _SmolVLAEmbedding:
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"RKNN SmolVLA embedding requires PyTorch to load {embedding_path}: {exc}",
                code="missing_dependency",
            ) from exc
        try:
            embedding_state = torch.load(embedding_path, map_location="cpu", weights_only=True)
            projection_state = torch.load(state_projection_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to load RKNN embedding artifacts {embedding_path} and {state_projection_path}: {exc}",
                code="invalid_embedding",
            ) from exc
        if not isinstance(embedding_state, Mapping) or not isinstance(projection_state, Mapping):
            raise BackendLoadError(
                "RKNN SmolVLA embedding and state_projection artifacts must contain tensor mappings",
                code="invalid_embedding",
            )
        token_weight = embedding_state.get("token_embedding.weight", embedding_state.get("weight"))
        state_weight = projection_state.get("state_proj.weight", projection_state.get("weight"))
        state_bias = projection_state.get("state_proj.bias", projection_state.get("bias"))
        if token_weight is None or state_weight is None or state_bias is None:
            raise BackendLoadError(
                "RKNN embedding artifacts must contain token_embedding.weight and state projection weight/bias",
                code="invalid_embedding",
            )
        return _SmolVLAEmbedding(
            token_weight=RKNNBackend._to_numpy_weight(token_weight, embedding_path, "token_embedding.weight"),
            state_weight=RKNNBackend._to_numpy_weight(state_weight, state_projection_path, "state_proj.weight"),
            state_bias=RKNNBackend._to_numpy_weight(state_bias, state_projection_path, "state_proj.bias"),
        )

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
    return RKNNBackend()
