"""Manifest-driven Houmo TCIM backend for PI0.5 and SmolVLA HMM deployments."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_manifest.json_utils import load_json_strict
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendResult, InferenceRequest, RuntimeContext
from inference_service.codecs import BoundInputs, ExecutionPlan, build_execution_plan

_ALLOWED_RUNTIME_OPTIONS = frozenset({"device_id", "random_seed"})
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})


@dataclass(frozen=True)
class _EmbeddingWeights:
    token_weight: np.ndarray
    state_weight: np.ndarray | None = None
    state_bias: np.ndarray | None = None


class HMMModule:
    """One TCIM module with manifest-validated name-based runtime I/O."""

    def __init__(
        self,
        runtime: object,
        role: str,
        path: Path,
        bindings: ArtifactBindings,
        *,
        option: object | None,
    ) -> None:
        self.role = role
        self._bindings = bindings
        self._module: object | None = None
        self._input_names: dict[str, str] = {}
        self._output_names: dict[str, str] = {}
        self._closed = False
        try:
            load = runtime.load
            self._module = load(str(path), option) if option is not None else load(str(path))
            self._validate_descriptor()
        except Exception:
            self.close()
            raise

    def execute(
        self,
        semantic_inputs: Mapping[str, object],
        *,
        device_input_semantics: set[str],
        read_semantics: set[str],
    ) -> dict[object, np.ndarray]:
        module = self._require_module()
        for binding in self._bindings.inputs:
            if binding.semantic in device_input_semantics:
                continue
            try:
                value = semantic_inputs[binding.semantic]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"HMM role {self.role!r} is missing input semantic {binding.semantic!r}",
                    code="missing_runtime_input",
                ) from exc
            module.set_input(
                self._input_names[binding.semantic],
                self._convert_value(binding, value, direction="input"),
            )
        module.run()
        sync = getattr(module, "sync", None)
        if callable(sync):
            sync()

        outputs: dict[object, np.ndarray] = {}
        for binding in self._bindings.outputs:
            if binding.semantic not in read_semantics:
                continue
            value = module.get_output(self._output_names[binding.semantic])
            to_numpy = getattr(value, "numpy", None)
            if callable(to_numpy):
                value = to_numpy()
            converted = self._convert_value(binding, value, direction="output")
            if binding.index is not None:
                outputs[int(binding.index)] = converted
            if binding.runtime_name is not None:
                outputs[binding.runtime_name] = converted
        return outputs

    def get_device_source(self, binding: TensorBinding, source: str) -> object:
        module = self._require_module()
        if source == "input":
            method = getattr(module, "get_dev_input", None)
            runtime_name = self._input_names[binding.semantic]
        else:
            method = getattr(module, "get_dev_output", None)
            runtime_name = self._output_names[binding.semantic]
        if not callable(method):
            raise BackendLoadError(
                f"TCIM role {self.role!r} does not support get_dev_{source} for {binding.semantic!r}",
                code="unsupported_device_link",
            )
        return method(runtime_name)

    def set_device_input(self, binding: TensorBinding, handle: object) -> None:
        module = self._require_module()
        method = getattr(module, "set_dev_input", None)
        if not callable(method):
            raise BackendLoadError(
                f"TCIM role {self.role!r} does not support set_dev_input for {binding.semantic!r}",
                code="unsupported_device_link",
            )
        method(self._input_names[binding.semantic], handle)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        module = self._module
        self._module = None
        if module is None:
            return
        for method_name in ("release", "close", "destroy"):
            method = getattr(module, method_name, None)
            if callable(method):
                method()
                return

    def _validate_descriptor(self) -> None:
        module = self._require_module()
        input_names = tuple(module.get_input_name(index) for index in range(module.get_num_inputs()))
        output_names = tuple(module.get_output_name(index) for index in range(module.get_num_outputs()))
        self._input_names = self._resolve_bindings(self._bindings.inputs, input_names, "input")
        self._output_names = self._resolve_bindings(self._bindings.outputs, output_names, "output")
        for direction, bindings, names in (
            ("input", self._bindings.inputs, self._input_names),
            ("output", self._bindings.outputs, self._output_names),
        ):
            info_method = getattr(module, f"get_{direction}_info", None)
            if not callable(info_method):
                continue
            for binding in bindings:
                info = info_method(names[binding.semantic])
                shape = getattr(info, "shape", None)
                if shape is not None and not self._compatible_shape(binding.shape, tuple(int(item) for item in shape)):
                    raise BackendLoadError(
                        f"HMM role {self.role!r} {direction} {binding.semantic!r} runtime shape {tuple(shape)} "
                        f"does not match manifest shape {binding.shape}",
                        code="runtime_shape_mismatch",
                    )
                runtime_dtype = self._runtime_dtype_name(getattr(info, "dtype", None))
                if runtime_dtype is not None and runtime_dtype != binding.dtype:
                    raise BackendLoadError(
                        f"HMM role {self.role!r} {direction} {binding.semantic!r} runtime dtype "
                        f"{runtime_dtype!r} does not match manifest dtype {binding.dtype!r}",
                        code="runtime_dtype_mismatch",
                    )

    def _resolve_bindings(
        self,
        bindings: Sequence[TensorBinding],
        runtime_names: tuple[str, ...],
        direction: str,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for binding in bindings:
            runtime_name = binding.runtime_name
            if binding.index is not None:
                index = int(binding.index)
                if index >= len(runtime_names):
                    raise BackendLoadError(
                        f"HMM role {self.role!r} {direction} index {index} exceeds runtime count {len(runtime_names)}",
                        code="runtime_index_mismatch",
                    )
                indexed_name = runtime_names[index]
                if runtime_name is not None and runtime_name != indexed_name:
                    raise BackendLoadError(
                        f"HMM role {self.role!r} {direction} index {index} is named {indexed_name!r}, "
                        f"not {runtime_name!r}",
                        code="runtime_name_mismatch",
                    )
                runtime_name = indexed_name
            if runtime_name is None or runtime_name not in runtime_names:
                raise BackendLoadError(
                    f"HMM role {self.role!r} has no runtime {direction} named {runtime_name!r}",
                    code="runtime_name_mismatch",
                )
            result[binding.semantic] = runtime_name
        return result

    def _convert_value(self, binding: TensorBinding, value: object, *, direction: str) -> np.ndarray:
        try:
            converted = np.ascontiguousarray(np.asarray(value, dtype=self._numpy_dtype(binding.dtype)))
        except (TypeError, ValueError) as exc:
            raise BackendInferenceError(
                f"HMM role {self.role!r} {direction} {binding.semantic!r} cannot convert to {binding.dtype}",
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
        if not self._compatible_shape(binding.shape, converted.shape):
            raise BackendInferenceError(
                f"HMM role {self.role!r} {direction} {binding.semantic!r} shape {converted.shape} "
                f"does not match manifest shape {binding.shape}",
                code=f"runtime_{direction}_shape_mismatch",
            )
        return converted

    def _require_module(self) -> object:
        if self._module is None:
            raise BackendInferenceError(f"HMM role {self.role!r} is closed", code="runtime_not_loaded")
        return self._module

    @staticmethod
    def _compatible_shape(expected: tuple[int, ...], actual: tuple[int, ...]) -> bool:
        return len(expected) == len(actual) and all(
            declared == -1 or declared == observed for declared, observed in zip(expected, actual, strict=True)
        )

    @staticmethod
    def _runtime_dtype_name(value: object) -> str | None:
        if value is None:
            return None
        try:
            return np.dtype(value).name
        except TypeError:
            pass
        text = str(value).lower()
        aliases = {
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
            "bf16": "bfloat16",
        }
        for alias, canonical in aliases.items():
            if alias in text:
                return canonical
        for canonical in (
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        ):
            if canonical in text:
                return canonical
        code = str(getattr(value, "code", "")).lower()
        bits = getattr(value, "bits", None)
        if code in {"float", "fp"} and bits in {16, 32, 64}:
            return f"float{bits}"
        if code in {"int", "uint"} and bits in {8, 16, 32, 64}:
            return f"{code}{bits}"
        if code == "bool":
            return "bool"
        return None

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
                    "HMM bfloat16 bindings require NumPy bfloat16 support or ml_dtypes",
                    code="unsupported_runtime_dtype",
                ) from exc
            return np.dtype(extension.bfloat16)


class HMMBackend(LifecycleBackend):
    """Execute PI0.5 and SmolVLA through manifest-declared TCIM modules."""

    def __init__(self, device_id: int = 0, *, runtime_loader: Callable[[], object] | None = None) -> None:
        super().__init__(
            "hmm",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                resource_domain=f"hmm:{device_id}",
                max_in_flight_per_resource_domain=1,
            ),
        )
        self._device_id = device_id
        self._runtime_loader = runtime_loader or self._import_tcim_runtime
        self._modules: dict[str, HMMModule] = {}
        self._weight_manager: object | None = None
        self._device_handles: tuple[object, ...] = ()
        self._embedding: _EmbeddingWeights | None = None
        self._context: RuntimeContext | None = None
        self._policy_config: dict[str, object] = {}
        self._options: dict[str, object] = {}
        self._random: np.random.Generator | None = None

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hmm":
            raise BackendLoadError("HMMBackend requires a compiled hmm deployment", code="invalid_deployment")
        if context.policy.policy_type not in {"pi05", "smolvla"}:
            raise BackendLoadError(
                f"HMMBackend does not support policy family {context.policy.policy_type!r}",
                code="unsupported_policy_backend_pair",
            )
        options = self._validate_runtime_options(context.runtime_options)
        if int(options["device_id"]) != self._device_id:
            raise BackendLoadError("HMM device_id changed after construction", code="deployment_context_mismatch")
        if not (deployment.target.runtime.startswith("hmm") or deployment.target.runtime.startswith("tcim")):
            raise BackendLoadError(
                f"HMM target.runtime {deployment.target.runtime!r} is not in the TCIM runtime family",
                code="incompatible_backend_target",
            )

        build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
        policy_config = self._load_policy_config(context.validated_manifest.bundle_root / "config.json")
        embedding = self._load_policy_artifacts(context)
        if context.policy.policy_type == "pi05":
            self._validate_pi05_plan(deployment, policy_config, embedding)
        else:
            self._validate_smolvla_plan(deployment, policy_config, embedding)

        runtime = self._runtime_loader()
        weight_manager = self._create_weight_manager(runtime, self._device_id)
        modules: dict[str, HMMModule] = {}
        handles: list[object] = []

        def cleanup_resources() -> None:
            errors: list[Exception] = []
            for module in reversed(tuple(modules.values())):
                try:
                    module.close()
                except Exception as exc:
                    errors.append(exc)
            try:
                self._release_resource(weight_manager)
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        rollback.defer(cleanup_resources)
        for role in deployment.execution:
            artifact = deployment.artifacts[role]
            if role == "embedding":
                if artifact.format not in {"pt", "pytorch"}:
                    raise BackendLoadError(
                        "HMM embedding artifact must use format 'pt' or 'pytorch'",
                        code="invalid_artifact_format",
                    )
                continue
            if artifact.format != "hmm":
                raise BackendLoadError(
                    f"HMM execution role {role!r} artifact format must be 'hmm'",
                    code="invalid_artifact_format",
                )
            try:
                option = self._create_option(runtime, weight_manager)
                modules[role] = HMMModule(
                    runtime,
                    role,
                    self._require_artifact(context, role),
                    deployment.bindings[role],
                    option=option,
                )
            except Exception as exc:
                raise BackendLoadError(
                    f"HMM role {role!r} failed to load from manifest artifact: {exc}",
                    code="runtime_load_failed",
                ) from exc

        for link in deployment.device_links:
            source_bindings = (
                deployment.bindings[link.producer].inputs
                if link.producer_binding == "input"
                else deployment.bindings[link.producer].outputs
            )
            source = self._binding_for_semantic(source_bindings, link.semantic, "device-link source")
            target = self._binding_for_semantic(
                deployment.bindings[link.consumer].inputs,
                link.semantic,
                "device-link consumer",
            )
            handle = modules[link.producer].get_device_source(source, link.producer_binding)
            modules[link.consumer].set_device_input(target, handle)
            handles.append(handle)

        self._modules = modules
        self._weight_manager = weight_manager
        self._device_handles = tuple(handles)
        self._embedding = embedding
        self._context = context
        self._policy_config = policy_config
        self._options = options
        self._random = np.random.default_rng(options["random_seed"])

    def _infer(self, request: InferenceRequest) -> BackendResult:
        context = self._context
        if context is None:
            raise BackendInferenceError("HMMBackend is not fully loaded", code="runtime_not_loaded")
        plan, role_inputs = self._request_execution(request)
        if plan.role_names != context.deployment.execution:
            raise BackendInferenceError(
                f"HMM request execution {list(plan.role_names)} does not match deployment execution "
                f"{list(context.deployment.execution)}",
                code="invalid_request",
            )
        started = time.perf_counter()
        if context.policy.policy_type == "pi05":
            outputs = self._infer_pi05(plan, role_inputs)
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
                "device_id": self._device_id,
                "target_soc": context.target.soc if context.target is not None else None,
                "deployment_name": context.deployment_name,
                "deployment_fingerprint": context.deployment_fingerprint,
            },
        )

    def _close(self) -> None:
        modules = self._modules
        weight_manager = self._weight_manager
        self._modules = {}
        self._weight_manager = None
        self._device_handles = ()
        self._embedding = None
        self._context = None
        self._policy_config = {}
        self._options = {}
        self._random = None
        errors: list[Exception] = []
        for module in reversed(tuple(modules.values())):
            try:
                module.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._release_resource(weight_manager)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _infer_pi05(
        self,
        plan: ExecutionPlan,
        role_inputs: Mapping[str, BoundInputs],
    ) -> dict[str, dict[object, np.ndarray]]:
        vision_roles = plan.role_names[:-6]
        values: dict[str, np.ndarray] = {}
        for role in vision_roles:
            self._execute_role(plan, role, role_inputs[role], values)
        values.update(self._execute_pi05_embedding(plan.role("embedding").bindings, role_inputs["embedding"], values))
        self._execute_role(plan, "prefill", role_inputs["prefill"], values)

        action_in_bindings = plan.role("action_in_proj").bindings
        noise_binding = self._binding_for_semantics(action_in_bindings.inputs, _NOISE_SEMANTICS, "noise")
        external_noise = self._bound_semantics(role_inputs["action_in_proj"]).get(noise_binding.semantic)
        noise = (
            self._convert_runtime_value(noise_binding, external_noise, "action_in_proj", "input")
            if external_noise is not None
            else self._sample_noise(noise_binding)
        )
        steps = self._positive_config_int("num_inference_steps")
        dt = -1.0 / steps
        final_outputs: dict[object, np.ndarray] = {}
        for step in range(steps):
            self._execute_role(
                plan,
                "action_in_proj",
                role_inputs["action_in_proj"],
                values,
                {noise_binding.semantic: noise},
            )
            time_binding = self._binding_for_semantics(
                plan.role("time_mlp").bindings.inputs,
                _TIME_SEMANTICS,
                "time",
            )
            time_value = 1.0 - step / steps
            time_embedding = self._sinusoidal_time_embedding(time_binding, time_value)
            self._execute_role(
                plan,
                "time_mlp",
                role_inputs["time_mlp"],
                values,
                {time_binding.semantic: time_embedding},
            )
            self._execute_role(plan, "decode", role_inputs["decode"], values)
            final_outputs = self._execute_role(
                plan,
                "action_out_proj",
                role_inputs["action_out_proj"],
                values,
            )
            action_binding = self._binding_for_semantics(
                plan.role("action_out_proj").bindings.outputs,
                {"action"},
                "action output",
            )
            velocity = self._output_value(final_outputs, action_binding)
            noise = noise.astype(np.float32) + dt * velocity.astype(np.float32)
        action_binding = self._binding_for_semantics(
            plan.role("action_out_proj").bindings.outputs,
            {"action"},
            "action output",
        )
        final_action = self._convert_runtime_value(action_binding, noise, "action_out_proj", "output")
        return {"action_out_proj": self._replace_output(final_outputs, action_binding, final_action)}

    def _infer_smolvla(
        self,
        plan: ExecutionPlan,
        role_inputs: Mapping[str, BoundInputs],
    ) -> dict[str, dict[object, np.ndarray]]:
        vision_roles = plan.role_names[:-3]
        values: dict[str, np.ndarray] = {}
        for role in vision_roles:
            self._execute_role(plan, role, role_inputs[role], values)
        values.update(
            self._execute_smolvla_embedding(plan.role("embedding").bindings, role_inputs["embedding"], values)
        )
        self._execute_role(plan, "prefill", role_inputs["prefill"], values)

        action_bindings = plan.role("action").bindings
        noise_binding = self._binding_for_semantics(action_bindings.inputs, _NOISE_SEMANTICS, "noise")
        time_binding = self._binding_for_semantics(action_bindings.inputs, _TIME_SEMANTICS, "time")
        external_noise = self._bound_semantics(role_inputs["action"]).get(noise_binding.semantic)
        noise = (
            self._convert_runtime_value(noise_binding, external_noise, "action", "input")
            if external_noise is not None
            else self._sample_noise(noise_binding)
        )
        steps = self._positive_config_int("num_steps")
        dt = -1.0 / steps
        outputs: dict[object, np.ndarray] = {}
        action_binding = self._binding_for_semantics(action_bindings.outputs, {"action"}, "action output")
        for step in range(steps):
            time_value = np.full(
                self._static_shape(time_binding),
                1.0 - step / steps,
                dtype=HMMModule._numpy_dtype(time_binding.dtype),
            )
            outputs = self._execute_role(
                plan,
                "action",
                role_inputs["action"],
                values,
                {
                    noise_binding.semantic: noise,
                    time_binding.semantic: time_value,
                },
            )
            velocity = self._output_value(outputs, action_binding)
            noise = noise.astype(np.float32) + dt * velocity.astype(np.float32)
        final_action = self._convert_runtime_value(action_binding, noise, "action", "output")
        return {"action": self._replace_output(outputs, action_binding, final_action)}

    def _execute_role(
        self,
        plan: ExecutionPlan,
        role: str,
        external_inputs: BoundInputs,
        values: dict[str, np.ndarray],
        overrides: Mapping[str, object] | None = None,
    ) -> dict[object, np.ndarray]:
        semantic_inputs: dict[str, object] = dict(values)
        semantic_inputs.update(self._bound_semantics(external_inputs))
        if overrides:
            semantic_inputs.update(overrides)
        device_semantics = {
            link.semantic
            for link in plan.device_links
            if link.consumer == role or (link.producer == role and link.producer_binding == "input")
        }
        read_semantics = {link.semantic for link in plan.host_links if link.producer == role}
        read_semantics.update(
            binding.semantic for binding in plan.role(role).bindings.outputs if binding.semantic == "action"
        )
        outputs = self._modules[role].execute(
            semantic_inputs,
            device_input_semantics=device_semantics,
            read_semantics=read_semantics,
        )
        for binding in plan.role(role).bindings.outputs:
            if binding.semantic in read_semantics:
                values[binding.semantic] = self._output_value(outputs, binding)
        return outputs

    def _execute_pi05_embedding(
        self,
        bindings: ArtifactBindings,
        external_inputs: BoundInputs,
        host_inputs: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        embedding = self._require_embedding(state_projection=False)
        values: dict[str, object] = dict(host_inputs)
        values.update(self._bound_semantics(external_inputs))
        token_binding = self._binding_for_semantics(
            bindings.inputs,
            {"observation.language.tokens"},
            "language tokens",
        )
        mask_binding = self._binding_for_semantics(
            bindings.inputs,
            {"observation.language.attention_mask"},
            "language mask",
        )
        tokens = np.asarray(values[token_binding.semantic], dtype=np.int64)
        self._validate_token_ids(tokens, embedding.token_weight.shape[0])
        language = embedding.token_weight[tokens] * math.sqrt(embedding.token_weight.shape[1])
        language_mask = np.asarray(values[mask_binding.semantic], dtype=bool)
        image_semantics = [
            binding.semantic for binding in bindings.inputs if binding.semantic.startswith("internal.image_embedding.")
        ]
        images = [np.asarray(values[semantic], dtype=language.dtype) for semantic in image_semantics]
        prefix = np.concatenate((*images, language), axis=1)
        image_masks = [np.ones(image.shape[:2], dtype=bool) for image in images]
        prefix_mask = np.concatenate((*image_masks, language_mask), axis=1)
        actual_length = prefix.shape[1]

        prefix_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.prefix_embeddings"},
            "prefix embeddings",
        )
        attention_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.prefix_attention"},
            "prefix attention",
        )
        decode_attention_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.decode_attention"},
            "decode attention",
        )
        prefix = self._pad_axis_one(prefix, prefix_binding.shape, 0.0)
        prefix_mask = self._pad_axis_one(prefix_mask, prefix_binding.shape[:2], False)

        query_length = attention_binding.shape[-2]
        key_length = attention_binding.shape[-1]
        if query_length != prefix.shape[1] or key_length < actual_length:
            raise BackendInferenceError(
                "PI0.5 prefix attention binding has incompatible dimensions", code="invalid_bindings"
            )
        key_mask = np.zeros((prefix.shape[0], key_length), dtype=bool)
        key_mask[:, : prefix_mask.shape[1]] = prefix_mask
        prefix_attention = prefix_mask[:, None, :, None] & key_mask[:, None, None, :]

        chunk_size = self._positive_config_int("chunk_size")
        decode_key_length = decode_attention_binding.shape[-1]
        if decode_attention_binding.shape[-2] != chunk_size or actual_length + chunk_size > decode_key_length:
            raise BackendInferenceError(
                "PI0.5 decode attention binding has incompatible dimensions", code="invalid_bindings"
            )
        decode_keys = np.zeros((prefix.shape[0], decode_key_length), dtype=bool)
        decode_keys[:, :actual_length] = prefix_mask[:, :actual_length]
        decode_keys[:, actual_length : actual_length + chunk_size] = True
        decode_attention = np.broadcast_to(
            decode_keys[:, None, None, :],
            (prefix.shape[0], 1, chunk_size, decode_key_length),
        ).copy()

        generated: dict[str, object] = {
            prefix_binding.semantic: prefix,
            attention_binding.semantic: self._to_additive_attention(prefix_attention, attention_binding.dtype),
            decode_attention_binding.semantic: self._to_additive_attention(
                decode_attention,
                decode_attention_binding.dtype,
            ),
            "internal.prefill_valid_length": np.zeros((prefix.shape[0],), dtype=np.int32),
            "internal.prefill_current_length": np.full((prefix.shape[0],), actual_length, dtype=np.int32),
            "internal.decode_valid_length": np.full((prefix.shape[0],), actual_length, dtype=np.int32),
            "internal.decode_current_length": np.full((prefix.shape[0],), chunk_size, dtype=np.int32),
        }
        return self._convert_semantic_outputs(bindings.outputs, generated, "embedding")

    def _execute_smolvla_embedding(
        self,
        bindings: ArtifactBindings,
        external_inputs: BoundInputs,
        host_inputs: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        embedding = self._require_embedding(state_projection=True)
        assert embedding.state_weight is not None
        assert embedding.state_bias is not None
        values: dict[str, object] = dict(host_inputs)
        values.update(self._bound_semantics(external_inputs))
        token_binding = self._binding_for_semantics(
            bindings.inputs,
            {"observation.language.tokens"},
            "language tokens",
        )
        mask_binding = self._binding_for_semantics(
            bindings.inputs,
            {"observation.language.attention_mask"},
            "language mask",
        )
        state_binding = self._binding_for_semantics(bindings.inputs, {"observation.state"}, "state")
        tokens = np.asarray(values[token_binding.semantic], dtype=np.int64)
        self._validate_token_ids(tokens, embedding.token_weight.shape[0])
        language = embedding.token_weight[tokens]
        state = np.asarray(values[state_binding.semantic], dtype=np.float32)
        state_embedding = state @ embedding.state_weight.T + embedding.state_bias
        state_embedding = state_embedding[:, None, :]
        image_semantics = [
            binding.semantic for binding in bindings.inputs if binding.semantic.startswith("internal.image_embedding.")
        ]
        images = [np.asarray(values[semantic], dtype=language.dtype) for semantic in image_semantics]
        hidden_size = embedding.token_weight.shape[1]
        prefix = np.concatenate(
            (*(image * math.sqrt(hidden_size) for image in images), language * math.sqrt(hidden_size), state_embedding),
            axis=1,
        )
        language_mask = np.asarray(values[mask_binding.semantic], dtype=bool)
        image_masks = [np.ones(image.shape[:2], dtype=bool) for image in images]
        state_mask = np.ones(state_embedding.shape[:2], dtype=bool)
        prefix_mask = np.concatenate((*image_masks, language_mask, state_mask), axis=1)
        attention_markers = np.concatenate(
            (
                *(np.zeros(image.shape[:2], dtype=np.int32) for image in images),
                np.zeros(language_mask.shape, dtype=np.int32),
                np.ones(state_mask.shape, dtype=np.int32),
            ),
            axis=1,
        )

        prefix_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.prefix_embeddings"},
            "prefix embeddings",
        )
        pad_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.prefix_pad_masks"},
            "prefix pad masks",
        )
        attention_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.attention_mask"},
            "attention mask",
        )
        position_binding = self._binding_for_semantics(
            bindings.outputs,
            {"internal.position_ids"},
            "position ids",
        )
        prefix = self._pad_axis_one(prefix, prefix_binding.shape, 0.0)
        prefix_mask = self._pad_axis_one(prefix_mask, pad_binding.shape, False)
        attention_markers = self._pad_axis_one(attention_markers, pad_binding.shape, 0)
        cumulative = np.cumsum(attention_markers, axis=1)
        attention = cumulative[:, None, :] <= cumulative[:, :, None]
        attention &= prefix_mask[:, None, :] & prefix_mask[:, :, None]
        position_ids = np.cumsum(prefix_mask.astype(np.int32), axis=1) - 1
        position_ids = np.where(prefix_mask, position_ids, 0)
        generated = {
            prefix_binding.semantic: prefix,
            pad_binding.semantic: prefix_mask,
            attention_binding.semantic: attention,
            position_binding.semantic: position_ids,
        }
        return self._convert_semantic_outputs(bindings.outputs, generated, "embedding")

    def _convert_semantic_outputs(
        self,
        bindings: Sequence[TensorBinding],
        values: Mapping[str, object],
        role: str,
    ) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        for binding in bindings:
            try:
                value = values[binding.semantic]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"HMM CPU role {role!r} did not generate {binding.semantic!r}",
                    code="missing_runtime_output",
                ) from exc
            outputs[binding.semantic] = self._convert_runtime_value(binding, value, role, "output")
        return outputs

    def _sinusoidal_time_embedding(self, binding: TensorBinding, time_value: float) -> np.ndarray:
        shape = self._static_shape(binding)
        if len(shape) != 2 or shape[-1] % 2 != 0:
            raise BackendInferenceError(
                f"PI0.5 time MLP input requires shape (B, even_dimension), got {shape}",
                code="invalid_time_binding",
            )
        fraction = np.linspace(0.0, 1.0, shape[-1] // 2, dtype=np.float64)
        min_period = float(self._policy_config.get("min_period", 0.004))
        max_period = float(self._policy_config.get("max_period", 4.0))
        period = min_period * (max_period / min_period) ** fraction
        scaled = time_value * (2.0 * math.pi / period)
        value = np.concatenate((np.sin(scaled), np.cos(scaled)))[None, :]
        if shape[0] != 1:
            value = np.broadcast_to(value, shape)
        return self._convert_runtime_value(binding, value, "time_mlp", "input")

    def _sample_noise(self, binding: TensorBinding) -> np.ndarray:
        if self._random is None:
            raise BackendInferenceError("HMM random generator is unavailable", code="runtime_not_loaded")
        return np.ascontiguousarray(
            self._random.standard_normal(self._static_shape(binding)).astype(HMMModule._numpy_dtype(binding.dtype))
        )

    def _require_embedding(self, *, state_projection: bool) -> _EmbeddingWeights:
        embedding = self._embedding
        if embedding is None:
            raise BackendInferenceError("HMM embedding artifacts are unavailable", code="runtime_not_loaded")
        if state_projection and (embedding.state_weight is None or embedding.state_bias is None):
            raise BackendInferenceError("HMM state projection is unavailable", code="runtime_not_loaded")
        return embedding

    def _positive_config_int(self, key: str) -> int:
        value = self._policy_config.get(key)
        if type(value) is not int or value < 1:
            raise BackendInferenceError(
                f"HMM policy requires positive integer {key!r} in LeRobot config",
                code="invalid_policy_config",
            )
        return value

    @classmethod
    def _validate_pi05_plan(
        cls,
        deployment: CompiledDeployment,
        policy_config: Mapping[str, object],
        embedding: _EmbeddingWeights,
    ) -> None:
        suffix = ("embedding", "prefill", "action_in_proj", "time_mlp", "decode", "action_out_proj")
        vision_roles = deployment.execution[: -len(suffix)]
        if (
            not vision_roles
            or deployment.execution[-len(suffix) :] != suffix
            or any(role != "vision" and not role.startswith("vision_") for role in vision_roles)
        ):
            raise BackendLoadError(
                "HMM PI0.5 requires vision role(s) followed by embedding, prefill, action_in_proj, "
                "time_mlp, decode, and action_out_proj",
                code="invalid_execution_plan",
            )
        for key in ("chunk_size", "max_action_dim", "num_inference_steps"):
            cls._require_positive_config(policy_config, key, "PI0.5")
        if embedding.token_weight.ndim != 2:
            raise BackendLoadError("HMM PI0.5 token embedding must be rank 2", code="invalid_embedding")
        input_links = [link for link in deployment.device_links if link.producer_binding == "input"]
        if not input_links or any(link.producer != "prefill" or link.consumer != "decode" for link in input_links):
            raise BackendLoadError(
                "HMM PI0.5 requires prefill input to decode input device links",
                code="invalid_device_links",
            )
        if any(link.producer_binding != "input" for link in deployment.device_links):
            raise BackendLoadError("HMM PI0.5 supports only input-sourced cache links", code="invalid_device_links")
        noise = cls._binding_for_semantics(
            deployment.bindings["action_in_proj"].inputs,
            _NOISE_SEMANTICS,
            "noise",
        )
        action = cls._binding_for_semantics(
            deployment.bindings["action_out_proj"].outputs,
            {"action"},
            "action output",
        )
        expected = (1, int(policy_config["chunk_size"]), int(policy_config["max_action_dim"]))
        if noise.shape != expected or action.shape != expected:
            raise BackendLoadError(
                f"HMM PI0.5 noise and action bindings must use shape {expected}",
                code="invalid_bindings",
            )
        cls._validate_vision_embedding_bindings(deployment, vision_roles)

    @classmethod
    def _validate_smolvla_plan(
        cls,
        deployment: CompiledDeployment,
        policy_config: Mapping[str, object],
        embedding: _EmbeddingWeights,
    ) -> None:
        suffix = ("embedding", "prefill", "action")
        vision_roles = deployment.execution[: -len(suffix)]
        if (
            not vision_roles
            or deployment.execution[-len(suffix) :] != suffix
            or any(role != "vision" and not role.startswith("vision_") for role in vision_roles)
        ):
            raise BackendLoadError(
                "HMM SmolVLA requires vision role(s) followed by embedding, prefill, and action",
                code="invalid_execution_plan",
            )
        for key in ("chunk_size", "max_state_dim", "max_action_dim", "num_steps"):
            cls._require_positive_config(policy_config, key, "SmolVLA")
        if policy_config.get("add_image_special_tokens", False) is not False:
            raise BackendLoadError(
                "HMM SmolVLA does not support add_image_special_tokens=true",
                code="unsupported_policy_config",
            )
        if embedding.state_weight is None or embedding.state_bias is None:
            raise BackendLoadError("HMM SmolVLA requires state projection weights", code="invalid_embedding")
        state_binding = cls._binding_for_semantics(
            deployment.bindings["embedding"].inputs,
            {"observation.state"},
            "state",
        )
        hidden_size = embedding.token_weight.shape[1]
        if embedding.state_weight.shape != (hidden_size, state_binding.shape[-1]):
            raise BackendLoadError(
                "HMM SmolVLA state projection shape is incompatible with the embedding binding",
                code="invalid_embedding",
            )
        if embedding.state_bias.shape != (hidden_size,):
            raise BackendLoadError("HMM SmolVLA state projection bias has invalid shape", code="invalid_embedding")
        if not deployment.device_links or any(
            link.producer != "prefill" or link.consumer != "action" or link.producer_binding != "output"
            for link in deployment.device_links
        ):
            raise BackendLoadError(
                "HMM SmolVLA requires prefill output to action input device links",
                code="invalid_device_links",
            )
        noise = cls._binding_for_semantics(deployment.bindings["action"].inputs, _NOISE_SEMANTICS, "noise")
        action = cls._binding_for_semantics(deployment.bindings["action"].outputs, {"action"}, "action output")
        expected = (1, int(policy_config["chunk_size"]), int(policy_config["max_action_dim"]))
        if noise.shape != expected or action.shape != expected:
            raise BackendLoadError(
                f"HMM SmolVLA noise and action bindings must use shape {expected}",
                code="invalid_bindings",
            )
        cls._validate_vision_embedding_bindings(deployment, vision_roles)

    @classmethod
    def _validate_vision_embedding_bindings(
        cls,
        deployment: CompiledDeployment,
        vision_roles: Sequence[str],
    ) -> None:
        image_outputs: list[str] = []
        for role in vision_roles:
            bindings = deployment.bindings[role]
            if len(bindings.inputs) != 1 or len(bindings.outputs) != 1:
                raise BackendLoadError(
                    f"HMM vision role {role!r} requires exactly one input and one output",
                    code="invalid_bindings",
                )
            if not cls._is_image_semantic(bindings.inputs[0].semantic) or not bindings.outputs[0].semantic.startswith(
                "internal.image_embedding."
            ):
                raise BackendLoadError(
                    f"HMM vision role {role!r} must map one image to one internal image embedding",
                    code="invalid_bindings",
                )
            image_outputs.append(bindings.outputs[0].semantic)
        embedding_images = [
            binding.semantic
            for binding in deployment.bindings["embedding"].inputs
            if binding.semantic.startswith("internal.image_embedding.")
        ]
        if image_outputs != embedding_images:
            raise BackendLoadError(
                "HMM embedding image inputs must match vision execution order",
                code="invalid_bindings",
            )

    def _load_policy_artifacts(self, context: RuntimeContext) -> _EmbeddingWeights:
        embedding_path = self._require_artifact(context, "embedding")
        embedding_artifact = context.deployment.artifacts["embedding"]
        if embedding_artifact.format not in {"pt", "pytorch"}:
            raise BackendLoadError(
                "HMM embedding artifact must use format 'pt' or 'pytorch'",
                code="invalid_artifact_format",
            )
        token_state = self._load_torch_mapping(embedding_path, "embedding")
        token_weight = token_state.get("token_embedding.weight", token_state.get("weight"))
        if token_weight is None:
            raise BackendLoadError("HMM embedding artifact does not contain token weights", code="invalid_embedding")
        if context.policy.policy_type != "smolvla":
            return _EmbeddingWeights(token_weight=self._to_numpy_weight(token_weight, embedding_path, "weight"))

        try:
            projection_artifact = context.deployment.artifacts["state_projection"]
        except KeyError as exc:
            raise BackendLoadError(
                "HMM SmolVLA requires a manifest-declared state_projection artifact",
                code="missing_artifact_role",
            ) from exc
        if projection_artifact.format not in {"pt", "pytorch"}:
            raise BackendLoadError(
                "HMM state_projection artifact must use format 'pt' or 'pytorch'",
                code="invalid_artifact_format",
            )
        projection_path = self._require_artifact(context, "state_projection")
        projection_state = self._load_torch_mapping(projection_path, "state_projection")
        state_weight = projection_state.get("state_proj.weight", projection_state.get("weight"))
        state_bias = projection_state.get("state_proj.bias", projection_state.get("bias"))
        if state_weight is None or state_bias is None:
            raise BackendLoadError(
                "HMM state_projection artifact must contain weight and bias",
                code="invalid_embedding",
            )
        return _EmbeddingWeights(
            token_weight=self._to_numpy_weight(token_weight, embedding_path, "weight"),
            state_weight=self._to_numpy_weight(state_weight, projection_path, "weight"),
            state_bias=self._to_numpy_weight(state_bias, projection_path, "bias"),
        )

    @staticmethod
    def _load_torch_mapping(path: Path, description: str) -> Mapping[str, object]:
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"HMM {description} requires PyTorch to load {path}: {exc}",
                code="missing_dependency",
            ) from exc
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to load HMM {description} artifact {path}: {exc}", code="invalid_embedding"
            ) from exc
        if not isinstance(value, Mapping):
            raise BackendLoadError(
                f"HMM {description} artifact must contain a tensor mapping", code="invalid_embedding"
            )
        return value

    @staticmethod
    def _to_numpy_weight(value: object, path: Path, name: str) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        try:
            return np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        except (TypeError, ValueError) as exc:
            raise BackendLoadError(
                f"HMM artifact {path} contains invalid tensor {name!r}",
                code="invalid_embedding",
            ) from exc

    @staticmethod
    def _request_execution(request: InferenceRequest) -> tuple[ExecutionPlan, Mapping[str, BoundInputs]]:
        plan = request.inputs.get("execution_plan")
        role_inputs = request.inputs.get("role_inputs")
        if not isinstance(plan, ExecutionPlan):
            raise BackendInferenceError("HMMBackend request is missing execution_plan", code="invalid_request")
        if not isinstance(role_inputs, Mapping):
            raise BackendInferenceError("HMMBackend request is missing role_inputs", code="invalid_request")
        for role in plan.role_names:
            if not isinstance(role_inputs.get(role), BoundInputs):
                raise BackendInferenceError(
                    f"HMMBackend role {role!r} inputs are not bound tensors",
                    code="invalid_request",
                )
        return plan, role_inputs

    @staticmethod
    def _bound_semantics(inputs: BoundInputs) -> dict[str, np.ndarray]:
        return {tensor.semantic: tensor.value for tensor in inputs.tensors}

    @staticmethod
    def _output_value(outputs: Mapping[object, np.ndarray], binding: TensorBinding) -> np.ndarray:
        if binding.runtime_name is not None and binding.runtime_name in outputs:
            return outputs[binding.runtime_name]
        if binding.index is not None and int(binding.index) in outputs:
            return outputs[int(binding.index)]
        raise BackendInferenceError(
            f"HMM runtime did not return output {binding.semantic!r}",
            code="missing_runtime_output",
        )

    @staticmethod
    def _replace_output(
        outputs: Mapping[object, np.ndarray],
        binding: TensorBinding,
        value: np.ndarray,
    ) -> dict[object, np.ndarray]:
        result = dict(outputs)
        if binding.index is not None:
            result[int(binding.index)] = value
        if binding.runtime_name is not None:
            result[binding.runtime_name] = value
        return result

    @staticmethod
    def _raw_action(outputs: object, plan: ExecutionPlan) -> np.ndarray:
        action_role = next(
            role for role in plan.roles if any(binding.semantic == "action" for binding in role.bindings.outputs)
        )
        binding = next(binding for binding in action_role.bindings.outputs if binding.semantic == "action")
        role_outputs = (
            outputs[action_role.name] if isinstance(outputs, Mapping) and action_role.name in outputs else outputs
        )
        if not isinstance(role_outputs, Mapping):
            raise BackendInferenceError("HMM runtime did not return bound action outputs", code="missing_action_output")
        return HMMBackend._output_value(role_outputs, binding)

    @staticmethod
    def _chunk_size(action: np.ndarray) -> int:
        if action.ndim < 2 or action.shape[-2] < 1:
            raise BackendInferenceError(
                f"HMM action output has invalid shape {action.shape}",
                code="invalid_action_shape",
            )
        return int(action.shape[-2])

    @staticmethod
    def _binding_for_semantics(
        bindings: Sequence[TensorBinding],
        semantics: set[str] | frozenset[str],
        description: str,
    ) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic in semantics]
        if len(matches) != 1:
            raise BackendLoadError(
                f"HMM deployment requires exactly one {description} binding",
                code="invalid_bindings",
            )
        return matches[0]

    @staticmethod
    def _binding_for_semantic(
        bindings: Sequence[TensorBinding],
        semantic: str,
        description: str,
    ) -> TensorBinding:
        return HMMBackend._binding_for_semantics(bindings, {semantic}, description)

    @staticmethod
    def _static_shape(binding: TensorBinding) -> tuple[int, ...]:
        if any(dimension < 1 for dimension in binding.shape):
            raise BackendInferenceError(
                f"HMM runtime-generated input {binding.semantic!r} requires a static shape, got {binding.shape}",
                code="dynamic_runtime_input",
            )
        return binding.shape

    @staticmethod
    def _convert_runtime_value(
        binding: TensorBinding,
        value: object,
        role: str,
        direction: str,
    ) -> np.ndarray:
        try:
            converted = np.ascontiguousarray(np.asarray(value, dtype=HMMModule._numpy_dtype(binding.dtype)))
        except (TypeError, ValueError) as exc:
            raise BackendInferenceError(
                f"HMM role {role!r} {direction} {binding.semantic!r} cannot convert to {binding.dtype}",
                code=f"runtime_{direction}_dtype_mismatch",
            ) from exc
        if not HMMModule._compatible_shape(binding.shape, converted.shape):
            raise BackendInferenceError(
                f"HMM role {role!r} {direction} {binding.semantic!r} shape {converted.shape} "
                f"does not match manifest shape {binding.shape}",
                code=f"runtime_{direction}_shape_mismatch",
            )
        return converted

    @staticmethod
    def _pad_axis_one(value: np.ndarray, shape: tuple[int, ...], pad_value: object) -> np.ndarray:
        if len(shape) != value.ndim or shape[0] not in {-1, value.shape[0]}:
            raise BackendInferenceError(
                f"HMM prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
                code="invalid_prefix_shape",
            )
        target_length = shape[1]
        if target_length < value.shape[1] or any(
            expected != -1 and expected != actual for expected, actual in zip(shape[2:], value.shape[2:], strict=True)
        ):
            raise BackendInferenceError(
                f"HMM prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
                code="invalid_prefix_shape",
            )
        if target_length == value.shape[1]:
            return value
        pad_shape = list(value.shape)
        pad_shape[1] = target_length - value.shape[1]
        return np.concatenate((value, np.full(pad_shape, pad_value, dtype=value.dtype)), axis=1)

    @staticmethod
    def _to_additive_attention(mask: np.ndarray, dtype: str) -> np.ndarray:
        target_dtype = HMMModule._numpy_dtype(dtype)
        minimum = np.finfo(target_dtype).min
        return np.where(mask, 0.0, minimum).astype(target_dtype)

    @staticmethod
    def _validate_token_ids(tokens: np.ndarray, vocabulary_size: int) -> None:
        if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= vocabulary_size:
            raise BackendInferenceError("HMM token id is outside the embedding table", code="invalid_token_id")

    @staticmethod
    def _is_image_semantic(semantic: str) -> bool:
        return (
            semantic == "observation.image"
            or semantic.startswith("observation.image.")
            or semantic.startswith("observation.images.")
        )

    @staticmethod
    def _require_positive_config(config: Mapping[str, object], key: str, policy: str) -> None:
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise BackendLoadError(
                f"HMM {policy} requires positive integer {key!r} in LeRobot config",
                code="invalid_policy_config",
            )

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        try:
            path = context.resolved_artifacts[role]
        except KeyError as exc:
            raise BackendLoadError(
                f"HMM deployment is missing artifact role {role!r}", code="missing_artifact_role"
            ) from exc
        if not path.is_file():
            raise BackendLoadError(f"HMM artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
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
    def _create_weight_manager(runtime: object, device_id: int) -> object | None:
        constructor = getattr(runtime, "WeightManager", None)
        if not callable(constructor):
            return None
        try:
            return constructor(device=device_id)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to create TCIM WeightManager for device {device_id}: {exc}",
                code="runtime_load_failed",
            ) from exc

    @staticmethod
    def _create_option(runtime: object, weight_manager: object | None) -> object | None:
        constructor = getattr(runtime, "Option", None)
        if weight_manager is None or not callable(constructor):
            return None
        return constructor(weight_manager)

    @staticmethod
    def _release_resource(resource: object | None) -> None:
        if resource is None:
            return
        for method_name in ("release", "close", "destroy"):
            method = getattr(resource, method_name, None)
            if callable(method):
                method()
                return

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
        unknown = sorted(set(options) - _ALLOWED_RUNTIME_OPTIONS)
        if unknown:
            raise BackendLoadError(f"unknown HMM runtime options: {unknown}", code="invalid_runtime_options")
        device_id = options.get("device_id", 0)
        if type(device_id) is not int or device_id < 0:
            raise BackendLoadError("HMM device_id must be a non-negative integer", code="invalid_runtime_options")
        random_seed = options.get("random_seed")
        if random_seed is not None and type(random_seed) is not int:
            raise BackendLoadError("HMM random_seed must be an integer or null", code="invalid_runtime_options")
        return {"device_id": device_id, "random_seed": random_seed}

    @staticmethod
    def _import_tcim_runtime() -> object:
        try:
            module = importlib.import_module("tcim_lite")
            return module.runtime
        except (ImportError, OSError, AttributeError) as exc:
            raise BackendLoadError(
                f"TCIM dependency 'tcim_lite.runtime' is unavailable: {exc}",
                code="missing_dependency",
            ) from exc


def create_backend(context: RuntimeContext) -> HMMBackend:
    """Lazy registry factory for Houmo TCIM execution."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hmm":
        raise BackendLoadError("HMMBackend requires a compiled hmm deployment", code="invalid_deployment")
    options = HMMBackend._validate_runtime_options(context.runtime_options)
    return HMMBackend(int(options["device_id"]))
