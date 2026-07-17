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
