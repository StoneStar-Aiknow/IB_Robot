"""Houmo TCIM backend entry point and shared host-resource helpers.

PI0.5 and SmolVLA compiled deployments now run through ``HMMModelSession`` and
the shared family executors.  ``HMMBackend`` itself is fail-closed for every
policy family and is retained for registry construction, runtime-option
validation, and the shared host-resource helper statics consumed by the
PI0.5/SmolVLA executor host stages.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendResult, InferenceRequest, RuntimeContext

_ALLOWED_RUNTIME_OPTIONS = frozenset({"device_id", "random_seed"})
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})


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
    """Fail-closed registry entry point plus shared host-resource helpers.

    PI0.5 and SmolVLA have migrated to ``HMMModelSession`` and the shared family
    executors; this backend fails closed for every policy family so the factory
    routing is unambiguous. It is retained for registry construction, runtime
    option validation, and the shared host-resource helper statics used by the
    PI0.5/SmolVLA executor host stages.
    """

    def __init__(
        self,
        device_id: int = 0,
        *,
        runtime_loader: Callable[[], object] | None = None,
        expose_hardware_identity: bool = False,
    ) -> None:
        super().__init__(
            "hmm",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                hardware_resource_id=f"hmm:{device_id}" if expose_hardware_identity else None,
                resource_domain=f"hmm:{device_id}",
                max_in_flight_per_resource_domain=1,
            ),
        )
        self._device_id = device_id
        self._runtime_loader = runtime_loader

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        del rollback
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hmm":
            raise BackendLoadError("HMMBackend requires a compiled hmm deployment", code="invalid_deployment")
        policy = context.policy.policy_type
        if policy == "pi05":
            raise BackendLoadError(
                "HMMBackend no longer hosts PI0.5; PI0.5 runs through HMMModelSession",
                code="unsupported_policy_backend_pair",
            )
        if policy == "smolvla":
            raise BackendLoadError(
                "HMMBackend no longer hosts SmolVLA; SmolVLA runs through HMMModelSession",
                code="unsupported_policy_backend_pair",
            )
        raise BackendLoadError(
            f"HMMBackend does not support policy family {policy!r}",
            code="unsupported_policy_backend_pair",
        )

    def _infer(self, request: InferenceRequest) -> BackendResult:
        del request
        raise BackendInferenceError(
            "HMMBackend is fail-closed; compiled policies run through HMMModelSession",
            code="unsupported_policy_backend_pair",
        )

    def _close(self) -> None:
        return

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
        if not deployment.device_links or any(
            link.producer != "prefill" or link.consumer != "decode" or link.producer_binding != "output"
            for link in deployment.device_links
        ):
            raise BackendLoadError(
                "HMM PI0.5 requires prefill output to decode input device links",
                code="invalid_device_links",
            )
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

    @staticmethod
    def _convert_semantic_outputs(
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
            outputs[binding.semantic] = HMMBackend._convert_runtime_value(binding, value, role, "output")
        return outputs

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


def create_backend(context: RuntimeContext) -> HMMBackend:
    """Lazy registry factory for Houmo TCIM execution."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "hmm":
        raise BackendLoadError("HMMBackend requires a compiled hmm deployment", code="invalid_deployment")
    options = HMMBackend._validate_runtime_options(context.runtime_options)
    return HMMBackend(
        int(options["device_id"]),
        expose_hardware_identity=context.priority_scheduling,
    )
