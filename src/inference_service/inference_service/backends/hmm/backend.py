"""Houmo TCIM device runtime primitives used by :class:`HMMModelSession`."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLoadError

_ALLOWED_RUNTIME_OPTIONS = frozenset({"device_id", "random_seed"})


def validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
    """Validate HMM options shared by the backend, session, and pipeline factory."""

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
