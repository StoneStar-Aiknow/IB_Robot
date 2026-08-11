"""Low-level RKNNLite runtime helpers shared by model sessions and factories."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import RuntimeContext

_ALLOWED_RUNTIME_OPTIONS = frozenset({"target", "core_mask", "random_seed"})


def validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
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

    def infer(self, inputs: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        if self._closed:
            raise BackendInferenceError(f"RKNN role {self.role!r} is closed", code="runtime_not_loaded")
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


def import_rknn_type() -> type:
    try:
        module = importlib.import_module("rknnlite.api")
        return module.RKNNLite
    except (ImportError, OSError, AttributeError) as exc:
        raise BackendLoadError(
            f"RKNNLite dependency 'rknnlite.api.RKNNLite' is unavailable: {exc}",
            code="missing_dependency",
        ) from exc


def resolve_core_mask(rknn_type: type, value: object) -> int:
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


def session_cache_key(deployment: CompiledDeployment, role: str) -> tuple[object, ...]:
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


def runtime_data_format(bindings: ArtifactBindings) -> str | None:
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


def require_artifact(context: RuntimeContext, role: str) -> Path:
    try:
        path = context.resolved_artifacts[role]
    except KeyError as exc:
        raise BackendLoadError(
            f"RKNN deployment is missing artifact role {role!r}", code="missing_artifact_role"
        ) from exc
    if not path.is_file():
        raise BackendLoadError(f"RKNN artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
    return path


def convert_runtime_value(
    binding: TensorBinding,
    value: object,
    *,
    role: str,
    direction: str,
) -> np.ndarray:
    try:
        converted = np.ascontiguousarray(np.asarray(value, dtype=numpy_dtype(binding.dtype)))
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
        expected != -1 and expected != actual for expected, actual in zip(binding.shape, converted.shape, strict=True)
    ):
        raise BackendInferenceError(
            f"RKNN role {role!r} {direction} {binding.semantic!r} shape {converted.shape} "
            f"does not match manifest shape {binding.shape}",
            code=f"runtime_{direction}_shape_mismatch",
        )
    return converted


def numpy_dtype(dtype: str) -> np.dtype:
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
