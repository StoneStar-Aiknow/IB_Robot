from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from inference_service.core._policy_config import override_runtime_policy_device
from inference_service.core.ascend_om._sd3403_action import decode_sd3403_action_array
from inference_service.core.pure_inference_engine import PolicyWrapper

COMPILED_MANIFEST_BASENAME = "config.om.json"
COMPILED_BACKEND_CONFIG_KEY = "_compiled_backend_config"
DEFAULT_SD3403_ACTION_OUTPUT_INDEX = 1
DEFAULT_SD3403_PERF_ENABLED = False
DEFAULT_SD3403_PERF_LOG_EVERY = 1
SD3403_OUTPUT_LAYOUTS = {"direct", "strided"}
# Image keys consulted (in order) when deriving resize dims from config.json
# input_features. The first one with a usable [C,H,W] shape wins.
_SD3403_IMAGE_FEATURE_KEYS = ("observation.images.top", "observation.images.wrist")

# Per-backend compiled manifest filenames. The OM backend keeps the historical
# ``config.om.json``; the Houmo HMM backend uses ``config.hmm.json``.
_MANIFEST_BASENAMES: dict[str, str] = {
    "ascend_om": "config.om.json",
    "ascend_om_3403": "config.om.json",
    "hmm": "config.hmm.json",
    "rknn": "config.rknn.json",
}


def manifest_basename_for_backend(backend: str) -> str:
    """Return the compiled-manifest filename expected for a backend."""
    normalized = normalize_backend_name(backend)
    return _MANIFEST_BASENAMES.get(normalized, COMPILED_MANIFEST_BASENAME)


@dataclass(frozen=True)
class CompiledManifest:
    artifacts: dict[str, Path]
    execution: list[str]
    backend_config: dict[str, Any]

    def require_artifact(self, role: str, *, suffix: str | None = None) -> Path:
        try:
            artifact = self.artifacts[role]
        except KeyError as exc:
            roles = ", ".join(sorted(self.artifacts)) or "<none>"
            raise KeyError(f"Compiled manifest is missing artifact role {role!r}; available roles: {roles}") from exc
        if suffix is not None and artifact.suffix.lower() != suffix:
            raise ValueError(f"Compiled artifact {role!r} must be a {suffix} file: {artifact}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Compiled artifact {role!r} does not exist: {artifact}")
        return artifact.resolve()

    def require_execution(self, expected: list[str]) -> None:
        if not self.execution:
            return
        if self.execution != expected:
            raise ValueError(f"Compiled manifest execution must be {expected}, got {self.execution}")


class CompiledModelAdapter(Protocol):
    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> CompiledModelAdapter: ...

    def prepare_inputs(self, batch: dict[str, Tensor]) -> Any: ...

    def decode_outputs(self, raw: Any, device: torch.device) -> Tensor: ...

    def get_chunk_size(self) -> int: ...

    @property
    def policy_type(self) -> str: ...

    @property
    def uses_action_chunking(self) -> bool: ...


class RuntimeSession(Protocol):
    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None: ...

    def execute(self, inputs: Any) -> Any: ...

    def release(self) -> None: ...


def normalize_backend_name(device: str) -> str:
    return str(device).lower().strip().replace("-", "_")


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{key} must be a boolean value, got {value!r}")
    return bool(value)


def _as_optional_dict(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object, got {type(value).__name__}")
    return value


def _validate_sd3403_output_layouts(backend_config: dict[str, Any]) -> None:
    for section in ("action_output",):
        section_config = _as_optional_dict(backend_config.get(section), f"backend_config.{section}")
        if "layout" not in section_config:
            continue
        layout = str(section_config["layout"]).lower().strip()
        if layout not in SD3403_OUTPUT_LAYOUTS:
            allowed = ", ".join(sorted(SD3403_OUTPUT_LAYOUTS))
            raise ValueError(f"backend_config.{section}.layout must be one of {allowed}, got {layout!r}")


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(_config_value(config, key, default))


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    return float(_config_value(config, key, default))


def _sd3403_backend_int(
    backend_config: dict[str, Any],
    legacy_config: dict[str, Any],
    *,
    key: str,
    legacy_key: str,
    default: int,
    section: str | None = None,
    flat_key: str | None = None,
    top_key: str | None = None,
) -> int:
    """Resolve an int from ``backend_config`` with sectioned, flat and legacy fallbacks.

    Lookup order: ``backend_config[section][key]`` -> ``backend_config[flat_key]``
    -> ``backend_config[top_key]`` -> legacy config. ``top_key`` defaults to
    ``key``; callers whose section key is generic (e.g. ``"index"``) pass a
    distinct ``top_key`` so a stray top-level entry cannot be matched by accident.
    """
    if section:
        section_config = _as_optional_dict(backend_config.get(section), f"backend_config.{section}")
        if key in section_config:
            return int(section_config[key])
    if flat_key is not None and flat_key in backend_config:
        return int(backend_config[flat_key])
    effective_top_key = top_key if top_key is not None else key
    if effective_top_key in backend_config:
        return int(backend_config[effective_top_key])
    return _config_int(legacy_config, legacy_key, default)


def _sd3403_backend_float(
    backend_config: dict[str, Any],
    legacy_config: dict[str, Any],
    *,
    key: str,
    legacy_key: str,
    default: float,
) -> float:
    if key in backend_config:
        return float(backend_config[key])
    return _config_float(legacy_config, legacy_key, default)


def _sd3403_backend_bool(
    backend_config: dict[str, Any],
    legacy_config: dict[str, Any],
    *,
    key: str,
    legacy_key: str,
    default: bool,
) -> bool:
    if key in backend_config:
        return _config_bool(backend_config, key, default)
    return _config_bool(legacy_config, legacy_key, default)


def _compiled_backend_config(config: dict[str, Any]) -> dict[str, Any]:
    return _as_optional_dict(config.get(COMPILED_BACKEND_CONFIG_KEY), COMPILED_BACKEND_CONFIG_KEY)


def _attach_compiled_backend_config(path: str, backend: str, config: dict[str, Any]) -> dict[str, Any]:
    if normalize_backend_name(backend) != "ascend_om_3403":
        return config

    manifest = load_compiled_manifest(path, backend, str(config.get("type", "")).lower().strip())
    if not manifest.backend_config:
        return config

    merged = dict(config)
    merged[COMPILED_BACKEND_CONFIG_KEY] = manifest.backend_config
    return merged


def _policy_config_path(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        return candidate
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if config_path.is_file():
            return config_path
    return None


def _manifest_config_path(path: str, basename: str = COMPILED_MANIFEST_BASENAME) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == basename:
        return candidate
    if candidate.is_dir():
        manifest_path = candidate / basename
        if manifest_path.is_file():
            return manifest_path
    return None


def load_compiled_policy_config(
    path: str,
    backend: str,
    runtime_device: Any | None = None,
) -> dict[str, Any]:
    config_path = _policy_config_path(path)
    if config_path is None:
        raise FileNotFoundError(
            f"Compiled backend {backend} requires policy_path/config.json with policy type metadata"
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Compiled backend {backend} policy config must be a JSON object: {config_path}")
    return override_runtime_policy_device(data, runtime_device)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return data


def _resolve_manifest_artifact_path(base_dir: Path, value: Any, role: str) -> Path:
    if isinstance(value, str):
        artifact_path = Path(value).expanduser()
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        artifact_path = Path(value["path"]).expanduser()
    else:
        raise ValueError(f"Compiled manifest artifact {role!r} must be a path string or object with path")
    if not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path
    return artifact_path


def load_compiled_manifest(path: str, backend: str, policy_type: str | None = None) -> CompiledManifest | None:
    basename = manifest_basename_for_backend(backend)
    manifest_path = _manifest_config_path(path, basename)
    if manifest_path is None:
        raise FileNotFoundError(f"Compiled backend {backend} requires {basename} under policy_path {path}")
    data = _read_json_object(manifest_path)
    manifest_backend = str(data.get("backend", "")).lower().strip()
    if manifest_backend and normalize_backend_name(manifest_backend) != normalize_backend_name(backend):
        raise ValueError(
            f"Compiled manifest backend {manifest_backend!r} does not match requested backend {backend!r}: {manifest_path}"
        )
    manifest_policy = str(data.get("policy_type", "")).lower().strip()
    if policy_type and manifest_policy and manifest_policy != policy_type:
        raise ValueError(
            f"Compiled manifest policy_type {manifest_policy!r} does not match config type {policy_type!r}: {manifest_path}"
        )

    artifact_dir = data.get("artifact_dir", "")
    base_dir = manifest_path.parent
    if isinstance(artifact_dir, str) and artifact_dir:
        artifact_base = Path(artifact_dir).expanduser()
        if not artifact_base.is_absolute():
            artifact_base = base_dir / artifact_base
        base_dir = artifact_base

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError(f"Compiled manifest must define non-empty artifacts map: {manifest_path}")
    artifacts = {
        str(role): _resolve_manifest_artifact_path(base_dir, value, str(role)) for role, value in raw_artifacts.items()
    }

    raw_execution = data.get("execution", [])
    if raw_execution is None:
        execution: list[str] = []
    elif isinstance(raw_execution, list):
        execution = [str(role) for role in raw_execution]
    else:
        raise ValueError(f"Compiled manifest execution must be a list of artifact roles: {manifest_path}")

    backend_config = _as_optional_dict(data.get("backend_config"), "backend_config")
    if normalize_backend_name(backend) == "ascend_om_3403":
        _validate_sd3403_output_layouts(backend_config)
    return CompiledManifest(artifacts=artifacts, execution=execution, backend_config=dict(backend_config))


def _shape_from_feature(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    elif isinstance(feature, list | tuple):
        shape = feature
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return []
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _feature_type(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("type", "")).upper()
    feature_type = getattr(feature, "type", None)
    if feature_type is None:
        return ""
    return str(getattr(feature_type, "name", feature_type)).upper()


def _action_shape_from_config(config: dict[str, Any]) -> list[int]:
    output_features = config.get("output_features") or {}
    if isinstance(output_features, dict) and "action" in output_features:
        shape = _shape_from_feature(output_features["action"])
        if shape:
            return shape
    action_dim = config.get("action_dim")
    if action_dim is not None:
        return [int(action_dim)]
    return [6]


def _chunk_size_from_config(config: dict[str, Any]) -> int:
    for key in ("chunk_size", "n_action_steps", "action_chunk_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    action_shape = _action_shape_from_config(config)
    if len(action_shape) >= 2:
        return int(action_shape[-2])
    return 1


def _action_dim_from_config(config: dict[str, Any]) -> int:
    action_shape = _action_shape_from_config(config)
    if action_shape:
        return int(action_shape[-1])
    return 6


def _image_hw_from_config(config: dict[str, Any]) -> tuple[int, int] | None:
    """Derive image (height, width) from config.json input_features.

    The authoritative image resolution lives in
    ``input_features.<image_key>.shape`` (e.g. ``[3, 480, 640]``), matching the
    ONNX/OM model input contract. Returns ``None`` when no image feature carries
    a usable ``[C, H, W]`` shape, so callers can fall back to an explicit
    backend_config override or trust the incoming tensor as-is.
    """
    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict):
        return None
    for key in _SD3403_IMAGE_FEATURE_KEYS:
        feature_shape = _shape_from_feature(input_features.get(key))
        if len(feature_shape) >= 3:
            return int(feature_shape[-2]), int(feature_shape[-1])
    return None


# Default graceful close timeout for the SD3403 worker subprocess (seconds).
# Mirrors ACTWrapper_3403.DEFAULT_GRACEFUL_CLOSE_TIMEOUT to avoid a cross-module
# import; both must stay in sync.
DEFAULT_GRACEFUL_CLOSE_TIMEOUT = 5.0


def _resolve_sd3403_image_hw(
    backend_config: dict[str, Any],
    legacy_config: dict[str, Any],
    dim_key: str,
    legacy_key: str,
) -> int | None:
    """Resolve one image dimension (height/width) for the SD3403 worker.

    Priority:
      1. config.json input_features.<image>.shape (the model's input contract).
      2. backend_config / legacy config explicit override (backward compatible).
      3. None -> worker trusts the incoming tensor size as-is.

    ``dim_key`` is ``"image_height"`` or ``"image_width"``; the matching
    input_features axis is selected from the resolved (h, w) pair.
    """
    hw = _image_hw_from_config(legacy_config)
    if hw is not None:
        return hw[0] if dim_key == "image_height" else hw[1]
    if dim_key in backend_config:
        return int(backend_config[dim_key])
    value = legacy_config.get(legacy_key)
    if value is not None:
        return int(value)
    return None


def _real_action_dim_from_config(config: dict[str, Any], fallback: int) -> int:
    output_features = config.get("output_features") or {}
    if isinstance(output_features, dict) and "action" in output_features:
        action_shape = _shape_from_feature(output_features["action"])
        if action_shape:
            return int(action_shape[-1])
    action_dim = config.get("action_dim")
    if action_dim is not None:
        return int(action_dim)
    return int(fallback)


def _input_order_from_config(config: dict[str, Any]) -> list[str]:
    for key in ("compiled_runtime_input_order", "runtime_input_order", "input_order"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]

    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError("compiled policy config must define non-empty input_features")

    input_keys = [key for key in input_features if key == "observation.state" or key.startswith("observation.images.")]
    if not input_keys:
        raise ValueError("compiled policy config does not expose supported state/image input features")
    return input_keys


def _to_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.astype(np.float32, copy=False))
    if isinstance(value, Tensor):
        if value.device.type == "cpu" and value.dtype == torch.float32 and value.is_contiguous():
            return value.detach().numpy()
        return np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _to_nhwc(value: np.ndarray) -> np.ndarray:
    """Convert a 4-D NCHW (1,C,H,W) array to NHWC (1,H,W,C).

    RKNN models embed an NHWC layout; RKNNLite expects 4-D image inputs already in
    NHWC. Non-image inputs (e.g. state vectors) are returned unchanged.
    """
    if isinstance(value, np.ndarray) and value.ndim == 4:
        return np.ascontiguousarray(np.transpose(value, (0, 2, 3, 1)))
    return value


def _to_numpy_int64(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.int64))


def _to_numpy_bool(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.bool_))


def _to_numpy_optional(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _as_action_tensor(output: Any, device: torch.device) -> Tensor:
    tensor = output if isinstance(output, Tensor) else torch.as_tensor(output, dtype=torch.float32)
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.to(device)


@dataclass
class VLARuntimeInputs:
    images: list[np.ndarray]
    tokens: np.ndarray
    masks: np.ndarray
    noise: np.ndarray | None = None


class _FeatureTypeView:
    def __init__(self, name: str):
        self.name = (name or "").upper()


class _FeatureSpecView:
    def __init__(self, feature_type: str, shape: list[int]):
        self.type = _FeatureTypeView(feature_type)
        self.shape = list(shape)


def _ordered_pi05_image_features(config: dict[str, Any]) -> dict[str, _FeatureSpecView]:
    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict):
        return {}
    result: dict[str, _FeatureSpecView] = {}
    for key, value in input_features.items():
        feature_type = _feature_type(value)
        if feature_type == "VISUAL" or key.startswith("observation.images."):
            result[key] = _FeatureSpecView(feature_type or "VISUAL", _shape_from_feature(value))
    return result


class _PI05ConfigView:
    def __init__(self, config: dict[str, Any]):
        self.chunk_size = _chunk_size_from_config(config)
        self.max_action_dim = int(config.get("max_action_dim", 32))
        self.num_inference_steps = int(config.get("num_inference_steps", 10))
        # SmolVLA's denoise loop uses ``num_steps`` (flow-matching steps).
        self.num_steps = int(config.get("num_steps", self.num_inference_steps))
        self.min_period = float(config.get("min_period", 0.004))
        self.max_period = float(config.get("max_period", 4.0))
        self.image_features = _ordered_pi05_image_features(config)
        # SmolVLA RKNN/HMM architecture params. Not present in the PI05 policy
        # config dict; SmolVLARKNNRuntimeSession.load merges them here from
        # the compiled manifest's backend_config so SmolVLARKNNModel reads the
        # real values via getattr instead of falling back to hardcoded
        # defaults (which silently corrupt KV-cache shapes for non-256M
        # SmolVLA variants). PI05 backends never read these attributes.
        self.num_layers = int(config.get("num_layers", 16))
        self.prefix_length = int(config.get("prefix_length", 177))
        self.prefix_hidden_size = int(config.get("prefix_hidden_size", 960))


class ACTCompiledAdapter:
    def __init__(self, config: dict[str, Any], backend: str):
        self._config = config
        self._backend = backend
        self._input_features = config.get("input_features") or {}
        self._input_keys = _input_order_from_config(config)
        self._chunk_size = _chunk_size_from_config(config)
        self._action_dim = _action_dim_from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> ACTCompiledAdapter:
        policy_type = str(config.get("type", "")).lower().strip()
        if not policy_type:
            raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
        if policy_type != "act":
            raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
        return cls(config, backend)

    @property
    def policy_type(self) -> str:
        return "act"

    @property
    def uses_action_chunking(self) -> bool:
        return True

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def prepare_inputs(self, batch: dict[str, Tensor]) -> list[np.ndarray]:
        inputs: list[np.ndarray] = []
        for key in self._input_keys:
            if key not in batch:
                raise KeyError(f"Missing compiled policy input tensor for {self._backend}: {key}")
            tensor = batch[key]
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)
            if key.startswith("observation.images."):
                tensor = self._prepare_image_tensor(key, tensor)
            elif tensor.ndim == 1:
                tensor = tensor.reshape(1, -1)
            inputs.append(_to_numpy_float32(tensor))
        return inputs

    def _prepare_image_tensor(self, key: str, tensor: Tensor) -> Tensor:
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.reshape(1, *tensor.shape)
        if tensor.ndim != 4:
            raise RuntimeError(f"{key} must be NCHW tensor, got shape={tuple(tensor.shape)}")

        target_hw = self._image_target_hw(key)
        if target_hw is not None and tuple(tensor.shape[-2:]) != target_hw:
            tensor = functional.interpolate(
                tensor,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return tensor.contiguous()

    def _image_target_hw(self, key: str) -> tuple[int, int] | None:
        feature_shape = _shape_from_feature(self._input_features.get(key))
        if len(feature_shape) >= 3:
            return int(feature_shape[-2]), int(feature_shape[-1])
        # No resolvable target dims: trust the incoming tensor size as-is.
        # Image resolution is the model's input contract (config.json
        # input_features.<key>.shape, baked into the OM via ATC --input_shape),
        # not a sidecar-only value, so we do not fall back to a hardcoded size.
        return None

    def decode_outputs(self, raw: list[np.ndarray], device: torch.device) -> Tensor:
        if not raw:
            raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
        if self._backend == "ascend_om_3403":
            return self._decode_sd3403_output(raw[0], device)
        if self._backend == "ascend_om":
            return self._decode_om_output(raw[0], device)
        return self._decode_first_action_output(raw[0], device)

    def _decode_om_output(self, output: Any, device: torch.device) -> Tensor:
        action = np.asarray(output, dtype=np.float32)
        if action.ndim == 1:
            expected_size = self._chunk_size * self._action_dim
            if action.size == expected_size:
                action = action.reshape(1, self._chunk_size, self._action_dim)
            elif action.size % self._action_dim == 0:
                action = action.reshape(1, -1, self._action_dim)
            else:
                raise RuntimeError(
                    f"unexpected ACT OM action tensor size={action.size}, "
                    f"not divisible by action_dim={self._action_dim}"
                )
        return _as_action_tensor(action, device)

    def _decode_sd3403_output(self, output: Any, device: torch.device) -> Tensor:
        action = decode_sd3403_action_array(output, self._action_dim)
        self._chunk_size = int(action.shape[-2])
        return _as_action_tensor(action, device)

    def _decode_first_action_output(self, output: Any, device: torch.device) -> Tensor:
        action = np.asarray(output, dtype=np.float32)
        if action.ndim == 1 and action.size % self._action_dim == 0:
            action = action.reshape(-1, self._action_dim)
        return _as_action_tensor(action, device)


class PI05CompiledAdapter:
    def __init__(self, config: dict[str, Any], backend: str):
        self._config = config
        self._backend = backend
        self._chunk_size = _chunk_size_from_config(config)
        self._max_action_dim = int(config.get("max_action_dim", 32))
        self._action_dim = _real_action_dim_from_config(config, self._max_action_dim)
        self._image_features = _ordered_pi05_image_features(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> PI05CompiledAdapter:
        policy_type = str(config.get("type", "")).lower().strip()
        if not policy_type:
            raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
        if policy_type != "pi05":
            raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
        normalized_backend = normalize_backend_name(backend)
        if normalized_backend not in ("ascend_om", "hmm"):
            raise ValueError(f"Compiled backend {backend} does not support PI05 policy")
        return cls(config, backend)

    @property
    def policy_type(self) -> str:
        return "pi05"

    @property
    def uses_action_chunking(self) -> bool:
        return True

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def prepare_inputs(self, batch: dict[str, Tensor]) -> VLARuntimeInputs:
        images: list[np.ndarray] = []
        for key in self._image_features:
            if key not in batch:
                raise KeyError(f"Missing PI05 image tensor for {self._backend}: {key}")
            images.append(_to_numpy_float32(batch[key]))
        if not images:
            raise ValueError("PI05 compiled policy config must define at least one VISUAL input feature")

        tokens = batch.get("observation.language.tokens", batch.get("lang_tokens"))
        masks = batch.get("observation.language.attention_mask", batch.get("lang_masks"))
        if tokens is None or masks is None:
            raise KeyError("Missing PI05 language tokens or attention masks")

        return VLARuntimeInputs(
            images=images,
            tokens=_to_numpy_int64(tokens),
            masks=_to_numpy_bool(masks),
            noise=_to_numpy_optional(batch.get("_noise")),
        )

    def decode_outputs(self, raw: Any, device: torch.device) -> Tensor:
        if isinstance(raw, list):
            if not raw:
                raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
            raw = raw[0]
        if raw is None:
            raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
        if getattr(raw, "shape", None) is not None and raw.shape[-1] > self._action_dim:
            raw = raw[..., : self._action_dim]
        action = _as_action_tensor(raw, device)
        if action.ndim >= 2:
            self._chunk_size = int(action.shape[-2])
        return action


class SmolVLACompiledAdapter:
    """Input/output adapter for SmolVLA on the Houmo HMM backend.

    SmolVLA shares PI05's observation contract (per-camera images + language
    tokens/masks + optional noise) and its compiled HMM model returns a single
    ``[chunk_size, action_dim]`` action tensor, so the plumbing mirrors
    :class:`PI05CompiledAdapter`.
    """

    def __init__(self, config: dict[str, Any], backend: str):
        self._config = config
        self._backend = backend
        self._chunk_size = _chunk_size_from_config(config)
        self._max_action_dim = int(config.get("max_action_dim", 32))
        self._action_dim = _real_action_dim_from_config(config, self._max_action_dim)
        self._image_features = _ordered_pi05_image_features(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> SmolVLACompiledAdapter:
        policy_type = str(config.get("type", "")).lower().strip()
        if not policy_type:
            raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
        if policy_type != "smolvla":
            raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
        if normalize_backend_name(backend) not in ("hmm", "rknn"):
            raise ValueError(f"Compiled backend {backend} does not support SmolVLA policy")
        return cls(config, backend)

    @property
    def policy_type(self) -> str:
        return "smolvla"

    @property
    def uses_action_chunking(self) -> bool:
        return True

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def prepare_inputs(self, batch: dict[str, Tensor]) -> VLARuntimeInputs:
        images: list[np.ndarray] = []
        for key in self._image_features:
            if key not in batch:
                raise KeyError(f"Missing SmolVLA image tensor for {self._backend}: {key}")
            images.append(_to_numpy_float32(batch[key]))
        if not images:
            raise ValueError("SmolVLA compiled policy config must define at least one VISUAL input feature")

        tokens = batch.get("observation.language.tokens", batch.get("lang_tokens"))
        masks = batch.get("observation.language.attention_mask", batch.get("lang_masks"))
        if tokens is None or masks is None:
            raise KeyError("Missing SmolVLA language tokens or attention masks")

        return VLARuntimeInputs(
            images=images,
            tokens=_to_numpy_int64(tokens),
            masks=_to_numpy_bool(masks),
            noise=_to_numpy_optional(batch.get("_noise")),
        )

    def decode_outputs(self, raw: Any, device: torch.device) -> Tensor:
        if isinstance(raw, list):
            if not raw:
                raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
            raw = raw[0]
        if raw is None:
            raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
        if getattr(raw, "shape", None) is not None and raw.shape[-1] > self._action_dim:
            raw = raw[..., : self._action_dim]
        action = _as_action_tensor(raw, device)
        if action.ndim >= 2:
            self._chunk_size = int(action.shape[-2])
        return action


ADAPTER_REGISTRY: dict[str, type[CompiledModelAdapter]] = {
    "act": ACTCompiledAdapter,
    "pi05": PI05CompiledAdapter,
    "smolvla": SmolVLACompiledAdapter,
}


def create_compiled_model_adapter(config: dict[str, Any], backend: str) -> CompiledModelAdapter:
    policy_type = str(config.get("type", "")).lower().strip()
    if not policy_type:
        raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
    adapter_cls = ADAPTER_REGISTRY.get(policy_type)
    if adapter_cls is None:
        raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
    return adapter_cls.from_config(config, backend)


def resolve_om_model_path(
    path: str,
    config: dict[str, Any] | None = None,
    manifest: CompiledManifest | None = None,
) -> Path:
    del config
    if manifest is None:
        manifest = load_compiled_manifest(path, "ascend_om")
    return manifest.require_artifact("policy", suffix=".om")


def resolve_pi05_om_paths(
    path: str,
    config: dict[str, Any] | None = None,
    manifest: CompiledManifest | None = None,
) -> tuple[Path, Path]:
    del config
    if manifest is None:
        manifest = load_compiled_manifest(path, "ascend_om", "pi05")
    manifest.require_execution(["vlm", "action_expert"])
    return (
        manifest.require_artifact("vlm", suffix=".om"),
        manifest.require_artifact("action_expert", suffix=".om"),
    )


def resolve_rknn_model_path(path: str) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_file() and raw_path.suffix == ".rknn":
        candidates.append(raw_path)
    if raw_path.is_dir():
        candidates.extend([raw_path / "model.rknn", raw_path / f"{raw_path.name}.rknn"])
        candidates.extend(sorted(raw_path.glob("*.rknn")))

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and candidate.suffix == ".rknn":
            return candidate.resolve()
    raise FileNotFoundError("RKNN model file not found under policy_path. Checked: " + ", ".join(checked))


class OMRuntimeSession:
    def __init__(self) -> None:
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om", str(config.get("type", "")).lower().strip())
        manifest.require_execution(["policy"])
        model_path = resolve_om_model_path(policy_path, config, manifest)
        from inference_service.core.ascend_om.OMmodel import OMmodel

        self._model = OMmodel(str(model_path))

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._model is None:
            raise RuntimeError("OMRuntimeSession is not loaded")
        return list(self._model.forward(inputs))

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None


class PI05OMRuntimeSession:
    def __init__(self) -> None:
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om", str(config.get("type", "")).lower().strip())
        vlm_path, action_expert_path = resolve_pi05_om_paths(policy_path, config, manifest)
        from inference_service.core.ascend_om.pi05.PI05OMModel import PI05OMModel

        self._model = PI05OMModel(str(vlm_path), str(action_expert_path), _PI05ConfigView(config))

    def execute(self, inputs: VLARuntimeInputs) -> Tensor:
        if self._model is None:
            raise RuntimeError("PI05OMRuntimeSession is not loaded")
        if not isinstance(inputs, VLARuntimeInputs):
            raise TypeError("PI05OMRuntimeSession expects VLARuntimeInputs")
        from inference_service.core.ascend_om.pi05.prefix_mask_utils import (
            build_prefix_att_2d_masks_4d_np,
        )

        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=len(inputs.images),
            lang_masks=inputs.masks,
            prefix_seq_len=self._model.prefix_seq_len,
        )
        return self._model.forward(
            inputs.images,
            inputs.tokens,
            inputs.masks,
            prefix_mask,
            noise=inputs.noise,
        )

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None


class SD3403RuntimeSession:
    def __init__(self) -> None:
        self._worker: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om_3403", str(config.get("type", "")).lower().strip())
        manifest.require_execution(["policy", "worker"])
        model_path = resolve_om_model_path(policy_path, config, manifest)
        worker_path = manifest.require_artifact("worker")
        if not os.access(worker_path, os.X_OK):
            raise FileNotFoundError(f"Compiled artifact 'worker' is not executable: {worker_path}")
        from inference_service.core.ascend_om.ACTWrapper_3403 import ACT3403Policy

        action_dim = _action_dim_from_config(config)
        backend_config = manifest.backend_config
        self._worker = ACT3403Policy(
            str(worker_path),
            str(model_path),
            action_dim=action_dim,
            action_output_index=_sd3403_backend_int(
                backend_config,
                config,
                key="index",
                legacy_key="sd3403_action_output_index",
                default=DEFAULT_SD3403_ACTION_OUTPUT_INDEX,
                section="action_output",
                flat_key="action_output_index",
                top_key="action_output_index",
            ),
            image_height=_resolve_sd3403_image_hw(backend_config, config, "image_height", "sd3403_image_height"),
            image_width=_resolve_sd3403_image_hw(backend_config, config, "image_width", "sd3403_image_width"),
            perf_enabled=_sd3403_backend_bool(
                backend_config,
                config,
                key="perf_enabled",
                legacy_key="sd3403_perf_enabled",
                default=DEFAULT_SD3403_PERF_ENABLED,
            ),
            perf_log_every=_sd3403_backend_int(
                backend_config,
                config,
                key="perf_log_every",
                legacy_key="sd3403_perf_log_every",
                default=DEFAULT_SD3403_PERF_LOG_EVERY,
            ),
            graceful_close_timeout=_sd3403_backend_float(
                backend_config,
                config,
                key="graceful_close_timeout",
                legacy_key="sd3403_graceful_close_timeout",
                default=DEFAULT_GRACEFUL_CLOSE_TIMEOUT,
            ),
            force_close=_sd3403_backend_bool(
                backend_config,
                config,
                key="force_close",
                legacy_key="sd3403_force_close",
                default=True,
            ),
        )

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._worker is None:
            raise RuntimeError("SD3403RuntimeSession is not loaded")
        execute_arrays = getattr(self._worker, "execute_arrays", None)
        if execute_arrays is None:
            raise RuntimeError("SD3403 worker does not expose execute_arrays")
        return [execute_arrays(inputs)]

    def release(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


class RKNNRuntimeSession:
    def __init__(self) -> None:
        self._rknn: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del config, device
        model_path = resolve_rknn_model_path(policy_path)
        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed with ret={ret}")
        ret = self._rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_ALL)
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with ret={ret}")

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._rknn is None:
            raise RuntimeError("RKNNRuntimeSession is not loaded")
        # RKNN models embed an NHWC layout, so RKNNLite expects 4-D image inputs
        # in NHWC (1,H,W,C). The adapter layer (ACTCompiledAdapter) produces a
        # backend-agnostic NCHW (1,C,H,W) layout shared with the Ascend OM
        # backends; the NHWC conversion is RKNN-specific, so it lives here in
        # the RKNN session rather than in the shared adapter. Without it,
        # RKNNLite silently rearranges the buffer the wrong way (treating
        # C/H/W as H/W/C), corrupting image channels and yielding garbage
        # actions while the simulator stays unaffected.
        rknn_inputs = [_to_nhwc(arr) for arr in inputs]
        outputs = self._rknn.inference(inputs=rknn_inputs)
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs")
        return list(outputs)

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


class SmolVLARKNNRuntimeSession:
    """Runtime session for the 3-module SmolVLA RKNN pipeline (vision/prefill/action).

    Loads ``config.rknn.json`` manifest, creates a ``SmolVLARKNNModel``
    orchestrator, and delegates ``execute`` to the model's ``forward`` method.
    The denoise loop runs on host CPU; each step calls the action NPU module.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        policy_type = str(config.get("type", "")).lower().strip()
        if policy_type != "smolvla":
            raise ValueError(f"SmolVLARKNNRuntimeSession does not support policy type {policy_type!r}")
        manifest = load_compiled_manifest(policy_path, "rknn", policy_type)
        manifest.require_execution(["vision", "prefill", "action"])
        vision_path = manifest.require_artifact("vision", suffix=".rknn")
        prefill_path = manifest.require_artifact("prefill", suffix=".rknn")
        action_path = manifest.require_artifact("action", suffix=".rknn")
        embedding_path = manifest.require_artifact("embedding")
        from inference_service.core.rknn.smolvla.SmolVLARKNNModel import SmolVLARKNNModel

        # Merge manifest backend_config (num_layers / prefix_length /
        # prefix_hidden_size / ...) over the policy config so _PI05ConfigView
        # exposes the real VLM architecture params to SmolVLARKNNModel instead
        # of the getattr fallback defaults (see _PI05ConfigView docstring).
        merged_config = {**config, **manifest.backend_config}
        self._model = SmolVLARKNNModel(
            vision_path=str(vision_path),
            prefill_path=str(prefill_path),
            action_path=str(action_path),
            embedding_path=str(embedding_path),
            config=_PI05ConfigView(merged_config),
        )

    def execute(self, inputs: Any) -> Tensor:
        if self._model is None:
            raise RuntimeError("SmolVLARKNNRuntimeSession is not loaded")
        if not isinstance(inputs, VLARuntimeInputs):
            raise TypeError("SmolVLARKNNRuntimeSession expects VLARuntimeInputs")
        return self._model.forward(
            inputs.images,
            torch.as_tensor(inputs.tokens),
            torch.as_tensor(inputs.masks),
            noise=torch.as_tensor(inputs.noise) if inputs.noise is not None else None,
        )

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None


def resolve_hmm_model_path(path: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve the compiled Houmo ``.hmm`` artifact for the single-module ACT HMM backend.

    Priority: ``config.hmm.json`` manifest ``policy`` role > env override >
    directory conventions (``model.hmm``, ``<dir>.hmm``, any ``*.hmm``).
    """
    del config
    env_path = os.environ.get("HMM_MODEL_PATH", "").strip()
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    if env_path:
        candidates.append(Path(env_path).expanduser())

    if raw_path.is_file() and raw_path.suffix == ".hmm":
        candidates.append(raw_path)
    if raw_path.is_dir():
        candidates.extend([raw_path / "model.hmm", raw_path / f"{raw_path.name}.hmm"])
        candidates.extend(sorted(raw_path.glob("*.hmm")))

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and candidate.suffix == ".hmm":
            return candidate.resolve()
    raise FileNotFoundError("HMM model file not found under policy_path. Checked: " + ", ".join(checked))


class HMMRuntimeSession:
    """Runtime session for the Houmo HMM (LQ50 / M50 xh2) backend.

    Dispatches by ``policy_type``:

    - ``act``: single compiled ``.hmm`` module via ``tcim_lite.runtime`` with the
      same ``list[np.ndarray]`` I/O contract as ``RKNNRuntimeSession`` (shares
      ``ACTCompiledAdapter``).
    - ``pi05`` / ``smolvla``: multi-module orchestrator (``PI05HMMModel`` /
      ``SmolVLAHMMModel``) that drives ``tcim_lite.runtime`` and runs the denoise
      loop on the host, with KV-cache handoff by device-pointer sharing.
    """

    def __init__(self) -> None:
        self._mode: str | None = None
        # ACT single-module path
        self._module: Any = None
        self._input_names: list[str] = []
        # pi05 / smolvla multi-module path
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        policy_type = str(config.get("type", "")).lower().strip()

        if policy_type == "act":
            model_path = resolve_hmm_model_path(policy_path)
            import tcim_lite as tcim  # type: ignore[import-not-found]

            self._module = tcim.runtime.load(str(model_path))
            self._input_names = [self._module.get_input_name(i) for i in range(self._module.get_num_inputs())]
            self._mode = "act"
            return

        manifest = load_compiled_manifest(policy_path, "hmm", policy_type)
        if manifest is None:
            raise FileNotFoundError(f"HMM backend requires config.hmm.json under policy_path {policy_path}")

        if policy_type == "pi05":
            manifest.require_execution(["vision", "prefill", "decode", "time_mlp", "action_in_proj", "action_out_proj"])
            from inference_service.core.hmm.pi05.PI05HMMModel import PI05HMMModel

            self._model = PI05HMMModel(
                vision_path=str(manifest.require_artifact("vision", suffix=".hmm")),
                prefill_path=str(manifest.require_artifact("prefill", suffix=".hmm")),
                decode_path=str(manifest.require_artifact("decode", suffix=".hmm")),
                time_mlp_path=str(manifest.require_artifact("time_mlp", suffix=".hmm")),
                action_in_proj_path=str(manifest.require_artifact("action_in_proj", suffix=".hmm")),
                action_out_proj_path=str(manifest.require_artifact("action_out_proj", suffix=".hmm")),
                embedding_path=str(manifest.require_artifact("embedding")),
                config=_PI05ConfigView(config),
            )
            self._mode = "pi05"
        elif policy_type == "smolvla":
            manifest.require_execution(["vision", "prefill", "action"])
            from inference_service.core.hmm.smolvla.SmolVLAHMMModel import SmolVLAHMMModel

            self._model = SmolVLAHMMModel(
                vision_path=str(manifest.require_artifact("vision", suffix=".hmm")),
                prefill_path=str(manifest.require_artifact("prefill", suffix=".hmm")),
                action_path=str(manifest.require_artifact("action", suffix=".hmm")),
                embedding_path=str(manifest.require_artifact("embedding")),
                config=_PI05ConfigView(config),
            )
            self._mode = "smolvla"
        else:
            raise ValueError(f"HMM backend does not support policy type {policy_type!r}")

    def execute(self, inputs: list[np.ndarray] | VLARuntimeInputs) -> list[np.ndarray] | Tensor:
        if self._mode is None:
            raise RuntimeError("HMMRuntimeSession is not loaded")

        if self._mode == "act":
            return self._execute_act(inputs)  # type: ignore[arg-type]
        return self._execute_multi(inputs)  # type: ignore[arg-type]

    def _execute_act(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._module is None:
            raise RuntimeError("HMMRuntimeSession is not loaded")
        if len(inputs) != len(self._input_names):
            raise RuntimeError(f"HMMRuntimeSession expected {len(self._input_names)} inputs, got {len(inputs)}")
        for name, data in zip(self._input_names, inputs, strict=False):
            self._module.set_input(name, data)
        self._module.run()
        self._module.sync()

        outputs: list[np.ndarray] = []
        for i in range(self._module.get_num_outputs()):
            name = self._module.get_output_name(i)
            output = self._module.get_output(name)
            cast = getattr(output, "astype", None)
            if callable(cast):
                output = cast(np.float32)
            to_numpy = getattr(output, "numpy", None)
            if callable(to_numpy):
                output = to_numpy()
            outputs.append(np.ascontiguousarray(np.asarray(output, dtype=np.float32)))
        if not outputs:
            raise RuntimeError("HMM inference returned no outputs")
        return outputs

    def _execute_multi(self, inputs: VLARuntimeInputs) -> Tensor:
        if self._model is None:
            raise RuntimeError("HMMRuntimeSession is not loaded")
        if not isinstance(inputs, VLARuntimeInputs):
            raise TypeError("HMMRuntimeSession (multi-module) expects VLARuntimeInputs")
        from inference_service.core.hmm.pi05.PI05HMMModel import (
            build_prefix_att_2d_masks_4d_np,
        )

        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=len(inputs.images),
            lang_masks=inputs.masks,
            prefix_seq_len=self._model.prefix_seq_len,
        )
        return self._model.forward(
            inputs.images,
            torch.as_tensor(inputs.tokens),
            torch.as_tensor(inputs.masks),
            prefix_mask,
            noise=inputs.noise,
        )

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None
        self._module = None
        self._input_names = []
        self._mode = None


def create_runtime_session(backend: str, config: dict[str, Any] | None = None) -> RuntimeSession:
    normalized = normalize_backend_name(backend)
    if normalized == "ascend_om":
        policy_type = str((config or {}).get("type", "")).lower().strip()
        if policy_type == "pi05":
            return PI05OMRuntimeSession()
        return OMRuntimeSession()
    if normalized == "ascend_om_3403":
        return SD3403RuntimeSession()
    if normalized == "rknn":
        policy_type = str((config or {}).get("type", "")).lower().strip()
        if policy_type == "smolvla":
            return SmolVLARKNNRuntimeSession()
        return RKNNRuntimeSession()
    if normalized == "hmm":
        return HMMRuntimeSession()
    raise ValueError(f"Unsupported compiled inference backend: {backend}")


class CompiledPolicyWrapper(PolicyWrapper):
    def __init__(self, backend: str, runtime_session: RuntimeSession | None = None) -> None:
        self._backend = normalize_backend_name(backend)
        self._runtime_session = runtime_session
        self._adapter: CompiledModelAdapter | None = None
        self._device = torch.device("cpu")

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = load_compiled_policy_config(path, self._backend, runtime_device=device)
        config = _attach_compiled_backend_config(path, self._backend, config)
        self._adapter = create_compiled_model_adapter(config, self._backend)
        if self._runtime_session is None:
            self._runtime_session = create_runtime_session(self._backend, config)
        self._runtime_session.load(path, config, device)

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._adapter is None or self._runtime_session is None:
            raise RuntimeError(f"CompiledPolicyWrapper for {self._backend} is not loaded")
        inputs = self._adapter.prepare_inputs(batch)
        outputs = self._runtime_session.execute(inputs)
        return self._adapter.decode_outputs(outputs, self._device)

    def get_chunk_size(self) -> int:
        if self._adapter is None:
            return 1
        return self._adapter.get_chunk_size()

    @property
    def policy_type(self) -> str:
        if self._adapter is None:
            return ""
        return self._adapter.policy_type

    @property
    def backend_type(self) -> str:
        return self._backend

    @property
    def uses_action_chunking(self) -> bool:
        return bool(self._adapter is not None and self._adapter.uses_action_chunking)

    def close(self) -> None:
        if self._runtime_session is not None:
            self._runtime_session.release()
