"""Thin PolicyWrapper adapters for the migrated Ascend OM ACT wrappers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from inference_service.core.ascend_om.ACTWrapper import ACTWrapper
from inference_service.core.ascend_om.ACTWrapper_3403 import (
    ACT3403Policy,
    DEFAULT_OM_BASENAME,
    _guess_worker_path_from_model,
    _is_om_path,
)
from inference_service.core.pure_inference_engine import PolicyWrapper


class _FeatureConfig:
    def __init__(self, shape: Any):
        self.shape = list(shape) if isinstance(shape, (list, tuple)) else [int(shape)]


class _ACTConfigView:
    """Config view with the fields consumed by the migrated ACTWrapper."""

    def __init__(self, raw_config: Dict[str, Any]):
        self.chunk_size = _chunk_size_from_config(raw_config)
        self.input_features = raw_config.get("input_features") or {}
        self.output_features = {
            "action": _FeatureConfig(_action_shape_from_config(raw_config)),
        }


def _normalize_device_name(device: str) -> str:
    return str(device).lower().strip().replace("-", "_")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _policy_config_path(path: str) -> Optional[Path]:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        return candidate
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if config_path.is_file():
            return config_path
    return None


def _load_policy_config(path: str) -> Dict[str, Any]:
    config_path = _policy_config_path(path)
    return _read_json(config_path) if config_path is not None else {}


def _chunk_size_from_config(config: Dict[str, Any]) -> int:
    for key in ("chunk_size", "n_action_steps", "action_chunk_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return 1


def _shape_from_feature(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return []
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _action_shape_from_config(config: Dict[str, Any]) -> list[int]:
    output_features = config.get("output_features") or {}
    if isinstance(output_features, dict) and "action" in output_features:
        shape = _shape_from_feature(output_features["action"])
        if shape:
            return shape
    action_dim = config.get("action_dim")
    if action_dim is not None:
        return [int(action_dim)]
    return [6]


def _candidate_om_paths(path: str, config: Dict[str, Any]) -> list[Path]:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    for env_name in ("ASCEND_OM_MODEL_PATH", "OM_MODEL_PATH", "SVP_MODEL_PATH"):
        env_path = os.environ.get(env_name, "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())

    config_om = str(config.get("om_model_path") or "").strip()
    if config_om:
        config_path = Path(config_om).expanduser()
        candidates.append(config_path)
        if not config_path.is_absolute() and raw_path.is_dir():
            candidates.append(raw_path / config_path)

    if raw_path.is_file() and _is_om_path(str(raw_path)):
        candidates.append(raw_path)

    if raw_path.is_dir():
        candidates.extend(
            [
                raw_path / "model.om",
                raw_path / "model" / "model.om",
                raw_path / "model" / DEFAULT_OM_BASENAME,
            ]
        )
        candidates.extend(sorted(raw_path.glob("*.om")))
        model_dir = raw_path / "model"
        if model_dir.is_dir():
            candidates.extend(sorted(model_dir.glob("*.om")))

    return candidates


def resolve_om_model_path(path: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve an OM model from policy config, environment, or policy layout."""
    config = config or _load_policy_config(path)
    checked: list[str] = []
    for candidate in _candidate_om_paths(path, config):
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and _is_om_path(str(candidate)):
            return candidate.resolve()

    raise FileNotFoundError("Ascend OM model file not found. Checked: " + ", ".join(checked))


def resolve_3403_worker_path(path: str, model_path: Path, config: Dict[str, Any]) -> Path:
    """Resolve the SD3403 worker executable from config, env, or common layout."""
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    for env_name in ("SVP_WORKER_EXECUTABLE", "SVP_CPP_EXECUTABLE"):
        env_path = os.environ.get(env_name, "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())

    config_worker = str(config.get("cpp_executable") or "").strip()
    if config_worker:
        worker_path = Path(config_worker).expanduser()
        candidates.append(worker_path)
        if not worker_path.is_absolute() and raw_path.is_dir():
            candidates.append(raw_path / worker_path)

    if raw_path.is_file() and not _is_om_path(str(raw_path)):
        candidates.append(raw_path)

    if raw_path.is_dir():
        candidates.extend([raw_path / "out" / "main", raw_path / "main", raw_path / "build" / "main"])

    candidates.append(Path(_guess_worker_path_from_model(str(model_path))).expanduser())

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise FileNotFoundError(
        "Ascend 3403 worker executable not found or not executable. Checked: "
        + ", ".join(checked)
    )


class AscendOMPolicyWrapper(PolicyWrapper):
    """PolicyWrapper adapter for the migrated generic ACTWrapper."""

    def __init__(self) -> None:
        self._impl: Optional[ACTWrapper] = None
        self._device = torch.device("cpu")
        self._chunk_size = 1
        self._policy_type = "ascend_om"

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = _load_policy_config(path)
        config_view = _ACTConfigView(config)
        self._chunk_size = config_view.chunk_size
        model_path = resolve_om_model_path(path, config)
        self._impl = ACTWrapper(str(model_path), config_view)

    def infer(self, batch: Dict[str, Tensor]) -> Tensor:
        if self._impl is None:
            raise RuntimeError("AscendOMPolicyWrapper is not loaded")
        output = self._impl.predict(batch)
        if not output:
            raise RuntimeError("Ascend OM model returned no outputs")
        return _as_action_tensor(output[0], self._device)

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        self._impl = None


class AscendOM3403PolicyWrapper(PolicyWrapper):
    """PolicyWrapper adapter for the migrated SD3403 ACT wrapper."""

    def __init__(self) -> None:
        self._impl: Optional[ACT3403Policy] = None
        self._device = torch.device("cpu")
        self._chunk_size = int(os.environ.get("SVP_ACTION_CHUNK_SIZE", "1"))
        self._policy_type = "ascend_om_3403"

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = _load_policy_config(path)
        self._chunk_size = _chunk_size_from_config(config)
        model_path = resolve_om_model_path(path, config)
        worker_path = resolve_3403_worker_path(path, model_path, config)
        self._impl = ACT3403Policy(str(worker_path), str(model_path))

    def infer(self, batch: Dict[str, Tensor]) -> Tensor:
        if self._impl is None:
            raise RuntimeError("AscendOM3403PolicyWrapper is not loaded")
        output = self._impl.predict(batch)
        if not output:
            raise RuntimeError("Ascend 3403 OM model returned no outputs")
        action = _as_action_tensor(output[0], self._device)
        if action.ndim >= 2:
            self._chunk_size = int(action.shape[0])
        return action

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None


def _as_action_tensor(output: Any, device: torch.device) -> Tensor:
    tensor = output if isinstance(output, Tensor) else torch.as_tensor(output)
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.to(device)


def create_ascend_om_policy_wrapper(device: str) -> PolicyWrapper:
    normalized = _normalize_device_name(device)
    if normalized == "ascend_om_3403":
        return AscendOM3403PolicyWrapper()
    if normalized == "ascend_om":
        return AscendOMPolicyWrapper()
    raise ValueError(f"Unsupported Ascend OM inference device: {device}")
