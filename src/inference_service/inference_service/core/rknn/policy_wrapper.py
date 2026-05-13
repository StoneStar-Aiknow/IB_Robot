from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from inference_service.core.pure_inference_engine import PolicyWrapper


def _normalize_device_name(device: str) -> str:
    return str(device).lower().strip().replace("-", "_")


def _load_policy_config(path: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        config_path = candidate
    elif candidate.is_dir():
        config_path = candidate / "config.json"
    else:
        raise FileNotFoundError(f"RKNN policy config.json not found under policy_path {candidate}")
    if not config_path.is_file():
        raise FileNotFoundError(f"RKNN policy config.json not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"RKNN policy config must be a JSON object: {config_path}")
    return data


def _chunk_size_from_config(config: dict[str, Any]) -> int:
    for key in ("chunk_size", "n_action_steps", "action_chunk_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return 1


def _input_keys_from_config(config: dict[str, Any]) -> list[str]:
    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError("RKNN policy config must define non-empty input_features")

    input_keys: list[str] = []
    for k, v in input_features.items():
        if not isinstance(v, dict):
            continue
        if k == "observation.state" or k.startswith("observation.images."):
            input_keys.append(k)
    if not input_keys:
        raise ValueError("RKNN policy config does not expose supported state/image input features")
    return input_keys


def _resolve_rknn_model_path(path: str) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_file() and raw_path.suffix == ".rknn":
        candidates.append(raw_path)

    if raw_path.is_dir():
        candidates.extend(
            [
                raw_path / "model.rknn",
                raw_path / f"{raw_path.name}.rknn",
            ]
        )
        candidates.extend(sorted(raw_path.glob("*.rknn")))

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and candidate.suffix == ".rknn":
            return candidate.resolve()

    raise FileNotFoundError(
        "RKNN model file not found under policy_path. Checked: " + ", ".join(checked)
    )


class RKNNPolicyWrapper(PolicyWrapper):
    def __init__(self) -> None:
        self._rknn: Any = None
        self._device = torch.device("cpu")
        self._chunk_size: int = 1
        self._policy_type: str = "rknn"
        self._input_keys: list[str] = []

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = _load_policy_config(path)
        self._chunk_size = _chunk_size_from_config(config)
        self._input_keys = _input_keys_from_config(config)
        model_path = _resolve_rknn_model_path(path)

        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed with ret={ret}")
        ret = self._rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with ret={ret}")

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._rknn is None:
            raise RuntimeError("RKNNPolicyWrapper is not loaded")

        inputs: list[np.ndarray] = []

        for key in self._input_keys:
            tensor = batch.get(key)
            if tensor is None:
                raise KeyError(f"Missing RKNN input tensor: {key}")

            input_np = self._to_numpy(tensor)
            if input_np.ndim == 1:
                input_np = input_np.reshape(1, -1)
            elif key.startswith("observation.images.") and input_np.ndim == 3:
                input_np = input_np.reshape(1, *input_np.shape)
            inputs.append(input_np.astype(np.float32))

        outputs = self._rknn.inference(inputs=inputs)
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs")

        action = outputs[0]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if action.ndim >= 3 and action.shape[0] == 1:
            action = action.squeeze(0)
        return action.to(self._device)

    @staticmethod
    def _to_numpy(t: Any) -> np.ndarray:
        if isinstance(t, np.ndarray):
            return t
        if isinstance(t, Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def create_rknn_policy_wrapper(device: str) -> PolicyWrapper:
    normalized = _normalize_device_name(device)
    if normalized == "rknn":
        return RKNNPolicyWrapper()
    raise ValueError(f"Unsupported RKNN inference device: {device}")
