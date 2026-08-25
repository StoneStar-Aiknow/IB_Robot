"""Model path, backend, and runtime identity helpers for perception services."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from .model_contracts import ModelManifest, sha256_file


def find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
        if (parent / "models").is_dir():
            return parent
    return Path(os.environ.get("IB_ROBOT_WORKSPACE", ".")).resolve()


WORKSPACE_ROOT = find_workspace_root()
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "models"


def resolve_model_path(path_str: str, model_dir: str | Path | None = None) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    base = Path(model_dir).expanduser() if model_dir else DEFAULT_MODEL_DIR
    candidate = base / path
    if candidate.exists():
        return candidate
    workspace_candidate = WORKSPACE_ROOT / "models" / path
    return workspace_candidate if workspace_candidate.exists() else candidate


def resolve_text_encoder(text_encoder: str, model_dir: str | Path | None = None) -> str:
    path = Path(text_encoder).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = resolve_model_path(text_encoder, model_dir)
    return str(candidate) if candidate.exists() else text_encoder


@dataclass(frozen=True)
class BackendStatus:
    backend: str
    ready: bool
    runtime_version: str
    message: str = ""


def inspect_backend(backend: str) -> BackendStatus:
    if backend == "cuda":
        try:
            import torch
        except ModuleNotFoundError:
            return BackendStatus(backend, False, "", "torch is not installed")
        if not torch.cuda.is_available():
            return BackendStatus(backend, False, str(torch.__version__), "CUDA is not available")
        return BackendStatus(backend, True, str(torch.__version__))
    if backend == "cpu":
        try:
            import torch
        except ModuleNotFoundError:
            return BackendStatus(backend, False, "", "torch is not installed")
        return BackendStatus(backend, True, str(torch.__version__))
    if backend == "ascend":
        if importlib.util.find_spec("acl") is None:
            return BackendStatus(backend, False, "", "Ascend ACL Python runtime is not installed")
        try:
            import acl
        except (ImportError, OSError) as exc:
            return BackendStatus(backend, False, "", f"Ascend ACL import failed: {exc}")
        return BackendStatus(backend, True, str(getattr(acl, "__version__", "unknown")))
    raise ValueError("backend must be 'cpu', 'cuda', or 'ascend'")


def build_model_manifest(
    *,
    model_name: str,
    model_version: str,
    weights_path: str | Path,
    config_path: str | Path | None,
    backend: str,
    preprocessing_hash: str,
    embedding_dim: int = 0,
    normalization: str = "",
) -> ModelManifest:
    status = inspect_backend(backend)
    weights = Path(weights_path)
    config = None if config_path is None else Path(config_path)
    return ModelManifest(
        model_name=model_name,
        model_version=model_version,
        weights_hash=sha256_file(weights),
        config_hash="" if config is None else sha256_file(config),
        backend=backend,
        runtime_version=status.runtime_version,
        preprocessing_hash=preprocessing_hash,
        embedding_dim=embedding_dim,
        normalization=normalization,
    )
