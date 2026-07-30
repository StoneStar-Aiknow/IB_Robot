"""Readiness contracts for perception Ascend OM artifacts.

The registry is intentionally fail-closed: an artifact being present on disk is
not sufficient to declare an adapter ready when its runtime ABI is incomplete.
"""

from dataclasses import dataclass
from pathlib import Path

from .model_utils import WORKSPACE_ROOT


@dataclass(frozen=True)
class OmAdapterReadiness:
    model: str
    ready: bool
    artifacts: tuple[Path, ...]
    reason: str = ""


def _existing(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if path.is_file())


def inspect_om_adapter(model: str) -> OmAdapterReadiness:
    """Return local OM readiness without importing ACL or executing a model."""
    if model == "ram_plus":
        artifacts = _existing(
            (
                WORKSPACE_ROOT
                / "models/ram_plus_swin_large_14m/model_utils_work/candidates/ram_plus_swin_large_14m_fp16.om",
            )
        )
        if artifacts:
            return OmAdapterReadiness(model, True, artifacts)
        return OmAdapterReadiness(model, False, artifacts, "RAM++ OM artifact is missing")

    if model == "sam2":
        artifacts = _existing(
            (
                WORKSPACE_ROOT / "sam2" / "models" / "om" / "sam2_encoder.om",
                WORKSPACE_ROOT / "sam2" / "models" / "om" / "sam2_decoder.om",
            )
        )
        return OmAdapterReadiness(
            model,
            False,
            artifacts,
            "SAM2 encoder/decoder tensor ABI and preprocessing contract are not finalized",
        )

    if model == "siglip2":
        return OmAdapterReadiness(
            model,
            False,
            (),
            "SigLIP2 production OM artifact and tensor ABI are not available",
        )

    if model == "grounding_dino":
        artifacts = _existing(
            (
                WORKSPACE_ROOT / "mmdetection" / "grounding_dino_backbone_aarch64_linux_aarch64.om",
                WORKSPACE_ROOT / "mmdetection" / "grounding_dino_neck.om",
                WORKSPACE_ROOT / "mmdetection" / "grounding_dino_transformer_head.om",
                WORKSPACE_ROOT / "mmdetection" / "grounding_dino_bert.om",
            )
        )
        return OmAdapterReadiness(
            model,
            False,
            artifacts,
            "Grounding DINO multi-artifact ABI and tokenizer contract are not finalized",
        )

    raise ValueError("model must be 'ram_plus', 'sam2', 'siglip2', or 'grounding_dino'")


def require_om_adapter_ready(model: str) -> OmAdapterReadiness:
    """Raise a diagnostic error unless the local adapter contract is ready."""
    readiness = inspect_om_adapter(model)
    if not readiness.ready:
        raise RuntimeError(f"{model} Ascend OM adapter is not ready: {readiness.reason}")
    return readiness
