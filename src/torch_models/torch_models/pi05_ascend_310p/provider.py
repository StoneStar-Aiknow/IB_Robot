"""Runtime provider for PI0.5 native Torch inference on Ascend310P."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from packaging.version import Version

from torch_models.pi05_ascend_310p.inference_config import PI05AscendInferenceConfig
from torch_models.pi05_ascend_310p.input_optimization import configure_input_processors, resize_host_images
from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import (
    PI05Ascend310PPolicy,
    configure_pi05_ascend_310p_config,
)
from torch_models.policy_provider import PolicyProvider

PI05_ASCEND_310P_LOAD_OPTIONS = {"skip_weight_init": True}


def resolve_pi05_load_options(*, config, extra_parameters):
    if not extra_parameters:
        return {}
    inference_config = PI05AscendInferenceConfig.parse(extra_parameters)
    # The session owns this fresh config object. Pass the same immutable
    # settings to construction and processors instead of re-reading env vars.
    config.ascend_inference_config = inference_config
    config.ascend_explicit_deployment = True
    return {"inference_config": inference_config}


def validate_pi05_ascend_310p(*, config, bundle_root: Path, tokenizer_path: str | None, device_name: str) -> None:
    """Validate the platform-specific contract before allocating model weights."""

    if "Ascend310P" not in device_name:
        raise ValueError(f"PI05 Ascend310P provider requires Ascend310P, got {device_name!r}")
    if tokenizer_path is None:
        raise ValueError("PI05 Ascend310P provider requires a bundled tokenizer")
    if not (bundle_root / "model.safetensors").is_file():
        raise ValueError("PI05 Ascend310P provider requires bundled model.safetensors")
    try:
        transformers_version = Version(package_version("transformers"))
    except (PackageNotFoundError, ValueError) as exc:
        raise ValueError(f"unable to identify Transformers: {exc}") from exc
    if transformers_version.base_version != "5.3.0":
        raise ValueError(f"PI05 Ascend310P provider requires Transformers 5.3.0, got {transformers_version}")
    if getattr(config, "ascend_cpu_image_resize", False):
        from lerobot.configs import FeatureType, NormalizationMode

        if config.normalization_mapping.get(FeatureType.VISUAL) != NormalizationMode.IDENTITY:
            raise ValueError("PI05 CPU image resize requires IDENTITY visual normalization")


def prepare_pi05_ascend_310p(*, policy, deployment_fingerprint, torch_module, torch_npu_module, device_name):
    return policy.prepare_for_inference(
        deployment_fingerprint=deployment_fingerprint,
        torch_module=torch_module,
        torch_npu_module=torch_npu_module,
        device_name=device_name,
    )


def _execution_metadata(policy) -> dict[str, object]:
    records = policy.model.get_action_fused_stage_timing_records()
    metadata = {"pi05_stage_timing": records[-1]} if records else {}
    inference_config = getattr(policy, "inference_config", None)
    if inference_config is not None:
        metadata["pi05_inference_config"] = inference_config.to_dict()
    return metadata


def create_provider() -> PolicyProvider:
    return PolicyProvider(
        policy_class=PI05Ascend310PPolicy,
        configure_config=configure_pi05_ascend_310p_config,
        validate=validate_pi05_ascend_310p,
        prepare=prepare_pi05_ascend_310p,
        load_options=PI05_ASCEND_310P_LOAD_OPTIONS,
        execution_metadata=_execution_metadata,
        preprocess_inputs=resize_host_images,
        configure_processors=configure_input_processors,
        resolve_load_options=resolve_pi05_load_options,
    )


__all__ = [
    "PI05_ASCEND_310P_LOAD_OPTIONS",
    "PI05Ascend310PPolicy",
    "configure_pi05_ascend_310p_config",
    "create_provider",
    "prepare_pi05_ascend_310p",
    "validate_pi05_ascend_310p",
]
