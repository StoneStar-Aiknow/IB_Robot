"""Native Torch provider for PI0.5 inference on Ascend 910B-family NPUs.

The provider deliberately owns the 910B runtime contract.  The target
LeRobot v0.6 configuration parser predates the NPU graph and quantization
fields used by this implementation, so ``load_pi05_ascend_910b_config``
filters those fields before parsing and attaches the validated runtime values
afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import Version

from torch_models.pi05_ascend_910b.modeling_pi05_ascend_910b import PI05Ascend910BPolicy
from torch_models.pi05_ascend_910b.quantization import QuantizationConfig
from torch_models.policy_provider import PolicyProvider

TRANSFORMERS_MIN = Version("5.4.0")
TRANSFORMERS_MAX = Version("5.6.0")
TORCH_BASE_VERSION = "2.10.0"
TORCH_NPU_BASE_VERSION = "2.10.0"
ASCEND_910B_DEVICE_MARKERS = ("Ascend910B", "Ascend910_93")
PI05_ASCEND_910B_LOAD_OPTIONS: dict[str, object] = {}

_CONFIG_DEFAULTS = {
    "compile_inference_graph": True,
    "compile_inference_backend": "torchair",
    "compile_inference_fullgraph": True,
    "compile_inference_dynamic": False,
    "compile_frozen_parameter": True,
    "compile_tiling_schedule_optimize": True,
    "shared_prefix_fias": True,
    "adarms_bias_fusion": True,
    "adarms_bias_fusion_stage": "all",
    "denoise_solver": "euler",
}


def _config_payload(bundle_root: Path) -> dict[str, object]:
    config_path = bundle_root / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read PI05 Ascend910B config.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("PI05 Ascend910B config.json must contain an object")
    if payload.get("type") not in {None, "pi05"}:
        raise ValueError(f"PI05 Ascend910B provider requires type='pi05', got {payload.get('type')!r}")
    return payload


def _is_ascend_910b_device(device_name: str) -> bool:
    return any(marker in device_name for marker in ASCEND_910B_DEVICE_MARKERS)


def _attach_910b_fields(config, payload: Mapping[str, object], *, model_dtype: str) -> object:
    if model_dtype not in {"native", "bf16"}:
        raise ValueError(
            f"PI05 Ascend910B optimized inference requires model_dtype='native' or 'bf16', got {model_dtype!r}"
        )
    config.dtype = "bfloat16"

    # The checkpoint's generic LeRobot flag wraps the whole policy with the
    # default torch.compile backend during construction.  On Ascend that can
    # select Inductor before this provider installs its dedicated TorchAir
    # prefix/denoise graphs.  The 910B provider owns graph compilation, so the
    # generic training-time wrapper must stay disabled.
    config.compile_model = False

    for name, default in _CONFIG_DEFAULTS.items():
        setattr(config, name, payload.get(name, default))

    quantization = payload.get("quantization", getattr(config, "quantization", None))
    if quantization is None:
        config.quantization = None
    elif isinstance(quantization, QuantizationConfig):
        # The unified session loads the provider-owned config first and then
        # invokes configure_config as a separate lifecycle hook.  Keep that
        # second pass idempotent instead of rejecting the already validated
        # local configuration object.
        config.quantization = quantization
    elif isinstance(quantization, Mapping):
        try:
            config.quantization = QuantizationConfig(**dict(quantization))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid PI05 Ascend910B quantization config: {exc}") from exc
    else:
        raise ValueError("PI05 Ascend910B quantization must be an object or null")
    return config


def load_pi05_ascend_910b_config(
    bundle_root: str | Path | None = None,
    *,
    bundle_path: str | Path | None = None,
    config_type=None,
    model_dtype: str = "bf16",
    **_unused,
):
    """Load a PI0.5 config while dropping fields unknown to target LeRobot v0.6.

    The temporary sanitized config is parsed by the target ``PI05Config`` so
    nested feature/RTC values retain LeRobot's normal decoding semantics.
    910B-only graph and quantization fields are then attached explicitly.
    """

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if bundle_root is None:
        bundle_root = bundle_path
    if bundle_root is None:
        raise ValueError("PI05 Ascend910B config requires bundle_root")
    bundle_path = Path(bundle_root)
    payload = _config_payload(bundle_path)
    # The session supplies the generic PreTrainedConfig symbol.  This provider
    # owns the concrete PI05 schema and must select it before parsing.
    parser_class = config_type or PreTrainedConfig
    target_fields = {field.name for field in fields(PI05Config)}
    sanitized = {name: value for name, value in payload.items() if name in target_fields}
    sanitized["type"] = "pi05"
    # The shared LeRobot config validator does not know the Torch-NPU device
    # type. Parse with a neutral CPU device; LeRobotTorchModelSession replaces
    # it with the manifest-owned ``npu`` device before model construction.
    sanitized["device"] = "cpu"
    with TemporaryDirectory(prefix="ibrobot-pi05-910b-config-") as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "config.json").write_text(json.dumps(sanitized), encoding="utf-8")
        try:
            config = parser_class.from_pretrained(temporary_path, local_files_only=True)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"unable to parse sanitized PI05 Ascend910B config: {exc}") from exc
    return _attach_910b_fields(config, payload, model_dtype=model_dtype)


def configure_pi05_ascend_910b_config(config, *, model_dtype: str = "bf16"):
    """Apply 910B defaults to a target ``PI05Config`` instance."""

    payload = {name: getattr(config, name) for name in _CONFIG_DEFAULTS if hasattr(config, name)}
    return _attach_910b_fields(config, payload, model_dtype=model_dtype)


def validate_pi05_ascend_910b(*, config, bundle_root: Path, tokenizer_path: str | None, device_name: str) -> None:
    """Validate all fail-closed platform and bundle prerequisites."""

    if not _is_ascend_910b_device(device_name):
        raise ValueError(f"PI05 Ascend910B provider requires an Ascend910B/Ascend910_93 device, got {device_name!r}")
    if tokenizer_path is None:
        raise ValueError("PI05 Ascend910B provider requires a bundled tokenizer")
    if not (bundle_root / "model.safetensors").is_file():
        raise ValueError("PI05 Ascend910B provider requires bundled model.safetensors")
    try:
        transformers_version = Version(package_version("transformers"))
        torch_version = Version(package_version("torch"))
        torch_npu_version = Version(package_version("torch-npu"))
    except (PackageNotFoundError, ValueError) as exc:
        raise ValueError(f"unable to identify PI05 Ascend910B runtime packages: {exc}") from exc
    if not TRANSFORMERS_MIN <= transformers_version < TRANSFORMERS_MAX:
        raise ValueError(f"PI05 Ascend910B provider requires Transformers >=5.4,<5.6, got {transformers_version}")
    if torch_version.base_version != TORCH_BASE_VERSION:
        raise ValueError(f"PI05 Ascend910B provider requires PyTorch {TORCH_BASE_VERSION}, got {torch_version}")
    if torch_npu_version.base_version != TORCH_NPU_BASE_VERSION:
        raise ValueError(
            f"PI05 Ascend910B provider requires Torch-NPU {TORCH_NPU_BASE_VERSION}, got {torch_npu_version}"
        )
    if getattr(config, "quantization", None) is not None and not isinstance(config.quantization, QuantizationConfig):
        raise ValueError("PI05 Ascend910B quantization must be a QuantizationConfig")


def prepare_pi05_ascend_910b(*, policy, deployment_fingerprint, torch_module, torch_npu_module, device_name):
    """Enable the complete validated 910B eager/graph optimization path."""

    del deployment_fingerprint, torch_module
    if not _is_ascend_910b_device(device_name):
        raise ValueError(f"PI05 Ascend910B preparation requires an Ascend910B/Ascend910_93 device, got {device_name!r}")
    if torch_npu_module is None:
        raise RuntimeError("PI05 Ascend910B preparation requires torch_npu")
    prepare = getattr(policy, "prepare_inference_optimizations", None)
    if not callable(prepare):
        raise RuntimeError("PI05 Ascend910B policy does not expose inference optimizations")
    optimization_info = prepare(
        enable_npu_fused_ops=True,
        enable_graph_compile=True,
        enable_qkv_fusion=True,
        enable_mlp_fusion=getattr(policy.config, "quantization", None) is not None,
        mlp_fusion_scope="prefix",
        enable_shared_prefix_fias=True,
        enable_adarms_bias_fusion=True,
        adarms_bias_fusion_stage="all",
    )
    if not isinstance(optimization_info, Mapping):
        raise RuntimeError("PI05 Ascend910B optimization setup returned invalid metadata")
    policy._pi05_ascend_910b_optimization_info = dict(optimization_info)
    return dict(optimization_info)


def _execution_metadata(policy) -> dict[str, object]:
    info = getattr(policy, "_pi05_ascend_910b_optimization_info", None)
    return {"pi05_ascend_910b_optimizations": dict(info)} if isinstance(info, Mapping) else {}


def create_provider() -> PolicyProvider:
    return PolicyProvider(
        policy_class=PI05Ascend910BPolicy,
        configure_config=configure_pi05_ascend_910b_config,
        validate=validate_pi05_ascend_910b,
        prepare=prepare_pi05_ascend_910b,
        load_options=PI05_ASCEND_910B_LOAD_OPTIONS,
        load_config=load_pi05_ascend_910b_config,
        execution_metadata=_execution_metadata,
    )


__all__ = [
    "ASCEND_910B_DEVICE_MARKERS",
    "PI05_ASCEND_910B_LOAD_OPTIONS",
    "configure_pi05_ascend_910b_config",
    "create_provider",
    "load_pi05_ascend_910b_config",
    "prepare_pi05_ascend_910b",
    "validate_pi05_ascend_910b",
]
