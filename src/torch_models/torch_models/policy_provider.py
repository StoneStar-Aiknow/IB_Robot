"""Lazy registry for repository-owned Torch policy implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PolicyProvider:
    """Model-owned hooks consumed by the native Torch session."""

    policy_class: type
    configure_config: Callable[..., object]
    validate: Callable[..., None]
    prepare: Callable[..., object]
    load_options: Mapping[str, object]
    load_config: Callable[..., object] | None = None
    execution_metadata: Callable[[object], Mapping[str, object]] | None = None


_NPU_PROVIDERS = {
    "Ascend310P": "torch_models.pi05_ascend_310p.provider",
    "Ascend910B": "torch_models.pi05_ascend_910b.provider",
    # CANN/Torch-NPU exposes 910B 93-series products with names such as
    # Ascend910_9362 instead of the marketing-family string Ascend910B3.
    "Ascend910_93": "torch_models.pi05_ascend_910b.provider",
}


def resolve_policy_provider(
    model_type: str,
    backend: str,
    device: str,
    *,
    device_name: str | None = None,
) -> PolicyProvider | None:
    """Resolve a repository-owned policy by runtime identity and physical NPU SKU."""

    if (model_type, backend, device) != ("pi05", "torch", "npu"):
        return None
    # Keep the original three-argument resolver usable by callers that predate
    # hardware-aware dispatch. The inference service always supplies device_name.
    if device_name is None:
        device_name = "Ascend310P"
    module_name = next((module for sku, module in _NPU_PROVIDERS.items() if sku in device_name), None)
    if module_name is None:
        raise ValueError(f"unsupported PI0.5 native Torch NPU device {device_name!r}")
    return import_module(module_name).create_provider()


__all__ = ["PolicyProvider", "resolve_policy_provider"]
