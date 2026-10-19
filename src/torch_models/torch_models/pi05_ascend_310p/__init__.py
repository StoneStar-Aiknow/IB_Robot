"""PI0.5 Ascend310P configuration is importable without loading Torch/Transformers."""

from importlib import import_module

from .inference_config import DenoisingConfig, PI05AscendInferenceConfig, deployment_parameters


def __getattr__(name):
    if name in {"PI05Ascend310PPolicy", "configure_pi05_ascend_310p_config"}:
        module = import_module(".modeling_pi05_ascend_310p", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DenoisingConfig",
    "PI05AscendInferenceConfig",
    "PI05Ascend310PPolicy",
    "configure_pi05_ascend_310p_config",
    "deployment_parameters",
]
