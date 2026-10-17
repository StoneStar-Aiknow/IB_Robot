"""GraspGenX served on an Ascend 310P through native Torch.

The encoders are this package's PTV3, the diffusion loop runs as an offline
model built by `model_utils.graspgenx_export`, and the custom Ascend C
operators the encoder calls are built at setup time -- see `custom_ops` for how
they are located.

Importing this package pulls in torch_npu, so it is imported by the loader for
an `npu` deployment, never from `torch_models/__init__.py`.

The exports resolve lazily. `GraspGenGeneratorAscend` and the assembly helpers
reach the upstream graspgenx wheel at import time, while `DiffusionLoopOM`
drives a compiled graph through ACL and needs none of it: binding every name
eagerly made a board that only has to execute a graph -- or to check the
operators against a reference computed elsewhere -- install the whole wheel.
"""

from typing import TYPE_CHECKING

_EXPORTS = {
    "DiffusionLoopOM": "diffusion_loop_om",
    "GraspGenGeneratorAscend": "generator_ascend",
    "assemble": "model_assembly",
    "verify_device": "model_assembly",
    "PointTransformerV3Ascend": "ptv3_ascend",
    "shared_geometry": "ptv3_ascend",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover - these bind at runtime through __getattr__
    from torch_models.graspgenx_ascend_310p.diffusion_loop_om import DiffusionLoopOM  # noqa: F401
    from torch_models.graspgenx_ascend_310p.generator_ascend import GraspGenGeneratorAscend  # noqa: F401
    from torch_models.graspgenx_ascend_310p.model_assembly import assemble, verify_device  # noqa: F401
    from torch_models.graspgenx_ascend_310p.ptv3_ascend import (  # noqa: F401
        PointTransformerV3Ascend,
        shared_geometry,
    )


def __getattr__(name: str):
    """Import the module that defines ``name`` on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value  # resolve once; later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
