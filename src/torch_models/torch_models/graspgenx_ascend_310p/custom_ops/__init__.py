"""Locate the custom Ascend C operators GraspGenX's PTV3 encoder calls.

Two operators carry the encoder's irregular work on a 310P: `graspgenx_grid`
holds the integer space-filling-curve encoding and `graspgenx_subm` the
submanifold convolution the CPE uses. Each ships as a CANN vendor package plus
a Torch bridge, both built at setup time rather than downloaded, because the
bridge links the installed Torch's ABI and the vendor package is compiled for a
CANN release and an `ascend310p` target.

Where setup put them is therefore not knowable from this file, so it is read
from the environment with a documented default, in the same way
`pi05_ascend_310p` locates its TorchAir cache. The layout is::

    <root>/graspgenx_grid/torch_bridge.so
    <root>/graspgenx_subm/torch_bridge.so
    <root>/vendors/<vendor_name>/...

Note that the vendor packages themselves are found by CANN through
`ASCEND_CUSTOM_OPP_PATH`, which has to be exported before the process starts;
nothing here can set it after the fact.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["operator_root", "load_bridge"]

ENVIRONMENT_VARIABLE = "IBROBOT_GRASPGENX_ASCEND_OPS"
# The operators are built per host, for one CANN release and the installed
# Torch's ABI, so the products are a cache rather than part of the tree: the
# source under kernels/ is versioned, what comes out of it is not. This is the
# only definition of that location; a launch that has to carry
# ASCEND_CUSTOM_OPP_PATH forwards the variable above instead of recomputing it.
_CACHE_DEFAULT = "~/.cache/ibrobot/graspgenx_ascend_ops"


def operator_root() -> Path:
    """The directory setup built the operators into."""
    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    return Path(_CACHE_DEFAULT).expanduser()


def load_bridge(vendor: str, operator: str) -> Path:
    """Load one vendor's Torch bridge, or say how to build it.

    Loading the same library twice is harmless -- Torch keeps a registry keyed
    by path -- so each wrapper may call this at import without coordinating.
    """
    import torch

    library = operator_root() / vendor / "torch_bridge.so"
    if not library.is_file():
        raise FileNotFoundError(
            f"The {operator} Torch bridge is missing: {library}. Build the Ascend "
            f"operators with scripts/setup/install_graspgenx_ascend_ops.sh, or point "
            f"{ENVIRONMENT_VARIABLE} at the directory holding them."
        )
    torch.ops.load_library(str(library))
    return library
