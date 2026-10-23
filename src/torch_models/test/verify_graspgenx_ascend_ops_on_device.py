#!/usr/bin/env python3
"""Check the operators IB_Robot built itself, through the module that loads them.

This runs against the operator cache and the `torch_models`
wrappers only: nothing here reads the GraspGenX-ascend checkout, so a pass
means the workspace can produce and use the operators on its own.

`GridEncode` is checked against the reference encoding it has to match column
for column, and `SubmConv3d` against a dense gather-and-accumulate over the
same neighbour map, so each is compared with an independent computation rather
than with a recorded output.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# <workspace>/src/torch_models/test/<this file>, so the workspace is three
# parents up. A hardcoded default would only ever be right on one host.
WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(WORKSPACE / "src/torch_models"))

# Importing graspgenx runs a hook that clones gripper descriptions and
# checkpoints from Hugging Face unless both roots exist. This test only needs
# the package's reference encoding, so they are pointed at empty directories;
# without this the import blocks for minutes on a host with no such network.
_ASSET_ROOT = tempfile.mkdtemp(prefix="graspgenx-ops-check-")
for _variable in ("GRASPGENX_GRIPPER_CFG_DIR", "GRASPGENX_CHECKPOINT_DIR"):
    os.environ.setdefault(_variable, _ASSET_ROOT)

# This exercises operators that only exist on an Ascend device, built by
# scripts/setup/install_graspgenx_ascend_ops.sh. Collection must not fail on a
# host without them, so the import guard comes before anything else.
try:
    import torch  # noqa: E402
    import torch_npu  # noqa: E402, F401 -- registers the NPU device
except ImportError as exc:  # pragma: no cover - the common case off-device
    import pytest

    pytest.skip(f"Ascend Torch runtime unavailable: {exc}", allow_module_level=True)

from torch_models.graspgenx_ascend_310p.custom_ops import operator_root  # noqa: E402
from torch_models.graspgenx_ascend_310p.custom_ops.grid_encode import (  # noqa: E402
    grid_encode,
)
from torch_models.graspgenx_ascend_310p.custom_ops.submconv3d import (  # noqa: E402
    build_subm_map,
    subm_conv3d,
)

print(f"operator root: {operator_root()}")
torch.npu.set_device(0)
# Without this the runtime recompiles the operator from its dynamic
# implementation instead of using the binary kernels the package ships, and
# that recompile needs an Ascend C toolchain it cannot find
# ("inner_interface/inner_kernel_operator_intf.h file not found"). The encoder
# sets the same mode; a test that calls the operators directly has to as well.
torch.npu.set_compile_mode(jit_compile=False)
print(f"device:        {torch.npu.get_device_name(0)}")
print(f"jit_compile:   {not torch.npu.is_jit_compile_false()}")

failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


# --- GridEncode against the reference serialization -------------------------
print("\nGridEncode")
rng = np.random.default_rng(0)
depth = 6
coordinates = rng.integers(0, 2**depth, size=(257, 3), dtype=np.int32)
produced = grid_encode(torch.from_numpy(coordinates).npu().contiguous(), depth).cpu().numpy()

# The reference encoding is plain PyTorch, so it does not have to be computed
# here. A board's job is to execute the operators; making it carry the upstream
# wheel for one function means installing a training-time dependency tree on
# it, and pip resolving that tree once moved Torch off the version torch_npu's
# ABI is built against. So: use the wheel when it is already installed,
# otherwise read a fixture a host wrote with tools/graspgenx_encode_fixture.py.
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
FIXTURE = Path(os.environ.get("GRASPGENX_ENCODE_FIXTURE", Path(__file__).with_name("graspgenx_encode_reference.npz")))
try:
    from graspgenx.models.ptv3.ptv3_vanilla import encode as reference_encode  # noqa: E402

    expected = np.stack(
        [
            reference_encode(
                torch.from_numpy(coordinates).long(),
                torch.zeros(coordinates.shape[0], dtype=torch.long),
                depth,
                order=order,
            ).numpy()
            for order in ORDERS
        ],
        axis=1,
    )
    reference_source = "the installed graspgenx wheel"
except ModuleNotFoundError:
    if not FIXTURE.is_file():
        raise SystemExit(
            f"no graspgenx wheel here and no reference fixture at {FIXTURE}. Write one on a host "
            f"that has the wheel:\n    python3 tools/graspgenx_encode_fixture.py {FIXTURE}"
        ) from None
    stored = np.load(FIXTURE)
    if int(stored["depth"]) != depth or not np.array_equal(stored["coordinates"], coordinates):
        raise SystemExit(f"{FIXTURE} was written for different inputs; regenerate it") from None
    expected = stored["expected"]
    reference_source = f"fixture {FIXTURE.name}"

print(f"reference:     {reference_source}")
check("shape", produced.shape == expected.shape, f"{produced.shape} vs {expected.shape}")
check(
    "values match the reference encoding",
    bool((produced == expected).all()),
    f"{int((produced != expected).sum())} differing entries",
)

# --- SubmConv3d against a dense gather-and-accumulate ------------------------
print("\nSubmConv3d")
rows, channels_in, channels_out, kernel = 96, 32, 48, 3
volume = kernel**3
coordinates = rng.integers(0, 8, size=(rows, 3), dtype=np.int32)
coordinates = np.unique(coordinates, axis=0).astype(np.int32)
rows = coordinates.shape[0]
batch = np.zeros((rows, 1), dtype=np.int32)
indices = torch.from_numpy(np.concatenate([batch, coordinates], axis=1)).npu().contiguous()

neighbors = build_subm_map(indices, kernel)
check(
    "neighbour width is the padded kernel volume",
    neighbors.shape == (rows, (volume + 7) // 8 * 8),
    f"{tuple(neighbors.shape)} for volume {volume}",
)

features = torch.from_numpy(rng.standard_normal((rows, channels_in)).astype(np.float16)).npu()
weight = torch.from_numpy(rng.standard_normal((volume, channels_in, channels_out)).astype(np.float16)).npu()
# The wrapper takes the coordinates and will build the map itself; passing the
# one already built exercises the reuse path the encoder relies on. (The raw
# op underneath is the one that takes a map, which is what PTV3 calls.)
with torch.no_grad():
    produced = subm_conv3d(features, indices, weight, neighbor_map=neighbors).to(torch.float32).cpu().numpy()

# The same sum, written independently: for each output row, add the
# contribution of every present neighbour through that offset's weight.
neighbour_map = neighbors.cpu().numpy()[:, :volume]
features_host = features.to(torch.float32).cpu().numpy()
weight_host = weight.to(torch.float32).cpu().numpy()
expected = np.zeros((rows, channels_out), dtype=np.float32)
for row in range(rows):
    for offset in range(volume):
        source = neighbour_map[row, offset]
        if source >= 0:
            expected[row] += features_host[source] @ weight_host[offset]

scale = max(float(np.abs(expected).max()), 1e-6)
error = float(np.abs(produced - expected).max()) / scale
check("matches a dense gather-and-accumulate", error < 5e-2, f"relative max error {error:.2e}")

print()
if failures:
    raise SystemExit(f"FAILED: {', '.join(failures)}")
print("the operators IB_Robot built are loadable and correct")
