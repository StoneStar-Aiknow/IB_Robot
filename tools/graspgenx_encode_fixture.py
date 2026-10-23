#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Freeze the upstream PTV3 serialization reference for an off-wheel board.

`verify_graspgenx_ascend_ops_on_device.py` compares the GridEncode operator
against `graspgenx.models.ptv3.ptv3_vanilla.encode`. That reference is plain
PyTorch and does not have to run where the operator runs, so a board with no
graspgenx wheel reads what this wrote instead. The inputs are fixed by the
same seed on both sides, so only the expected codes travel.

Run it anywhere the wheel is installed:

    source .shrc_local
    python3 tools/graspgenx_encode_fixture.py \\
        src/torch_models/test/graspgenx_encode_reference.npz
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Importing graspgenx runs a hook that clones gripper descriptions and
# checkpoints from Hugging Face unless both roots exist; this needs neither.
_ASSETS = tempfile.mkdtemp(prefix="graspgenx-fixture-")
for _variable in ("GRASPGENX_GRIPPER_CFG_DIR", "GRASPGENX_CHECKPOINT_DIR"):
    os.environ.setdefault(_variable, _ASSETS)

import torch  # noqa: E402
from graspgenx.models.ptv3.ptv3_vanilla import encode  # noqa: E402

# Must match the verification script exactly, or the board rejects the file.
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
SEED = 0
DEPTH = 6
POINTS = 257


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <output.npz>")
    destination = Path(sys.argv[1]).expanduser()

    coordinates = np.random.default_rng(SEED).integers(0, 2**DEPTH, size=(POINTS, 3), dtype=np.int32)
    expected = np.stack(
        [
            encode(
                torch.from_numpy(coordinates).long(),
                torch.zeros(coordinates.shape[0], dtype=torch.long),
                DEPTH,
                order=order,
            ).numpy()
            for order in ORDERS
        ],
        axis=1,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        coordinates=coordinates,
        depth=DEPTH,
        expected=expected,
        orders=np.asarray(ORDERS),
    )
    print(f"wrote {destination}: {expected.shape} {expected.dtype} for orders {', '.join(ORDERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
