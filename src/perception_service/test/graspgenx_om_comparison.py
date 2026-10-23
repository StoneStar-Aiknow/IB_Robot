#!/usr/bin/env python3
"""Compare two compiled diffusion loops output by output, on one device.

The graph shipped in the bundle was compiled by CANN 8.3.RC1's ATC. Rebuilding
it from the export toolchain on a host whose pinned toolkit is 8.1 produces a
different file, and "different file" says nothing about whether it computes the
same thing. This feeds both the same inputs and compares all four outputs.

Both models are driven the way the deployment drives them -- through
DiffusionLoopOM, on Torch's own stream -- so the comparison includes the
interop, not only the graph.

Usage:
    graspgenx_om_comparison.py <reference.om> <candidate.om>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401 -- registers the NPU device

# <workspace>/src/<package>/test/<this file>, so the workspace is three
# parents up. A hardcoded default would only ever be right on one host.
WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for package in sorted(p.name for p in (WORKSPACE / "src").iterdir() if p.is_dir()):
    sys.path.insert(0, str(WORKSPACE / "src" / package))

from torch_models.graspgenx_ascend_310p import DiffusionLoopOM  # noqa: E402

OUTPUTS = ("grasps", "likelihood", "noise_predictions", "latents")


def run(model_path: Path, inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Execute one offline model and bring its outputs back to the host."""
    loop = DiffusionLoopOM(str(model_path))
    try:
        produced = loop.run(
            inputs["observation"],
            inputs["initial_noise"],
            inputs["position_noise"],
            inputs["rotation_noise"],
        )
        # The runner's buffers are reused by the next call, so clone now.
        # strict: this script's claim is that it compares all four outputs, so
        # a graph returning a different number must fail, not be truncated.
        return {
            name: tensor.detach().to(torch.float32).cpu().numpy().copy()
            for name, tensor in zip(OUTPUTS, produced, strict=True)
        }
    finally:
        loop.close()


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    reference_path, candidate_path = (Path(argument) for argument in sys.argv[1:])

    torch.npu.set_device(0)
    # The operators the graph embeds are compiled in, but the compile mode still
    # decides whether anything else is rebuilt at run time.
    torch.npu.set_compile_mode(jit_compile=False)
    print(f"device     {torch.npu.get_device_name(0)}")
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        print(f"{label:10} {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    reference_loop = DiffusionLoopOM(str(reference_path))
    metadata = dict(reference_loop.metadata)
    reference_loop.close()
    grasps, steps = int(metadata["grasps"]), int(metadata["steps"])
    observation, sample, half = (int(metadata["observation"]), int(metadata["sample"]), int(metadata["half"]))
    print(f"shape      {grasps} grasps, {steps} steps, observation {observation}")

    # Fixed inputs: the comparison is between two compilations of one graph, so
    # the inputs only have to be the same for both and in a plausible range.
    rng = np.random.default_rng(0)
    inputs = {
        "observation": rng.standard_normal((grasps, observation), dtype=np.float32),
        "initial_noise": rng.standard_normal((grasps, sample), dtype=np.float32),
        "position_noise": rng.standard_normal((steps - 1, grasps, half), dtype=np.float32),
        "rotation_noise": rng.standard_normal((steps - 1, grasps, half), dtype=np.float32),
    }
    device_inputs = {name: torch.from_numpy(value).npu() for name, value in inputs.items()}

    reference = run(reference_path, device_inputs)
    candidate = run(candidate_path, device_inputs)

    print(f"\n  {'output':20} {'cosine':>12} {'max_abs':>12} {'rel_l2':>12}")
    failures = []
    for name in OUTPUTS:
        left = reference[name].reshape(-1).astype(np.float64)
        right = candidate[name].reshape(-1).astype(np.float64)
        norms = np.linalg.norm(left) * np.linalg.norm(right)
        cosine = float(left @ right / norms) if norms else 1.0
        max_abs = float(np.abs(left - right).max())
        scale = np.linalg.norm(left)
        relative = float(np.linalg.norm(left - right) / scale) if scale else 0.0
        print(f"  {name:20} {cosine:12.8f} {max_abs:12.3e} {relative:12.3e}")
        # The two compilations differ only in which ATC built them, so anything
        # beyond float32 rounding means they are not the same computation.
        if cosine < 0.999999 or relative > 1e-5:
            failures.append(name)

    # What production actually consumes is the ranking, so check it directly.
    print()
    left = reference["likelihood"].reshape(-1)
    right = candidate["likelihood"].reshape(-1)
    order_left, order_right = np.argsort(-left), np.argsort(-right)
    for keep in (10, 50, 200, grasps):
        shared = len(set(order_left[:keep].tolist()) & set(order_right[:keep].tolist()))
        print(f"  likelihood top-{keep:<4} overlap {shared}/{keep}")

    if failures:
        raise SystemExit(f"FAILED: {', '.join(failures)} differ beyond rounding")
    print("\nthe two compilations compute the same loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
