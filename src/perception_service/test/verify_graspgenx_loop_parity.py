#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Compare the compiled diffusion loop against the eager one, output by output.

`verify_graspgenx_pose_parity.py` serves a whole request, which needs the
upstream wheel and its import chain on both sides. This takes the narrower
route the user asked for: the loop's inputs are four plain tensors, so fixing
them makes the two implementations directly comparable without a bundle, a
service or a wheel on the device.

    # anywhere with the bundle and the wheel -- writes inputs and the reference
    verify_graspgenx_loop_parity.py --reference loop_cuda.npz
    # on the board, needs only DiffusionLoopOM and the .om
    verify_graspgenx_loop_parity.py --run-om loop_cuda.npz loop_npu.npz
    # anywhere
    verify_graspgenx_loop_parity.py --compare loop_cuda.npz loop_npu.npz

`grasps` is the r3_so3 parameterisation, before division by kappa and restoring
the cloud centre. The reported translation converts to metres using kappa.

What this covers: the denoising head's weights, both compositional DDPM
schedulers, the likelihood accumulation and the pose decode. What it does not:
the PTV3 encoder, whose output is `observation` here -- an input to both sides.
That half needs the wheel on the device and is measured separately.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for _package in ("model_utils", "torch_models"):
    sys.path.insert(0, str(WORKSPACE / "src" / _package))

_ASSETS = tempfile.mkdtemp(prefix="graspgenx-loop-parity-")
for _variable in ("GRASPGENX_GRIPPER_CFG_DIR", "GRASPGENX_CHECKPOINT_DIR"):
    os.environ.setdefault(_variable, _ASSETS)

OUTPUTS = ("grasps", "likelihood", "noise_predictions", "latents")


def loop_inputs(constants: Path, seed: int) -> dict[str, np.ndarray]:
    """The four tensors the graph takes, drawn from one seed on both sides."""
    stored = np.load(constants)
    steps = int(stored["timesteps"].shape[0])
    grasps, observation_dim, half = int(stored["grasps"]), 1024, 3
    rng = np.random.default_rng(seed)
    return {
        "observation": rng.standard_normal((grasps, observation_dim), dtype=np.float32),
        "initial_noise": rng.standard_normal((grasps, 2 * half), dtype=np.float32),
        "position_noise": rng.standard_normal((steps - 1, grasps, half), dtype=np.float32),
        "rotation_noise": rng.standard_normal((steps - 1, grasps, half), dtype=np.float32),
    }


def eager_reference(bundle: Path, constants: Path, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the loop eagerly with the upstream head and the dumped constants.

    Use upstream's scheduler.step with explicit replayed variance noise. Do not
    reuse exported coefficients: an exporter bug must fail this comparison.
    """
    import torch

    stored = np.load(constants)
    assets = json.loads((bundle / "assets" / "adapter.json").read_text(encoding="utf-8"))
    checkpoints = bundle / assets["checkpoints_dir"]
    # Point the upstream import hook at the bundle before importing graspgenx,
    # exactly as torch_model_loaders.load_graspgenx does, or it clones from
    # Hugging Face into site-packages.
    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(bundle / "assets" / "x_grippers")
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(checkpoints)

    from graspgenx.grasp_server import load_grasp_gen_model
    from graspgenx.utils.checkpoint_io import load_model_cfg

    # The same wiring the production loader uses, so the head here is the head
    # the deployment serves with.
    config = load_model_cfg(
        str(checkpoints / "gen"),
        str(checkpoints / "dis"),
        gen_pth=str(bundle / assets["generator_checkpoint"]),
        dis_pth=str(bundle / assets["discriminator_checkpoint"]),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # patch 0004 adds the device argument. An environment installed before that
    # patch reports the same local version, so dispatch on the signature rather
    # than on the version, and say which build is actually loaded.
    if "device" in inspect.signature(load_grasp_gen_model).parameters:
        model = load_grasp_gen_model(config, device=device)
    else:
        print(f"note: the installed graspgenx predates patch 0004; loading on {device} via .to()")
        model = load_grasp_gen_model(config).to(device)
    generator = model.grasp_generator if hasattr(model, "grasp_generator") else model
    timesteps = [int(t) for t in stored["timesteps"]]
    return eager_loop(generator, inputs, timesteps)


def eager_loop(generator, inputs, timesteps):
    """Independent scheduler reference, also runnable on a CPU test model."""
    import torch

    device = next(generator.diffusion_head.parameters()).device
    half = inputs["position_noise"].shape[-1]

    sample = torch.from_numpy(inputs["initial_noise"]).to(device)
    observation = torch.from_numpy(inputs["observation"]).to(device)
    noise = {
        "pos": torch.from_numpy(inputs["position_noise"]).to(device),
        "rot": torch.from_numpy(inputs["rotation_noise"]).to(device),
    }
    predictions, latents = [], []
    likelihood = torch.zeros((len(sample), 1), device=device)
    schedulers = {"pos": generator.noise_scheduler_pos, "rot": generator.noise_scheduler_rot}
    for scheduler in schedulers.values():
        scheduler.set_timesteps(len(timesteps))
        if [int(t) for t in scheduler.timesteps] != timesteps:
            raise ValueError("reference scheduler does not match the exported timestep schedule")

    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            predicted = generator.diffusion_head(observation, torch.tensor(timestep, device=device), sample)
            predictions.append(predicted.detach().clone())
            halves = []
            for offset, prefix in ((0, "pos"), (half, "rot")):
                current = sample[:, offset : offset + half]
                model_output = predicted[:, offset : offset + half]
                scheduler = schedulers[prefix]
                replay = noise[prefix][index] if timestep > 0 else torch.zeros_like(current)
                with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor", return_value=replay):
                    result = scheduler.step(model_output, timestep, current)
                if timestep > 0:
                    distribution = torch.distributions.Normal(
                        result.pred_original_sample, scheduler.betas[timestep].sqrt().to(device)
                    )
                    likelihood += distribution.log_prob(current).sum(-1, keepdim=True)
                halves.append(result.prev_sample)
            sample = torch.cat(halves, dim=1)
            # The graph records each updated sample, not the input to that step.
            latents.append(sample.detach().clone())

    return {
        "grasps": sample.float().cpu().numpy(),
        "likelihood": likelihood.float().cpu().numpy(),
        "noise_predictions": torch.stack(predictions).float().cpu().numpy(),
        "latents": torch.stack(latents).float().cpu().numpy(),
    }


def run_om(reference_path: Path, output: Path) -> int:
    """Drive the compiled graph with the inputs the reference run recorded."""
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU device

    from torch_models.graspgenx_ascend_310p import DiffusionLoopOM

    stored = np.load(reference_path)
    graph = Path(os.environ.get("GRASPGENX_DIFFUSION_LOOP", stored["graph"].item()))
    if not graph.is_file():
        raise SystemExit(f"no compiled loop at {graph}; set GRASPGENX_DIFFUSION_LOOP")

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    names = ("observation", "initial_noise", "position_noise", "rotation_noise")
    tensors = {name: torch.from_numpy(stored[name]).npu() for name in names}

    loop = DiffusionLoopOM(str(graph))
    try:
        produced = loop.run(*(tensors[name] for name in names))
    finally:
        loop.close()

    values = produced if isinstance(produced, dict) else dict(zip(OUTPUTS, produced, strict=True))
    np.savez(output, **{name: np.asarray(values[name].detach().cpu().clone()) for name in OUTPUTS})
    print(f"wrote {output}")
    return 0


def compare(left_path: Path, right_path: Path) -> int:
    left, right = np.load(left_path), np.load(right_path)
    print(f"{left_path.name}  vs  {right_path.name}\n")
    shared = [name for name in OUTPUTS if name in left.files and name in right.files]

    print(f"  {'output':20} {'cosine':>12} {'max_abs':>12} {'rel_l2':>12}")
    for name in shared:
        a = left[name].reshape(-1).astype(np.float64)
        b = right[name].reshape(-1).astype(np.float64)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(a @ b / norms) if norms else 1.0
        relative = float(np.linalg.norm(a - b) / np.linalg.norm(a)) if np.linalg.norm(a) else 0.0
        print(f"  {name:20} {cosine:12.8f} {np.abs(a - b).max():12.3e} {relative:12.3e}")

    # The point of the exercise: grasps[:, :3] is the pose translation in metres.
    if "kappa" not in left:
        raise ValueError("reference lacks kappa; regenerate it before reporting physical translation")
    translation = np.linalg.norm(left["grasps"][:, :3] - right["grasps"][:, :3], axis=1) / float(left["kappa"])
    rotation = np.linalg.norm(left["grasps"][:, 3:] - right["grasps"][:, 3:], axis=1)
    print(f"\n  translation  max {translation.max() * 1000:9.4f} mm   mean {translation.mean() * 1000:9.4f} mm")
    print(f"  rotation     max {rotation.max():9.6f}      mean {rotation.mean():9.6f}   (r3_so3 units)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, help="Write inputs plus the eager outputs here.")
    parser.add_argument("--run-om", type=Path, nargs=2, metavar=("REFERENCE", "OUTPUT"))
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--bundle", type=Path, default=WORKSPACE / "models/graspgenx_omnihand_left_x2")
    parser.add_argument(
        "--constants",
        type=Path,
        default=WORKSPACE / "models/_work/graspgenx_omnihand_left_x2/diffusion_om_g400/loop_constants_g400.npz",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=WORKSPACE / "models/graspgenx_omnihand_left_x2/artifacts/ascend_310p/diffusion_loop.om",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    arguments = parser.parse_args()

    if arguments.compare:
        return compare(*arguments.compare)
    if arguments.run_om:
        return run_om(*arguments.run_om)
    if arguments.reference is None:
        parser.error("pass --reference, --run-om or --compare")

    inputs = loop_inputs(arguments.constants, arguments.seed)
    print(f"seed {arguments.seed}; inputs " + ", ".join(f"{k}{v.shape}" for k, v in inputs.items()), flush=True)
    produced = eager_reference(arguments.bundle, arguments.constants, inputs)
    import yaml

    assets = json.loads((arguments.bundle / "assets/adapter.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (arguments.bundle / assets["checkpoints_dir"] / "gen/config.yaml").read_text(encoding="utf-8")
    )
    kappa = config["diffusion"].get("kappa") or 1.0
    np.savez(arguments.reference, graph=str(arguments.graph), seed=arguments.seed, kappa=kappa, **inputs, **produced)
    print(f"\ngrasps     {produced['grasps'].shape}")
    translation = produced["grasps"][:, :3] / kappa
    print(f"translation range (m) {translation.min():.4f} .. {translation.max():.4f}")
    print(f"wrote      {arguments.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
