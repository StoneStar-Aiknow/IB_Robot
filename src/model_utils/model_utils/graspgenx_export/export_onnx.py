#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Assemble the GraspGen-X diffusion loop into one ONNX graph.

The eager loop is dispatch bound -- a 400-grasp request issues roughly 2200
operators for the generator alone, for 78 ms of device time -- so the twenty
steps are unrolled into a single graph covering the denoising head, both
compositional DDPM schedulers and the likelihood accumulation. One
`aclmdlExecute` then replaces about 1500 launches.

Every per-timestep constant (step embedding, scheduler coefficients,
likelihood constants) is read out of the constructed model rather than
re-derived, so the graph uses the numbers the eager path uses. The head's
weights are shared initializers, so unrolling twenty steps does not duplicate
them.

The graph is float32 end to end. Compile it with `--precision-mode force_fp32`,
which on Ascend310P1 means float16 multiplies with float32 accumulation -- the
arithmetic class the eager path already gets, since the chip has no float32
cube. `must_keep_origin_dtype` is genuinely float32 but ran a single 2048x2048
projection in 25 ms against 0.25 ms, so it is not a usable option.

Reading the constants needs Torch-NPU and the bundle's checkpoints, so this
runs on the device. It also needs `onnx`, which is not in the base image:
install it into the workspace venv first.

Outputs, for G grasps and S steps:
  grasps            (G, 6)      final sample, before rt_to_matrix
  likelihood        (G, 1)      accumulated as the eager loop accumulates it
  noise_predictions (S, G, 6)   per-step head output, for baseline comparison
  latents           (S, G, 6)   per-step sample, for baseline comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from model_utils.export_paths import export_work_dir
from model_utils.graspgenx_export.binding import generator_binding, validate_binding
from torch_models.graspgenx_ascend_310p.diffusion_scheduler import AscendDDPMStep, ddpm_coefficients

MANIFEST_NAME = "graspgenx.onnx.json"


def _bundle_assets(bundle_root: Path) -> dict:
    return json.loads((bundle_root / "assets/adapter.json").read_text(encoding="utf-8"))


def _assembly():
    """The deployment's own model assembly, from an install or from source.

    Adding just this one package beats asking the caller for a PYTHONPATH over
    every package in src/: that puts each package's directory on the import
    path, where a directory name can shadow a module something else needs.
    """
    try:
        from torch_models.graspgenx_ascend_310p import assemble
    except ImportError:
        # <workspace>/src/model_utils/model_utils/graspgenx_export/<this file>
        workspace = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[4])
        candidate = workspace / "src/torch_models"
        if not (candidate / "torch_models").is_dir():
            raise
        sys.path.insert(0, str(candidate))
        from torch_models.graspgenx_ascend_310p import assemble
    return assemble


def dump_constants(bundle_root: Path, destination: Path, grasps: int) -> None:
    """Read the loop's per-timestep constants out of the assembled model.

    The model comes from the same assembly the deployment uses, so the graph
    cannot be built from a differently configured generator than the one it
    will stand in for.
    """
    assemble = _assembly()

    assets = _bundle_assets(bundle_root)
    model = assemble(
        bundle_root,
        assets,
        device_name=torch.npu.get_device_name(0),
        num_grasps=grasps,
    )
    generator = model.grasp_generator
    device = next(generator.diffusion_head.parameters()).device
    generator._install_inference_head()
    steps = generator.num_diffusion_iters_eval
    schedulers = {"pos": generator.noise_scheduler_pos, "rot": generator.noise_scheduler_rot}
    for scheduler in schedulers.values():
        scheduler.set_timesteps(steps)
        if not AscendDDPMStep(scheduler).supported():
            raise ValueError("the graph exporter requires a supported epsilon DDPM scheduler")
    timesteps = [int(timestep) for timestep in schedulers["pos"].timesteps]

    with torch.inference_mode():
        embeddings = np.stack(
            [
                generator.diffusion_head._step_embedding(
                    torch.tensor(timestep),
                    device,
                )
                .float()
                .cpu()
                .numpy()[0]
                for timestep in timesteps
            ]
        )
        coefficients, likelihood = {}, {}
        for name, scheduler in schedulers.items():
            rows = []
            for timestep in timesteps:
                rows.append(ddpm_coefficients(scheduler, timestep))
            coefficients[name] = np.array(rows, dtype=np.float64)
            scales = generator._prepare_likelihood_scales(scheduler.betas, device)
            likelihood[name] = np.array(
                [[float(scales[timestep][0]), scales[timestep][1]] for timestep in timesteps],
                dtype=np.float64,
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        timesteps=np.array(timesteps, dtype=np.int64),
        step_embeddings=embeddings.astype(np.float32),
        coefficients_pos=coefficients["pos"],
        coefficients_rot=coefficients["rot"],
        likelihood_pos=likelihood["pos"],
        likelihood_rot=likelihood["rot"],
        clip_sample=np.array(bool(schedulers["pos"].config.clip_sample)),
        generator_binding=json.dumps(
            generator_binding(
                bundle_root / assets["generator_checkpoint"],
                bundle_root / assets["checkpoints_dir"] / "gen/config.yaml",
            )
        ),
        grasps=grasps,
        steps=steps,
    )
    print(f"wrote {destination}: {len(timesteps)} timesteps", flush=True)


# Graph assembly, carried over verbatim from the implementation that produced
# the graph the shipped bundle was compiled from. Any edit here changes the
# ONNX, so parity against that graph is checked by digest, not by review.
def build(constants, tensors, grasps: int, steps: int):
    """Emit the unrolled loop. Returns (an ONNX graph, metadata)."""
    from onnx import TensorProto, helper, numpy_helper

    embed = tensors["query_pos_enc.weight"].shape[1]
    step_dim = constants["step_embeddings"].shape[1]
    sample_dim = tensors["sample_encoder.0.weight"].shape[1]
    sample_embed = tensors["sample_encoder.2.weight"].shape[0]
    observation_dim = embed - step_dim - sample_embed
    half = sample_dim // 2
    initializers, nodes, seen = [], [], set()

    def constant(name, array):
        if name not in seen:
            seen.add(name)
            initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def scalar(name, value):
        return constant(name, np.float32(value))

    def node(op, inputs, output, **attributes):
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
        return output

    def gemm(name, source, weight_key, bias_key):
        return node(
            "Gemm",
            [source, constant(f"{weight_key}_w", tensors[weight_key]), constant(f"{bias_key}_b", tensors[bias_key])],
            name,
            transB=1,
        )

    def layer_norm(name, source, prefix):
        return node(
            "LayerNormalization",
            [
                source,
                constant(f"{prefix}.weight_g", tensors[f"{prefix}.weight"]),
                constant(f"{prefix}.bias_g", tensors[f"{prefix}.bias"]),
            ],
            name,
            axis=-1,
            epsilon=1e-5,
        )

    def gelu(name, source):
        """nn.GELU()'s exact error-function form, not the tanh approximation."""
        scaled = node("Mul", [source, scalar("rsqrt2", 0.7071067811865476)], f"{name}_scaled")
        shifted = node("Add", [node("Erf", [scaled], f"{name}_erf"), scalar("one", 1.0)], f"{name}_shifted")
        halved = node("Mul", [shifted, scalar("pointfive", 0.5)], f"{name}_halved")
        return node("Mul", [source, halved], name)

    def folded_attention(index):
        """out_proj(W_v x + b_v) with the constant position encoding in the bias.

        At sequence length 1 the softmax is 1 and the query and key projections
        cannot reach the output, so the layer is one affine map; see
        generator_ascend._CollapsedSelfAttention.
        """
        prefix = f"self_attention_layers.{index}.attn"
        output_weight = tensors[f"{prefix}.out_proj.weight"]
        weight = output_weight @ tensors[f"{prefix}.in_proj_weight"][2 * embed :]
        bias = (
            output_weight @ tensors[f"{prefix}.in_proj_bias"][2 * embed :]
            + tensors[f"{prefix}.out_proj.bias"]
            + weight @ tensors["query_pos_enc.weight"].reshape(embed)
        )
        return weight, bias

    attention = [folded_attention(index) for index in range(3)]

    def component(name, source, start, end):
        return node(
            "Slice",
            [
                source,
                constant(f"start_{start}", np.array([start], dtype=np.int64)),
                constant(f"end_{end}", np.array([end], dtype=np.int64)),
                constant("axis_last", np.array([-1], dtype=np.int64)),
            ],
            name,
        )

    def scheduler_step(prefix, index, sample, prediction, noise):
        beta_sqrt, alpha_sqrt, original_coeff, sample_coeff, variance_sqrt = constants[f"coefficients_{prefix}"][index]
        scaled = node("Mul", [prediction, scalar(f"{prefix}_beta_{index}", beta_sqrt)], f"{prefix}_scaled_{index}")
        original = node(
            "Div",
            [node("Sub", [sample, scaled], f"{prefix}_centred_{index}"), scalar(f"{prefix}_alpha_{index}", alpha_sqrt)],
            f"{prefix}_original_raw_{index}",
        )
        if bool(constants["clip_sample"]):
            original = node(
                "Clip", [original, scalar("clip_low", -1.0), scalar("clip_high", 1.0)], f"{prefix}_original_{index}"
            )
        previous = node(
            "Add",
            [
                node("Mul", [original, scalar(f"{prefix}_ocoeff_{index}", original_coeff)], f"{prefix}_oterm_{index}"),
                node("Mul", [sample, scalar(f"{prefix}_scoeff_{index}", sample_coeff)], f"{prefix}_sterm_{index}"),
            ],
            f"{prefix}_previous_raw_{index}",
        )
        if noise is not None:
            previous = node(
                "Add",
                [
                    previous,
                    node("Mul", [noise, scalar(f"{prefix}_var_{index}", variance_sqrt)], f"{prefix}_noise_{index}"),
                ],
                f"{prefix}_previous_{index}",
            )
        return previous, original

    def likelihood_term(prefix, index, sample, original):
        negative_double_variance, offset = constants[f"likelihood_{prefix}"][index]
        difference = node("Sub", [sample, original], f"{prefix}_difference_{index}")
        squared = node(
            "ReduceSum",
            [
                node("Mul", [difference, difference], f"{prefix}_squared_{index}"),
                constant("axis_last", np.array([-1], dtype=np.int64)),
            ],
            f"{prefix}_sum_{index}",
            keepdims=1,
        )
        return node(
            "Add",
            [
                node(
                    "Div",
                    [squared, scalar(f"{prefix}_ndv_{index}", negative_double_variance)],
                    f"{prefix}_scaled_sum_{index}",
                ),
                scalar(f"{prefix}_offset_{index}", half * offset),
            ],
            f"{prefix}_likelihood_{index}",
        )

    sample = "initial_noise"
    accumulated = None
    predictions, latents = [], []
    for index in range(steps):
        timestep = int(constants["timesteps"][index])
        tag = f"s{index}"
        embedded = gemm(f"{tag}_sample_0", sample, "sample_encoder.0.weight", "sample_encoder.0.bias")
        embedded = node("Relu", [embedded], f"{tag}_sample_relu")
        embedded = gemm(f"{tag}_sample_2", embedded, "sample_encoder.2.weight", "sample_encoder.2.bias")
        step_embedding = node(
            "Expand",
            [
                constant(f"step_embedding_{index}", constants["step_embeddings"][index].reshape(1, step_dim)),
                constant("step_shape", np.array([grasps, step_dim], dtype=np.int64)),
            ],
            f"{tag}_step",
        )
        features = node("Concat", [embedded, step_embedding, "observation"], f"{tag}_features", axis=1)
        for layer in range(3):
            weight, bias = attention[layer]
            projected = node(
                "Gemm",
                [features, constant(f"attention_{layer}_w", weight), constant(f"attention_{layer}_b", bias)],
                f"{tag}_attention_{layer}",
                transB=1,
            )
            features = layer_norm(
                f"{tag}_attention_norm_{layer}",
                node("Add", [features, projected], f"{tag}_attention_res_{layer}"),
                f"self_attention_layers.{layer}.norm",
            )
            inner = gemm(
                f"{tag}_ffn_{layer}_0", features, f"ffn_layers.{layer}.ff.0.weight", f"ffn_layers.{layer}.ff.0.bias"
            )
            inner = gelu(f"{tag}_ffn_{layer}_gelu", inner)
            inner = gemm(
                f"{tag}_ffn_{layer}_2", inner, f"ffn_layers.{layer}.ff.2.weight", f"ffn_layers.{layer}.ff.2.bias"
            )
            features = layer_norm(
                f"{tag}_ffn_norm_{layer}",
                node("Add", [features, inner], f"{tag}_ffn_res_{layer}"),
                f"ffn_layers.{layer}.norm",
            )
        features = node(
            "Relu",
            [gemm(f"{tag}_prediction_0", features, "prediction_head.0.weight", "prediction_head.0.bias")],
            f"{tag}_prediction_relu_0",
        )
        features = node(
            "Relu",
            [gemm(f"{tag}_prediction_2", features, "prediction_head.2.weight", "prediction_head.2.bias")],
            f"{tag}_prediction_relu_1",
        )
        prediction = gemm(f"{tag}_prediction", features, "prediction_head.4.weight", "prediction_head.4.bias")
        predictions.append(prediction)

        parts = {
            "pos": (component(f"{tag}_sample_pos", sample, 0, half), component(f"{tag}_pred_pos", prediction, 0, half)),
            "rot": (
                component(f"{tag}_sample_rot", sample, half, sample_dim),
                component(f"{tag}_pred_rot", prediction, half, sample_dim),
            ),
        }
        noises = {"pos": None, "rot": None}
        if timestep > 0:
            # ATC keeps the gathered rank, so the row is reshaped to rank 2
            # explicitly: a rank-3 operand broadcasts through the scheduler and
            # the next step's Gemm rejects it.
            selector = constant(f"gather_{index}", np.array(index, dtype=np.int64))
            shape = constant("noise_shape", np.array([grasps, half], dtype=np.int64))
            for prefix, source in (("pos", "position_noise"), ("rot", "rotation_noise")):
                noises[prefix] = node(
                    "Reshape",
                    [node("Gather", [source, selector], f"{tag}_noise_{prefix}_row", axis=0), shape],
                    f"{tag}_noise_{prefix}",
                )

        updated, originals = {}, {}
        for prefix, (component_sample, component_prediction) in parts.items():
            updated[prefix], originals[prefix] = scheduler_step(
                prefix, index, component_sample, component_prediction, noises[prefix]
            )
        if timestep > 0:
            term = node(
                "Add",
                [likelihood_term(prefix, index, parts[prefix][0], originals[prefix]) for prefix in ("pos", "rot")],
                f"{tag}_likelihood",
            )
            accumulated = term if accumulated is None else node("Add", [accumulated, term], f"{tag}_likelihood_total")
        sample = node("Concat", [updated["pos"], updated["rot"]], f"{tag}_next", axis=-1)
        latents.append(sample)

    node("Identity", [sample], "grasps")
    node("Identity", [accumulated], "likelihood")
    for name, values in (("noise_predictions", predictions), ("latents", latents)):
        expanded = [
            node("Unsqueeze", [value, constant("axis_zero", np.array([0], dtype=np.int64))], f"{value}_expanded")
            for value in values
        ]
        nodes.append(helper.make_node("Concat", expanded, [name], name=name, axis=0))

    metadata = {
        "grasps": grasps,
        "steps": steps,
        "observation": observation_dim,
        "sample": sample_dim,
        "half": half,
        "step": step_dim,
        "nodes": len(nodes),
        "initializers": len(initializers),
        "timesteps": [int(timestep) for timestep in constants["timesteps"]],
        "inputs": ["observation", "initial_noise", "position_noise", "rotation_noise"],
        "outputs": ["grasps", "likelihood", "noise_predictions", "latents"],
    }
    graph = helper.make_graph(
        nodes,
        "graspgenx_diffusion_loop",
        [
            helper.make_tensor_value_info("observation", TensorProto.FLOAT, [grasps, observation_dim]),
            helper.make_tensor_value_info("initial_noise", TensorProto.FLOAT, [grasps, sample_dim]),
            helper.make_tensor_value_info("position_noise", TensorProto.FLOAT, [steps - 1, grasps, half]),
            helper.make_tensor_value_info("rotation_noise", TensorProto.FLOAT, [steps - 1, grasps, half]),
        ],
        [
            helper.make_tensor_value_info("grasps", TensorProto.FLOAT, [grasps, sample_dim]),
            helper.make_tensor_value_info("likelihood", TensorProto.FLOAT, [grasps, 1]),
            helper.make_tensor_value_info("noise_predictions", TensorProto.FLOAT, [steps, grasps, sample_dim]),
            helper.make_tensor_value_info("latents", TensorProto.FLOAT, [steps, grasps, sample_dim]),
        ],
        initializers,
    )
    return graph, metadata


def resolve_grasps(declared: int, requested: int | None) -> int:
    """Return the row count to build for, which the bundle decides.

    The graph is shape specialised and the generator only reaches for it at
    exactly the bundle's ``num_grasps``, so a default of this exporter's own
    could only ever disagree with the bundle the packager wrote. An explicit
    request is still checked rather than quietly honoured: it says the caller
    meant a different number, and that graph would be declined at serve time.
    """
    if requested is None:
        return declared
    if requested != declared:
        raise SystemExit(
            f"the bundle declares num_grasps={declared} but this would build "
            f"a {requested}-row graph; the generator would decline it and run "
            f"the eager loop instead"
        )
    return requested


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root", type=Path, required=True, help="GraspGen-X bundle holding the generator checkpoint."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Defaults to models/_work/<bundle>/graspgenx_export."
    )
    parser.add_argument(
        "--grasps",
        type=int,
        default=None,
        help="Rows the graph is built for; defaults to the bundle's num_grasps, which it has to match.",
    )
    parser.add_argument(
        "--constants", type=Path, default=None, help="Reuse a constants dump instead of reading the model again."
    )
    parser.add_argument(
        "--dump-only", action="store_true", help="Write the constants and stop, for a host without onnx."
    )
    arguments = parser.parse_args()

    bundle_root = arguments.bundle_root.resolve()
    assets = _bundle_assets(bundle_root)
    grasps = resolve_grasps(int(assets["num_grasps"]), arguments.grasps)

    # export_work_dir resolves models/_work/<bundle>/graspgenx_export and, for
    # an explicit --output-dir too, refuses a destination inside the bundle: a
    # published bundle carries only manifest-referenced artifacts, and ONNX,
    # the constants dump and ATC's leavings are none of them.
    output_dir = export_work_dir(bundle_root, "graspgenx_export", arguments.output_dir)
    constants_path = arguments.constants or (output_dir / f"loop_constants_g{grasps}.npz")

    if arguments.constants is None:
        dump_constants(bundle_root, constants_path, grasps)
    if arguments.dump_only:
        print(f"constants only: {constants_path}")
        return 0

    import onnx
    from onnx import helper

    constants = np.load(constants_path)
    binding = generator_binding(
        bundle_root / assets["generator_checkpoint"],
        bundle_root / assets["checkpoints_dir"] / "gen/config.yaml",
    )
    # Old or copied constants must not attest a newly changed checkpoint/config.
    validate_binding(
        {
            "generator_binding": json.loads(str(constants["generator_binding"]))
            if "generator_binding" in constants
            else None,
            "grasps": int(constants["grasps"]) if "grasps" in constants else None,
            "steps": int(constants["timesteps"].shape[0]),
        },
        binding,
        grasps,
    )
    # The head's weights come from the checkpoint the bundle ships, so the
    # graph and the deployment cannot drift apart. The release checkpoints hold
    # the generator under "model" alongside the epoch and optimiser state;
    # "state_dict" is accepted too, since that is what a bare export looks like.
    checkpoint = torch.load(bundle_root / assets["generator_checkpoint"], map_location="cpu", weights_only=True)
    for container in ("model", "state_dict"):
        if isinstance(checkpoint.get(container), dict):
            checkpoint = checkpoint[container]
            break
    prefix = "diffusion_head."
    tensors = {key[len(prefix) :]: value.float().numpy() for key, value in checkpoint.items() if key.startswith(prefix)}
    if not tensors:
        raise SystemExit(
            f"no diffusion-head weights under {prefix!r}; the checkpoint's top-level keys are {sorted(checkpoint)[:6]}"
        )

    steps = int(constants["timesteps"].shape[0])
    graph, metadata = build(constants, tensors, grasps, steps)
    metadata["generator_binding"] = binding
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = output_dir / f"diffusion_loop_g{grasps}.onnx"
    onnx.save(model, onnx_path)
    metadata_path = onnx_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "model_type": "graspgenx",
        "grasps": grasps,
        "steps": steps,
        "onnx": onnx_path.name,
        "metadata": metadata_path.name,
        "sha256": _sha256(onnx_path),
        "metadata_sha256": _sha256(metadata_path),
        "generator_binding": binding,
        # Recorded so onnx2om can keep ATC's output and its leavings out of the
        # bundle the graph was built from; the compile step never reads it.
        "bundle_root": str(bundle_root),
        "generator_checkpoint_sha256": _sha256(bundle_root / assets["generator_checkpoint"]),
        "input_shape": {
            "observation": [grasps, metadata["observation"]],
            "initial_noise": [grasps, metadata["sample"]],
            "position_noise": [steps - 1, grasps, metadata["half"]],
            "rotation_noise": [steps - 1, grasps, metadata["half"]],
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {onnx_path}: {len(graph.node)} nodes, {len(graph.initializer)} initializers")
    print(f"sha256 {manifest['sha256']}")
    print(f"manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
