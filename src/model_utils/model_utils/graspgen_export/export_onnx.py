# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Export GraspGen neural subgraphs to static ONNX models."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from model_utils.graspgen_export.modeling import (
    GraspGenDenoiser,
    GraspGenDiscriminatorHead,
    build_pointnet_segments,
    load_checkpoint_state,
)

MANIFEST_NAME = "graspgen.onnx.json"
ARTIFACT_ORDER = [
    "generator_sa1",
    "generator_sa2",
    "generator_encoder_head",
    "discriminator_sa1",
    "discriminator_sa2",
    "discriminator_encoder_head",
    "denoiser",
    "discriminator_head",
]


@dataclass
class ExportArtifact:
    name: str
    model: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    output_names: list[str]


def _resolve_checkpoint(config_path: Path, configured_path: str, explicit_path: str | None) -> Path:
    candidate = Path(explicit_path).expanduser() if explicit_path else Path(str(configured_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    return candidate


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"config must be a YAML object: {path}")
    diffusion = config.get("diffusion", {})
    discriminator = config.get("discriminator", {})
    required = {
        "diffusion.obs_backbone": diffusion.get("obs_backbone"),
        "diffusion.pose_repr": diffusion.get("pose_repr"),
        "diffusion.grasp_repr": diffusion.get("grasp_repr"),
        "discriminator.obs_backbone": discriminator.get("obs_backbone"),
        "discriminator.pose_repr": discriminator.get("pose_repr"),
        "discriminator.grasp_repr": discriminator.get("grasp_repr"),
    }
    expected = {
        "diffusion.obs_backbone": "pointnet",
        "diffusion.pose_repr": "mlp",
        "diffusion.grasp_repr": "r3_so3",
        "discriminator.obs_backbone": "pointnet",
        "discriminator.pose_repr": "mlp",
        "discriminator.grasp_repr": "r3_so3",
    }
    mismatches = [
        f"{key}={required[key]!r}, expected {value!r}" for key, value in expected.items() if required[key] != value
    ]
    if mismatches:
        raise ValueError("unsupported GraspGen configuration: " + "; ".join(mismatches))
    return config


def _artifact_specs(
    generator_state: dict[str, torch.Tensor],
    discriminator_state: dict[str, torch.Tensor],
    grasp_batch_size: int,
) -> dict[str, ExportArtifact]:
    generator_sa1, generator_sa2, generator_head = build_pointnet_segments(generator_state)
    discriminator_sa1, discriminator_sa2, discriminator_head = build_pointnet_segments(discriminator_state)
    denoiser = GraspGenDenoiser.from_state_dict(generator_state)
    scorer = GraspGenDiscriminatorHead.from_state_dict(discriminator_state)

    return {
        "generator_sa1": ExportArtifact(
            "generator_sa1",
            generator_sa1,
            (torch.randn(1, 3, 256, 64),),
            ["grouped_features"],
            ["features"],
        ),
        "generator_sa2": ExportArtifact(
            "generator_sa2",
            generator_sa2,
            (torch.randn(1, 131, 64, 128),),
            ["grouped_features"],
            ["features"],
        ),
        "generator_encoder_head": ExportArtifact(
            "generator_encoder_head",
            generator_head,
            (torch.randn(1, 259, 1, 64),),
            ["grouped_features"],
            ["object_embedding"],
        ),
        "discriminator_sa1": ExportArtifact(
            "discriminator_sa1",
            discriminator_sa1,
            (torch.randn(1, 3, 256, 64),),
            ["grouped_features"],
            ["features"],
        ),
        "discriminator_sa2": ExportArtifact(
            "discriminator_sa2",
            discriminator_sa2,
            (torch.randn(1, 131, 64, 128),),
            ["grouped_features"],
            ["features"],
        ),
        "discriminator_encoder_head": ExportArtifact(
            "discriminator_encoder_head",
            discriminator_head,
            (torch.randn(1, 259, 1, 64),),
            ["grouped_features"],
            ["object_embedding"],
        ),
        "denoiser": ExportArtifact(
            "denoiser",
            denoiser,
            (
                torch.randn(1, 512),
                torch.randn(grasp_batch_size, 6),
                torch.full((1,), 9.0),
            ),
            ["object_embedding", "sample", "timestep"],
            ["predicted_noise"],
        ),
        "discriminator_head": ExportArtifact(
            "discriminator_head",
            scorer,
            (torch.randn(1, 512), torch.randn(grasp_batch_size, 6)),
            ["object_embedding", "grasp_rt"],
            ["logits", "confidence"],
        ),
    }


def _as_output_list(value: Any) -> list[torch.Tensor]:
    return list(value) if isinstance(value, tuple | list) else [value]


def _cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    first = reference.reshape(-1).astype(np.float64)
    second = candidate.reshape(-1).astype(np.float64)
    if max(np.max(np.abs(first)), np.max(np.abs(second))) < 1e-8:
        return 1.0
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm < 1e-12 and second_norm < 1e-12:
        return 1.0
    if first_norm < 1e-12 or second_norm < 1e-12:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def _verify_onnx(path: Path, artifact: ExportArtifact) -> None:
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        print("  onnxruntime is not installed; skipping numerical verification")
        return

    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(artifact.input_names, artifact.example_inputs, strict=True)
    }
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, feed)
    with torch.no_grad():
        torch_outputs = _as_output_list(artifact.model(*artifact.example_inputs))
    for name, reference, candidate in zip(artifact.output_names, torch_outputs, onnx_outputs, strict=True):
        reference_np = reference.detach().cpu().numpy()
        diff = np.abs(reference_np.astype(np.float64) - candidate.astype(np.float64))
        print(
            f"  verify {name}: max_abs={diff.max():.6e}, "
            f"mean_abs={diff.mean():.6e}, cosine={_cosine(reference_np, candidate):.8f}"
        )


def _export_artifact(
    artifact: ExportArtifact,
    output_dir: Path,
    opset: int,
    verify: bool,
    constant_folding: bool,
    simplify: bool,
) -> dict[str, Any]:
    path = output_dir / f"{artifact.name}.onnx"
    print(f"Exporting {artifact.name} -> {path}")
    artifact.model.eval()
    with torch.no_grad():
        torch.onnx.export(
            artifact.model,
            artifact.example_inputs,
            str(path),
            input_names=artifact.input_names,
            output_names=artifact.output_names,
            opset_version=opset,
            do_constant_folding=constant_folding,
            verbose=False,
        )

    onnx = importlib.import_module("onnx")

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    if simplify:
        onnxsim = importlib.import_module("onnxsim")
        model, valid = onnxsim.simplify(model)
        if not valid:
            raise RuntimeError(f"onnxsim validation failed for {artifact.name}")
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    operators = Counter(f"{node.domain}::{node.op_type}" if node.domain else node.op_type for node in model.graph.node)
    if verify:
        _verify_onnx(path, artifact)
    return {
        "onnx": path.name,
        "inputs": {
            name: list(tensor.shape) for name, tensor in zip(artifact.input_names, artifact.example_inputs, strict=True)
        },
        "outputs": list(artifact.output_names),
        "operators": dict(sorted(operators.items())),
        "file_size_bytes": path.stat().st_size,
    }


def _write_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    generator_checkpoint: Path,
    discriminator_checkpoint: Path,
    artifacts: dict[str, Any],
    grasp_batch_size: int,
    opset: int,
    constant_folding: bool,
    simplify: bool,
) -> Path:
    diffusion = config["diffusion"]
    data = config["data"]
    manifest = {
        "schema_version": 1,
        "model_type": "graspgen",
        "backend": "onnx",
        "source": {
            "config": str(config_path),
            "generator_checkpoint": str(generator_checkpoint),
            "discriminator_checkpoint": str(discriminator_checkpoint),
        },
        "artifacts": artifacts,
        "execution": [name for name in ARTIFACT_ORDER if name in artifacts],
        "backend_config": {
            "opset": opset,
            "constant_folding": constant_folding,
            "simplified": simplify,
            "grasp_batch_size": grasp_batch_size,
            "point_count": int(data.get("num_points", 2048)),
            "grasp_repr": str(diffusion["grasp_repr"]),
            "kappa": float(diffusion["kappa"]),
            "diffusion_steps": int(diffusion["num_diffusion_iters_eval"]),
            "compositional_scheduler": bool(diffusion["compositional_schedular"]),
            "geometry": {
                "npoints": [256, 64, None],
                "radii": [0.02, 0.04, None],
                "nsamples": [64, 128, None],
                "fps_start_index": 0,
                "ball_query_order": "input_index",
            },
        },
    }
    path = output_dir / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GraspGen neural subgraphs to ONNX")
    parser.add_argument("--config", required=True, help="GraspGen gripper YAML config")
    parser.add_argument("--generator-checkpoint", default=None)
    parser.add_argument("--discriminator-checkpoint", default=None)
    parser.add_argument("--output-dir", default="./output/onnx")
    parser.add_argument("--grasp-batch-size", type=int, default=1000)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--disable-constant-folding",
        action="store_true",
        help="Disable ONNX constant folding (enabled by default for ATC-friendly graphs)",
    )
    parser.add_argument(
        "--disable-simplify",
        action="store_true",
        help="Disable onnxsim cleanup (enabled by default to remove export shape scaffolding)",
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        choices=["all", *ARTIFACT_ORDER],
        default=["all"],
        help="Subset to export; default exports all eight subgraphs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grasp_batch_size <= 0:
        raise ValueError("--grasp-batch-size must be positive")
    torch.manual_seed(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = _load_config(config_path)
    generator_checkpoint = _resolve_checkpoint(
        config_path,
        config["eval"]["checkpoint"],
        args.generator_checkpoint,
    )
    discriminator_checkpoint = _resolve_checkpoint(
        config_path,
        config["discriminator"]["checkpoint"],
        args.discriminator_checkpoint,
    )

    print(f"Loading generator checkpoint: {generator_checkpoint}")
    generator_state = load_checkpoint_state(str(generator_checkpoint))
    print(f"Loading discriminator checkpoint: {discriminator_checkpoint}")
    discriminator_state = load_checkpoint_state(str(discriminator_checkpoint))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = ARTIFACT_ORDER if "all" in args.artifacts else args.artifacts
    specs = _artifact_specs(generator_state, discriminator_state, args.grasp_batch_size)
    artifacts = {
        name: _export_artifact(
            specs[name],
            output_dir,
            args.opset,
            not args.skip_verify,
            not args.disable_constant_folding,
            not args.disable_simplify,
        )
        for name in requested
    }
    manifest = _write_manifest(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        generator_checkpoint=generator_checkpoint,
        discriminator_checkpoint=discriminator_checkpoint,
        artifacts=artifacts,
        grasp_batch_size=args.grasp_batch_size,
        opset=args.opset,
        constant_folding=not args.disable_constant_folding,
        simplify=not args.disable_simplify,
    )
    print(f"ONNX export complete. Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
