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

from inference_manifest import (
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_EXECUTION,
    GRASPGEN_NPOINTS,
    GRASPGEN_NSAMPLES,
    graspgen_geometry,
)
from model_utils.graspgen_export.modeling import (
    GraspGenDenoiser,
    GraspGenDiscriminatorHead,
    build_pointnet_segments,
    load_checkpoint_state,
)

MANIFEST_NAME = "graspgen.onnx.json"

# Numerical envelope every exported subgraph must stay inside before its manifest is
# published. The metrics are normalised by the largest reference activation so that a
# single envelope covers subgraphs whose outputs span very different magnitudes
# (PointNet features around 1e0, denoiser noise around 1e0, discriminator logits an
# order of magnitude higher).
#
# The envelope is sized for float32 CPU onnxruntime against float32 CPU PyTorch, where
# the only legitimate difference is GEMM/Conv reassociation. Measured across all eight
# subgraphs at production dimensions (opset 14, constant folding on, grasp batch 1000)
# the worst observed deviation was max_relative 1.8e-6 and mean_relative 3.7e-7 on the
# denoiser, with cosine 1.0 everywhere; the limits below keep roughly fifty times that
# headroom. Every structural defect the export can introduce - a mis-mapped weight
# slice, a dropped bias, a wrong activation, a transposed grouping axis - is at least
# 1e-2 relative, four orders of magnitude outside, and also breaks the cosine gate.
# Each run records its own measured numbers under ``artifacts[*].verification`` in the
# manifest, so the envelope can be re-baselined against real checkpoints;
# ``--verify-tolerance-scale`` widens it without editing code.
VERIFY_MAX_RELATIVE = 1e-4
VERIFY_MEAN_RELATIVE = 1e-5
VERIFY_COSINE_DEFICIT = 1e-6

ARTIFACT_ORDER = list(GRASPGEN_EXECUTION)


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

    # The traced example shapes become the OMs' static input shapes, so they have to be
    # the same neighbourhood sizes the backend groups points into at runtime. The channel
    # counts come from the checkpoints: stage two sees stage one's 128 features plus the
    # 3 relative coordinates, and the head sees stage two's 256 plus the same 3.
    stage1_shape = (1, 3, GRASPGEN_NPOINTS[0], GRASPGEN_NSAMPLES[0])
    stage2_shape = (1, 131, GRASPGEN_NPOINTS[1], GRASPGEN_NSAMPLES[1])
    head_shape = (1, 259, 1, GRASPGEN_NPOINTS[1])

    return {
        "generator_sa1": ExportArtifact(
            "generator_sa1",
            generator_sa1,
            (torch.randn(*stage1_shape),),
            ["grouped_features"],
            ["features"],
        ),
        "generator_sa2": ExportArtifact(
            "generator_sa2",
            generator_sa2,
            (torch.randn(*stage2_shape),),
            ["grouped_features"],
            ["features"],
        ),
        "generator_encoder_head": ExportArtifact(
            "generator_encoder_head",
            generator_head,
            (torch.randn(*head_shape),),
            ["grouped_features"],
            ["object_embedding"],
        ),
        "discriminator_sa1": ExportArtifact(
            "discriminator_sa1",
            discriminator_sa1,
            (torch.randn(*stage1_shape),),
            ["grouped_features"],
            ["features"],
        ),
        "discriminator_sa2": ExportArtifact(
            "discriminator_sa2",
            discriminator_sa2,
            (torch.randn(*stage2_shape),),
            ["grouped_features"],
            ["features"],
        ),
        "discriminator_encoder_head": ExportArtifact(
            "discriminator_encoder_head",
            discriminator_head,
            (torch.randn(*head_shape),),
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


def _output_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Measure one output against its PyTorch reference, absolutely and relatively."""
    reference64 = reference.astype(np.float64)
    diff = np.abs(reference64 - candidate.astype(np.float64))
    scale = float(np.max(np.abs(reference64)))
    if not scale > 0.0:
        # An all-zero reference has no relative scale; fall back to the absolute
        # numbers so a nonzero candidate still trips the gate instead of dividing
        # by zero into a NaN that compares false against every limit.
        scale = 1.0
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_relative": float(diff.max()) / scale,
        "mean_relative": float(diff.mean()) / scale,
        "cosine": _cosine(reference, candidate),
        "reference_scale": scale,
    }


def _tolerance_violations(name: str, metrics: dict[str, float], tolerance_scale: float) -> list[str]:
    max_limit = VERIFY_MAX_RELATIVE * tolerance_scale
    mean_limit = VERIFY_MEAN_RELATIVE * tolerance_scale
    cosine_limit = 1.0 - VERIFY_COSINE_DEFICIT * tolerance_scale
    violations = []
    if not metrics["max_relative"] <= max_limit:
        violations.append(f"{name} max_relative={metrics['max_relative']:.6e} exceeds {max_limit:.6e}")
    if not metrics["mean_relative"] <= mean_limit:
        violations.append(f"{name} mean_relative={metrics['mean_relative']:.6e} exceeds {mean_limit:.6e}")
    if not metrics["cosine"] >= cosine_limit:
        violations.append(f"{name} cosine={metrics['cosine']:.8f} below {cosine_limit:.8f}")
    return violations


def _verify_onnx(path: Path, artifact: ExportArtifact, tolerance_scale: float = 1.0) -> dict[str, dict[str, float]]:
    """Compare the exported graph against PyTorch and fail closed outside the envelope.

    Returns the measured metrics so the manifest can publish the numbers the gate was
    decided on. Raises ``RuntimeError`` when any output leaves the envelope; the caller
    is responsible for making sure neither the graph nor a manifest describing it is
    published after that.
    """
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError as error:
        # Silently skipping verification would publish an unverified graph under the
        # same manifest shape as a verified one, so an unusable verifier is a failure.
        raise RuntimeError(
            "onnxruntime is required to verify exported GraspGen subgraphs; "
            "install it or pass --skip-verify to export without numerical verification"
        ) from error

    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(artifact.input_names, artifact.example_inputs, strict=True)
    }
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, feed)
    with torch.no_grad():
        torch_outputs = _as_output_list(artifact.model(*artifact.example_inputs))

    metrics: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    for name, reference, candidate in zip(artifact.output_names, torch_outputs, onnx_outputs, strict=True):
        measured = _output_metrics(reference.detach().cpu().numpy(), candidate)
        metrics[name] = measured
        print(
            f"  verify {name}: max_abs={measured['max_abs']:.6e}, "
            f"mean_abs={measured['mean_abs']:.6e}, cosine={measured['cosine']:.8f}"
        )
        violations.extend(_tolerance_violations(name, measured, tolerance_scale))
    if violations:
        raise RuntimeError(f"ONNX verification failed for {artifact.name}: " + "; ".join(violations))
    return metrics


def _export_artifact(
    artifact: ExportArtifact,
    output_dir: Path,
    opset: int,
    verify: bool,
    constant_folding: bool,
    simplify: bool,
    tolerance_scale: float = 1.0,
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
    verification: dict[str, dict[str, float]] | None = None
    if verify:
        try:
            verification = _verify_onnx(path, artifact, tolerance_scale)
        except RuntimeError:
            # A graph that failed the numerical gate must not survive on disk, where a
            # later onnx2om run would happily compile it into an OM that nothing has
            # checked. The manifest for this run is never written either, because the
            # exception propagates out of main() before _write_manifest.
            path.unlink(missing_ok=True)
            raise
    record: dict[str, Any] = {
        "onnx": path.name,
        "inputs": {
            name: list(tensor.shape) for name, tensor in zip(artifact.input_names, artifact.example_inputs, strict=True)
        },
        "outputs": list(artifact.output_names),
        "operators": dict(sorted(operators.items())),
        "file_size_bytes": path.stat().st_size,
    }
    if verification is not None:
        record["verification"] = verification
    return record


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
        "contract_version": GRASPGEN_CONTRACT_VERSION,
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
            # One entry per exported set-abstraction stage, so the encoder head's null
            # stage is listed alongside the two sampled ones.
            "geometry": graspgen_geometry(include_head_stage=True),
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
        "--verify-tolerance-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the baseline verification envelope "
            f"(max_relative={VERIFY_MAX_RELATIVE:g}, mean_relative={VERIFY_MEAN_RELATIVE:g}, "
            f"cosine>=1-{VERIFY_COSINE_DEFICIT:g}); use it to re-baseline, not to hide a regression"
        ),
    )
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
    if not args.verify_tolerance_scale > 0.0:
        raise ValueError("--verify-tolerance-scale must be positive")
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
    # The manifest is the publication point: onnx2om and every downstream consumer read
    # it, not the loose .onnx files. Clearing it before the export means a run that
    # fails the numerical gate leaves no manifest at all, rather than an older one that
    # still advertises artifacts this run has just replaced or deleted.
    (output_dir / MANIFEST_NAME).unlink(missing_ok=True)
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
            args.verify_tolerance_scale,
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
