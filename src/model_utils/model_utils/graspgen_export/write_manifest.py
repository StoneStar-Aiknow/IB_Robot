# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Package the eight compiled GraspGen OM artifacts into a unified manifest.

This module bridges the Huawei GraspGen toolchain (which emits a flat
``graspgen.om.json``) and the IB-Robot unified ``inference_manifest.json`` so
that ``inference_service.backends.ascend.AscendBackend`` can load the eight
roles exactly as it loads ACT's single ``policy`` role or PI0.5's
``vlm``/``action_expert`` roles.

The semantic names assigned to bindings intentionally avoid the ``internal.*``
prefix: those bindings describe host-side geometry byproducts (FPS / ball
query / feature grouping) that the backend computes itself rather than reading
from another OM role, so they must not be enrolled in the manifest's
producer/consumer link validation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inference_manifest import ArtifactBindings, DeviceLink, TensorBinding
from model_utils.inference_manifest_export import (
    compiled_deployment,
    package_deployment_artifact,
    read_runtime_abi,
    upsert_deployment,
    write_acl_om_abi,
)

GRASPGEN_EXECUTION = (
    "generator_sa1",
    "generator_sa2",
    "generator_encoder_head",
    "discriminator_sa1",
    "discriminator_sa2",
    "discriminator_encoder_head",
    "denoiser",
    "discriminator_head",
)

_DEFAULT_GEOMETRY = {
    "npoints": [256, 64],
    "radii": [0.02, 0.04],
    "nsamples": [64, 128],
    "fps_start_index": 0,
    "ball_query_order": "input_index",
}

_GENERATOR_EMBEDDING = "internal.graspgen.generator_embedding"
_DISCRIMINATOR_EMBEDDING = "internal.graspgen.discriminator_embedding"
_GROUPED_FEATURE_ROLES = {
    "generator_sa1",
    "generator_sa2",
    "generator_encoder_head",
    "discriminator_sa1",
    "discriminator_sa2",
    "discriminator_encoder_head",
}


def _input_semantics(role: str) -> dict[str, str]:
    if role in {"generator_sa1", "discriminator_sa1"}:
        return {"grouped_features": "observation.object_points"}
    if role in {"generator_sa2", "discriminator_sa2"}:
        return {"grouped_features": "stage2_features"}
    if role in {"generator_encoder_head", "discriminator_encoder_head"}:
        return {"grouped_features": "global_features"}
    if role == "denoiser":
        return {
            "object_embedding": _GENERATOR_EMBEDDING,
            "sample": "diffusion.sample",
            "timestep": "diffusion.time",
        }
    if role == "discriminator_head":
        return {
            "object_embedding": _DISCRIMINATOR_EMBEDDING,
            "grasp_rt": "sample_rt",
        }
    raise ValueError(f"unknown graspgen role {role!r}")


def _output_semantics(role: str) -> dict[str, str]:
    if role in {"generator_sa1", "generator_sa2", "discriminator_sa1", "discriminator_sa2"}:
        return {"features": "features"}
    if role == "generator_encoder_head":
        return {"object_embedding": _GENERATOR_EMBEDDING}
    if role == "discriminator_encoder_head":
        return {"object_embedding": _DISCRIMINATOR_EMBEDDING}
    if role == "denoiser":
        return {"predicted_noise": "predicted_noise"}
    if role == "discriminator_head":
        return {"logits": "grasp.poses", "confidence": "grasp.confidence"}
    raise ValueError(f"unknown graspgen role {role!r}")


def _abi_for_role(
    om_abi_dir: Path,
    role: str,
    *,
    om_path: Path,
    inspect_missing_abi: bool,
    abi_device_id: int,
    acl_config_path: str | None,
) -> Any:
    abi_path = om_abi_dir / f"{role}.om.abi.json"
    if not abi_path.is_file():
        if not inspect_missing_abi:
            raise FileNotFoundError(f"Runtime-introspected OM ABI for role {role!r} not found: {abi_path}")
        write_acl_om_abi(
            om_path,
            abi_path,
            device_id=abi_device_id,
            acl_config_path=acl_config_path,
        )
    return read_runtime_abi(abi_path)


def _role_bindings(role: str, abi: Any) -> ArtifactBindings:
    input_semantics = tuple(_input_semantics(role).values())
    output_semantics = tuple(_output_semantics(role).values())
    if len(abi.inputs) != len(input_semantics):
        raise ValueError(f"role {role!r} runtime ABI has {len(abi.inputs)} inputs; expected {len(input_semantics)}")
    if len(abi.outputs) != len(output_semantics):
        raise ValueError(f"role {role!r} runtime ABI has {len(abi.outputs)} outputs; expected {len(output_semantics)}")

    inputs = [
        TensorBinding(
            semantic=semantic,
            runtime_name=tensor.name,
            index=tensor.index,
            dtype=tensor.dtype,
            shape=tensor.shape,
            layout=(
                tensor.layout
                if tensor.layout is not None
                else "NCHW"
                if role in _GROUPED_FEATURE_ROLES and len(tensor.shape) == 4
                else None
            ),
        )
        for tensor, semantic in zip(sorted(abi.inputs, key=lambda item: item.index), input_semantics, strict=True)
    ]
    outputs = [
        TensorBinding(
            semantic=semantic,
            runtime_name=tensor.name,
            index=tensor.index,
            dtype=tensor.dtype,
            shape=tensor.shape,
            layout=tensor.layout,
        )
        for tensor, semantic in zip(sorted(abi.outputs, key=lambda item: item.index), output_semantics, strict=True)
    ]
    return ArtifactBindings(inputs=tuple(inputs), outputs=tuple(outputs))


def _ensure_lerobot_config(
    bundle_root: Path,
    onnx_manifest: Mapping[str, object],
    *,
    point_count: int,
    grasp_batch_size: int,
) -> None:
    config_path = bundle_root / "config.json"
    if config_path.is_file():
        return
    backend_config = dict(onnx_manifest.get("backend_config") or {})
    diffusion_steps = int(backend_config.get("diffusion_steps", 10))
    kappa = float(backend_config.get("kappa", 2.02217))
    geometry = dict(backend_config.get("geometry") or _DEFAULT_GEOMETRY)
    config: dict[str, Any] = {
        "type": "graspgen",
        "kappa": kappa,
        "diffusion_steps": diffusion_steps,
        "grasp_batch_size": grasp_batch_size,
        "point_count": point_count,
        "geometry": geometry,
        "input_features": {
            "observation.object_points": {"type": "POINTCLOUD", "shape": [point_count, 3]},
        },
        "output_features": {
            "grasp.poses": {"type": "POSE", "shape": [grasp_batch_size, 4, 4]},
            "grasp.confidence": {"type": "CONFIDENCE", "shape": [grasp_batch_size]},
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    preprocessor = bundle_root / "policy_preprocessor.json"
    preprocessor.write_text(json.dumps({"name": "graspgen_preprocessor", "steps": []}, indent=2), encoding="utf-8")
    postprocessor = bundle_root / "policy_postprocessor.json"
    postprocessor.write_text(json.dumps({"name": "graspgen_postprocessor", "steps": []}, indent=2), encoding="utf-8")


def write_graspgen_ascend_deployment(
    bundle_root: Path,
    *,
    deployment_name: str,
    om_dir: Path,
    om_abi_dir: Path,
    soc_version: str,
    onnx_manifest: Mapping[str, object],
    grasp_batch_size: int,
    point_count: int,
    inspect_missing_abi: bool = True,
    abi_device_id: int = 0,
    acl_config_path: str | None = None,
) -> Path:
    _ensure_lerobot_config(
        bundle_root,
        onnx_manifest,
        point_count=point_count,
        grasp_batch_size=grasp_batch_size,
    )
    packaged_artifacts: dict[str, tuple[Path, str]] = {}
    role_bindings: dict[str, ArtifactBindings] = {}
    for role in GRASPGEN_EXECUTION:
        om_path = om_dir / f"{role}.om"
        if not om_path.is_file():
            raise FileNotFoundError(f"OM artifact for role {role!r} not found: {om_path}")
        abi = _abi_for_role(
            om_abi_dir,
            role,
            om_path=om_path,
            inspect_missing_abi=inspect_missing_abi,
            abi_device_id=abi_device_id,
            acl_config_path=acl_config_path,
        )
        packaged = package_deployment_artifact(
            bundle_root,
            om_path,
            backend="ascend",
            deployment_name=deployment_name,
            role=role,
            force_copy=True,
        )
        packaged_artifacts[role] = (packaged, "om")
        role_bindings[role] = _role_bindings(role, abi)
    deployment = compiled_deployment(
        bundle_root,
        backend="ascend",
        target_soc=soc_version,
        target_runtime="acl",
        artifacts=packaged_artifacts,
        execution=GRASPGEN_EXECUTION,
        bindings=role_bindings,
        device_links=(
            DeviceLink(
                semantic=_GENERATOR_EMBEDDING,
                producer="generator_encoder_head",
                consumer="denoiser",
                transport="device_pointer",
                owner="producer",
            ),
            DeviceLink(
                semantic=_DISCRIMINATOR_EMBEDDING,
                producer="discriminator_encoder_head",
                consumer="discriminator_head",
                transport="device_pointer",
                owner="producer",
            ),
        ),
    )
    validated = upsert_deployment(bundle_root, deployment_name, deployment)
    return validated.manifest_path


def _load_onnx_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX manifest not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("model_type") != "graspgen":
        raise ValueError(f"not a GraspGen ONNX manifest: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package compiled GraspGen OM artifacts into a unified inference_manifest.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-root",
        required=True,
        help="Target bundle directory; inference_manifest.json is written here.",
    )
    parser.add_argument(
        "--onnx-manifest",
        required=True,
        help="Path to graspgen.onnx.json emitted by export_onnx.py.",
    )
    parser.add_argument(
        "--om-dir",
        default=None,
        help="Directory containing the eight compiled OM files (defaults to bundle_root/artifacts/ascend/<deployment>).",
    )
    parser.add_argument(
        "--om-abi-dir",
        default=None,
        help="Directory containing or receiving <role>.om.abi.json runtime descriptors (defaults to --om-dir).",
    )
    parser.add_argument(
        "--inspect-missing-abi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inspect OMs with ACL when runtime ABI sidecars are missing.",
    )
    parser.add_argument("--abi-device-id", type=int, default=0, help="Ascend device used for OM ABI inspection.")
    parser.add_argument(
        "--acl-config-path", default=None, help="Optional ACL initialization config for ABI inspection."
    )
    parser.add_argument("--soc-version", required=True, help="Target Ascend SoC, e.g. Ascend310P3.")
    parser.add_argument("--deployment", default="ascend", help="Unified manifest deployment name.")
    parser.add_argument(
        "--grasp-batch-size",
        type=int,
        default=None,
        help="Override grasp batch size (defaults to backend_config.grasp_batch_size).",
    )
    parser.add_argument(
        "--point-count",
        type=int,
        default=None,
        help="Override input point cloud size (defaults to backend_config.point_count).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve(strict=True)
    onnx_manifest_path = Path(args.onnx_manifest).expanduser().resolve(strict=True)
    onnx_manifest = _load_onnx_manifest(onnx_manifest_path)
    backend_config = dict(onnx_manifest.get("backend_config") or {})
    grasp_batch_size = int(
        args.grasp_batch_size if args.grasp_batch_size is not None else backend_config.get("grasp_batch_size", 1000)
    )
    point_count = int(args.point_count if args.point_count is not None else backend_config.get("point_count", 2048))
    om_dir = (
        Path(args.om_dir).expanduser().resolve(strict=True)
        if args.om_dir
        else bundle_root / "artifacts" / "ascend" / args.deployment
    )
    om_abi_dir = Path(args.om_abi_dir).expanduser().resolve(strict=True) if args.om_abi_dir else om_dir
    manifest_path = write_graspgen_ascend_deployment(
        bundle_root,
        deployment_name=args.deployment,
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
        soc_version=args.soc_version,
        onnx_manifest=onnx_manifest,
        grasp_batch_size=grasp_batch_size,
        point_count=point_count,
        inspect_missing_abi=args.inspect_missing_abi,
        abi_device_id=args.abi_device_id,
        acl_config_path=args.acl_config_path,
    )
    print(f"GraspGen ascend deployment written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
