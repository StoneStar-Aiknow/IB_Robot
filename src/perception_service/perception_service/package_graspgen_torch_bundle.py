"""Write the schema-v3 bundle used by the CUDA GraspGen executor.

GraspGen is stored in the grasp model domain. The generic model runtime represents
its executable contract with the canonical tensor-model identity; that identity does
not determine the on-disk domain directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

import yaml

from inference_manifest import (
    BundleFile,
    DeploymentTarget,
    Digest,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    TorchDeployment,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)
from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_NPOINTS,
    GRASPGEN_NSAMPLES,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
    GRASPGEN_RADII,
)

from .graspgen_adapter import GRASPGEN_POSTPROCESSING, GRASPGEN_PREPROCESSING

_SOURCE_TO_ASSET = (
    ("checkpoints/graspgen_robotiq_2f_140.yml", "assets/graspgen_config.yml"),
    ("checkpoints/graspgen_robotiq_2f_140_gen.pth", "assets/generator_checkpoint.pth"),
    ("checkpoints/graspgen_robotiq_2f_140_dis.pth", "assets/discriminator_checkpoint.pth"),
)
_REQUIRED = tuple(asset for _source, asset in _SOURCE_TO_ASSET)

_ADAPTER_ASSET = "assets/adapter.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_descriptor(point_count: int, grasp_batch_size: int) -> ModelDescriptor:
    del point_count, grasp_batch_size
    return ModelDescriptor(
        interface="tensor_model",
        model_type="graspgen",
        operation="generate_grasps",
        inputs=(SemanticTensor(semantic=GRASPGEN_POINT_CLOUD_SEMANTIC, dtype="float32", shape=(-1, 3)),),
        outputs=(
            SemanticTensor(semantic=GRASPGEN_POSE_SEMANTIC, dtype="float32", shape=(-1, 4, 4)),
            SemanticTensor(semantic=GRASPGEN_CONFIDENCE_SEMANTIC, dtype="float32", shape=(-1,)),
        ),
        semantic_identity=SemanticIdentity(
            logical_model_revision=f"graspgen@v{GRASPGEN_CONTRACT_VERSION}",
            preprocessing_contract=GRASPGEN_PREPROCESSING,
            output_semantics=GRASPGEN_POSTPROCESSING,
        ),
    )


def _adapter_assets(
    bundle_root: Path,
    config: dict[str, object],
    *,
    point_count: int,
    grasp_batch_size: int,
) -> dict[str, object]:
    """Describe the Torch callable using the same adapter contract as Ascend bundles."""
    diffusion = config.get("diffusion", {})
    if not isinstance(diffusion, dict):
        raise ValueError("GraspGen config diffusion must be a mapping")
    return {
        "interface": "tensor_model",
        "model_type": "graspgen",
        "operation": "generate_grasps",
        "preprocessing": GRASPGEN_PREPROCESSING,
        "postprocessing": GRASPGEN_POSTPROCESSING,
        "kappa": float(diffusion.get("kappa", 2.02217)),
        "diffusion_steps": int(diffusion.get("num_diffusion_iters_eval", 10)),
        "grasp_batch_size": grasp_batch_size,
        "point_count": point_count,
        "geometry": {
            "npoints": list(GRASPGEN_NPOINTS),
            "radii": list(GRASPGEN_RADII),
            "nsamples": list(GRASPGEN_NSAMPLES),
        },
        "torch_module_loader": "perception_service.torch_model_loaders:load_graspgen",
        "gripper_config": _REQUIRED[0],
        "generator_checkpoint": _REQUIRED[1],
        "discriminator_checkpoint": _REQUIRED[2],
        "source_sha256": {relative: _sha256(bundle_root / relative) for relative in _REQUIRED},
    }


def _materialize_assets(bundle_root: Path, source_root: Path | None) -> None:
    source = source_root
    if source is None and all((bundle_root / relative).is_file() for relative, _asset in _SOURCE_TO_ASSET):
        source = bundle_root
    if source is not None:
        missing = [relative for relative, _asset in _SOURCE_TO_ASSET if not (source / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"GraspGen CUDA source is missing required files: {missing}")
        for relative, asset in _SOURCE_TO_ASSET:
            source_path = source / relative
            asset_path = bundle_root / asset
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            if not asset_path.is_file() or _sha256(source_path) != _sha256(asset_path):
                shutil.copy2(source_path, asset_path)

    missing = [relative for relative in _REQUIRED if not (bundle_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "GraspGen CUDA bundle is missing standard assets: "
            f"{missing}; provide --source-root containing the checkpoints directory"
        )


def package_graspgen_torch_bundle(
    bundle_root: Path,
    *,
    deployment_name: str = "torch_cuda",
    source_root: Path | None = None,
) -> Path:
    """Create/update a standard bundle, optionally importing legacy checkpoint sources."""
    bundle_root.mkdir(parents=True, exist_ok=True)
    _materialize_assets(bundle_root, source_root)
    config = yaml.safe_load((bundle_root / _REQUIRED[0]).read_text(encoding="utf-8")) or {}
    data = config.get("data", {})
    diffusion = config.get("diffusion", {})
    if not isinstance(data, dict) or not isinstance(diffusion, dict):
        raise ValueError("GraspGen config data and diffusion must be mappings")
    point_count = int(data.get("num_points", 2048))
    batch_size = int(data.get("num_grasps_per_object", diffusion.get("num_grasps_per_object", 500)))
    if point_count <= 0 or batch_size <= 0:
        raise ValueError("GraspGen config point count and grasp batch size must be positive")

    manifest_path = bundle_root / "inference_manifest.json"
    existing = None
    previous_adapter = ""
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = load_inference_manifest(bundle_root, next(iter(raw["deployments"]))).manifest
    files = tuple(BundleFile(path=relative) for relative in _REQUIRED)
    # Include provenance so the bundle records the selected checkpoint identity.
    adapter_path = bundle_root / _ADAPTER_ASSET
    if adapter_path.is_file():
        previous_adapter = adapter_path.read_text(encoding="utf-8")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_document = (
        json.dumps(
            _adapter_assets(bundle_root, config, point_count=point_count, grasp_batch_size=batch_size),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    adapter_path.write_text(adapter_document, encoding="utf-8")
    files = (*files, BundleFile(path=_ADAPTER_ASSET))
    bundle_uuid = existing.bundle.uuid if existing is not None else str(uuid4())
    previous_revision = existing.bundle.revision if existing is not None else 0
    changed = existing is not None and (
        existing.bundle.files != files
        or existing.model != _model_descriptor(point_count, batch_size)
        or previous_adapter != adapter_document
    )
    bundle_revision = previous_revision + int(changed) if existing is not None else 1
    deployment = TorchDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device="cuda"),
        ),
    )
    previous_deployment = existing.deployments.get(deployment_name) if existing is not None else None
    if isinstance(previous_deployment, TorchDeployment):
        deployment = deployment.model_copy(
            update={"uuid": previous_deployment.uuid, "revision": previous_deployment.revision}
        )
    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=bundle_root.name,
            files=tuple(files),
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_root.name, files),
            ),
        ),
        model=_model_descriptor(point_count, batch_size),
        deployments={**(existing.deployments if existing is not None else {}), deployment_name: deployment},
    )
    write_inference_manifest(manifest_path, manifest)
    load_inference_manifest(bundle_root, deployment_name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional source directory containing checkpoints/; assets are copied into the bundle.",
    )
    parser.add_argument("--deployment", default="torch_cuda")
    args = parser.parse_args()
    print(
        package_graspgen_torch_bundle(
            args.bundle_root,
            deployment_name=args.deployment,
            source_root=args.source_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
