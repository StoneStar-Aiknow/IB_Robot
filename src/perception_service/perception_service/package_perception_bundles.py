"""Finalize downloaded perception assets as schema-v2 inference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    BundleFile,
    Digest,
    EmbeddingMetadata,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    SemanticIdentity,
    SemanticTensor,
    TorchDeployment,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)

from .ram_plus_adapter import RAM_PLUS_POSTPROCESSING, RAM_PLUS_PREPROCESSING
from .semantic_model_adapters import GroundingDINOAdapter, SAM2Adapter, SigLIP2ImageAdapter


@dataclass(frozen=True)
class BundleSpec:
    family: str
    name: str
    adapter: dict[str, object]
    model: ModelDescriptor
    required_paths: tuple[str, ...]


def _tensor(semantic: str, dtype: str, shape: tuple[int, ...], layout: str | None = None) -> SemanticTensor:
    return SemanticTensor(semantic=semantic, dtype=dtype, shape=shape, layout=layout)


def _identity(family: str, preprocessing: str, output_semantics: str, embedding=None) -> SemanticIdentity:
    return SemanticIdentity(
        logical_model_revision=family,
        preprocessing_contract=preprocessing,
        output_semantics=output_semantics,
        embedding=embedding,
    )


def _specs() -> dict[str, BundleSpec]:
    siglip_embedding = EmbeddingMetadata(
        embedding_space_id="google/siglip2-so400m-patch14-384@main",
        dimension=1152,
        normalization="l2",
        image_preprocessing="masked-crop-gray127-resize384-bilinear-normalize0.5-v1",
        text_preprocessing="photo-template-gemma-tokenizer-max64-v1",
    )
    return {
        "sam2": BundleSpec(
            family="sam2",
            name="sam2.1_hiera_tiny",
            adapter={
                "family": "sam2",
                "preprocessing": SAM2Adapter.identity.preprocessing,
                "postprocessing": SAM2Adapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_sam2",
                "checkpoint": "assets/sam2.1_hiera_tiny.pt",
                "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "points_per_batch": 64,
            },
            model=ModelDescriptor(
                kind="perception",
                family="sam2",
                inputs=(_tensor("observation.image", "uint8", (-1, -1, 3)),),
                outputs=(
                    _tensor("masks", "uint8", (-1, -1, -1)),
                    _tensor("boxes", "float32", (-1, 4)),
                    _tensor("scores", "float32", (-1,)),
                    _tensor("stability_scores", "float32", (-1,)),
                ),
                semantic_identity=_identity(
                    "sam2.1-hiera-tiny@092824",
                    SAM2Adapter.identity.preprocessing,
                    SAM2Adapter.identity.postprocessing,
                ),
            ),
            required_paths=("assets/sam2.1_hiera_tiny.pt",),
        ),
        "ram_plus": BundleSpec(
            family="ram_plus",
            name="ram_plus_swin_large_14m",
            adapter={
                "family": "ram_plus",
                "preprocessing": RAM_PLUS_PREPROCESSING,
                "postprocessing": RAM_PLUS_POSTPROCESSING,
                "torch_module_loader": "perception_service.torch_model_loaders:load_ram_plus",
                "checkpoint": "assets/ram_plus_swin_large_14m.pth",
                "text_encoder": "assets/bert-base-uncased",
            },
            model=ModelDescriptor(
                kind="perception",
                family="ram_plus",
                inputs=(_tensor("observation.image", "float32", (1, 3, 384, 384), "NCHW"),),
                outputs=(_tensor("tag_logits", "float32", (1, 4585)),),
                semantic_identity=_identity(
                    "ram-plus-swin-large-14m@v1", RAM_PLUS_PREPROCESSING, RAM_PLUS_POSTPROCESSING
                ),
            ),
            required_paths=(
                "assets/ram_plus_swin_large_14m.pth",
                "assets/bert-base-uncased/config.json",
                "assets/ram_tag_list.txt",
                "assets/ram_tag_list_threshold.txt",
            ),
        ),
        "siglip2": BundleSpec(
            family="siglip2",
            name="siglip2_so400m_patch14_384",
            adapter={
                "family": "siglip2",
                "preprocessing": SigLIP2ImageAdapter.identity.preprocessing,
                "postprocessing": SigLIP2ImageAdapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_siglip2",
                "model_path": "assets/model",
            },
            model=ModelDescriptor(
                kind="perception",
                family="siglip2",
                inputs=(
                    _tensor("masked_images", "float32", (-1, 3, 384, 384), "NCHW"),
                    _tensor("text_tokens", "int64", (-1, 64)),
                    _tensor("text_attention_mask", "int64", (-1, 64)),
                ),
                outputs=(
                    _tensor("image_embeddings", "float32", (-1, 1152)),
                    _tensor("text_embeddings", "float32", (-1, 1152)),
                ),
                semantic_identity=_identity(
                    "google/siglip2-so400m-patch14-384@main",
                    SigLIP2ImageAdapter.identity.preprocessing,
                    SigLIP2ImageAdapter.identity.postprocessing,
                    siglip_embedding,
                ),
            ),
            required_paths=("assets/model/config.json", "assets/model/model.safetensors"),
        ),
        "grounded_sam2": BundleSpec(
            family="grounding_dino",
            name="grounded_sam2_swint_ogc",
            adapter={
                "family": "grounding_dino",
                "preprocessing": GroundingDINOAdapter.identity.preprocessing,
                "postprocessing": GroundingDINOAdapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_grounded_sam2",
                "gdino_checkpoint": "assets/groundingdino_swint_ogc.pth",
                "text_encoder": "assets/bert-base-uncased",
                "sam_checkpoint": "assets/sam2.1_hiera_tiny.pt",
                "sam_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            },
            model=ModelDescriptor(
                kind="perception",
                family="grounding_dino",
                inputs=(
                    _tensor("observation.image", "uint8", (-1, -1, 3)),
                    _tensor("text_prompt", "uint8", (-1,)),
                    _tensor("box_threshold", "float32", (1,)),
                    _tensor("text_threshold", "float32", (1,)),
                ),
                outputs=(
                    _tensor("boxes", "float32", (-1, 4)),
                    _tensor("scores", "float32", (-1,)),
                    _tensor("masks", "uint8", (-1, -1, -1)),
                    _tensor("label_indices", "int32", (-1,)),
                ),
                semantic_identity=_identity(
                    "groundingdino-swint-ogc+sam2.1-hiera-tiny@v1",
                    GroundingDINOAdapter.identity.preprocessing,
                    GroundingDINOAdapter.identity.postprocessing,
                ),
            ),
            required_paths=(
                "assets/groundingdino_swint_ogc.pth",
                "assets/bert-base-uncased/config.json",
                "assets/sam2.1_hiera_tiny.pt",
            ),
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(root))
            for path in (root / "assets").rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )


def package_bundle(root: Path, spec: BundleSpec) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if spec.family == "grounding_dino":
        (root / "assets" / "GroundingDINO_SwinT_OGC.py").unlink(missing_ok=True)
    for relative in spec.required_paths:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"required {spec.family} bundle asset is missing: {root / relative}")

    adapter_path = root / "assets" / "adapter.json"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(json.dumps(spec.adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    digest_path = root / "assets" / "artifact-digests.json"
    asset_paths = tuple(path for path in _asset_paths(root) if path != "assets/artifact-digests.json")
    digests = {path: _sha256(root / path) for path in asset_paths}
    previous_digests = digest_path.read_text(encoding="utf-8") if digest_path.is_file() else ""
    digest_document = json.dumps(digests, indent=2, sort_keys=True) + "\n"
    digest_path.write_text(digest_document, encoding="utf-8")

    manifest_path = root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        existing = load_inference_manifest(
            root, next(iter(json.loads(manifest_path.read_text())["deployments"]))
        ).manifest
    files = tuple(BundleFile(path=path) for path in _asset_paths(root))
    structure_changed = existing is not None and (
        existing.bundle.name != spec.name
        or existing.bundle.files != files
        or existing.model != spec.model
        or previous_digests != digest_document
    )
    bundle_uuid = existing.bundle.uuid if existing is not None else str(uuid4())
    bundle_revision = existing.bundle.revision + int(structure_changed) if existing is not None else 1
    deployments = dict(existing.deployments) if existing is not None else {}
    for name, device in (("torch_cpu", "cpu"), ("torch_cuda", "cuda")):
        candidate = TorchDeployment(backend="torch", device=device)
        previous = deployments.get(name)
        if previous is not None:
            candidate = candidate.model_copy(update={"uuid": previous.uuid, "revision": previous.revision})
        deployments[name] = candidate
    manifest = InferenceManifest(
        schema_version=2,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=spec.name,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, spec.name, files),
            ),
        ),
        model=spec.model,
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for deployment in sorted(deployments):
        load_inference_manifest(root, deployment)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(_specs()) + ("all",), default="all")
    args = parser.parse_args()
    specs = _specs()
    selected = specs if args.family == "all" else {args.family: specs[args.family]}
    for spec in selected.values():
        print(package_bundle(args.models_root / spec.name, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
