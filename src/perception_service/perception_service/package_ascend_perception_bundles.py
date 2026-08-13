"""Package audited Ascend perception OM candidates as schema-v2 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ArtifactBindings,
    BundleFile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    DeviceLink,
    Digest,
    EmbeddingMetadata,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    SemanticIdentity,
    SemanticTensor,
    TensorBinding,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    source: str
    destination: str
    bindings: ArtifactBindings


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    soc: str
    artifacts: tuple[ArtifactSpec, ...]
    device_links: tuple[DeviceLink, ...] = ()


@dataclass(frozen=True)
class BundleSpec:
    name: str
    family: str
    inputs: tuple[SemanticTensor, ...]
    outputs: tuple[SemanticTensor, ...]
    preprocessing: str
    postprocessing: str
    deployments: tuple[DeploymentSpec, ...]
    assets: tuple[tuple[str, str], ...] = ()
    operation: str = ""
    embedding: EmbeddingMetadata | None = None


def _semantic(semantic: str, dtype: str, shape: tuple[int, ...], layout: str | None = None) -> SemanticTensor:
    return SemanticTensor(semantic=semantic, dtype=dtype, shape=shape, layout=layout)


def _binding(
    semantic: str,
    index: int,
    dtype: str,
    shape: tuple[int, ...],
    layout: str | None = None,
    runtime_name: str | None = None,
) -> TensorBinding:
    return TensorBinding(
        semantic=semantic,
        runtime_name=runtime_name,
        index=index,
        dtype=dtype,
        shape=shape,
        layout=layout,
    )


def _bindings(inputs: tuple[TensorBinding, ...], outputs: tuple[TensorBinding, ...]) -> ArtifactBindings:
    return ArtifactBindings(inputs=inputs, outputs=outputs)


def _artifact(role: str, source: str, bindings: ArtifactBindings) -> ArtifactSpec:
    target = source.split("/")[-2].removesuffix("_verified")
    return ArtifactSpec(role, source, f"artifacts/{target}/{Path(source).name}", bindings)


def _link(semantic: str, producer: str, consumer: str) -> DeviceLink:
    return DeviceLink(
        semantic=semantic,
        producer=producer,
        consumer=consumer,
        transport="device_pointer",
        owner="producer",
    )


def _sam2_deployment(name: str, soc: str, candidate: str, decoder_name: str, batch: int) -> DeploymentSpec:
    encoder = _artifact(
        "encoder",
        f"sam2.1_hiera_tiny/model_utils_work/candidates/{candidate}/sam2_encoder.om",
        _bindings(
            (_binding("image", 0, "float32", (1, 3, 1024, 1024), "NCHW", "image"),),
            (
                _binding(
                    "internal.high_res_feats_0",
                    0,
                    "float32",
                    (1, 32, 256, 256),
                    "NCHW",
                    "/Reshape_5:0:high_res_feats_0",
                ),
                _binding(
                    "internal.high_res_feats_1",
                    1,
                    "float32",
                    (1, 64, 128, 128),
                    "NCHW",
                    "/Reshape_4:0:high_res_feats_1",
                ),
                _binding(
                    "internal.image_embed",
                    2,
                    "float32",
                    (1, 256, 64, 64),
                    "NCHW",
                    "/Reshape_3:0:image_embed",
                ),
            ),
        ),
    )
    decoder = _artifact(
        "decoder",
        f"sam2.1_hiera_tiny/model_utils_work/candidates/{candidate}/{decoder_name}",
        _bindings(
            (
                _binding("internal.image_embed", 0, "float32", (1, 256, 64, 64), "NCHW"),
                _binding("internal.high_res_feats_0", 1, "float32", (1, 32, 256, 256), "NCHW"),
                _binding("internal.high_res_feats_1", 2, "float32", (1, 64, 128, 128), "NCHW"),
                _binding("point_coords", 3, "float32", (batch, 2, 2), runtime_name="point_coords"),
                _binding("point_labels", 4, "int8", (batch, 2), runtime_name="point_labels"),
                _binding("mask_input", 5, "float32", (batch, 1, 256, 256), "NCHW", "mask_input"),
                _binding("has_mask_input", 6, "int8", (batch,), runtime_name="has_mask_input"),
            ),
            (
                _binding("mask_logits", 0, "float32", (batch, 1, 256, 256), "NCHW", "/Where_8:0:masks"),
                _binding("iou_predictions", 1, "float32", (batch, 1), runtime_name="/Where_9:0:iou_predictions"),
                _binding(
                    "low_res_masks",
                    2,
                    "float32",
                    (batch, 1, 256, 256),
                    "NCHW",
                    "/Clip:0:low_res_masks",
                ),
            ),
        ),
    )
    return DeploymentSpec(
        name,
        soc,
        (encoder, decoder),
        (
            _link("internal.image_embed", "encoder", "decoder"),
            _link("internal.high_res_feats_0", "encoder", "decoder"),
            _link("internal.high_res_feats_1", "encoder", "decoder"),
        ),
    )


def _siglip_deployment(name: str, soc: str, candidate: str, text_name: str, vision_name: str, batch: int):
    vision = _artifact(
        "vision",
        f"siglip2_so400m_patch14_384/model_utils_work/candidates/{candidate}/{vision_name}",
        _bindings(
            (_binding("host.siglip2.image", 0, "float32", (1, 3, 384, 384), "NCHW", "image"),),
            (
                _binding("internal.image_tokens", 0, "float32", (1, 729, 1152)),
                _binding("host.siglip2.image_embedding", 1, "float32", (1, 1152)),
            ),
        ),
    )
    text = _artifact(
        "text",
        f"siglip2_so400m_patch14_384/model_utils_work/candidates/{candidate}/{text_name}",
        _bindings(
            (_binding("host.siglip2.input_ids", 0, "int64", (batch, 64), runtime_name="input_ids"),),
            (
                _binding("internal.text_tokens", 0, "float32", (batch, 64, 1152)),
                _binding("host.siglip2.text_embeddings", 1, "float32", (batch, 1152)),
            ),
        ),
    )
    return DeploymentSpec(name, soc, (vision, text))


def _grounding_dino_deployment() -> DeploymentSpec:
    root = "grounded_sam2_swint_ogc/model_utils_work/candidates/ascend_310p"
    text_mask = _binding("text_token_mask", 2, "int64", (1, 8), runtime_name="text_token_mask")
    position_ids = _binding("position_ids", 3, "int64", (1, 8), runtime_name="position_ids")
    attention = _binding("text_self_attention_masks", 4, "int64", (1, 8, 8), runtime_name="text_self_attention_masks")
    artifacts = [
        _artifact(
            "text",
            f"{root}/gdino_text.om",
            _bindings(
                (
                    _binding("input_ids", 0, "int64", (1, 8), runtime_name="input_ids"),
                    _binding("text_self_attention_masks", 1, "int64", (1, 8, 8), runtime_name="attention_mask"),
                    _binding("token_type_ids", 2, "int64", (1, 8), runtime_name="token_type_ids"),
                    _binding("position_ids", 3, "int64", (1, 8), runtime_name="position_ids"),
                ),
                (_binding("internal.text_0", 0, "float32", (1, 8, 256), runtime_name="/feat_map/Add:0:encoded_text"),),
            ),
        ),
        _artifact(
            "vision",
            f"{root}/gdino_vision.om",
            _bindings(
                (_binding("image", 0, "float32", (1, 3, 720, 1280), "NCHW", "image"),),
                tuple(
                    _binding(
                        f"internal.src{index}",
                        index,
                        "float32",
                        shape,
                        "NCHW",
                        f"src{index}",
                    )
                    for index, shape in enumerate(
                        ((1, 256, 90, 160), (1, 256, 45, 80), (1, 256, 23, 40), (1, 256, 12, 20))
                    )
                ),
            ),
        ),
        _artifact(
            "flatten",
            f"{root}/gdino_vision_flatten.om",
            _bindings(
                tuple(
                    _binding(f"internal.src{index}", index, "float32", shape, "NCHW", f"src{index}")
                    for index, shape in enumerate(
                        ((1, 256, 90, 160), (1, 256, 45, 80), (1, 256, 23, 40), (1, 256, 12, 20))
                    )
                ),
                (_binding("internal.visual_0", 0, "float32", (1, 19160, 256), runtime_name="visual"),),
            ),
        ),
    ]
    for index in range(6):
        artifacts.append(
            _artifact(
                f"encoder_{index}",
                f"{root}/gdino_encoder_layer{index}_batch2_fp16.om",
                _bindings(
                    (
                        _binding(f"internal.visual_{index}", 0, "float32", (1, 19160, 256), runtime_name="visual"),
                        _binding(f"internal.text_{index}", 1, "float32", (1, 8, 256), runtime_name="memory_text"),
                        text_mask,
                        position_ids,
                        attention,
                    ),
                    (
                        _binding(
                            f"internal.visual_{index + 1}",
                            0,
                            "float32",
                            (1, 19160, 256),
                            runtime_name=("PartitionedCall_/visual/norm2/LayerNormalization_LayerNorm_96:0:visual_out"),
                        ),
                        _binding(
                            f"internal.text_{index + 1}",
                            1,
                            "float32",
                            (1, 8, 256),
                            runtime_name="PartitionedCall_/Transpose_2_Transpose_155:0:memory_text_out",
                        ),
                    ),
                ),
            )
        )
    artifacts.extend(
        (
            _artifact(
                "proposal",
                f"{root}/gdino_proposal_cube.om",
                _bindings(
                    (
                        _binding("internal.visual_6", 0, "float32", (1, 19160, 256), runtime_name="memory"),
                        _binding("internal.text_6", 1, "float32", (1, 8, 256), runtime_name="memory_text"),
                        _binding("text_token_mask", 2, "int64", (1, 8), runtime_name="text_token_mask"),
                    ),
                    (
                        _binding(
                            "internal.refpoint", 0, "float32", (1, 900, 4), runtime_name="/GatherElements:0:refpoint"
                        ),
                    ),
                ),
            ),
            _artifact(
                "decoder",
                f"{root}/gdino_decoder_cube.om",
                _bindings(
                    (
                        _binding("encoder_tgt", 0, "float32", (1, 900, 256), runtime_name="tgt"),
                        _binding("internal.refpoint", 1, "float32", (1, 900, 4), runtime_name="refpoint"),
                        _binding("internal.visual_6", 2, "float32", (1, 19160, 256), runtime_name="memory"),
                        _binding("internal.text_6", 3, "float32", (1, 8, 256), runtime_name="memory_text"),
                        _binding("text_token_mask", 4, "int64", (1, 8), runtime_name="text_token_mask"),
                    ),
                    (
                        _binding("internal.decoder_hidden", 0, "float32", (1, 900, 256)),
                        _binding("internal.decoder_reference", 1, "float32", (1, 900, 4)),
                        _binding("internal.decoder_text", 2, "float32", (1, 8, 256)),
                    ),
                ),
            ),
            _artifact(
                "head",
                f"{root}/gdino_head_origin.om",
                _bindings(
                    (
                        _binding("internal.decoder_hidden", 0, "float32", (1, 900, 256)),
                        _binding("internal.decoder_reference", 1, "float32", (1, 900, 4)),
                        _binding("internal.decoder_text", 2, "float32", (1, 8, 256)),
                        _binding("text_token_mask", 3, "int64", (1, 8), runtime_name="text_token_mask"),
                    ),
                    (
                        _binding(
                            "pred_logits",
                            0,
                            "float32",
                            (1, 900, 256),
                            runtime_name="/class_embed/ScatterND:0:pred_logits",
                        ),
                        _binding("pred_boxes", 1, "float32", (1, 900, 4), runtime_name="/Sigmoid:0:pred_boxes"),
                    ),
                ),
            ),
        )
    )
    links = [_link(f"internal.src{index}", "vision", "flatten") for index in range(4)]
    links.extend((_link("internal.visual_0", "flatten", "encoder_0"), _link("internal.text_0", "text", "encoder_0")))
    for index in range(5):
        links.extend(
            (
                _link(f"internal.visual_{index + 1}", f"encoder_{index}", f"encoder_{index + 1}"),
                _link(f"internal.text_{index + 1}", f"encoder_{index}", f"encoder_{index + 1}"),
            )
        )
    links.extend(
        (
            _link("internal.visual_6", "encoder_5", "proposal"),
            _link("internal.text_6", "encoder_5", "proposal"),
            _link("internal.visual_6", "encoder_5", "decoder"),
            _link("internal.text_6", "encoder_5", "decoder"),
            _link("internal.refpoint", "proposal", "decoder"),
            _link("internal.decoder_hidden", "decoder", "head"),
            _link("internal.decoder_reference", "decoder", "head"),
            _link("internal.decoder_text", "decoder", "head"),
        )
    )
    return DeploymentSpec("ascend_310p", "Ascend310P1", tuple(artifacts), tuple(links))


def _specs() -> dict[str, BundleSpec]:
    return {
        "sam2": BundleSpec(
            name="sam2_hiera_tiny_ascend",
            family="sam2",
            operation="prompt",
            inputs=(
                _semantic("image", "float32", (1, 3, 1024, 1024), "NCHW"),
                _semantic("point_coords", "float32", (-1, 2, 2)),
                _semantic("point_labels", "int8", (-1, 2)),
                _semantic("mask_input", "float32", (-1, 1, 256, 256), "NCHW"),
                _semantic("has_mask_input", "int8", (-1,)),
            ),
            outputs=(
                _semantic("mask_logits", "float32", (-1, 1, 256, 256), "NCHW"),
                _semantic("iou_predictions", "float32", (-1, 1)),
                _semantic("low_res_masks", "float32", (-1, 1, 256, 256), "NCHW"),
            ),
            preprocessing="sam2-longest-side1024-imagenet-box-prompt-v1",
            postprocessing="sam2-mask-logits-iou-v1",
            deployments=(
                _sam2_deployment("ascend_310p", "Ascend310P1", "ascend_310p", "sam2_decoder_bs4.om", 4),
                _sam2_deployment("ascend_310b", "Ascend310B1", "ascend_310b", "sam2_decoder.om", 1),
            ),
        ),
        "siglip2": BundleSpec(
            name="siglip2_so400m_patch14_384_ascend",
            family="siglip2",
            inputs=(
                _semantic("masked_images", "float32", (-1, 3, 384, 384), "NCHW"),
                _semantic("text_tokens", "int64", (-1, 64)),
                _semantic("text_attention_mask", "int64", (-1, 64)),
            ),
            outputs=(
                _semantic("image_embeddings", "float32", (-1, 1152)),
                _semantic("text_embeddings", "float32", (-1, 1152)),
            ),
            preprocessing="siglip2-dual-encoder-v2",
            postprocessing="normalized-embedding-v1",
            embedding=EmbeddingMetadata(
                embedding_space_id="google/siglip2-so400m-patch14-384@main",
                dimension=1152,
                normalization="l2",
                image_preprocessing="masked-crop-gray127-resize384-bilinear-normalize0.5-v1",
                text_preprocessing="photo-template-gemma-tokenizer-max64-v1",
            ),
            deployments=(
                _siglip_deployment(
                    "ascend_310p",
                    "Ascend310P1",
                    "ascend_310p_verified",
                    "siglip2_text_310p.om",
                    "siglip2_vision_310p_linux_aarch64.om",
                    8,
                ),
                _siglip_deployment(
                    "ascend_310b",
                    "Ascend310B1",
                    "ascend_310b",
                    "siglip2_text_64.om",
                    "siglip2_vision.om",
                    5,
                ),
            ),
            assets=(
                ("siglip2_so400m_patch14_384/assets/model/tokenizer.json", "assets/model/tokenizer.json"),
                (
                    "siglip2_so400m_patch14_384/assets/model/tokenizer_config.json",
                    "assets/model/tokenizer_config.json",
                ),
                (
                    "siglip2_so400m_patch14_384/assets/model/special_tokens_map.json",
                    "assets/model/special_tokens_map.json",
                ),
                ("siglip2_so400m_patch14_384/assets/model/config.json", "assets/model/config.json"),
            ),
        ),
        "grounding_dino": BundleSpec(
            name="grounding_dino_swint_seq8_1280x720_ascend",
            family="grounding_dino",
            operation="raw",
            inputs=(
                _semantic("image", "float32", (1, 3, 720, 1280), "NCHW"),
                _semantic("input_ids", "int64", (1, 8)),
                _semantic("token_type_ids", "int64", (1, 8)),
                _semantic("position_ids", "int64", (1, 8)),
                _semantic("text_self_attention_masks", "int64", (1, 8, 8)),
                _semantic("text_token_mask", "int64", (1, 8)),
                _semantic("encoder_tgt", "float32", (1, 900, 256)),
            ),
            outputs=(
                _semantic("pred_logits", "float32", (1, 900, 256)),
                _semantic("pred_boxes", "float32", (1, 900, 4)),
            ),
            preprocessing="grounding-dino-swint-rgb720x1280-bert-seq8-v1",
            postprocessing="grounding-dino-raw-logits-cxcywh-v1",
            deployments=(_grounding_dino_deployment(),),
            assets=(
                (
                    "grounded_sam2_swint_ogc/model_utils_work/candidates/ascend_310p/encoder_tgt.npy",
                    "assets/encoder_tgt.npy",
                ),
                (
                    "grounded_sam2_swint_ogc/assets/bert-base-uncased/vocab.txt",
                    "assets/bert-base-uncased/vocab.txt",
                ),
            ),
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required Ascend bundle source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(source) != _sha256(destination):
        shutil.copy2(source, destination)


def package_bundle(models_root: Path, spec: BundleSpec) -> Path:
    root = models_root / spec.name
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "inference_manifest.json"
    digest_path = root / "assets/artifact-digests.json"
    previous_digests = json.loads(digest_path.read_text(encoding="utf-8")) if digest_path.is_file() else {}
    existing = None
    if manifest_path.is_file():
        name = next(iter(json.loads(manifest_path.read_text(encoding="utf-8"))["deployments"]))
        existing = load_inference_manifest(root, name).manifest
    adapter = root / "assets/adapter.json"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter_identity = {
        "family": spec.family,
        "preprocessing": spec.preprocessing,
        "postprocessing": spec.postprocessing,
    }
    if spec.operation:
        adapter_identity["operation"] = spec.operation
    adapter.write_text(
        json.dumps(
            adapter_identity,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for source, destination in spec.assets:
        _copy(models_root / source, root / destination)

    available = []
    for deployment in spec.deployments:
        if all((models_root / artifact.source).is_file() for artifact in deployment.artifacts):
            for artifact in deployment.artifacts:
                _copy(models_root / artifact.source, root / artifact.destination)
            available.append(deployment)
    if not available:
        raise FileNotFoundError(f"no complete Ascend deployment candidates are available for {spec.name}")

    digests = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"artifact-digests.json", "inference_manifest.json"}
    }
    digest_path.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = tuple(
        BundleFile(path=str(path.relative_to(root))) for path in sorted((root / "assets").rglob("*")) if path.is_file()
    )

    model = ModelDescriptor(
        kind="perception",
        family=spec.family,
        operation=spec.operation,
        inputs=spec.inputs,
        outputs=spec.outputs,
        semantic_identity=SemanticIdentity(
            logical_model_revision=f"{spec.family}@v1",
            preprocessing_contract=spec.preprocessing,
            output_semantics=spec.postprocessing,
            embedding=spec.embedding,
        ),
    )
    structure_changed = existing is not None and (
        existing.bundle.name != spec.name
        or existing.bundle.files != files
        or existing.model != model
        or previous_digests != digests
    )
    bundle_uuid = existing.bundle.uuid if existing else str(uuid4())
    bundle_revision = existing.bundle.revision + int(structure_changed) if existing else 1
    deployments = {}
    for deployment in available:
        previous = existing.deployments.get(deployment.name) if existing else None
        value = CompiledDeployment(
            backend="ascend",
            target=DeploymentTarget(soc=deployment.soc, runtime="acl"),
            artifacts={
                artifact.role: DeploymentArtifact(path=artifact.destination, format="om")
                for artifact in deployment.artifacts
            },
            execution=tuple(artifact.role for artifact in deployment.artifacts),
            bindings={artifact.role: artifact.bindings for artifact in deployment.artifacts},
            device_links=deployment.device_links,
        )
        if isinstance(previous, CompiledDeployment):
            artifact_changed = any(
                previous_digests.get(artifact.destination) != digests[artifact.destination]
                for artifact in deployment.artifacts
            )
            contract_changed = previous.model_dump(exclude={"uuid", "revision"}) != value.model_dump(
                exclude={"uuid", "revision"}
            )
            value = value.model_copy(
                update={
                    "uuid": previous.uuid,
                    "revision": previous.revision + int(artifact_changed or contract_changed),
                }
            )
        deployments[deployment.name] = value

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
        model=model,
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for deployment in deployments:
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
        print(package_bundle(args.models_root, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
