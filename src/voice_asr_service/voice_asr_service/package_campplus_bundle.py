"""Write the schema-v3 bundle for the speaker-embedding (voiceprint) model.

The addressee gate compares one utterance's voiceprint with the anchor's, so the
model that produced them is part of the comparison: vectors from two models are
not comparable at all, and a swap that keeps the dimension is invisible in the
numbers. Packaging the model as a bundle is what makes that identity checkable --
``model_id`` is derived from the manifest's semantic identity rather than a file
name (see ``speaker_id.embedder.voiceprint_model_id``), and the artifact digest
is recorded where the downloader checks it, the same as for the
ASR, Silero VAD and FullSubNet bundles this file mirrors. Loading does not
re-hash the artifact (``declared_only``, as for every other model here), so the
digest guards provisioning rather than later tampering.

The model is 3D-Speaker's CAM++ (``speech_campplus_sv_zh-cn_16k-common``,
Apache-2.0) as exported by sherpa-onnx. It consumes 80-bin Kaldi log-mel
features, not PCM, so the host owns feature extraction; see
``voice_asr_service/speaker_id/features.py`` for the exact settings and the
parity test that pins them to sherpa-onnx's own extractor.

Two deployments are written. ``torch_cpu`` runs the whole graph under ONNX
Runtime. ``ascend_310p_cann81`` runs a trunk on the NPU, and its shape is
dictated by two measurements made 2026-10-29 on an Ascend310P1 with CANN
8.1.RC1: ATC compiles this graph only at exactly 200 frames (every other fixed
length, and dynamic dims, fail its ``AvgPoolV2FusionPass``), which is why the
host feeds it fixed windows; and the whole graph compiled as one OM corrupts
four elements at the statistics pooling ``Concat``, so this one stops at mean
and std and leaves the concat, dense layer and batch norm to the host. Cut
there it agrees with CPU fp32 to cosine 0.999976 at worst on the same windows,
and embeds an utterance in 113 ms against 357 ms. The OM is kept per CANN
release because ATC output is not portable across them; ``runtime_abi`` is what
the loader checks before ACL would fail at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ArtifactBindings,
    AscendRuntimeProfile,
    AudioContract,
    BundleFile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    Digest,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    ONNXRuntimeProfile,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    TensorBinding,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)

BUNDLE_NAME = "campplus-speaker"
DEFAULT_ONNX_REL = "assets/campplus_zh.onnx"
DEFAULT_ONNX_SOURCE_REL = "speaker_id/campplus_zh.onnx"
#: The OM is built on the board it runs on, by the ATC of that board's CANN
#: release; see voice_asr_service/README.md for the command.
DEFAULT_OM_REL = "artifacts/ascend_310p_cann81/campplus_trunk_t200.om"
_ADAPTER_ASSET = "assets/adapter.json"

#: The features the host computes, and the embedding the model returns.
CAMPPLUS_FEATURES_SEMANTIC = "host.campplus.features"
CAMPPLUS_EMBEDDING_SEMANTIC = "host.campplus.embedding"

#: Time is the utterance's length in 10 ms frames, so it is dynamic.
#: One fixed window; the host splits an utterance into these.
CAMPPLUS_FEATURES_SHAPE = (1, 200, 80)
CAMPPLUS_EMBEDDING_SHAPE = (1, 192)

#: Includes the fixed windowing (see ``speaker_id.features.utterance_windows``):
#: a bundle built before it produces different vectors from the same audio, so
#: the version is what stops the two being compared.
CAMPPLUS_PREPROCESSING = "kaldi-fbank80-16khz-25ms-10ms-povey-globalmean-tile200-v2"
CAMPPLUS_POSTPROCESSING = "speaker-embedding-192-l2-normalized-v1"
CAMPPLUS_LOGICAL_REVISION = "campplus@sv-zh-cn-16k-common"

CAMPPLUS_EXECUTION_ROLE = "campplus"

#: The Ascend OM stops before the statistics are concatenated, so it reports the
#: two halves and the host joins them. That is not a preference: compiled whole,
#: the OM corrupts the first four elements of the standard-deviation half
#: (measured 2026-10-29 on an Ascend310P1, one element came back 1638 against a
#: reference 0.769), which is enough to destroy the embedding's direction. Cut
#: here, every one of its 732 comparable nodes agrees with CPU ONNX Runtime to
#: 1e-3 or better, and the finished vectors agree to cosine 0.999976.
CAMPPLUS_STATS_MEAN_SEMANTIC = "host.campplus.stats_mean"
CAMPPLUS_STATS_STD_SEMANTIC = "host.campplus.stats_std"
CAMPPLUS_STATS_SHAPE = (1, 512)

#: The ONNX artifact doubles as the source of the host-side head (the dense
#: layer and its BatchNorm). Keeping one copy of the weights means the two
#: deployments cannot drift apart, and it needs no second artifact to verify.
CAMPPLUS_HEAD_ROLE = "head_source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ascend_bindings() -> ArtifactBindings:
    """The OM's own tensors: one window in, the two statistics halves out."""
    return ArtifactBindings(
        inputs=(
            TensorBinding(semantic=CAMPPLUS_FEATURES_SEMANTIC, dtype="float32", shape=CAMPPLUS_FEATURES_SHAPE, index=0),
        ),
        outputs=(
            TensorBinding(semantic=CAMPPLUS_STATS_MEAN_SEMANTIC, dtype="float32", shape=CAMPPLUS_STATS_SHAPE, index=0),
            TensorBinding(semantic=CAMPPLUS_STATS_STD_SEMANTIC, dtype="float32", shape=CAMPPLUS_STATS_SHAPE, index=1),
        ),
    )


def _ascend_deployment(om_path: Path, onnx_path: Path, *, rel: str, soc: str, runtime_abi: str) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_request_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(runtime="acl", runtime_abi=runtime_abi, soc=soc),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts={
            CAMPPLUS_EXECUTION_ROLE: DeploymentArtifact(path=rel, format="om", sha256=_sha256(om_path)),
            # Auxiliary, not executed: the host reads the head's weights from it.
            # Bundle-relative, like every artifact path; the loader rejects absolute ones.
            CAMPPLUS_HEAD_ROLE: DeploymentArtifact(path=DEFAULT_ONNX_REL, format="onnx", sha256=_sha256(onnx_path)),
        },
        execution=(CAMPPLUS_EXECUTION_ROLE,),
        bindings={CAMPPLUS_EXECUTION_ROLE: _ascend_bindings()},
        audio_contract=_audio_contract(),
    )


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            TensorBinding(semantic=CAMPPLUS_FEATURES_SEMANTIC, dtype="float32", shape=CAMPPLUS_FEATURES_SHAPE, index=0),
        ),
        outputs=(
            TensorBinding(
                semantic=CAMPPLUS_EMBEDDING_SEMANTIC, dtype="float32", shape=CAMPPLUS_EMBEDDING_SHAPE, index=0
            ),
        ),
    )


def _request_contract() -> ExecutionContract:
    """One utterance in, one vector out: no state survives a request."""
    return ExecutionContract(
        state_scope="request",
        execution_structure="direct",
        cancellation_granularity="request_boundary",
        stateful=False,
    )


def _audio_contract() -> AudioContract:
    return AudioContract(
        sample_rate_hz=16000,
        channels=1,
        channel_semantics="mono",
        sample_dtype="float32",
        execution_mode="offline",
    )


def _model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        interface="tensor_model",
        model_type="campplus",
        operation="embed",
        inputs=(SemanticTensor(semantic=CAMPPLUS_FEATURES_SEMANTIC, dtype="float32", shape=CAMPPLUS_FEATURES_SHAPE),),
        outputs=(
            SemanticTensor(semantic=CAMPPLUS_EMBEDDING_SEMANTIC, dtype="float32", shape=CAMPPLUS_EMBEDDING_SHAPE),
        ),
        semantic_identity=SemanticIdentity(
            logical_model_revision=CAMPPLUS_LOGICAL_REVISION,
            preprocessing_contract=CAMPPLUS_PREPROCESSING,
            output_semantics=CAMPPLUS_POSTPROCESSING,
        ),
    )


def _onnx_deployment(onnx_path: Path) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_request_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="onnx",
            target=DeploymentTarget(runtime="onnx"),
            # Two threads, because the consumer has a deadline rather than a
            # preference: the gate holds a transcript only until its voiceprint
            # lands, and the voiceprint node finalizes an END-less utterance
            # 0.8 s before the ASR does. Measured on an Ascend 310P compute
            # pack, 6 s of audio takes 771 ms on one thread and 440 ms on two,
            # so one thread spends 96% of that budget; on the X2 it is
            # 340 ms and 191 ms.
            profile=ONNXRuntimeProfile(device="cpu", intra_op_num_threads=2),
        ),
        artifacts={
            CAMPPLUS_EXECUTION_ROLE: DeploymentArtifact(
                path=DEFAULT_ONNX_REL,
                format="onnx",
                sha256=_sha256(onnx_path),
            )
        },
        execution=(CAMPPLUS_EXECUTION_ROLE,),
        bindings={CAMPPLUS_EXECUTION_ROLE: _bindings()},
        audio_contract=_audio_contract(),
    )


def _materialize(bundle_root: Path, source: Path, relative: str) -> Path:
    destination = bundle_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(source) != _sha256(destination):
        shutil.copy2(source, destination)
    return destination


def _resolve_source(explicit: Path | None, workspace: Path, relative: str) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"speaker embedding source not found: {explicit}")
        return explicit
    candidate = workspace / "models" / relative
    return candidate if candidate.is_file() else None


def _deployment_revision(existing: dict | None, name: str, deployment: CompiledDeployment) -> tuple[str, int]:
    """Keep one deployment's identity stable across repackaging runs."""
    previous = None
    if existing is not None:
        raw = existing.get("deployments", {}).get(name)
        if isinstance(raw, dict):
            previous = raw
    if previous is None:
        return str(uuid4()), 1
    uuid = str(previous.get("uuid") or uuid4())
    revision = int(previous.get("revision", 1))
    candidate = json.loads(
        json.dumps(
            deployment.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"uuid": True, "revision": True}
            )
        )
    )

    def _normalized(value: object) -> object:
        if value is None or value == [] or value == {}:
            return None
        return value

    if any(_normalized(previous.get(key)) != _normalized(value) for key, value in candidate.items()):
        revision += 1
    return uuid, revision


def package_campplus_bundle(
    bundle_root: Path,
    *,
    workspace: Path | None = None,
    onnx_source: Path | None = None,
    om_source: Path | None = None,
) -> Path:
    """Create or refresh the standalone speaker-embedding bundle under ``bundle_root``."""
    root = workspace or bundle_root.parent.parent
    bundle_root.mkdir(parents=True, exist_ok=True)

    onnx = _resolve_source(onnx_source, root, DEFAULT_ONNX_SOURCE_REL)
    if onnx is None:
        onnx = bundle_root / DEFAULT_ONNX_REL
    if not onnx.is_file():
        raise FileNotFoundError(
            f"speaker embedding ONNX missing: {onnx}; see voice_asr_service/README.md for where to get it"
        )
    onnx_path = _materialize(bundle_root, onnx, DEFAULT_ONNX_REL)

    adapter_path = bundle_root / _ADAPTER_ASSET
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_document = (
        json.dumps(
            {
                "interface": "tensor_model",
                "model_type": "campplus",
                "operation": "embed",
                "preprocessing": CAMPPLUS_PREPROCESSING,
                "postprocessing": CAMPPLUS_POSTPROCESSING,
                "logical_model_revision": CAMPPLUS_LOGICAL_REVISION,
                "checkpoint": "campplus_zh.onnx",
                "checkpoint_sha256": _sha256(onnx_path),
                "embedding_dim": 192,
                "num_mel_bins": 80,
                "frame_length_ms": 25,
                "frame_shift_ms": 10,
                "feature_normalize": "global-mean",
                "sample_rate": 16000,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    adapter_path.write_text(adapter_document, encoding="utf-8")

    manifest_path = bundle_root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = (BundleFile(path=_ADAPTER_ASSET),)
    previous_bundle = existing.get("bundle", {}) if isinstance(existing, dict) else {}
    bundle_uuid = str(previous_bundle.get("uuid") or uuid4())
    previous_revision = int(previous_bundle.get("revision", 0)) if previous_bundle else 0
    changed = isinstance(existing, dict) and previous_bundle.get("files") != [{"path": entry.path} for entry in files]
    bundle_revision = previous_revision + int(changed) if existing is not None else 1

    deployments_raw: dict[str, CompiledDeployment] = {"torch_cpu": _onnx_deployment(onnx_path)}
    if om_source is not None:
        om_path = _materialize(bundle_root, om_source, DEFAULT_OM_REL)
        deployments_raw["ascend_310p_cann81"] = _ascend_deployment(
            om_path,
            onnx_path,
            rel=DEFAULT_OM_REL,
            soc="Ascend310P1",
            runtime_abi="cann-8.1.RC1",
        )
    deployments: dict[str, object] = {}
    for name, deployment in deployments_raw.items():
        uuid, revision = _deployment_revision(existing, name, deployment)
        deployments[name] = deployment.model_copy(update={"uuid": uuid, "revision": revision})

    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=BUNDLE_NAME,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, BUNDLE_NAME, files),
            ),
        ),
        model=_model_descriptor(),
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for deployment_name in manifest.deployments:
        load_inference_manifest(bundle_root, deployment_name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root used to resolve default sources")
    parser.add_argument("--onnx-source", type=Path, default=None)
    parser.add_argument(
        "--om-source",
        type=Path,
        default=None,
        help="Ascend OM of the trunk (two outputs, cut before the statistics concat); omit for a CPU-only bundle",
    )
    args = parser.parse_args()
    print(
        package_campplus_bundle(
            args.bundle_root,
            workspace=args.workspace,
            onnx_source=args.onnx_source,
            om_source=args.om_source,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
