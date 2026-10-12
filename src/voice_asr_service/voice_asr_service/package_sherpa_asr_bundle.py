"""Package explicit sherpa-onnx artifacts as a schema-v3 ASR bundle."""

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
    ModelIdentity,
    ONNXRuntimeProfile,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    StateLink,
    TensorBinding,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)

IDENTITY = ("tensor_model", "sherpa_onnx", "recognize")
TOKENS_SEMANTIC = "host.asr.tokens"
MODEL_INPUT_SEMANTIC = "host.asr.audio"
MODEL_OUTPUT_SEMANTIC = "host.asr.text"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bindings(role: str) -> ArtifactBindings:
    if role == "tokens":
        return ArtifactBindings(
            inputs=(TensorBinding(semantic=TOKENS_SEMANTIC, dtype="int32", shape=(1,), index=0),),
            outputs=(TensorBinding(semantic=TOKENS_SEMANTIC, dtype="int32", shape=(1,), index=0),),
        )
    return ArtifactBindings(
        inputs=(TensorBinding(semantic=MODEL_INPUT_SEMANTIC, dtype="float32", shape=(-1,), index=0),),
        outputs=(TensorBinding(semantic=MODEL_OUTPUT_SEMANTIC, dtype="float32", shape=(-1,), index=0),),
    )


def _deployment(artifacts: dict[str, DeploymentArtifact], *, streaming: bool, device: str) -> CompiledDeployment:
    roles = tuple(artifacts)
    contract_values = {
        "state_scope": "stream" if streaming else "request",
        "execution_structure": "iterative" if streaming else "direct",
        "cancellation_granularity": "checkpoint" if streaming else "request_boundary",
        "stateful": streaming,
    }
    if streaming:
        contract_values.update(
            {
                "orchestration_visibility": "session",
                "state_bank_mode": "per_stream",
                "max_open_streams": 1,
                "state_links": (
                    StateLink(
                        role="decoder",
                        state_name="decoder_state",
                        owner="session",
                        source="state.decoder_in",
                        target="state.decoder_out",
                        scope="stream",
                        state_bank="sherpa.decoder.bank",
                    ),
                ),
            }
        )
    return CompiledDeployment(
        execution_contract=ExecutionContract(**contract_values),
        runtime_profile=RoleRuntimeProfile(
            backend="onnx",
            target=DeploymentTarget(runtime="onnx"),
            profile=ONNXRuntimeProfile(device=device),
        ),
        artifacts=artifacts,
        execution=roles,
        bindings={role: _bindings(role) for role in roles},
        audio_contract=AudioContract(
            sample_rate_hz=16000,
            channels=1,
            channel_semantics="mono",
            sample_dtype="float32",
            frame_size=512,
            chunk_size=512,
            execution_mode="streaming" if streaming else "offline",
        ),
    )


def _encoder_om_bindings(encoder_onnx: Path) -> tuple[ArtifactBindings, int]:
    """Bind every static encoder input/output by index; states pair by name via semantics.

    onnxsim leaves output shapes undeclared, so one zero-input ONNX Runtime pass records the
    concrete output shapes the OM exposes (ATC keeps the ONNX input/output order).
    """
    import numpy as np
    import onnx
    import onnxruntime as ort

    from voice_asr_service.transducer_ascend import (
        ENCODER_FEATURES_SEMANTIC,
        ENCODER_OUT_SEMANTIC,
        ENCODER_STATE_IN_PREFIX,
        ENCODER_STATE_OUT_PREFIX,
    )

    session = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    feeds = {}
    for value in session.get_inputs():
        if not value.shape or any(not isinstance(dim, int) or dim <= 0 for dim in value.shape):
            raise ValueError(f"encoder input {value.name!r} must have a static shape, got {value.shape}")
        feeds[value.name] = np.zeros(value.shape, dtype=np.int64 if "int64" in value.type else np.float32)
    names = [value.name for value in session.get_outputs()]
    results = session.run(names, feeds)
    dtypes = {np.dtype(np.float32): "float32", np.dtype(np.int64): "int64"}

    inputs = tuple(
        TensorBinding(
            semantic=ENCODER_FEATURES_SEMANTIC if index == 0 else ENCODER_STATE_IN_PREFIX + name,
            index=index,
            dtype=dtypes[array.dtype],
            shape=array.shape,
        )
        for index, (name, array) in enumerate(feeds.items())
    )
    outputs = tuple(
        TensorBinding(
            semantic=ENCODER_OUT_SEMANTIC if index == 0 else ENCODER_STATE_OUT_PREFIX + name.removeprefix("new_"),
            index=index,
            dtype=dtypes[array.dtype],
            shape=array.shape,
        )
        for index, (name, array) in enumerate(zip(names, results, strict=True))
    )
    metadata = {prop.key: prop.value for prop in onnx.load(str(encoder_onnx), load_external_data=False).metadata_props}
    return ArtifactBindings(inputs=inputs, outputs=outputs), int(metadata["decode_chunk_len"])


def _ascend_deployment(
    artifacts: dict[str, DeploymentArtifact],
    encoder_bindings: ArtifactBindings,
    decode_chunk_len: int,
    *,
    soc: str,
    runtime_abi: str,
) -> CompiledDeployment:
    """Encoder OM on Ascend, the small decoder/joiner on ONNX Runtime CPU, greedy search on host."""
    cpu = RoleRuntimeProfile(
        backend="onnx", target=DeploymentTarget(runtime="onnx"), profile=ONNXRuntimeProfile(device="cpu")
    )
    base = _deployment(
        {role: artifacts[role] for role in ("tokens", "encoder", "decoder", "joiner")}, streaming=True, device="cpu"
    )
    # Unlike the sherpa deployments, the recurrent state lives in the encoder OM: the runner keeps the
    # cached tensors device-resident in ping-pong banks it owns exclusively (as SileroVadAclRunner does),
    # so declare the encoder role and runtime_exclusive banks instead of the sherpa decoder state link.
    encoder_state_contract = ExecutionContract(
        state_scope="stream",
        execution_structure="iterative",
        cancellation_granularity="checkpoint",
        stateful=True,
        orchestration_visibility="session",
        state_bank_mode="runtime_exclusive",
        max_open_streams=1,
        state_links=(
            StateLink(
                role="encoder",
                state_name="cached",
                owner="session",
                source="state.encoder_in",
                target="state.encoder_out",
                scope="runtime",
                state_bank="asr.encoder.bank",
            ),
        ),
    )
    return base.model_copy(
        update={
            "execution_contract": encoder_state_contract,
            "runtime_profile": None,
            "role_identities": {
                "encoder": ModelIdentity(interface="tensor_model", model_type="zipformer2_encoder", operation="encode"),
                "decoder": ModelIdentity(interface="tensor_model", model_type="transducer_decoder", operation="decode"),
                "joiner": ModelIdentity(interface="tensor_model", model_type="transducer_joiner", operation="join"),
            },
            "role_runtime_profiles": {
                "encoder": RoleRuntimeProfile(
                    backend="ascend",
                    target=DeploymentTarget(runtime="acl", runtime_abi=runtime_abi, soc=soc),
                    profile=AscendRuntimeProfile(device_id=0),
                ),
                "decoder": cpu,
                "joiner": cpu,
            },
            "bindings": {**base.bindings, "encoder": encoder_bindings},
            "metadata": {"decode_chunk_len": decode_chunk_len},
        }
    )


def package_sherpa_asr_bundle(
    bundle_root: Path,
    *,
    artifacts: dict[str, Path],
    streaming: bool,
    streaming_kind: str = "transducer",
    include_cuda: bool = True,
    ascend_encoder_om: Path | None = None,
    ascend_encoder_onnx: Path | None = None,
    ascend_soc: str = "Ascend310P1",
    ascend_runtime_abi: str = "cann-8.3.RC1",
) -> Path:
    """Copy explicitly named artifacts and write/validate the manifest."""
    if streaming and streaming_kind not in {"transducer", "paraformer"}:
        raise ValueError("streaming_kind must be transducer or paraformer")
    expected = (
        {"tokens", "encoder", "decoder", "joiner"}
        if streaming and streaming_kind == "transducer"
        else {"tokens", "encoder", "decoder"}
        if streaming
        else {"tokens", "model"}
    )
    missing = sorted(expected - set(artifacts))
    if missing:
        raise ValueError(f"missing ASR artifact roles: {', '.join(missing)}")
    root = bundle_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    deployed: dict[str, DeploymentArtifact] = {}
    for role, source in artifacts.items():
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ASR artifact for role {role!r} not found: {source}")
        relative = f"artifacts/{role}/{source.name}"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        deployed[role] = DeploymentArtifact(
            path=relative, format=source.suffix.lstrip(".") or "file", sha256=_sha256(destination)
        )

    metadata_path = root / "bundle.metadata"
    metadata_path.write_text("sherpa-onnx ASR bundle\n", encoding="utf-8")
    bundle_files = (BundleFile(path="bundle.metadata"),)
    deployments_raw = {"torch_cpu": _deployment(deployed, streaming=streaming, device="cpu")}
    if include_cuda:
        deployments_raw["torch_cuda"] = _deployment(deployed, streaming=streaming, device="cuda")
    if ascend_encoder_om is not None:
        if not streaming or streaming_kind != "transducer" or ascend_encoder_onnx is None:
            raise ValueError("the ascend deployment needs a streaming transducer plus the static encoder ONNX")
        source = ascend_encoder_om.resolve()
        relative = f"artifacts/encoder_ascend/{source.name}"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        encoder_bindings, decode_chunk_len = _encoder_om_bindings(ascend_encoder_onnx)
        ascend_artifacts = {
            **deployed,
            "encoder": DeploymentArtifact(path=relative, format="om", sha256=_sha256(destination)),
        }
        deployments_raw["ascend_310p"] = _ascend_deployment(
            ascend_artifacts, encoder_bindings, decode_chunk_len, soc=ascend_soc, runtime_abi=ascend_runtime_abi
        )

    # Repackaging keeps a stable bundle identity, mirroring the silero/fullsubnet
    # packagers: reuse the existing uuid and only bump the revision when the
    # bundle files or any deployment artifacts change.
    manifest_path = root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
    previous_bundle = existing.get("bundle", {}) if isinstance(existing, dict) else {}
    bundle_uuid = str(previous_bundle.get("uuid") or uuid4())
    previous_revision = int(previous_bundle.get("revision", 0)) if previous_bundle else 0

    # Per-deployment identity stability (mirrors the silero/fullsubnet
    # packagers): keep the same uuid, bump the revision only when the
    # deployment's semantic content (null/empty normalized) changes.
    deployments: dict[str, object] = {}
    for name, deployment in deployments_raw.items():
        previous_deployment = None
        if isinstance(existing, dict):
            raw = existing.get("deployments", {}).get(name)
            if isinstance(raw, dict):
                previous_deployment = raw
        if previous_deployment is None:
            deployment_uuid, deployment_revision = str(uuid4()), 1
        else:
            deployment_uuid = str(previous_deployment.get("uuid") or uuid4())
            deployment_revision = int(previous_deployment.get("revision", 1))
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

            if any(_normalized(previous_deployment.get(key)) != _normalized(value) for key, value in candidate.items()):
                deployment_revision += 1
        deployments[name] = deployment.model_copy(update={"uuid": deployment_uuid, "revision": deployment_revision})

    previous_artifacts = (
        {name: deployment.get("artifacts") for name, deployment in existing.get("deployments", {}).items()}
        if isinstance(existing, dict)
        else None
    )
    candidate_artifacts = {
        name: json.loads(deployment.model_dump_json(exclude_none=True)).get("artifacts")
        for name, deployment in deployments_raw.items()
    }
    changed = isinstance(existing, dict) and (
        previous_bundle.get("files") != [{"path": entry.path} for entry in bundle_files]
        or previous_artifacts != candidate_artifacts
    )
    bundle_revision = previous_revision + int(changed) if isinstance(existing, dict) else 1

    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name="sherpa-onnx-asr",
            files=bundle_files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, "sherpa-onnx-asr", bundle_files),
            ),
        ),
        model=ModelDescriptor(
            interface=IDENTITY[0],
            model_type=IDENTITY[1],
            operation=IDENTITY[2],
            inputs=(SemanticTensor(semantic=MODEL_INPUT_SEMANTIC, dtype="float32", shape=(-1,)),),
            outputs=(SemanticTensor(semantic=MODEL_OUTPUT_SEMANTIC, dtype="float32", shape=(-1,)),),
            semantic_identity=SemanticIdentity(
                logical_model_revision="sherpa-onnx-asr@bundle-v1",
                preprocessing_contract="mono-16khz-float32",
                output_semantics="utf8-transcript",
            ),
        ),
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for name in deployments:
        load_inference_manifest(root, name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--joiner", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--paraformer", action="store_true", help="Package streaming Paraformer roles")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--ascend-encoder-om", type=Path, help="Encoder OM for the ascend_310p deployment")
    parser.add_argument("--ascend-encoder-onnx", type=Path, help="Static ONNX the encoder OM was compiled from")
    parser.add_argument("--ascend-soc", default="Ascend310P1")
    parser.add_argument("--ascend-runtime-abi", default="cann-8.3.RC1")
    args = parser.parse_args()
    artifacts = {"tokens": args.tokens}
    if args.offline:
        if args.model is None:
            parser.error("--model is required with --offline")
        artifacts["model"] = args.model
    else:
        required_roles = (("encoder", args.encoder), ("decoder", args.decoder))
        if not args.paraformer:
            required_roles += (("joiner", args.joiner),)
        for role, path in required_roles:
            if path is None:
                parser.error(f"--{role} is required unless --offline is used")
            artifacts[role] = path
    print(
        package_sherpa_asr_bundle(
            args.bundle_root,
            artifacts=artifacts,
            streaming=not args.offline,
            streaming_kind="paraformer" if args.paraformer else "transducer",
            include_cuda=not args.skip_cuda,
            ascend_encoder_om=args.ascend_encoder_om,
            ascend_encoder_onnx=args.ascend_encoder_onnx,
            ascend_soc=args.ascend_soc,
            ascend_runtime_abi=args.ascend_runtime_abi,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
