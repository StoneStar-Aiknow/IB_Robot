"""Turn one utterance's PCM into a speaker embedding, through a manifest bundle.

The model is described by a schema-v3 ``inference_manifest.json`` and reached
through the unified runtime, like every other model in this repository: the
deployment is named, the artifact digest is declared where the downloader
checks it, and the identity that produced a vector travels with it. That identity matters more here than
usual -- the addressee gate only ever compares two vectors, so swapping the
model behind them changes every comparison while leaving the numbers looking
perfectly reasonable.

The model takes 80-bin Kaldi log-mel features rather than PCM, so the host
computes them (``features``) and this module owns only the request. See
``package_campplus_bundle.py`` for what the bundle declares.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: Shortest utterance worth embedding. Below roughly a second the vector is
#: dominated by whatever phonemes happened to be said, not by the voice.
MIN_SPEECH_SECONDS = 1.0

#: Mono signed 16-bit PCM, the chunk contract's sample format.
_PCM_SCALE = 32768.0


def l2_normalized(vector) -> tuple[float, ...]:
    """Scale to unit length, so a cosine similarity is a plain dot product.

    Returns empty for a zero vector: there is no direction to compare, and
    silently emitting zeros would read as "maximally unlike everything".
    """
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if not norm > 0.0:
        return ()
    return tuple(item / norm for item in values)


def voiceprint_model_id(model_type: str, semantic_identity: Any) -> str:
    """Name the vector space a voiceprint lives in: equal exactly when two vectors are comparable.

    Built from the manifest's semantic identity -- the logical model revision,
    the preprocessing contract (which includes the windowing) and the output
    semantics -- and nothing about the deployment that ran it. The deployments
    are meant to agree (an OM and CPU ONNX Runtime reach cosine 0.999976 on the
    same windows), and their own settings do not move a vector either (one and
    two threads agree to 1.00000000), so folding the deployment fingerprint in
    would only make identical vectors look incomparable. A file name could not
    carry this at all: the same name can hold another model, and the same model
    can arrive under another name.
    """
    canonical = json.dumps(semantic_identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"{model_type}@{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _load_head(onnx_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dense layer and its BatchNorm, folded, read from the bundle's ONNX.

    Taken from the same artifact the CPU deployment runs, so the two
    deployments cannot drift apart and no second set of weights has to be
    verified. Folding the BatchNorm into a scale and a shift keeps the host
    arithmetic to one matrix product per window.

    A parameter is read from the initializers *or* from a ``Constant`` node,
    because an exporter may put it in either -- this very checkpoint keeps the
    BatchNorm's scale and bias in ``Constant`` nodes. Every one of the four is
    then required: substituting a default for a parameter that the graph does
    declare would silently rescale every embedding, which no shape or dtype
    check can see, and this one happens to carry an identity scale and a zero
    bias, so a wrong reading would look right here and be wrong on the next
    checkpoint.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    values = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        attribute = next((a for a in node.attribute if a.name == "value"), None)
        if attribute is not None:
            values.setdefault(node.output[0], numpy_helper.to_array(attribute.t))
    dense = values.get("xvector.dense.linear.weight")
    if dense is None:
        raise ValueError(f"{onnx_path} carries no xvector.dense.linear.weight")
    batchnorm = next((node for node in model.graph.node if node.output[0] == "embedding"), None)
    if batchnorm is None or batchnorm.op_type != "BatchNormalization":
        raise ValueError(f"{onnx_path} does not end in the embedding BatchNormalization")
    epsilon = next((a.f for a in batchnorm.attribute if a.name == "epsilon"), 1e-5)
    labels = ("scale", "bias", "running mean", "running variance")
    missing = [label for label, name in zip(labels, batchnorm.input[1:5], strict=True) if values.get(name) is None]
    if missing:
        raise ValueError(
            f"{onnx_path} BatchNormalization declares {', '.join(missing)} that this reader cannot "
            "resolve to a value; refusing to substitute a default, which would rescale every embedding"
        )
    gamma, beta, mean, variance = (values[name] for name in batchnorm.input[1:5])
    output_dim = dense.shape[0]
    scale = (gamma / np.sqrt(variance + epsilon)).astype(np.float32)
    return (
        dense.astype(np.float32).reshape(output_dim, -1),
        scale,
        (beta - mean * scale).astype(np.float32),
    )


class _RoleExecutor:
    """Run one manifest role instead of the whole model.

    The Ascend deployment stops before the statistics are concatenated, so its
    outputs are the two halves rather than the embedding the model declares.
    Running the role validates against that role's own tensor bindings, which
    is where those halves are declared; going through the model would check the
    deployment against an output it deliberately does not produce. The same
    shape of indirection as ``vad_runtime``'s streaming adapter.
    """

    def __init__(self, session, role: str) -> None:
        self._session = session
        self._role = role

    def execute(self, request, context):
        return self._session.execute_role(self._role, request.inputs, request, context)


class SpeakerEmbedder:
    """One loaded speaker-embedding deployment, one vector per utterance.

    ``model_id`` names the vector space (``voiceprint_model_id``), so two
    vectors are comparable exactly when their ``model_id`` matches.
    """

    def __init__(self, bundle_path: str | Path, deployment: str = "torch_cpu") -> None:
        self._bundle_path = Path(bundle_path)
        self._deployment = deployment
        self._handle = None
        self._providers = None
        self._requests = 0
        self.model_id = ""
        self.dim = 0

    def initialize(self) -> None:
        """Load the deployment; raises when the bundle, digest or identity is wrong."""
        from inference_manifest import load_inference_manifest
        from inference_service.backends import RuntimeContext
        from inference_service.runtime_composition import build_runtime_dependencies
        from inference_service.unified_runtime import ModelRuntimeHandle, ResultAdapter, RuntimeAssembly
        from voice_asr_service.package_campplus_bundle import (
            CAMPPLUS_EMBEDDING_SEMANTIC,
            CAMPPLUS_EXECUTION_ROLE,
        )
        from voice_asr_service.speaker_id.session_builders import register_speaker_embedding_session_builders

        if self._handle is not None:
            raise RuntimeError("speaker embedding runtime is already initialized")
        validated = load_inference_manifest(self._bundle_path, self._deployment)
        identity = validated.top_level_identity
        if (identity.interface, identity.operation) != ("tensor_model", "embed"):
            raise ValueError(
                f"selected deployment is not a tensor_model/embed model: {identity.interface}/{identity.operation}"
            )
        if identity_contract := validated.manifest.model.semantic_identity.preprocessing_contract:
            from voice_asr_service.package_campplus_bundle import CAMPPLUS_PREPROCESSING

            if identity_contract != CAMPPLUS_PREPROCESSING:
                # The windowing is part of the preprocessing the bundle declares;
                # a bundle built before it would be embedded differently and its
                # vectors are not comparable with these.
                raise ValueError(
                    f"bundle declares preprocessing {identity_contract!r}, this build implements "
                    f"{CAMPPLUS_PREPROCESSING!r}; repackage the bundle"
                )
        outputs = validated.manifest.model.outputs
        embedding = next((item for item in outputs if item.semantic == CAMPPLUS_EMBEDDING_SEMANTIC), None)
        if embedding is None:
            raise ValueError(f"deployment must declare the {CAMPPLUS_EMBEDDING_SEMANTIC} output")
        # The Ascend OM stops before the statistics are concatenated, so this
        # deployment returns the two halves and the head runs here. Compiled
        # whole, that concatenation is corrupted on the device; see
        # package_campplus_bundle.
        self._head = None
        if validated.deployment.backend == "ascend":
            from voice_asr_service.package_campplus_bundle import (
                CAMPPLUS_HEAD_ROLE,
                CAMPPLUS_STATS_MEAN_SEMANTIC,
                CAMPPLUS_STATS_STD_SEMANTIC,
            )

            required_outputs = (CAMPPLUS_STATS_MEAN_SEMANTIC, CAMPPLUS_STATS_STD_SEMANTIC)
            self._stats_semantics = required_outputs
            head_path = validated.resolved_artifacts.get(CAMPPLUS_HEAD_ROLE)
            if head_path is None:
                raise ValueError(f"the ascend deployment must carry the {CAMPPLUS_HEAD_ROLE!r} artifact for its head")
            self._head = _load_head(head_path)
            self._abi_guard(validated)
        else:
            required_outputs = (CAMPPLUS_EMBEDDING_SEMANTIC,)
        context = RuntimeContext(
            validated_manifest=validated,
            runtime_profile=getattr(validated, "runtime_profile", None),
        )
        dependencies = build_runtime_dependencies(
            lambda session_registry, _assemblers: register_speaker_embedding_session_builders(session_registry)
        )
        self._providers = dependencies.providers
        try:
            session = dependencies.registry_set.session_builder_registry.create(
                context,
                backend_registry=dependencies.registry_set.backend_registry,
                providers=dependencies.providers,
            )
            executor = _RoleExecutor(session, CAMPPLUS_EXECUTION_ROLE) if self._head is not None else session
            self._handle = ModelRuntimeHandle(
                RuntimeAssembly(
                    runtime_executor=executor,
                    session=session,
                    request_adapter=ResultAdapter(required_outputs=required_outputs),
                    execution_contract=validated.deployment.execution_contract,
                    runtime_id=f"speaker-embedding-{self._deployment}",
                    load_context=context,
                    providers=dependencies.providers,
                    identity=identity,
                    deployment_fingerprint=validated.deployment_fingerprint,
                    runtime_profile_fingerprint=validated.runtime_profile_fingerprint,
                    artifact_integrity=validated.integrity_status,
                )
            )
            self._handle.load(context)
        except Exception:
            self.close()
            raise
        self._role = CAMPPLUS_EXECUTION_ROLE
        self._embedding_semantic = CAMPPLUS_EMBEDDING_SEMANTIC
        self.dim = int(embedding.shape[-1])
        self.model_id = voiceprint_model_id(identity.model_type, validated.manifest.model.semantic_identity)

    @staticmethod
    def _abi_guard(validated) -> None:
        """Refuse an OM built for another CANN release instead of failing inside ACL.

        A mismatch otherwise surfaces as ACL error 500002 during model load,
        after every other node has started -- and because this consumer fails
        open, a voiceprint that never arrives leaves the addressee gate
        accepting everyone while the pipeline looks healthy.
        """
        from inference_service.backends.ascend.host_runtime import detect_cann_release, runtime_abi_matches

        declared = getattr(validated.deployment.runtime_profile.target, "runtime_abi", None)
        host = detect_cann_release()
        if not runtime_abi_matches(declared, host):
            raise ValueError(
                f"deployment was built for CANN {declared!r} but this host runs {host!r}; "
                "pick the deployment matching the host or rebuild the OM with its ATC"
            )

    def embed(self, pcm: bytes) -> tuple[float, ...]:
        """Embed mono S16LE PCM; empty when there was too little to work with.

        The utterance is split into the model's fixed windows (see
        ``utterance_windows``), each embedded, and the unit vectors averaged.
        A single mean over the windows is what keeps a long utterance's vector
        close to the one the whole utterance would have produced, and it is the
        same arithmetic on every deployment, so vectors stay comparable.
        """
        from voice_asr_service.speaker_id.features import SAMPLE_RATE, fbank, global_mean_normalized, utterance_windows

        if self._handle is None:
            raise RuntimeError("speaker embedding runtime is not initialized")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _PCM_SCALE
        features = global_mean_normalized(fbank(samples, SAMPLE_RATE))
        windows = utterance_windows(features)
        if not windows:
            return ()
        total = np.zeros(self.dim, dtype=np.float64)
        for window in windows:
            vector = np.asarray(self._embed_window(window), dtype=np.float64)
            norm = np.linalg.norm(vector)
            if norm > 0.0:
                total += vector / norm
        return l2_normalized(total)

    def _embed_window(self, window: np.ndarray) -> np.ndarray:
        """One window through the loaded deployment, head included."""
        from inference_service.unified_runtime import ExecutionContext, ModelRequest
        from voice_asr_service.package_campplus_bundle import CAMPPLUS_FEATURES_SEMANTIC

        self._requests += 1
        result = self._handle.execute(
            ModelRequest(
                {CAMPPLUS_FEATURES_SEMANTIC: window[None].astype(np.float32)},
                {"deployment": self._deployment},
            ),
            ExecutionContext(f"speaker-embedding-{self._requests}"),
        )
        if self._head is None:
            return np.asarray(result.outputs[self._embedding_semantic]).reshape(-1)
        mean_semantic, std_semantic = self._stats_semantics
        statistics = np.concatenate(
            [
                np.asarray(result.outputs[mean_semantic], dtype=np.float32).reshape(-1),
                np.asarray(result.outputs[std_semantic], dtype=np.float32).reshape(-1),
            ]
        )
        weight, scale, shift = self._head
        return weight @ statistics * scale + shift

    def close(self) -> None:
        handle, self._handle = self._handle, None
        providers, self._providers = self._providers, None
        try:
            if handle is not None:
                handle.close()
        finally:
            if providers is not None:
                providers.close()
