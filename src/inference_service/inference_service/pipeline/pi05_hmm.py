"""Executor-owned PI0.5 HMM family resources and host-stage construction.

The modular HMM PI0.5 deployment splits one denoising step into several TCIM
modules and declares a synthetic ``embedding`` host role whose outputs feed the
compiled ``prefill`` and ``decode`` modules.  This module owns the deterministic
host computation (embedding construction and sinusoidal timestep embedding) and
the runtime-loaded embedding weights, moving that family logic out of
:class:`HMMModelSession` and into executor-owned stages/resources as required by the
unified pipeline migration.

The embedding and time-embedding algorithms are extracted from the original
HMM backend so that compiled HMM PI0.5 runs through ``HMMModelSession`` and the
shared ``create_pi05_executor`` / ``IterativeStage`` path without duplicating
numerical logic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import ArtifactBindings, CompiledDeployment, TensorBinding
from inference_manifest.json_utils import load_json_strict
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.hmm.host_utils import (
    binding_for_semantic,
    binding_for_semantics,
    convert_runtime_value,
    convert_semantic_outputs,
    load_torch_mapping,
    pad_axis_one,
    require_artifact,
    static_shape,
    to_additive_attention,
    to_numpy_weight,
    validate_token_ids,
)
from inference_service.backends.hmm.host_utils import (
    validate_pi05_plan as validate_hmm_pi05_plan,
)
from inference_service.backends.types import RuntimeContext
from inference_service.pipeline.stages import HostComputeStage, HostRoleStage, InferenceStage

_TOKEN_KEY_CANDIDATES = ("token_embedding.weight", "weight")
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_LANGUAGE_TOKEN_SEMANTICS = frozenset({"observation.language.tokens"})
_LANGUAGE_MASK_SEMANTICS = frozenset({"observation.language.attention_mask"})


@dataclass(frozen=True)
class _PI05EmbeddingWeights:
    token_weight: np.ndarray


def load_pi05_embedding_weights(context: RuntimeContext) -> _PI05EmbeddingWeights:
    """Load PI0.5 token embedding weights from the manifest artifact."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment):
        raise BackendLoadError("HMM PI0.5 embedding requires a compiled deployment", code="invalid_deployment")
    embedding_artifact = deployment.artifacts.get("embedding")
    if embedding_artifact is None:
        raise BackendLoadError("HMM PI0.5 requires an embedding artifact", code="missing_artifact_role")
    if embedding_artifact.format not in {"pt", "pytorch"}:
        raise BackendLoadError(
            "HMM PI0.5 embedding artifact must use format 'pt' or 'pytorch'", code="invalid_artifact_format"
        )
    embedding_path = _require_artifact(context, "embedding")
    token_state = load_torch_mapping(embedding_path, "embedding")
    token_weight = next((token_state.get(key) for key in _TOKEN_KEY_CANDIDATES if key in token_state), None)
    if token_weight is None:
        raise BackendLoadError("HMM PI0.5 embedding artifact does not contain token weights", code="invalid_embedding")
    return _PI05EmbeddingWeights(token_weight=to_numpy_weight(token_weight, embedding_path, "weight"))


def load_pi05_policy_config(context: RuntimeContext) -> dict[str, object]:
    """Load and validate the LeRobot policy config for HMM PI0.5."""

    try:
        value = load_json_strict(context.validated_manifest.bundle_root / "config.json")
    except Exception as exc:
        raise BackendLoadError(f"Unable to read LeRobot config: {exc}", code="invalid_policy_config") from exc
    if not isinstance(value, dict):
        raise BackendLoadError("LeRobot config must be an object", code="invalid_policy_config")
    return value


def validate_pi05_plan(
    deployment: CompiledDeployment,
    policy_config: Mapping[str, object],
    embedding: _PI05EmbeddingWeights,
) -> None:
    """Validate the modular HMM PI0.5 execution plan, bindings, and device links."""

    validate_hmm_pi05_plan(deployment, policy_config, embedding.token_weight)


def build_embedding_stage(
    deployment: CompiledDeployment,
    resource: PI05HMMFamilyResource,
    policy_config: Mapping[str, object],
) -> InferenceStage:
    """Build the :class:`HostRoleStage` for the synthetic PI0.5 embedding role."""

    bindings = deployment.bindings["embedding"]
    chunk_size = _positive_config_int(policy_config, "chunk_size")
    operation = _embedding_operation(bindings, lambda: resource.embedding.token_weight, chunk_size)
    return HostRoleStage(role="embedding", operation=operation)


def build_time_prep_stage(
    deployment: CompiledDeployment,
    policy_config: Mapping[str, object],
) -> InferenceStage:
    """Build a :class:`HostComputeStage` that prepares sinusoidal timestep embeddings.

    The stage reads the scalar ``time`` semantic set by the iteration state
    adapter and replaces it with the sinusoidal embedding expected by the
    compiled ``time_mlp`` module.
    """

    time_binding = binding_for_semantics(
        deployment.bindings["time_mlp"].inputs,
        frozenset({"time", "timestep", "action.time", "_time"}),
        "time",
    )
    min_period = float(policy_config.get("min_period", 0.004))
    max_period = float(policy_config.get("max_period", 4.0))
    operation = _time_prep_operation(time_binding, min_period, max_period)
    return HostComputeStage(operation=operation)


class PI05HMMFamilyResource:
    """Executor-owned PI0.5 HMM family resource: embedding weights and config.

    Loaded and closed by the :class:`SequentialModelExecutor` component
    lifecycle so runtime-owned embedding assets have explicit ownership outside
    :class:`HMMModelSession`.
    """

    def __init__(self, deployment: CompiledDeployment, policy_config: Mapping[str, object]) -> None:
        self._deployment = deployment
        self._configured_policy = dict(policy_config)
        self._embedding: _PI05EmbeddingWeights | None = None
        self._policy_config: dict[str, object] = {}

    def load(self, context: RuntimeContext) -> None:
        embedding = load_pi05_embedding_weights(context)
        validate_pi05_plan(self._deployment, self._configured_policy, embedding)
        self._embedding = embedding
        self._policy_config = dict(self._configured_policy)

    def close(self) -> None:
        self._embedding = None
        self._policy_config = {}

    @property
    def policy_config(self) -> Mapping[str, object]:
        if not self._policy_config:
            raise BackendInferenceError("PI0.5 HMM family resource is not loaded", code="runtime_not_loaded")
        return self._policy_config

    @property
    def embedding(self) -> _PI05EmbeddingWeights:
        if self._embedding is None:
            raise BackendInferenceError("PI0.5 HMM family resource is not loaded", code="runtime_not_loaded")
        return self._embedding


def _embedding_operation(
    bindings: ArtifactBindings,
    token_weight_provider: Callable[[], np.ndarray],
    chunk_size: int,
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Create the deterministic host operation for PI0.5 embedding construction."""

    token_binding = binding_for_semantics(bindings.inputs, _LANGUAGE_TOKEN_SEMANTICS, "language tokens")
    mask_binding = binding_for_semantics(bindings.inputs, _LANGUAGE_MASK_SEMANTICS, "language mask")
    image_semantics = tuple(
        binding.semantic for binding in bindings.inputs if binding.semantic.startswith("internal.image_embedding.")
    )
    prefix_binding = binding_for_semantic(bindings.outputs, "internal.prefix_embeddings", "prefix embeddings")
    attention_binding = binding_for_semantic(bindings.outputs, "internal.prefix_attention", "prefix attention")
    decode_attention_binding = binding_for_semantic(bindings.outputs, "internal.decode_attention", "decode attention")
    prefix_positions_binding = binding_for_semantic(bindings.outputs, "internal.prefix_positions", "prefix positions")
    decode_positions_binding = binding_for_semantic(bindings.outputs, "internal.decode_positions", "decode positions")

    def operation(values: Mapping[str, object]) -> Mapping[str, object]:
        token_weight = token_weight_provider()
        hidden_size = token_weight.shape[1]
        tokens = np.asarray(values[token_binding.semantic], dtype=np.int64)
        validate_token_ids(tokens, token_weight.shape[0])
        language = token_weight[tokens] * math.sqrt(hidden_size)
        language_mask = np.asarray(values[mask_binding.semantic], dtype=bool)
        images = [np.asarray(values[semantic], dtype=language.dtype) for semantic in image_semantics]
        prefix = np.concatenate((*images, language), axis=1)
        image_masks = [np.ones(image.shape[:2], dtype=bool) for image in images]
        prefix_mask = np.concatenate((*image_masks, language_mask), axis=1)
        actual_length = prefix.shape[1]

        prefix = pad_axis_one(prefix, prefix_binding.shape, 0.0)
        prefix_mask = pad_axis_one(prefix_mask, prefix_binding.shape[:2], False)

        query_length = attention_binding.shape[-2]
        key_length = attention_binding.shape[-1]
        if query_length != prefix.shape[1] or key_length < actual_length:
            raise BackendInferenceError(
                "PI0.5 prefix attention binding has incompatible dimensions", code="invalid_bindings"
            )
        key_mask = np.zeros((prefix.shape[0], key_length), dtype=bool)
        key_mask[:, : prefix_mask.shape[1]] = prefix_mask
        prefix_attention = prefix_mask[:, None, :, None] & key_mask[:, None, None, :]

        decode_key_length = decode_attention_binding.shape[-1]
        if decode_attention_binding.shape[-2] != chunk_size or actual_length + chunk_size > decode_key_length:
            raise BackendInferenceError(
                "PI0.5 decode attention binding has incompatible dimensions", code="invalid_bindings"
            )
        decode_keys = np.zeros((prefix.shape[0], decode_key_length), dtype=bool)
        decode_keys[:, :actual_length] = prefix_mask[:, :actual_length]
        decode_keys[:, actual_length : actual_length + chunk_size] = True
        decode_attention = np.broadcast_to(
            decode_keys[:, None, None, :],
            (prefix.shape[0], 1, chunk_size, decode_key_length),
        ).copy()

        generated: dict[str, object] = {
            prefix_binding.semantic: prefix,
            attention_binding.semantic: to_additive_attention(prefix_attention, attention_binding.dtype),
            decode_attention_binding.semantic: to_additive_attention(decode_attention, decode_attention_binding.dtype),
            prefix_positions_binding.semantic: np.arange(prefix.shape[1], dtype=np.int64)[None, :],
            decode_positions_binding.semantic: np.arange(prefix.shape[1], prefix.shape[1] + chunk_size, dtype=np.int64)[
                None, :
            ],
        }
        return convert_semantic_outputs(bindings.outputs, generated, "embedding")

    return operation


def _time_prep_operation(
    time_binding: TensorBinding,
    min_period: float,
    max_period: float,
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Create the deterministic host operation for sinusoidal timestep embedding."""

    shape = static_shape(time_binding)
    if len(shape) != 2 or shape[-1] % 2 != 0:
        raise BackendInferenceError(
            f"PI0.5 time MLP input requires shape (B, even_dimension), got {shape}",
            code="invalid_time_binding",
        )
    half = shape[-1] // 2
    fraction = np.linspace(0.0, 1.0, half, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction

    def operation(values: Mapping[str, object]) -> Mapping[str, object]:
        raw = values.get("time")
        is_scalar_array = isinstance(raw, np.ndarray) and raw.shape == ()
        if is_scalar_array or isinstance(raw, int | float | np.floating | np.integer):
            time_value = float(raw)
        else:
            raise BackendInferenceError(
                "PI0.5 time preparation requires a scalar time value", code="invalid_time_value"
            )
        scaled = time_value * (2.0 * math.pi / period)
        value = np.concatenate((np.sin(scaled), np.cos(scaled)))[None, :]
        if shape[0] != 1:
            value = np.broadcast_to(value, shape)
        return {"time": convert_runtime_value(time_binding, value, "time_mlp", "input")}

    return operation


def _positive_config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            f"HMM PI0.5 requires positive integer {key!r} in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _require_artifact(context: RuntimeContext, role: str) -> Path:
    return require_artifact(context, role)


__all__ = [
    "PI05HMMFamilyResource",
    "build_embedding_stage",
    "build_time_prep_stage",
    "load_pi05_embedding_weights",
    "load_pi05_policy_config",
    "validate_pi05_plan",
]
