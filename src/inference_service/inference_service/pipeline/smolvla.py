"""Shared SmolVLA topology, executor factory, and host embedding resource.

The SmolVLA family is compiled for both HMM and RKNN backends with the same
algorithmic structure: one or more vision roles, a synthetic host ``embedding``
role, a compiled ``prefill`` role, and a compiled ``action`` role that is the
iterative denoising loop body.  This module owns the topology derivation, the
runtime-loaded token/state-projection embedding weights, and the deterministic
host embedding construction, moving that family logic out of the backend
classes into executor-owned stages/resources as required by the unified
pipeline migration (OpenSpec task 4.3).

The embedding algorithm is extracted from the original HMM and RKNN backends so
that both compiled SmolVLA deployments run through the shared
``create_smolvla_executor`` / ``IterativeStage`` path without duplicating
numerical logic.  Topology and binding validation is backend-neutral: it does
not enforce the HMM device-link policy or the RKNN host-link policy, which
remain backend-owned concerns.
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
from inference_service.backends.hmm.backend import HMMBackend
from inference_service.backends.types import RuntimeContext
from inference_service.codecs import ExecutionPlan
from inference_service.pipeline.executor import SequentialModelExecutor
from inference_service.pipeline.stages import (
    EulerIterationStateAdapter,
    HostRoleStage,
    InferenceStage,
    IterationStep,
    IterativeStage,
    ModelStage,
    ResultAdapter,
)

_TOKEN_KEY_CANDIDATES = ("token_embedding.weight", "weight")
_STATE_WEIGHT_KEY_CANDIDATES = ("state_proj.weight", "weight")
_STATE_BIAS_KEY_CANDIDATES = ("state_proj.bias", "bias")
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})
_VELOCITY_SEMANTICS = frozenset({"action"})
_LANGUAGE_TOKEN_SEMANTICS = frozenset({"observation.language.tokens"})
_LANGUAGE_MASK_SEMANTICS = frozenset({"observation.language.attention_mask"})
_STATE_SEMANTICS = frozenset({"observation.state"})
_IMAGE_EMBEDDING_PREFIX = "internal.image_embedding."
_ROLE_SUFFIX = ("embedding", "prefill", "action")
_REQUIRED_CONFIG_INTS = ("chunk_size", "max_action_dim", "num_steps")
_REQUIRED_EMBEDDING_INPUTS = (
    ("observation.language.tokens", "language tokens"),
    ("observation.language.attention_mask", "language mask"),
    ("observation.state", "state"),
)
_REQUIRED_EMBEDDING_OUTPUTS = (
    ("internal.prefix_embeddings", "prefix embeddings"),
    ("internal.prefix_pad_masks", "prefix pad masks"),
    ("internal.attention_mask", "attention mask"),
    ("internal.position_ids", "position ids"),
)


class SmolVLATopologyError(ValueError):
    """Raised when a manifest execution plan is not a supported SmolVLA topology."""


@dataclass(frozen=True)
class SmolVLAEmbeddingWeights:
    """Token embedding plus state projection weights loaded once from artifacts."""

    token_weight: np.ndarray
    state_weight: np.ndarray
    state_bias: np.ndarray


@dataclass(frozen=True)
class SmolVLATopology:
    """Derived SmolVLA role composition and iterative-loop semantic edges."""

    vision_roles: tuple[str, ...]
    pre_loop_roles: tuple[str, ...]
    loop_roles: tuple[str, ...]
    state_semantic: str
    timestep_semantic: str
    velocity_semantic: str


def derive_smolvla_topology(plan: ExecutionPlan) -> SmolVLATopology:
    """Validate and classify SmolVLA role composition without inspecting backend identity."""

    roles = plan.role_names
    if len(roles) <= len(_ROLE_SUFFIX) or roles[-len(_ROLE_SUFFIX) :] != _ROLE_SUFFIX:
        raise SmolVLATopologyError("SmolVLA topology must be vision role(s) followed by embedding, prefill, and action")
    vision_roles = roles[: -len(_ROLE_SUFFIX)]
    if any(role != "vision" and not role.startswith("vision_") for role in vision_roles):
        raise SmolVLATopologyError("SmolVLA topology has invalid vision roles")

    state_semantic = _single_action_input_semantic(plan, _NOISE_SEMANTICS, "noise")
    timestep_semantic = _single_action_input_semantic(plan, _TIME_SEMANTICS, "timestep")
    velocity_semantic = _single_action_output_semantic(plan, _VELOCITY_SEMANTICS, "velocity")
    pre_loop_roles = (*vision_roles, "embedding", "prefill")
    loop_roles = ("action",)
    return SmolVLATopology(
        vision_roles=vision_roles,
        pre_loop_roles=pre_loop_roles,
        loop_roles=loop_roles,
        state_semantic=state_semantic,
        timestep_semantic=timestep_semantic,
        velocity_semantic=velocity_semantic,
    )


def load_smolvla_embedding_weights(context: RuntimeContext) -> SmolVLAEmbeddingWeights:
    """Load token embedding and state projection weights from manifest artifacts.

    Both the ``embedding`` (token table) and ``state_projection`` (linear
    projection) artifacts are loaded exactly once from their manifest-resolved
    ``pt``/``pytorch`` files and converted to contiguous float32 NumPy arrays.
    """

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment):
        raise BackendLoadError("SmolVLA embedding requires a compiled deployment", code="invalid_deployment")
    embedding_artifact = deployment.artifacts.get("embedding")
    if embedding_artifact is None:
        raise BackendLoadError("SmolVLA requires an embedding artifact", code="missing_artifact_role")
    if embedding_artifact.format not in {"pt", "pytorch"}:
        raise BackendLoadError(
            "SmolVLA embedding artifact must use format 'pt' or 'pytorch'", code="invalid_artifact_format"
        )
    projection_artifact = deployment.artifacts.get("state_projection")
    if projection_artifact is None:
        raise BackendLoadError("SmolVLA requires a state_projection artifact", code="missing_artifact_role")
    if projection_artifact.format not in {"pt", "pytorch"}:
        raise BackendLoadError(
            "SmolVLA state_projection artifact must use format 'pt' or 'pytorch'", code="invalid_artifact_format"
        )

    embedding_path = _require_artifact(context, "embedding")
    projection_path = _require_artifact(context, "state_projection")
    token_state = _load_torch_mapping(embedding_path, "embedding")
    token_weight = next((token_state.get(key) for key in _TOKEN_KEY_CANDIDATES if key in token_state), None)
    if token_weight is None:
        raise BackendLoadError("SmolVLA embedding artifact does not contain token weights", code="invalid_embedding")
    projection_state = _load_torch_mapping(projection_path, "state_projection")
    state_weight = next(
        (projection_state.get(key) for key in _STATE_WEIGHT_KEY_CANDIDATES if key in projection_state), None
    )
    state_bias = next(
        (projection_state.get(key) for key in _STATE_BIAS_KEY_CANDIDATES if key in projection_state), None
    )
    if state_weight is None or state_bias is None:
        raise BackendLoadError(
            "SmolVLA state_projection artifact must contain weight and bias", code="invalid_embedding"
        )
    return SmolVLAEmbeddingWeights(
        token_weight=_to_numpy_weight(token_weight, embedding_path, "token_embedding.weight"),
        state_weight=_to_numpy_weight(state_weight, projection_path, "state_proj.weight"),
        state_bias=_to_numpy_weight(state_bias, projection_path, "state_proj.bias"),
    )


def validate_smolvla_plan(
    deployment: CompiledDeployment,
    policy_config: Mapping[str, object],
    embedding: SmolVLAEmbeddingWeights,
) -> None:
    """Validate the SmolVLA execution plan, bindings, and weights without backend policy.

    This validates the structural contract shared by HMM and RKNN deployments:
    vision role shape, embedding input/output semantics, token/state-projection
    weight shapes, prefix binding consistency, positive config integers, and
    noise/action binding shapes.  It deliberately does not enforce the
    HMM device-link requirement or the RKNN host-link prohibition; those remain
    backend-owned concerns so the same resource serves both runtimes.
    """

    roles = tuple(deployment.execution)
    if len(roles) <= len(_ROLE_SUFFIX) or roles[-len(_ROLE_SUFFIX) :] != _ROLE_SUFFIX:
        raise BackendLoadError(
            "SmolVLA requires vision role(s) followed by embedding, prefill, and action",
            code="invalid_execution_plan",
        )
    vision_roles = roles[: -len(_ROLE_SUFFIX)]
    if any(role != "vision" and not role.startswith("vision_") for role in vision_roles):
        raise BackendLoadError(
            "SmolVLA vision roles must be named 'vision' or 'vision_*'", code="invalid_execution_plan"
        )
    for key in _REQUIRED_CONFIG_INTS:
        _require_positive_config(policy_config, key)
    if policy_config.get("add_image_special_tokens", False) is not False:
        raise BackendLoadError(
            "SmolVLA does not support add_image_special_tokens=true", code="unsupported_policy_config"
        )

    if embedding.token_weight.ndim != 2:
        raise BackendLoadError(
            f"SmolVLA token embedding weight must be rank 2, got {embedding.token_weight.shape}",
            code="invalid_embedding",
        )
    _validate_vision_embedding_bindings(deployment, vision_roles)

    embedding_bindings = deployment.bindings["embedding"]
    for semantic, description in _REQUIRED_EMBEDDING_INPUTS:
        _binding_for_semantic(embedding_bindings.inputs, semantic, description)
    state_binding = _binding_for_semantic(embedding_bindings.inputs, "observation.state", "state")
    prefix_embeddings = _binding_for_semantic(
        embedding_bindings.outputs, "internal.prefix_embeddings", "prefix embeddings"
    )
    prefix_pad_masks = _binding_for_semantic(
        embedding_bindings.outputs, "internal.prefix_pad_masks", "prefix pad masks"
    )
    attention_binding = _binding_for_semantic(embedding_bindings.outputs, "internal.attention_mask", "attention mask")
    position_binding = _binding_for_semantic(embedding_bindings.outputs, "internal.position_ids", "position ids")

    hidden_size = embedding.token_weight.shape[1]
    state_dim = state_binding.shape[-1]
    if state_dim < 1 or embedding.state_weight.shape != (hidden_size, state_dim):
        raise BackendLoadError(
            f"SmolVLA state projection shape {embedding.state_weight.shape} must be ({hidden_size}, {state_dim})",
            code="invalid_embedding",
        )
    if embedding.state_bias.shape != (hidden_size,):
        raise BackendLoadError(
            f"SmolVLA state projection bias shape {embedding.state_bias.shape} must be ({hidden_size},)",
            code="invalid_embedding",
        )
    if len(prefix_embeddings.shape) != 3 or prefix_embeddings.shape[-1] != hidden_size:
        raise BackendLoadError(
            "SmolVLA prefix embedding binding is incompatible with the embedding hidden size",
            code="invalid_bindings",
        )
    prefix_length = prefix_embeddings.shape[1]
    if prefix_length < 1 or prefix_pad_masks.shape != (prefix_embeddings.shape[0], prefix_length):
        raise BackendLoadError(
            "SmolVLA prefix pad mask binding must use one consistent static prefix length",
            code="invalid_bindings",
        )
    if attention_binding.shape != (prefix_embeddings.shape[0], prefix_length, prefix_length):
        raise BackendLoadError(
            "SmolVLA attention mask binding must use one consistent static prefix length",
            code="invalid_bindings",
        )
    if position_binding.shape != (prefix_embeddings.shape[0], prefix_length):
        raise BackendLoadError(
            "SmolVLA position ids binding must use one consistent static prefix length",
            code="invalid_bindings",
        )

    action_bindings = deployment.bindings["action"]
    noise_binding = _binding_for_semantics(action_bindings.inputs, _NOISE_SEMANTICS, "noise")
    action_binding = _binding_for_semantics(action_bindings.outputs, _VELOCITY_SEMANTICS, "action output")
    expected = (1, int(policy_config["chunk_size"]), int(policy_config["max_action_dim"]))
    if noise_binding.shape != expected or action_binding.shape != expected:
        raise BackendLoadError(f"SmolVLA noise and action bindings must use shape {expected}", code="invalid_bindings")


def build_embedding_stage(plan: ExecutionPlan, resource: SmolVLAFamilyResource) -> InferenceStage:
    """Build the :class:`HostRoleStage` for the synthetic SmolVLA embedding role."""

    bindings = plan.role("embedding").bindings
    operation = _embedding_operation(bindings, lambda: resource.embedding)
    return HostRoleStage(role="embedding", operation=operation)


class SmolVLAFamilyResource:
    """Executor-owned SmolVLA family resource: embedding weights and config.

    Loaded and closed by the :class:`SequentialModelExecutor` component
    lifecycle so runtime-owned embedding assets have explicit ownership outside
    any backend or model session.  Validation is backend-neutral and does not
    enforce device-link policy.
    """

    def __init__(self, deployment: CompiledDeployment, policy_config: Mapping[str, object]) -> None:
        self._deployment = deployment
        self._configured_policy = dict(policy_config)
        self._weights: SmolVLAEmbeddingWeights | None = None
        self._policy_config: dict[str, object] = {}

    def load(self, context: RuntimeContext) -> None:
        weights = load_smolvla_embedding_weights(context)
        validate_smolvla_plan(self._deployment, self._configured_policy, weights)
        self._weights = weights
        self._policy_config = dict(self._configured_policy)

    def close(self) -> None:
        self._weights = None
        self._policy_config = {}

    @property
    def policy_config(self) -> Mapping[str, object]:
        if not self._policy_config:
            raise BackendInferenceError("SmolVLA family resource is not loaded", code="runtime_not_loaded")
        return self._policy_config

    @property
    def embedding(self) -> SmolVLAEmbeddingWeights:
        if self._weights is None:
            raise BackendInferenceError("SmolVLA family resource is not loaded", code="runtime_not_loaded")
        return self._weights


def create_smolvla_executor(
    plan: ExecutionPlan,
    session: object,
    resource: SmolVLAFamilyResource,
    result_adapter: ResultAdapter,
    *,
    num_inference_steps: int,
    initializer: Callable[[Mapping[str, object]], np.ndarray] | None = None,
    stage_for_role: Callable[[str], InferenceStage] | None = None,
    velocity_trace: list[np.ndarray] | None = None,
    embedding_stage: InferenceStage | None = None,
    extra_components: tuple[object, ...] = (),
) -> SequentialModelExecutor:
    """Build pre-loop and iterative stages from a validated SmolVLA execution plan.

    The executor composes ``ModelStage`` vision/prefill/action roles with a
    :class:`HostRoleStage` synthetic embedding role.  The iterative loop body is
    the ``action`` role only; it repeats over a uniform Euler schedule derived
    from ``num_inference_steps`` (constant timestep decrement ``-1/N`` and
    timesteps ``1 - i/N``) and applies the shared
    :class:`EulerIterationStateAdapter`.

    The executor owns no backend loop or resource beyond the supplied session
    and family ``resource`` components, which are loaded/closed by
    :class:`SequentialModelExecutor`.
    """

    topology = derive_smolvla_topology(plan)
    stage_factory = stage_for_role or (lambda role: ModelStage(role, session))
    embedding = embedding_stage or build_embedding_stage(plan, resource)
    pre_loop = _expand_pre_loop_stages(topology, stage_factory, embedding)
    body = _expand_loop_body_stages(topology, stage_factory)
    steps = _uniform_euler_steps(num_inference_steps)
    timestep_shape = _static_timestep_shape(plan, topology.timestep_semantic)
    state_adapter = EulerIterationStateAdapter(
        state_semantic=topology.state_semantic,
        velocity_semantic=topology.velocity_semantic,
        timestep_semantic=topology.timestep_semantic,
        initializer=initializer,
        timestep_shape=timestep_shape,
        velocity_trace=velocity_trace,
    )
    iterative = IterativeStage(steps, body, state_adapter, loop_roles=topology.loop_roles)
    return SequentialModelExecutor(
        (*pre_loop, iterative),
        result_adapter,
        components=(session, resource, *extra_components),
        execution_plan=plan,
    )


def _expand_pre_loop_stages(
    topology: SmolVLATopology,
    stage_factory: Callable[[str], InferenceStage],
    embedding_stage: InferenceStage,
) -> tuple[InferenceStage, ...]:
    stages: list[InferenceStage] = [stage_factory(role) for role in topology.vision_roles]
    stages.append(embedding_stage)
    stages.append(stage_factory("prefill"))
    return tuple(stages)


def _expand_loop_body_stages(
    topology: SmolVLATopology,
    stage_factory: Callable[[str], InferenceStage],
) -> tuple[InferenceStage, ...]:
    del topology
    return (stage_factory("action"),)


def _uniform_euler_steps(num_inference_steps: int) -> tuple[IterationStep, ...]:
    if type(num_inference_steps) is not int or num_inference_steps < 1:
        raise SmolVLATopologyError("SmolVLA requires a positive integer num_inference_steps")
    delta = -1.0 / num_inference_steps
    return tuple(
        IterationStep(index=step, timestep=1.0 - step / num_inference_steps, delta=delta)
        for step in range(num_inference_steps)
    )


def _static_timestep_shape(plan: ExecutionPlan, timestep_semantic: str) -> tuple[int, ...] | None:
    binding = _action_input_binding(plan, timestep_semantic)
    if any(dimension < 1 for dimension in binding.shape):
        return None
    return binding.shape


def _embedding_operation(
    bindings: ArtifactBindings,
    weights_provider: Callable[[], SmolVLAEmbeddingWeights],
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Create the deterministic host operation for SmolVLA embedding construction."""

    token_binding = _binding_for_semantics(bindings.inputs, _LANGUAGE_TOKEN_SEMANTICS, "language tokens")
    mask_binding = _binding_for_semantics(bindings.inputs, _LANGUAGE_MASK_SEMANTICS, "language mask")
    state_binding = _binding_for_semantic(bindings.inputs, "observation.state", "state")
    image_semantics = tuple(
        binding.semantic for binding in bindings.inputs if binding.semantic.startswith(_IMAGE_EMBEDDING_PREFIX)
    )
    prefix_binding = _binding_for_semantic(bindings.outputs, "internal.prefix_embeddings", "prefix embeddings")
    pad_binding = _binding_for_semantic(bindings.outputs, "internal.prefix_pad_masks", "prefix pad masks")
    attention_binding = _binding_for_semantic(bindings.outputs, "internal.attention_mask", "attention mask")
    position_binding = _binding_for_semantic(bindings.outputs, "internal.position_ids", "position ids")

    def operation(values: Mapping[str, object]) -> Mapping[str, object]:
        weights = weights_provider()
        token_weight = weights.token_weight
        state_weight = weights.state_weight
        state_bias = weights.state_bias
        tokens = np.asarray(values[token_binding.semantic], dtype=np.int64)
        _validate_token_ids(tokens, token_weight.shape[0])
        language = np.ascontiguousarray(token_weight[tokens])
        hidden_size = language.shape[-1]
        state = np.asarray(values[state_binding.semantic], dtype=np.float32)
        if state.ndim != 2 or state.shape[-1] != state_weight.shape[1]:
            raise BackendInferenceError(
                f"SmolVLA state shape {state.shape} is incompatible with projection input {state_weight.shape[1]}",
                code="invalid_state_shape",
            )
        state_embedding = state @ state_weight.T + state_bias
        state_embedding = np.ascontiguousarray(state_embedding[:, None, :], dtype=language.dtype)
        images = [np.asarray(values[semantic], dtype=language.dtype) for semantic in image_semantics]
        if not images or any(image.shape[-1] != hidden_size for image in images):
            raise BackendInferenceError(
                "SmolVLA image and language embeddings must share one hidden dimension",
                code="invalid_embedding_shape",
            )
        scaled_images = [image * math.sqrt(hidden_size) for image in images]
        language = language * math.sqrt(hidden_size)
        prefix = np.concatenate((*scaled_images, language, state_embedding), axis=1)
        language_mask = np.asarray(values[mask_binding.semantic], dtype=bool)
        image_masks = [np.ones(image.shape[:2], dtype=bool) for image in images]
        state_mask = np.ones(state_embedding.shape[:2], dtype=bool)
        prefix_mask = np.concatenate((*image_masks, language_mask, state_mask), axis=1)
        attention_markers = np.concatenate(
            (
                *(np.zeros(image.shape[:2], dtype=np.int32) for image in images),
                np.zeros(language_mask.shape, dtype=np.int32),
                np.ones(state_mask.shape, dtype=np.int32),
            ),
            axis=1,
        )

        prefix = _pad_axis_one(prefix, prefix_binding.shape, np.float32(0.0))
        prefix_mask = _pad_axis_one(prefix_mask, pad_binding.shape, False)
        attention_markers = _pad_axis_one(attention_markers, pad_binding.shape, 0)
        cumulative = np.cumsum(attention_markers, axis=1)
        attention = cumulative[:, None, :] <= cumulative[:, :, None]
        attention &= prefix_mask[:, None, :] & prefix_mask[:, :, None]
        position_ids = np.cumsum(prefix_mask.astype(np.int32), axis=1) - 1
        position_ids = np.where(prefix_mask, position_ids, 0)
        generated: dict[str, object] = {
            prefix_binding.semantic: prefix,
            pad_binding.semantic: prefix_mask,
            attention_binding.semantic: attention,
            position_binding.semantic: position_ids,
        }
        return _convert_semantic_outputs(bindings.outputs, generated, "embedding")

    return operation


def _validate_vision_embedding_bindings(deployment: CompiledDeployment, vision_roles: tuple[str, ...]) -> None:
    image_outputs: list[str] = []
    for role in vision_roles:
        bindings = deployment.bindings[role]
        if len(bindings.inputs) != 1 or len(bindings.outputs) != 1:
            raise BackendLoadError(
                f"SmolVLA vision role {role!r} requires exactly one input and one output",
                code="invalid_bindings",
            )
        if not _is_image_semantic(bindings.inputs[0].semantic) or not bindings.outputs[0].semantic.startswith(
            _IMAGE_EMBEDDING_PREFIX
        ):
            raise BackendLoadError(
                f"SmolVLA vision role {role!r} must map one image to one image embedding",
                code="invalid_bindings",
            )
        image_outputs.append(bindings.outputs[0].semantic)
    embedding_images = [
        binding.semantic
        for binding in deployment.bindings["embedding"].inputs
        if binding.semantic.startswith(_IMAGE_EMBEDDING_PREFIX)
    ]
    if image_outputs != embedding_images:
        raise BackendLoadError(
            "SmolVLA embedding image inputs must match vision execution order",
            code="invalid_bindings",
        )


def _single_action_input_semantic(plan: ExecutionPlan, candidates: frozenset[str], label: str) -> str:
    binding = _action_input_binding(plan, candidates, label)
    return binding.semantic


def _action_input_binding(
    plan: ExecutionPlan,
    candidates: frozenset[str] | str,
    label: str | None = None,
) -> TensorBinding:
    semantics = candidates if isinstance(candidates, frozenset) else frozenset({candidates})
    matches = [binding for binding in plan.role("action").bindings.inputs if binding.semantic in semantics]
    if len(matches) != 1:
        detail = label or "matching"
        raise SmolVLATopologyError(
            f"SmolVLA action role requires exactly one {detail} input semantic, found {sorted(b.semantic for b in matches)}"
        )
    return matches[0]


def _single_action_output_semantic(plan: ExecutionPlan, candidates: frozenset[str], label: str) -> str:
    matches = {binding.semantic for binding in plan.role("action").bindings.outputs if binding.semantic in candidates}
    if len(matches) != 1:
        raise SmolVLATopologyError(
            f"SmolVLA action role requires exactly one {label} output semantic, found {sorted(matches)}"
        )
    return matches.pop()


def _require_positive_config(config: Mapping[str, object], key: str) -> None:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            f"SmolVLA requires positive integer {key!r} in LeRobot config", code="invalid_policy_config"
        )


def _binding_for_semantics(
    bindings: tuple[TensorBinding, ...], semantics: frozenset[str], description: str
) -> TensorBinding:
    return HMMBackend._binding_for_semantics(bindings, semantics, description)


def _binding_for_semantic(bindings: tuple[TensorBinding, ...], semantic: str, description: str) -> TensorBinding:
    return HMMBackend._binding_for_semantic(bindings, semantic, description)


def _convert_semantic_outputs(
    bindings: tuple[TensorBinding, ...],
    values: Mapping[str, object],
    role: str,
) -> dict[str, np.ndarray]:
    return HMMBackend._convert_semantic_outputs(bindings, values, role)


def _pad_axis_one(value: np.ndarray, shape: tuple[int, ...], pad_value: object) -> np.ndarray:
    return HMMBackend._pad_axis_one(value, shape, pad_value)


def _validate_token_ids(tokens: np.ndarray, vocabulary_size: int) -> None:
    HMMBackend._validate_token_ids(tokens, vocabulary_size)


def _load_torch_mapping(path: Path, description: str) -> Mapping[str, object]:
    return HMMBackend._load_torch_mapping(path, description)


def _to_numpy_weight(value: object, path: Path, name: str) -> np.ndarray:
    return HMMBackend._to_numpy_weight(value, path, name)


def _require_artifact(context: RuntimeContext, role: str) -> Path:
    return HMMBackend._require_artifact(context, role)


def _is_image_semantic(semantic: str) -> bool:
    return HMMBackend._is_image_semantic(semantic)


def load_smolvla_policy_config(context: RuntimeContext) -> dict[str, object]:
    """Load and validate the LeRobot policy config for SmolVLA."""

    try:
        value = load_json_strict(context.validated_manifest.bundle_root / "config.json")
    except Exception as exc:
        raise BackendLoadError(f"Unable to read LeRobot config: {exc}", code="invalid_policy_config") from exc
    if not isinstance(value, dict):
        raise BackendLoadError("LeRobot config must be an object", code="invalid_policy_config")
    return value


__all__ = [
    "SmolVLAEmbeddingWeights",
    "SmolVLAFamilyResource",
    "SmolVLATopology",
    "SmolVLATopologyError",
    "build_embedding_stage",
    "create_smolvla_executor",
    "derive_smolvla_topology",
    "load_smolvla_embedding_weights",
    "load_smolvla_policy_config",
    "validate_smolvla_plan",
]
