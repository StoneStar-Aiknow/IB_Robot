"""Typed, hardware-independent models for unified inference manifests."""

from __future__ import annotations

import re
from typing import Annotated, Literal, TypeAlias
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from inference_manifest.paths import normalize_bundle_path

_ROLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DEPLOYMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RETURN_SEMANTICS = frozenset({"action", "grasp.poses", "detection.boxes", "segmentation.masks"})

# Two prefixes carve tensors out of a deployment's external contract, for different reasons.
# ``internal.`` names a tensor one role produces and a later role consumes, so it must have
# a declared producer. ``host.`` names a tensor the host computes between roles - a
# PointNet++ neighbourhood, a diffusion sample, an integrated pose - so it has no in-graph
# producer by construction, and a deployment that declares one is driven role by role
# rather than as a single straight-through graph.
INTERNAL_SEMANTIC_PREFIX = "internal."
HOST_SEMANTIC_PREFIX = "host."


def _validate_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError("must be a canonical UUID") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ValueError("must be a canonical non-nil lowercase UUID")
    return value


def _validate_role(value: str) -> str:
    if not _ROLE_PATTERN.fullmatch(value):
        raise ValueError("must start with a letter and contain only letters, digits, '.', '_', or '-'")
    return value


def _validate_shape(value: tuple[int, ...]) -> tuple[int, ...]:
    if any(dimension == 0 or dimension < -1 for dimension in value):
        raise ValueError("dimensions must be positive integers or -1 for dynamic dimensions")
    return value


def _validate_layout(shape: tuple[int, ...], layout: str | None, semantic: str) -> None:
    if len(shape) == 4 and layout is None:
        raise ValueError(f"rank-4 tensor {semantic!r} requires NCHW or NHWC layout")
    if len(shape) != 4 and layout is not None:
        raise ValueError(f"non-rank-4 tensor {semantic!r} must omit layout")


StrictString: TypeAlias = Annotated[str, StringConstraints(strict=True, min_length=1)]
BundlePath: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(normalize_bundle_path),
]
Sha256: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=64, max_length=64),
    AfterValidator(_validate_sha256),
]
ManifestUUID: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=36, max_length=36),
    AfterValidator(_validate_uuid),
]
Revision: TypeAlias = Annotated[int, Field(strict=True, ge=1)]
ExecutionRole: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_validate_role),
]
TensorShape: TypeAlias = Annotated[tuple[int, ...], AfterValidator(_validate_shape)]
TensorDType: TypeAlias = Literal[
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]


class StrictFrozenModel(BaseModel):
    """Base class that rejects coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Digest(StrictFrozenModel):
    algorithm: Literal["sha256"]
    scope: Literal["structure"]
    value: Sha256


class BundleFile(StrictFrozenModel):
    path: BundlePath


class ManifestBundle(StrictFrozenModel):
    uuid: ManifestUUID = Field(default_factory=lambda: str(uuid4()))
    revision: Revision = 1
    name: StrictString
    files: tuple[BundleFile, ...] = Field(min_length=1)
    digest: Digest

    @model_validator(mode="after")
    def validate_unique_paths(self) -> ManifestBundle:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle.files contains duplicate paths")
        return self


class DeploymentTarget(StrictFrozenModel):
    soc: StrictString
    runtime: StrictString


class DeploymentArtifact(StrictFrozenModel):
    path: BundlePath
    format: StrictString
    share_group: StrictString | None = None
    sha256: Sha256 | None = None


class TensorBinding(StrictFrozenModel):
    semantic: StrictString
    runtime_name: StrictString | None = None
    index: int | None = Field(default=None, ge=0)
    dtype: TensorDType
    shape: TensorShape
    layout: Literal["NCHW", "NHWC"] | None = None

    @model_validator(mode="after")
    def validate_runtime_slot_and_layout(self) -> TensorBinding:
        if self.runtime_name is None and self.index is None:
            raise ValueError("a tensor binding requires runtime_name, index, or both")
        _validate_layout(self.shape, self.layout, self.semantic)
        return self


class SemanticTensor(StrictFrozenModel):
    """Model-owned semantic tensor contract shared by all deployments."""

    semantic: StrictString
    dtype: TensorDType
    shape: TensorShape
    layout: Literal["NCHW", "NHWC"] | None = None

    @model_validator(mode="after")
    def validate_layout(self) -> SemanticTensor:
        _validate_layout(self.shape, self.layout, self.semantic)
        return self


class EmbeddingMetadata(StrictFrozenModel):
    embedding_space_id: StrictString
    dimension: Annotated[int, Field(strict=True, gt=0)]
    normalization: StrictString
    image_preprocessing: StrictString
    text_preprocessing: StrictString


class SemanticIdentity(StrictFrozenModel):
    """Logical model-space contract independent of its deployment."""

    logical_model_revision: StrictString
    preprocessing_contract: StrictString
    output_semantics: StrictString
    embedding: EmbeddingMetadata | None = None


class ModelDescriptor(StrictFrozenModel):
    kind: Literal["policy", "perception", "generic"] = "policy"
    family: StrictString = "lerobot"
    inputs: tuple[SemanticTensor, ...] = ()
    outputs: tuple[SemanticTensor, ...] = ()
    semantic_identity: SemanticIdentity | None = None

    @model_validator(mode="after")
    def validate_unique_semantics(self) -> ModelDescriptor:
        for direction, descriptors in (("inputs", self.inputs), ("outputs", self.outputs)):
            semantics = [descriptor.semantic for descriptor in descriptors]
            if len(semantics) != len(set(semantics)):
                raise ValueError(f"model.{direction} contains duplicate semantic descriptors")
        if self.kind != "policy" and (not self.inputs or not self.outputs):
            raise ValueError("non-policy models must declare non-empty model inputs and outputs")
        return self


class ArtifactBindings(StrictFrozenModel):
    inputs: tuple[TensorBinding, ...] = Field(min_length=1)
    outputs: tuple[TensorBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_slots(self) -> ArtifactBindings:
        for direction, bindings in (("inputs", self.inputs), ("outputs", self.outputs)):
            semantics = [binding.semantic for binding in bindings]
            if len(semantics) != len(set(semantics)):
                raise ValueError(f"{direction} contains duplicate semantic bindings")

            runtime_names = [binding.runtime_name for binding in bindings if binding.runtime_name is not None]
            if len(runtime_names) != len(set(runtime_names)):
                raise ValueError(f"{direction} contains duplicate runtime_name values")

            indices = [binding.index for binding in bindings if binding.index is not None]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{direction} contains duplicate runtime indices")
            if indices and len(indices) != len(bindings):
                raise ValueError(f"{direction} must declare indices for every binding or omit them from every binding")
            if direction == "inputs" and indices and sorted(indices) != list(range(len(indices))):
                raise ValueError(f"{direction} runtime indices must be contiguous and start at zero")
        return self


class DeviceLink(StrictFrozenModel):
    semantic: Annotated[str, StringConstraints(strict=True, pattern=r"^internal\.[A-Za-z0-9_.-]+$")]
    producer: ExecutionRole
    consumer: ExecutionRole
    producer_binding: Literal["output", "input"] = "output"
    transport: Literal["device_pointer"]
    owner: Literal["producer", "consumer"]
    lifetime: Literal["inference"] = "inference"

    @model_validator(mode="after")
    def validate_input_source_ownership(self) -> DeviceLink:
        if self.producer_binding == "input" and self.owner != "producer":
            raise ValueError("input-sourced device links require owner='producer'")
        return self


class DeploymentIdentity(StrictFrozenModel):
    uuid: ManifestUUID = Field(default_factory=lambda: str(uuid4()))
    revision: Revision = 1


class TorchDeployment(DeploymentIdentity):
    backend: Literal["torch"]
    device: Literal["cpu", "cuda", "mps", "npu"]


class CompiledDeployment(DeploymentIdentity):
    backend: Literal["ascend", "hisilicon", "rknn", "hmm"]
    target: DeploymentTarget
    artifacts: dict[ExecutionRole, DeploymentArtifact] = Field(min_length=1)
    execution: tuple[ExecutionRole, ...] = Field(min_length=1)
    bindings: dict[ExecutionRole, ArtifactBindings] = Field(min_length=1)
    device_links: tuple[DeviceLink, ...] = ()

    @model_validator(mode="after")
    def validate_execution_graph(self) -> CompiledDeployment:
        if len(self.execution) != len(set(self.execution)):
            raise ValueError("execution contains duplicate roles")

        execution_roles = set(self.execution)
        artifact_roles = set(self.artifacts)
        binding_roles = set(self.bindings)
        if not execution_roles.issubset(artifact_roles):
            raise ValueError(
                f"artifact roles must contain every execution role (missing={sorted(execution_roles - artifact_roles)})"
            )
        if binding_roles != execution_roles:
            raise ValueError(
                "binding roles must exactly match execution roles "
                f"(missing={sorted(execution_roles - binding_roles)}, unexpected={sorted(binding_roles - execution_roles)})"
            )

        roles_by_path: dict[str, list[str]] = {}
        for role, artifact in self.artifacts.items():
            roles_by_path.setdefault(artifact.path, []).append(role)
            if self.backend != "rknn" and artifact.share_group is not None:
                raise ValueError("share_group is only valid for RKNN artifacts")
        for path, roles in roles_by_path.items():
            if len(roles) < 2:
                continue
            groups = {self.artifacts[role].share_group for role in roles}
            if None in groups or len(groups) != 1:
                raise ValueError(
                    f"duplicate artifact path {path!r} requires one shared non-empty share_group for roles {roles}"
                )
        roles_by_group: dict[str, list[str]] = {}
        for role, artifact in self.artifacts.items():
            if artifact.share_group is not None:
                roles_by_group.setdefault(artifact.share_group, []).append(role)
        for group, roles in roles_by_group.items():
            paths = {self.artifacts[role].path for role in roles}
            if len(paths) != 1:
                raise ValueError(f"RKNN share_group {group!r} must reference one artifact path")

            def signature(role: str) -> tuple[tuple[object, ...], tuple[object, ...]]:
                bindings = self.bindings[role]

                def slots(values: tuple[TensorBinding, ...]) -> tuple[object, ...]:
                    return tuple(
                        (value.runtime_name, value.index, value.dtype, value.shape, value.layout) for value in values
                    )

                return slots(bindings.inputs), slots(bindings.outputs)

            if len({signature(role) for role in roles}) != 1:
                raise ValueError(f"RKNN share_group {group!r} roles must expose identical runtime ABI")

        role_positions = {role: index for index, role in enumerate(self.execution)}
        produced_internal: dict[str, str] = {}
        for role in self.execution:
            for binding in self.bindings[role].outputs:
                if binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                    produced_internal[binding.semantic] = role

        linked_inputs = {(link.consumer, link.semantic) for link in self.device_links}
        linked_source_inputs = {
            (link.producer, link.semantic) for link in self.device_links if link.producer_binding == "input"
        }
        input_sourced_semantics = {link.semantic for link in self.device_links if link.producer_binding == "input"}
        ambiguous_semantics = sorted(input_sourced_semantics & set(produced_internal))
        if ambiguous_semantics:
            raise ValueError(
                f"input-sourced device links cannot share semantics with internal outputs: {ambiguous_semantics}"
            )
        for role in self.execution:
            for binding in self.bindings[role].inputs:
                if not binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                    continue
                producer = produced_internal.get(binding.semantic)
                endpoint = (role, binding.semantic)
                if producer is None and endpoint not in linked_inputs and endpoint not in linked_source_inputs:
                    raise ValueError(f"internal input {binding.semantic!r} for role {role!r} has no declared producer")
                if producer is not None and role_positions[producer] >= role_positions[role]:
                    raise ValueError(
                        f"internal input {binding.semantic!r} must be produced before role {role!r} executes"
                    )

        for link in self.device_links:
            if link.producer not in execution_roles or link.consumer not in execution_roles:
                raise ValueError(f"device link {link.semantic!r} references an unknown execution role")
            if role_positions[link.producer] >= role_positions[link.consumer]:
                raise ValueError(f"device link {link.semantic!r} producer must execute before its consumer")
            producer_bindings = (
                self.bindings[link.producer].inputs
                if link.producer_binding == "input"
                else self.bindings[link.producer].outputs
            )
            producer_semantics = {binding.semantic for binding in producer_bindings}
            consumer_inputs = {binding.semantic for binding in self.bindings[link.consumer].inputs}
            if link.semantic not in producer_semantics or link.semantic not in consumer_inputs:
                raise ValueError(
                    f"device link {link.semantic!r} must match producer {link.producer_binding} "
                    "and consumer input bindings"
                )

        return self


Deployment: TypeAlias = Annotated[TorchDeployment | CompiledDeployment, Field(discriminator="backend")]


class InferenceManifest(StrictFrozenModel):
    schema_version: Literal[2]
    bundle: ManifestBundle
    model: ModelDescriptor = Field(default_factory=ModelDescriptor)
    deployments: dict[str, Deployment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_deployment_names(self) -> InferenceManifest:
        invalid = [name for name in self.deployments if not _DEPLOYMENT_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid deployment names: {sorted(invalid)}")
        return self
