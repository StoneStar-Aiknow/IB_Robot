"""Strict semantic identity, deployment provenance, and mapping-run pinning."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

REQUIRED_MAPPING_ROLES = ("sam2", "ram_plus", "siglip2_image")
_HEX_DIGITS = frozenset("0123456789abcdef")


class MappingRunPinMismatch(ValueError):
    """A runtime changed an identity pinned for the active mapping run."""


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    value = _required_string(value, field_name)
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True)
class EmbeddingSpaceIdentity:
    embedding_space_id: str
    dimension: int
    normalization: str
    image_preprocessing: str
    text_preprocessing: str

    @classmethod
    def from_dict(cls, value: Any) -> "EmbeddingSpaceIdentity":
        if not isinstance(value, dict) or set(value) != {
            "embedding_space_id",
            "dimension",
            "normalization",
            "image_preprocessing",
            "text_preprocessing",
        }:
            raise ValueError("embedding identity has missing or unknown fields")
        dimension = value["dimension"]
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("embedding.dimension must be a positive integer")
        return cls(
            embedding_space_id=_required_string(value["embedding_space_id"], "embedding.embedding_space_id"),
            dimension=dimension,
            normalization=_required_string(value["normalization"], "embedding.normalization"),
            image_preprocessing=_required_string(value["image_preprocessing"], "embedding.image_preprocessing"),
            text_preprocessing=_required_string(value["text_preprocessing"], "embedding.text_preprocessing"),
        )


@dataclass(frozen=True)
class SemanticIdentity:
    logical_model_revision: str
    preprocessing_contract: str
    output_semantics: str
    embedding: EmbeddingSpaceIdentity | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "SemanticIdentity":
        if not isinstance(value, dict):
            raise ValueError("semantic identity must be an object")
        allowed = {"logical_model_revision", "preprocessing_contract", "output_semantics", "embedding"}
        required = allowed - {"embedding"}
        if set(value) - allowed or not required <= set(value):
            raise ValueError("semantic identity has missing or unknown fields")
        return cls(
            logical_model_revision=_required_string(value["logical_model_revision"], "logical_model_revision"),
            preprocessing_contract=_required_string(value["preprocessing_contract"], "preprocessing_contract"),
            output_semantics=_required_string(value["output_semantics"], "output_semantics"),
            embedding=(EmbeddingSpaceIdentity.from_dict(value["embedding"]) if "embedding" in value else None),
        )

    @classmethod
    def from_json(cls, value: str) -> "SemanticIdentity":
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("semantic_identity_json must be valid JSON") from exc
        identity = cls.from_dict(decoded)
        if value != identity.canonical_json:
            raise ValueError("semantic_identity_json must use canonical JSON serialization")
        return identity

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def to_dict(self) -> dict:
        value = {
            "logical_model_revision": self.logical_model_revision,
            "preprocessing_contract": self.preprocessing_contract,
            "output_semantics": self.output_semantics,
        }
        if self.embedding is not None:
            value["embedding"] = asdict(self.embedding)
        return value


def parse_semantic_identities(value: Any, *, required_roles=REQUIRED_MAPPING_ROLES) -> dict[str, SemanticIdentity]:
    if not isinstance(value, dict):
        raise ValueError("semantic identities must be an object keyed by role")
    missing = sorted(set(required_roles) - set(value))
    if missing:
        raise ValueError(f"missing semantic identities: {', '.join(missing)}")
    parsed = {
        role: identity if isinstance(identity, SemanticIdentity) else SemanticIdentity.from_dict(identity)
        for role, identity in value.items()
    }
    for role in ("sam2", "ram_plus"):
        if role in parsed and parsed[role].embedding is not None:
            raise ValueError(f"{role} semantic identity must not define an embedding space")
    siglip = parsed.get("siglip2_image")
    if siglip is not None and siglip.embedding is None:
        raise ValueError("siglip2_image semantic identity must define embedding metadata")
    return parsed


def semantic_identities_dict(value: dict[str, SemanticIdentity]) -> dict[str, dict]:
    return {role: identity.to_dict() for role, identity in sorted(value.items())}


def require_embedding_compatibility(left: SemanticIdentity, right: SemanticIdentity) -> None:
    if left.embedding is None or right.embedding is None:
        raise ValueError("both semantic identities must define embedding metadata")
    if left.embedding != right.embedding:
        raise ValueError("embedding semantic identity is incompatible")


@dataclass(frozen=True)
class DeploymentProvenance:
    instance_id: str
    model_name: str
    model_version: str
    manifest_fingerprint: str
    deployment_name: str
    deployment_fingerprint: str
    backend: str
    runtime_version: str

    @classmethod
    def from_runtime_info(cls, info) -> "DeploymentProvenance":
        return cls(
            instance_id=_required_string(info.instance_id, "instance_id"),
            model_name=_required_string(info.model_name, "model_name"),
            model_version=_required_string(info.model_version, "model_version"),
            manifest_fingerprint=_fingerprint(info.manifest_fingerprint, "manifest_fingerprint"),
            deployment_name=_required_string(info.deployment_name, "deployment_name"),
            deployment_fingerprint=_fingerprint(info.deployment_fingerprint, "deployment_fingerprint"),
            backend=_required_string(info.backend, "backend"),
            runtime_version=_required_string(info.runtime_version, "runtime_version"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeDiagnostic:
    semantic_identity: SemanticIdentity
    provenance: DeploymentProvenance
    configuration_generation: int

    @classmethod
    def from_runtime_info(cls, info) -> "RuntimeDiagnostic":
        if not info.ready:
            raise ValueError(f"model runtime is not ready: {info.failure_reason or info.message}")
        identity = SemanticIdentity.from_json(info.semantic_identity_json)
        if _fingerprint(info.semantic_identity_fingerprint, "semantic_identity_fingerprint") != identity.fingerprint:
            raise ValueError("semantic identity fingerprint does not match semantic_identity_json")
        if identity.embedding is not None:
            embedding = identity.embedding
            if (
                info.embedding_space_id != embedding.embedding_space_id
                or info.embedding_dimension != embedding.dimension
                or info.normalization != embedding.normalization
            ):
                raise ValueError("runtime embedding summary does not match semantic_identity_json")
        generation = info.configuration_generation
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("configuration_generation must be a non-negative integer")
        return cls(identity, DeploymentProvenance.from_runtime_info(info), generation)


@dataclass(frozen=True)
class MappingRunPin:
    run_id: str
    configuration_generation: int
    expected_service_instance_ids: dict[str, str]
    required_semantic_identities: dict[str, SemanticIdentity]

    def __post_init__(self) -> None:
        _required_string(self.run_id, "run_id")
        if self.configuration_generation < 0:
            raise ValueError("configuration_generation must be non-negative")
        identities = parse_semantic_identities(self.required_semantic_identities)
        object.__setattr__(self, "required_semantic_identities", identities)
        missing = sorted(set(REQUIRED_MAPPING_ROLES) - set(self.expected_service_instance_ids))
        if missing:
            raise ValueError(f"missing expected service instances: {', '.join(missing)}")
        for role, instance_id in self.expected_service_instance_ids.items():
            _required_string(instance_id, f"expected_service_instance_ids.{role}")

    def validate_frame(
        self, diagnostics: dict[str, tuple[object, ...] | list[object]]
    ) -> dict[str, DeploymentProvenance]:
        provenances = {}
        for role in REQUIRED_MAPPING_ROLES:
            values = diagnostics.get(role, ())
            if not values:
                raise ValueError(f"frame has no {role} runtime diagnostic")
            parsed = [
                value if isinstance(value, RuntimeDiagnostic) else RuntimeDiagnostic.from_runtime_info(value)
                for value in values
            ]
            for diagnostic in parsed:
                if diagnostic.configuration_generation != self.configuration_generation:
                    raise MappingRunPinMismatch(f"{role} configuration generation does not match the pinned run")
                if diagnostic.provenance.instance_id != self.expected_service_instance_ids[role]:
                    raise MappingRunPinMismatch(f"{role} service instance does not match the pinned run")
                if diagnostic.semantic_identity != self.required_semantic_identities[role]:
                    raise MappingRunPinMismatch(f"{role} semantic identity does not match the pinned run")
            if any(item.semantic_identity != parsed[0].semantic_identity for item in parsed[1:]):
                raise MappingRunPinMismatch(f"{role} batch diagnostics have inconsistent semantic identities")
            provenances[role] = parsed[-1].provenance
        return provenances
