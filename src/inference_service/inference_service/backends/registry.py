"""Static, lazy backend registry and compatibility validation."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from inference_manifest import CompiledDeployment, Deployment, TorchDeployment
from inference_service.backends.errors import BackendCompatibilityError, BackendRegistryError
from inference_service.backends.types import InferenceBackend, RuntimeContext

CANONICAL_BACKENDS = ("torch", "ascend", "hisilicon", "rknn", "hmm")
POLICY_FAMILIES = frozenset({"act", "diffusion", "pi05", "smolvla"})
NON_POLICY_MODEL_KINDS = frozenset({"perception", "generic"})
PERCEPTION_FAMILIES = frozenset({"ram_plus", "sam2", "siglip2", "grounding_dino", "graspgen", "dummy_echo"})
_FACTORY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")

TargetValidator = Callable[[Deployment], str | None]


@dataclass(frozen=True)
class ConformanceEvidence:
    """Immutable record that one kind/family/backend pair passed support conformance."""

    kind: str
    family: str
    reference: str = ""


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    factory: str
    target_validator: TargetValidator
    supported_policy_families: frozenset[str] = frozenset()
    supported_model_kinds: frozenset[str] = frozenset()
    supported_model_families: frozenset[str] = frozenset()
    conformance_evidence: frozenset[ConformanceEvidence] = frozenset()

    @property
    def evidence_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset((evidence.kind, evidence.family) for evidence in self.conformance_evidence)

    def validate_definition(self) -> None:
        if self.name not in CANONICAL_BACKENDS:
            raise BackendRegistryError(
                f"backend descriptor uses non-canonical name {self.name!r}",
                code="non_canonical_backend",
            )
        if not _FACTORY_PATTERN.fullmatch(self.factory):
            raise BackendRegistryError(
                f"backend {self.name!r} has invalid factory import string {self.factory!r}",
                code="invalid_factory",
            )
        if not (self.supported_policy_families or self.supported_model_kinds or self.supported_model_families):
            raise BackendRegistryError(
                f"backend {self.name!r} must declare policy or model support",
                code="empty_model_support",
            )
        unsupported = self.supported_policy_families - POLICY_FAMILIES
        if unsupported:
            raise BackendRegistryError(
                f"backend {self.name!r} declares unknown policy families: {sorted(unsupported)}",
                code="unknown_policy_family",
            )
        unsupported_kinds = self.supported_model_kinds - NON_POLICY_MODEL_KINDS
        if unsupported_kinds:
            raise BackendRegistryError(
                f"backend {self.name!r} declares unknown or policy-only model kinds: {sorted(unsupported_kinds)}",
                code="unknown_model_kind",
            )
        invalid_families = sorted(family for family in self.supported_model_families if not family.strip())
        if invalid_families:
            raise BackendRegistryError(
                f"backend {self.name!r} declares empty model families",
                code="invalid_model_family",
            )
        if not callable(self.target_validator):
            raise BackendRegistryError(
                f"backend {self.name!r} target validator is not callable",
                code="invalid_target_validator",
            )
        self._validate_evidence_coherence()

    def _validate_evidence_coherence(self) -> None:
        for evidence in self.conformance_evidence:
            if evidence.kind == "policy":
                if evidence.family not in self.supported_policy_families:
                    raise BackendRegistryError(
                        f"backend {self.name!r} conformance evidence claims undeclared policy "
                        f"family {evidence.family!r}",
                        code="evidence_overclaims_support",
                    )
                continue
            if evidence.kind not in self.supported_model_kinds and evidence.family not in self.supported_model_families:
                raise BackendRegistryError(
                    f"backend {self.name!r} conformance evidence claims undeclared model "
                    f"kind/family ({evidence.kind!r}, {evidence.family!r})",
                    code="evidence_overclaims_support",
                )


class BackendRegistry:
    """Immutable registry whose descriptors are validated without resolving factories."""

    def __init__(self, descriptors: Mapping[str, BackendDescriptor]) -> None:
        copied = dict(descriptors)
        for key, descriptor in copied.items():
            if key != descriptor.name:
                raise BackendRegistryError(
                    f"backend descriptor key {key!r} does not match descriptor name {descriptor.name!r}",
                    code="descriptor_name_mismatch",
                )
            descriptor.validate_definition()
        self._descriptors = MappingProxyType(copied)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def validate_static_registry(self) -> None:
        names = tuple(self._descriptors)
        if names != CANONICAL_BACKENDS:
            raise BackendRegistryError(
                f"static backend registry must contain exactly {list(CANONICAL_BACKENDS)}, got {list(names)}",
                code="invalid_static_registry",
            )

    def descriptor(self, backend_name: str) -> BackendDescriptor:
        try:
            return self._descriptors[backend_name]
        except KeyError as exc:
            raise BackendRegistryError(
                f"backend {backend_name!r} is not registered; registered backends: {list(self._descriptors)}",
                code="backend_not_registered",
            ) from exc

    def validate(
        self,
        context: RuntimeContext,
        *,
        allowed_deployments: frozenset[str] | None = None,
    ) -> BackendDescriptor:
        deployment = context.deployment
        descriptor = self.descriptor(deployment.backend)
        model = context.model
        if model.kind == "policy":
            policy_family = context.policy.policy_type
            if policy_family not in POLICY_FAMILIES:
                raise BackendCompatibilityError(
                    f"policy family {policy_family!r} is not recognized for deployment {context.deployment_name!r}",
                    code="unknown_policy_family",
                )
            if policy_family not in descriptor.supported_policy_families:
                raise BackendCompatibilityError(
                    f"backend {descriptor.name!r} does not support policy family {policy_family!r} "
                    f"for deployment {context.deployment_name!r}",
                    code="unsupported_policy_backend_pair",
                )
            kind, family = "policy", policy_family
        elif (
            model.kind not in descriptor.supported_model_kinds
            and model.family not in descriptor.supported_model_families
        ):
            raise BackendCompatibilityError(
                f"backend {descriptor.name!r} does not support model family {model.family!r} "
                f"(kind {model.kind!r}) for deployment {context.deployment_name!r}",
                code="unsupported_model_backend_pair",
            )
        else:
            kind, family = model.kind, model.family
        if (kind, family) not in descriptor.evidence_pairs:
            raise BackendCompatibilityError(
                f"backend {descriptor.name!r} declares support for family {family!r} (kind {kind!r}) "
                f"but lacks conformance evidence for deployment {context.deployment_name!r}",
                code="missing_conformance_evidence",
            )
        incompatibility = descriptor.target_validator(deployment)
        if incompatibility is not None:
            raise BackendCompatibilityError(
                f"deployment {context.deployment_name!r} is incompatible with backend {descriptor.name!r}: "
                f"{incompatibility}",
                code="incompatible_backend_target",
            )
        if allowed_deployments is not None and context.deployment_name not in allowed_deployments:
            raise BackendCompatibilityError(
                f"deployment {context.deployment_name!r} is not in the adapter supported deployments "
                f"{sorted(allowed_deployments)} for backend {descriptor.name!r}",
                code="adapter_deployment_mismatch",
            )
        return descriptor

    def create(self, context: RuntimeContext) -> InferenceBackend:
        descriptor = self.validate(context)
        factory = self._load_factory(descriptor)
        try:
            backend = factory(context)
        except Exception as exc:
            raise BackendRegistryError(
                f"factory {descriptor.factory!r} failed to create backend {descriptor.name!r}: {exc}",
                code="factory_failed",
            ) from exc
        if not isinstance(backend, InferenceBackend):
            raise BackendRegistryError(
                f"factory {descriptor.factory!r} did not return an InferenceBackend",
                code="invalid_factory_result",
            )
        if backend.name != descriptor.name:
            raise BackendRegistryError(
                f"factory {descriptor.factory!r} returned backend {backend.name!r}, expected {descriptor.name!r}",
                code="factory_backend_mismatch",
            )
        return backend

    @staticmethod
    def _load_factory(descriptor: BackendDescriptor) -> Callable[[RuntimeContext], InferenceBackend]:
        module_name, attribute = descriptor.factory.split(":", maxsplit=1)
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise BackendRegistryError(
                f"backend {descriptor.name!r} factory module {module_name!r} is unavailable: {exc}",
                code="factory_unavailable",
            ) from exc
        try:
            factory = getattr(module, attribute)
        except AttributeError as exc:
            raise BackendRegistryError(
                f"backend {descriptor.name!r} factory {descriptor.factory!r} is not defined",
                code="factory_unavailable",
            ) from exc
        if not callable(factory):
            raise BackendRegistryError(
                f"backend {descriptor.name!r} factory {descriptor.factory!r} is not callable",
                code="invalid_factory",
            )
        return factory


def _validate_torch(deployment: Deployment) -> str | None:
    if not isinstance(deployment, TorchDeployment):
        return "torch requires a native Torch deployment with a cpu, cuda, mps, or npu device"
    return None


def _validate_ascend(deployment: Deployment) -> str | None:
    if not isinstance(deployment, CompiledDeployment):
        return "ascend requires a compiled deployment"
    runtime = deployment.target.runtime
    if not (runtime.startswith("acl") or runtime.startswith("ascend")):
        return f"target.runtime {runtime!r} must identify the Ascend ACL runtime family"
    invalid_formats = sorted({deployment.artifacts[role].format for role in deployment.execution} - {"om"})
    if invalid_formats:
        return f"Ascend execution artifacts must use format 'om', got {invalid_formats}"
    return None


def _validate_hisilicon(deployment: Deployment) -> str | None:
    if not isinstance(deployment, CompiledDeployment):
        return "hisilicon requires a compiled deployment"
    if deployment.target.soc != "sd3403":
        return f"target.soc must be 'sd3403', got {deployment.target.soc!r}"
    if deployment.target.runtime != "hisilicon-worker":
        return f"target.runtime must be 'hisilicon-worker', got {deployment.target.runtime!r}"
    if deployment.execution != ("policy",):
        return f"execution must be ['policy'], got {list(deployment.execution)}"
    missing = sorted({"policy", "worker"} - set(deployment.artifacts))
    if missing:
        return f"required artifact roles are missing: {missing}"
    if deployment.artifacts["policy"].format != "om":
        return "policy artifact must use format 'om'"
    if deployment.artifacts["worker"].format != "executable":
        return "worker artifact must use format 'executable'"
    return None


def _validate_rknn(deployment: Deployment) -> str | None:
    if not isinstance(deployment, CompiledDeployment):
        return "rknn requires a compiled deployment"
    if not re.fullmatch(r"rk3588[a-z0-9_.-]*", deployment.target.soc):
        return f"target.soc {deployment.target.soc!r} is not in the RK3588 family"
    if not re.fullmatch(r"rknn(?:-lite(?:2)?|-toolkit-lite2)?", deployment.target.runtime):
        return f"target.runtime {deployment.target.runtime!r} is not an RKNN runtime"
    invalid_formats = sorted(
        {deployment.artifacts[role].format for role in deployment.execution if role != "embedding"} - {"rknn"}
    )
    if invalid_formats:
        return f"RKNN execution artifacts must use format 'rknn', got {invalid_formats}"
    if "embedding" in deployment.execution and deployment.artifacts["embedding"].format not in {"pt", "pytorch"}:
        return "RKNN embedding artifact must use format 'pt' or 'pytorch'"
    return None


def _validate_hmm(deployment: Deployment) -> str | None:
    if not isinstance(deployment, CompiledDeployment):
        return "hmm requires a compiled deployment"
    if not (deployment.target.runtime.startswith("hmm") or deployment.target.runtime.startswith("tcim")):
        return f"target.runtime {deployment.target.runtime!r} must identify an HMM or TCIM runtime"
    invalid_formats = sorted({artifact.format for artifact in deployment.artifacts.values()} - {"hmm", "pt", "pytorch"})
    if invalid_formats:
        return f"HMM artifacts must use format 'hmm', 'pt', or 'pytorch', got {invalid_formats}"
    return None


def _policy_evidence(*families: str) -> frozenset[ConformanceEvidence]:
    return frozenset(ConformanceEvidence("policy", family, reference="software-conformance") for family in families)


def _perception_evidence(*families: str) -> frozenset[ConformanceEvidence]:
    return frozenset(ConformanceEvidence("perception", family, reference="software-conformance") for family in families)


STATIC_BACKEND_DESCRIPTORS: Mapping[str, BackendDescriptor] = MappingProxyType(
    {
        "torch": BackendDescriptor(
            name="torch",
            factory="inference_service.backends.torch:create_backend",
            supported_policy_families=frozenset({"act", "diffusion", "pi05", "smolvla"}),
            supported_model_families=frozenset({"ram_plus", "sam2", "siglip2", "grounding_dino", "dummy_echo"}),
            conformance_evidence=_policy_evidence("act", "diffusion", "pi05", "smolvla")
            | _perception_evidence("ram_plus", "sam2", "siglip2", "grounding_dino", "dummy_echo"),
            target_validator=_validate_torch,
        ),
        "ascend": BackendDescriptor(
            name="ascend",
            factory="inference_service.backends.ascend:create_backend",
            supported_policy_families=frozenset({"act", "pi05"}),
            supported_model_families=frozenset({"ram_plus", "graspgen"}),
            conformance_evidence=_policy_evidence("act", "pi05") | _perception_evidence("ram_plus", "graspgen"),
            target_validator=_validate_ascend,
        ),
        "hisilicon": BackendDescriptor(
            name="hisilicon",
            factory="inference_service.backends.hisilicon:create_backend",
            supported_policy_families=frozenset({"act"}),
            conformance_evidence=_policy_evidence("act"),
            target_validator=_validate_hisilicon,
        ),
        "rknn": BackendDescriptor(
            name="rknn",
            factory="inference_service.backends.rknn:create_backend",
            supported_policy_families=frozenset({"act", "smolvla"}),
            conformance_evidence=_policy_evidence("act", "smolvla"),
            target_validator=_validate_rknn,
        ),
        "hmm": BackendDescriptor(
            name="hmm",
            factory="inference_service.backends.hmm:create_backend",
            supported_policy_families=frozenset({"pi05", "smolvla"}),
            conformance_evidence=_policy_evidence("pi05", "smolvla"),
            target_validator=_validate_hmm,
        ),
    }
)

BACKEND_REGISTRY = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
BACKEND_REGISTRY.validate_static_registry()
