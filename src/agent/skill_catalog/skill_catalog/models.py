"""Immutable data models for the skill catalog.

The dataclasses in this module are the pure-data contract between
``robot_config`` (hardware facts), ``skill_catalog`` (manifest/profile/digest)
and ``skill_library`` (runtime execution). ``skill_catalog`` owns these models
so it does not need to import either of the other packages; callers construct
the context objects and hand them to the compiler.

References: ``docs/lightweight_skill_package_registry_design_zh.md`` sections
5, 6, 9.1, 9.5, 10.2, 11.1 and 17.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_common.primitive_contracts import PrimitiveDescriptor
from skill_catalog.digest import (
    deep_freeze,
    derive_capability_digest,
    derive_capability_preimage,
    derive_provenance_digest,
    derive_provenance_preimage,
    derive_registry_digest,
    derive_registry_preimage,
    to_canonical_json,
)

# --------------------------------------------------------------------------- #
# Error hierarchy (section 17)                                                #
# --------------------------------------------------------------------------- #


class SkillCatalogError(Exception):
    """Base class for all skill catalog errors.

    Carries a stable ``code`` (from the v1 error vocabulary) plus optional
    ``source_relative_path`` and ``field_path`` so diagnostics can be produced
    without parsing free text.
    """

    code: str = "SKILL_SCHEMA_INVALID"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        source_relative_path: str | None = None,
        field_path: str | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.source_relative_path = source_relative_path
        self.field_path = field_path

    def diagnostic(self, *, severity: int = 1) -> SkillDiagnostic:
        return SkillDiagnostic(
            schema_version=1,
            severity=severity,
            error_code=self.code,
            source_relative_path=self.source_relative_path or "",
            field_path=self.field_path or "",
            message=str(self),
        )


class SkillSchemaError(SkillCatalogError):
    code = "SKILL_SCHEMA_INVALID"


class SkillProfileError(SkillCatalogError):
    code = "SKILL_PROFILE_NOT_FOUND"


class SkillReferenceError(SkillCatalogError):
    code = "SKILL_REFERENCE_MISSING"


class SkillRobotCompatibilityError(SkillCatalogError):
    code = "SKILL_LIMIT_VIOLATION"


class SkillRegistryError(SkillCatalogError):
    code = "SKILL_REGISTRY_NOT_READY"


class SkillCompileError(SkillCatalogError):
    def __init__(self, diagnostics: Sequence[SkillDiagnostic]) -> None:
        self.diagnostics = tuple(sort_diagnostics(list(diagnostics)))
        primary = next((diagnostic for diagnostic in self.diagnostics if diagnostic.severity == SEVERITY_ERROR), None)
        if primary is None:
            primary = SkillDiagnostic.error("SKILL_SCHEMA_INVALID", "catalog compilation failed")
        super().__init__(
            primary.message,
            code=primary.error_code,
            source_relative_path=primary.source_relative_path,
            field_path=primary.field_path,
        )


# --------------------------------------------------------------------------- #
# Diagnostics (section 11.1)                                                   #
# --------------------------------------------------------------------------- #

SEVERITY_ERROR = 1
SEVERITY_WARNING = 2


@dataclass(frozen=True)
class SkillDiagnostic:
    """Single structured diagnostic produced by the compiler/validator."""

    schema_version: int
    severity: int
    error_code: str
    source_relative_path: str
    field_path: str
    message: str

    @classmethod
    def error(
        cls,
        error_code: str,
        message: str,
        *,
        source_relative_path: str = "",
        field_path: str = "",
    ) -> SkillDiagnostic:
        return cls(
            schema_version=1,
            severity=SEVERITY_ERROR,
            error_code=error_code,
            source_relative_path=source_relative_path,
            field_path=field_path,
            message=message,
        )

    @classmethod
    def warning(
        cls,
        error_code: str,
        message: str,
        *,
        source_relative_path: str = "",
        field_path: str = "",
    ) -> SkillDiagnostic:
        return cls(
            schema_version=1,
            severity=SEVERITY_WARNING,
            error_code=error_code,
            source_relative_path=source_relative_path,
            field_path=field_path,
            message=message,
        )


def sort_diagnostics(diagnostics: list[SkillDiagnostic]) -> list[SkillDiagnostic]:
    """Deterministic diagnostic ordering (section 9.2 / 11.1)."""

    return sorted(
        diagnostics,
        key=lambda d: (d.source_relative_path, d.error_code, d.field_path, d.message),
    )


# --------------------------------------------------------------------------- #
# Compiler inputs (section 9.1)                                               #
# --------------------------------------------------------------------------- #

CONTEXT_SCHEMA_VERSION = 3

# Closed execution endpoint role sets by context schema version (section 9.1).
EXECUTION_ENDPOINT_ROLES_V1: frozenset[str] = frozenset(
    {
        "skill_action",
        "primitive_action",
        "validate_skill_service",
        "validate_primitive_service",
        "gateway_status_service",
        "begin_workflow_service",
        "finalize_workflow_service",
        "task_executor_action",
        "arm_trajectory_action",
        "move_configuration_service",
    }
)
EXECUTION_ENDPOINT_ROLES: frozenset[str] = EXECUTION_ENDPOINT_ROLES_V1 | {"navigation_action"}
EXECUTION_ENDPOINT_ROLES_BY_CONTEXT_VERSION: Mapping[int, frozenset[str]] = {
    1: EXECUTION_ENDPOINT_ROLES_V1,
    2: EXECUTION_ENDPOINT_ROLES,
    3: EXECUTION_ENDPOINT_ROLES,
}

# Closed set of dispatch kinds and the readiness capability they imply (5.1).
DISPATCH_KIND_CAPABILITY_V1: Mapping[str, str] = {
    "task_executor_action": "task_executor",
    "arm_trajectory_action": "arm_trajectory",
    "move_configuration_service": "move_configuration",
}
DISPATCH_KIND_CAPABILITY: Mapping[str, str] = {
    **DISPATCH_KIND_CAPABILITY_V1,
    "navigation_action": "navigation",
}
DISPATCH_KIND_CAPABILITY_BY_CONTEXT_VERSION: Mapping[int, Mapping[str, str]] = {
    1: DISPATCH_KIND_CAPABILITY_V1,
    2: DISPATCH_KIND_CAPABILITY,
    3: DISPATCH_KIND_CAPABILITY,
}

REQUIRED_RUNTIME_CAPABILITIES_V1: frozenset[str] = frozenset(
    {"validate_skill", "task_executor", "arm_trajectory", "fresh_ee_pose", "move_configuration"}
)
REQUIRED_RUNTIME_CAPABILITIES: frozenset[str] = REQUIRED_RUNTIME_CAPABILITIES_V1 | {"navigation"}
REQUIRED_RUNTIME_CAPABILITIES_BY_CONTEXT_VERSION: Mapping[int, frozenset[str]] = {
    1: REQUIRED_RUNTIME_CAPABILITIES_V1,
    2: REQUIRED_RUNTIME_CAPABILITIES,
    3: REQUIRED_RUNTIME_CAPABILITIES,
}

DELEGATED_ENDPOINT_KINDS: frozenset[str] = frozenset({"ros_action", "ros_service"})


@dataclass(frozen=True)
class SkillRobotContext:
    """Frozen robot execution context owned by ``robot_config``.

    ``skill_catalog`` only reads this object; it never imports the package that
    builds it. Field additions must bump ``context_schema_version``.
    """

    robot_name: str
    context_schema_version: int
    robot_config_digest: str
    named_poses: Mapping[str, Mapping[str, Any]]
    named_targets: Mapping[str, Mapping[str, Any]]
    arm_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, Mapping[str, float]]
    workspace_limits: Mapping[str, tuple[float, float]]
    required_control_mode: str
    timeout_policy: Mapping[str, float]
    relative_motion_reference_frame: str
    relative_motion_step_m: float
    relative_motion_direction_mapping: Mapping[str, tuple[float, float, float]]
    gripper_open_position: float
    gripper_closed_position: float
    execution_endpoints: Mapping[str, str]
    supported_control_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "named_poses",
            "named_targets",
            "joint_limits",
            "workspace_limits",
            "timeout_policy",
            "relative_motion_direction_mapping",
            "execution_endpoints",
        ):
            object.__setattr__(self, field_name, deep_freeze(getattr(self, field_name)))
        object.__setattr__(self, "arm_joint_names", tuple(self.arm_joint_names))
        object.__setattr__(self, "supported_control_modes", tuple(self.supported_control_modes))


@dataclass(frozen=True)
class DelegatedExecutorDescriptor:
    """Identity of a delegated executor (section 9.1).

    Model fields must be all-empty or all-non-empty. Non-model executors use
    empty strings for the three model fields.
    """

    name: str
    contract_version: str
    endpoint_kind: str
    endpoint_name: str
    configuration_digest: str
    model_deployment_name: str
    model_fingerprint: str
    model_bundle_digest: str


@dataclass(frozen=True)
class SkillCompileContext:
    """Pure-data inputs to the compiler (section 9.1)."""

    robot: SkillRobotContext
    primitive_contracts: Mapping[str, PrimitiveDescriptor]
    primitive_contract_digest: str
    delegated_executors: Mapping[str, DelegatedExecutorDescriptor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitive_contracts", deep_freeze(self.primitive_contracts))
        object.__setattr__(self, "delegated_executors", deep_freeze(self.delegated_executors))


# --------------------------------------------------------------------------- #
# Compiler outputs (section 9.5) / runtime bundle (section 10.2)              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompiledSkill:
    """A single compiled catalog entry (Atomic Operator or Skill)."""

    name: str
    version: str
    semantic_level: str
    skill_package_digest: str
    source_relative_path: str
    implementation_identity: str
    implementation_relative_path: str
    template: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "template", deep_freeze(self.template))


@dataclass(frozen=True)
class SkillSnapshot:
    """Immutable, fully-validated catalog snapshot (section 9.5).

    Digest and preimage fields are ``init=False`` derived values computed in
    ``__post_init__`` from the frozen core data; callers cannot supply a
    separate set of digests. Consumers re-derive the three digests from the
    published preimages and compare them to these values.
    """

    robot_name: str
    profile_name: str
    primitive_contract_digest: str
    robot_context: SkillRobotContext
    delegated_executors: Mapping[str, DelegatedExecutorDescriptor]
    templates: Mapping[str, Mapping[str, Any]]
    semantic_levels: Mapping[str, str]
    aliases: Mapping[str, tuple[str, ...]]
    parameter_schemas: Mapping[str, Mapping[str, Any]]
    requirements: Mapping[str, frozenset[str]]
    provenance: Mapping[str, Any]
    enabled_skill_names: tuple[str, ...]
    planner_visible_skill_names: tuple[str, ...]
    capability_view: Mapping[str, Any]

    registry_digest: str = field(init=False, default="")
    capability_digest: str = field(init=False, default="")
    provenance_digest: str = field(init=False, default="")
    registry_preimage_json: str = field(init=False, default="")
    capability_preimage_json: str = field(init=False, default="")
    provenance_preimage_json: str = field(init=False, default="")
    snapshot_json: str = field(init=False, default="")

    def __post_init__(self) -> None:
        # Recursively freeze mutable containers so callers only ever observe
        # read-only references (section 9.5).
        frozen = deep_freeze(
            {
                "robot_context": self.robot_context,
                "delegated_executors": self.delegated_executors,
                "templates": self.templates,
                "semantic_levels": self.semantic_levels,
                "aliases": self.aliases,
                "parameter_schemas": self.parameter_schemas,
                "requirements": self.requirements,
                "provenance": self.provenance,
                "capability_view": self.capability_view,
                "enabled_skill_names": self.enabled_skill_names,
                "planner_visible_skill_names": self.planner_visible_skill_names,
            }
        )
        object.__setattr__(self, "robot_context", frozen["robot_context"])
        object.__setattr__(self, "delegated_executors", frozen["delegated_executors"])
        object.__setattr__(self, "templates", frozen["templates"])
        object.__setattr__(self, "semantic_levels", frozen["semantic_levels"])
        object.__setattr__(self, "aliases", frozen["aliases"])
        object.__setattr__(self, "parameter_schemas", frozen["parameter_schemas"])
        object.__setattr__(self, "requirements", frozen["requirements"])
        object.__setattr__(self, "provenance", frozen["provenance"])
        object.__setattr__(self, "capability_view", frozen["capability_view"])
        object.__setattr__(self, "enabled_skill_names", tuple(frozen["enabled_skill_names"]))
        object.__setattr__(
            self,
            "planner_visible_skill_names",
            tuple(frozen["planner_visible_skill_names"]),
        )

        registry_preimage = derive_registry_preimage(
            robot_name=self.robot_name,
            profile_name=self.profile_name,
            primitive_contract_digest=self.primitive_contract_digest,
            robot_context=self.robot_context,
            delegated_executors=self.delegated_executors,
            templates=self.templates,
            semantic_levels=self.semantic_levels,
            aliases=self.aliases,
            parameter_schemas=self.parameter_schemas,
            requirements=self.requirements,
            capability_view=self.capability_view,
            enabled_skill_names=self.enabled_skill_names,
            planner_visible_skill_names=self.planner_visible_skill_names,
        )
        capability_preimage = derive_capability_preimage(
            robot_name=self.robot_name,
            profile_name=self.profile_name,
            capability_view=self.capability_view,
            enabled_skill_names=self.enabled_skill_names,
            planner_visible_skill_names=self.planner_visible_skill_names,
            named_pose_names=tuple(sorted(self.robot_context.named_poses.keys())),
            timeout_policy=self.robot_context.timeout_policy,
        )
        provenance_preimage = derive_provenance_preimage(self.provenance)

        registry_json = to_canonical_json(registry_preimage)
        capability_json = to_canonical_json(capability_preimage)
        provenance_json = to_canonical_json(provenance_preimage)

        object.__setattr__(self, "registry_preimage_json", registry_json)
        object.__setattr__(self, "capability_preimage_json", capability_json)
        object.__setattr__(self, "provenance_preimage_json", provenance_json)
        object.__setattr__(self, "registry_digest", derive_registry_digest(registry_preimage))
        object.__setattr__(self, "capability_digest", derive_capability_digest(capability_preimage))
        object.__setattr__(self, "provenance_digest", derive_provenance_digest(provenance_preimage))

        snapshot_payload = {
            "schema_version": 1,
            "registry_preimage": registry_preimage,
            "capability_preimage": capability_preimage,
            "provenance_preimage": provenance_preimage,
        }
        object.__setattr__(self, "snapshot_json", to_canonical_json(snapshot_payload))


@dataclass(frozen=True)
class GatewayPolicyView:
    """Generation-specific immutable admission rules (section 10.2).

    Holds only read-only indexes into the frozen snapshot; it never owns
    active leases, request ledgers or busy state.
    """

    enabled_skill_names: tuple[str, ...]
    planner_visible_skill_names: tuple[str, ...]
    requirements: Mapping[str, frozenset[str]]
    timeout_policy: Mapping[str, float]
    capability_view: Mapping[str, Any]
    parameter_schemas: Mapping[str, Mapping[str, Any]]
    semantic_levels: Mapping[str, str]

    @classmethod
    def from_snapshot(cls, snapshot: SkillSnapshot) -> GatewayPolicyView:
        return cls(
            enabled_skill_names=snapshot.enabled_skill_names,
            planner_visible_skill_names=snapshot.planner_visible_skill_names,
            requirements=snapshot.requirements,
            timeout_policy=snapshot.robot_context.timeout_policy,
            capability_view=snapshot.capability_view,
            parameter_schemas=snapshot.parameter_schemas,
            semantic_levels=snapshot.semantic_levels,
        )


@dataclass(frozen=True)
class SkillRuntimeBundle:
    """Immutable runtime bundle (section 10.2).

    ``registry_epoch`` and ``generation`` are runtime-only identity; they do
    NOT enter any content digest (section 6.4).
    """

    registry_epoch: str
    generation: int
    snapshot: SkillSnapshot
    gateway_policy_view: GatewayPolicyView

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SkillSnapshot,
        *,
        registry_epoch: str,
        generation: int,
    ) -> SkillRuntimeBundle:
        return cls(
            registry_epoch=registry_epoch,
            generation=generation,
            snapshot=snapshot,
            gateway_policy_view=GatewayPolicyView.from_snapshot(snapshot),
        )
