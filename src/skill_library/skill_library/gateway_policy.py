"""ROS-independent policy objects for Gateway skill admission and idempotency."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import Any

from embodied_common import skill_request
from embodied_common.primitive_contracts import PRIMITIVE_DESCRIPTORS

MOTION_NOT_AUTHORIZED = "MOTION_NOT_AUTHORIZED"
CONTROL_MODE_MISMATCH = "CONTROL_MODE_MISMATCH"
SKILL_BUSY = "SKILL_BUSY"
TIMEOUT_EXCEEDS_POLICY = "TIMEOUT_EXCEEDS_POLICY"
CAPABILITY_NOT_READY = "CAPABILITY_NOT_READY"
DUPLICATE_TASK_ID = "DUPLICATE_TASK_ID"
TASK_ID_CONFLICT = "TASK_ID_CONFLICT"
INVALID_ARGUMENT = "INVALID_ARGUMENT"
SKILL_REJECTED = "SKILL_REJECTED"
GATEWAY_FINALIZATION_FAILED = "GATEWAY_FINALIZATION_FAILED"

_ROOT_OWNER_KINDS = {"skill_command", "workflow", "external_primitive"}
_CAPABILITY_ORDER = (
    "validate_skill",
    "task_executor",
    "arm_trajectory",
    "fresh_ee_pose",
    "navigation",
)
_CAPABILITY_UNAVAILABLE_MESSAGES = {
    "validate_skill": "validate skill service unavailable",
    "task_executor": "task executor action unavailable",
    "arm_trajectory": "arm trajectory action unavailable",
    "fresh_ee_pose": "ee pose unavailable or stale",
    "navigation": "navigation action unavailable",
}
_REQUEST_PARAMETER_FIELDS = (
    "target_name",
    "container_name",
    "place_name",
    "motion_direction",
    "motion_distance",
    "direction",
    "distance",
    "degree",
    "x",
    "y",
    "yaw",
)

__all__ = [
    "BoundedRequestLedger",
    "CAPABILITY_NOT_READY",
    "CONTROL_MODE_MISMATCH",
    "DUPLICATE_TASK_ID",
    "ExecutionOwner",
    "GATEWAY_FINALIZATION_FAILED",
    "GatewayDecision",
    "GatewayPolicy",
    "GatewayRequest",
    "INVALID_ARGUMENT",
    "LedgerQuery",
    "LedgerRecord",
    "MOTION_NOT_AUTHORIZED",
    "PreparedRequest",
    "ReadinessReason",
    "RequestIdentity",
    "RootExecutionLease",
    "RuntimeSnapshot",
    "SKILL_BUSY",
    "SKILL_REJECTED",
    "SkillRequirements",
    "TASK_ID_CONFLICT",
    "TIMEOUT_EXCEEDS_POLICY",
    "build_request_identity",
    "build_skill_parameter_schemas",
    "build_skill_requirements",
]


@dataclass(frozen=True)
class GatewayRequest:
    """The normalized-input boundary for a Gateway skill request."""

    task_id: Any
    skill_name: Any
    schema_version: Any = 1
    target_name: Any = None
    container_name: Any = None
    place_name: Any = None
    motion_direction: Any = None
    motion_distance: Any = None
    direction: Any = None
    distance: Any = None
    degree: Any = None
    x: Any = None
    y: Any = None
    yaw: Any = None
    has_x: Any = None
    has_y: Any = None
    has_yaw: Any = None
    timeout_sec: Any = None


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Runtime state consumed by policy without importing ROS types."""

    motion_authorized: bool
    active_control_mode: str
    required_control_mode: str
    busy: bool = False
    active_task_id: str = ""
    validate_ready: bool = False
    task_executor_ready: bool = False
    arm_trajectory_ready: bool = False
    ee_pose_fresh: bool = False
    navigation_ready: bool = False
    control_mode_switching_enabled: bool = False

    @property
    def is_busy(self) -> bool:
        """Whether the runtime currently has an active skill execution."""
        return self.busy or bool(self.active_task_id)


@dataclass(frozen=True)
class SkillRequirements:
    """Private runtime capabilities required by an expanded skill template."""

    validate_skill: bool = False
    task_executor: bool = False
    arm_trajectory: bool = False
    fresh_ee_pose: bool = False
    navigation: bool = False


@dataclass(frozen=True)
class ReadinessReason:
    """A stable, ordered explanation of a skill's capability readiness."""

    required: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Return whether every required capability is available."""
        return not self.unavailable


@dataclass(frozen=True)
class RequestIdentity:
    """A task identifier paired with the canonical shared-payload digest."""

    task_id: str
    payload_hash: str


@dataclass(frozen=True)
class PreparedRequest:
    """A request after canonical payload resolution by embodied_common."""

    identity: RequestIdentity
    payload: Mapping[str, str | float | bool | None]
    effective_timeout_sec: float


@dataclass(frozen=True)
class GatewayDecision:
    """The result of evaluating a skill request against one runtime snapshot."""

    admitted: bool
    error_code: str = ""
    message: str = ""
    effective_timeout_sec: float = 0.0
    readiness: ReadinessReason = field(default_factory=ReadinessReason)
    prepared_request: PreparedRequest | None = None
    lease_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ExecutionOwner:
    """The root command or external primitive owning an execution lease."""

    root_task_id: str
    kind: str
    root_kind: str = ""
    name: str = ""

    @classmethod
    def skill_command(cls, task_id: str) -> ExecutionOwner:
        """Create the owner for a root SkillCommand."""
        return cls(root_task_id=task_id, kind="skill_command")

    @classmethod
    def external_primitive(cls, task_id: str) -> ExecutionOwner:
        """Create the owner for a primitive requested outside a SkillCommand."""
        return cls(root_task_id=task_id, kind="external_primitive")

    @classmethod
    def workflow(cls, task_id: str) -> ExecutionOwner:
        """Create the owner for one typed Workflow root."""
        return cls(root_task_id=task_id, kind="workflow")

    @classmethod
    def internal_child(cls, root_owner: ExecutionOwner, name: str) -> ExecutionOwner:
        """Create a child that may reuse its root owner's active lease."""
        if root_owner.kind not in _ROOT_OWNER_KINDS:
            raise ValueError("internal child requires a root execution owner")
        return cls(
            root_task_id=root_owner.root_task_id,
            kind="internal_child",
            root_kind=root_owner.kind,
            name=name,
        )

    def is_internal_child_of(self, root_owner: ExecutionOwner) -> bool:
        """Return whether this owner is a child of the supplied root execution."""
        return (
            self.kind == "internal_child"
            and self.root_task_id == root_owner.root_task_id
            and self.root_kind == root_owner.kind
        )


class _ExecutionLeaseToken:
    """Token constructible only by RootExecutionLease."""

    def __init__(self, factory: object, expected_factory: object) -> None:
        if factory is not expected_factory:
            raise TypeError("execution lease tokens are created by RootExecutionLease")


class _ExecutionLeaseBorrow:
    """Non-releasable capability issued to an internal child primitive."""

    def __init__(self, factory: object, expected_factory: object) -> None:
        if factory is not expected_factory:
            raise TypeError("execution lease borrows are created by RootExecutionLease")


class RootExecutionLease:
    """Thread-safe, in-process exclusive lease shared by root Gateway executions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._token_factory = object()
        self._borrow_factory = object()
        self._owner: ExecutionOwner | None = None
        self._token: _ExecutionLeaseToken | None = None

    @property
    def owner(self) -> ExecutionOwner | None:
        """Return the root owner currently holding the lease."""
        with self._lock:
            return self._owner

    def acquire(self, owner: ExecutionOwner) -> object | None:
        """Acquire a new opaque token, or return the token held by the same owner object."""
        with self._lock:
            if self._owner is None:
                if owner.kind not in _ROOT_OWNER_KINDS:
                    return None
                self._owner = owner
                self._token = _ExecutionLeaseToken(self._token_factory, self._token_factory)
                return self._token
            if owner is self._owner:
                return self._token
            return None

    def reuse(self, owner: ExecutionOwner, token: object) -> object | None:
        """Issue a non-releasable borrow to an internal child of the current root owner."""
        with self._lock:
            if token is not self._token or self._owner is None:
                return None
            if owner.is_internal_child_of(self._owner):
                return _ExecutionLeaseBorrow(self._borrow_factory, self._borrow_factory)
            return None

    def release(self, token: object) -> bool:
        """Release only when the caller presents the current opaque token object."""
        with self._lock:
            if token is not self._token:
                return False
            self._token = None
            self._owner = None
            return True

    def holds(self, token: object) -> bool:
        """Return whether token is the token currently held by this lease."""
        with self._lock:
            return token is self._token


class BoundedRequestLedger:
    """Thread-safe active tracking with FIFO-bounded terminal history."""

    def __init__(self, terminal_capacity: int) -> None:
        if isinstance(terminal_capacity, bool) or not isinstance(terminal_capacity, int) or terminal_capacity <= 0:
            raise ValueError("terminal_capacity must be a finite positive integer")
        self._lock = RLock()
        self._terminal_capacity = terminal_capacity
        self._active: dict[str, LedgerRecord] = {}
        self._terminal: OrderedDict[str, LedgerRecord] = OrderedDict()

    def begin(self, task_id: str, payload_hash: str) -> LedgerRecord:
        """Record an active task, returning its existing record for the same identity."""
        with self._lock:
            existing = self._record_for(task_id)
            if existing is not None:
                if existing.payload_hash == payload_hash:
                    return existing
                raise ValueError("task_id is already recorded with a different payload hash")
            record = LedgerRecord(task_id=task_id, payload_hash=payload_hash, state="active")
            self._active[task_id] = record
            return record

    def begin_request(self, request: GatewayRequest, *, default_timeout_sec: float) -> LedgerRecord:
        """Create identity through embodied_common before recording an active request."""
        prepared = build_request_identity(request, default_timeout_sec=default_timeout_sec)
        return self.begin(prepared.identity.task_id, prepared.identity.payload_hash)

    def terminal(
        self,
        task_id: str,
        payload_hash: str,
        *,
        error_code: str = "",
        terminal_metadata: Mapping[str, Any] | None = None,
    ) -> LedgerRecord:
        """Move one active request to FIFO-bounded terminal history."""
        with self._lock:
            active = self._active.get(task_id)
            if active is None:
                existing_terminal = self._terminal.get(task_id)
                if existing_terminal is not None and existing_terminal.payload_hash == payload_hash:
                    return existing_terminal
                raise ValueError("cannot terminalize a task that is not active")
            if active.payload_hash != payload_hash:
                raise ValueError("task_id is active with a different payload hash")

            record = LedgerRecord(
                task_id=task_id,
                payload_hash=payload_hash,
                state="terminal",
                error_code=error_code,
                terminal_metadata=MappingProxyType(dict(terminal_metadata or {})),
            )
            del self._active[task_id]
            self._terminal[task_id] = record
            if len(self._terminal) > self._terminal_capacity:
                self._terminal.popitem(last=False)
            return record

    def record(
        self,
        task_id: str,
        payload_hash: str,
        state: str,
        *,
        error_code: str = "",
        terminal_metadata: Mapping[str, Any] | None = None,
    ) -> LedgerRecord:
        """Record an active or terminal state for testable executor lifecycle use.

        Warning: this helper bypasses GatewayPolicy.admit/finalize and therefore
        does not coordinate the RootExecutionLease. Business code must drive the
        ledger through GatewayPolicy.admit/finalize so the lease and ledger stay
        atomic; this method exists only for focused unit tests of the executor
        lifecycle that do not need lease semantics.
        """
        with self._lock:
            if state == "active":
                return self.begin(task_id, payload_hash)
            if state == "terminal":
                if self._record_for(task_id) is None:
                    self.begin(task_id, payload_hash)
                return self.terminal(
                    task_id,
                    payload_hash,
                    error_code=error_code,
                    terminal_metadata=terminal_metadata,
                )
            raise ValueError("state must be 'active' or 'terminal'")

    def query(self, task_id: str = "", payload_hash: str = "") -> LedgerQuery:
        """Return the specified task/hash query matrix without exposing terminal errors."""
        with self._lock:
            if not task_id:
                return LedgerQuery(error_code=INVALID_ARGUMENT) if payload_hash else LedgerQuery()

            record = self._record_for(task_id)
            if record is None:
                return LedgerQuery()
            if not payload_hash:
                return LedgerQuery(state=record.state)
            if payload_hash == record.payload_hash:
                return LedgerQuery(state=record.state, error_code=DUPLICATE_TASK_ID)
            return LedgerQuery(state=record.state, error_code=TASK_ID_CONFLICT)

    def get(self, task_id: str) -> LedgerRecord | None:
        """Return the immutable record for internal replay bookkeeping."""
        with self._lock:
            return self._record_for(task_id)

    def _record_for(self, task_id: str) -> LedgerRecord | None:
        return self._active.get(task_id) or self._terminal.get(task_id)


@dataclass(frozen=True)
class LedgerRecord:
    """The minimal active or terminal record retained for one task identifier."""

    task_id: str
    payload_hash: str
    state: str
    error_code: str = ""
    terminal_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LedgerQuery:
    """The externally stable state/error pair for an idempotency lookup."""

    state: str = ""
    error_code: str = ""


def build_skill_requirements(
    expanded_skill_templates: Mapping[str, Any],
    primitive_descriptors: Mapping[str, Any] | None = None,
) -> dict[str, SkillRequirements]:
    """Build private capability requirements from already-expanded skill templates."""
    if not isinstance(expanded_skill_templates, Mapping):
        raise ValueError("expanded_skill_templates must be a mapping")
    descriptors = PRIMITIVE_DESCRIPTORS if primitive_descriptors is None else primitive_descriptors
    if not isinstance(descriptors, Mapping):
        raise ValueError("primitive_descriptors must be a mapping")

    requirements: dict[str, SkillRequirements] = {}
    for skill_name, template in expanded_skill_templates.items():
        if not isinstance(template, Mapping):
            raise ValueError(f"skill template '{skill_name}' must be a mapping")
        merged: dict[str, bool] = {
            "validate_skill": bool(str(template.get("initial_gripper_state", "")).strip()),
            "task_executor": bool(str(template.get("initial_gripper_state", "")).strip()),
            "arm_trajectory": False,
            "fresh_ee_pose": False,
            "navigation": False,
        }
        primitive_sequence = template.get("primitive_sequence", [])
        if not isinstance(primitive_sequence, list):
            raise ValueError(f"skill template '{skill_name}' primitive_sequence must be a list")

        for step in primitive_sequence:
            if not isinstance(step, Mapping):
                raise ValueError(f"skill template '{skill_name}' contains a non-object step")
            primitive_name = str(step.get("primitive_name", "")).strip()
            descriptor = descriptors.get(primitive_name)
            if descriptor is None:
                raise ValueError(f"skill template '{skill_name}' uses unknown primitive '{primitive_name}'")
            for field_name in descriptor.required_runtime_capabilities:
                if field_name in merged:
                    merged[field_name] = True

        requirements[str(skill_name)] = SkillRequirements(
            validate_skill=merged["validate_skill"],
            task_executor=merged["task_executor"],
            arm_trajectory=merged["arm_trajectory"],
            fresh_ee_pose=merged["fresh_ee_pose"],
            navigation=merged["navigation"],
        )
    return requirements


def build_skill_parameter_schemas(expanded_skill_templates: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Extract normalized public parameter schemas from expanded skill templates."""
    if not isinstance(expanded_skill_templates, Mapping):
        raise ValueError("expanded_skill_templates must be a mapping")

    schemas: dict[str, Mapping[str, Any]] = {}
    for skill_name, template in expanded_skill_templates.items():
        if not isinstance(template, Mapping):
            raise ValueError(f"skill template '{skill_name}' must be a mapping")
        capability = template.get("capability")
        if not isinstance(capability, Mapping):
            raise ValueError(f"skill template '{skill_name}' capability must be a mapping")
        parameters = capability.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"skill template '{skill_name}' capability.parameters must be a mapping")
        schemas[str(skill_name)] = MappingProxyType(copy.deepcopy(dict(parameters)))
    return schemas


def build_request_identity(request: GatewayRequest, *, default_timeout_sec: float) -> PreparedRequest:
    """Use embodied_common's canonical payload and hash functions to prepare a request."""
    if not isinstance(request.task_id, str):
        raise ValueError("task_id must be a string")
    task_id = request.task_id.strip()
    if not task_id:
        raise ValueError("task_id must be non-empty")
    # Reuse the same normalized value to guarantee UUID/ledger consistency.
    skill_request.skill_goal_uuid(task_id)
    payload = skill_request.canonical_skill_payload(
        request.skill_name,
        target_name=request.target_name,
        container_name=request.container_name,
        place_name=request.place_name,
        motion_direction=request.motion_direction,
        motion_distance=request.motion_distance,
        direction=request.direction,
        distance=request.distance,
        degree=request.degree,
        x=request.x,
        y=request.y,
        yaw=request.yaw,
        has_x=request.has_x,
        has_y=request.has_y,
        has_yaw=request.has_yaw,
        timeout_sec=request.timeout_sec,
        schema_version=request.schema_version,
        default_timeout_sec=default_timeout_sec,
    )
    payload_hash = skill_request.skill_payload_hash(payload)
    immutable_payload = MappingProxyType(dict(payload))
    effective_timeout_sec = float(immutable_payload["timeout_sec"])
    return PreparedRequest(
        identity=RequestIdentity(task_id=task_id, payload_hash=payload_hash),
        payload=immutable_payload,
        effective_timeout_sec=effective_timeout_sec,
    )


class GatewayPolicy:
    """Evaluate Gateway admission with optional process-local atomic execution transitions."""

    def __init__(
        self,
        timeout_policy: Mapping[str, Any],
        skill_requirements: Mapping[str, SkillRequirements],
        *,
        parameter_schemas: Mapping[str, Mapping[str, Any]],
        skill_timeout_caps: Mapping[str, float] | None = None,
        skill_schema_versions: Mapping[str, int] | None = None,
        skill_control_modes: Mapping[str, str] | None = None,
        ledger: BoundedRequestLedger | None = None,
        lease: RootExecutionLease | None = None,
    ) -> None:
        if not isinstance(timeout_policy, Mapping):
            raise ValueError("timeout_policy must be a mapping")
        self._default_timeout_sec = _finite_positive(
            timeout_policy.get("default_skill_timeout_sec"),
            "timeout_policy.default_skill_timeout_sec",
        )
        self._task_budget_sec = _finite_positive(
            timeout_policy.get("task_budget_sec"),
            "timeout_policy.task_budget_sec",
        )
        if self._default_timeout_sec > self._task_budget_sec:
            raise ValueError("timeout_policy.default_skill_timeout_sec must be <= task_budget_sec")
        self._skill_requirements = dict(skill_requirements)
        if not isinstance(parameter_schemas, Mapping):
            raise ValueError("parameter_schemas must be a mapping")
        self._parameter_schemas = dict(parameter_schemas)
        self._skill_timeout_caps = self._validate_timeout_caps(skill_timeout_caps or {})
        self._skill_schema_versions = self._validate_schema_versions(skill_schema_versions or {})
        self._skill_control_modes = {str(name): str(mode) for name, mode in (skill_control_modes or {}).items()}
        self._ledger = ledger
        self._lease = lease
        self._transition_lock = RLock()
        self._active_admission: GatewayDecision | None = None

    def prepare(self, request: GatewayRequest) -> PreparedRequest:
        """Resolve a request's canonical identity and effective timeout."""
        skill_name = str(request.skill_name).strip()
        timeout_cap = self._skill_timeout_caps.get(skill_name, self._default_timeout_sec)
        return build_request_identity(request, default_timeout_sec=timeout_cap)

    def replace_catalog(
        self,
        timeout_policy: Mapping[str, Any],
        skill_requirements: Mapping[str, SkillRequirements],
        *,
        parameter_schemas: Mapping[str, Mapping[str, Any]],
        skill_timeout_caps: Mapping[str, float] | None = None,
        skill_schema_versions: Mapping[str, int] | None = None,
        skill_control_modes: Mapping[str, str] | None = None,
    ) -> None:
        """Replace immutable admission indexes for the next root request.

        Active admissions retain their prepared identity and lease.  The node
        must keep their execution template separately until finalization.
        """
        default_timeout = _finite_positive(
            timeout_policy.get("default_skill_timeout_sec"),
            "timeout_policy.default_skill_timeout_sec",
        )
        task_budget = _finite_positive(timeout_policy.get("task_budget_sec"), "timeout_policy.task_budget_sec")
        if default_timeout > task_budget:
            raise ValueError("timeout_policy.default_skill_timeout_sec must be <= task_budget_sec")
        if not isinstance(parameter_schemas, Mapping):
            raise ValueError("parameter_schemas must be a mapping")
        with self._transition_lock:
            self._default_timeout_sec = default_timeout
            self._task_budget_sec = task_budget
            self._skill_requirements = dict(skill_requirements)
            self._parameter_schemas = dict(parameter_schemas)
            self._skill_timeout_caps = self._validate_timeout_caps(skill_timeout_caps or {})
            self._skill_schema_versions = self._validate_schema_versions(skill_schema_versions or {})
            self._skill_control_modes = {str(name): str(mode) for name, mode in (skill_control_modes or {}).items()}

    def _validate_timeout_caps(self, values: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError("skill_timeout_caps must be a mapping")
        caps = {str(name): _finite_positive(value, f"skill_timeout_caps.{name}") for name, value in values.items()}
        if any(value > self._task_budget_sec for value in caps.values()):
            raise ValueError("skill timeout cap must not exceed task_budget_sec")
        return caps

    @staticmethod
    def _validate_schema_versions(values: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(values, Mapping):
            raise ValueError("skill_schema_versions must be a mapping")
        versions: dict[str, int] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2}:
                raise ValueError(f"skill_schema_versions.{name} must be 1 or 2")
            versions[str(name)] = value
        return versions

    def evaluate(
        self,
        request: GatewayRequest,
        snapshot: RuntimeSnapshot,
        *,
        owner: ExecutionOwner | None = None,
        validate_parameters: bool = True,
    ) -> GatewayDecision:
        """Evaluate admission with fixed motion, mode, busy, timeout, readiness priority."""
        try:
            prepared = self.prepare(request)
            skill_name = str(prepared.payload["skill_name"])
            expected_schema_version = self._skill_schema_versions.get(skill_name, 1)
            if prepared.payload["schema_version"] != expected_schema_version:
                return GatewayDecision(
                    admitted=False,
                    error_code="SKILL_SCHEMA_INVALID",
                    message=(f"skill '{skill_name}' requires schema_version {expected_schema_version}"),
                    effective_timeout_sec=prepared.effective_timeout_sec,
                    prepared_request=prepared,
                )
            requirements = self._requirements_for(prepared)
            if validate_parameters:
                self._validate_parameters(prepared)
        except ValueError as exc:
            return GatewayDecision(
                admitted=False,
                error_code=SKILL_REJECTED,
                message=str(exc) or "skill request rejected",
            )
        readiness = _readiness_reason(requirements, snapshot)
        decision_args = {
            "effective_timeout_sec": prepared.effective_timeout_sec,
            "readiness": readiness,
            "prepared_request": prepared,
        }

        if not snapshot.motion_authorized:
            return GatewayDecision(
                admitted=False,
                error_code=MOTION_NOT_AUTHORIZED,
                message="operator authorization is disabled",
                **decision_args,
            )
        if (
            not snapshot.control_mode_switching_enabled
            and snapshot.active_control_mode != snapshot.required_control_mode
        ):
            return GatewayDecision(
                admitted=False,
                error_code=CONTROL_MODE_MISMATCH,
                message=(f"requires {snapshot.required_control_mode}, active mode is {snapshot.active_control_mode}"),
                **decision_args,
            )
        if snapshot.is_busy and not _is_active_owner(owner, snapshot.active_task_id):
            return GatewayDecision(
                admitted=False,
                error_code=SKILL_BUSY,
                message="another root execution is active",
                **decision_args,
            )
        timeout_cap = self._skill_timeout_caps.get(str(prepared.payload["skill_name"]), self._task_budget_sec)
        if prepared.effective_timeout_sec > timeout_cap or prepared.effective_timeout_sec > self._task_budget_sec:
            return GatewayDecision(admitted=False, error_code=TIMEOUT_EXCEEDS_POLICY, **decision_args)
        if not readiness.ready:
            return GatewayDecision(
                admitted=False,
                error_code=CAPABILITY_NOT_READY,
                message=_capability_unavailable_message(readiness),
                **decision_args,
            )
        return GatewayDecision(admitted=True, **decision_args)

    def admit(
        self,
        request: GatewayRequest,
        snapshot: RuntimeSnapshot,
        owner: ExecutionOwner,
    ) -> GatewayDecision:
        """Atomically evaluate, reserve the in-process lease, and record an active request.

        The lock only coordinates callers within this Gateway process. It is not a ROS graph lock.
        """
        ledger, lease = self._atomic_resources()
        with self._transition_lock:
            decision = self.evaluate(request, snapshot, owner=owner)
            if not decision.admitted:
                return decision
            prepared = decision.prepared_request
            if prepared is None:
                raise ValueError("admitted Gateway decision must include a prepared request")
            if owner.root_task_id != prepared.identity.task_id:
                return replace(decision, admitted=False, error_code=SKILL_REJECTED)

            duplicate = ledger.query(prepared.identity.task_id, prepared.identity.payload_hash)
            if duplicate.error_code:
                return replace(decision, admitted=False, error_code=duplicate.error_code)

            token = lease.acquire(owner)
            if token is None:
                return replace(
                    decision,
                    admitted=False,
                    error_code=SKILL_BUSY,
                    message="another root execution is active",
                )
            try:
                ledger.begin(prepared.identity.task_id, prepared.identity.payload_hash)
            except Exception:
                lease.release(token)
                raise
            admission = replace(decision, lease_token=token)
            self._active_admission = admission
            return admission

    def finalize(
        self,
        admission: GatewayDecision,
        *,
        error_code: str = "",
        terminal_metadata: Mapping[str, Any] | None = None,
    ) -> LedgerRecord:
        """Atomically write terminal state and release the admission's opaque lease token."""
        ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if not isinstance(admission, GatewayDecision) or admission is not self._active_admission:
                raise ValueError("finalize requires the original active admission")
            prepared = admission.prepared_request
            token = admission.lease_token
            if not admission.admitted or prepared is None or token is None:
                raise ValueError("finalize requires an admitted Gateway decision with a lease token")
            if not lease.holds(token):
                raise ValueError("finalize requires the currently held admission lease token")
            try:
                terminal = ledger.terminal(
                    prepared.identity.task_id,
                    prepared.identity.payload_hash,
                    error_code=error_code,
                    terminal_metadata=terminal_metadata,
                )
            except Exception as exc:
                raise RuntimeError("Gateway terminalization failed; root lease remains held") from exc
            if not lease.release(token):
                raise RuntimeError("Gateway terminalized request but could not release root lease")
            self._active_admission = None
            return terminal

    def borrow_internal(
        self,
        admission: GatewayDecision,
        task_id: str,
        child_name: str,
    ) -> object | None:
        """Issue a non-releasable borrow only for the exact active root admission."""
        _ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if not isinstance(admission, GatewayDecision) or admission is not self._active_admission:
                return None
            prepared = admission.prepared_request
            token = admission.lease_token
            root_owner = lease.owner
            if (
                not admission.admitted
                or prepared is None
                or token is None
                or root_owner is None
                or root_owner.kind != "skill_command"
                or prepared.identity.task_id != task_id
            ):
                return None
            return lease.reuse(ExecutionOwner.internal_child(root_owner, child_name), token)

    def admit_external_primitive(
        self, task_id: str, payload_hash: str | RuntimeSnapshot, snapshot: RuntimeSnapshot | None = None
    ) -> tuple[str, object | None]:
        """Atomically ledger and lease an external primitive root."""
        if snapshot is None:
            snapshot = payload_hash
            payload_hash = f"external:{task_id}"
        if not isinstance(snapshot, RuntimeSnapshot):
            raise TypeError("snapshot is required")
        ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if not snapshot.motion_authorized:
                return MOTION_NOT_AUTHORIZED, None
            if (
                not snapshot.control_mode_switching_enabled
                and snapshot.active_control_mode != snapshot.required_control_mode
            ):
                return CONTROL_MODE_MISMATCH, None
            if lease.owner is not None:
                return SKILL_BUSY, None
            if snapshot.is_busy:
                return SKILL_BUSY, None
            duplicate = ledger.query(task_id, payload_hash)
            if duplicate.error_code:
                return duplicate.error_code, None
            token = lease.acquire(ExecutionOwner.external_primitive(task_id))
            if token is None:
                return SKILL_BUSY, None
            try:
                ledger.begin(task_id, payload_hash)
            except Exception:
                lease.release(token)
                raise
            return "", token

    def terminal_external_primitive(
        self, task_id: str, payload_hash: str, token: object, *, error_code: str, terminal_metadata=None
    ) -> LedgerRecord:
        """Record external terminal identity while deliberately retaining the lease."""
        ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if not lease.holds(token):
                raise ValueError("external primitive lease is not active")
            return ledger.terminal(
                task_id,
                payload_hash,
                error_code=error_code,
                terminal_metadata=terminal_metadata,
            )

    def admit_workflow(
        self,
        owner: ExecutionOwner,
        snapshot: RuntimeSnapshot,
        *,
        timeout_sec: float,
    ) -> tuple[str, object | None]:
        """Atomically acquire the root lease for a prevalidated typed Workflow."""
        _ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if owner.kind != "workflow" or not owner.root_task_id:
                return SKILL_REJECTED, None
            if not snapshot.motion_authorized:
                return MOTION_NOT_AUTHORIZED, None
            if (
                not snapshot.control_mode_switching_enabled
                and snapshot.active_control_mode != snapshot.required_control_mode
            ):
                return CONTROL_MODE_MISMATCH, None
            if snapshot.is_busy and not _is_active_owner(owner, snapshot.active_task_id):
                return SKILL_BUSY, None
            if timeout_sec > self._task_budget_sec:
                return TIMEOUT_EXCEEDS_POLICY, None
            token = lease.acquire(owner)
            if token is None:
                return SKILL_BUSY, None
            return "", token

    def borrow_workflow_internal(
        self,
        root_owner: ExecutionOwner,
        lease_token: object,
        task_id: str,
        child_name: str,
    ) -> object | None:
        """Issue a non-releasable borrow for one Workflow child dispatch."""
        _ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if root_owner.kind != "workflow" or lease.owner is not root_owner or not task_id:
                return None
            return lease.reuse(ExecutionOwner.internal_child(root_owner, child_name), lease_token)

    def release_workflow(self, owner: ExecutionOwner, lease_token: object) -> bool:
        """Release only the exact currently active Workflow root lease."""
        _ledger, lease = self._atomic_resources()
        with self._transition_lock:
            if owner.kind != "workflow" or lease.owner is not owner:
                return False
            return lease.release(lease_token)

    def release_external_primitive(self, token: object) -> bool:
        """Release a successfully cleaned-up external primitive lease."""
        _ledger, lease = self._atomic_resources()
        with self._transition_lock:
            return lease.release(token)

    def _requirements_for(self, prepared: PreparedRequest) -> SkillRequirements:
        skill_name = str(prepared.payload["skill_name"])
        try:
            return self._skill_requirements[skill_name]
        except KeyError as exc:
            raise ValueError(f"unknown skill '{skill_name}'") from exc

    def required_control_mode(self, skill_name: str) -> str:
        """Return the mode declared by the immutable catalog capability."""
        return self._skill_control_modes.get(str(skill_name), "")

    def _validate_parameters(self, prepared: PreparedRequest) -> None:
        skill_name = str(prepared.payload["skill_name"])
        try:
            parameters = self._parameter_schemas[skill_name]
        except KeyError as exc:
            raise ValueError(f"missing parameter schema for skill '{skill_name}'") from exc
        properties = parameters.get("properties")
        required = parameters.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ValueError(f"invalid parameter schema for skill '{skill_name}'")

        for field_name in _REQUEST_PARAMETER_FIELDS:
            value = prepared.payload.get(field_name)
            if field_name in {"x", "y", "yaw"}:
                missing = not bool(prepared.payload.get(f"has_{field_name}", False))
            else:
                missing = value is None or (isinstance(value, str) and not value)
            definition = properties.get(field_name)
            if definition is None:
                if missing or (field_name in {"motion_distance", "distance", "degree"} and value == 0.0):
                    continue
                raise ValueError(f"{field_name} is not accepted by skill '{skill_name}'")
            if not isinstance(definition, Mapping):
                raise ValueError(f"invalid parameter schema for skill '{skill_name}'")
            if missing:
                if field_name in required:
                    raise ValueError(f"{field_name} is required for skill '{skill_name}'")
                continue
            if definition.get("type") == "string":
                if definition.get("freeform") is True:
                    if not isinstance(value, str):
                        raise ValueError(f"{field_name} is invalid for skill '{skill_name}'")
                    continue
                enum = definition.get("enum")
                if not isinstance(value, str) or not isinstance(enum, list) or value not in enum:
                    raise ValueError(f"{field_name} is invalid for skill '{skill_name}'")
                continue
            if definition.get("type") == "number":
                exclusive_minimum = definition.get("exclusiveMinimum")
                invalid_number = (
                    isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
                )
                if exclusive_minimum is not None and (
                    isinstance(exclusive_minimum, bool)
                    or not isinstance(exclusive_minimum, int | float)
                    or (not invalid_number and value <= exclusive_minimum)
                ):
                    invalid_number = True
                if invalid_number:
                    raise ValueError(f"{field_name} is invalid for skill '{skill_name}'")
                continue
            raise ValueError(f"invalid parameter schema for skill '{skill_name}'")

    def _atomic_resources(self) -> tuple[BoundedRequestLedger, RootExecutionLease]:
        if self._ledger is None or self._lease is None:
            raise ValueError("GatewayPolicy.admit/finalize require both ledger and lease")
        return self._ledger, self._lease


def _finite_positive(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return number


def _readiness_reason(requirements: SkillRequirements, snapshot: RuntimeSnapshot) -> ReadinessReason:
    available = {
        "validate_skill": snapshot.validate_ready,
        "task_executor": snapshot.task_executor_ready,
        "arm_trajectory": snapshot.arm_trajectory_ready,
        "fresh_ee_pose": snapshot.ee_pose_fresh,
        "navigation": snapshot.navigation_ready,
    }
    required = tuple(name for name in _CAPABILITY_ORDER if getattr(requirements, name))
    unavailable = tuple(name for name in required if not available[name])
    return ReadinessReason(required=required, unavailable=unavailable)


def _capability_unavailable_message(readiness: ReadinessReason) -> str:
    if not readiness.unavailable:
        return ""
    return _CAPABILITY_UNAVAILABLE_MESSAGES[readiness.unavailable[0]]


def _is_active_owner(owner: ExecutionOwner | None, active_task_id: str) -> bool:
    return owner is not None and bool(active_task_id) and owner.root_task_id == active_task_id
