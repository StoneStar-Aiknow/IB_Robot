"""Dependency-neutral values shared by the unified runtime core.

The values in this module deliberately do not import ROS, a model SDK, or a
legacy backend contract.  They are the small boundary that a facade,
executor, stage, and session can all share.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


class LifecycleState(str, Enum):
    """Public lifecycle states of a :class:`ModelRuntimeHandle`."""

    CREATED = "created"
    LOADING = "loading"
    READY = "ready"
    RESET_REQUIRED = "reset_required"
    RESETTING = "resetting"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


# These aliases make the state name discoverable without creating competing
# enum types in downstream code.
RuntimeLifecycleState = LifecycleState
ModelRuntimeState = LifecycleState


class OutcomeState(str, Enum):
    """How certain the runtime is about one operation's execution."""

    NOT_STARTED = "not_started"
    STARTED = "started"
    COMPLETED = "completed"


EvidenceState = OutcomeState


class StreamState(str, Enum):
    """Lifecycle of one stream-owned state bank."""

    OPEN = "open"
    STEPPING = "stepping"
    RESET_REQUIRED = "reset_required"
    RESETTING = "resetting"
    FAILED = "failed"
    CLOSED = "closed"


StreamLifecycleState = StreamState


class ExecutionPhase(str, Enum):
    """Standard evidence phases.  Custom phase strings remain supported."""

    ADMISSION = "admission"
    DEADLINE = "deadline"
    CANCELLATION = "cancellation"
    BACKEND = "backend"
    ACL_ASYNC = "acl_async"
    STATEFUL_SESSION = "stateful_session"
    OUTPUT_VALIDATION = "output_validation"
    ADAPTATION = "adaptation"
    TRANSPORT = "transport"
    RESET = "reset"


EvidencePhase = ExecutionPhase
OutcomeEvidenceState = OutcomeState


class CancellationGranularity(str, Enum):
    STAGE = "stage"
    CHECKPOINT = "checkpoint"
    REQUEST_BOUNDARY = "request_boundary"


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _enum_value(enum_type: type[Enum], value: Enum | str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError:
        try:
            return enum_type[str(value)]
        except KeyError as exc:
            raise ValueError(f"invalid {enum_type.__name__}: {value!r}") from exc


def _freeze_json(value: Any, *, path: str = "value") -> Any:
    """Validate and recursively freeze a JSON-safe value."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain only JSON-safe values, got {type(value).__name__}")


def _freeze_json_mapping(value: Mapping[str, Any] | None, *, path: str) -> Mapping[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    frozen = _freeze_json(value, path=path)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by input validation
        raise TypeError(f"{path} must be a mapping")
    return frozen


def json_safe_data(value: Any) -> Any:
    """Return a mutable JSON-serializable copy for diagnostics and wire use."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _normalize_datetime(value).isoformat().replace("+00:00", "Z")
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        return json_safe_data(serializer())
    if isinstance(value, Mapping):
        return {str(key): json_safe_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe_data(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic values must contain finite floats")
        return value
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


@dataclass(frozen=True)
class Deadline:
    """One effective, absolute deadline shared by all execution layers.

    ``expires_at=None`` means unbounded.  A deadline is never extended by a
    downstream layer; callers can create a derived context only by explicitly
    supplying another absolute deadline.
    """

    expires_at: datetime | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc), compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.expires_at is not None:
            if not isinstance(self.expires_at, datetime):
                raise TypeError("Deadline.expires_at must be a datetime or None")
            object.__setattr__(self, "expires_at", _normalize_datetime(self.expires_at))
        if not callable(self.clock):
            raise TypeError("Deadline.clock must be callable")

    @classmethod
    def at(cls, expires_at: datetime | None) -> Deadline:
        return cls(expires_at)

    @classmethod
    def after(
        cls,
        seconds: float,
        *,
        now: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> Deadline:
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise TypeError("deadline timeout must be numeric")
        if not math.isfinite(float(seconds)) or seconds < 0:
            raise ValueError("deadline timeout must be finite and non-negative")
        origin = _normalize_datetime(now) if now is not None else datetime.now(timezone.utc)
        return cls(origin + timedelta(seconds=float(seconds)), clock or (lambda: datetime.now(timezone.utc)))

    @classmethod
    def from_timeout(
        cls,
        seconds: float,
        *,
        now: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> Deadline:
        return cls.after(seconds, now=now, clock=clock)

    @classmethod
    def unbounded(cls) -> Deadline:
        return cls(None)

    @property
    def bounded(self) -> bool:
        return self.expires_at is not None

    @property
    def deadline_at(self) -> datetime | None:
        return self.expires_at

    @property
    def is_unbounded(self) -> bool:
        return self.expires_at is None

    @property
    def expired(self) -> bool:
        return self.is_expired()

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = _normalize_datetime(now) if now is not None else _normalize_datetime(self.clock())
        return current >= self.expires_at

    def remaining_seconds(self, *, now: datetime | None = None) -> float | None:
        if self.expires_at is None:
            return None
        current = _normalize_datetime(now) if now is not None else _normalize_datetime(self.clock())
        return max(0.0, (self.expires_at - current).total_seconds())

    remaining = remaining_seconds

    def check(
        self,
        phase: str = ExecutionPhase.DEADLINE.value,
        *,
        now: datetime | None = None,
    ) -> None:
        if self.is_expired(now=now):
            from .errors import DeadlineExceeded

            raise DeadlineExceeded(phase=phase)


class CancellationToken:
    """Thread-safe cooperative cancellation shared by one execution."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._callbacks: list[Callable[[], object]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled

    @property
    def cancellation_requested(self) -> bool:
        return self.cancelled

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str | None = None) -> bool:
        """Request cancellation and invoke registered callbacks once.

        The return value is ``True`` only for the caller that changed the
        token from active to cancelled.
        """

        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            callbacks = tuple(self._callbacks)
            self._event.set()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Cancellation must not be prevented by an observer.
                continue
        return True

    def add_callback(self, callback: Callable[[], object]) -> None:
        if not callable(callback):
            raise TypeError("cancellation callback must be callable")
        with self._lock:
            if self._event.is_set():
                invoke = True
            else:
                self._callbacks.append(callback)
                invoke = False
        if invoke:
            with suppress(Exception):
                callback()

    def remove_callback(self, callback: Callable[[], object]) -> None:
        with self._lock, suppress(ValueError):
            self._callbacks.remove(callback)

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def throw_if_cancelled(self, phase: str = ExecutionPhase.CANCELLATION.value) -> None:
        if self.cancelled:
            from .errors import CancellationRequested

            raise CancellationRequested(phase=phase, reason=self.reason)

    raise_if_cancelled = throw_if_cancelled
    raise_if_requested = throw_if_cancelled


@dataclass(frozen=True)
class ModelRequest:
    """Read-only semantic model inputs and JSON-safe business metadata."""

    inputs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Mapping):
            raise TypeError("ModelRequest.inputs must be a mapping")
        if any(not isinstance(name, str) or not name for name in self.inputs):
            raise TypeError("ModelRequest input names must be non-empty strings")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, path="metadata"))

    @property
    def semantic_inputs(self) -> Mapping[str, object]:
        return self.inputs


@dataclass(frozen=True, init=False)
class ExecutionContext:
    """Request identity and shared execution controls."""

    request_id: str
    deadline: Deadline = field(default_factory=Deadline.unbounded)
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)

    def __init__(
        self,
        request_id: str,
        deadline: Deadline | datetime | None = None,
        cancellation_token: CancellationToken | None = None,
        *,
        cancellation: CancellationToken | None = None,
        token: CancellationToken | None = None,
    ) -> None:
        aliases = [value for value in (cancellation, token) if value is not None]
        if cancellation_token is not None and aliases and any(value is not cancellation_token for value in aliases):
            raise ValueError("cancellation_token, cancellation, and token disagree")
        selected_token = cancellation_token or (aliases[0] if aliases else CancellationToken())
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "cancellation_token", selected_token)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("ExecutionContext.request_id must be non-empty")
        deadline = self.deadline
        if deadline is None:
            deadline = Deadline.unbounded()
        elif isinstance(deadline, datetime):
            deadline = Deadline.at(deadline)
        elif not isinstance(deadline, Deadline):
            raise TypeError("ExecutionContext.deadline must be a Deadline")
        if not isinstance(self.cancellation_token, CancellationToken):
            raise TypeError("ExecutionContext.cancellation_token must be a CancellationToken")
        object.__setattr__(self, "deadline", deadline)

    @classmethod
    def create(
        cls,
        request_id: str,
        *,
        deadline: Deadline | datetime | None = None,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ExecutionContext:
        if deadline is not None and timeout is not None:
            raise ValueError("provide either deadline or timeout, not both")
        effective = Deadline.unbounded() if deadline is None else deadline
        if timeout is not None:
            effective = Deadline.after(timeout)
        return cls(request_id, effective, cancellation_token or CancellationToken())

    @property
    def token(self) -> CancellationToken:
        return self.cancellation_token

    @property
    def cancellation(self) -> CancellationToken:
        return self.cancellation_token

    def check(self, phase: str = CancellationGranularity.REQUEST_BOUNDARY.value) -> None:
        self.deadline.check(phase)
        self.cancellation_token.throw_if_cancelled(phase)

    checkpoint = check

    def with_deadline(self, deadline: Deadline | datetime | None) -> ExecutionContext:
        return replace(self, deadline=Deadline.unbounded() if deadline is None else deadline)

    def derive(self) -> ExecutionContext:
        """Return a value copy while retaining the exact shared token/deadline."""

        return replace(self)


@dataclass(frozen=True)
class OutcomeEvidence:
    """Evidence attached to both successful results and normalized failures."""

    state: OutcomeState = OutcomeState.NOT_STARTED
    outcome_known: bool = True
    state_mutated: bool = False
    phase: str = ExecutionPhase.ADMISSION.value
    details: Mapping[str, object] = field(default_factory=dict)
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        state = _enum_value(OutcomeState, self.state)
        if not isinstance(self.outcome_known, bool) or not isinstance(self.state_mutated, bool):
            raise TypeError("OutcomeEvidence flags must be bool")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("OutcomeEvidence.phase must be non-empty")
        observed_at = self.observed_at
        if observed_at is not None:
            if not isinstance(observed_at, datetime):
                raise TypeError("OutcomeEvidence.observed_at must be a datetime or None")
            observed_at = _normalize_datetime(observed_at)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "details", _freeze_json_mapping(self.details, path="evidence.details"))
        object.__setattr__(self, "observed_at", observed_at)

    @classmethod
    def not_started(cls, phase: str = ExecutionPhase.ADMISSION.value, **details: object) -> OutcomeEvidence:
        return cls(OutcomeState.NOT_STARTED, True, False, phase, details)

    @classmethod
    def started(
        cls,
        phase: str = ExecutionPhase.BACKEND.value,
        *,
        outcome_known: bool = False,
        state_mutated: bool = True,
        **details: object,
    ) -> OutcomeEvidence:
        return cls(OutcomeState.STARTED, outcome_known, state_mutated, phase, details)

    @classmethod
    def completed(
        cls,
        phase: str = ExecutionPhase.ADAPTATION.value,
        *,
        state_mutated: bool = False,
        **details: object,
    ) -> OutcomeEvidence:
        return cls(OutcomeState.COMPLETED, True, state_mutated, phase, details)

    def with_phase(self, phase: str, **details: object) -> OutcomeEvidence:
        merged = dict(self.details)
        merged.update(details)
        return replace(self, phase=phase, details=merged)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "state": self.state.value,
            "outcome_known": self.outcome_known,
            "state_mutated": self.state_mutated,
            "phase": self.phase,
            "details": json_safe_data(self.details),
        }
        if self.observed_at is not None:
            result["observed_at"] = json_safe_data(self.observed_at)
        return result


class OutcomeEvidenceTracker:
    """Small synchronized state machine used at runtime checkpoints."""

    def __init__(self, *, phase: str = ExecutionPhase.ADMISSION.value) -> None:
        self._lock = threading.Lock()
        self._evidence = OutcomeEvidence.not_started(phase)

    @property
    def evidence(self) -> OutcomeEvidence:
        return self.snapshot()

    def snapshot(self) -> OutcomeEvidence:
        with self._lock:
            return self._evidence

    @property
    def operation_started(self) -> bool:
        return self.snapshot().state is not OutcomeState.NOT_STARTED

    @property
    def outcome_known(self) -> bool:
        return self.snapshot().outcome_known

    @property
    def state_mutated(self) -> bool:
        return self.snapshot().state_mutated

    def mark_started(
        self,
        phase: str = ExecutionPhase.BACKEND.value,
        *,
        state_mutated: bool = True,
        outcome_known: bool = False,
        **details: object,
    ) -> OutcomeEvidence:
        with self._lock:
            self._evidence = OutcomeEvidence.started(
                phase,
                outcome_known=outcome_known,
                state_mutated=state_mutated,
                **details,
            )
            return self._evidence

    def mark_completed(
        self,
        phase: str = ExecutionPhase.ADAPTATION.value,
        *,
        state_mutated: bool | None = None,
        **details: object,
    ) -> OutcomeEvidence:
        with self._lock:
            self._evidence = OutcomeEvidence.completed(
                phase,
                state_mutated=self._evidence.state_mutated if state_mutated is None else state_mutated,
                **details,
            )
            return self._evidence

    def mark_failed(
        self,
        phase: str,
        *,
        state: OutcomeState | str | None = None,
        outcome_known: bool | None = None,
        state_mutated: bool | None = None,
        **details: object,
    ) -> OutcomeEvidence:
        with self._lock:
            current = self._evidence
            selected_state = (
                current.state
                if state is None
                else (state if isinstance(state, OutcomeState) else _enum_value(OutcomeState, state))
            )
            selected_known = current.outcome_known if outcome_known is None else outcome_known
            selected_mutated = current.state_mutated if state_mutated is None else state_mutated
            self._evidence = OutcomeEvidence(
                selected_state,
                selected_known,
                selected_mutated,
                phase,
                {**dict(current.details), **details},
            )
            return self._evidence


@dataclass(frozen=True)
class RuntimeLatency:
    """Optional phase-separated latency measurement in milliseconds."""

    total_ms: float
    backend_ms: float | None = None
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.total_ms,
            self.backend_ms if self.backend_ms is not None else self.total_ms,
            self.preprocess_ms,
            self.postprocess_ms,
        )
        if any(
            not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) < 0
            for value in values
        ):
            raise ValueError("runtime latency values must be finite and non-negative")
        object.__setattr__(self, "total_ms", float(self.total_ms))
        if self.backend_ms is not None:
            object.__setattr__(self, "backend_ms", float(self.backend_ms))
        object.__setattr__(self, "preprocess_ms", float(self.preprocess_ms))
        object.__setattr__(self, "postprocess_ms", float(self.postprocess_ms))


@dataclass(frozen=True)
class ModelResult:
    """The only successful result crossing the unified runtime boundary."""

    outputs: object
    latency: float | RuntimeLatency
    evidence: OutcomeEvidence
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.latency, bool) or not isinstance(self.latency, int | float | RuntimeLatency):
            raise TypeError("ModelResult.latency must be a number or RuntimeLatency")
        if isinstance(self.latency, int | float) and (
            not math.isfinite(float(self.latency)) or float(self.latency) < 0
        ):
            raise ValueError("ModelResult.latency must be finite and non-negative")
        if not isinstance(self.evidence, OutcomeEvidence):
            raise TypeError("ModelResult.evidence must be OutcomeEvidence")
        if isinstance(self.outputs, Mapping):
            object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, path="result.metadata"))

    @property
    def latency_ms(self) -> float:
        if isinstance(self.latency, RuntimeLatency):
            return self.latency.total_ms
        return float(self.latency)

    @property
    def successful(self) -> bool:
        return self.evidence.state is OutcomeState.COMPLETED and self.evidence.outcome_known

    @property
    def ok(self) -> bool:
        return self.successful

    def with_evidence(self, evidence: OutcomeEvidence) -> ModelResult:
        return replace(self, evidence=evidence)

    def to_dict(self) -> dict[str, object]:
        latency: object = self.latency
        if isinstance(latency, RuntimeLatency):
            latency = {
                "total_ms": latency.total_ms,
                "backend_ms": latency.backend_ms,
                "preprocess_ms": latency.preprocess_ms,
                "postprocess_ms": latency.postprocess_ms,
            }
        return {
            "outputs": json_safe_data(self.outputs),
            "latency": latency,
            "evidence": self.evidence.to_dict(),
            "metadata": json_safe_data(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionContract:
    """Minimal execution contract needed by the lifecycle/stream core."""

    state_scope: str = "request"
    execution_structure: str = "direct"
    orchestration_visibility: str | None = None
    cancellation_granularity: str = CancellationGranularity.REQUEST_BOUNDARY.value
    state_bank_mode: str | None = None
    max_open_streams: int | None = None
    state_links: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.state_scope not in {"request", "stream"}:
            raise ValueError("state_scope must be request or stream")
        if self.execution_structure not in {"direct", "iterative"}:
            raise ValueError("execution_structure must be direct or iterative")
        if self.execution_structure == "direct" and self.orchestration_visibility is not None:
            raise ValueError("direct execution cannot declare orchestration_visibility")
        if self.execution_structure == "iterative" and self.orchestration_visibility not in {"executor", "session"}:
            raise ValueError("iterative execution requires executor or session visibility")
        if self.cancellation_granularity not in {item.value for item in CancellationGranularity}:
            raise ValueError("invalid cancellation_granularity")
        if self.state_scope == "request":
            if self.state_bank_mode is not None or self.max_open_streams is not None or self.state_links:
                raise ValueError("request contracts cannot declare stream state")
        else:
            if self.state_bank_mode not in {"per_stream", "runtime_exclusive"}:
                raise ValueError("stream contracts require a state_bank_mode")
            if (
                not isinstance(self.max_open_streams, int)
                or isinstance(self.max_open_streams, bool)
                or self.max_open_streams < 1
            ):
                raise ValueError("stream contracts require positive max_open_streams")
            if self.state_bank_mode == "runtime_exclusive" and self.max_open_streams != 1:
                raise ValueError("runtime_exclusive contracts require max_open_streams=1")
        object.__setattr__(self, "state_links", tuple(self.state_links))

    @property
    def name(self) -> str:
        return f"{self.state_scope}-{self.execution_structure}"

    def to_dict(self) -> dict[str, object]:
        return {
            "state_scope": self.state_scope,
            "execution_structure": self.execution_structure,
            "orchestration_visibility": self.orchestration_visibility,
            "cancellation_granularity": self.cancellation_granularity,
            "state_bank_mode": self.state_bank_mode,
            "max_open_streams": self.max_open_streams,
            "state_links": json_safe_data(self.state_links),
        }


@dataclass(frozen=True)
class RuntimeHealth:
    state: LifecycleState
    ready: bool
    reason_code: str | None = None
    message: str | None = None
    recoverable: bool = False
    failure_count: int = 0
    last_successful_at: datetime | None = None
    recovery_requirement: object | None = None

    def __post_init__(self) -> None:
        state = _enum_value(LifecycleState, self.state)
        if self.ready != (state is LifecycleState.READY):
            raise ValueError("runtime readiness must match lifecycle state")
        if self.failure_count < 0:
            raise ValueError("failure_count cannot be negative")
        object.__setattr__(self, "state", state)
        if self.last_successful_at is not None:
            object.__setattr__(self, "last_successful_at", _normalize_datetime(self.last_successful_at))

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self.state

    @property
    def healthy(self) -> bool:
        return self.ready and self.state is LifecycleState.READY

    def __call__(self) -> RuntimeHealth:
        """Allow both ``handle.health`` and legacy ``handle.health()`` usage."""

        return self


@dataclass(frozen=True)
class RuntimeDiagnostics:
    runtime_id: str
    state: LifecycleState
    health: RuntimeHealth
    active_executions: int
    open_streams: int
    recovery_requirement: object | None = None
    last_failure: object | None = None
    last_outcome: OutcomeEvidence | None = None
    stream_diagnostics: tuple[object, ...] = ()
    identity: object | None = None
    execution_contract: object | None = None
    deployment_fingerprint: str | None = None
    runtime_profile_fingerprint: str | None = None
    artifact_integrity: object | None = None
    runtime_version: str | None = None
    capabilities: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.runtime_id:
            raise ValueError("runtime_id must be non-empty")
        object.__setattr__(self, "state", _enum_value(LifecycleState, self.state))
        object.__setattr__(
            self, "capabilities", _freeze_json_mapping(self.capabilities, path="diagnostics.capabilities")
        )
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, path="diagnostics.metadata"))
        object.__setattr__(self, "stream_diagnostics", tuple(self.stream_diagnostics))

    @property
    def active_streams(self) -> int:
        return self.open_streams

    @property
    def ready(self) -> bool:
        return self.state is LifecycleState.READY and self.health.ready

    @property
    def active_requests(self) -> int:
        return self.active_executions

    @property
    def open_stream_count(self) -> int:
        return self.open_streams

    @property
    def state_bank_mode(self) -> object | None:
        return self.capabilities.get("state_bank_mode")

    @property
    def max_open_streams(self) -> object | None:
        return self.capabilities.get("max_open_streams")

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self.state

    @property
    def streams(self) -> tuple[object, ...]:
        return self.stream_diagnostics

    def to_dict(self) -> dict[str, object]:
        failure = self.last_failure
        if hasattr(failure, "to_dict"):
            failure = failure.to_dict()
        return {
            "runtime_id": self.runtime_id,
            "state": self.state.value,
            "health": {
                "state": self.health.state.value,
                "ready": self.health.ready,
                "reason_code": self.health.reason_code,
                "message": self.health.message,
                "recoverable": self.health.recoverable,
                "failure_count": self.health.failure_count,
                "recovery_requirement": json_safe_data(self.health.recovery_requirement)
                if self.health.recovery_requirement is not None
                else None,
            },
            "active_executions": self.active_executions,
            "open_streams": self.open_streams,
            "recovery_requirement": json_safe_data(self.recovery_requirement)
            if self.recovery_requirement is not None
            else None,
            "last_failure": failure,
            "last_outcome": self.last_outcome.to_dict() if self.last_outcome is not None else None,
            "streams": [
                item.to_dict() if hasattr(item, "to_dict") else json_safe_data(item) for item in self.stream_diagnostics
            ],
            "identity": json_safe_data(self.identity) if self.identity is not None else None,
            "execution_contract": json_safe_data(self.execution_contract)
            if self.execution_contract is not None
            else None,
            "deployment_fingerprint": self.deployment_fingerprint,
            "runtime_profile_fingerprint": self.runtime_profile_fingerprint,
            "artifact_integrity": json_safe_data(self.artifact_integrity)
            if self.artifact_integrity is not None
            else None,
            "runtime_version": self.runtime_version,
            "capabilities": json_safe_data(self.capabilities),
            "metadata": json_safe_data(self.metadata),
        }


__all__ = [
    "CancellationGranularity",
    "CancellationToken",
    "Deadline",
    "EvidenceState",
    "EvidencePhase",
    "ExecutionContract",
    "ExecutionContext",
    "ExecutionPhase",
    "LifecycleState",
    "ModelRequest",
    "ModelResult",
    "ModelRuntimeState",
    "OutcomeEvidence",
    "OutcomeEvidenceState",
    "OutcomeEvidenceTracker",
    "OutcomeState",
    "RuntimeDiagnostics",
    "RuntimeHealth",
    "RuntimeLatency",
    "RuntimeLifecycleState",
    "StreamLifecycleState",
    "StreamState",
    "json_safe_data",
]


def __getattr__(name: str) -> object:
    """Lazily expose recovery values without introducing an import cycle."""

    if name in {"RecoveryAction", "RecoveryRequirement", "RecoveryScope"}:
        from .errors import RecoveryAction, RecoveryRequirement, RecoveryScope

        return {
            "RecoveryAction": RecoveryAction,
            "RecoveryRequirement": RecoveryRequirement,
            "RecoveryScope": RecoveryScope,
        }[name]
    raise AttributeError(name)
