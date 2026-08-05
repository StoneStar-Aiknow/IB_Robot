"""Short-lived immutable Agent plan state owned by ``embodied_agent``."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_common.primitive_contracts import canonical_json
from embodied_common.workflow_contracts import CanonicalWorkflowStep, normalize_workflow_steps

MAX_PLAN_STEPS = 16
DEFAULT_PLAN_TTL_SEC = 300.0
DEFAULT_MAX_RECORDS = 1024


class AgentPlanError(Exception):
    """Stable plan-store failure with no secret-bearing message."""

    def __init__(self, code: str, message: str = "agent plan request rejected") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentPlan:
    schema_version: int
    plan_id: str
    plan_token: str = field(repr=False)
    plan_kind: int
    raw_command: str
    workflow_steps: tuple[CanonicalWorkflowStep, ...]
    plan_digest: str
    registry_epoch: str
    registry_generation: int
    registry_digest: str
    expires_at: tuple[int, int]

    SINGLE_SKILL = 1
    WORKFLOW = 2


@dataclass(frozen=True)
class PlanConfirmation:
    confirmed: bool
    confirmation_token: str = field(repr=False)
    plan_digest: str
    task_id: str


@dataclass(frozen=True)
class AgentPlanExecution:
    plan: AgentPlan
    task_id: str
    state: str
    confirmation_token: str = field(repr=False)
    terminal_code: str = ""
    terminal_message: str = ""
    workflow_digest: str = ""
    completed_step_count: int = 0
    newly_accepted: bool = False


@dataclass
class _PlanRecord:
    plan: AgentPlan
    request_id: str
    monotonic_expires_at: float
    state: str = "PLANNED"
    task_id: str = ""
    confirmation_token: str = field(default="", repr=False)
    terminal_code: str = ""
    terminal_message: str = ""
    workflow_digest: str = ""
    completed_step_count: int = 0


class AgentPlanStore:
    """In-memory bounded store; callers provide their own state lock."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        token_factory: Callable[[], str] | None = None,
        plan_id_factory: Callable[[], str] | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        ttl_sec: float = DEFAULT_PLAN_TTL_SEC,
    ) -> None:
        if max_records <= 0 or ttl_sec <= 0:
            raise ValueError("max_records and ttl_sec must be positive")
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._plan_id_factory = plan_id_factory or (lambda: str(uuid.uuid4()))
        self._max_records = max_records
        self._ttl_sec = float(ttl_sec)
        self._records: dict[str, _PlanRecord] = {}
        self._request_index: dict[str, str] = {}
        self._expired_tokens: OrderedDict[str, None] = OrderedDict()

    def create_plan(
        self,
        *,
        request_id: str,
        raw_command: str,
        workflow_steps: Sequence[Any],
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> AgentPlan:
        self._purge()
        if not request_id.strip() or not raw_command.strip() or not registry_epoch or not registry_digest:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "plan request fields are incomplete")
        try:
            steps = normalize_workflow_steps(workflow_steps, max_steps=MAX_PLAN_STEPS)
        except (TypeError, ValueError) as exc:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", str(exc)) from exc
        if registry_generation <= 0:
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "registry generation must be positive")

        existing_token = self._request_index.get(request_id)
        if existing_token:
            existing = self._records.get(existing_token)
            if existing is not None:
                if _plan_payload_matches(
                    existing.plan,
                    raw_command=raw_command,
                    workflow_steps=steps,
                    registry_epoch=registry_epoch,
                    registry_generation=registry_generation,
                    registry_digest=registry_digest,
                ):
                    return existing.plan
                raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "request_id payload conflicts with the stored plan")

        self._ensure_capacity()
        plan = AgentPlan(
            schema_version=1,
            plan_id=self._plan_id_factory(),
            plan_token=_new_opaque_token(self._token_factory),
            plan_kind=AgentPlan.SINGLE_SKILL if len(steps) == 1 else AgentPlan.WORKFLOW,
            raw_command=raw_command,
            workflow_steps=steps,
            plan_digest=compute_plan_digest(
                raw_command=raw_command,
                workflow_steps=steps,
                registry_epoch=registry_epoch,
                registry_generation=registry_generation,
                registry_digest=registry_digest,
            ),
            registry_epoch=registry_epoch,
            registry_generation=registry_generation,
            registry_digest=registry_digest,
            expires_at=_wall_time(self._wall_clock() + self._ttl_sec),
        )
        self._records[plan.plan_token] = _PlanRecord(
            plan=plan,
            request_id=request_id,
            monotonic_expires_at=self._clock() + self._ttl_sec,
        )
        self._request_index[request_id] = plan.plan_token
        return plan

    def get(self, plan_token: str) -> AgentPlan:
        record = self._get_record(plan_token)
        return record.plan

    def validate(
        self,
        *,
        plan_token: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> AgentPlan:
        record = self._get_record(plan_token)
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        return record.plan

    def confirm(
        self,
        *,
        plan_token: str,
        plan_digest: str,
        task_id: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> PlanConfirmation:
        record = self._get_record(plan_token)
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        if not hmac.compare_digest(record.plan.plan_digest, plan_digest):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan digest does not match")
        if not task_id.strip():
            raise AgentPlanError("SKILL_SCHEMA_INVALID", "task_id must be non-empty")
        if record.state != "PLANNED":
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan has already been confirmed or accepted")
        confirmation_token = _new_opaque_token(self._token_factory)
        record.state = "CONFIRMED"
        record.task_id = task_id
        record.confirmation_token = confirmation_token
        return PlanConfirmation(True, confirmation_token, record.plan.plan_digest, task_id)

    def accept_execution(
        self,
        *,
        plan_token: str,
        confirmation_token: str,
        task_id: str,
        registry_epoch: str,
        registry_generation: int,
        registry_digest: str,
    ) -> AgentPlanExecution:
        record = self._get_record(plan_token, allow_terminal=True)
        self._require_identity(record.plan, registry_epoch, registry_generation, registry_digest)
        if record.task_id and record.task_id != task_id:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan is bound to a different task")
        if record.state in {"ACCEPTED", "TERMINAL"}:
            if not hmac.compare_digest(record.confirmation_token, confirmation_token):
                raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "confirmation token does not match")
            return AgentPlanExecution(
                plan=record.plan,
                task_id=record.task_id,
                state=record.state,
                confirmation_token=record.confirmation_token,
                terminal_code=record.terminal_code,
                terminal_message=record.terminal_message,
                workflow_digest=record.workflow_digest,
                completed_step_count=record.completed_step_count,
            )
        if record.state != "CONFIRMED" or not hmac.compare_digest(record.confirmation_token, confirmation_token):
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan confirmation is invalid")
        record.state = "ACCEPTED"
        return AgentPlanExecution(
            record.plan,
            task_id,
            record.state,
            record.confirmation_token,
            newly_accepted=True,
        )

    def mark_terminal(
        self,
        *,
        plan_token: str,
        task_id: str,
        terminal_code: str = "",
        terminal_message: str = "",
        workflow_digest: str = "",
        completed_step_count: int = 0,
    ) -> AgentPlanExecution:
        record = self._get_record(plan_token, allow_terminal=True)
        if record.task_id != task_id:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan is bound to a different task")
        if record.state not in {"ACCEPTED", "TERMINAL"}:
            raise AgentPlanError("SKILL_REQUEST_ID_CONFLICT", "plan has not been accepted")
        record.state = "TERMINAL"
        record.terminal_code = terminal_code
        record.terminal_message = terminal_message
        record.workflow_digest = workflow_digest
        record.completed_step_count = completed_step_count
        record.monotonic_expires_at = self._clock() + self._ttl_sec
        return AgentPlanExecution(
            plan=record.plan,
            task_id=task_id,
            state=record.state,
            confirmation_token=record.confirmation_token,
            terminal_code=terminal_code,
            terminal_message=terminal_message,
            workflow_digest=workflow_digest,
            completed_step_count=completed_step_count,
        )

    def cancel(self, *, plan_token: str, task_id: str) -> AgentPlanExecution:
        return self.mark_terminal(plan_token=plan_token, task_id=task_id, terminal_code="SKILL_CANCELLED")

    def _get_record(self, plan_token: str, *, allow_terminal: bool = False) -> _PlanRecord:
        self._purge()
        record = self._records.get(plan_token)
        if record is None:
            if plan_token in self._expired_tokens:
                raise AgentPlanError("SKILL_AGENT_PLAN_EXPIRED")
            raise AgentPlanError("SKILL_AGENT_PLAN_NOT_FOUND")
        if record.state == "TERMINAL" and not allow_terminal:
            raise AgentPlanError("SKILL_AGENT_PLAN_EXPIRED", "plan is no longer active")
        return record

    def _require_identity(self, plan: AgentPlan, epoch: str, generation: int, digest: str) -> None:
        if (plan.registry_epoch, plan.registry_generation, plan.registry_digest) != (epoch, generation, digest):
            raise AgentPlanError("SKILL_REGISTRY_VERSION_MISMATCH", "plan snapshot identity is stale")

    def _ensure_capacity(self) -> None:
        self._purge()
        while len(self._records) >= self._max_records:
            candidates = [
                (record.monotonic_expires_at, token)
                for token, record in self._records.items()
                if record.state != "ACCEPTED"
            ]
            if not candidates:
                raise AgentPlanError("SKILL_EXECUTION_BUSY", "plan store is full of active executions")
            _, token = min(candidates)
            self._remove(token)

    def _purge(self) -> None:
        now = self._clock()
        for token, record in list(self._records.items()):
            if record.monotonic_expires_at <= now and record.state != "ACCEPTED":
                self._expired_tokens[token] = None
                self._expired_tokens.move_to_end(token)
                while len(self._expired_tokens) > self._max_records:
                    self._expired_tokens.popitem(last=False)
                self._remove(token)

    def _remove(self, token: str) -> None:
        record = self._records.pop(token, None)
        if record is not None and self._request_index.get(record.request_id) == token:
            self._request_index.pop(record.request_id, None)


def compute_plan_digest(
    *,
    raw_command: str,
    workflow_steps: Sequence[Any],
    registry_epoch: str,
    registry_generation: int,
    registry_digest: str,
) -> str:
    steps = normalize_workflow_steps(workflow_steps, max_steps=MAX_PLAN_STEPS)
    preimage = {
        "schema_version": 1,
        "raw_command": raw_command,
        "workflow_steps": [step.to_dict() for step in steps],
        "registry_epoch": registry_epoch,
        "registry_generation": registry_generation,
        "registry_digest": registry_digest,
    }
    return hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()


def _plan_payload_matches(
    plan: AgentPlan,
    *,
    raw_command: str,
    workflow_steps: Sequence[CanonicalWorkflowStep],
    registry_epoch: str,
    registry_generation: int,
    registry_digest: str,
) -> bool:
    return (
        plan.raw_command == raw_command
        and plan.workflow_steps == tuple(workflow_steps)
        and plan.registry_epoch == registry_epoch
        and plan.registry_generation == registry_generation
        and plan.registry_digest == registry_digest
    )


def _new_opaque_token(factory: Callable[[], str]) -> str:
    token = factory()
    if not isinstance(token, str) or len(token.encode("utf-8")) < 16:
        raise AgentPlanError("SKILL_SCHEMA_INVALID", "token factory did not provide an opaque token")
    return token


def _wall_time(timestamp: float) -> tuple[int, int]:
    seconds = int(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return seconds, nanoseconds
