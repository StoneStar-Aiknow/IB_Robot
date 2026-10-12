"""Canonical typed Program contract shared by execution hosts and the Gateway."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.workflow_contracts import CanonicalWorkflowStep, normalize_workflow_steps

PROGRAM_SCHEMA_VERSION = 1
MAX_PROGRAM_STEPS = 16
MAX_ID_LENGTH = 128


@dataclass(frozen=True)
class ProgramIdentity:
    """Exact deployment identity frozen into one Program."""

    registry_epoch: str
    registry_generation: int
    registry_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry_epoch", _required_text(self.registry_epoch, "registry_epoch"))
        if isinstance(self.registry_generation, bool) or int(self.registry_generation) <= 0:
            raise ValueError("registry_generation must be positive")
        object.__setattr__(self, "registry_generation", int(self.registry_generation))
        object.__setattr__(self, "registry_digest", _required_text(self.registry_digest, "registry_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_epoch": self.registry_epoch,
            "registry_generation": self.registry_generation,
            "registry_digest": self.registry_digest,
        }


@dataclass(frozen=True)
class CanonicalProgram:
    """Immutable bounded Program accepted by the robot execution runtime."""

    schema_version: int
    request_id: str
    caller_scope: str
    robot_id: str
    identity: ProgramIdentity
    steps: tuple[CanonicalWorkflowStep, ...]
    task_budget_sec: float
    program_digest: str

    @property
    def execution_key(self) -> tuple[str, str, str]:
        return self.caller_scope, self.robot_id, self.request_id

    @property
    def root_task_id(self) -> str:
        """Return the deterministic Gateway root ID for this request scope."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ibrobot:{self.caller_scope}:{self.robot_id}:{self.request_id}"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "caller_scope": self.caller_scope,
            "robot_id": self.robot_id,
            **self.identity.to_dict(),
            "workflow_steps": [step.to_dict() for step in self.steps],
            "task_budget_sec": self.task_budget_sec,
            "program_digest": self.program_digest,
        }


def canonicalize_program(
    *,
    schema_version: Any,
    request_id: Any,
    caller_scope: Any,
    robot_id: Any,
    expected_registry_epoch: Any,
    expected_registry_generation: Any,
    expected_registry_digest: Any,
    workflow_steps: Sequence[Any],
    task_budget_sec: Any,
) -> CanonicalProgram:
    """Validate and freeze all execution-semantic request fields."""
    if isinstance(schema_version, bool) or schema_version != PROGRAM_SCHEMA_VERSION:
        raise ValueError(f"Program.schema_version must be {PROGRAM_SCHEMA_VERSION}")
    normalized_request_id = _bounded_id(request_id, "request_id")
    normalized_caller_scope = _bounded_id(caller_scope, "caller_scope")
    normalized_robot_id = _bounded_id(robot_id, "robot_id")
    identity = ProgramIdentity(
        registry_epoch=expected_registry_epoch,
        registry_generation=expected_registry_generation,
        registry_digest=expected_registry_digest,
    )
    steps = normalize_workflow_steps(workflow_steps, max_steps=MAX_PROGRAM_STEPS)
    if isinstance(task_budget_sec, bool) or not isinstance(task_budget_sec, int | float):
        raise ValueError("task_budget_sec must be a finite positive number")
    normalized_budget = float(task_budget_sec)
    if not math.isfinite(normalized_budget) or normalized_budget <= 0.0:
        raise ValueError("task_budget_sec must be a finite positive number")
    preimage = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "request_id": normalized_request_id,
        "caller_scope": normalized_caller_scope,
        "robot_id": normalized_robot_id,
        **identity.to_dict(),
        "workflow_steps": [step.to_dict() for step in steps],
        "task_budget_sec": normalized_budget,
    }
    return CanonicalProgram(
        schema_version=PROGRAM_SCHEMA_VERSION,
        request_id=normalized_request_id,
        caller_scope=normalized_caller_scope,
        robot_id=normalized_robot_id,
        identity=identity,
        steps=steps,
        task_budget_sec=normalized_budget,
        program_digest=sha256_text(to_canonical_json(preimage)),
    )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _bounded_id(value: Any, field_name: str) -> str:
    result = _required_text(value, field_name)
    if len(result) > MAX_ID_LENGTH:
        raise ValueError(f"{field_name} must not exceed {MAX_ID_LENGTH} characters")
    return result


__all__ = [
    "CanonicalProgram",
    "MAX_PROGRAM_STEPS",
    "PROGRAM_SCHEMA_VERSION",
    "ProgramIdentity",
    "canonicalize_program",
]
