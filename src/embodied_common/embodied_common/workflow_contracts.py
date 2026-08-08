"""Canonical typed Workflow contracts shared by planners and executors."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from embodied_common.canon import to_canonical_json


@dataclass(frozen=True)
class CanonicalWorkflowStep:
    schema_version: int
    skill_name: str
    target_name: str = ""
    place_name: str = ""
    motion_direction: str = ""
    motion_distance: float = 0.0
    timeout_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("WorkflowStep.schema_version must be 1")
        if not self.skill_name.strip():
            raise ValueError("WorkflowStep.skill_name must be non-empty")
        for field_name in ("target_name", "place_name", "motion_direction"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"WorkflowStep.{field_name} must be a string")
        for field_name in ("motion_distance", "timeout_sec"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"WorkflowStep.{field_name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"WorkflowStep.{field_name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "skill_name": self.skill_name.strip(),
            "target_name": self.target_name.strip(),
            "place_name": self.place_name.strip(),
            "motion_direction": self.motion_direction.strip().lower(),
            "motion_distance": _finite_float(self.motion_distance),
            "timeout_sec": _finite_float(self.timeout_sec),
        }


def normalize_workflow_step(step: Any) -> CanonicalWorkflowStep:
    if isinstance(step, CanonicalWorkflowStep):
        return step
    if isinstance(step, Mapping):
        values = step
    else:
        values = {
            field: getattr(step, field)
            for field in (
                "schema_version",
                "skill_name",
                "target_name",
                "place_name",
                "motion_direction",
                "motion_distance",
                "timeout_sec",
            )
        }
    return CanonicalWorkflowStep(
        schema_version=int(values.get("schema_version", 0)),
        skill_name=str(values.get("skill_name", "")),
        target_name=str(values.get("target_name", "")),
        place_name=str(values.get("place_name", "")),
        motion_direction=str(values.get("motion_direction", "")),
        motion_distance=_finite_float(values.get("motion_distance", 0.0)),
        timeout_sec=_finite_float(values.get("timeout_sec", 0.0)),
    )


def normalize_workflow_steps(steps: Sequence[Any], *, max_steps: int = 16) -> tuple[CanonicalWorkflowStep, ...]:
    if not steps:
        raise ValueError("workflow_steps must not be empty")
    if len(steps) > max_steps:
        raise ValueError(f"workflow_steps exceeds maximum of {max_steps}")
    return tuple(normalize_workflow_step(step) for step in steps)


def workflow_digest_preimage(
    *,
    root_task_id: str,
    task_budget: Any,
    expected_registry_epoch: str,
    expected_registry_generation: int,
    expected_registry_digest: str,
    workflow_steps: Sequence[Any],
) -> dict[str, Any]:
    if not root_task_id.strip():
        raise ValueError("root_task_id must be non-empty")
    if not expected_registry_epoch or not expected_registry_digest:
        raise ValueError("expected registry identity must be complete")
    if expected_registry_generation <= 0:
        raise ValueError("expected_registry_generation must be positive")
    steps = normalize_workflow_steps(workflow_steps)
    return {
        "schema_version": 1,
        "root_task_id": root_task_id,
        "task_budget": _task_budget_dict(task_budget),
        "expected_registry_epoch": expected_registry_epoch,
        "expected_registry_generation": int(expected_registry_generation),
        "expected_registry_digest": expected_registry_digest,
        "workflow_steps": [step.to_dict() for step in steps],
    }


def compute_workflow_digest(**kwargs: Any) -> str:
    payload = workflow_digest_preimage(**kwargs)
    return hashlib.sha256(to_canonical_json(payload).encode()).hexdigest()


def _task_budget_dict(task_budget: Any) -> dict[str, Any]:
    if is_dataclass(task_budget):
        task_budget = asdict(task_budget)
    started_at = _value(task_budget, "started_at")
    deadline = _value(task_budget, "deadline")
    schema_version = int(_value(task_budget, "schema_version"))
    if schema_version != 1:
        raise ValueError("TaskBudget.schema_version must be 1")
    return {
        "started_at": _time_dict(started_at),
        "deadline": _time_dict(deadline),
    }


def _time_dict(value: Any) -> dict[str, int]:
    sec = int(_value(value, "sec"))
    nanosec = int(_value(value, "nanosec"))
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("invalid builtin time")
    return {"sec": sec, "nanosec": nanosec}


def _value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value[key]
    return getattr(value, key)


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("number must be finite")
    return 0.0 if result == 0.0 else result
