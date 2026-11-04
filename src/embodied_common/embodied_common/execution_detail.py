"""Versioned Skill execution facts, independent of ROS and of any robot.

Display normalization is deliberately tolerant: malformed/legacy detail must
never turn an authoritative terminal into an unknown physical outcome.
Producers use the strict dataclass; consumers use ``from_value`` at boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

PHASE_STATES = frozenset({"RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"})
RECOVERY_STATES = frozenset({"NOT_NEEDED", "IN_PROGRESS", "COMPLETED", "FAILED", "UNKNOWN"})
SUGGESTED_ACTIONS = frozenset({"NONE", "WAIT_SERVICE", "WAIT_USER", "MANUAL_CHECK"})
MESSAGE_MAX = 300


def short_message(value: object, limit: int = MESSAGE_MAX) -> str:
    """Bound the entire display summary, not just its exception fragment."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class ExecutionDetail:
    schema_version: int = 0
    phase: str = ""
    phase_state: str = "UNKNOWN"
    failed_phase: str = ""
    reason_code: str = ""
    recovery_state: str = "UNKNOWN"
    retry_count: int = 0
    suggested_action: str = "MANUAL_CHECK"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {0, 1}:
            raise ValueError("unsupported execution detail version")
        if type(self.retry_count) is not int or not 0 <= self.retry_count <= 0xFFFFFFFF:
            raise ValueError("retry_count must be a uint32")
        for name in ("phase", "failed_phase", "reason_code"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 64:
                raise ValueError(f"{name} must be a string of at most 64 characters")
        for name, allowed in (
            ("phase_state", PHASE_STATES),
            ("recovery_state", RECOVERY_STATES),
            ("suggested_action", SUGGESTED_ACTIONS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"invalid {name}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> ExecutionDetail:
        if isinstance(value, cls):
            return value
        names = (field.name for field in fields(cls))
        try:
            data = (
                {name: value[name] for name in names}
                if isinstance(value, Mapping)
                else {name: getattr(value, name) for name in names}
            )
            if data.get("schema_version") != 1:
                return cls()
            return cls(**data)
        except (AttributeError, KeyError, TypeError, ValueError):
            return cls()


def fill_execution_detail(target: object, value: object) -> None:
    """Fill an existing generated ExecutionDetail without importing ROS."""
    for name, item in ExecutionDetail.from_value(value).to_dict().items():
        setattr(target, name, item)


def execution_context(value: object) -> dict:
    """Common projection for result, query and event consumers."""
    get = value.get if isinstance(value, Mapping) else lambda key, default=None: getattr(value, key, default)
    skill = get("current_skill", "")
    index = get("workflow_step_index", 0)
    return {
        "current_skill": skill if isinstance(skill, str) else "",
        "workflow_step_index": index if type(index) is int and 0 <= index <= 0xFFFFFFFF else 0,
        "execution_detail": ExecutionDetail.from_value(get("execution_detail")).to_dict(),
    }


def fill_execution_context(target: object, value: object) -> None:
    context = execution_context(value)
    target.current_skill = context["current_skill"]
    target.workflow_step_index = context["workflow_step_index"]
    fill_execution_detail(target.execution_detail, context["execution_detail"])
