"""Versioned execution facts shared by ROS producers and tolerant consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

PHASE_STATES = frozenset({"RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"})
RECOVERY_STATES = frozenset({"NOT_NEEDED", "IN_PROGRESS", "COMPLETED", "FAILED", "UNKNOWN"})
SUGGESTED_ACTIONS = frozenset({"NONE", "WAIT_SERVICE", "WAIT_USER", "MANUAL_CHECK"})
RELEASE_STATES = frozenset({"UNKNOWN", "NOT_RELEASED", "RELEASED"})
VERIFICATION_STATES = frozenset({"UNKNOWN", "NOT_RUN", "SUCCESS", "FAILED", "UNCERTAIN"})
# PlaceObject.action has a distinct numeric ABI: zero means NOT_RELEASED/NOT_RUN,
# not UNKNOWN. Keep these explicit and test against the generated action constants.
PLACE_RELEASE_STATES = {0: "NOT_RELEASED", 1: "RELEASED", 2: "UNKNOWN"}
PLACE_VERIFICATION_STATES = {0: "NOT_RUN", 1: "SUCCESS", 2: "FAILED", 3: "UNCERTAIN"}
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
    release_status: str = "UNKNOWN"
    verification_status: str = "UNKNOWN"
    place_succeeded: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {0, 1, 2}:
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
            ("release_status", RELEASE_STATES),
            ("verification_status", VERIFICATION_STATES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"invalid {name}")
        if type(self.place_succeeded) is not bool:
            raise ValueError("place_succeeded must be a bool")
        if self.schema_version < 2 and (
            self.release_status != "UNKNOWN" or self.verification_status != "UNKNOWN" or self.place_succeeded
        ):
            raise ValueError("placement facts require version two")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> ExecutionDetail:
        if isinstance(value, cls):
            return value
        get = value.__getitem__ if isinstance(value, Mapping) else lambda name: getattr(value, name)
        try:
            version = get("schema_version")
            if type(version) is not int or version not in {1, 2}:
                return cls()
            names = [field.name for field in fields(cls)]
            # v1 wire objects and mappings predate the last three fields. Never
            # infer physical facts from their phase, reason, or attached extras.
            data = {name: get(name) for name in (names[:8] if version == 1 else names)}
            return cls(**data)
        except (AttributeError, KeyError, TypeError, ValueError):
            return cls()


def place_execution_detail(
    value: object, *, release_status: str, verification_status: str, place_succeeded: bool
) -> ExecutionDetail:
    """Attach explicit placement facts without changing generic execution facts."""
    return replace(
        ExecutionDetail.from_value(value),
        schema_version=2,
        release_status=release_status,
        verification_status=verification_status,
        place_succeeded=place_succeeded,
    )


def place_result_execution_detail(value: object) -> ExecutionDetail:
    """Project PlaceObject's numeric terminal facts using its fixed action ABI."""

    def state(name, mapping):
        number = getattr(value, name, None)
        return mapping.get(number, "UNKNOWN") if type(number) is int else "UNKNOWN"

    return place_execution_detail(
        getattr(value, "execution_detail", None),
        release_status=state("release_status", PLACE_RELEASE_STATES),
        verification_status=state("verification_status", PLACE_VERIFICATION_STATES),
        place_succeeded=getattr(value, "place_succeeded", False) is True,
    )


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
