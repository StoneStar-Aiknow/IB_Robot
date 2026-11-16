"""PickObject vocabulary shared by grasp executors and their callers.

* ``GraspSpec``: the optional grasp intent of a goal (``ibrobot_msgs/GraspSpec``).
  Every field is optional and empty means "executor's choice". An executor
  refuses, with ``GRASP_SPEC_UNSUPPORTED``, any non-empty field its effector's
  ``GraspCapability`` cannot honour; it never ignores a field silently.
* ``FAILURE_CLASSES``: the category a failed result reports in
  ``failure_class``, next to ``recoverable`` and ``measured_phase``.

No ROS or package dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

GRASP_SPEC_SCHEMA_VERSION = 1
GRASP_SPEC_UNSUPPORTED = "GRASP_SPEC_UNSUPPORTED"
GRASP_SPEC_INVALID = "GRASP_SPEC_INVALID"

#: Why a pick failed, independent of the robot. ``recoverable`` says separately
#: whether a new plan from the current state may succeed.
FAILURE_CLASSES = frozenset(
    {
        "PERCEPTION",  # target, part or scene not found or not trusted
        "PLANNING",  # no reachable, collision-free candidate
        "EXECUTION",  # the arm or hand did not follow the plan
        "GRIP",  # nothing held, or the grip was lost
        "VERIFICATION",  # held, but not as required (part, fingers, object)
        "SAFETY",  # refused or stopped by a safety check
        "UNSUPPORTED",  # the request asks for what this robot cannot do
        "CANCELED",
        "TIMEOUT",
        "INTERNAL",
    }
)


class GraspSpecVersionError(ValueError):
    """A grasp spec version this contract does not know (a newer caller)."""


_NAME_LIMIT = 64
_LIST_LIMIT = 16


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > _NAME_LIMIT or value != value.strip():
        raise ValueError(f"{field} must be a trimmed string of at most {_NAME_LIMIT} characters")
    return value


def _names(values: object, field: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError(f"{field} must be a list of names")
    names = tuple(_name(value, field) for value in values)
    if len(names) > _LIST_LIMIT or "" in names or len(set(names)) != len(names):
        raise ValueError(f"{field} must hold at most {_LIST_LIMIT} distinct non-empty names")
    return names


@dataclass(frozen=True)
class GraspSpec:
    schema_version: int = 0
    part: str = ""
    avoid_parts: tuple[str, ...] = ()
    grasp_type: str = ""
    fingers: tuple[str, ...] = ()
    target_force_n: float = 0.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {0, GRASP_SPEC_SCHEMA_VERSION}:
            raise GraspSpecVersionError("unsupported grasp spec version")
        _name(self.part, "part")
        _name(self.grasp_type, "grasp_type")
        object.__setattr__(self, "avoid_parts", _names(self.avoid_parts, "avoid_parts"))
        object.__setattr__(self, "fingers", _names(self.fingers, "fingers"))
        force = self.target_force_n
        if isinstance(force, bool) or not isinstance(force, int | float) or not math.isfinite(force) or force < 0:
            raise ValueError("target_force_n must be a finite non-negative number")
        object.__setattr__(self, "target_force_n", float(force))
        if self.schema_version == 0 and self.requested():
            raise ValueError("a version-zero grasp spec must be empty")
        if self.part and self.part in self.avoid_parts:
            raise ValueError("the grasped part cannot also be avoided")

    def requested(self) -> tuple[str, ...]:
        """The fields this spec constrains, in declaration order."""
        return tuple(
            name
            for name, value in (
                ("part", self.part),
                ("avoid_parts", self.avoid_parts),
                ("grasp_type", self.grasp_type),
                ("fingers", self.fingers),
                ("target_force_n", self.target_force_n),
            )
            if value
        )

    @classmethod
    def from_value(cls, value: object) -> GraspSpec:
        """From a generated message, a mapping or None (no spec). Invalid values raise."""
        if value is None or isinstance(value, cls):
            return value or cls()
        get = value.__getitem__ if isinstance(value, Mapping) else lambda name: getattr(value, name)
        try:
            data = {name: get(name) for name in cls.__dataclass_fields__}
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"incomplete grasp spec: {exc}") from exc
        return cls(**data)

    def fill(self, target: object) -> None:
        """Fill a generated ``ibrobot_msgs/GraspSpec`` without importing ROS."""
        target.schema_version = self.schema_version
        target.part = self.part
        target.avoid_parts = list(self.avoid_parts)
        target.grasp_type = self.grasp_type
        target.fingers = list(self.fingers)
        target.target_force_n = self.target_force_n


@dataclass(frozen=True)
class GraspCapability:
    """What one effector declares it can honour; the default honours no spec field."""

    grasp_types: tuple[str, ...] = ()
    fingers: tuple[str, ...] = ()
    part_targeting: bool = False
    force_control: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasp_types", _names(self.grasp_types, "grasp_types"))
        object.__setattr__(self, "fingers", _names(self.fingers, "fingers"))
        if type(self.part_targeting) is not bool or type(self.force_control) is not bool:
            raise ValueError("part_targeting and force_control must be bools")

    def unsupported(self, spec: GraspSpec) -> tuple[str, ...]:
        """The spec fields this effector cannot honour; empty means the goal may proceed."""
        problems = []
        if (spec.part or spec.avoid_parts) and not self.part_targeting:
            problems.extend(name for name in ("part", "avoid_parts") if getattr(spec, name))
        if spec.grasp_type and spec.grasp_type not in self.grasp_types:
            problems.append("grasp_type")
        if spec.fingers and not set(spec.fingers) <= set(self.fingers):
            problems.append("fingers")
        if spec.target_force_n and not self.force_control:
            problems.append("target_force_n")
        return tuple(problems)


def check_grasp_spec(value: object, capability: GraspCapability) -> tuple[GraspSpec, str]:
    """An executor's one call on a goal's spec: (spec, "") or (empty spec, error code).

    An unknown version or a field the effector cannot honour is
    ``GRASP_SPEC_UNSUPPORTED``; a malformed spec is ``GRASP_SPEC_INVALID``.
    """
    try:
        spec = GraspSpec.from_value(value)
    except GraspSpecVersionError:
        return GraspSpec(), GRASP_SPEC_UNSUPPORTED
    except ValueError:
        return GraspSpec(), GRASP_SPEC_INVALID
    return (GraspSpec(), GRASP_SPEC_UNSUPPORTED) if capability.unsupported(spec) else (spec, "")
