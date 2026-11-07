"""Neutral free-pose results. Body frame: x forward, y left, z up; SI units."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class ArmSolution:
    q: dict[str, float]
    cost: float
    residual_deg: dict[str, float]
    residual_m: dict[str, float]
    approx: bool


@dataclass
class Report:
    ok: bool
    limit_violations: list[str] = field(default_factory=list)
    clearance_violations: list[tuple[str, float]] = field(default_factory=list)
    min_clearance_m: float = 1.0
    gravity_share: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "ok": bool(self.ok),
            "limit_violations": self.limit_violations,
            "clearance_violations": [(n, round(d, 3)) for n, d in self.clearance_violations],
            "min_clearance_m": round(float(self.min_clearance_m), 3),
            "gravity_share": round(float(self.gravity_share), 3),
            "notes": self.notes,
        }


@dataclass
class Plan:
    traj: Trajectory
    ok: bool
    via: list[str]
    report: Report


@dataclass
class Trajectory:
    times: Sequence[float]  # (N,)
    positions: Sequence[Sequence[float]]  # (N, 14) in ARM_JOINTS order


class SpecError(ValueError):
    """A pose description that must not be executed."""


@dataclass
class Composition:
    kind: str
    frames: list[dict[str, float]] = field(default_factory=list)
    holds: list[float] = field(default_factory=list)
    user_message: str = ""
    reason: str = ""
    approx: bool = False
    residual: dict = field(default_factory=dict)


def intersect_joint_limits(joints, vendor: Mapping, configured: Mapping | None = None):
    """Never widen runtime travel; optional robot YAML limits must cover every joint."""
    if not isinstance(vendor, Mapping) or (configured is not None and not isinstance(configured, Mapping)):
        raise ValueError("joint limits must be mappings")
    out = {}
    for name in joints:
        ranges = [vendor.get(name)]
        if configured is not None:
            ranges.append(configured.get(name))
        for bounds in ranges:
            if (
                not isinstance(bounds, list | tuple)
                or len(bounds) != 2
                or any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) for x in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"invalid or missing joint limits: {name}")
        lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
        if lo >= hi:
            raise ValueError(f"empty joint limit intersection: {name}")
        out[name] = (float(lo), float(hi))
    return out
