"""Shared stowed-arm envelope and rest-window logic (lekiwi-nav-grasp-sim-
platforms review rework; D7/D5).

The navigation gates record the arm as ``(sim_stamp, {joint: value})``
samples — lekiwi_robot's chassis gate and stop cross gate, robot_navigation's
lidar-odometry gate and robot_config's navigation e2e all use the same shape.
The freshness rules live here, as pure functions over those samples, so they
are unit-testable without a simulator and identical across the gates (the
review found each gate checking only its internal inter-sample gaps, leaving
the interval's leading and trailing edges uncovered).

Both verdicts share one freshness rule, applied to the WHOLE interval:

- the first sample must arrive within ``max_gap`` of the interval start,
- no internal inter-sample gap may exceed ``max_gap``,
- the last sample must be within ``max_gap`` of the interval end.

Every counted sample must carry EVERY required joint (a missing joint field
is a failure, never a skip) with finite values. ``envelope_verdict`` bounds
every joint within ``tolerance`` of the reset pose; ``rest_verdict``
accumulates the worst |velocity| over the whole window (a moving start
followed by a still final frame is still motion, and NO observations is a
failure, not a clean zero).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

#: One recorded arm sample: (simulation stamp, {joint name: value}).
Sample = tuple[float, dict[str, float]]


def _finite(value) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _freshness_violations(samples: list[Sample], begin: float, end: float, max_gap: float) -> list[str]:
    """Why the sample stream does not freshly cover ``[begin, end]``."""
    violations: list[str] = []
    if not samples:
        return ["no arm samples in the window"]
    stamps = [stamp for stamp, _positions in samples]
    if stamps[0] - begin > max_gap:
        violations.append(f"first sample {stamps[0] - begin:.2f} s after the window start")
    if end - stamps[-1] > max_gap:
        violations.append(f"last sample {end - stamps[-1]:.2f} s before the window end")
    worst_gap = max((b - a for a, b in zip(stamps, stamps[1:], strict=False)), default=0.0)
    if worst_gap > max_gap:
        violations.append(f"internal sample gap {worst_gap:.2f} s")
    return violations


def _checked_samples(
    samples: Sequence[Sample], joints: Sequence[str], violations: list[str]
) -> list[tuple[float, dict[str, float]]]:
    """The samples carrying every joint, plus one violation entry per problem.

    A sample missing a joint field or carrying a non-finite value is a
    failure: it must never be silently skipped (the envelope would then be
    asserted over less than the whole arm).
    """
    checked: list[tuple[float, dict[str, float]]] = []
    for stamp, positions in samples:
        if not isinstance(positions, dict):
            violations.append(f"sample at {stamp:.2f} s is not a joint mapping")
            continue
        missing = [joint for joint in joints if joint not in positions]
        if missing:
            violations.append(f"sample at {stamp:.2f} s missing joints {missing}")
            continue
        nonfinite = [joint for joint in joints if not _finite(positions[joint])]
        if nonfinite:
            violations.append(f"sample at {stamp:.2f} s has non-finite values for {nonfinite}")
            continue
        checked.append((stamp, positions))
    return checked


def envelope_verdict(
    samples: Iterable[Sample],
    begin: float,
    end: float,
    *,
    reset: dict[str, float],
    tolerance: float,
    max_gap: float,
) -> dict:
    """The stowed-arm envelope verdict over ``[begin, end]`` of simulation time.

    ``within`` requires: fresh coverage of the whole interval (the shared
    freshness rule), every sample carrying every reset joint with finite
    values, and every joint within ``tolerance`` of its reset pose.
    """
    joints = sorted(reset)
    window = sorted((stamp, positions) for stamp, positions in samples if begin <= stamp <= end)
    violations = _freshness_violations(window, begin, end, max_gap)
    checked = _checked_samples(window, joints, violations)
    worst = max(
        (abs(positions[joint] - value) for _stamp, positions in checked for joint, value in reset.items()),
        default=float("inf"),
    )
    if checked and worst > tolerance:
        violations.append(f"worst {worst:.3f} rad of the reset pose (tolerance {tolerance})")
    stamps = [stamp for stamp, _positions in window]
    return {
        "within": not violations and bool(checked),
        "violations": violations,
        "worst_rad": worst,
        "samples": len(checked),
        "max_gap_sec": max((b - a for a, b in zip(stamps, stamps[1:], strict=False)), default=0.0),
    }


def rest_verdict(
    velocity_samples: Iterable[Sample],
    begin: float,
    end: float,
    *,
    joints: Sequence[str],
    max_velocity: float,
    max_gap: float,
) -> dict:
    """The at-rest verdict over ``[begin, end]`` of simulation time.

    ``at_rest`` requires: fresh coverage of the whole interval (the shared
    freshness rule), every sample carrying every joint with finite velocity
    values, and the worst |velocity| ACCUMULATED over the whole window
    staying within ``max_velocity``. No observations is a failure (the
    measured zero of an empty stream must never read as "at rest").
    """
    window = sorted((stamp, velocities) for stamp, velocities in velocity_samples if begin <= stamp <= end)
    violations = _freshness_violations(window, begin, end, max_gap)
    checked = _checked_samples(window, joints, violations)
    worst = max(
        (abs(velocities[joint]) for _stamp, velocities in checked for joint in joints),
        default=float("inf"),
    )
    if checked and worst > max_velocity:
        violations.append(f"worst |velocity| {worst:.3e} rad/s (limit {max_velocity})")
    return {
        "at_rest": not violations and bool(checked),
        "violations": violations,
        "worst_velocity_rad_s": worst,
        "samples": len(checked),
    }
