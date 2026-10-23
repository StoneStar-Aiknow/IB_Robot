"""Summary statistics in the shape phase zero reports them: p50, p95, max.

Pure Python on purpose, so a summary means the same thing on every host and a
test does not need numpy to pin it. Percentiles use linear interpolation
between closest ranks, which is numpy's default, so a figure here matches one
computed from the raw samples with ``numpy.percentile``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

# Below this, a distribution is exploratory only (design §12): fine for
# choosing a direction, not for a release P95 or a stop threshold.
MIN_EXPLORATORY_RUNS = 30


def percentile(values: Iterable[float], q: float) -> float:
    """The ``q``-th percentile (0–100) by linear interpolation."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile of no values")
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarise(values: Iterable[float]) -> dict[str, float | int | bool]:
    """Count, mean, p50, p95, max, and whether the sample is too small to cite.

    An empty sample is reported as ``{"n": 0}`` rather than raising, because
    "this condition never produced a value" is itself a result — a run where
    every round was quiet has no speak-stage timings, and that must show.
    """
    samples = [float(value) for value in values]
    if not samples:
        return {"n": 0}
    for value in samples:
        if not math.isfinite(value):
            raise ValueError("a timing must be finite")
    return {
        "n": len(samples),
        "mean": sum(samples) / len(samples),
        "min": min(samples),
        "p50": percentile(samples, 50.0),
        "p95": percentile(samples, 95.0),
        "max": max(samples),
        "below_exploratory_minimum": len(samples) < MIN_EXPLORATORY_RUNS,
    }


def counts(values: Iterable[object]) -> dict[str, int]:
    """How often each value occurred, keyed by its text, in a stable order."""
    tally: dict[str, int] = {}
    for value in values:
        key = str(value)
        tally[key] = tally.get(key, 0) + 1
    return dict(sorted(tally.items()))


__all__ = ["MIN_EXPLORATORY_RUNS", "counts", "percentile", "summarise"]
