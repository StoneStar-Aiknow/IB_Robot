"""Golden comparison: the same input through two builds, array by array.

The first migration copies the demo as it is, so with the same input, chat
template, model, ABI and greedy decoding it must give the same tokens and
features within tolerance (design §12). The tolerances are frozen in a file
*before* the comparison and the report records that file's digest: a
difference is located first and judged after, and a tolerance is never
loosened to make a run pass.

Once cropping, the template, the enumeration or constrained decoding change,
tokens are no longer expected to match the old script; quality is compared
separately and this check stops applying.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ToleranceError(ValueError):
    """The tolerance file is missing, malformed or does not cover an array."""


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if not math.isfinite(value) or value < 0.0:
                raise ToleranceError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class Tolerances:
    """Which arrays must match exactly and which within a numeric tolerance."""

    tensors: Mapping[str, Tolerance]
    exact: frozenset[str]

    def __post_init__(self) -> None:
        both = self.exact & set(self.tensors)
        if both:
            raise ToleranceError(f"{sorted(both)} listed as both exact and toleranced")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Tolerances:
        unknown = set(raw) - {"tensors", "exact", "note"}
        if unknown:
            raise ToleranceError(f"unknown tolerance keys {sorted(unknown)}")
        tensors_raw = raw.get("tensors", {})
        exact_raw = raw.get("exact", [])
        if not isinstance(tensors_raw, Mapping) or not isinstance(exact_raw, list):
            raise ToleranceError("'tensors' must be a mapping and 'exact' a list")
        tensors = {}
        for name, spec in tensors_raw.items():
            if not isinstance(spec, Mapping) or set(spec) != {"atol", "rtol"}:
                raise ToleranceError(f"tensor {name!r} needs exactly atol and rtol")
            tensors[str(name)] = Tolerance(atol=float(spec["atol"]), rtol=float(spec["rtol"]))
        return cls(tensors=tensors, exact=frozenset(str(name) for name in exact_raw))

    @classmethod
    def load(cls, path: str | Path) -> Tolerances:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ToleranceError(f"cannot read tolerances from {path}: {error}") from error
        if not isinstance(raw, Mapping):
            raise ToleranceError("the tolerance file must hold a JSON object")
        return cls.from_mapping(raw)

    def covers(self, name: str) -> bool:
        return name in self.exact or name in self.tensors


def compare_exact(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Equal shape and equal values; for sequences, where they first part."""
    result: dict[str, object] = {
        "mode": "exact",
        "shape": [list(reference.shape), list(candidate.shape)],
    }
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    common = min(ref.size, cand.size)
    differ = np.flatnonzero(ref[:common] != cand[:common])
    first = int(differ[0]) if differ.size else (common if ref.size != cand.size else None)
    result["first_divergence"] = first
    result["passed"] = reference.shape == candidate.shape and first is None
    return result


def compare_tensor(reference: np.ndarray, candidate: np.ndarray, tolerance: Tolerance) -> dict[str, object]:
    """``|cand - ref| <= atol + rtol·|ref|`` everywhere, with the worst offender.

    Arrays that differ only in their leading length — per-step logits from a
    generation that stopped at a different token — are compared over the
    common prefix for diagnosis and still fail.
    """
    result: dict[str, object] = {
        "mode": "tolerance",
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
        "shape": [list(reference.shape), list(candidate.shape)],
    }
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    same_shape = ref.shape == cand.shape
    if not same_shape:
        if ref.shape[1:] != cand.shape[1:] or ref.ndim == 0:
            result["passed"] = False
            result["detail"] = "shapes differ beyond the leading length"
            return result
        steps = min(ref.shape[0], cand.shape[0])
        ref, cand = ref[:steps], cand[:steps]
        result["compared_leading"] = steps
    if ref.size == 0:
        result["passed"] = same_shape
        return result
    if not (np.isfinite(ref).all() and np.isfinite(cand).all()):
        result["passed"] = False
        result["detail"] = "non-finite values"
        return result
    diff = np.abs(cand - ref)
    allowed = tolerance.atol + tolerance.rtol * np.abs(ref)
    over = diff > allowed
    worst = int(np.argmax(diff))
    result.update(
        {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "violations": int(over.sum()),
            "worst_index": [int(i) for i in np.unravel_index(worst, diff.shape)],
            "passed": same_shape and not bool(over.any()),
        }
    )
    return result


def compare_archives(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    tolerances: Tolerances,
) -> dict[str, object]:
    """Compare every array; an array the tolerances do not cover is an error.

    Skipping an uncovered array would let a new output pass unexamined, so the
    tolerance file has to be extended — on purpose, before the run.
    """
    names = sorted(set(reference) | set(candidate))
    uncovered = [name for name in names if not tolerances.covers(name)]
    if uncovered:
        raise ToleranceError(f"no tolerance for {uncovered}; freeze one before comparing")
    arrays: dict[str, object] = {}
    for name in names:
        if name not in reference or name not in candidate:
            arrays[name] = {
                "passed": False,
                "detail": "missing from " + ("candidate" if name in reference else "reference"),
            }
        elif name in tolerances.exact:
            arrays[name] = compare_exact(np.asarray(reference[name]), np.asarray(candidate[name]))
        else:
            arrays[name] = compare_tensor(
                np.asarray(reference[name]), np.asarray(candidate[name]), tolerances.tensors[name]
            )
    return {"passed": all(item["passed"] for item in arrays.values()), "arrays": arrays}


__all__ = [
    "Tolerance",
    "ToleranceError",
    "Tolerances",
    "compare_archives",
    "compare_exact",
    "compare_tensor",
]
