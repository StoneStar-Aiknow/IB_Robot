"""Evaluate DWB-versus-MPPI evidence against the navigation promotion gate."""

import argparse
import json
import math
from pathlib import Path

import yaml


def _success_rate(runs):
    return sum(run["goal_result"] == "succeeded" for run in runs) / len(runs)


def _p95(values):
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(index, 0)]


def evaluate_regression(baseline, candidate, gate):
    """Return gate failures for two non-empty lists of measured run dictionaries."""
    if not baseline or not candidate:
        raise ValueError("baseline and candidate evidence must contain at least one run")

    failures = []
    success_delta = _success_rate(candidate) - _success_rate(baseline)
    if success_delta < gate["success_rate_delta_min"]:
        failures.append(f"success rate delta {success_delta:.3f} is below the required minimum")

    comparisons = (
        ("collision_count", "additional_collisions_max"),
        ("sustained_oscillation_count", "additional_sustained_oscillations_max"),
    )
    for metric, gate_key in comparisons:
        delta = sum(run[metric] for run in candidate) - sum(run[metric] for run in baseline)
        if delta > gate[gate_key]:
            failures.append(f"{metric} delta {delta} exceeds {gate[gate_key]}")

    command_violations = sum(run["command_limit_violation_count"] for run in candidate)
    if command_violations > gate["command_limit_violations_max"]:
        failures.append(f"command limit violations {command_violations} exceed the gate")

    computation_times = [sample for run in candidate for sample in run["controller_computation_ms"]]
    if not computation_times:
        failures.append("candidate evidence has no controller computation samples")
    elif _p95(computation_times) >= gate["controller_computation_p95_ms_max_exclusive"]:
        failures.append(f"controller computation p95 {_p95(computation_times):.3f} ms misses the exclusive gate")

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    gate = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))["promotion_gate"]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["runs"]
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))["runs"]
    failures = evaluate_regression(baseline, candidate, gate)
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
