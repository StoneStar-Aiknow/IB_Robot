# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Licensed under the Mulan PSL v2.
"""Build strict PI0.5 denoising schedules from runtime curvature logs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from inference_manifest.json_utils import loads_json_strict
from inference_service.pi05_schedule import PI05DenoisingSchedule, parse_pi05_schedule, write_pi05_schedule

_CURVATURE_RECORD_KEYS = frozenset({"schedule", "curvature_scores"})


def load_curvature_log(path: str | Path) -> tuple[PI05DenoisingSchedule, np.ndarray]:
    """Load JSONL records and average scores for one strict dense schedule."""

    log_path = Path(path).expanduser().resolve(strict=True)
    dense_schedule = None
    score_rows = []
    with log_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source = f"{log_path}:{line_number}"
            value = loads_json_strict(line, source)
            if type(value) is not dict or set(value) != _CURVATURE_RECORD_KEYS:
                raise ValueError(f"{source} must contain exactly {sorted(_CURVATURE_RECORD_KEYS)}")
            schedule = parse_pi05_schedule(value["schedule"], source=f"{source} schedule")
            if dense_schedule is None:
                dense_schedule = schedule
            elif schedule != dense_schedule:
                raise ValueError(f"{source} uses a different dense schedule")

            scores = value["curvature_scores"]
            if type(scores) is not list or len(scores) != schedule.step_count:
                raise ValueError(f"{source} curvature_scores must contain {schedule.step_count} values")
            if any(
                type(score) not in (int, float) or type(score) is bool or not math.isfinite(score) or score < 0.0
                for score in scores
            ):
                raise ValueError(f"{source} curvature_scores must contain only finite non-negative numbers")
            score_rows.append(np.asarray(scores, dtype=np.float64))

    if dense_schedule is None:
        raise ValueError(f"No curvature records found in {log_path}")
    return dense_schedule, np.mean(np.stack(score_rows, axis=0), axis=0)


def build_curvature_schedule(
    dense_schedule: PI05DenoisingSchedule,
    scores: np.ndarray,
    *,
    num_steps: int,
    base: float = 1e-3,
    eta: float = 1.0,
    name: str = "curvature",
) -> PI05DenoisingSchedule:
    """Allocate Euler nodes so each step covers approximately equal curvature mass."""

    if not isinstance(dense_schedule, PI05DenoisingSchedule):
        raise TypeError("dense_schedule must be a PI05DenoisingSchedule")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if type(base) not in (int, float) or type(base) is bool or not math.isfinite(base) or base < 0.0:
        raise ValueError("base must be a finite non-negative number")
    if type(eta) not in (int, float) or type(eta) is bool or not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be a finite positive number")

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (dense_schedule.step_count,):
        raise ValueError(f"scores must contain {dense_schedule.step_count} values")
    if not np.all(np.isfinite(score_array)) or np.any(score_array < 0.0):
        raise ValueError("scores must contain only finite non-negative values")

    dense_timesteps = np.asarray(dense_schedule.timesteps, dtype=np.float64)
    interval_dt = np.abs(np.diff(dense_timesteps))
    density = float(base) + np.power(score_array, float(eta))
    mass = density * interval_dt
    total_mass = float(np.sum(mass))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("curvature mass must be finite and positive")

    cdf = np.concatenate(([0.0], np.cumsum(mass) / total_mass))
    targets = np.linspace(0.0, 1.0, num_steps + 1)
    timesteps = np.interp(targets, cdf, dense_timesteps)
    timesteps[0] = 1.0
    timesteps[-1] = 0.0
    return PI05DenoisingSchedule(name=name, timesteps=tuple(float(value) for value in timesteps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Curvature JSONL written by the Ascend runtime")
    parser.add_argument("--num-steps", nargs="+", required=True, type=int, help="Output schedule step count(s)")
    parser.add_argument("--base", type=float, default=1e-3, help="Density floor added to every dense interval")
    parser.add_argument("--eta", type=float, default=1.0, help="Curvature score exponent")
    parser.add_argument("--name-prefix", default="curvature", help="Output schedule name prefix")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for strict schedule JSON files")
    parser.add_argument("--force", action="store_true", help="Replace existing output schedule files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dense_schedule, scores = load_curvature_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for num_steps in args.num_steps:
        name = f"{args.name_prefix}_{num_steps}"
        schedule = build_curvature_schedule(
            dense_schedule,
            scores,
            num_steps=num_steps,
            base=args.base,
            eta=args.eta,
            name=name,
        )
        destination = args.output_dir / f"{name}.json"
        if destination.exists() and not args.force:
            raise SystemExit(f"Refusing to replace {destination}; pass --force")
        write_pi05_schedule(schedule, destination)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
