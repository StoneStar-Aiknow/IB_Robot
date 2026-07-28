"""Command-line interface for raw observation batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_utils.observation_batch import (
    FieldSpec,
    extract_lerobot_observations,
    generate_random_observations,
    load_observation_batch,
    save_observation_batch,
)


def parse_field(value: str) -> FieldSpec:
    """Parse NAME=SHAPE,DTYPE,MIN,MAX, where SHAPE uses ``x`` separators."""
    try:
        name, body = value.split("=", 1)
        shape_text, dtype, minimum, maximum = body.rsplit(",", 3)
        shape = tuple(int(part) for part in shape_text.lower().split("x") if part)
        if not name or not shape:
            raise ValueError
        return FieldSpec(name=name, shape=shape, dtype=dtype, minimum=float(minimum), maximum=float(maximum))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "field must use NAME=SHAPE,DTYPE,MIN,MAX (for example state=6,float32,-1,1)"
        ) from exc


def _load_specs(path: str | None, inline: list[FieldSpec]) -> list[FieldSpec]:
    specs = list(inline)
    if path:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            payload = payload.get("fields")
        if not isinstance(payload, list):
            raise ValueError("JSON field spec must be a list or an object containing 'fields'")
        specs.extend(FieldSpec(**entry) for entry in payload)
    if not specs:
        raise ValueError("At least one --field or --spec is required")
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and inspect raw IB-Robot observation batches")
    subparsers = parser.add_subparsers(dest="command", required=True)
    random_parser = subparsers.add_parser("random", help="Generate deterministic random observations")
    random_parser.add_argument("--output", required=True)
    random_parser.add_argument("--samples", type=int, required=True)
    random_parser.add_argument("--seed", type=int, default=0)
    random_parser.add_argument("--field", action="append", type=parse_field, default=[])
    random_parser.add_argument("--spec", help="Optional JSON field specification")
    random_parser.add_argument("--force", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Extract observations from a local LeRobot dataset")
    dataset_parser.add_argument("--dataset-root", required=True)
    dataset_parser.add_argument("--output", required=True)
    dataset_parser.add_argument("--samples", type=int, required=True)
    dataset_parser.add_argument("--field", dest="fields", action="append")
    dataset_parser.add_argument("--seed", type=int, default=0)
    dataset_parser.add_argument(
        "--sampling",
        dest="strategy",
        choices=("episode-stratified", "global-random", "global-uniform"),
        default="episode-stratified",
    )
    dataset_parser.add_argument("--dataset-repo-id", dest="repo_id")
    dataset_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Print batch schema and provenance")
    inspect_parser.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "random":
            specs = _load_specs(args.spec, args.field)
            samples = generate_random_observations(specs, args.samples, seed=args.seed)
            save_observation_batch(
                args.output,
                samples,
                force=args.force,
                provenance={"source": "random", "seed": args.seed},
            )
        elif args.command == "dataset":
            samples, provenance, sample_provenance = extract_lerobot_observations(
                args.dataset_root,
                args.samples,
                fields=args.fields,
                seed=args.seed,
                strategy=args.strategy,
                repo_id=args.repo_id or f"local/{Path(args.dataset_root).resolve().name}",
            )
            save_observation_batch(
                args.output, samples, force=args.force, provenance=provenance, sample_provenance=sample_provenance
            )
        else:
            batch = load_observation_batch(args.path)
            output: dict[str, Any] = {
                "samples": len(batch),
                "fields": batch.fields,
                "provenance": batch.provenance,
            }
            print(json.dumps(output, indent=2, sort_keys=True))
    except (FileExistsError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    main()
