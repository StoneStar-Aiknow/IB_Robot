"""Compare two ``.npz`` archives of model outputs under frozen tolerances.

Each archive holds the arrays one build produced for the same input — vision
features, per-step logits, token ids — saved by whatever ran the model. This
tool only compares them, so it needs no model, no NPU and no demo code.

Usage::

    source .shrc_local
    python3 -m visual_interaction.phase0.golden_compare \\
        --reference ref.npz --candidate cand.npz \\
        --tolerances tolerances.json --output measurement.json

The tolerance file is written and committed *before* the run; the report
records its digest so a later loosening shows. Exit status 0 means every array
passed, 1 means at least one did not, 2 means the comparison could not run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from visual_interaction.phase0.golden import ToleranceError, Tolerances, compare_archives
from visual_interaction.phase0.report import build_report, sha256_file, write_report

TOOL = "golden_compare"


def _load(path: Path) -> dict[str, np.ndarray]:
    # allow_pickle stays off: an archive is arrays, never objects to unpickle.
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True, help="frozen before the run; see module docs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="checkout whose commit is recorded")
    args = parser.parse_args(argv)

    try:
        tolerances = Tolerances.load(args.tolerances)
        result = compare_archives(_load(args.reference), _load(args.candidate), tolerances)
    except (OSError, ValueError) as error:
        # ToleranceError is a ValueError; a malformed archive raises either.
        kind = "tolerances" if isinstance(error, ToleranceError) else "archive"
        print(f"{TOOL}: cannot compare ({kind}): {error}", file=sys.stderr)
        return 2

    report = build_report(
        tool=TOOL,
        conditions={},
        results=result,
        repo=args.repo,
        assets={
            "reference": sha256_file(args.reference),
            "candidate": sha256_file(args.candidate),
            "tolerances": sha256_file(args.tolerances),
        },
    )
    write_report(args.output, report)
    failed = sorted(name for name, item in result["arrays"].items() if not item["passed"])
    print(f"{TOOL}: {'passed' if result['passed'] else 'FAILED ' + ', '.join(failed)} -> {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
