"""Summarise VL segment timings from the demo's own run log.

The board demo already records, per round, the preprocessing, vision OM,
prefill, per-token decode and speak-stage times into ``out/live/<ts>/talk.jsonl``.
This tool reads that file and reports the distributions §12 asks for, rather
than importing and re-timing the demo: its licence is unconfirmed, so its code
stays where it is, and timings taken from its own log describe exactly what it
ran.

Usage::

    source .shrc_local
    python3 -m visual_interaction.phase0.vl_segments \\
        --log talk.jsonl --label idle --output measurement.json

The outcome split (none/uncertain/valid) is derived from the fields the log
already carries: ``nobody`` marks a no-person round, an empty ``text`` after
retries marks a round that produced nothing sayable, and ``tries`` gives the
retry distribution the design flagged as a latency risk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from visual_interaction.phase0.report import build_report, sha256_file, write_report
from visual_interaction.phase0.stats import counts, summarise

TOOL = "vl_segments"

# Fields the demo writes per round that this tool reads. Missing ones are
# tolerated (an older log), an unreadable line is not.
_SEGMENTS = {
    "preprocess_ms": "pre_ms",
    "vision_ms": "vision_ms",
    "observe_prefill_ms": "obs_prefill_ms",
    "observe_per_token_ms": "obs_decode_ms",
    "observe_total_s": "observe_s",
    "speak_total_s": "speak_s",
    "round_total_s": "total_s",
    "capture_to_print_s": "capture_to_print_s",
    "frame_age_ms": "frame_age_ms",
}


def read_rounds(path: Path) -> Iterator[dict[str, object]]:
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield record


def outcome_of(record: dict[str, object]) -> str:
    """How this round ended, in the vocabulary the design reports.

    The demo answers in free text, so this is its own classification mapped on:
    a no-person round is the demo's ``nobody``, and a round that retried out
    without usable text is the closest it has to "nothing worth saying".
    """
    if record.get("nobody"):
        return "no_person"
    if not str(record.get("text", "")).strip():
        return "no_usable_text"
    return "spoke"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="talk.jsonl from a demo run")
    parser.add_argument("--label", required=True, help="the condition this run was under")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--skip", type=int, default=1, help="leading rounds to discard as warm-up")
    args = parser.parse_args(argv)

    try:
        rounds = list(read_rounds(args.log))
    except (OSError, ValueError) as error:
        print(f"{TOOL}: {error}", file=sys.stderr)
        return 2
    measured = rounds[args.skip :]
    if not measured:
        print(f"{TOOL}: {args.log} has no rounds after skipping {args.skip}", file=sys.stderr)
        return 2

    segments = {
        name: summarise(record[key] for record in measured if isinstance(record.get(key), int | float))
        for name, key in _SEGMENTS.items()
    }
    outcomes = counts(outcome_of(record) for record in measured)
    spoke = [record for record in measured if outcome_of(record) == "spoke"]

    results = {
        "rounds": len(measured),
        "skipped_warmup": min(args.skip, len(rounds)),
        "segments": segments,
        "outcomes": outcomes,
        # The retry loop the design called a latency risk: a worst-case round
        # runs the speak stage several times over.
        "speak_tries": counts(record.get("tries") for record in measured if record.get("tries") is not None),
        "observe_tokens": summarise(
            record["obs_tokens"] for record in measured if isinstance(record.get("obs_tokens"), int | float)
        ),
        "speak_tokens": summarise(
            record["speak_tokens"] for record in spoke if isinstance(record.get("speak_tokens"), int | float)
        ),
        "frames_in_window": summarise(
            record["frames_in_window"] for record in measured if isinstance(record.get("frames_in_window"), int | float)
        ),
    }
    report = build_report(
        tool=TOOL,
        conditions={"label": args.label, "log_name": args.log.name},
        results=results,
        repo=args.repo,
        assets={"run_log": sha256_file(args.log)},
        notes=[
            "Timings are the demo's own per-round records, not a re-timing by this tool.",
            "Outcome labels are mapped from the demo's free-text answer, which asks a different "
            "question than this feature's closed schema; they are not the contract's kinds.",
        ],
    )
    write_report(args.output, report)

    print(f"  rounds={len(measured)} outcomes={outcomes}")
    for name in ("preprocess_ms", "vision_ms", "observe_prefill_ms", "observe_per_token_ms"):
        stat = segments[name]
        if stat.get("n"):
            print(f"  {name:22s} p50={stat['p50']:8.2f} p95={stat['p95']:8.2f} max={stat['max']:8.2f}")
    for name in ("observe_total_s", "speak_total_s", "round_total_s", "capture_to_print_s"):
        stat = segments[name]
        if stat.get("n"):
            print(f"  {name:22s} p50={stat['p50']:8.3f}s p95={stat['p95']:8.3f}s max={stat['max']:8.3f}s")
    if results["speak_tries"]:
        print(f"  speak retries: {results['speak_tries']}")
    print(f"{TOOL}: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
