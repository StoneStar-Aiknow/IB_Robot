"""Evaluate timestamped single-target estimates against optional annotations."""

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix.casefold() == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return list(csv.DictReader(stream))


def evaluate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("tracking results are empty")
    timestamps = [int(row["stamp_ns"]) for row in rows]
    if timestamps != sorted(timestamps):
        raise ValueError("tracking results must be ordered by stamp_ns")
    sessions = {str(row["session_id"]) for row in rows}
    objects = {str(row["object_id"]) for row in rows}
    measured = [row for row in rows if str(row.get("measured", "")).casefold() in {"1", "true", "yes"}]
    annotated = [row for row in rows if row.get("gt_x") not in {None, ""} and row.get("gt_y") not in {None, ""}]
    errors = [
        math.hypot(float(row["x"]) - float(row["gt_x"]), float(row["y"]) - float(row["gt_y"])) for row in annotated
    ]
    duration_s = max(0.0, (timestamps[-1] - timestamps[0]) / 1e9)
    return {
        "row_count": len(rows),
        "duration_s": duration_s,
        "output_rate_hz": (len(rows) - 1) / duration_s if duration_s > 0.0 else 0.0,
        "measured_ratio": len(measured) / len(rows),
        "session_count": len(sessions),
        "object_id_count": len(objects),
        "identity_retained": len(sessions) == 1 and len(objects) == 1,
        "annotated_count": len(annotated),
        "position_rmse_m": math.sqrt(sum(error**2 for error in errors) / len(errors)) if errors else None,
        "position_max_error_m": max(errors) if errors else None,
    }


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="CSV or JSONL frame diagnostics")
    parser.add_argument("--output", type=Path, help="write the summary as JSON")
    parsed = parser.parse_args(args)
    summary = evaluate(_read_rows(parsed.results))
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if parsed.output:
        parsed.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
