#!/usr/bin/env python3
"""Release-based latency accounting. Dropped frames count as deadline misses."""

import argparse
import csv
import json
import math
from itertools import pairwise
from pathlib import Path


def quantile(values, probability):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)]


def summarize(rows, deadline_ms):
    completed = [row for row in rows if row["event"] == "complete"]
    responses = [float(row["response_ms"]) for row in completed]
    service = [float(row["service_ms"]) for row in completed]
    late = sum((int(row["finish_ns"]) - int(row["release_ns"])) > deadline_ms * 1e6 for row in completed)
    drops = len(rows) - len(completed)
    rank = math.ceil(0.99 * len(rows))
    all_p99 = sorted(responses)[rank - 1] if rank and rank <= len(responses) else None
    return {
        "offered": len(rows),
        "completed": len(completed),
        "dropped": drops,
        "late_completed": late,
        "deadline_misses": late + drops,
        "deadline_miss_pct": 100 * (late + drops) / len(rows) if rows else None,
        "response_p50_ms": quantile(responses, 0.5),
        "response_p95_ms": quantile(responses, 0.95),
        "response_p99_ms": quantile(responses, 0.99),
        "response_max_ms": max(responses) if responses else None,
        "service_p99_ms": quantile(service, 0.99),
        "p99_all_releases_ms": all_p99,
        "p99_all_releases_defined": all_p99 is not None,
        "p99_all_releases_note": "drops treated as infinite; null when the 99th percentile is uncompleted",
    }


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def analyze_case(folder):
    config = json.loads((folder / "config.json").read_text())
    assert json.loads((folder / "completed.json").read_text())["status"] == "PASS"
    start = json.loads((folder / "go.json").read_text())["release_ns"]
    begin, end = start + config["trim"] * 1_000_000_000, start + config["duration"] * 1_000_000_000
    all_rows = read_rows(folder / "pose-pear.csv")
    assert [int(row["index"]) for row in all_rows] == list(range(config["duration"] * config["hz"]))
    for row in all_rows:
        assert row["event"] in ("complete", "drop")
        assert int(row["release_ns"]) == start + (int(row["index"]) * 1_000_000_000) // config["hz"]
        if row["event"] == "complete":
            assert int(row["finish_ns"]) >= int(row["start_ns"]) >= int(row["release_ns"])
    rows = [row for row in all_rows if begin <= int(row["release_ns"]) < end]
    result = {
        "case": folder.name,
        "config": config,
        "pear": summarize(rows, config["deadline_ms"]),
        "measurement_seconds": config["duration"] - config["trim"],
        "windows": [],
        "background": {},
    }
    for offset in range(config["trim"], config["duration"], 10):
        left, right = start + offset * 1_000_000_000, min(end, start + (offset + 10) * 1_000_000_000)
        window = [row for row in rows if left <= int(row["release_ns"]) < right]
        result["windows"].append({"offset_s": offset, **summarize(window, config["deadline_ms"])})
    result["pear"]["completed_hz"] = result["pear"]["completed"] / result["measurement_seconds"]
    for role in ("tts", "vl"):
        path = folder / f"{role}-chains.csv"
        if not path.exists():
            continue
        chains = read_rows(path)
        selected = [row for row in chains if begin <= int(row["release_ns"]) < end]
        completed = [row for row in selected if row["event"] == "complete"]
        finishes = sorted(
            int(row["finish_ns"])
            for row in chains
            if row["event"] == "complete" and begin <= int(row["finish_ns"]) < end
        )
        points = [begin, *finishes, end]
        result["background"][role] = {
            "offered_chains": len(selected),
            "completed_chains": len(completed),
            "dropped_chains": len(selected) - len(completed),
            "completed_inside_window": len(finishes),
            "max_completion_silence_s": max(b - a for a, b in pairwise(points)) / 1e9,
            "chain_response_p99_ms": quantile([float(row["response_ms"]) for row in completed], 0.99),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cases = []
    for folder in sorted((args.root / "cases").iterdir()):
        if (folder / "completed.json").exists():
            cases.append(analyze_case(folder))
    result = {
        "percentile": "nearest-rank",
        "deadline_origin": "scheduled release; finish includes synchronous host inference",
        "selection": "common window by scheduled release, including completions after window end",
        "cases": cases,
    }
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        p = case["pear"]
        print(
            case["case"],
            "n",
            p["offered"],
            "done",
            p["completed"],
            "drop",
            p["dropped"],
            "p99",
            p["response_p99_ms"],
            "all-p99",
            p["p99_all_releases_ms"],
            "miss%",
            p["deadline_miss_pct"],
        )


if __name__ == "__main__":
    main()
