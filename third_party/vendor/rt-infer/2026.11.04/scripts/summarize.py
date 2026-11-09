#!/usr/bin/env python3
"""Summarize all rows; combine only CSVs with the same experiment conditions."""
import argparse
import csv
import json
import math

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", nargs="+")
args = parser.parse_args()
rows = []
for name in args.csv:
    with open(name, newline="", encoding="utf-8") as stream:
        rows.extend(csv.DictReader(stream))
values = sorted(float(row["response_ms"]) for row in rows)
if not values:
    raise SystemExit("No completed samples")
result = {"samples": len(values), "mean_ms": sum(values) / len(values), "max_ms": values[-1],
          "late_fraction": sum(int(row["late"]) for row in rows) / len(rows)}
for p in (50, 95, 99):
    result[f"p{p}_ms"] = values[max(0, math.ceil(len(values) * p / 100) - 1)]
result["timing"] = "scheduled release to execute+stream-sync completion; excludes copies, preprocessing and ROS"
result["periodic_policy"] = "one outstanding inference; overdue releases execute serially, no automatic dropping"
print(json.dumps(result, indent=2))
