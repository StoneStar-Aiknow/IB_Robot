import csv
import json
import math
from pathlib import Path

root = Path("/tmp/rtinfer-default-root-review/final-v3-data/final-v3")


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def q(vals, p):
    return sorted(vals)[max(0, math.ceil(len(vals) * p) - 1)] if vals else None


result = []
for case in sorted(root.iterdir()):
    if not case.is_dir() or not (case / "campaign-complete.json").exists():
        continue
    cfg = json.loads((case / "config.json").read_text())
    complete = json.loads((case / "campaign-complete.json").read_text())
    assert complete["status"] == "executed" and complete["numeric_checks"] == "passed"
    exits = complete["process_exits"]
    assert all(p["returncode"] == 0 for p in exits)
    if cfg["mode"] == "soft":
        assert len(exits) == 5
        st = json.loads((case / "runtime-after-stop/state.json").read_text())
        assert st["status"] == "stopped" and not st["clients"]
        assert st["prefix"] == "/opt/rt-infer/2026.11.06"
    for rnd in sorted(case.glob("round-*")):
        go = json.loads((rnd / "go.json").read_text())
        start = go["release_ns"]
        lo = start + int(go["trim_sec"] * 1e9)
        hi = start + int(go["duration"] * 1e9)
        report = {"case": case.name, "mode": cfg["mode"], "round": rnd.name, "roles": {}}
        for role, hz in [("pear", 18), ("yolox", 1), ("tts", 2 / 15), ("vl", 2 / 5)]:
            path = rnd / (role + ".csv")
            if not path.exists():
                continue
            data = [r for r in rows(path) if r["model"] == role]
            count = math.ceil(go["duration"] * hz)
            assert len(data) == count, (case.name, role, len(data), count)
            assert [int(r["index"]) for r in data] == list(range(count))
            num, den = {"pear": (18, 1), "yolox": (1, 1), "tts": (2, 15), "vl": (2, 5)}[role]
            for r in data:
                assert int(r["release_ns"]) == start + int(r["index"]) * 10**9 * den // num
                assert r["event"] in ("complete", "drop", "incomplete")
                if r["event"] == "complete":
                    assert int(r["finish_ns"]) >= int(r["start_ns"]) >= int(r["release_ns"])
                    assert abs(float(r["response_ms"]) - (int(r["finish_ns"]) - int(r["release_ns"])) / 1e6) < 1e-8
            selected = [r for r in data if lo <= int(r["release_ns"]) < hi]
            values = [(int(r["finish_ns"]) - int(r["release_ns"])) / 1e6 for r in selected if r["event"] == "complete"]
            drops = len(selected) - len(values)
            rank = math.ceil(0.99 * len(selected))
            finishes = sorted(
                int(r["finish_ns"]) for r in data if r["event"] == "complete" and lo <= int(r["finish_ns"]) < hi
            )
            points = [lo, *finishes, hi]
            report["roles"][role] = {
                "offered": len(selected),
                "completed": len(values),
                "drops": drops,
                "window_finished": len(finishes),
                "window_released_and_finished": sum(
                    r["event"] == "complete" and lo <= int(r["finish_ns"]) < hi for r in selected
                ),
                "p50_ms": q(values, 0.5),
                "completed_p99_ms": q(values, 0.99),
                "all_release_p99_ms": sorted(values)[rank - 1] if rank <= len(values) else None,
                "late_55_56ms": sum(v > 1000 / 18 for v in values) if role == "pear" else None,
                "longest_silence_s": max(b - a for a, b in zip(points, points[1:], strict=False)) / 1e9,
            }
        result.append(report)
for r in result:
    p = r["roles"]["pear"]
    print(
        r["case"],
        "p99",
        p["completed_p99_ms"],
        "all99",
        p["all_release_p99_ms"],
        "drop",
        p["drops"],
        "late",
        p["late_55_56ms"],
        [
            (k, v["window_finished"], v["offered"], v["longest_silence_s"])
            for k, v in r["roles"].items()
            if k in ("tts", "vl")
        ],
    )
Path("/tmp/rtinfer-default-root-review/final-v3-independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print("PASS INTEGER TIMESTAMP/ALL RELEASE/EXIT/STOP AUDIT", len(result), "cases")
