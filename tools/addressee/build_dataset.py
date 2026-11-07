"""Turn shadow logs into training data: a labelled corpus plus a queue to label.

Two inputs, one rule: a human label always wins, and nothing is ever labelled by
the gate itself. The shadow log is a sampling device -- it tells you which
sentences the gate is unsure about, which is exactly where a label is worth the
reviewer's time -- not a source of truth.

    # after a session's shadow log exists
    python3 tools/addressee/build_dataset.py --logs /var/log/ibrobot/semantic_gate.jsonl \
        --labelled corpus.txt --out corpus_next.txt --candidates to_label.txt

``to_label.txt`` is plain text, most-uncertain first; append `` 是`` or `` 不是``
to a line and move it into the labelled file. Lines that carry a ``label`` field
already (a reviewer wrote it back into the log) are used directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_labelled(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            record = json.loads(line)
            rows[str(record["text"]).strip()] = int(record["label"])
            continue
        head, _, tail = line.rpartition(" ")
        if tail in {"是", "不是"} and head.strip():
            rows[head.strip()] = 1 if tail == "是" else 0
    return rows


def read_logs(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", nargs="*", default=[], help="shadow JSONL files (one sample per line)")
    parser.add_argument("--labelled", default="", help="existing labelled corpus to carry forward")
    parser.add_argument("--out", default="corpus_next.txt")
    parser.add_argument("--candidates", default="to_label.txt")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--max-candidates", type=int, default=400)
    args = parser.parse_args()

    labelled = read_labelled(Path(args.labelled)) if args.labelled else {}
    unknown: dict[str, dict] = {}
    decided = 0
    for path in args.logs:
        for record in read_logs(Path(path)):
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            if "label" in record:  # a reviewer wrote the answer back into the log
                labelled[text] = int(record["label"])
                decided += 1
                continue
            if text in labelled:
                continue
            unknown.setdefault(text, record)

    out_lines = [f"{text} {'是' if label else '不是'}" for text, label in sorted(labelled.items())]
    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    ranked = sorted(unknown.items(), key=lambda item: abs(float(item[1].get("score", 0.5)) - args.threshold))
    candidates = [text for text, _ in ranked[: args.max_candidates]]
    Path(args.candidates).write_text("\n".join(candidates) + ("\n" if candidates else ""), encoding="utf-8")

    positives = sum(labelled.values())
    print(f"labelled={len(labelled)} (是 {positives} / 不是 {len(labelled) - positives}) -> {args.out}")
    print(f"from logs with a written-back label: {decided}")
    print(f"unlabelled in logs: {len(unknown)}; queued for labelling: {len(candidates)} -> {args.candidates}")
    if candidates:
        print("first to label: " + " | ".join(candidates[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
