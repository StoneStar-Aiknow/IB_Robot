"""Evaluate the L3 gate's full rule (model + skill lexicon + wake prefixes) on a labelled corpus.

This is the acceptance gate from the design note (§5): a candidate model ships
only if command recall stays high while a useful share of chatter is dropped.
Exit code 2 means "do not ship this model".

    python3 tools/addressee/eval_model.py --model models/addressee/addressee_zh_v1.json \
        --corpus corpus.txt --threshold 0.10 --emit-lexicon

The lexicon is read from the skill catalog by default, so it cannot drift from
the skills the robot actually has; pass ``--no-lexicon`` to score the model alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_lexicon import catalog_lexicon  # noqa: E402  (after sys.path edit)


def _load_gate_module():
    """Import ``semantic_gate`` by path: the agent package __init__ pulls in ROS-side deps."""
    import importlib.util

    path = REPO_ROOT / "src" / "agent" / "ibrobot_agent" / "ibrobot_agent" / "semantic_gate.py"
    spec = importlib.util.spec_from_file_location("ibrobot_semantic_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_gate_module = _load_gate_module()
SemanticGate = _gate_module.SemanticGate
normalize_text = _gate_module.normalize_text


def load_rows(path: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            record = json.loads(line)
            rows.append((str(record["text"]).strip(), int(record["label"])))
            continue
        head, _, tail = line.rpartition(" ")
        if tail in {"是", "不是"} and head.strip():
            rows.append((head.strip(), 1 if tail == "是" else 0))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-recall", type=float, default=0.95, help="command (positive) recall floor")
    parser.add_argument("--max-false-accept", type=float, default=0.15, help="chatter (negative) leak ceiling")
    parser.add_argument("--no-lexicon", action="store_true")
    parser.add_argument("--show", type=int, default=8, help="how many errors to print per side")
    args = parser.parse_args()

    rows = load_rows(args.corpus)
    if not rows:
        print(f"no labelled rows in {args.corpus}", file=sys.stderr)
        return 1
    lexicon = () if args.no_lexicon else catalog_lexicon()
    gate = SemanticGate.load(args.model, threshold=args.threshold, lexicon=lexicon)

    kept_pos = 0
    dropped_neg = 0
    positives = negatives = 0
    dropped_positives: list[tuple[str, float, str]] = []
    leaked_negatives: list[tuple[str, float, str]] = []
    for text, label in rows:
        decision = gate.judge(text)
        if label:
            positives += 1
            if decision.rejected:
                dropped_positives.append((text, decision.score, decision.reason))
            else:
                kept_pos += 1
        else:
            negatives += 1
            if decision.rejected:
                dropped_neg += 1
            else:
                leaked_negatives.append((text, decision.score, decision.reason))

    recall = kept_pos / max(1, positives)
    specificity = dropped_neg / max(1, negatives)
    false_accept = 1.0 - specificity
    precision = kept_pos / max(1, kept_pos + (negatives - dropped_neg))
    accuracy = (kept_pos + dropped_neg) / len(rows)
    print(f"corpus={len(rows)} pos={positives} neg={negatives} threshold={args.threshold:.2f} "
          f"lexicon={len(lexicon)}")
    print("(rule metrics below are on this corpus; if it is the training corpus they are in-sample)")
    print(f"  accuracy      {accuracy:.4f}")
    print(f"  command recall {recall:.4f}  (dropped {positives - kept_pos}/{positives})")
    print(f"  precision     {precision:.4f}")
    print(f"  chatter leak  {false_accept:.4f}  (dropped {dropped_neg}/{negatives})")

    for title, examples in (("dropped commands", dropped_positives), ("chatter that got through", leaked_negatives)):
        if examples and args.show:
            print(f"\n{title} ({len(examples)}):")
            for text, score, reason in sorted(examples, key=lambda row: row[1])[: args.show]:
                print(f"  {score:0.3f}  {text}  ({reason})")

    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    cv = payload.get("cross_validation") or {}
    if cv:
        print("\nmodel-only, grouped 5-fold CV recorded in the artefact:")
        for threshold in sorted(cv, key=float):
            row = cv[threshold]
            print(f"  thr={threshold} acc={row['accuracy']:.4f} recall={row['recall']:.4f} "
                  f"chatter_dropped={row['specificity']:.4f}")

    reasons = sorted({reason.split(" ")[0] for text, _, reason in dropped_positives if normalize_text(text)})
    if reasons:
        print(f"\ndropped-command reasons: {reasons}")

    failures = []
    if recall < args.min_recall:
        failures.append(f"command recall {recall:.4f} < {args.min_recall:.2f}")
    if false_accept > args.max_false_accept:
        failures.append(f"chatter kept {false_accept:.4f} > {args.max_false_accept:.2f}")
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 2
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
