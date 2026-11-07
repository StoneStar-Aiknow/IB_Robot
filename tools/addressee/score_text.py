"""Score sentences with the L3 semantic gate model, for debugging a corpus.

Scores come from the runtime module itself (``ibrobot_agent.semantic_gate``), so
what you see here is exactly what the bridge decides -- there is no second copy
of the scoring rule to drift.

    python3 tools/addressee/score_text.py --model models/addressee/addressee_zh_v1.json \
        --lexicon 挥挥手 抓取 -- 灵犀灵犀挥挥手 今天开会 帮我抓取橘子

Without positional texts it reads lines from stdin.
"""

from __future__ import annotations

import argparse
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


SemanticGate = _load_gate_module().SemanticGate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--lexicon", nargs="*", default=[])
    parser.add_argument("texts", nargs="*")
    args = parser.parse_args()

    lexicon = tuple(args.lexicon) or catalog_lexicon()
    gate = SemanticGate.load(args.model, threshold=args.threshold, lexicon=lexicon)
    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    for text in texts:
        decision = gate.judge(text)
        verdict = "REJECT" if decision.rejected else "accept"
        print(f"{decision.score:0.3f}  {verdict:6s}  {text}  ({decision.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
