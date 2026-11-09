"""Score sentences with the intent gate exactly as the voice bridge builds it, without ROS.

    rig_env.sh python3 tools/gate_probe.py <robot config> <intent model> "句子一" "句子二" ...

Prints pass/REJECT, the model score and the deciding rule. The session is assumed open, as
for a follow-up after the wake phrase.
"""

from __future__ import annotations

import sys

from ibrobot_agent.semantic_gate import SemanticGate
from robot_config.loader import _load_robot_section


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    bridge = _load_robot_section(sys.argv[1])[1]["embodied"]["agent"]["voice_bridge"]
    gate = SemanticGate.load(
        sys.argv[2],
        threshold=bridge.get("semantic_gate_threshold", 0.1),
        lexicon=tuple(bridge.get("semantic_gate_lexicon", ())),
        dialogue_requests=tuple(bridge.get("semantic_gate_dialogue_requests", ())),
        wake_prefixes=tuple(bridge.get("semantic_gate_wake_prefixes", ())) + tuple(bridge.get("wake_phrases", ())),
        model_only_threshold=bridge.get("semantic_gate_model_only_threshold", 0.9),
    )
    for text in sys.argv[3:]:
        decision = gate.judge(text, session_open=True)
        verdict = "REJECT" if decision.rejected else "pass  "
        print(f"{verdict} {decision.score:.3f} {text}  [{decision.reason}]")


if __name__ == "__main__":
    main()
