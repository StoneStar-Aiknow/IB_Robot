"""Score sentences with the intent gate exactly as the voice bridge builds it, without ROS.

    rig_env.sh python3 tools/gate_probe.py [--followup] <robot config> <intent model> "句子一" "句子二" ...

Prints pass/REJECT, the model score, the deciding rule and, for an admitted sentence, its
grade (``conversation`` is answer-only). The session is assumed open, as for a follow-up
after the wake phrase; ``--followup`` also assumes the robot has just finished speaking.
The configured text corrections and rewrites are applied first, as in the bridge.
"""

from __future__ import annotations

import sys

from ibrobot_agent.semantic_gate import SemanticGate
from ibrobot_agent.voice_bridge import VoiceBridgeConfig
from robot_config.loader import _load_robot_section


def main() -> None:
    args = sys.argv[1:]
    followup = "--followup" in args
    args = [arg for arg in args if arg != "--followup"]
    if len(args) < 3:
        raise SystemExit(__doc__)
    bridge = _load_robot_section(args[0])[1]["embodied"]["agent"]["voice_bridge"]
    corrector = VoiceBridgeConfig(
        robot_scope="gate_probe",
        text_corrections=tuple(
            (term, tuple(variants)) for term, variants in bridge.get("text_corrections", {}).items()
        ),
        text_rewrites=tuple(tuple(pair) for pair in bridge.get("text_rewrites", ())),
    )
    conversation = bool(bridge.get("semantic_gate_conversation", False))
    gate = SemanticGate.load(
        args[1],
        threshold=bridge.get("semantic_gate_threshold", 0.1),
        lexicon=tuple(bridge.get("semantic_gate_lexicon", ())),
        dialogue_requests=tuple(bridge.get("semantic_gate_dialogue_requests", ())),
        wake_prefixes=tuple(bridge.get("semantic_gate_wake_prefixes", ())) + tuple(bridge.get("wake_phrases", ())),
        model_only_threshold=bridge.get("semantic_gate_model_only_threshold", 0.9),
    )
    for heard in args[2:]:
        text = corrector.correct(heard)
        decision = gate.judge(text, session_open=True, followup_open=followup, conversation=conversation)
        verdict = "REJECT" if decision.rejected else "pass  "
        grade = "" if decision.rejected else f" ({decision.grade})"
        sent = f" -> {decision.admitted_text}" if decision.admitted_text else ""
        print(f"{verdict} {decision.score:.3f} {text}{sent}  [{decision.reason}]{grade}")


if __name__ == "__main__":
    main()
