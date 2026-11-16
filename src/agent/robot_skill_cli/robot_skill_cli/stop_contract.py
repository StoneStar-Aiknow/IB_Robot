"""ROS-free stop contract for the CLI, importable as ``robot_skill_cli.stop_contract``.

The vocabulary is the single ``embodied_common.control_phrases`` module. The
installed Hermes copy imports the sibling ``control_phrases.py`` that
``hermes_configure`` writes from that same module; there is no local copy and
no ``ImportError`` fallback that could silently switch sources at runtime.
"""

from .control_phrases import STOP_PHRASES as STOP_PHRASES


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
