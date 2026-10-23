"""ROS-free stop vocabulary shared with the installed Hermes adapter."""

try:
    from .control_phrases import STOP_PHRASES as _STOP_PHRASES
except ImportError:
    from embodied_common.control_phrases import STOP_PHRASES as _STOP_PHRASES

STOP_PHRASES = frozenset(_STOP_PHRASES)


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
