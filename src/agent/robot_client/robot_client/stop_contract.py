"""ROS-free stop vocabulary shared with the installed Hermes adapter."""

try:
    from .control_phrases import STOP_PHRASES as _STOP_PHRASES
except ImportError:
    from embodied_common.control_phrases import STOP_PHRASES as _STOP_PHRASES

STOP_PHRASES = frozenset(_STOP_PHRASES)

# Keep the public ROS transport error contract independent of the CLI output
# layer while preserving the historical process exit values.
EXIT_ROS_UNAVAILABLE = 4
EXIT_TIMEOUT = 124


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
