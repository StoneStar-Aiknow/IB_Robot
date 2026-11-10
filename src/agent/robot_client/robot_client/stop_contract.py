"""ROS-free stop vocabulary shared with the installed Hermes adapter."""

from embodied_common.exit_codes import EXIT_ROS_UNAVAILABLE as EXIT_ROS_UNAVAILABLE
from embodied_common.exit_codes import EXIT_TIMEOUT as EXIT_TIMEOUT

try:
    from .control_phrases import STOP_PHRASES as _STOP_PHRASES
except ImportError:
    from embodied_common.control_phrases import STOP_PHRASES as _STOP_PHRASES

STOP_PHRASES = frozenset(_STOP_PHRASES)

# EXIT_ROS_UNAVAILABLE/EXIT_TIMEOUT remain importable here for the historical
# robot_client.stop_contract and robot_client.ros_bridge paths; the values are
# owned by embodied_common.exit_codes.


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
