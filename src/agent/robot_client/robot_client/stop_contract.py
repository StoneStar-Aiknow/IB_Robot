"""ROS-free re-export of the shared stop vocabulary.

The words themselves are owned by ``embodied_common.control_phrases``. This
module keeps the historical ``robot_client.stop_contract`` import path and the
``EXIT_*`` re-exports; it deliberately contains no vocabulary copy and no
``ImportError`` fallback, so the source cannot change with the import order.
"""

from embodied_common.control_phrases import STOP_PHRASES as STOP_PHRASES
from embodied_common.exit_codes import EXIT_ROS_UNAVAILABLE as EXIT_ROS_UNAVAILABLE
from embodied_common.exit_codes import EXIT_TIMEOUT as EXIT_TIMEOUT

# EXIT_ROS_UNAVAILABLE/EXIT_TIMEOUT remain importable here for the historical
# robot_client.stop_contract and robot_client.ros_bridge paths; the values are
# owned by embodied_common.exit_codes.


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
