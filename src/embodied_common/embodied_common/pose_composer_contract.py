"""Names shared by the pose composer and the packages that wire or call it.

The composer node (pose_composer), its launch builder, the Gateway's
pose_composer executor (skill_library), the skill catalog, robot_skill_cli and
the Agent's plan-time preview each need the same names. They live only here.
Keep this module dependency-free: importing it must not pull in any launch
builder or ROS package.
"""

from collections.abc import Mapping

NODE_NAME = "pose_composer"
# ibrobot_msgs/srv/ComposeArmPose: the Agent composes while it prepares a plan.
COMPOSE_SERVICE = f"/{NODE_NAME}/compose"
# ibrobot_msgs/action/ExecuteArmPose: the Gateway's executor runs a composition.
EXECUTE_ACTION = f"/{NODE_NAME}/execute"
EXECUTOR = "pose_composer"
# The only argument a pose_composer skill takes: the pose in the user's words.
ARGUMENT = "target_name"
MAX_TEXT_CHARS = 200
# A composition prepared for a plan is executed only if no older than this;
# otherwise the text is composed again at execution and the result says so.
PREPARED_TTL_SEC = 180.0
# ibrobot_msgs/srv/LookupArmPose: the Agent asks whether a confirmed remembered pose
# answers the user's words before it plans at all.
LOOKUP_SERVICE = f"/{NODE_NAME}/lookup"
# ibrobot_msgs/srv/ArmPoseFeedback: the user's verdict on the last pose of a conversation.
FEEDBACK_SERVICE = f"/{NODE_NAME}/feedback"


def lookup_enabled(config: Mapping) -> bool:
    """Read-only pose lookup is independent of writable memory and defaults on."""
    lookup = config.get("lookup", {})
    if not isinstance(lookup, Mapping):
        raise ValueError("pose_composer.lookup must be a mapping")
    if set(lookup) - {"enabled"}:
        raise ValueError("pose_composer.lookup has unknown keys")
    enabled = lookup.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("pose_composer.lookup.enabled must be a boolean")
    return enabled
