"""Names shared by the pose composer and the packages that wire or call it.

The composer node (pose_composer), its launch builder, the Gateway's
pose_composer executor (skill_library), the skill catalog, robot_skill_cli and
the Agent's plan-time preview each need the same names. They live only here.
Keep this module dependency-free: importing it must not pull in any launch
builder or ROS package.
"""

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
