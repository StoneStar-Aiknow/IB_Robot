"""Constants shared by the X2 mirror session and the packages that wire or call it.

The session node (manipulation_execution), its launch builder (robot_config),
the Gateway (skill_library), the skill catalog and robot_skill_cli each need the
same names and timing. They live only here. Keep this module dependency-free:
importing it must not pull in any launch builder or ROS package.
"""

# The session node's name and its start/stop service (std_srvs/SetBool:
# true = start, false = stop), which the Gateway's mirror_session executor calls.
NODE_NAME = "mirror_session"
SESSION_SERVICE = f"/{NODE_NAME}/set_session"

# The commands a mirror_session skill binds in its catalog implementation
# (``binding: {command: start|stop}``).
COMMANDS = ("start", "stop")

# How long the session may take to answer stop: wrap-up plus the release to idle.
DEFAULT_STOP_TIMEOUT_SEC = 4.5
# The part of the stop timeout kept for the release to idle; the wrap-up gets the rest.
RELEASE_RESERVE_SEC = 2.0
# The Gateway's mirror_stop timeout must be at least the session's stop timeout
# plus this, so the session's answer still arrives in time.
STOP_TIMEOUT_MARGIN_SEC = 0.5
