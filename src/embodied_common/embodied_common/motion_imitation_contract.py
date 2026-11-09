"""Non-motion control boundary of the resident mirror perception stream.

The Gateway pauses the stream around robot motion; the stream answers the pause
service and republishes its paused state on ``PAUSED_STATE_TOPIC`` (latched, so a
late or restarted reader sees the current value). The other camera readers of the
mirror -- hand perception, the session's cover input and the operator's rear
camera -- follow that state, so one acknowledged pause frees every camera reader.
"""

PAUSE_SERVICE = "/motion_imitation_stream/set_paused"
# std_msgs/Bool, reliable + transient local, depth 1: True while any owner holds a pause.
PAUSED_STATE_TOPIC = "/motion_imitation_stream/perception_paused"

# Which skills the Gateway pauses the stream for.
PAUSE_SCOPE_MANIPULATION = "manipulation"  # return_to_task_pose, pick and place
PAUSE_SCOPE_ROBOT_MOTION = "robot_motion"  # every moves_robot skill except the mirror's own
PAUSE_SCOPES = (PAUSE_SCOPE_MANIPULATION, PAUSE_SCOPE_ROBOT_MOTION)


def _flag(config, key):
    value = config.get(key, False)
    if type(value) is not bool:
        raise ValueError(f"motion_imitation_stream.{key} must be a boolean")
    return value


def perception_pause_settings(robot_config):
    """``(pause service, scope)`` the Gateway uses, or ``("", "")`` when the deployment opts out.

    ``pause_during_robot_motion`` widens ``pause_during_manipulation``: every skill whose
    capability declares ``moves_robot: true`` pauses the stream, except the mirror skills,
    which need it. Pick, place and return-to-task-pose stay paused either way.
    """
    config = robot_config.get("motion_imitation_stream", {})
    manipulation = _flag(config, "pause_during_manipulation")
    robot_motion = _flag(config, "pause_during_robot_motion")
    if not (manipulation or robot_motion):
        return "", ""
    if config.get("enabled") is not True:
        key = "pause_during_robot_motion" if robot_motion else "pause_during_manipulation"
        raise ValueError(f"{key} requires an enabled motion_imitation_stream")
    return PAUSE_SERVICE, PAUSE_SCOPE_ROBOT_MOTION if robot_motion else PAUSE_SCOPE_MANIPULATION


def manipulation_pause_service(robot_config):
    """Opt in per deployment; robots without a resident stream keep the old path."""
    return perception_pause_settings(robot_config)[0]
