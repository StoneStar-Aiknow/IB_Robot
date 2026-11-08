"""Non-motion control boundary of the resident mirror perception stream."""

PAUSE_SERVICE = "/motion_imitation_stream/set_paused"


def manipulation_pause_service(robot_config):
    """Opt in per deployment; robots without a resident stream keep the old path."""
    config = robot_config.get("motion_imitation_stream", {})
    enabled = config.get("pause_during_manipulation", False)
    if type(enabled) is not bool:
        raise ValueError("motion_imitation_stream.pause_during_manipulation must be a boolean")
    if enabled and config.get("enabled") is not True:
        raise ValueError("pause_during_manipulation requires an enabled motion_imitation_stream")
    return PAUSE_SERVICE if enabled else ""
