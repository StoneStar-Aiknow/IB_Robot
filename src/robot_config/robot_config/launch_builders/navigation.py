"""Navigation launch builders."""

from launch_ros.actions import Node

from robot_config.utils import parse_bool, resolve_ros_path


def generate_navigation_nodes(
    robot_config: dict,
    use_sim=False,
    navigation_mode: str = "",
    force_enable: bool = False,
):
    """Generate navigation nodes from robot_config navigation section."""
    navigation_config = robot_config.get("navigation", {})
    if not force_enable and not navigation_config.get("enabled", False):
        return []

    resolved_mode = navigation_mode or navigation_config.get("default_mode", "")
    mode_configs = navigation_config.get("modes", {})
    if not resolved_mode:
        raise ValueError("robot.navigation.default_mode must be set when navigation is enabled")
    if resolved_mode not in mode_configs:
        raise ValueError(
            f"Unknown navigation mode '{resolved_mode}'. Available: {list(mode_configs.keys())}"
        )

    mode_config = mode_configs[resolved_mode]
    config_path = mode_config.get("config")
    if not config_path:
        raise ValueError(f"Navigation mode '{resolved_mode}' is missing a config path")

    node_parameters = [resolve_ros_path(config_path)]
    if parse_bool(use_sim, default=False):
        node_parameters.append({"use_sim_time": True})

    return [
        Node(
            package=navigation_config.get("package", "robot_localization"),
            executable=navigation_config.get("executable", "ekf_node"),
            name=navigation_config.get("node_name", "ekf_node"),
            output="screen",
            parameters=node_parameters,
        )
    ]
