"""Launch the base robot stack plus embodied runtime nodes."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource

from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
from robot_config.loader import load_robot_config_dict, validate_embodied_launch_dict
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("embodied_bringup.launch")


def _load_config(robot_config_name: str, config_path_override: str) -> dict:
    try:
        robot_config_share = get_package_share_directory("robot_config")
    except Exception:
        robot_config_share = str(Path(__file__).parents[2] / "robot_config")

    config_path = (
        Path(config_path_override)
        if config_path_override
        else Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml"
    )
    config = load_robot_config_dict(config_path)
    config["_config_path"] = str(config_path)
    return config


def _parallel_ik_worker_action(config: dict, use_sim_time: str):
    grasp_execution = config.get("grasp_execution", {})
    if not isinstance(grasp_execution, dict) or not bool(grasp_execution.get("enabled", False)):
        return None
    if not bool(grasp_execution.get("auto_start_dependencies", True)):
        return None
    ik_config = grasp_execution.get("ik", {})
    worker_count = int(ik_config.get("worker_count", 0)) if isinstance(ik_config, dict) else 0
    if worker_count <= 0 or not bool(ik_config.get("auto_start_workers", True)):
        return None
    if worker_count > 8:
        raise ValueError("grasp_execution.ik.worker_count must be between 0 and 8")
    namespace_prefix = str(ik_config.get("worker_namespace_prefix", "/ik_worker")).strip("/")
    if not namespace_prefix:
        raise ValueError("grasp_execution.ik.worker_namespace_prefix must not be empty")
    worker_launch_path = Path(get_package_share_directory("robot_moveit")) / "launch" / "so101_ik_workers.launch.py"
    logger.info(f"Launching {worker_count} parallel grasp IK/FK workers under /{namespace_prefix}_<n>")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(worker_launch_path)),
        launch_arguments={
            "worker_count": str(worker_count),
            "namespace_prefix": namespace_prefix,
            "use_sim_time": use_sim_time,
        }.items(),
    )


def launch_setup(context, *_args, **_kwargs):
    robot_config_name = context.launch_configurations.get("robot_config", "so101_single_arm")
    config_path_override = context.launch_configurations.get("config_path", "")
    control_mode_override = context.launch_configurations.get("control_mode", "")
    with_embodied_str = context.launch_configurations.get("with_embodied", "true")
    with_perception_str = context.launch_configurations.get("with_perception", "")

    config = _load_config(robot_config_name, config_path_override)
    if control_mode_override:
        config["default_control_mode"] = control_mode_override

    embodied_config = config.setdefault("embodied", {})
    embodied_config["enabled"] = parse_bool(with_embodied_str, default=True)
    if with_perception_str != "":
        perception_config = embodied_config.setdefault("perception", {})
        perception_config["enabled"] = parse_bool(with_perception_str, default=False)

    # Fail fast on inconsistent launch overrides (e.g. a visual game enabled while
    # with_perception:=false) instead of starting a node graph that routes to a
    # dead topic. Reuses the same rules as robot_config.validate_config.
    launch_errors = validate_embodied_launch_dict(config)
    if launch_errors:
        for error in launch_errors:
            logger.error(f"Invalid embodied launch configuration: {error}")
        raise RuntimeError("embodied launch configuration is inconsistent: " + "; ".join(launch_errors))

    active_control_mode = config.get("default_control_mode", "moveit_planning")
    base_launch_path = Path(get_package_share_directory("robot_config")) / "launch" / "robot.launch.py"
    base_launch_arguments = {
        "robot_config": robot_config_name,
        "config_path": config_path_override,
        "use_sim": context.launch_configurations.get("use_sim", "false"),
        "use_mock": context.launch_configurations.get("use_mock", "false"),
        "auto_start_controllers": context.launch_configurations.get("auto_start_controllers", "true"),
        "control_mode": active_control_mode,
        "with_moveit": context.launch_configurations.get("with_moveit", ""),
        "moveit_display": context.launch_configurations.get("moveit_display", "false"),
        "with_embodied": "false",
        "with_perception": "false",
    }

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(base_launch_path)),
            launch_arguments=base_launch_arguments.items(),
        )
    ]
    if embodied_config["enabled"]:
        logger.info("Launching embodied runtime nodes from embodied_bringup")
        worker_action = _parallel_ik_worker_action(config, base_launch_arguments["use_sim"])
        if worker_action is not None:
            actions.append(worker_action)
        actions.extend(generate_embodied_nodes(config, active_control_mode))
    else:
        logger.info("Embodied runtime disabled by with_embodied:=false")
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="so101_single_arm"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("use_sim", default_value="false"),
            DeclareLaunchArgument("use_mock", default_value="false"),
            DeclareLaunchArgument("auto_start_controllers", default_value="true"),
            DeclareLaunchArgument("control_mode", default_value="moveit_planning"),
            DeclareLaunchArgument("with_moveit", default_value=""),
            DeclareLaunchArgument("moveit_display", default_value="false"),
            DeclareLaunchArgument("with_embodied", default_value="true"),
            DeclareLaunchArgument("with_perception", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
