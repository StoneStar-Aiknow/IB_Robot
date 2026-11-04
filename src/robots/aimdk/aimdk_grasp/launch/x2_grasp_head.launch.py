"""Launch the X2 grasp head on the Orin, bound to the runtime the robot-side host runs.

The X2's temporary two-host grasp (aimdk_grasp.split): the robot-side host (the
310P or 203) runs ``embodied_pipeline.launch.py`` with the X2 grasp
configuration of its accelerator, and this launch runs the grasp head with the
same configuration. Only that configuration starts it: another robot, runtime
or grasp backend is refused before anything starts. By default it waits for
the runtime however long it takes to come up, so the head can be started
first, or at boot.
"""

from aimdk_grasp.split import head_node, isolate_launch_cpus, split_config
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from robot_config.config_path import resolve_robot_config_path
from robot_config.dds_environment import check_launch_dds
from robot_config.launch_builders.runtime import generate_bound_runtime_actions
from robot_config.loader import load_robot_config_dict


def launch_setup(context, *_args, **_kwargs):
    check_launch_dds(context)
    configurations = context.launch_configurations
    config_path = resolve_robot_config_path(
        config_name=configurations.get("robot_config") or None, config_path=configurations.get("config_path") or None
    )
    config = load_robot_config_dict(config_path, defer_interface_binding=True)
    config["_config_path"] = str(config_path)
    if configurations.get("control_mode"):
        config["default_control_mode"] = configurations["control_mode"]
    head = split_config(config)
    isolate_launch_cpus(head)
    timeout_s = float(configurations.get("runtime_timeout", "0"))
    return generate_bound_runtime_actions(
        config,
        lambda effective: [head_node(effective, head)],
        use_sim=False,
        readiness_timeout_s=timeout_s if timeout_s > 0 else float("inf"),
        start_provider=False,
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value=""),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("control_mode", default_value=""),
            # Seconds to wait for the other host's runtime to be ready; 0 waits until it is.
            DeclareLaunchArgument("runtime_timeout", default_value="0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
