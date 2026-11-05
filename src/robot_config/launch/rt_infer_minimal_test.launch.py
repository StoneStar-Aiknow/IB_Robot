"""Minimal RT-Infer integration launch (board validation only).

Starts exactly the RT-Infer owner and the PEAR perception model service from
the real board configuration, through the real launch registry and the real
robot_config loader — no runtime provider, no controllers, no other
consumers. Exiting the launch (SIGINT) proves review #4's ordered shutdown:
the wrapped client exits first, then the owner stops the scheduler through
the vendor CLI.

This launch moves no robot: the model service only answers typed service
requests, and nothing in this graph sends commands.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction


def _load_minimal_config():
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("robot_config")
    return str(Path(share) / "config" / "robots" / "aimdk_x2_magicmirror_board.yaml")


def launch_setup(context):
    config_path = _load_minimal_config()
    from robot_config.loader import load_robot_config_dict

    config = load_robot_config_dict(config_path, defer_interface_binding=True)
    config["perception_services"] = {
        "services": [
            service
            for service in config.get("perception_services", {}).get("services", [])
            if service.get("id") == "hri_pear_parameters"
        ]
    }
    config["voice_tts"] = {"enabled": False}
    config["visual_interaction"] = {"enabled": False}
    config["motion_imitation_stream"] = {"enabled": False}

    # 1. The RT-Infer owner, exactly as the registry would build it.
    from robot_config.launch_builders.rt_infer_scheduler import build_rt_infer_scheduler
    from robot_config.launch_registry.registry import LaunchBuildContext

    context_value = LaunchBuildContext(robot_config=config, use_sim=False)
    owner_result = build_rt_infer_scheduler(context_value)

    # 2. The PEAR model service, wrapped at its role.
    from inference_service.launch_builders.perception_models import generate_perception_model_nodes

    nodes = generate_perception_model_nodes(config)

    return [*owner_result.actions, *nodes]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
