"""Launch builder modules for robot_config.

This package contains modules for building ROS2 launch components:
- description.py: URDF building (xacro processing + camera injection)
- control.py: ros2_control nodes, controller spawners
- perception.py: Camera drivers, TF publishers
- simulation.py: Gazebo and simulation nodes
- execution.py: Action dispatcher and inference nodes

Compatibility lazy facade (openspec ``refactor-launch-builder-registry``,
design decision 6): every legacy public symbol resolves through a lookup
table on first access. Importing this package imports NO builder
submodule, so ``import robot_config.launch_builders.registry`` (or
``launch_options``) does not pull benchmark/tensormsg or any other
business dependency. Each access re-resolves the attribute on the source
module (no cached callable), preserving source-module monkeypatch seams;
``from robot_config.launch_builders import <submodule>`` keeps working
through the standard from-import fallback.
"""

from __future__ import annotations

import importlib

# Legacy public symbol -> (submodule, attribute). The table is the single
# compatibility surface: new registry entries do NOT need to be added here.
_EXPORTS = {
    "generate_audio_io_actions": ("audio_io", "generate_audio_io_actions"),
    "generate_benchmark_nodes": ("benchmark", "generate_benchmark_nodes"),
    "generate_controller_spawners": ("control", "generate_controller_spawners"),
    "generate_ros2_control_nodes": ("control", "generate_ros2_control_nodes"),
    "validate_joint_config": ("control", "validate_joint_config"),
    "validate_runtime_resources": ("control", "validate_runtime_resources"),
    "generate_robot_description": ("description", "generate_robot_description"),
    "generate_hand_source_nodes": ("hand_sources", "generate_hand_source_nodes"),
    "generate_hardware_mock_nodes": ("hardware_mock", "generate_hardware_mock_nodes"),
    "mock_mode_skips_subsystem": ("hardware_mock", "mock_mode_skips_subsystem"),
    "generate_navigation_nodes": ("navigation", "generate_navigation_nodes"),
    "generate_camera_nodes": ("perception", "generate_camera_nodes"),
    "generate_lidar_nodes": ("perception", "generate_lidar_nodes"),
    "generate_tf_nodes": ("perception", "generate_tf_nodes"),
    "generate_virtual_camera_relays": ("perception", "generate_virtual_camera_relays"),
    "generate_perception_model_nodes": ("perception_models", "generate_perception_model_nodes"),
    "generate_runtime_provider_actions": ("runtime", "generate_runtime_provider_actions"),
    "runtime_provider": ("runtime", "runtime_provider"),
    "generate_gazebo_nodes": ("simulation", "generate_gazebo_nodes"),
    "generate_voice_asr_nodes": ("voice_asr", "generate_voice_asr_nodes"),
    "generate_voice_tts_nodes": ("voice_tts", "generate_voice_tts_nodes"),
    "generate_speech_direction_actions": ("speech_direction", "generate_speech_direction_actions"),
}


def __getattr__(name: str):
    entry = _EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    module = importlib.import_module(f".{module_name}", __name__)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"module {module.__name__!r} has no attribute {attribute!r}") from exc


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # Description
    "generate_robot_description",
    # Control
    "generate_ros2_control_nodes",
    "generate_controller_spawners",
    "generate_hand_source_nodes",
    "validate_runtime_resources",
    "validate_joint_config",
    # Perception
    "generate_camera_nodes",
    "generate_lidar_nodes",
    "generate_tf_nodes",
    "generate_virtual_camera_relays",
    "generate_perception_model_nodes",
    # Simulation
    "generate_gazebo_nodes",
    # MoveIt
    "generate_runtime_provider_actions",
    "runtime_provider",
    # Voice ASR
    "generate_voice_asr_nodes",
    "generate_audio_io_actions",
    "generate_voice_tts_nodes",
    "generate_speech_direction_actions",
    # Navigation
    "generate_navigation_nodes",
    # Hardware mock
    "generate_hardware_mock_nodes",
    "mock_mode_skips_subsystem",
    # Benchmark
    "generate_benchmark_nodes",
]
