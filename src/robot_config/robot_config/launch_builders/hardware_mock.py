"""Hardware mock launch builder for robot_config.

The top-level launch uses this when ``use_sim:=true`` and
``simulation.platform: mock`` selects the contract-level backend.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.hardware_mock")

# Subsystems that hardware_mock intentionally replaces or has no analogue
# for. Centralised here so subsystem sections in robot.launch.py do not
# each hard-code their own knowledge of "what mock implies".
_MOCK_SKIPPED_SUBSYSTEMS = frozenset(
    {
        "control",  # contract_mock owns /joint_states; no controller_manager
        "perception",  # contract_mock publishes synthetic camera / lidar topics
        "voice_asr",  # voice ASR is out of mock scope
        "navigation",  # navigation stack is out of mock scope
    }
)

# Control modes compatible with hardware_mock. Mock currently only models
# the model_inference observation/action loop; teleop devices and MoveIt
# action servers are not implemented behind the mock surface.
_MOCK_SUPPORTED_CONTROL_MODES = frozenset({"model_inference"})


def validate_mock_control_mode(active_control_mode: str) -> None:
    """Validate control mode compatibility for any hardware_mock entry point."""
    if active_control_mode not in _MOCK_SUPPORTED_CONTROL_MODES:
        supported = ", ".join(sorted(_MOCK_SUPPORTED_CONTROL_MODES))
        raise RuntimeError(
            f"hardware_mock only supports control_mode={supported}, "
            f"got '{active_control_mode}'. hardware_mock does not implement "
            "teleop devices or MoveIt action servers."
        )


def mock_mode_skips_subsystem(mock_active: bool, subsystem: str) -> bool:
    """Return ``True`` when ``subsystem`` should be skipped under mock mode.

    The caller is expected to log the skip reason itself so each subsystem
    section retains a single, locally-readable narrative.
    """
    if not mock_active:
        return False
    return subsystem in _MOCK_SKIPPED_SUBSYSTEMS


def generate_hardware_mock_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Build the hardware_mock Node list for mock-mode launches.

    Currently returns a single ``contract_mock`` node. Future mock backends
    (different topic surfaces, debug instrumentation, multi-arm fan-out)
    can be added here without growing ``robot.launch.py``.
    """
    config_path = robot_config.get("_config_path", "")
    logger.info("Generating hardware_mock contract_mock node")
    return [
        Node(
            package="hardware_mock",
            executable="contract_mock",
            name="contract_mock",
            output="screen",
            parameters=[
                {
                    "robot_config_path": config_path,
                    "use_sim_time": False,
                }
            ],
        )
    ]
