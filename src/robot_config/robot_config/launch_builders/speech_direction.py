"""Speech direction launch builder for robot_config."""

from pathlib import Path
from typing import Any

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.speech_direction")
_SPEECH_DIRECTION_MISSING_ERROR = "speech_direction.enabled=true requires the voice_asr_service package to be installed"


def _voice_asr_service_share() -> Path:
    try:
        return Path(get_package_share_directory("voice_asr_service"))
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(_SPEECH_DIRECTION_MISSING_ERROR) from exc


def generate_speech_direction_actions(robot_config: dict[str, Any]) -> list[IncludeLaunchDescription]:
    """Include the speech-direction launch when enabled by the robot SSOT."""

    config = robot_config.get("speech_direction", {})
    if not config.get("enabled", False):
        logger.info("Speech direction disabled, skipping")
        return []

    profile = str(config.get("profile", "ascend_310p")).strip()
    if not profile:
        raise ValueError("speech_direction.profile must be non-empty when enabled")

    package_share = _voice_asr_service_share()
    launch_file = package_share / "launch" / "speech_direction.launch.py"
    if not launch_file.is_file():
        raise FileNotFoundError(f"Speech direction launch file not found: {launch_file}")

    launch_arguments = {"profile": profile}
    for key in ("config_file", "profiles_file", "models_root"):
        value = str(config.get(key, "")).strip()
        if value:
            launch_arguments[key] = resolve_ros_path(value)

    microphone_name = str(config.get("microphone", "")).strip()
    peripherals = robot_config.get("peripherals", [])
    microphone_params = find_microphone_params(peripherals, microphone_name)
    launch_argument_names = {
        "channels": "microphone_channels",
        "sample_rate": "microphone_sample_rate",
        "channel_indices": "microphone_channel_indices",
    }
    for parameter_name, launch_name in launch_argument_names.items():
        if parameter_name in microphone_params:
            value = microphone_params[parameter_name]
            launch_arguments[launch_name] = str(value) if parameter_name != "channel_indices" else repr(value)
    parameter_launch_names = {
        "mount_yaw_deg": "speech_direction_mount_yaw_deg",
    }
    for parameter_name, launch_name in parameter_launch_names.items():
        if parameter_name in config.get("parameters", {}):
            launch_arguments[launch_name] = str(config["parameters"][parameter_name])

    audio_io = robot_config.get("audio_io", {})
    if not is_audio_io_enabled(audio_io):
        raise ValueError("speech_direction.enabled=true requires audio_io.enabled=true")
    launch_arguments["speech_direction_audio_topic"] = str(
        audio_io.get("capture_stamped_topic", "/audio/capture_stamped")
    )

    logger.info(f"Speech direction enabled with profile {profile!r}")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_file)),
            launch_arguments=launch_arguments.items(),
        )
    ]
