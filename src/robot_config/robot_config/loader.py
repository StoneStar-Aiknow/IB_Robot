"""Configuration loader and validator for robot_config."""

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from robot_config.config import (
    CameraConfig,
    ContractAction,
    ContractExtensionConfig,
    ContractObservation,
    EmbodiedConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
    VoiceASRConfig,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy

from .utils import resolve_calibration_paths_from_config, resolve_ros_path

logger = logging.getLogger(__name__)


def _load_robot_section(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the raw ``robot`` section from a robot_config YAML file."""
    resolved_config_path = Path(config_path).expanduser().resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Robot configuration not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid robot config: expected mapping in {resolved_config_path}")

    if "robot" not in data:
        raise ValueError(f"Invalid robot config: missing 'robot' section in {resolved_config_path}")

    robot_data = data["robot"]
    if not isinstance(robot_data, dict):
        raise ValueError(f"Invalid robot config: 'robot' section must be a mapping in {resolved_config_path}")

    name = robot_data.get("name")
    if not name:
        raise ValueError(f"Invalid robot config: missing 'name' in {resolved_config_path}")

    return resolved_config_path, robot_data


def load_robot_config_dict(config_path: str | Path) -> dict[str, Any]:
    """Load robot configuration as a complete dict.

    This is the canonical loader for launch/builders/runtime consumers. It preserves
    the full YAML schema under ``robot`` and annotates the resolved source path for
    downstream users that need provenance.
    """
    resolved_config_path, robot_data = _load_robot_section(config_path)
    robot_config = copy.deepcopy(robot_data)
    robot_config["_config_path"] = str(resolved_config_path)
    return robot_config


def load_camera_config(data: dict[str, Any]) -> CameraConfig:
    """Load camera configuration from dict.

    Example:
    ```yaml
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame
      camera_info_url: file:///path/to/calibration.yaml
    ```
    """
    driver = data.get("driver", "opencv")

    # Handle different index/port naming conventions
    index_or_port = data.get("index", data.get("port", data.get("serial_number", 0)))
    if driver == "realsense" and "serial_number" in data:
        index_or_port = data["serial_number"]

    return CameraConfig(
        name=data["name"],
        driver=driver,
        index_or_port=index_or_port,
        width=data.get("width", 640),
        height=data.get("height", 480),
        fps=data.get("fps", 30),
        frame_id=data.get("frame_id", f"camera_{data['name']}_frame"),
        optical_frame_id=data.get("optical_frame_id", f"camera_{data['name']}_optical_frame"),
        camera_info_url=data.get("camera_info_url"),
        pixel_format=data.get("pixel_format", "bgr8"),
        depth_width=data.get("depth_width"),
        depth_height=data.get("depth_height"),
        depth_fps=data.get("depth_fps"),
    )


def load_ros2_control_config(data: dict[str, Any], config_dir: Path | None = None) -> Ros2ControlConfig:
    """Load ros2_control configuration from dict.

    Example:
    ```yaml
    ros2_control:
      hardware_plugin: so101_hardware/SO101SystemHardware
      port: /dev/ttyACM0
      calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
      reset_positions: {1: 0.0, 2: 0.0}
      urdf_path: $(find robot_description)/urdf/lerobot/so101/so101.urdf.xacro
    ```
    """
    params = {}
    for key, value in data.items():
        if key not in ["hardware_plugin", "urdf_path"]:
            # Resolve paths in parameters
            if isinstance(value, str):
                value = resolve_ros_path(value)
            params[key] = value

    return Ros2ControlConfig(
        hardware_plugin=data.get("hardware_plugin", ""),
        params=params,
        urdf_path=resolve_ros_path(data.get("urdf_path")),
    )


def load_contract_config(data: dict[str, Any]) -> ContractExtensionConfig:
    """Load contract extension configuration from dict.

    Example:
    ```yaml
    contract:
      base_contract: $(find robot_config)/config/contracts/act_grab_pan.yaml
      observations:
        - key: observation.images.top
          topic: /camera/top
          peripheral: top
      actions:
        - key: action
          publish:
            topic: /joint_commands
            type: sensor_msgs/msg/JointState
    ```
    """
    observations = []
    for obs_data in data.get("observations", []):
        observations.append(
            ContractObservation(
                key=obs_data["key"],
                topic=obs_data.get("topic"),
                type=obs_data.get("type"),
                peripheral=obs_data.get("peripheral"),
                selector=obs_data.get("selector"),
                image=obs_data.get("image"),
                align=obs_data.get("align"),
                qos=obs_data.get("qos"),
            )
        )

    actions = []
    for action_data in data.get("actions", []):
        actions.append(
            ContractAction(
                key=action_data["key"],
                publish=action_data.get("publish", {}),
                selector=action_data.get("selector"),
                from_tensor=action_data.get("from_tensor"),
                safety_behavior=action_data.get("safety_behavior", "zeros"),
            )
        )

    return ContractExtensionConfig(
        base_contract=data.get("base_contract"),
        observations=observations,
        actions=actions,
        rate_hz=data.get("rate_hz", 20.0),
        max_duration_s=data.get("max_duration_s", 30.0),
    )


def load_voice_asr_config(data: dict[str, Any]) -> VoiceASRConfig:
    """Load voice ASR configuration from dict."""
    defaults = VoiceASRConfig()
    model_path = data.get("model_path", "")
    tokens_path = data.get("tokens_path", "")

    return VoiceASRConfig(
        enabled=data.get("enabled", defaults.enabled),
        auto_download_model=data.get("auto_download_model", defaults.auto_download_model),
        active_mode=data.get("active_mode", defaults.active_mode),
        language=data.get("language", defaults.language),
        model_path=resolve_ros_path(model_path) if model_path else "",
        tokens_path=resolve_ros_path(tokens_path) if tokens_path else "",
        provider=data.get("provider", defaults.provider),
        model_type=data.get("model_type", defaults.model_type),
        max_recording_duration=data.get("max_recording_duration", defaults.max_recording_duration),
        vad_sensitivity=data.get("vad_sensitivity", defaults.vad_sensitivity),
        realtime_pre_roll_seconds=data.get("realtime_pre_roll_seconds", defaults.realtime_pre_roll_seconds),
        publish_partial=data.get("publish_partial", defaults.publish_partial),
        output_topic=data.get("output_topic", defaults.output_topic),
        sample_rate=data.get("sample_rate", defaults.sample_rate),
        chunk_size=data.get("chunk_size", defaults.chunk_size),
        buffer_seconds=data.get("buffer_seconds", defaults.buffer_seconds),
        device_index=data.get("device_index", defaults.device_index),
        device_name=data.get("device_name", defaults.device_name),
        exit_on_init_failure=data.get("exit_on_init_failure", defaults.exit_on_init_failure),
    )


def load_embodied_config(data: dict[str, Any]) -> EmbodiedConfig:
    """Load embodied minimal-closure configuration from dict."""
    execution = data.get("execution", {})
    safety = data.get("safety", {})
    direction_mapping = execution.get("relative_motion_direction_mapping", {})
    planner = data.get("planner", {})
    perception = data.get("perception", {})
    timeout_policy = resolve_embodied_timeout_policy(data)

    return EmbodiedConfig(
        enabled=data.get("enabled", False),
        debug_tracing=data.get("debug_tracing", True),
        task_input_topic=data.get("task_input_topic", "/voice_command"),
        task_command_topic=data.get("task_command_topic", "/embodied/task_command"),
        planned_task_topic=data.get("planned_task_topic", "/embodied/planned_task"),
        status_topic=data.get("status_topic", "/embodied/task_status"),
        skill_action_name=data.get("skill_action_name", "/embodied/execute_skill"),
        primitive_action_name=data.get("primitive_action_name", "/embodied/execute_primitive"),
        validate_skill_service=data.get("validate_skill_service", "/embodied/validate_skill"),
        validate_primitive_service=data.get("validate_primitive_service", "/embodied/validate_primitive"),
        default_target_name=data.get("default_target_name", "demo_object"),
        default_place_name=data.get("default_place_name", "tray_right"),
        skill_timeout_sec=execution.get("skill_timeout_sec", 15.0),
        primitive_timeout_sec=execution.get("primitive_timeout_sec", 5.0),
        primitive_wait_sec=execution.get("primitive_wait_sec", 1.0),
        timeouts=timeout_policy,
        relative_motion_step_m=execution.get("relative_motion_step_m", 0.03),
        relative_motion_reference_frame=execution.get("relative_motion_reference_frame", "base"),
        relative_motion_direction_mapping=direction_mapping,
        planner=planner,
        perception=perception,
        gripper_open_position=execution.get("gripper_open_position", 1.0),
        gripper_closed_position=execution.get("gripper_closed_position", 0.0),
        skill_templates=data.get("skill_templates", {}),
        named_poses=data.get("named_poses", {}),
        named_targets=data.get("named_targets", {}),
        workspace=safety.get("workspace", {}),
    )


def load_robot_config(config_path: str | Path) -> RobotConfig:
    """Load robot configuration from YAML file.

    Args:
        config_path: Path to robot configuration YAML file

    Returns:
        RobotConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    resolved_config_path, robot_data = _load_robot_section(config_path)
    config_dir = resolved_config_path.parent

    # Load required fields
    name = robot_data.get("name")

    robot_type = robot_data.get("robot_type", robot_data.get("type", name))
    type_ = robot_data.get("type", name)

    # Load ros2_control config
    ros2_control_data = robot_data.get("ros2_control", {})
    ros2_control = load_ros2_control_config(ros2_control_data, config_dir)

    # Load peripherals (cameras)
    peripherals = []
    for periph_data in robot_data.get("peripherals", []):
        if periph_data.get("type") == "camera":
            peripherals.append(load_camera_config(periph_data))
        else:
            # Generic peripheral
            peripherals.append(
                PeripheralConfig(
                    type=periph_data["type"],
                    name=periph_data["name"],
                    driver=periph_data.get("driver", "generic"),
                    params=periph_data.get("params", {}),
                    frame_id=periph_data.get("frame_id"),
                )
            )

    # Load contract config
    contract_data = robot_data.get("contract", {})
    contract = load_contract_config(contract_data)

    voice_asr = load_voice_asr_config(robot_data.get("voice_asr", {}))
    embodied = load_embodied_config(robot_data.get("embodied", {}))

    return RobotConfig(
        name=name,
        type=type_,
        robot_type=robot_type,
        ros2_control=ros2_control,
        peripherals=peripherals,
        contract=contract,
        voice_asr=voice_asr,
        embodied=embodied,
    )


def build_contract_from_robot_config_dict(robot_config: dict[str, Any]):
    """Build a runtime contract directly from the canonical dict loader output."""
    from robot_config.generators.contract import build_contract_from_robot_config_dict as _build

    return _build(robot_config)


def _robot_config_to_validation_dict(config: RobotConfig) -> dict[str, Any]:
    params = dict(config.ros2_control.params)
    return {"ros2_control": params}


def validate_config(config: RobotConfig) -> list[str]:
    """Validate robot configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate ros2_control config
    if not config.ros2_control.hardware_plugin:
        errors.append("ros2_control.hardware_plugin is required")

    try:
        calibration_paths = resolve_calibration_paths_from_config(_robot_config_to_validation_dict(config))
    except (TypeError, ValueError) as exc:
        errors.append(f"Invalid ros2_control calibration configuration: {exc}")
    else:
        for calib_file in calibration_paths:
            if calib_file and not Path(calib_file).exists():
                errors.append(f"Calibration file not found: {calib_file}")

    # Validate peripherals
    peripheral_names = set()
    for periph in config.peripherals:
        if periph.name in peripheral_names:
            if isinstance(periph, CameraConfig):
                errors.append(f"Duplicate camera name: {periph.name}")
            else:
                errors.append(f"Duplicate peripheral name: {periph.name}")
        peripheral_names.add(periph.name)

        if not isinstance(periph, CameraConfig):
            continue

        # Validate camera parameters
        if periph.width <= 0 or periph.height <= 0:
            errors.append(f"Invalid camera dimensions for {periph.name}: {periph.width}x{periph.height}")
        if periph.fps <= 0:
            errors.append(f"Invalid FPS for {periph.name}: {periph.fps}")

        # Validate calibration file if specified
        if periph.camera_info_url and periph.camera_info_url.startswith("file://"):
            calib_path = Path(periph.camera_info_url.replace("file://", ""))
            if not calib_path.exists():
                errors.append(f"Camera calibration file not found: {calib_path}")

    # Validate contract-peripheral references
    for obs in config.contract.observations:
        if obs.peripheral and obs.peripheral not in peripheral_names:
            errors.append(f"Observation '{obs.key}' references undefined peripheral: {obs.peripheral}")

    if config.voice_asr.enabled and not config.voice_asr.model_path and not config.voice_asr.auto_download_model:
        errors.append("voice_asr.model_path is required when voice_asr.enabled is true")

    if config.embodied.enabled:
        valid_directions = {"forward", "backward", "left", "right", "up", "down"}
        valid_planner_modes = {"rule", "vlm_api", "hybrid"}
        valid_skills = {
            "inspect_scene",
            "observe_target_area",
            "approach_named_target",
            "hover_named_target",
            "pick_named_target",
            "lift_named_target",
            "retreat_from_target",
            "place_named_pose",
            "release_at_named_pose",
            "open_gripper_skill",
            "close_gripper_skill",
            "recover_safe_pose",
            "move_relative_ee",
            "gripper_point_down",
            "rotate_gripper_cw",
            "rotate_gripper_ccw",
        }
        required_pose_names = {"home", "observe_table", config.embodied.default_place_name}
        missing_pose_names = sorted(p for p in required_pose_names if p not in config.embodied.named_poses)
        if missing_pose_names:
            errors.append("embodied.named_poses is missing required pose(s): " + ", ".join(missing_pose_names))

        if config.embodied.default_target_name not in config.embodied.named_targets:
            errors.append(
                f"embodied.default_target_name '{config.embodied.default_target_name}' "
                "must exist in embodied.named_targets"
            )

        for target_name, target_cfg in config.embodied.named_targets.items():
            for pose_key in ("observe_pose", "pregrasp_pose", "hover_pose", "grasp_pose", "lift_pose", "retreat_pose"):
                pose_name = target_cfg.get(pose_key)
                if pose_name and pose_name not in config.embodied.named_poses:
                    errors.append(
                        f"embodied.named_targets.{target_name}.{pose_key} references undefined pose '{pose_name}'"
                    )

        skill_templates = config.embodied.skill_templates or {}
        supported_primitives = {
            "move_to_named_pose",
            "move_relative_ee",
            "open_gripper",
            "close_gripper",
            "rotate_gripper_cw",
            "rotate_gripper_ccw",
        }
        for skill_name, template in skill_templates.items():
            if skill_name not in valid_skills:
                errors.append(f"embodied.skill_templates contains unsupported skill key: {skill_name}")
                continue
            primitive_sequence = template.get("primitive_sequence", [])
            # inspect_scene is intentionally vision-only with no primitives
            if skill_name == "inspect_scene":
                continue
            if not isinstance(primitive_sequence, list) or not primitive_sequence:
                errors.append(f"embodied.skill_templates.{skill_name}.primitive_sequence must be a non-empty list")
                continue
            for step in primitive_sequence:
                if not isinstance(step, dict):
                    errors.append(f"embodied.skill_templates.{skill_name}.primitive_sequence entries must be objects")
                    continue
                primitive_name = str(step.get("primitive_name", "")).strip()
                if primitive_name not in supported_primitives:
                    errors.append(
                        f"embodied.skill_templates.{skill_name} uses unsupported primitive '{primitive_name}'"
                    )
                    continue
                if primitive_name == "move_to_named_pose":
                    pose_name = str(step.get("pose_name", "")).strip()
                    target_pose_key = str(step.get("target_pose_key", "")).strip()
                    place_from_request = bool(step.get("place_name_from_request", False))
                    reference_count = int(bool(pose_name)) + int(bool(target_pose_key)) + int(place_from_request)
                    if reference_count != 1:
                        errors.append(
                            f"embodied.skill_templates.{skill_name} move_to_named_pose step must define exactly one "
                            "of pose_name, target_pose_key, or place_name_from_request"
                        )
                    if pose_name and pose_name not in config.embodied.named_poses:
                        errors.append(f"embodied.skill_templates.{skill_name} references undefined pose '{pose_name}'")
                    if target_pose_key:
                        for target_name, target_cfg in config.embodied.named_targets.items():
                            resolved_pose_name = str(target_cfg.get(target_pose_key, "")).strip()
                            if not resolved_pose_name:
                                errors.append(
                                    f"embodied.named_targets.{target_name}.{target_pose_key} is required by "
                                    f"skill template '{skill_name}'"
                                )
                            elif resolved_pose_name not in config.embodied.named_poses:
                                errors.append(
                                    f"embodied.skill_templates.{skill_name} references undefined pose "
                                    f"'{resolved_pose_name}' via target '{target_name}.{target_pose_key}'"
                                )
                if primitive_name == "move_relative_ee":
                    literal_direction = str(step.get("motion_direction", "")).strip()
                    if (
                        not step.get("motion_direction_from_request", False)
                        and literal_direction not in valid_directions
                    ):
                        errors.append(
                            f"embodied.skill_templates.{skill_name} move_relative_ee step must provide a valid "
                            "motion_direction or enable motion_direction_from_request"
                        )
                    literal_distance = float(step.get("motion_distance", 0.0) or 0.0)
                    if not step.get("motion_distance_from_request", False) and literal_distance <= 0.0:
                        errors.append(
                            f"embodied.skill_templates.{skill_name} move_relative_ee step must provide a positive "
                            "motion_distance or enable motion_distance_from_request"
                        )

        for axis in ("x", "y", "z"):
            axis_limits = config.embodied.workspace.get(axis)
            if axis_limits is None:
                continue
            if not isinstance(axis_limits, list) or len(axis_limits) != 2:
                errors.append(f"embodied.safety.workspace.{axis} must be a [min, max] list")
                continue
            if axis_limits[0] >= axis_limits[1]:
                errors.append(f"embodied.safety.workspace.{axis} must satisfy min < max")

        if config.embodied.relative_motion_step_m <= 0.0:
            errors.append("embodied.execution.relative_motion_step_m must be greater than zero")

        if config.embodied.relative_motion_reference_frame != "base":
            errors.append("embodied.execution.relative_motion_reference_frame currently must be 'base'")

        direction_mapping = config.embodied.relative_motion_direction_mapping
        if direction_mapping:
            missing_directions = valid_directions.difference(direction_mapping.keys())
            if missing_directions:
                errors.append(
                    "embodied.execution.relative_motion_direction_mapping is missing directions: "
                    + ", ".join(sorted(missing_directions))
                )
            for direction, vector in direction_mapping.items():
                if direction not in valid_directions:
                    errors.append(
                        "embodied.execution.relative_motion_direction_mapping contains unsupported "
                        f"direction: {direction}"
                    )
                    continue
                if not isinstance(vector, list) or len(vector) != 3:
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must be a 3-element list"
                    )
                    continue
                try:
                    normalized = [float(v) for v in vector]
                except (TypeError, ValueError):
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must contain numeric values"
                    )
                    continue
                if all(abs(v) < 1e-9 for v in normalized):
                    errors.append(
                        f"embodied.execution.relative_motion_direction_mapping.{direction} must not be a zero vector"
                    )

        planner = config.embodied.planner or {}
        timeout_policy = config.embodied.timeouts or resolve_embodied_timeout_policy(
            {
                "execution": {
                    "skill_timeout_sec": config.embodied.skill_timeout_sec,
                    "primitive_timeout_sec": config.embodied.primitive_timeout_sec,
                    "primitive_wait_sec": config.embodied.primitive_wait_sec,
                },
                "planner": planner,
                "perception": config.embodied.perception or {},
                "timeouts": {},
            }
        )
        planner_mode = str(planner.get("mode", "rule")).lower()
        valid_vlm_api_providers = {"kimicode", "openai_compatible"}
        if planner_mode not in valid_planner_modes:
            errors.append("embodied.planner.mode must be one of: " + ", ".join(sorted(valid_planner_modes)))

        scene_sources = planner.get("scene_sources", {})
        if planner_mode in {"vlm_api", "hybrid"}:
            if not scene_sources.get("primary_camera_topic"):
                errors.append(
                    "embodied.planner.scene_sources.primary_camera_topic is required when planner.mode uses VLM"
                )
            if float(timeout_policy.get("scene_freshness_sec", 0.0)) <= 0.0:
                errors.append("embodied.timeouts.scene_freshness_sec must be greater than zero")
            if bool(scene_sources.get("require_depth", False)) and not (
                scene_sources.get("primary_aligned_depth_topic") or scene_sources.get("wrist_aligned_depth_topic")
            ):
                errors.append(
                    "embodied.planner.scene_sources.require_depth=true requires at least one aligned depth topic"
                )
            if bool(scene_sources.get("require_pointcloud", False)) and not (
                scene_sources.get("primary_pointcloud_topic") or scene_sources.get("wrist_pointcloud_topic")
            ):
                errors.append(
                    "embodied.planner.scene_sources.require_pointcloud=true requires at least one pointcloud topic"
                )
            vlm_api = planner.get("vlm_api", {})
            planner_provider = str(vlm_api.get("provider", "")).strip()
            if not planner_provider:
                errors.append("embodied.planner.vlm_api.provider is required when planner.mode uses VLM")
            elif planner_provider not in valid_vlm_api_providers:
                errors.append(
                    "embodied.planner.vlm_api.provider must be one of: " + ", ".join(sorted(valid_vlm_api_providers))
                )
            if not vlm_api.get("model"):
                errors.append("embodied.planner.vlm_api.model is required when planner.mode uses VLM")
            if planner_provider == "kimicode" and not str(vlm_api.get("api_key_env", "")).strip():
                errors.append("embodied.planner.vlm_api.api_key_env is required when planner.mode uses VLM")
            if not vlm_api.get("base_url"):
                errors.append("embodied.planner.vlm_api.base_url is required when planner.mode uses VLM")
            if float(timeout_policy.get("model_idle_timeout_sec", 0.0)) <= 0.0:
                errors.append("embodied.timeouts.model_idle_timeout_sec must be greater than zero")

        planning_policy = planner.get("planning_policy", {})
        allowed_skills = planning_policy.get("allowed_skills", [])
        if planner_mode in {"vlm_api", "hybrid"}:
            if not isinstance(allowed_skills, list) or not allowed_skills:
                errors.append(
                    "embodied.planner.planning_policy.allowed_skills must be a non-empty list "
                    "when planner.mode uses VLM"
                )
            else:
                unsupported = sorted(s for s in allowed_skills if s not in valid_skills)
                if unsupported:
                    errors.append(
                        "embodied.planner.planning_policy.allowed_skills contains unsupported skill(s): "
                        + ", ".join(unsupported)
                    )
            min_confidence = float(planning_policy.get("min_confidence", 0.7))
            if min_confidence < 0.0 or min_confidence > 1.0:
                errors.append("embodied.planner.planning_policy.min_confidence must be in [0.0, 1.0]")

        perception = config.embodied.perception or {}
        if perception.get("enabled", False):
            if not str(perception.get("request_topic", "")).strip():
                errors.append("embodied.perception.request_topic is required when perception is enabled")
            if not str(perception.get("result_topic", "")).strip():
                errors.append("embodied.perception.result_topic is required when perception is enabled")
            p_sources = perception.get("scene_sources", {})
            if not p_sources.get("primary_camera_topic"):
                errors.append(
                    "embodied.perception.scene_sources.primary_camera_topic is required when perception is enabled"
                )
            if float(timeout_policy.get("scene_freshness_sec", 0.0)) <= 0.0:
                errors.append("embodied.timeouts.scene_freshness_sec must be greater than zero")
            if bool(p_sources.get("require_depth", False)) and not (
                p_sources.get("primary_aligned_depth_topic") or p_sources.get("wrist_aligned_depth_topic")
            ):
                errors.append(
                    "embodied.perception.scene_sources.require_depth=true requires at least one aligned depth topic"
                )
            if bool(p_sources.get("require_pointcloud", False)) and not (
                p_sources.get("primary_pointcloud_topic") or p_sources.get("wrist_pointcloud_topic")
            ):
                errors.append(
                    "embodied.perception.scene_sources.require_pointcloud=true requires at least one pointcloud topic"
                )
            p_vlm_api = perception.get("vlm_api", {})
            p_provider = str(p_vlm_api.get("provider", "")).strip()
            if not p_provider:
                errors.append("embodied.perception.vlm_api.provider is required when perception is enabled")
            elif p_provider not in valid_vlm_api_providers:
                errors.append(
                    "embodied.perception.vlm_api.provider must be one of: " + ", ".join(sorted(valid_vlm_api_providers))
                )
            if not p_vlm_api.get("model"):
                errors.append("embodied.perception.vlm_api.model is required when perception is enabled")
            if p_provider == "kimicode" and not str(p_vlm_api.get("api_key_env", "")).strip():
                errors.append("embodied.perception.vlm_api.api_key_env is required when perception is enabled")
            if not p_vlm_api.get("base_url"):
                errors.append("embodied.perception.vlm_api.base_url is required when perception is enabled")
            if float(timeout_policy.get("model_idle_timeout_sec", 0.0)) <= 0.0:
                errors.append("embodied.timeouts.model_idle_timeout_sec must be greater than zero")

        if float(timeout_policy.get("task_budget_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.task_budget_sec must be greater than zero")
        if float(timeout_policy.get("rpc_timeout_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.rpc_timeout_sec must be greater than zero")
        if float(timeout_policy.get("gripper_settle_sec", 0.0)) <= 0.0:
            errors.append("embodied.timeouts.gripper_settle_sec must be greater than zero")

        conversation = perception.get("conversation", {})
        try:
            max_history_turns = int(conversation.get("max_history_turns", 0))
        except (TypeError, ValueError):
            errors.append("embodied.perception.conversation.max_history_turns must be an integer")
        else:
            if max_history_turns < 0:
                errors.append("embodied.perception.conversation.max_history_turns must be >= 0")

    return errors


def validate_config_file(config_path: str | Path) -> bool:
    """Validate a robot configuration file.

    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_robot_config(config_path)
        errors = validate_config(config)

        if errors:
            logger.error(f"Configuration errors in {config_path}:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info(f"Configuration {config_path} is valid")
        return True

    except Exception as e:
        logger.error(f"Failed to validate {config_path}: {e}")
        return False
