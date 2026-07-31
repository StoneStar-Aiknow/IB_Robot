"""Launch builders for validated named inference pipelines and dispatch routing."""

from __future__ import annotations

import json

from launch_ros.actions import Node

from robot_config.inference_config import (
    ControlModeInferenceConfig,
    InferenceConfigError,
    InferencePipelineConfig,
    parse_inference_config,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool, prepare_lerobot_env

logger = get_colored_logger("robot_config.execution")


def _resolve_use_sim_time(use_sim: object, use_sim_time: object | None = None) -> bool:
    if use_sim_time is None:
        use_sim_time = use_sim
    return parse_bool(use_sim_time, default=False)


def _attention_viz_request(inference_config: dict) -> tuple[bool, str, dict]:
    """Compatibility helper for the inactive attention sidecar configuration."""

    config = inference_config.get("attention_viz", {}) or {}
    return parse_bool(config.get("enabled", False), default=False), str(config.get("mode") or "file"), config


def _validated_inference(robot_config: dict, control_mode: str) -> ControlModeInferenceConfig:
    return parse_inference_config(robot_config, control_mode)


def _selected_executor_pipeline(
    robot_config: dict,
    control_mode: str,
    inference: ControlModeInferenceConfig | None = None,
) -> InferencePipelineConfig:
    validated = inference or _validated_inference(robot_config, control_mode)
    if not validated.enabled or not validated.pipelines:
        raise InferenceConfigError(f"control mode {control_mode!r} has no enabled inference pipeline")

    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    selection = executor_config.get("inference_pipeline")
    if selection is None:
        if len(validated.pipelines) != 1:
            raise InferenceConfigError(
                f"control mode {control_mode!r} configures multiple inference pipelines; "
                "executor.inference_pipeline must select one explicitly"
            )
        return next(iter(validated.pipelines.values()))
    if not isinstance(selection, str) or not selection:
        raise InferenceConfigError("executor.inference_pipeline must be a non-empty pipeline ID")
    try:
        return validated.pipelines[selection]
    except KeyError as exc:
        raise InferenceConfigError(
            f"executor.inference_pipeline selects unknown pipeline {selection!r}; "
            f"available pipelines: {list(validated.pipelines)}"
        ) from exc


def generate_inference_node(
    robot_config: dict,
    control_mode: str,
    use_sim: object = False,
    use_sim_time: object | None = None,
) -> list[Node]:
    """Create one unified local or distributed-edge process per pipeline."""

    inference = _validated_inference(robot_config, control_mode)
    if not inference.enabled:
        return []
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")

    environment = prepare_lerobot_env()
    use_sim_clock = _resolve_use_sim_time(use_sim, use_sim_time)
    is_sim = parse_bool(use_sim, default=False)
    nodes: list[Node] = []
    for pipeline in inference.pipelines.values():
        transport = pipeline.transport
        parameters = {
            "pipeline_id": pipeline.pipeline_id,
            "model_path": str(pipeline.model_path),
            "deployment": pipeline.deployment,
            "execution_mode": pipeline.execution_mode,
            "request_timeout": pipeline.request_timeout,
            "default_task": pipeline.default_task,
            "runtime_options_json": json.dumps(dict(pipeline.runtime_options), sort_keys=True, separators=(",", ":")),
            "robot_config_path": str(robot_config_path),
            "use_sim": is_sim,
            "use_sim_time": use_sim_clock,
            "node_name": transport.node_name,
            "action_server": transport.action_server,
            "reset_service": transport.reset_service,
            "health_topic": transport.health_topic,
            "action_topic": transport.action_topic,
            "request_topic": transport.request_topic or "",
            "result_topic": transport.result_topic or "",
            "heartbeat_topic": transport.heartbeat_topic or "",
            "video_descriptor_topic": transport.video_descriptor_topic or "",
            "video_status_topic": transport.video_status_topic or "",
        }
        nodes.append(
            Node(
                package="inference_service",
                executable="pipeline_policy_node",
                name=transport.node_name,
                env=environment,
                parameters=[parameters],
                output="screen",
            )
        )
        logger.info(
            f"Configured inference pipeline {pipeline.pipeline_id!r}: "
            f"{pipeline.model_path} deployment={pipeline.deployment} action={transport.action_server}"
        )
    return nodes


def generate_action_dispatcher_node(robot_config: dict, control_mode: str, use_sim: object = False) -> Node:
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")

    inference = _validated_inference(robot_config, control_mode)
    pipeline = _selected_executor_pipeline(robot_config, control_mode, inference)
    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    robot_joints = robot_config.get("joints", {})
    return Node(
        package="action_dispatch",
        executable="action_dispatcher_node",
        name="action_dispatcher",
        parameters=[
            {
                "enable_dual_mode": executor_config.get("type", "topic") == "topic",
                "executor_mode": executor_config.get("mode", control_mode),
                "robot_name": robot_config.get("name", "so101"),
                "joint_names": robot_joints.get("all", ["1", "2", "3", "4", "5", "6"]),
                "queue_size": executor_config.get("queue_size", 100),
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "min_queue_size": executor_config.get("min_queue_size", 10),
                "control_frequency": executor_config.get("control_frequency", 100.0),
                "temporal_smoothing_enabled": executor_config.get("temporal_smoothing_enabled", False),
                "temporal_ensemble_coeff": executor_config.get("temporal_ensemble_coeff", 0.01),
                "chunk_size": executor_config.get("chunk_size", 100),
                "smoothing_device": executor_config.get("smoothing_device", ""),
                "control_mode": control_mode,
                "interpolation_enabled": True,
                "interpolation_step": 0.1,
                "max_interpolation_time": 2.0,
                "on_inference_failure": "hold",
                "on_queue_exhausted": "hold",
                "max_inference_timeout": 1.0,
                "max_retry_attempts": 3,
                "retry_backoff_base": 0.5,
                "stale_obs_threshold_ms": 500,
                "exhaustion_timeout": 2.0,
                "joint_state_topic": "/joint_states",
                "dispatch_action_topic": "/action_dispatch/dispatch_action",
                "robot_config_path": str(robot_config_path),
                "inference_action_server": pipeline.transport.action_server,
                "inference_reset_service": pipeline.transport.reset_service,
                "inference_timeout_sec": pipeline.request_timeout,
                "policy_reset_timeout_sec": executor_config.get("policy_reset_timeout_sec", 2.0),
                "inference_prompt": executor_config.get("inference_prompt", ""),
                "navigation_mode": executor_config.get("navigation_mode", False),
                "use_sim_time": parse_bool(use_sim, default=False),
            }
        ],
        output="screen",
    )


def generate_robot_evaluate_node(robot_config: dict, control_mode: str, use_sim: object = False) -> Node:
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    pipeline = _selected_executor_pipeline(robot_config, control_mode)
    executor_config = robot_config.get("control_modes", {}).get(control_mode, {}).get("executor", {}) or {}
    return Node(
        package="robot_evaluate",
        executable="robot_evaluate_node",
        name="robot_evaluate",
        parameters=[
            {
                "robot_config_path": str(robot_config_path),
                "inference_action_server": pipeline.transport.action_server,
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "enable_stable_mode": executor_config.get("enable_stable_mode", False),
                "use_sim_time": parse_bool(use_sim, default=False),
            }
        ],
        output="screen",
    )


def generate_execution_nodes(
    robot_config: dict,
    control_mode: str = "model_inference",
    use_sim: object = False,
    use_sim_time: object | None = None,
) -> list[Node]:
    if not control_mode or control_mode == "default":
        control_mode = robot_config.get("default_control_mode", "model_inference")
    inference_nodes = generate_inference_node(robot_config, control_mode, use_sim, use_sim_time)
    dispatcher = generate_action_dispatcher_node(
        robot_config,
        control_mode,
        _resolve_use_sim_time(use_sim, use_sim_time),
    )
    return [*inference_nodes, dispatcher]
