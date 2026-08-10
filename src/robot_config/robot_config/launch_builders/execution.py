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
        # Scheduled-path endpoints are passed to the pipeline node only
        # when the scheduler is enabled; the node registers the scheduled
        # action servers + serving status iff scheduler_enabled=true.
        scheduler = inference.scheduler if inference.scheduler is not None else None
        if scheduler is not None and scheduler.enable:
            parameters["scheduled_open_session"] = transport.open_session
            parameters["scheduled_dispatch"] = transport.dispatch
            parameters["scheduled_close_session"] = transport.close_session
            parameters["scheduled_serving_status"] = transport.serving_status
            parameters["runtime_policy_json"] = pipeline.runtime_policy_json or ""
            parameters["runtime_policy_fingerprint"] = pipeline.runtime_policy_fingerprint or ""
            parameters["hardware_resource_id"] = pipeline.hardware_resource_id or ""
            parameters["session_idle_timeout_ns"] = scheduler.session_idle_timeout_ns
            parameters["max_prompt_bytes"] = scheduler.max_prompt_bytes
            parameters["max_error_message_bytes"] = scheduler.max_error_message_bytes
            parameters["max_error_details_bytes"] = scheduler.max_error_details_bytes
            parameters["max_session_records"] = scheduler.max_session_records
            parameters["terminal_result_cache_entries"] = scheduler.terminal_result_cache_entries
            parameters["max_duplicate_waiters_per_request"] = scheduler.max_duplicate_waiters_per_request
            parameters["terminal_session_retention_ns"] = scheduler.terminal_session_retention_ns
            parameters["public_capacity_json"] = json.dumps(
                {wc.work_class: {"max_in_flight": wc.max_in_flight} for wc in pipeline.public_capacity.values()},
                sort_keys=True,
            )
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


def _scheduled_process_exit_shutdown_factory(node: object):
    """Build an OnProcessExit on_exit handler that tears down the whole scheduled
    topology when one required process exits."""
    from launch.events import Shutdown

    def _on_exit(_event, _context) -> object:
        logger.error(f"scheduled process {getattr(node, 'node_executable', '?')} exited; shutting down")
        return Shutdown(reason="scheduled process exited")

    return _on_exit


def generate_execution_nodes(
    robot_config: dict,
    control_mode: str = "model_inference",
    use_sim: object = False,
    use_sim_time: object | None = None,
) -> list:
    """Generate the execution subgraph for one control mode.

    Returns a list of launch actions (Node or RegisterEventHandler wrappers).
    On the scheduled branch each required process (required pipelines, Scheduler,
    ScheduledActionDispatcher) is wrapped in OnProcessExit -> Shutdown so that a
    required process exit or readiness timeout tears down the whole scheduled
    topology. The false/absent branch returns bare Nodes as before.
    """
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnProcessExit

    if not control_mode or control_mode == "default":
        control_mode = robot_config.get("default_control_mode", "model_inference")

    inference = _validated_inference(robot_config, control_mode)
    scheduler = inference.scheduler if inference.scheduler is not None else None

    # scheduler.enable=true selects the scheduled topology.
    if scheduler is not None and scheduler.enable:
        inference_nodes = generate_inference_node(robot_config, control_mode, use_sim, use_sim_time)
        scheduler_node = generate_global_inference_scheduler_node(
            robot_config, control_mode, scheduler, _resolve_use_sim_time(use_sim, use_sim_time)
        )
        scheduled_dispatcher = generate_scheduled_action_dispatcher_node(
            robot_config, control_mode, scheduler, _resolve_use_sim_time(use_sim, use_sim_time)
        )
        required_pipeline_nodes = [
            node
            for node, pipeline in zip(inference_nodes, inference.pipelines.values(), strict=True)
            if pipeline.required
        ]
        # Optional pipelines may disappear without killing required serving.
        # Global readiness/routing already excludes their stale or missing status.
        required_nodes = [*required_pipeline_nodes, scheduler_node, scheduled_dispatcher]
        exit_handlers = [
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=node,
                    on_exit=_scheduled_process_exit_shutdown_factory(node),
                )
            )
            for node in required_nodes
        ]
        return [*inference_nodes, scheduler_node, scheduled_dispatcher, *exit_handlers]

    # False or absent selects the unchanged legacy behavior.
    inference_nodes = generate_inference_node(robot_config, control_mode, use_sim, use_sim_time)
    dispatcher = generate_action_dispatcher_node(
        robot_config,
        control_mode,
        _resolve_use_sim_time(use_sim, use_sim_time),
    )
    return [*inference_nodes, dispatcher]


def generate_global_inference_scheduler_node(
    robot_config: dict,
    control_mode: str,
    scheduler: object,
    use_sim_time: bool,
) -> Node:
    """Independent GlobalInferenceScheduler process and single writable Scheduler."""
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    inference = _validated_inference(robot_config, control_mode)
    # Read-only per-pipeline status and transport parameters keep Scheduler and
    # pipeline endpoint names aligned.
    pipeline_endpoints: list[dict] = []
    for pipeline in inference.pipelines.values():
        pipeline_endpoints.append(
            {
                "pipeline_id": pipeline.pipeline_id,
                "required": pipeline.required,
                "serving_status": pipeline.transport.serving_status,
                "open_session": pipeline.transport.open_session,
                "dispatch": pipeline.transport.dispatch,
                "close_session": pipeline.transport.close_session,
                "compatibility_group": pipeline.compatibility_group or "",
                "hardware_resource_id": pipeline.hardware_resource_id or "",
                "hardware_profile_fingerprint": pipeline.hardware_profile_fingerprint or "",
                "deployment_fingerprint": pipeline.validated_manifest.fingerprint,
                "runtime_policy_fingerprint": pipeline.runtime_policy_fingerprint or "",
                "profile_compatibility_fingerprint": pipeline.profile_compatibility_fingerprint or "",
                "profile_path": str(pipeline.profile_path) if pipeline.profile_path else "",
                "public_capacity": {
                    capacity.work_class: {
                        "max_in_flight": capacity.max_in_flight,
                    }
                    for capacity in pipeline.public_capacity.values()
                },
            }
        )
    return Node(
        package="inference_service",
        executable="global_inference_scheduler_node",
        name="global_inference_scheduler",
        parameters=[
            {
                "readiness_endpoint": scheduler.global_endpoints.readiness,
                "open_session_endpoint": scheduler.global_endpoints.open_session,
                "dispatch_endpoint": scheduler.global_endpoints.dispatch,
                "close_session_endpoint": scheduler.global_endpoints.close_session,
                "default_target_pipeline_id": inference.inference_pipeline or "",
                "pipelines_json": json.dumps(pipeline_endpoints, sort_keys=True),
                "default_open_timeout_ns": scheduler.default_open_timeout_ns,
                "default_request_timeout_ns": scheduler.default_request_timeout_ns,
                "status_stale_timeout_ns": scheduler.status_stale_timeout_ns,
                "clock_skew_tolerance_ns": scheduler.clock_skew_tolerance_ns,
                "goal_acceptance_timeout_ns": scheduler.goal_acceptance_timeout_ns,
                "goal_acceptance_safety_margin_ms": scheduler.goal_acceptance_safety_margin_ms,
                "dispatch_safety_margin_ms": scheduler.dispatch_safety_margin_ms,
                "dispatch_goal_contexts": scheduler.dispatch_goal_contexts,
                "lower_priority_dispatch_goal_contexts": scheduler.lower_priority_dispatch_goal_contexts,
                "session_idle_timeout_ns": scheduler.session_idle_timeout_ns,
                "profile_min_samples": scheduler.profile_min_samples,
                "profile_max_age_days": scheduler.profile_max_age_days,
                "max_product_requests_per_session": scheduler.max_product_requests_per_session,
                "terminal_result_cache_entries": scheduler.terminal_result_cache_entries,
                "max_duplicate_waiters_per_request": scheduler.max_duplicate_waiters_per_request,
                "max_prompt_bytes": scheduler.max_prompt_bytes,
                "max_fallback_pipelines": scheduler.max_fallback_pipelines,
                "max_error_message_bytes": scheduler.max_error_message_bytes,
                "max_error_details_bytes": scheduler.max_error_details_bytes,
                "terminal_session_retention_ns": scheduler.terminal_session_retention_ns,
                "max_session_records": scheduler.max_session_records,
                "default_priority": inference.inference_priority,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )


def generate_scheduled_action_dispatcher_node(
    robot_config: dict,
    control_mode: str,
    scheduler: object,
    use_sim_time: bool,
) -> Node:
    """ScheduledActionDispatcher executable. Shares node name
    `/action_dispatcher` with the legacy dispatcher; the two never coexist."""
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    inference = _validated_inference(robot_config, control_mode)
    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    return Node(
        package="action_dispatch",
        executable="scheduled_action_dispatcher_node",
        name="action_dispatcher",
        parameters=[
            {
                "queue_size": executor_config.get("queue_size", 100),
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "control_frequency": executor_config.get("control_frequency", 100.0),
                "temporal_smoothing_enabled": executor_config.get("temporal_smoothing_enabled", False),
                "temporal_ensemble_coeff": executor_config.get("temporal_ensemble_coeff", 0.01),
                "chunk_size": executor_config.get("chunk_size", 100),
                "smoothing_device": executor_config.get("smoothing_device", ""),
                "joint_state_topic": "/joint_states",
                "robot_config_path": str(robot_config_path),
                # Product callers only touch the Global Scheduler endpoints.
                "scheduler_readiness_endpoint": scheduler.global_endpoints.readiness,
                "open_session_endpoint": scheduler.global_endpoints.open_session,
                "dispatch_endpoint": scheduler.global_endpoints.dispatch,
                "close_session_endpoint": scheduler.global_endpoints.close_session,
                "startup_readiness_timeout_ns": scheduler.startup_readiness_timeout_ns,
                "default_open_timeout_ns": scheduler.default_open_timeout_ns,
                "default_request_timeout_ns": scheduler.default_request_timeout_ns,
                # executor.inference_* fields are consumed only by this dispatcher.
                "inference_pipeline": inference.inference_pipeline or "",
                "inference_fallback_chain": json.dumps(list(inference.inference_fallback_chain)),
                "inference_priority": inference.inference_priority,
                "inference_retry_json": json.dumps(dict(inference.inference_retry), sort_keys=True),
                "inference_prompt": executor_config.get("inference_prompt", ""),
                # navigation_mode=false opens on READY; true waits in STOPPED.
                "navigation_mode": executor_config.get("navigation_mode", False),
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )
