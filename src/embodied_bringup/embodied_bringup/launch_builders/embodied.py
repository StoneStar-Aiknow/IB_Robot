"""Embodied minimal-closure launch builder."""

import json
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.loader import robot_config_digest
from robot_config.logger_utils import get_colored_logger
from robot_config.timeout_policy import resolve_embodied_timeout_policy

logger = get_colored_logger("embodied_bringup")


def _resolve_development_source_root(config_path: Path, configured_root: str) -> Path | None:
    module_path = Path(__file__).resolve()
    anchors = [config_path]
    repository_root = next((parent.parent for parent in module_path.parents if parent.name == "src"), None)
    if repository_root is not None:
        install_root = repository_root / "install"
        try:
            config_path.absolute().relative_to(install_root)
        except ValueError:
            pass
        else:
            anchors.append(module_path)
    for anchor in anchors:
        try:
            repository_root = next(parent for parent in anchor.parents if parent.name == "src").parent
        except StopIteration:
            continue
        candidate = (repository_root / configured_root).resolve()
        if candidate.is_dir():
            return candidate
    return None


def _node_runtime_settings(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    node_params = dict(params)
    host_runtime = node_params.pop("host_runtime", {})
    environment = {}
    if "omp_threads" in host_runtime:
        environment.update(
            {
                "OMP_NUM_THREADS": str(host_runtime["omp_threads"]),
                "OMP_DYNAMIC": "FALSE",
                "OMP_WAIT_POLICY": "PASSIVE",
                "GOMP_SPINCOUNT": "0",
            }
        )
    if "blas_threads" in host_runtime:
        environment["OPENBLAS_NUM_THREADS"] = str(host_runtime["blas_threads"])
    return node_params, environment


def generate_embodied_nodes(
    robot_config: dict[str, Any],
    active_control_mode: str,
    *,
    motion_authorized: bool = False,
) -> list[Node]:
    """Generate embodied minimum-closure nodes from robot_config YAML."""
    embodied_config = robot_config.get("embodied", {})
    if not embodied_config.get("enabled", False):
        logger.info("Embodied minimal closure disabled, skipping")
        return []

    if "moveit" not in active_control_mode.lower():
        raise ValueError(
            "embodied minimal closure currently requires control_mode:=moveit_planning "
            "or with_moveit-compatible moveit control mode."
        )

    execution = embodied_config.get("execution", {})
    entry_mode = str(embodied_config.get("entry_mode", "hermes")).lower()
    if entry_mode != "hermes":
        raise ValueError("embodied.entry_mode must be hermes")
    named_poses = embodied_config.get("named_poses", {})
    named_targets = embodied_config.get("named_targets", {})
    safety = embodied_config.get("safety", {})
    joint_config = robot_config.get("joints", {})
    teleoperation = robot_config.get("teleoperation", {})
    perception = embodied_config.get("perception", {})
    grasp_execution = robot_config.get("grasp_execution", {})
    perception_scene_sources = perception.get("scene_sources", {})
    perception_vlm_api = perception.get("vlm_api", {})
    perception_conversation = perception.get("conversation", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied_config)
    skill_catalog_source_mode = embodied_config.get("skill_catalog_source_mode", "installed")
    skill_catalog_source_root = embodied_config.get("skill_catalog_source_root", "")
    if skill_catalog_source_root and not Path(skill_catalog_source_root).is_absolute():
        config_path = Path(robot_config.get("_config_path", ""))
        resolved_source_root = _resolve_development_source_root(config_path, skill_catalog_source_root)
        if resolved_source_root is None:
            skill_catalog_source_mode = "installed"
            skill_catalog_source_root = ""
        else:
            skill_catalog_source_root = str(resolved_source_root)

    common_params = {
        "debug_tracing": embodied_config.get("debug_tracing", True),
        "named_poses_json": json.dumps(named_poses),
        "named_targets_json": json.dumps(named_targets),
        "workspace_json": json.dumps(safety.get("workspace", {})),
        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
        "joint_limits_json": json.dumps(teleoperation.get("safety", {}).get("joint_limits", {})),
        "default_target_name": embodied_config.get("default_target_name", "demo_object"),
        "default_place_name": embodied_config.get("default_place_name", "home"),
        "skill_action_name": embodied_config.get("skill_action_name", "/embodied/execute_skill"),
        "primitive_action_name": embodied_config.get("primitive_action_name", "/embodied/execute_primitive"),
        "validate_skill_service": embodied_config.get("validate_skill_service", "/embodied/validate_skill"),
        "validate_primitive_service": embodied_config.get("validate_primitive_service", "/embodied/validate_primitive"),
        "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
        "motion_authorized": motion_authorized,
        "active_control_mode": active_control_mode,
        "skill_required_control_mode": robot_config.get("skill_required_control_mode", ""),
        "skill_gateway_status_service": embodied_config.get(
            "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
        ),
        "skill_catalog_source_mode": skill_catalog_source_mode,
        "skill_catalog_source_root": skill_catalog_source_root,
        "skill_catalog_profile": embodied_config.get("skill_catalog_profile", robot_config.get("name", "")),
        "skill_catalog_snapshot_service": embodied_config.get(
            "skill_catalog_snapshot_service", "/embodied/get_skill_snapshot"
        ),
        "skill_registry_event_topic": embodied_config.get(
            "skill_registry_event_topic", "/embodied/skill_registry_events"
        ),
        "begin_workflow_service": embodied_config.get("begin_workflow_service", "/embodied/begin_workflow_execution"),
        "finalize_workflow_service": embodied_config.get(
            "finalize_workflow_service", "/embodied/finalize_workflow_execution"
        ),
        "robot_name": robot_config.get("name", "unknown"),
        "config_digest": robot_config_digest(robot_config),
        "default_skill_timeout_sec": timeout_policy["default_skill_timeout_sec"],
        "task_budget_sec": timeout_policy["task_budget_sec"],
        "robot_state_freshness_sec": timeout_policy["robot_state_freshness_sec"],
        "scene_freshness_sec": timeout_policy["scene_freshness_sec"],
        "model_idle_timeout_sec": timeout_policy["model_idle_timeout_sec"],
        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
        "gripper_settle_sec": timeout_policy["gripper_settle_sec"],
        "relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
        "relative_motion_reference_frame": execution.get("relative_motion_reference_frame", "base"),
        "relative_motion_direction_mapping_json": json.dumps(execution.get("relative_motion_direction_mapping", {})),
        "gripper_open_position": execution.get("gripper_open_position", 1.0),
        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
        "task_executor_action_name": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
        "pick_action_name": grasp_execution.get("action_name", "/manipulation/execute_pick"),
        "grasp_execution_json": json.dumps(grasp_execution),
        "move_configuration_service": execution.get(
            "move_configuration_service", "/moveit_gateway/move_to_configuration"
        ),
    }
    perception_node = None
    if perception.get("enabled", False):
        perception_node = Node(
            package="perception_service",
            executable="perception_service_node",
            name="perception_service_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "request_topic": perception.get("request_topic", "/embodied/perception_request"),
                    "text_input_topic": perception.get("text_input_topic", "/embodied/perception_text"),
                    "result_topic": perception.get("result_topic", "/embodied/perception_result"),
                    "summary_topic": perception.get("summary_topic", "/embodied/perception_summary"),
                    "observation_topic": perception.get("observation_topic", "/embodied/perception_observation"),
                    "default_session_id": perception.get("default_session_id", "default"),
                    "primary_camera_topic": perception_scene_sources.get(
                        "primary_camera_topic", "/camera/top/image_raw"
                    ),
                    "wrist_camera_topic": perception_scene_sources.get("wrist_camera_topic", "/camera/wrist/image_raw"),
                    "primary_camera_info_topic": perception_scene_sources.get("primary_camera_info_topic", ""),
                    "primary_aligned_depth_topic": perception_scene_sources.get("primary_aligned_depth_topic", ""),
                    "primary_pointcloud_topic": perception_scene_sources.get("primary_pointcloud_topic", ""),
                    "wrist_camera_info_topic": perception_scene_sources.get("wrist_camera_info_topic", ""),
                    "wrist_aligned_depth_topic": perception_scene_sources.get("wrist_aligned_depth_topic", ""),
                    "wrist_pointcloud_topic": perception_scene_sources.get("wrist_pointcloud_topic", ""),
                    "ee_pose_topic": perception_scene_sources.get("ee_pose_topic", "/robot_status/ee_pose"),
                    "joint_state_topic": perception_scene_sources.get("joint_state_topic", "/joint_states"),
                    "max_scene_age_sec": timeout_policy["scene_freshness_sec"],
                    "require_depth": perception_scene_sources.get("require_depth", False),
                    "require_pointcloud": perception_scene_sources.get("require_pointcloud", False),
                    "api_provider": perception_vlm_api.get("provider", "openai_compatible"),
                    "api_base_url": perception_vlm_api.get("base_url", "http://localhost:8000/v1"),
                    "api_key_env": perception_vlm_api.get("api_key_env", ""),
                    "api_model": perception_vlm_api.get("model", "Qwen3.5-9B"),
                    "api_timeout_sec": timeout_policy["model_idle_timeout_sec"],
                    "api_max_image_width": perception_vlm_api.get("max_image_width", 320),
                    "api_jpeg_quality": perception_vlm_api.get("jpeg_quality", 70),
                    "max_history_turns": perception_conversation.get("max_history_turns", 4),
                    "max_concurrent_requests": perception.get("max_concurrent_requests", 1),
                    "min_object_confidence": perception.get("min_object_confidence", 0.0),
                }
            ],
        )

    logger.info("Embodied minimal closure enabled, launching task/safety/skill nodes")

    nodes = [
        Node(
            package="safety_guard",
            executable="safety_guard_node",
            name="safety_guard_node",
            output="screen",
            parameters=[common_params],
        ),
        Node(
            package="skill_library",
            executable="skill_executor_node",
            name="skill_executor_node",
            output="screen",
            parameters=[common_params],
        ),
        Node(
            package="embodied_agent",
            executable="agent_plan_node",
            name="agent_plan_node",
            output="screen",
            parameters=[
                {
                    "gateway_status_service": common_params["skill_gateway_status_service"],
                    "validate_skill_service": common_params["validate_skill_service"],
                    "skill_catalog_snapshot_service": common_params["skill_catalog_snapshot_service"],
                    "skill_action_name": common_params["skill_action_name"],
                    "begin_workflow_service": common_params["begin_workflow_service"],
                    "finalize_workflow_service": common_params["finalize_workflow_service"],
                    "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                    "plan_service": embodied_config.get("plan_service", "/embodied/plan_agent_command"),
                    "validate_plan_service": embodied_config.get(
                        "validate_plan_service", "/embodied/validate_agent_plan"
                    ),
                    "confirm_plan_service": embodied_config.get("confirm_plan_service", "/embodied/confirm_agent_plan"),
                    "execute_plan_action": embodied_config.get("execute_plan_action", "/embodied/execute_agent_plan"),
                }
            ],
        ),
    ]
    if perception_node is not None:
        nodes.append(perception_node)
    if grasp_execution.get("enabled", False):
        camera = grasp_execution.get("camera", {})
        planner_params, planner_env = _node_runtime_settings(grasp_execution.get("planner_node", {}))
        verifier_params = grasp_execution.get("verifier_node", {})
        if grasp_execution.get("auto_start_dependencies", True):
            nodes.extend(
                [
                    Node(
                        package="manipulation_service",
                        executable="grasp_planner_node",
                        name="grasp_planner",
                        output="screen",
                        additional_env=planner_env,
                        parameters=[
                            {
                                "rgb_topic": camera.get("rgb_topic", "/camera/wrist/image_raw"),
                                "depth_topic": camera.get(
                                    "depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw"
                                ),
                                "camera_info_topic": camera.get(
                                    "camera_info_topic", "/camera/wrist/aligned_depth_to_color/camera_info"
                                ),
                                "detect_service": grasp_execution.get("detect_service", "/perception/grounding_detect"),
                                "segment_service": grasp_execution.get("segment_service", ""),
                                "legacy_detect_service": grasp_execution.get(
                                    "fallback_detect_service", "/grasp_planner/detect_and_segment"
                                ),
                                "model_dir": grasp_execution.get("model_bundle_path", ""),
                                **planner_params,
                            }
                        ],
                    ),
                    Node(
                        package="manipulation_service",
                        executable="grasp_verifier_node",
                        name="grasp_verifier",
                        output="screen",
                        parameters=[
                            {
                                "joint_state_topic": verifier_params.get("joint_state_topic", "/joint_states"),
                                "joint_current_topic": verifier_params.get(
                                    "joint_current_topic", "/so101_follower/joint_currents"
                                ),
                                "wrist_depth_topic": verifier_params.get(
                                    "wrist_depth_topic",
                                    camera.get("depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw"),
                                ),
                                **verifier_params,
                            }
                        ],
                    ),
                ]
            )
        nodes.append(
            Node(
                package="manipulation_execution",
                executable="pick_executor_node",
                name="pick_executor_node",
                output="screen",
                parameters=[
                    {
                        "action_name": grasp_execution.get("action_name", "/manipulation/execute_pick"),
                        "primitive_action_name": embodied_config.get(
                            "primitive_action_name", "/embodied/execute_primitive"
                        ),
                        "grasp_execution_json": json.dumps(grasp_execution),
                        "workspace_json": json.dumps(safety.get("workspace", {})),
                        "home_joint_positions_json": json.dumps(
                            robot_config.get("ros2_control", {}).get("reset_positions", {})
                        ),
                        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
                        "gripper_open_position": execution.get("gripper_open_position", 1.0),
                        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
                        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                    }
                ],
            )
        )
    return nodes
