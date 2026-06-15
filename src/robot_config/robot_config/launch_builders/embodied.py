"""Embodied minimal-closure launch builder."""

import json
from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.timeout_policy import resolve_embodied_timeout_policy

logger = get_colored_logger("robot_config.embodied")


def generate_embodied_nodes(robot_config: dict[str, Any], active_control_mode: str) -> list[Node]:
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
    named_poses = embodied_config.get("named_poses", {})
    named_targets = embodied_config.get("named_targets", {})
    skill_templates = embodied_config.get("skill_templates", {})
    safety = embodied_config.get("safety", {})
    joint_config = robot_config.get("joints", {})
    teleoperation = robot_config.get("teleoperation", {})
    planner = embodied_config.get("planner", {})
    perception = embodied_config.get("perception", {})
    planner_mode = str(planner.get("mode", "rule")).lower()
    scene_sources = planner.get("scene_sources", {})
    vlm_api = planner.get("vlm_api", {})
    planning_policy = planner.get("planning_policy", {})
    perception_scene_sources = perception.get("scene_sources", {})
    perception_vlm_api = perception.get("vlm_api", {})
    perception_conversation = perception.get("conversation", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied_config)
    allowed_skills = planning_policy.get(
        "allowed_skills",
        [
            "inspect_scene",
            "recover_safe_pose",
            "recover_zero_pose",
            "move_relative_ee",
            "open_gripper_skill",
            "close_gripper_skill",
            "rotate_gripper_cw",
            "rotate_gripper_ccw",
            "dance_basic",
        ],
    )

    common_params = {
        "debug_tracing": embodied_config.get("debug_tracing", True),
        "named_poses_json": json.dumps(named_poses),
        "named_targets_json": json.dumps(named_targets),
        "skill_templates_json": json.dumps(skill_templates),
        "workspace_json": json.dumps(safety.get("workspace", {})),
        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
        "joint_limits_json": json.dumps(teleoperation.get("safety", {}).get("joint_limits", {})),
        "default_target_name": embodied_config.get("default_target_name", "demo_object"),
        "default_place_name": embodied_config.get("default_place_name", "tray_right"),
        "skill_action_name": embodied_config.get("skill_action_name", "/embodied/execute_skill"),
        "primitive_action_name": embodied_config.get("primitive_action_name", "/embodied/execute_primitive"),
        "validate_skill_service": embodied_config.get("validate_skill_service", "/embodied/validate_skill"),
        "validate_primitive_service": embodied_config.get("validate_primitive_service", "/embodied/validate_primitive"),
        "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
        "gripper_settle_sec": timeout_policy["gripper_settle_sec"],
        "relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
        "relative_motion_reference_frame": execution.get("relative_motion_reference_frame", "base"),
        "relative_motion_direction_mapping_json": json.dumps(execution.get("relative_motion_direction_mapping", {})),
        "gripper_open_position": execution.get("gripper_open_position", 1.0),
        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
        "task_executor_action_name": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
    }

    planner_node = Node(
        package="embodied_agent",
        executable="task_planner_node",
        name="task_planner_node",
        output="screen",
        parameters=[
            {
                "debug_tracing": embodied_config.get("debug_tracing", True),
                "input_topic": embodied_config.get("task_command_topic", "/embodied/task_command"),
                "output_topic": embodied_config.get("planned_task_topic", "/embodied/planned_task"),
                "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
                "default_target_name": embodied_config.get("default_target_name", "demo_object"),
                "default_place_name": embodied_config.get("default_place_name", "tray_right"),
                "default_relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
            }
        ],
    )
    if planner_mode in {"vlm_api", "hybrid"}:
        planner_node = Node(
            package="vlm_task_planner",
            executable="vlm_task_planner_node",
            name="vlm_task_planner_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "input_topic": embodied_config.get("task_command_topic", "/embodied/task_command"),
                    "output_topic": embodied_config.get("planned_task_topic", "/embodied/planned_task"),
                    "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
                    "default_target_name": embodied_config.get("default_target_name", "demo_object"),
                    "default_place_name": embodied_config.get("default_place_name", "tray_right"),
                    "default_relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
                    "named_poses_json": json.dumps(named_poses),
                    "named_targets_json": json.dumps(named_targets),
                    "workspace_json": json.dumps(safety.get("workspace", {})),
                    "relative_motion_reference_frame": execution.get("relative_motion_reference_frame", "base"),
                    "relative_motion_direction_mapping_json": json.dumps(
                        execution.get("relative_motion_direction_mapping", {})
                    ),
                    "planner_mode": planner_mode,
                    "primary_camera_topic": scene_sources.get("primary_camera_topic", "/camera/top/image_raw"),
                    "wrist_camera_topic": scene_sources.get("wrist_camera_topic", "/camera/wrist/image_raw"),
                    "primary_camera_info_topic": scene_sources.get("primary_camera_info_topic", ""),
                    "primary_aligned_depth_topic": scene_sources.get("primary_aligned_depth_topic", ""),
                    "primary_pointcloud_topic": scene_sources.get("primary_pointcloud_topic", ""),
                    "wrist_camera_info_topic": scene_sources.get("wrist_camera_info_topic", ""),
                    "wrist_aligned_depth_topic": scene_sources.get("wrist_aligned_depth_topic", ""),
                    "wrist_pointcloud_topic": scene_sources.get("wrist_pointcloud_topic", ""),
                    "ee_pose_topic": scene_sources.get("ee_pose_topic", "/robot_status/ee_pose"),
                    "joint_state_topic": scene_sources.get("joint_state_topic", "/joint_states"),
                    "max_scene_age_sec": timeout_policy["scene_freshness_sec"],
                    "require_depth": scene_sources.get("require_depth", False),
                    "require_pointcloud": scene_sources.get("require_pointcloud", False),
                    "api_provider": vlm_api.get("provider", "openai_compatible"),
                    "api_base_url": vlm_api.get("base_url", "http://localhost:8000/v1"),
                    "api_key_env": vlm_api.get("api_key_env", ""),
                    "api_model": vlm_api.get("model", "Qwen3.5-9B"),
                    "api_timeout_sec": timeout_policy["model_idle_timeout_sec"],
                    "api_max_image_width": vlm_api.get("max_image_width", 640),
                    "api_jpeg_quality": vlm_api.get("jpeg_quality", 80),
                    "fallback_to_rule_planner": planning_policy.get("fallback_to_rule_planner", True),
                    "min_confidence": planning_policy.get("min_confidence", 0.7),
                    "allowed_skills_json": json.dumps(allowed_skills),
                }
            ],
        )

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
            executable="task_entry_node",
            name="task_entry_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "input_topic": embodied_config.get("task_input_topic", "/voice_command"),
                    "output_topic": embodied_config.get("task_command_topic", "/embodied/task_command"),
                    "planned_output_topic": embodied_config.get("planned_task_topic", "/embodied/planned_task"),
                    "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
                    "default_target_name": embodied_config.get("default_target_name", "demo_object"),
                    "default_place_name": embodied_config.get("default_place_name", "tray_right"),
                    "default_relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
                    "default_task_timeout_sec": timeout_policy["task_budget_sec"],
                }
            ],
        ),
        planner_node,
        Node(
            package="embodied_agent",
            executable="task_executor_node",
            name="task_executor_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "input_topic": embodied_config.get("planned_task_topic", "/embodied/planned_task"),
                    "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
                    "skill_action_name": embodied_config.get("skill_action_name", "/embodied/execute_skill"),
                    "default_task_timeout_sec": timeout_policy["task_budget_sec"],
                    "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                }
            ],
        ),
    ]
    if perception_node is not None:
        nodes.append(perception_node)
    return nodes
