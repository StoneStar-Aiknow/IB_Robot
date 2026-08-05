"""VLM-backed task planner node with rule-planner fallback."""

from __future__ import annotations

import json
import time
from typing import Any

import rclpy

from embodied_common.base_node import BaseTaskNode
from embodied_common.command_parser import load_skill_aliases
from embodied_common.dispatch_binding import copy_binding, workflow_step
from embodied_common.json_utils import load_json_list, load_json_mapping
from embodied_common.scene_analysis import SceneAnalysis, parse_scene_analysis_response
from embodied_common.skill_templates import DEFAULT_ALLOWED_SKILLS
from embodied_common.vlm_prompt_builder import build_scene_analysis_messages
from embodied_common.workflow_contracts import compute_workflow_digest
from ibrobot_msgs.msg import TaskCommand, TaskStatus
from skill_catalog.ros_consumer import CatalogViewSynchronizer
from vlm_task_planner.api_client import VLMAPIClient
from vlm_task_planner.planner_fallback import fallback_plan_from_text
from vlm_task_planner.prompt_builder import build_chat_messages
from vlm_task_planner.response_parser import PlannerResult, parse_planner_response
from vlm_task_planner.scene_snapshot import SceneSnapshotBuffer


class VLMTaskPlannerNode(BaseTaskNode):
    """Plan a constrained skill sequence from ROS scene context and text."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("vlm_task_planner_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/embodied/task_command")
        self.declare_parameter("output_topic", "/embodied/planned_task")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("default_target_name", "demo_object")
        self.declare_parameter("default_place_name", "tray_right")
        self.declare_parameter("default_relative_motion_step_m", 0.03)
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("relative_motion_reference_frame", "base")
        self.declare_parameter("relative_motion_direction_mapping_json", "{}")
        self.declare_parameter("planner_mode", "hybrid")
        self.declare_parameter("primary_camera_topic", "/camera/top/image_raw")
        self.declare_parameter("wrist_camera_topic", "/camera/wrist/image_raw")
        self.declare_parameter("primary_camera_info_topic", "")
        self.declare_parameter("primary_aligned_depth_topic", "")
        self.declare_parameter("primary_pointcloud_topic", "")
        self.declare_parameter("wrist_camera_info_topic", "")
        self.declare_parameter("wrist_aligned_depth_topic", "")
        self.declare_parameter("wrist_pointcloud_topic", "")
        self.declare_parameter("ee_pose_topic", "/robot_status/ee_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("max_scene_age_sec", 0.5)
        self.declare_parameter("require_depth", False)
        self.declare_parameter("require_pointcloud", False)
        self.declare_parameter("api_provider", "openai_compatible")
        self.declare_parameter("api_base_url", "http://localhost:8000/v1")
        self.declare_parameter("api_key_env", "")
        self.declare_parameter("api_model", "Qwen3.5-9B")
        self.declare_parameter("api_timeout_sec", 120.0)
        self.declare_parameter("api_max_image_width", 640)
        self.declare_parameter("api_jpeg_quality", 80)
        self.declare_parameter("fallback_to_rule_planner", True)
        self.declare_parameter("min_confidence", 0.7)
        self.declare_parameter("allowed_skills_json", json.dumps(DEFAULT_ALLOWED_SKILLS))
        self.declare_parameter("skill_aliases_json", "")
        self.declare_parameter("skill_gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot")
        self.declare_parameter("skill_registry_event_topic", "/embodied/skill_registry_events")
        self.declare_parameter("snapshot_sync_period_sec", 0.5)
        self.declare_parameter("debug_tracing", False)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._default_target = self.get_parameter("default_target_name").get_parameter_value().string_value
        self._default_place = self.get_parameter("default_place_name").get_parameter_value().string_value
        self._default_relative_motion_step = (
            self.get_parameter("default_relative_motion_step_m").get_parameter_value().double_value
        )
        self._named_poses = load_json_mapping(
            self.get_parameter("named_poses_json").get_parameter_value().string_value,
            "named_poses_json",
        )
        self._named_targets = load_json_mapping(
            self.get_parameter("named_targets_json").get_parameter_value().string_value,
            "named_targets_json",
        )
        self._workspace = load_json_mapping(
            self.get_parameter("workspace_json").get_parameter_value().string_value,
            "workspace_json",
        )
        self._relative_motion_reference_frame = (
            self.get_parameter("relative_motion_reference_frame").get_parameter_value().string_value
        )
        self._relative_motion_direction_mapping = load_json_mapping(
            self.get_parameter("relative_motion_direction_mapping_json").get_parameter_value().string_value,
            "relative_motion_direction_mapping_json",
        )
        self._skill_aliases = load_skill_aliases(
            self.get_parameter("skill_aliases_json").get_parameter_value().string_value
        )
        self._planner_mode = self.get_parameter("planner_mode").get_parameter_value().string_value
        self._primary_camera_topic = self.get_parameter("primary_camera_topic").get_parameter_value().string_value
        self._wrist_camera_topic = self.get_parameter("wrist_camera_topic").get_parameter_value().string_value
        self._ee_pose_topic = self.get_parameter("ee_pose_topic").get_parameter_value().string_value
        self._joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self._max_scene_age_sec = self.get_parameter("max_scene_age_sec").get_parameter_value().double_value
        self._require_depth = self.get_parameter("require_depth").get_parameter_value().bool_value
        self._require_pointcloud = self.get_parameter("require_pointcloud").get_parameter_value().bool_value
        self._api_max_image_width = self.get_parameter("api_max_image_width").get_parameter_value().integer_value
        self._api_jpeg_quality = self.get_parameter("api_jpeg_quality").get_parameter_value().integer_value
        self._fallback_enabled = self.get_parameter("fallback_to_rule_planner").get_parameter_value().bool_value
        self._min_confidence = self.get_parameter("min_confidence").get_parameter_value().double_value
        self._allowed_skills = self._load_allowed_skills(
            self.get_parameter("allowed_skills_json").get_parameter_value().string_value
        )
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._catalog = CatalogViewSynchronizer(
            self,
            status_service=self.get_parameter("skill_gateway_status_service").get_parameter_value().string_value,
            snapshot_service=self.get_parameter("skill_catalog_snapshot_service").get_parameter_value().string_value,
            event_topic=self.get_parameter("skill_registry_event_topic").get_parameter_value().string_value,
            sync_period_sec=self.get_parameter("snapshot_sync_period_sec").get_parameter_value().double_value,
        )

        self._scene_buffer = SceneSnapshotBuffer.from_node(self)
        self._api_client = VLMAPIClient(
            provider=self.get_parameter("api_provider").get_parameter_value().string_value,
            base_url=self.get_parameter("api_base_url").get_parameter_value().string_value,
            api_key_env=self.get_parameter("api_key_env").get_parameter_value().string_value,
            model=self.get_parameter("api_model").get_parameter_value().string_value,
            timeout_sec=self.get_parameter("api_timeout_sec").get_parameter_value().double_value,
        )

        self._planned_publisher = self.create_publisher(TaskCommand, self._output_topic, 10)
        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self.create_subscription(TaskCommand, self._input_topic, self._handle_task_command, 10)

        self.get_logger().info(
            "[embodied-debug] vlm_task_planner ready: "
            f"mode={self._planner_mode}, input={self._input_topic}, output={self._output_topic}, "
            f"camera={self._primary_camera_topic}, provider={self.get_parameter('api_provider').value}"
        )

    @staticmethod
    def _load_allowed_skills(raw_value: str) -> list[str]:
        raw_value = raw_value.strip()
        if not raw_value:
            return []
        return [str(skill).strip() for skill in load_json_list(raw_value, "allowed_skills_json") if str(skill).strip()]

    def _reject_task(self, task_id: str, message: str, error_code: str) -> None:
        self._publish_status(
            task_id=task_id,
            state="rejected",
            success=False,
            message=message,
            error_code=error_code,
            recoverable=True,
            replan_requested=True,
        )

    @staticmethod
    def _load_request_context(msg: TaskCommand) -> dict[str, Any]:
        raw_context = (msg.context_json or "").strip()
        if not raw_context:
            return {}
        loaded = json.loads(raw_context)
        if not isinstance(loaded, dict):
            raise ValueError("task context_json must decode to a JSON object")
        return loaded

    def _remaining_task_budget_sec(self, msg: TaskCommand) -> float | None:
        try:
            request_context = self._load_request_context(msg)
        except ValueError:
            request_context = {}
        timeout_context = request_context.get("timeout_context", {})
        if isinstance(timeout_context, dict):
            deadline_unix_sec = timeout_context.get("deadline_unix_sec")
            if isinstance(deadline_unix_sec, int | float):
                return float(deadline_unix_sec) - time.time()
        if msg.timeout_sec > 0.0:
            return float(msg.timeout_sec)
        return None

    @staticmethod
    def _scene_analysis_to_payload(analysis: SceneAnalysis) -> dict[str, Any]:
        return {
            "scene_summary": analysis.scene_summary,
            "visible_objects": analysis.visible_objects,
            "robot_state_summary": analysis.robot_state_summary,
            "ee_pose_interpretation": analysis.ee_pose_interpretation,
            "risks": analysis.risks,
            "confidence": analysis.confidence,
        }

    @staticmethod
    def _requires_missing_skill_rejection(reason: str) -> bool:
        normalized = (reason or "").lower()
        return "unsupported skill" in normalized or "required missing skills" in normalized

    def _publish_planned_task(
        self,
        msg: TaskCommand,
        plan: PlannerResult,
        scene_analysis: SceneAnalysis | None = None,
        fallback_reason: str = "",
        request_context: dict[str, Any] | None = None,
        catalog=None,
    ) -> None:
        planned = TaskCommand()
        planned.dispatch_binding = copy_binding(msg.dispatch_binding)
        if catalog is not None:
            planned.dispatch_binding.expected_registry_epoch = catalog.identity.registry_epoch
            planned.dispatch_binding.expected_registry_generation = catalog.identity.generation
            planned.dispatch_binding.expected_registry_digest = catalog.identity.registry_digest
        planned.source = msg.source
        planned.raw_command = msg.raw_command
        planned.task_type = plan.task_type
        planned.target_name = plan.target_name
        planned.place_name = plan.place_name
        planned.motion_direction = plan.motion_direction
        planned.motion_distance = float(plan.motion_distance)
        planned.priority = msg.priority
        planned.timeout_sec = msg.timeout_sec
        if request_context is None:
            try:
                request_context = self._load_request_context(msg)
            except (ValueError, json.JSONDecodeError):
                request_context = {}
        request_context["planner_source"] = plan.planner_source
        request_context["planner_confidence"] = plan.confidence
        request_context["scene_summary"] = plan.scene_summary
        request_context["planner_reason"] = plan.planner_reason
        request_context["required_missing_skills"] = plan.required_missing_skills
        request_context["scene_analysis"] = (
            self._scene_analysis_to_payload(scene_analysis) if scene_analysis is not None else {}
        )
        request_context["fallback_reason"] = fallback_reason
        planned.context_json = json.dumps(request_context, ensure_ascii=False)
        planned.workflow_steps = [
            workflow_step(
                skill_name=skill_name,
                target_name=plan.target_name,
                place_name=plan.place_name,
                motion_direction=plan.motion_direction,
                motion_distance=plan.motion_distance,
                timeout_sec=float(catalog.timeout_policy.get("default_skill_timeout_sec", msg.timeout_sec))
                if catalog is not None
                else msg.timeout_sec,
            )
            for skill_name in plan.skill_sequence
        ]
        if catalog is not None and planned.dispatch_binding.task_budget.schema_version == 1:
            planned.dispatch_binding.workflow_digest = compute_workflow_digest(
                root_task_id=planned.dispatch_binding.root_task_id or planned.dispatch_binding.task_id,
                task_budget=planned.dispatch_binding.task_budget,
                expected_registry_epoch=catalog.identity.registry_epoch,
                expected_registry_generation=catalog.identity.generation,
                expected_registry_digest=catalog.identity.registry_digest,
                workflow_steps=planned.workflow_steps,
            )
        self._planned_publisher.publish(planned)
        self._publish_status(
            task_id=msg.dispatch_binding.task_id,
            state="planned",
            success=True,
            message=f"planned skills: {plan.skill_sequence}",
        )
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] vlm_task_planner planned "
                f"task_id={msg.dispatch_binding.task_id} source={plan.planner_source} "
                f"confidence={plan.confidence:.3f} target={plan.target_name or '-'} "
                f"place={plan.place_name or '-'} motion={plan.motion_direction or '-'} "
                f"skills={plan.skill_sequence} scene_summary='{plan.scene_summary}' "
                f"fallback_reason='{fallback_reason}'"
            )

    def _analyze_scene_for_planning(
        self,
        msg: TaskCommand,
        scene_snapshot: dict[str, Any],
    ) -> SceneAnalysis:
        remaining_budget = self._remaining_task_budget_sec(msg)
        if remaining_budget is not None and remaining_budget <= 0.0:
            raise RuntimeError("task deadline exceeded before scene analysis")
        request_context = self._load_request_context(msg)
        scene_analysis_context = request_context.get("scene_analysis_context", {})
        if not isinstance(scene_analysis_context, dict):
            scene_analysis_context = {}
        if "task_context" not in scene_analysis_context and request_context:
            scene_analysis_context["task_context"] = request_context
        scene_analysis_context["planning_source"] = msg.source

        messages = build_scene_analysis_messages(
            user_text=msg.raw_command,
            user_context=scene_analysis_context,
            scene_snapshot=scene_snapshot,
            conversation_history=[],
        )
        raw_content, _ = self._api_client.plan(messages, timeout_sec=remaining_budget)
        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] vlm_task_planner scene_analysis task_id={msg.dispatch_binding.task_id} preview={raw_content[:240]!r}"
            )
        return parse_scene_analysis_response(raw_content)

    def _capture_scene_snapshot(self, msg: TaskCommand) -> dict[str, Any]:
        """Capture a scene snapshot from camera buffers (no LLM call)."""
        scene_snapshot = self._scene_buffer.build_snapshot(
            max_scene_age_sec=self._max_scene_age_sec,
            max_image_width=self._api_max_image_width,
            jpeg_quality=self._api_jpeg_quality,
            require_depth=self._require_depth,
            require_pointcloud=self._require_pointcloud,
        )
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] vlm_task_planner scene_snapshot "
                f"task_id={msg.dispatch_binding.task_id} camera={scene_snapshot.get('camera_topic')} "
                f"errors={scene_snapshot.get('errors')}"
            )
        if scene_snapshot["errors"]:
            raise RuntimeError("; ".join(scene_snapshot["errors"]))
        return scene_snapshot

    def _call_planning_api(
        self,
        msg: TaskCommand,
        scene_snapshot: dict[str, Any],
        scene_analysis: SceneAnalysis,
    ) -> PlannerResult:
        """Second LLM call: select skill sequence based on scene analysis."""
        messages = build_chat_messages(
            task_text=msg.raw_command,
            scene_snapshot=scene_snapshot,
            scene_analysis=self._scene_analysis_to_payload(scene_analysis),
            allowed_skills=self._allowed_skills,
            named_poses=self._named_poses,
            named_targets=self._named_targets,
            workspace=self._workspace,
            relative_motion_reference_frame=self._relative_motion_reference_frame,
            relative_motion_direction_mapping=self._relative_motion_direction_mapping,
        )
        remaining_budget = self._remaining_task_budget_sec(msg)
        if remaining_budget is not None and remaining_budget <= 0.0:
            raise RuntimeError("task deadline exceeded before planner response")
        raw_content, _ = self._api_client.plan(messages, timeout_sec=remaining_budget)
        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] vlm_task_planner api_response task_id={msg.dispatch_binding.task_id} preview={raw_content[:240]!r}"
            )
        plan = parse_planner_response(
            raw_content,
            allowed_skills=self._allowed_skills,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
        )
        if plan.confidence < self._min_confidence:
            raise RuntimeError(
                f"planner confidence below threshold: {plan.confidence:.3f} < {self._min_confidence:.3f}"
            )
        return plan

    def _try_vlm_planning(self, msg: TaskCommand) -> tuple[PlannerResult | None, SceneAnalysis | None, str] | None:
        """Run VLM-based planning (scene analysis + skill planning).

        Returns ``(plan, scene_analysis, fallback_reason)`` on success or
        soft failure (plan may be None if both LLM calls failed but fallback
        is allowed).  Returns ``None`` when the task has already been
        rejected via ``_reject_task`` and the caller should return
        immediately.
        """
        scene_snapshot: dict[str, Any] | None = None
        scene_analysis: SceneAnalysis | None = None
        fallback_reason = ""

        # Step 1: capture scene snapshot + first LLM call (scene analysis)
        try:
            scene_snapshot = self._capture_scene_snapshot(msg)
            scene_analysis = self._analyze_scene_for_planning(msg, scene_snapshot)
            if self._debug and scene_analysis is not None:
                self.get_logger().info(
                    f"[embodied-debug] vlm_task_planner scene_summary "
                    f"task_id={msg.dispatch_binding.task_id} summary={scene_analysis.scene_summary!r}"
                )
        except Exception as exc:
            fallback_reason = str(exc)
            if self._requires_missing_skill_rejection(fallback_reason):
                self._reject_task(msg.dispatch_binding.task_id, fallback_reason, "MISSING_REQUIRED_SKILLS")
                return None
            self.get_logger().warning(
                f"[embodied-debug] vlm_task_planner scene analysis failed task_id={msg.dispatch_binding.task_id}: {fallback_reason}"
            )
            if self._planner_mode == "vlm_api" and not self._fallback_enabled:
                self._reject_task(
                    msg.dispatch_binding.task_id, f"scene analysis failed: {fallback_reason}", "SCENE_ANALYSIS_FAILED"
                )
                return None

        # Step 2: second LLM call (skill planning) - scene_analysis preserved on failure
        if scene_snapshot is not None and scene_analysis is not None:
            try:
                plan = self._call_planning_api(msg, scene_snapshot, scene_analysis)
                return plan, scene_analysis, fallback_reason
            except Exception as exc:
                fallback_reason = str(exc)
                if self._requires_missing_skill_rejection(fallback_reason):
                    self._reject_task(msg.dispatch_binding.task_id, fallback_reason, "MISSING_REQUIRED_SKILLS")
                    return None
                self.get_logger().warning(
                    f"[embodied-debug] vlm_task_planner VLM planning failed task_id={msg.dispatch_binding.task_id}: {fallback_reason}"
                )
                if self._planner_mode == "vlm_api" and not self._fallback_enabled:
                    self._reject_task(
                        msg.dispatch_binding.task_id,
                        f"vlm planning failed: {fallback_reason}",
                        "VLM_PLANNING_FAILED",
                    )
                    return None

        return None, scene_analysis, fallback_reason

    def _try_fallback_planning(self, msg: TaskCommand, fallback_reason: str) -> PlannerResult | None:
        """Run rule-based fallback planning.

        Returns a ``PlannerResult`` on success.  Returns ``None`` when the
        task has already been rejected and the caller should return.
        """
        if self._planner_mode == "vlm_api" and not self._fallback_enabled:
            self._reject_task(msg.dispatch_binding.task_id, "vlm planner produced no valid plan", "EMPTY_VLM_PLAN")
            return None

        plan = fallback_plan_from_text(
            msg.raw_command,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
            skill_aliases=self._skill_aliases or None,
        )
        if not plan.skill_sequence:
            self._reject_task(
                msg.dispatch_binding.task_id, fallback_reason or "unsupported command", "UNSUPPORTED_COMMAND"
            )
            return None
        return plan

    def _handle_task_command(self, msg: TaskCommand) -> None:
        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] vlm_task_planner received task_id={msg.dispatch_binding.task_id} text='{msg.raw_command}'"
            )

        catalog = self._catalog.current
        if catalog is None:
            self._reject_task(msg.dispatch_binding.task_id, "catalog snapshot is not ready", "SKILL_REGISTRY_NOT_READY")
            return
        self._allowed_skills = sorted(catalog.planner_visible_names)
        self._skill_aliases = {name: list(values) for name, values in catalog.aliases.items()}

        self._publish_status(
            task_id=msg.dispatch_binding.task_id, state="planning", success=True, message="planning task"
        )

        plan: PlannerResult | None = None
        scene_analysis: SceneAnalysis | None = None
        fallback_reason = ""

        if self._planner_mode in {"vlm_api", "hybrid"}:
            result = self._try_vlm_planning(msg)
            if result is None:
                return
            plan, scene_analysis, fallback_reason = result

        if plan is None:
            plan = self._try_fallback_planning(msg, fallback_reason)
            if plan is None:
                return

        if plan.required_missing_skills:
            rejection_reason = f"required missing skills: {', '.join(plan.required_missing_skills)}"
            if plan.planner_reason:
                rejection_reason = f"{rejection_reason}; {plan.planner_reason}"
            self._reject_task(msg.dispatch_binding.task_id, rejection_reason, "MISSING_REQUIRED_SKILLS")
            return
        if any(skill_name not in catalog.planner_visible_names for skill_name in plan.skill_sequence):
            self._reject_task(
                msg.dispatch_binding.task_id,
                "planner returned an entry outside the captured planner-visible set",
                "SKILL_SCHEMA_INVALID",
            )
            return

        try:
            _parsed_context: dict[str, Any] = self._load_request_context(msg)
        except (ValueError, json.JSONDecodeError):
            _parsed_context = {}
        self._publish_planned_task(
            msg,
            plan,
            scene_analysis=scene_analysis,
            fallback_reason=fallback_reason,
            request_context=_parsed_context,
            catalog=catalog,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLMTaskPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
