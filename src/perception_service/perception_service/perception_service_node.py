"""Continuous multimodal scene understanding node."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ibrobot_msgs.msg import SceneAnalysisRequest, SceneAnalysisResult
from perception_service.api_client import VLMAPIClient
from perception_service.prompt_builder import build_scene_analysis_messages
from perception_service.response_parser import SceneAnalysis, parse_scene_analysis_response
from perception_service.scene_snapshot import SceneSnapshotBuffer


class PerceptionServiceNode(Node):
    """Understand live scene content from image, robot state, and user context."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("perception_service_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("request_topic", "/embodied/perception_request")
        self.declare_parameter("text_input_topic", "/embodied/perception_text")
        self.declare_parameter("result_topic", "/embodied/perception_result")
        self.declare_parameter("summary_topic", "/embodied/perception_summary")
        self.declare_parameter("default_session_id", "default")
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
        self.declare_parameter("api_max_image_width", 320)
        self.declare_parameter("api_jpeg_quality", 70)
        self.declare_parameter("max_history_turns", 4)
        self.declare_parameter("debug_tracing", False)

        self._request_topic = self.get_parameter("request_topic").get_parameter_value().string_value
        self._text_input_topic = self.get_parameter("text_input_topic").get_parameter_value().string_value
        self._result_topic = self.get_parameter("result_topic").get_parameter_value().string_value
        self._summary_topic = self.get_parameter("summary_topic").get_parameter_value().string_value
        self._default_session_id = self.get_parameter("default_session_id").get_parameter_value().string_value
        self._max_scene_age_sec = self.get_parameter("max_scene_age_sec").get_parameter_value().double_value
        self._require_depth = self.get_parameter("require_depth").get_parameter_value().bool_value
        self._require_pointcloud = self.get_parameter("require_pointcloud").get_parameter_value().bool_value
        self._api_max_image_width = self.get_parameter("api_max_image_width").get_parameter_value().integer_value
        self._api_jpeg_quality = self.get_parameter("api_jpeg_quality").get_parameter_value().integer_value
        self._max_history_turns = self.get_parameter("max_history_turns").get_parameter_value().integer_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self._scene_buffer = SceneSnapshotBuffer.from_node(self)
        self._api_client = VLMAPIClient(
            provider=self.get_parameter("api_provider").get_parameter_value().string_value,
            base_url=self.get_parameter("api_base_url").get_parameter_value().string_value,
            api_key_env=self.get_parameter("api_key_env").get_parameter_value().string_value,
            model=self.get_parameter("api_model").get_parameter_value().string_value,
            timeout_sec=self.get_parameter("api_timeout_sec").get_parameter_value().double_value,
        )

        self._result_publisher = self.create_publisher(SceneAnalysisResult, self._result_topic, 10)
        self._summary_publisher = self.create_publisher(String, self._summary_topic, 10)
        self.create_subscription(SceneAnalysisRequest, self._request_topic, self._handle_request_cb, 10)
        self.create_subscription(String, self._text_input_topic, self._handle_text_input, 10)
        self._conversation_history: dict[str, list[dict[str, str]]] = {}
        self._max_history_sessions: int = 32

        self.get_logger().info(
            "[embodied-debug] perception_service ready: "
            f"request_topic={self._request_topic}, text_input_topic={self._text_input_topic}, "
            f"result_topic={self._result_topic}"
        )

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        return session_id.strip()

    def _generate_request_id(self) -> str:
        return f"scene-{time.time_ns()}"

    def _load_context_json(self, raw_text: str) -> dict[str, Any]:
        if not raw_text.strip():
            return {}
        loaded = json.loads(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError("context_json must decode to a JSON object")
        return loaded

    def _session_history(self, session_id: str) -> list[dict[str, str]]:
        if session_id not in self._conversation_history:
            if len(self._conversation_history) >= self._max_history_sessions:
                oldest = next(iter(self._conversation_history))
                del self._conversation_history[oldest]
            self._conversation_history[session_id] = []
        return self._conversation_history[session_id]

    def _remember_turn(self, session_id: str, user_text: str, analysis: SceneAnalysis) -> None:
        history = self._session_history(session_id)
        if user_text.strip():
            history.append({"role": "user", "text": user_text.strip()})
        assistant_text = (
            f"场景摘要：{analysis.scene_summary}\n"
            f"机械臂状态：{analysis.robot_state_summary}\n"
            f"风险：{'; '.join(analysis.risks) if analysis.risks else '无明显风险'}"
        ).strip()
        history.append({"role": "assistant", "text": assistant_text})
        keep_items = max(0, self._max_history_turns * 2)
        if keep_items:
            del history[:-keep_items]

    def _publish_result(
        self,
        request: SceneAnalysisRequest,
        success: bool,
        message: str,
        analysis: SceneAnalysis | None = None,
        raw_response: str = "",
        error_code: str = "",
    ) -> None:
        result = SceneAnalysisResult()
        result.request_id = request.request_id
        result.source = request.source
        result.session_id = request.session_id
        result.success = success
        result.scene_summary = analysis.scene_summary if analysis else ""
        result.visible_objects = analysis.visible_objects if analysis else []
        result.robot_state_summary = analysis.robot_state_summary if analysis else ""
        result.ee_pose_interpretation = analysis.ee_pose_interpretation if analysis else ""
        result.risks = analysis.risks if analysis else []
        result.confidence = float(analysis.confidence if analysis else 0.0)
        result.raw_response = raw_response
        result.error_code = error_code
        result.message = message
        self._result_publisher.publish(result)

        if success and analysis:
            summary = String()
            summary.data = f"[{request.session_id}] {analysis.scene_summary} | 机械臂: {analysis.robot_state_summary}"
            self._summary_publisher.publish(summary)
            self.get_logger().info(f"大模型回答: {analysis.scene_summary}")
        else:
            error_text = message
            if error_code:
                error_text = f"{error_code}: {message}"
            self.get_logger().warning(error_text)

    def _analyze(self, request: SceneAnalysisRequest) -> tuple[SceneAnalysis, str]:
        user_context = self._load_context_json(request.context_json)
        scene_snapshot = self._scene_buffer.build_snapshot(
            max_scene_age_sec=self._max_scene_age_sec,
            max_image_width=self._api_max_image_width,
            jpeg_quality=self._api_jpeg_quality,
            require_depth=self._require_depth,
            require_pointcloud=self._require_pointcloud,
        )
        if self._debug:
            self.get_logger().debug(
                "[embodied-debug] perception_service scene_snapshot "
                f"request_id={request.request_id} errors={scene_snapshot.get('errors')}"
            )
        if scene_snapshot["errors"]:
            raise RuntimeError("; ".join(scene_snapshot["errors"]))

        messages = build_scene_analysis_messages(
            user_text=request.user_text,
            user_context=user_context,
            scene_snapshot=scene_snapshot,
            conversation_history=self._session_history(request.session_id),
        )
        raw_content, _ = self._api_client.analyze(
            messages,
            timeout_sec=request.timeout_sec if request.timeout_sec > 0.0 else None,
        )
        if self._debug:
            self.get_logger().debug(
                "[embodied-debug] perception_service api_response "
                f"request_id={request.request_id} preview={raw_content[:240]!r}"
            )
        return parse_scene_analysis_response(raw_content), raw_content

    def _handle_text_input(self, msg: String) -> None:
        request = SceneAnalysisRequest()
        request.request_id = self._generate_request_id()
        request.source = "text_topic"
        request.session_id = self._default_session_id
        request.user_text = msg.data
        request.context_json = ""
        request.timeout_sec = 0.0
        threading.Thread(target=self._handle_request, args=(request,), daemon=True).start()

    def _handle_request_cb(self, request: SceneAnalysisRequest) -> None:
        threading.Thread(target=self._handle_request, args=(request,), daemon=True).start()

    def _handle_request(self, request: SceneAnalysisRequest) -> None:
        if not request.request_id:
            request.request_id = self._generate_request_id()
        request.session_id = self._normalize_session_id(request.session_id)
        if not request.session_id:
            request.session_id = self._default_session_id
        if not request.source:
            request.source = "request_topic"

        if self._debug:
            self.get_logger().debug(
                "[embodied-debug] perception_service request "
                f"request_id={request.request_id} session_id={request.session_id} "
                f"text={request.user_text!r}"
            )
        self.get_logger().info(f"用户询问: {request.user_text}")

        try:
            analysis, raw_response = self._analyze(request)
        except json.JSONDecodeError as exc:
            self._publish_result(
                request,
                success=False,
                message=f"invalid context_json: {exc}",
                error_code="INVALID_CONTEXT_JSON",
            )
            return
        except Exception as exc:
            self._publish_result(
                request,
                success=False,
                message=f"scene analysis failed: {exc}",
                error_code="SCENE_ANALYSIS_FAILED",
            )
            return

        self._remember_turn(request.session_id, request.user_text, analysis)
        self._publish_result(
            request,
            success=True,
            message="scene analysis completed",
            analysis=analysis,
            raw_response=raw_response,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
