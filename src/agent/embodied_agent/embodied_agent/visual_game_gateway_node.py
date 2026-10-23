"""Asynchronous Agent control plane for visual games."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from uuid import uuid4

import rclpy
from rclpy.node import Node

from embodied_agent.visual_game_qos import visual_game_event_qos
from embodied_agent.visual_games import build_game_request
from embodied_common.visual_game_contracts import (
    build_visual_game_capability_view,
    get_visual_game_terminal_error,
    load_visual_game_policies_json,
    validate_visual_game_result,
)
from ibrobot_msgs.msg import SceneAnalysisRequest, SceneAnalysisResult, VisualGameEvent
from ibrobot_msgs.srv import GetVisualGameResult, StartVisualGame

MAX_VISUAL_GAME_REQUEST_ID_LENGTH = 128


@dataclass
class _GameResultRecord:
    game_name: str
    handler: str
    announce: bool
    deadline_monotonic: float
    execution_id: str = ""
    terminal: bool = False
    terminal_at_monotonic: float | None = None
    success: bool = False
    scene_summary: str = ""
    result_json: str = ""
    error_code: str = ""
    message: str = "visual game is running"


class VisualGameGatewayNode(Node):
    """Start visual games and retain their results for later queries."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("visual_game_gateway_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("perception_enabled", False)
        self.declare_parameter("robot_name", "unknown")
        self.declare_parameter("visual_games_json", "{}")
        self.declare_parameter("perception_request_topic", "/embodied/perception_request")
        self.declare_parameter("perception_result_topic", "/embodied/perception_result")
        self.declare_parameter("start_service", "/embodied/start_visual_game")
        self.declare_parameter("result_service", "/embodied/get_visual_game_result")
        self.declare_parameter("event_topic", "/embodied/visual_game_events")
        self.declare_parameter("result_capacity", 128)
        self.declare_parameter("model_idle_timeout_sec", 120.0)
        self.declare_parameter("visual_game_timeout_sec", 130.0)
        self.declare_parameter("result_retention_sec", 300.0)
        self.declare_parameter("debug_tracing", False)

        self._perception_enabled = self.get_parameter("perception_enabled").value
        self._games = load_visual_game_policies_json(self.get_parameter("visual_games_json").value)
        self._result_capacity = int(self.get_parameter("result_capacity").value)
        self._model_idle_timeout_sec = float(self.get_parameter("model_idle_timeout_sec").value)
        self._visual_game_timeout_sec = float(self.get_parameter("visual_game_timeout_sec").value)
        self._result_retention_sec = float(self.get_parameter("result_retention_sec").value)
        self._debug = self.get_parameter("debug_tracing").value
        self._records: OrderedDict[str, _GameResultRecord] = OrderedDict()
        # Keep public idempotency keys separate from downstream execution generations.
        self._perception_request_ids: dict[str, str] = {}

        request_topic = self.get_parameter("perception_request_topic").value
        result_topic = self.get_parameter("perception_result_topic").value
        start_service = self.get_parameter("start_service").value
        result_service = self.get_parameter("result_service").value
        event_topic = self.get_parameter("event_topic").value
        game_view = build_visual_game_capability_view(
            str(self.get_parameter("robot_name").value),
            self._games,
            timeout_sec=self._visual_game_timeout_sec,
            result_retention_sec=self._result_retention_sec,
            result_capacity=self._result_capacity,
            start_service=start_service,
            result_service=result_service,
            event_topic=event_topic,
        )
        self._config_digest = game_view["config_digest"]
        self._request_publisher = self.create_publisher(SceneAnalysisRequest, request_topic, 10)
        self._event_publisher = self.create_publisher(
            VisualGameEvent,
            event_topic,
            visual_game_event_qos(
                depth=self._result_capacity * 2,
                lifespan_sec=self._result_retention_sec,
            ),
        )
        self.create_subscription(SceneAnalysisResult, result_topic, self._handle_perception_result, 10)
        self.create_service(StartVisualGame, start_service, self._handle_start_game)
        self.create_service(GetVisualGameResult, result_service, self._handle_get_result)
        self.create_timer(1.0, self._expire_records)

        self.get_logger().info(
            "[embodied-debug] visual game gateway ready: "
            f"start_service={start_service}, result_service={result_service}"
        )

    def _expire_records(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        expired_ids = []
        timed_out_ids = []
        for request_id, record in self._records.items():
            if not record.terminal and current >= record.deadline_monotonic:
                record.terminal = True
                record.terminal_at_monotonic = current
                record.error_code = "GAME_RESULT_TIMEOUT"
                record.message = "visual game did not reach a terminal result before its deadline"
                timed_out_ids.append(request_id)
            elif (
                record.terminal
                and record.terminal_at_monotonic is not None
                and current - record.terminal_at_monotonic >= self._result_retention_sec
            ):
                expired_ids.append(request_id)
        for request_id in timed_out_ids:
            self._records.move_to_end(request_id)
            self._publish_event(request_id, self._records[request_id], state="failed")
        for request_id in expired_ids:
            record = self._records.pop(request_id, None)
            if record is not None and record.execution_id:
                self._perception_request_ids.pop(record.execution_id, None)

    def _publish_event(self, request_id: str, record: _GameResultRecord, *, state: str) -> None:
        event = VisualGameEvent()
        event.request_id = request_id
        event.execution_id = record.execution_id
        event.game_name = record.game_name
        event.handler = record.handler
        event.announce = record.announce
        event.state = state
        event.success = record.success
        event.scene_summary = record.scene_summary
        event.result_json = record.result_json
        event.config_digest = self._config_digest
        event.error_code = record.error_code
        event.message = record.message
        self._event_publisher.publish(event)

    def _publish_rejection_event(
        self,
        request_id: str,
        game_name: str,
        error_code: str,
        message: str,
        *,
        announce_override: bool | None = None,
    ) -> None:
        """Publish a non-ledger failure for a request rejected before admission."""
        policy = self._games.get(game_name)
        if not request_id or not isinstance(policy, dict) or not policy.get("enabled", False):
            return
        announce = bool(announce_override) if announce_override is not None else bool(policy.get("announce", False))
        now = time.monotonic()
        record = _GameResultRecord(
            game_name=game_name,
            handler=policy["handler"],
            announce=announce,
            deadline_monotonic=now,
            terminal=True,
            terminal_at_monotonic=now,
            error_code=error_code,
            message=message,
        )
        self._publish_event(request_id, record, state="failed")

    def _reject(
        self,
        response,
        request_id: str,
        game_name: str,
        error_code: str,
        message: str,
        *,
        announce_override: bool | None = None,
    ):
        response.error_code = error_code
        response.message = message
        self._publish_rejection_event(request_id, game_name, error_code, message, announce_override=announce_override)
        return response

    def _reserve_result_slot(self, now: float) -> bool:
        self._expire_records(now)
        return len(self._records) < self._result_capacity

    def _active_request_id(self) -> str | None:
        return next((request_id for request_id, record in self._records.items() if not record.terminal), None)

    @staticmethod
    def _format_perception_failure_message(raw_error_code: str, mapped_error_code: str, message: str) -> str:
        """Preserve the perception-internal cause inside the Gateway failure message.

        When a perception-internal code is collapsed to ``PERCEPTION_FAILED`` the
        original code is folded into the message so callers and logs can still
        see the root cause without it leaking onto the public error_code field.
        Codes already on the Gateway public surface pass through unchanged.
        """
        if not raw_error_code or raw_error_code == mapped_error_code:
            return message or mapped_error_code
        return (
            f"{mapped_error_code}: {raw_error_code}: {message}" if message else f"{mapped_error_code}: {raw_error_code}"
        )

    def _handle_start_game(self, request, response):
        response.config_digest = self._config_digest
        request_id = request.request_id.strip()
        game_name = request.game_name.strip()
        if not request_id:
            response.error_code = "INVALID_REQUEST_ID"
            response.message = "visual game request_id must be non-empty"
            return response
        if len(request_id) > MAX_VISUAL_GAME_REQUEST_ID_LENGTH:
            response.error_code = "INVALID_REQUEST_ID"
            response.message = f"visual game request_id must be at most {MAX_VISUAL_GAME_REQUEST_ID_LENGTH} characters"
            return response
        if not request.expected_config_digest or request.expected_config_digest != self._config_digest:
            return self._reject(
                response,
                request_id,
                game_name,
                "CONFIG_MISMATCH",
                "local visual game configuration does not match the running gateway",
            )
        policy = self._games.get(game_name)
        if not isinstance(policy, dict) or not policy.get("enabled", False):
            response.error_code = "GAME_NOT_ENABLED"
            response.message = f"visual game is not enabled: {game_name}"
            return response
        if not self._perception_enabled:
            return self._reject(
                response,
                request_id,
                game_name,
                "PERCEPTION_DISABLED",
                "visual game perception is disabled",
            )
        existing = self._records.get(request_id)
        if existing is not None:
            if existing.game_name != game_name or existing.handler != policy["handler"]:
                response.error_code = "DUPLICATE_REQUEST_ID"
                response.message = "request_id is already bound to a different visual game request"
                return response
            response.accepted = True
            response.duplicate = True
            response.request_id = request_id
            response.message = "visual game request already accepted"
            return response
        now = time.monotonic()
        self._expire_records(now)
        active_request_id = self._active_request_id()
        if active_request_id is not None:
            return self._reject(
                response,
                request_id,
                game_name,
                "GAME_BUSY",
                f"visual game request is already running: {active_request_id}",
                announce_override=False,
            )
        if self._request_publisher.get_subscription_count() <= 0:
            return self._reject(
                response,
                request_id,
                game_name,
                "PERCEPTION_UNAVAILABLE",
                "visual game perception request subscriber is unavailable",
            )
        if not self._reserve_result_slot(now):
            return self._reject(
                response,
                request_id,
                game_name,
                "GAME_CAPACITY_EXHAUSTED",
                "visual game result ledger is full; retained records are not evicted early",
            )
        game_request = build_game_request(
            game_name,
            handler=policy["handler"],
            request_id=uuid4().hex,
            timeout_sec=self._model_idle_timeout_sec,
        )
        execution_id = game_request.request_id
        self._records[request_id] = _GameResultRecord(
            game_name=game_name,
            handler=policy["handler"],
            announce=bool(policy.get("announce", False)),
            deadline_monotonic=now + self._visual_game_timeout_sec,
            execution_id=execution_id,
        )
        self._perception_request_ids[execution_id] = request_id
        try:
            self._request_publisher.publish(game_request)
        except Exception:
            self._records.pop(request_id, None)
            self._perception_request_ids.pop(execution_id, None)
            raise
        response.accepted = True
        response.request_id = request_id
        response.message = "visual game request accepted"
        self._publish_event(request_id, self._records[request_id], state="accepted")
        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] visual game request accepted: game={game_name}, request_id={request_id}"
            )
        return response

    def _handle_perception_result(self, result: SceneAnalysisResult) -> None:
        self._expire_records()
        public_request_id = self._perception_request_ids.get(result.request_id)
        record = self._records.get(public_request_id) if public_request_id else None
        if record is None:
            if self._debug:
                self.get_logger().info(
                    "[embodied-debug] visual game perception result dropped: "
                    f"unknown request_id={result.request_id}, source={result.source}"
                )
            return
        if record.terminal:
            if self._debug:
                self.get_logger().info(
                    "[embodied-debug] visual game perception result dropped: "
                    f"request_id={result.request_id} already terminal, source={result.source}"
                )
            return
        expected_source = f"game.{record.game_name}"
        if result.source != expected_source:
            if self._debug:
                self.get_logger().info(
                    "[embodied-debug] visual game perception result dropped: "
                    f"request_id={result.request_id}, expected source={expected_source}, "
                    f"got source={result.source}"
                )
            return
        record.terminal = True
        record.terminal_at_monotonic = time.monotonic()
        result_payload = {
            "scene_summary": result.scene_summary,
            "visible_objects": list(result.visible_objects),
            "robot_state_summary": result.robot_state_summary,
            "ee_pose_interpretation": result.ee_pose_interpretation,
            "risks": list(result.risks),
            "confidence": float(result.confidence),
        }
        result_error = validate_visual_game_result(record.handler, result_payload) if result.success else None
        if result_error is not None:
            record.success = False
            record.error_code = "INVALID_GAME_RESULT"
            record.message = result_error
        elif result.success and (terminal_error := get_visual_game_terminal_error(record.handler, result_payload)):
            record.success = False
            record.error_code, record.message = terminal_error
        elif result.success:
            record.success = True
            record.scene_summary = result.scene_summary
            record.result_json = json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
            record.message = result.message
        else:
            record.success = False
            record.error_code = "PERCEPTION_FAILED"
            record.message = self._format_perception_failure_message(
                result.error_code, record.error_code, result.message
            )
        self._records.move_to_end(public_request_id)
        self._publish_event(public_request_id, record, state="succeeded" if record.success else "failed")

    def _handle_get_result(self, request, response):
        self._expire_records()
        response.config_digest = self._config_digest
        request_id = request.request_id.strip()
        record = self._records.get(request_id)
        if record is None:
            response.error_code = "GAME_REQUEST_NOT_FOUND"
            response.message = f"visual game request not found: {request_id}"
            return response
        response.found = True
        response.terminal = record.terminal
        response.success = record.success
        response.game_name = record.game_name
        response.scene_summary = record.scene_summary
        response.result_json = record.result_json
        response.error_code = record.error_code
        response.message = record.message
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualGameGatewayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
