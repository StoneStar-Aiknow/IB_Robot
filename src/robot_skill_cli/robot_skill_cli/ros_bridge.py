"""Lazy ROS 2 clients for the Capability Gateway runtime surface."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from typing import Any

from embodied_common.skill_request import skill_goal_uuid
from robot_skill_cli.output import EXIT_ROS_UNAVAILABLE, EXIT_TIMEOUT


class BridgeError(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


class RosBridge:
    """Own Gateway clients while keeping rclpy imports out of catalog commands."""

    def __init__(
        self,
        *,
        status_service: str,
        validate_skill_service: str,
        skill_action: str,
    ) -> None:
        self._status_service = status_service
        self._validate_skill_service = validate_skill_service
        self._skill_action = skill_action.rstrip("/")
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._owns_rclpy_context = False
        self._status_client = None
        self._validate_client = None
        self._skill_client = None
        self._cancel_client = None
        self._GetSkillGatewayStatus = None
        self._ValidateSkill = None
        self._SkillCommand = None
        self._CancelGoal = None

    def start(self) -> bool:
        try:
            import rclpy
            from action_msgs.srv import CancelGoal
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor

            from ibrobot_msgs.action import SkillCommand
            from ibrobot_msgs.srv import GetSkillGatewayStatus, ValidateSkill
        except Exception:
            return False

        try:
            self._owns_rclpy_context = not rclpy.ok()
            if self._owns_rclpy_context:
                rclpy.init()
            suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            self._node = rclpy.create_node(f"robot_skill_cli_{suffix}")
            callback_group = ReentrantCallbackGroup()
            self._status_client = self._node.create_client(
                GetSkillGatewayStatus,
                self._status_service,
                callback_group=callback_group,
            )
            self._validate_client = self._node.create_client(
                ValidateSkill,
                self._validate_skill_service,
                callback_group=callback_group,
            )
            self._skill_client = ActionClient(
                self._node,
                SkillCommand,
                self._skill_action,
                callback_group=callback_group,
            )
            self._cancel_client = self._node.create_client(
                CancelGoal,
                f"{self._skill_action}/_action/cancel_goal",
                callback_group=callback_group,
            )
            self._GetSkillGatewayStatus = GetSkillGatewayStatus
            self._ValidateSkill = ValidateSkill
            self._SkillCommand = SkillCommand
            self._CancelGoal = CancelGoal
            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()
        except Exception:
            self.close()
            return False
        return True

    @staticmethod
    def _wait_future(future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done()

    def wait_future(self, future, *, timeout_sec: float) -> bool:
        return self._wait_future(future, timeout_sec)

    def _call_service(self, client, request, *, service_name: str, timeout_sec: float):
        if client is None or not client.wait_for_service(timeout_sec=timeout_sec):
            raise BridgeError(
                "SERVER_UNAVAILABLE",
                f"service unavailable: {service_name}",
                exit_code=EXIT_ROS_UNAVAILABLE,
            )
        try:
            future = client.call_async(request)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc
        if not self._wait_future(future, timeout_sec):
            raise BridgeError(
                "RESULT_TIMEOUT",
                f"service response timed out: {service_name}",
                exit_code=EXIT_TIMEOUT,
            )
        try:
            return future.result()
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def get_status(
        self,
        *,
        task_id: str = "",
        payload_hash: str = "",
        timeout_sec: float,
    ) -> dict[str, Any]:
        if self._GetSkillGatewayStatus is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._GetSkillGatewayStatus.Request()
        request.task_id = task_id
        request.payload_hash = payload_hash
        response = self._call_service(
            self._status_client,
            request,
            service_name=self._status_service,
            timeout_sec=timeout_sec,
        )
        return {
            "schema_version": int(response.schema_version),
            "robot_name": response.robot_name,
            "motion_authorized": bool(response.motion_authorized),
            "active_control_mode": response.active_control_mode,
            "busy": bool(response.busy),
            "active_task_id": response.active_task_id,
            "default_skill_timeout_sec": float(response.default_skill_timeout_sec),
            "task_budget_sec": float(response.task_budget_sec),
            "rpc_timeout_sec": float(response.rpc_timeout_sec),
            "config_digest": response.config_digest,
            "request_state": response.request_state,
            "request_error_code": response.request_error_code,
            "capabilities": [
                {
                    "name": capability.name,
                    "ready": bool(capability.ready),
                    "reason": capability.reason,
                    "required_control_mode": capability.required_control_mode,
                }
                for capability in response.capabilities
            ],
        }

    def validate_skill(self, payload: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
        if self._ValidateSkill is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._ValidateSkill.Request()
        request.skill_name = payload["skill_name"]
        request.target_name = payload["target_name"]
        request.place_name = payload["place_name"]
        request.motion_direction = payload["motion_direction"]
        request.motion_distance = float(payload["motion_distance"] or 0.0)
        response = self._call_service(
            self._validate_client,
            request,
            service_name=self._validate_skill_service,
            timeout_sec=timeout_sec,
        )
        return {"allowed": bool(response.allowed), "reason": response.reason}

    def wait_for_skill_server(self, *, timeout_sec: float) -> bool:
        if self._skill_client is None:
            return False
        try:
            return self._skill_client.wait_for_server(timeout_sec=timeout_sec)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def send_skill_goal(self, payload: dict[str, Any], *, task_id: str, feedback_callback=None):
        from unique_identifier_msgs.msg import UUID

        if self._skill_client is None or self._SkillCommand is None:
            raise BridgeError("ROS_UNAVAILABLE", "skill action client is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        goal = self._SkillCommand.Goal()
        goal.task_id = task_id
        goal.skill_name = payload["skill_name"]
        goal.target_name = payload["target_name"]
        goal.place_name = payload["place_name"]
        goal.motion_direction = payload["motion_direction"]
        goal.motion_distance = float(payload["motion_distance"] or 0.0)
        goal.timeout_sec = float(payload["timeout_sec"])

        def on_feedback(feedback) -> None:
            if feedback_callback is None:
                return
            message = getattr(feedback, "feedback", feedback)
            feedback_callback({"state": str(message.state), "detail": str(message.detail)})

        goal_uuid = UUID(uuid=list(skill_goal_uuid(task_id).bytes))
        try:
            return self._skill_client.send_goal_async(goal, feedback_callback=on_feedback, goal_uuid=goal_uuid)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def cancel_task(self, task_id: str, *, timeout_sec: float) -> dict[str, Any]:
        if self._cancel_client is None or self._CancelGoal is None:
            raise BridgeError("ROS_UNAVAILABLE", "cancel client is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._CancelGoal.Request()
        request.goal_info.goal_id.uuid = list(skill_goal_uuid(task_id).bytes)
        response = self._call_service(
            self._cancel_client,
            request,
            service_name=f"{self._skill_action}/_action/cancel_goal",
            timeout_sec=timeout_sec,
        )
        return {
            "accepted": int(response.return_code) == 0 and bool(response.goals_canceling),
            "return_code": int(response.return_code),
        }

    def cancel_goal(self, goal_handle, result_future, *, timeout_sec: float) -> bool:
        if result_future.done():
            return True
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            return False
        if not self._wait_future(cancel_future, timeout_sec):
            return False
        try:
            cancel_future.result()
        except Exception:
            return False
        return self._wait_future(result_future, timeout_sec)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._executor is not None:
                self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            if self._node is not None:
                self._node.destroy_node()
        if self._owns_rclpy_context:
            with contextlib.suppress(Exception):
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
        self._node = None
        self._executor = None
        self._spin_thread = None
