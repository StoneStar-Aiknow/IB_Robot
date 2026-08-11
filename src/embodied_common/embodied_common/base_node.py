"""Shared base node for embodied pipeline task nodes."""

from __future__ import annotations

from rclpy.node import Node

from ibrobot_msgs.msg import TaskStatus


class BaseTaskNode(Node):
    """Base class providing shared status publishing utilities."""

    def _publish_status(
        self,
        task_id: str,
        state: str,
        success: bool,
        message: str,
        current_skill: str = "",
        completed_skills: list[str] | None = None,
        error_code: str = "",
        recoverable: bool = False,
        replan_requested: bool = False,
    ) -> None:
        status = TaskStatus()
        status.schema_version = 1
        status.task_id = task_id
        status.state = state
        status.success = success
        status.current_skill = current_skill
        status.completed_skills = completed_skills or []
        status.error_code = error_code
        status.message = message
        status.recoverable = recoverable
        status.replan_requested = replan_requested
        self._status_publisher.publish(status)
