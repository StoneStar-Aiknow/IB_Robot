"""Integration test for the Phase-1 ROS path against a mock skill_executor.

Spins a mock ValidateSkill service + SkillCommand action server (which emits
feedback and succeeds), then drives RosBridge + the execute_skill tool
end-to-end. Validates: service call, action goal acceptance, feedback
streaming, and result handling. Skips when rclpy/ibrobot_msgs are unavailable.

Run with the ROS overlay sourced and ROS pytest plugins disabled:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest src/robot_mcp/test/test_bridge.py
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

rclpy_available = True
try:
    import rclpy  # noqa: F401

    from ibrobot_msgs.action import SkillCommand  # noqa: F401
    from ibrobot_msgs.srv import ValidateSkill  # noqa: F401
except Exception:  # noqa: BLE001
    rclpy_available = False


class _RecordingContext:
    """Minimal stand-in for mcp's Context that captures info() notifications."""

    def __init__(self) -> None:
        self.logs: list[str] = []

    async def info(self, message: str) -> None:
        self.logs.append(message)


@pytest.mark.skipif(not rclpy_available, reason="rclpy / ibrobot_msgs not available")
def test_validate_and_execute_against_mock_skill_executor():
    import rclpy
    from rclpy.action import ActionServer
    from rclpy.executors import SingleThreadedExecutor

    from ibrobot_msgs.action import SkillCommand
    from ibrobot_msgs.msg import TaskStatus
    from ibrobot_msgs.srv import ValidateSkill
    from robot_mcp import server
    from robot_mcp.ros_bridge import RosBridge

    if not rclpy.ok():
        rclpy.init()

    # --- mock skill_executor + safety_guard ---
    mock = rclpy.create_node("mock_skill_executor_for_test")
    mock_exec = SingleThreadedExecutor()
    mock_exec.add_node(mock)
    spin_thread = threading.Thread(target=mock_exec.spin, daemon=True)
    spin_thread.start()

    skill_action = "/embodied/execute_skill_test"
    validate_service = "/embodied/validate_skill_test"

    def _validate_cb(_req, response):
        response.allowed = True
        response.reason = "allowed by mock safety_guard"
        return response

    mock.create_service(ValidateSkill, validate_service, _validate_cb)

    def _execute_cb(goal_handle):
        for i in range(3):
            fb = SkillCommand.Feedback()
            fb.state = "running"
            fb.detail = f"primitive {i + 1}/3"
            goal_handle.publish_feedback(fb)
            time.sleep(0.03)
        goal_handle.succeed()
        result = SkillCommand.Result()
        result.success = True
        result.message = "skill complete"
        result.executed_primitives = ["move_to_named_pose", "open_gripper"]
        return result

    ActionServer(mock, SkillCommand, skill_action, _execute_cb)
    time.sleep(0.8)  # let DDS discover the server

    # --- bridge + tool wiring ---
    server._catalog = server.build_catalog(config_name="so101_single_arm")
    bridge = RosBridge(skill_action=skill_action, validate_skill_service=validate_service)
    assert bridge.start() is True
    server._bridge = bridge
    try:
        assert bridge.wait_for_skill_server(timeout=3.0) is True

        # validate_skill (service dry-run)
        v = server.validate_skill("recover_safe_pose")
        assert v["allowed"] is True

        # execute_skill (action, with streaming feedback)
        ctx = _RecordingContext()
        result = asyncio.run(server.execute_skill("recover_safe_pose", timeout_sec=10.0, ctx=ctx))

        assert result["success"] is True
        assert result["message"] == "skill complete"
        assert result["executed_primitives"] == ["move_to_named_pose", "open_gripper"]
        # 1 dispatch notification + 3 primitive feedback notifications
        assert len(ctx.logs) == 4
        assert any("dispatching skill" in line for line in ctx.logs)
        assert any("primitive 2/3" in line for line in ctx.logs)

        # snapshot still works and reports the action name
        snap = bridge.snapshot()
        assert snap["skill_action"] == skill_action
    finally:
        bridge.shutdown()
        mock_exec.shutdown()
        mock.destroy_node()
        # sanity: TaskStatus import is unused at runtime but proves msg availability
        assert TaskStatus is not None
