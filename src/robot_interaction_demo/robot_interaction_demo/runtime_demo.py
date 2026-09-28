"""Execute parameterized speech and named-motion requests on injected endpoints."""

from __future__ import annotations

import math
import threading
import time
import uuid


def _wait(node, future, deadline: float, phase: str, *, externally_spun: bool):
    import rclpy

    remaining = max(0.0, deadline - time.monotonic())
    if externally_spun:
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        ready.wait(timeout=remaining)
    else:
        rclpy.spin_until_future_complete(node, future, timeout_sec=remaining)
    if not future.done():
        raise TimeoutError(
            f"Timeout during {phase}; an issued request may still execute. "
            "No retry or cancellation was sent; inspect the robot before another request."
        )
    return future.result()


def execute_speech(
    node,
    endpoint: str,
    *,
    text: str,
    language: str,
    priority: int,
    interrupt: bool,
    timeout_sec: float,
    externally_spun: bool = False,
) -> dict:
    """Request native robot speech; success means accepted, not playback complete."""
    from rclpy.callback_groups import ReentrantCallbackGroup

    from ibrobot_msgs.srv import SpeakText

    if not isinstance(text, str) or not text.strip():
        raise ValueError("speech text must be non-empty")
    if not isinstance(language, str):
        raise ValueError("speech language must be a string")
    if type(priority) is not int or not 0 <= priority <= 100:
        raise ValueError("speech priority must be an integer in 0..100")
    if type(interrupt) is not bool:
        raise ValueError("speech interrupt must be a boolean")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive and finite")

    deadline = time.monotonic() + timeout_sec
    client = node.create_client(SpeakText, endpoint, callback_group=ReentrantCallbackGroup())
    try:
        remaining = max(0.0, deadline - time.monotonic())
        if not client.wait_for_service(timeout_sec=remaining):
            raise TimeoutError("Speech service unavailable; no speech request sent")
        request = SpeakText.Request(
            text=text,
            language=language,
            priority=priority,
            interrupt=interrupt,
            trace_id=f"interaction-demo-{uuid.uuid4().hex}",
        )
        response = _wait(
            node,
            client.call_async(request),
            deadline,
            "speech acceptance",
            externally_spun=externally_spun,
        )
        return {
            "success": response.success,
            "phase": "accepted" if response.success else "rejected",
            "error_code": response.error_code,
            "message": response.message,
            "utterance_id": response.utterance_id,
            "note": "Acceptance does not confirm audible playback or speech completion.",
        }
    finally:
        node.destroy_client(client)


def execute_named_motion(
    node,
    endpoint: str,
    *,
    name: str,
    target: str,
    interrupt: bool,
    allow_motion: bool,
    timeout_sec: float,
    externally_spun: bool = False,
) -> dict:
    """Execute one runtime-advertised semantic motion and wait for its terminal result."""
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup

    from ibrobot_msgs.action import ExecuteNamedMotion

    if not allow_motion:
        raise ValueError("Motion requires allow_motion=true for this request")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("motion name must be non-empty")
    if target not in ("", "left", "right", "both"):
        raise ValueError("motion target must be empty, left, right or both")
    if type(interrupt) is not bool:
        raise ValueError("motion interrupt must be a boolean")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive and finite")

    deadline = time.monotonic() + timeout_sec
    client = ActionClient(node, ExecuteNamedMotion, endpoint, callback_group=ReentrantCallbackGroup())
    try:
        remaining = max(0.0, deadline - time.monotonic())
        if not client.wait_for_server(timeout_sec=remaining):
            raise TimeoutError("Named motion server unavailable; no motion goal sent")
        goal = ExecuteNamedMotion.Goal(name=name, target=target, interrupt=interrupt)
        handle = _wait(
            node,
            client.send_goal_async(goal),
            deadline,
            "motion goal acceptance",
            externally_spun=externally_spun,
        )
        if not handle.accepted:
            return {"success": False, "phase": "rejected", "message": "Runtime rejected the motion goal"}
        answer = _wait(
            node,
            handle.get_result_async(),
            deadline,
            "motion result",
            externally_spun=externally_spun,
        )
        result = answer.result
        return {
            "success": answer.status == GoalStatus.STATUS_SUCCEEDED and result.success,
            "phase": "terminal",
            "action_status": answer.status,
            "error_code": result.error_code,
            "message": result.message,
        }
    finally:
        client.destroy()
