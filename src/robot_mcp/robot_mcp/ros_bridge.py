"""Unified rclpy bridge for the MCP server.

Owns a single ROS 2 node + a MultiThreadedExecutor spinning in a daemon thread.
It combines three concerns behind one rclpy context:

  * status observation (subscribes to TaskStatus / JointState — non-blocking cache)
  * the ValidateSkill service client (safety_guard dry-run)
  * the SkillCommand action client (/embodied/execute_skill — real motion)

Design rules (see the MCP design doc, Phase 1 safety invariants):
  * This bridge only ever talks to ``/embodied/execute_skill`` and
    ``/embodied/validate_skill``. It never touches ``/task_executor/*`` or
    MoveIt directly, so ``safety_guard`` (white-list + workspace bounds) stays
    in the enforcement path. The MCP layer adds no safety logic of its own.
  * ROS is imported lazily so the module imports without a ROS environment;
    ``start()`` returns False and tools degrade gracefully when ROS is absent.

Long-lived operations (skill execution) are started non-blocking and polled
from the async MCP tool, so the executor stays responsive and ROS action
feedback streams back to the agent as MCP notifications.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Defaults mirror EmbodiedConfig (robot_config/config.py) and the
# skill_executor_node defaults; overridden at construction from the active
# robot_config (see Catalog.ros_interfaces).
DEFAULT_STATUS_TOPIC = "/embodied/task_status"
DEFAULT_JOINT_STATE_TOPIC = "/joint_states"
DEFAULT_SKILL_ACTION = "/embodied/execute_skill"
DEFAULT_VALIDATE_SKILL_SERVICE = "/embodied/validate_skill"
_STALE_AFTER_SEC = 5.0


class RosBridge:
    """Thread-safe ROS client: status cache + validate service + skill action."""

    def __init__(
        self,
        status_topic: str = DEFAULT_STATUS_TOPIC,
        joint_state_topic: str = DEFAULT_JOINT_STATE_TOPIC,
        skill_action: str = DEFAULT_SKILL_ACTION,
        validate_skill_service: str = DEFAULT_VALIDATE_SKILL_SERVICE,
    ) -> None:
        self._status_topic = status_topic
        self._joint_state_topic = joint_state_topic
        self._skill_action = skill_action
        self._validate_skill_service = validate_skill_service

        self._lock = threading.Lock()
        self._latest_status: dict | None = None
        self._status_recv_sec: float = 0.0
        self._latest_joints: dict[str, float] | None = None
        self._joints_recv_sec: float = 0.0

        self._ros_available = False
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._skill_client = None
        self._validate_client = None
        # Message/action classes captured at start() so we can build
        # goals/requests later without re-importing in every call.
        self._SkillCommand = None
        self._ValidateSkill = None

    @property
    def ros_available(self) -> bool:
        return self._ros_available

    @property
    def skill_action_name(self) -> str:
        return self._skill_action

    @property
    def validate_service_name(self) -> str:
        return self._validate_skill_service

    def start(self) -> bool:
        """Initialize rclpy, create entities, spin the executor in a daemon thread."""
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState

            from ibrobot_msgs.action import SkillCommand
            from ibrobot_msgs.msg import TaskStatus
            from ibrobot_msgs.srv import ValidateSkill
        except Exception as exc:  # noqa: BLE001 - ROS may be absent/un-sourced
            logger.warning("rclpy/messages unavailable, bridge disabled: %s", exc)
            self._ros_available = False
            return False

        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("robot_mcp_bridge")
        except Exception as exc:  # noqa: BLE001 - init failures must not crash server
            logger.warning("rclpy.init failed, bridge disabled: %s", exc)
            self._ros_available = False
            return False

        self._SkillCommand = SkillCommand
        self._ValidateSkill = ValidateSkill
        cb_group = ReentrantCallbackGroup()

        self._node.create_subscription(TaskStatus, self._status_topic, self._on_status, 10, callback_group=cb_group)
        self._node.create_subscription(JointState, self._joint_state_topic, self._on_joint, 10, callback_group=cb_group)
        self._skill_client = ActionClient(self._node, SkillCommand, self._skill_action, callback_group=cb_group)
        self._validate_client = self._node.create_client(
            ValidateSkill, self._validate_skill_service, callback_group=cb_group
        )

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, name="robot_mcp_bridge_spin", daemon=True)
        self._spin_thread.start()
        self._ros_available = True
        logger.info(
            "bridge ready (status=%s joints=%s skill=%s validate=%s)",
            self._status_topic,
            self._joint_state_topic,
            self._skill_action,
            self._validate_skill_service,
        )
        return True

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001
            logger.warning("bridge executor stopped: %s", exc)

    # ----------------------------- status cache -------------------------------

    def _on_status(self, msg) -> None:
        with self._lock:
            self._latest_status = _task_status_to_dict(msg)
            self._status_recv_sec = time.monotonic()

    def _on_joint(self, msg) -> None:
        joints = {n: float(p) for n, p in zip(msg.name, msg.position, strict=True)} if msg.position else {}
        with self._lock:
            self._latest_joints = joints or None
            self._joints_recv_sec = time.monotonic()

    def snapshot(self) -> dict:
        """Non-blocking, thread-safe snapshot of current status."""
        now = time.monotonic()
        with self._lock:
            task_status = self._latest_status
            status_stale = task_status is not None and (now - self._status_recv_sec) > _STALE_AFTER_SEC
            joints = self._latest_joints
            joints_stale = joints is not None and (now - self._joints_recv_sec) > _STALE_AFTER_SEC
        return {
            "ros_available": self._ros_available,
            "status_topic": self._status_topic,
            "joint_state_topic": self._joint_state_topic,
            "skill_action": self._skill_action,
            "validate_skill_service": self._validate_skill_service,
            "task_status": task_status,
            "task_status_stale": status_stale,
            "joints": joints,
            "joints_stale": joints_stale,
        }

    # ---------------------------- validate (service) --------------------------

    def validate_skill(
        self,
        skill_name: str,
        target_name: str = "",
        place_name: str = "",
        motion_direction: str = "",
        motion_distance: float = 0.0,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Dry-run a skill against safety_guard (ValidateSkill service).

        Blocking but fast (a service round-trip). Returns {allowed, reason}.
        """
        if not self._ros_available:
            return {"allowed": False, "reason": "ROS bridge offline"}
        if self._validate_client is None or self._ValidateSkill is None:
            return {"allowed": False, "reason": "validate client not initialized"}

        if not self._validate_client.service_is_ready() and not self._validate_client.wait_for_service(
            timeout_sec=float(timeout)
        ):
            return {
                "allowed": False,
                "reason": (
                    f"validate service {self._validate_skill_service} not available "
                    "(is safety_guard / skill_executor running?)"
                ),
            }

        req = self._ValidateSkill.Request()
        req.skill_name = skill_name
        req.target_name = target_name or ""
        req.place_name = place_name or ""
        req.motion_direction = motion_direction or ""
        req.motion_distance = float(motion_distance or 0.0)

        future = self._validate_client.call_async(req)
        if not self._wait_future(future, timeout):
            return {"allowed": False, "reason": "validate service call timed out"}
        resp = future.result()
        return {"allowed": bool(resp.allowed), "reason": resp.reason}

    # ---------------------------- execute (action) ----------------------------

    def wait_for_skill_server(self, timeout: float = 2.0) -> bool:
        """Return True if the SkillCommand action server is reachable."""
        if not self._ros_available or self._skill_client is None:
            return False
        return self._skill_client.wait_for_server(timeout_sec=float(timeout))

    def build_skill_goal(
        self,
        skill_name: str,
        target_name: str = "",
        place_name: str = "",
        motion_direction: str = "",
        motion_distance: float = 0.0,
        timeout_sec: float = 15.0,
        task_id: str | None = None,
    ):
        """Construct a SkillCommand.Goal from plain parameters."""
        goal = self._SkillCommand.Goal()
        goal.task_id = task_id or str(uuid.uuid4())
        goal.skill_name = skill_name
        goal.target_name = target_name or ""
        goal.place_name = place_name or ""
        goal.motion_direction = motion_direction or ""
        goal.motion_distance = float(motion_distance or 0.0)
        goal.timeout_sec = float(timeout_sec)
        return goal

    def send_skill_goal(self, goal):
        """Send a skill goal non-blocking.

        Returns ``(send_future, feedback_queue)``. The MCP tool polls
        ``send_future`` for acceptance, then ``get_result_async()`` for
        completion, draining ``feedback_queue`` (items: (state, detail)) in
        between to stream progress.
        """
        if self._skill_client is None:
            raise RuntimeError("skill action client not initialized")

        fb_queue: queue.Queue = queue.Queue()

        def _on_feedback(feedback) -> None:
            # rclpy client feedback callback receives a single argument. Some
            # versions wrap the message (exposing .feedback); others pass it
            # raw. Handle both so state/detail are always readable.
            msg = getattr(feedback, "feedback", feedback)
            fb_queue.put((getattr(msg, "state", ""), getattr(msg, "detail", "")))

        send_future = self._skill_client.send_goal_async(goal, feedback_callback=_on_feedback)
        return send_future, fb_queue

    # --------------------------------- helpers --------------------------------

    def _wait_future(self, future, timeout: float) -> bool:
        """Busy-wait for an rclpy future (executor spins in the bg thread)."""
        end = time.monotonic() + float(timeout)
        while not future.done() and time.monotonic() < end:
            time.sleep(0.02)
        return future.done()

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            if self._executor is not None:
                self._executor.shutdown()
        with contextlib.suppress(Exception):
            if self._skill_client is not None:
                self._skill_client.destroy()
        with contextlib.suppress(Exception):
            if self._node is not None:
                self._node.destroy_node()
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass


def _task_status_to_dict(msg) -> dict:
    """Flatten a TaskStatus message into a plain dict for the agent."""
    return {
        "task_id": msg.task_id,
        "state": msg.state,
        "success": bool(msg.success),
        "current_skill": msg.current_skill,
        "completed_skills": list(msg.completed_skills),
        "error_code": msg.error_code,
        "message": msg.message,
        "recoverable": bool(msg.recoverable),
        "replan_requested": bool(msg.replan_requested),
    }
