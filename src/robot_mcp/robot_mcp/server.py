"""FastMCP server exposing IB-Robot tools.

Phase 0 (read-only): ``list_skills`` / ``list_poses`` / ``get_status``.
Phase 1 (guarded commanding):
    * ``validate_skill`` - safety_guard dry-run (no motion)
    * ``execute_skill``  - run a real skill via /embodied/execute_skill

Safety invariants (see MCP design doc):
    * ``execute_skill`` ONLY calls ``/embodied/execute_skill``, which routes
      through ``skill_executor`` -> ``safety_guard`` (white-list + workspace
      bounds). The MCP layer adds no safety logic of its own.
    * Nothing exposes ``/task_executor/*`` or MoveIt directly.

opencode / Hermes namespace tools by the server name, e.g. server ``robot``
yields ``robot_list_skills``, ``robot_execute_skill`` ...

Usage:
    robot_mcp_server --config-name so101_single_arm                 # stdio (default)
    robot_mcp_server --transport streamable-http --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from robot_mcp.catalog import Catalog, build_catalog, valid_motion_directions
from robot_mcp.ros_bridge import RosBridge

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "IB-Robot MCP layer. list_skills/list_poses/get_status are read-only. "
    "validate_skill dry-runs a skill against safety_guard; execute_skill "
    "commands real motion through skill_executor (safety-guarded). Prefer "
    "validate_skill before execute_skill for unfamiliar skills."
)

# Set in main() before serving; tools treat None as "not ready".
_catalog: Catalog | None = None
_bridge: RosBridge | None = None

# Grace beyond the action's own timeout before the MCP side gives up.
_RESULT_GRACE_SEC = 5.0
_ACCEPT_TIMEOUT_SEC = 5.0
_CANCEL_DRAIN_TIMEOUT_SEC = 5.0
DEFAULT_HTTP_HOST = "127.0.0.1"


# =============================== Phase 0 tools ==============================


def list_skills() -> list[dict[str, Any]]:
    """List every skill the robot supports, with its primitive decomposition.

    Derived from the single source of truth (robot_config YAML). Each entry:
    name, primitives[], pose_targets[], accepts_motion, vision_only, doc.
    """
    if _catalog is None:  # pragma: no cover
        raise RuntimeError("Catalog not initialized")
    return list(_catalog.skills)


def list_poses() -> list[dict[str, Any]]:
    """List named end-effector poses (home/observe_table/zero + any custom).

    Each entry: {name, position:{x,y,z}, orientation:{x,y,z,w}} (quaternion).
    """
    if _catalog is None:  # pragma: no cover
        raise RuntimeError("Catalog not initialized")
    return list(_catalog.poses)


def get_status() -> dict[str, Any]:
    """Read the robot's current state (non-blocking snapshot).

    Returns ros_available, latest TaskStatus, joint positions, and the ROS
    interface names the bridge targets. Stale flags mark data older than a few
    seconds. If ROS is unavailable, ros_available is false but the catalog
    tools remain usable.
    """
    if _catalog is None:  # pragma: no cover
        raise RuntimeError("Catalog not initialized")
    snapshot: dict[str, Any] = {
        "robot": _catalog.to_summary(),
        "motion_directions": valid_motion_directions(),
        "workspace": _catalog.workspace,
        "ros_interfaces": _catalog.ros_interfaces,
    }
    if _bridge is None or not _bridge.ros_available:
        snapshot["ros_available"] = False
        snapshot["note"] = (
            "ROS not available (not sourced / no daemon running). "
            "list_skills/list_poses remain usable; validate_skill/execute_skill are offline."
        )
        return snapshot
    snapshot.update(_bridge.snapshot())
    return snapshot


# =============================== Phase 1 tools ==============================


def _catalog_has_skill(skill_name: str) -> bool:
    if _catalog is None:  # pragma: no cover
        raise RuntimeError("Catalog not initialized")
    return any(skill.get("name") == skill_name for skill in _catalog.skills)


def validate_skill(
    skill_name: str,
    target_name: str = "",
    place_name: str = "",
    motion_direction: str = "",
    motion_distance: float = 0.0,
) -> dict[str, Any]:
    """Dry-run a skill against safety_guard WITHOUT moving the robot.

    Use this to check whether a skill is allowed before calling execute_skill.
    Returns {allowed, reason}. Reason explains the rejection (unknown skill,
    workspace bound, missing target, etc.) straight from safety_guard.
    """
    if not _catalog_has_skill(skill_name):
        return {"allowed": False, "reason": f"unsupported skill: {skill_name}"}
    if _bridge is None or not _bridge.ros_available:
        return {"allowed": False, "reason": "ROS bridge offline (validate_skill unavailable)"}
    return _bridge.validate_skill(
        skill_name=skill_name,
        target_name=target_name,
        place_name=place_name,
        motion_direction=motion_direction,
        motion_distance=motion_distance,
    )


async def execute_skill(
    skill_name: str,
    target_name: str = "",
    place_name: str = "",
    motion_direction: str = "",
    motion_distance: float = 0.0,
    timeout_sec: float = 0.0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute a real robot skill via /embodied/execute_skill (moves the robot).

    SAFETY: this goes through skill_executor, which validates the skill and
    every primitive with safety_guard (white-list + workspace bounds). It never
    bypasses those checks. Prefer validate_skill first for unfamiliar skills.

    Feedback streams back as progress notifications. Returns
    {success, error_code, message, executed_primitives, task_id}.

    Args:
        skill_name: one of the skills from list_skills.
        target_name/place_name: object/place reference (skill-dependent).
        motion_direction: forward/backward/left/right/up/down (move_relative_ee).
        motion_distance: meters (for move_relative_ee).
        timeout_sec: action-side timeout for the whole skill. Zero uses the
            skill timeout declared in robot_config, then falls back to 15s.
    """
    if _catalog is None:  # pragma: no cover
        raise RuntimeError("Catalog not initialized")
    if not _catalog_has_skill(skill_name):
        return {
            "success": False,
            "error_code": "UNSUPPORTED_SKILL",
            "message": f"unsupported skill: {skill_name}",
        }
    if _bridge is None or not _bridge.ros_available:
        return {
            "success": False,
            "error_code": "BRIDGE_OFFLINE",
            "message": "ROS bridge offline; execute_skill unavailable.",
        }
    if not _bridge.wait_for_skill_server(timeout=2.0):
        return {
            "success": False,
            "error_code": "SERVER_UNAVAILABLE",
            "message": (f"{_bridge.skill_action_name} action server not available (is skill_executor running?)"),
        }

    skill_entry = next((skill for skill in _catalog.skills if skill["name"] == skill_name), None)
    effective_timeout = float(timeout_sec)
    if effective_timeout <= 0.0:
        effective_timeout = float((skill_entry or {}).get("timeout_sec", 0.0) or 15.0)

    admission_state = _bridge.try_claim_skill_execution()
    if admission_state is not None:
        recovering = admission_state == "recovering"
        return {
            "success": False,
            "error_code": "MOTION_RECOVERY_IN_PROGRESS" if recovering else "SKILL_IN_PROGRESS",
            "message": (
                "a timed-out skill is still reaching a terminal state."
                if recovering
                else "another MCP skill is already executing."
            ),
        }

    task_id = str(uuid.uuid4())
    try:
        if ctx is not None:
            await ctx.info(f"[{task_id}] dispatching skill '{skill_name}' to skill_executor")
        goal = _bridge.build_skill_goal(
            skill_name=skill_name,
            target_name=target_name,
            place_name=place_name,
            motion_direction=motion_direction,
            motion_distance=motion_distance,
            timeout_sec=effective_timeout,
            task_id=task_id,
        )
        send_future, fb_queue = _bridge.send_skill_goal(goal)
    except Exception as exc:  # noqa: BLE001 - transport failure must release admission
        _bridge.release_skill_execution()
        logger.warning("[%s] failed to dispatch skill %s: %s", task_id, skill_name, exc)
        return {
            "success": False,
            "error_code": "DISPATCH_FAILED",
            "message": "failed to send SkillCommand goal.",
            "task_id": task_id,
        }

    # Wait for goal acceptance.
    accept_deadline = time.monotonic() + _ACCEPT_TIMEOUT_SEC
    while not send_future.done() and time.monotonic() < accept_deadline:
        await asyncio.sleep(0.05)
    if not send_future.done():
        _bridge.mark_skill_recovering()
        _bridge.cancel_late_skill_goal(send_future, _CANCEL_DRAIN_TIMEOUT_SEC)
        return {
            "success": False,
            "error_code": "ACCEPT_TIMEOUT",
            "message": "skill_executor did not accept the goal in time; motion admission remains blocked.",
            "task_id": task_id,
        }

    try:
        goal_handle = send_future.result()
    except Exception as exc:  # noqa: BLE001 - a failed send cannot leave an active goal
        _bridge.release_skill_execution()
        logger.warning("[%s] SkillCommand acceptance failed: %s", task_id, exc)
        return {
            "success": False,
            "error_code": "DISPATCH_FAILED",
            "message": "SkillCommand acceptance failed.",
            "task_id": task_id,
        }
    if not goal_handle.accepted:
        _bridge.release_skill_execution()
        return {
            "success": False,
            "error_code": "REJECTED",
            "message": "skill_executor rejected the goal.",
            "task_id": task_id,
        }

    # Wait for completion, streaming feedback as MCP progress notifications.
    try:
        result_future = goal_handle.get_result_async()
    except Exception as exc:  # noqa: BLE001 - no result future means no tracked motion
        _bridge.release_skill_execution()
        logger.warning("[%s] failed to track SkillCommand result: %s", task_id, exc)
        return {
            "success": False,
            "error_code": "DISPATCH_FAILED",
            "message": "failed to track SkillCommand result.",
            "task_id": task_id,
        }
    overall_deadline = time.monotonic() + effective_timeout + _RESULT_GRACE_SEC
    while not result_future.done() and time.monotonic() < overall_deadline:
        drained = False
        while not fb_queue.empty():
            state, detail = fb_queue.get_nowait()
            drained = True
            if ctx is not None:
                await ctx.info(f"[{task_id}] {state}: {detail}")
        if not drained:
            await asyncio.sleep(0.05)

    if not result_future.done():
        _bridge.mark_skill_recovering()
        cleaned = await asyncio.to_thread(
            _bridge.cancel_and_drain_skill_goal,
            goal_handle,
            result_future,
            _CANCEL_DRAIN_TIMEOUT_SEC,
        )
        if cleaned:
            _bridge.release_skill_execution()
            return {
                "success": False,
                "error_code": "RESULT_TIMEOUT",
                "message": (
                    f"skill did not complete within {effective_timeout + _RESULT_GRACE_SEC:.1f}s; "
                    "goal canceled and drained."
                ),
                "task_id": task_id,
            }
        _bridge.release_skill_execution_when_terminal(result_future)
        return {
            "success": False,
            "error_code": "CANCEL_CLEANUP_TIMEOUT",
            "message": (
                f"skill did not complete within {effective_timeout + _RESULT_GRACE_SEC:.1f}s; "
                "motion admission remains blocked until cancellation is terminal."
            ),
            "task_id": task_id,
        }

    try:
        res = result_future.result().result
        # Flush any trailing feedback.
        if ctx is not None:
            while not fb_queue.empty():
                state, detail = fb_queue.get_nowait()
                await ctx.info(f"[{task_id}] {state}: {detail}")

        return {
            "success": bool(res.success),
            "error_code": res.error_code,
            "message": res.message,
            "executed_primitives": list(res.executed_primitives),
            "task_id": task_id,
        }
    finally:
        _bridge.release_skill_execution()


_ALL_TOOLS = (list_skills, list_poses, get_status, validate_skill, execute_skill)


def _build_server(transport: str, host: str, port: int) -> FastMCP:
    if transport == "streamable-http":
        server = FastMCP("robot", instructions=_INSTRUCTIONS, host=host, port=port)
    else:
        server = FastMCP("robot", instructions=_INSTRUCTIONS)
    for tool_fn in _ALL_TOOLS:
        server.add_tool(tool_fn)
    return server


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot_mcp_server", description=__doc__)
    parser.add_argument("--config-name", default=None, help="Robot config name (without .yaml).")
    parser.add_argument("--config-path", default=None, help="Full path to a robot_config YAML.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HTTP_HOST,
        help="HTTP host (http transport only; defaults to loopback).",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (http transport only).")
    parser.add_argument("--no-ros", action="store_true", help="Skip rclpy init (catalog-only mode).")
    return parser


def main(argv: list[str] | None = None) -> int:
    global _catalog, _bridge

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_arg_parser().parse_args(argv)

    _catalog = build_catalog(config_name=args.config_name, config_path=args.config_path)
    logger.info("catalog ready: %s", _catalog.to_summary())

    iface = _catalog.ros_interfaces
    _bridge = RosBridge(
        status_topic=iface.get("status_topic", "/embodied/task_status"),
        joint_state_topic=iface.get("joint_state_topic", "/joint_states"),
        skill_action=iface.get("skill_action", "/embodied/execute_skill"),
        validate_skill_service=iface.get("validate_skill_service", "/embodied/validate_skill"),
    )
    if not args.no_ros:
        _bridge.start()

    server = _build_server(args.transport, args.host, args.port)
    try:
        if args.transport == "streamable-http":
            logger.info("serving streamable-http on %s:%d/mcp", args.host, args.port)
        server.run(transport=args.transport)
    finally:
        if _bridge is not None:
            _bridge.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
