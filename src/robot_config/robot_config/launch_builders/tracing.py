"""Tracing launch builder helpers.

This module owns the LTTng session lifecycle used by ``robot.launch.py`` so the
main launch file can stay focused on orchestration.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnShutdown

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.tracing")

DEFAULT_TRACE_SESSION_NAME = "ib_robot_trace"


def _run_trace_command(command: list[str], failure_reason: str) -> None:
    """Run an LTTng CLI command and raise with captured output on failure."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{failure_reason}: lttng command not found ({exc})") from exc

    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout or "").strip()
    if not detail:
        detail = f"exit code {result.returncode}"
    raise RuntimeError(f"{failure_reason}: {detail}")


def _trace_session_exists(session_name: str) -> bool:
    """Return whether an LTTng session with the given name already exists."""
    try:
        result = subprocess.run(
            ["lttng", "list", session_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"failed to inspect tracing session '{session_name}': lttng command not found ({exc})"
        ) from exc

    if result.returncode == 0:
        return True

    detail = (result.stderr or result.stdout or "").strip().lower()
    if "not found" in detail or "no session" in detail:
        return False

    raise RuntimeError(f"failed to inspect tracing session '{session_name}': {detail}")


def _resolve_trace_session(session_name: str, trace_root: Path) -> tuple[str, Path]:
    """Resolve a non-destructive session name and output directory."""
    trace_dir = trace_root / session_name
    if not _trace_session_exists(session_name) and not trace_dir.exists():
        return session_name, trace_dir

    if session_name != DEFAULT_TRACE_SESSION_NAME:
        raise RuntimeError(
            f"tracing session '{session_name}' already exists or output directory already exists at "
            f"{trace_dir}; stop the existing session or choose a different trace_session_name"
        )

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    index = 0
    while True:
        candidate_suffix = f"{suffix}_{index}" if index else suffix
        candidate_name = f"{session_name}_{candidate_suffix}"
        candidate_dir = trace_root / candidate_name
        if not _trace_session_exists(candidate_name) and not candidate_dir.exists():
            return candidate_name, candidate_dir
        index += 1


def _start_trace_session(session_name: str, trace_dir: Path) -> None:
    """Create and start a mixed ROS UST + Python logging LTTng session."""
    trace_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_trace_command(
        ["lttng", "create", session_name, "--output", str(trace_dir)],
        "failed to create tracing session",
    )
    _run_trace_command(
        ["lttng", "enable-event", "--session", session_name, "--userspace", "ros2:*"],
        "failed to enable ROS 2 UST tracepoints",
    )
    _run_trace_command(
        ["lttng", "enable-event", "--session", session_name, "--python", "ib_trace.*"],
        "failed to enable Python tracing domain",
    )
    _run_trace_command(
        ["lttng", "start", session_name],
        "failed to start tracing session",
    )
    logger.info(f"[tracing] Writing tracing session to: {trace_dir}")


def _make_trace_shutdown_handler(session_name: str):
    """Return an OnShutdown callback that stops and destroys the session."""

    def _shutdown_trace(_event, _context):
        for command, action in (
            (["lttng", "stop", session_name], "stop"),
            (["lttng", "destroy", session_name], "destroy"),
        ):
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                if detail and "Session name not found" not in detail:
                    logger.warning(f"[tracing] Failed to {action} session '{session_name}': {detail}")
        return []

    return _shutdown_trace


def generate_tracing_actions(
    enable_tracing: bool,
    requested_session_name: str,
    trace_root: Path | None = None,
) -> list[RegisterEventHandler]:
    """Start the tracing session when requested and return launch shutdown actions."""
    if not enable_tracing:
        return []

    trace_root = trace_root or (Path.home() / ".ros" / "tracing")
    trace_session, trace_dir = _resolve_trace_session(requested_session_name, trace_root)
    if trace_session != requested_session_name:
        logger.warning(
            f"[tracing] Session '{requested_session_name}' already exists; using unique session "
            f"'{trace_session}' instead"
        )
    logger.info(f"[tracing] Enabling ros2_tracing session: {trace_session}")
    _start_trace_session(trace_session, trace_dir)
    return [RegisterEventHandler(event_handler=OnShutdown(on_shutdown=_make_trace_shutdown_handler(trace_session)))]
