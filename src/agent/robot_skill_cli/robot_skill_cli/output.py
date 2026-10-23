"""Stable JSON and JSONL output contracts for robot-skill."""

from __future__ import annotations

import json
from typing import Any

EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_GATEWAY_REJECTED = 3
EXIT_ROS_UNAVAILABLE = 4
EXIT_TIMEOUT = 124
EXIT_SIGINT = 130
EXIT_SIGTERM = 143


def json_dumps(value: Any) -> str:
    """Serialize one stable, compact, ASCII-only JSON document."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def success_envelope(command: str, data: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "command": command,
        "data": data,
        "error": None,
    }


def error_envelope(command: str, code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "command": command,
        "data": None,
        "error": {"code": code, "message": message},
    }


def feedback_event(task_id: str, payload_hash: str, state: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "feedback",
        "task_id": task_id,
        "payload_hash": payload_hash,
        "data": {"state": state, "detail": detail},
    }


def result_event(
    task_id: str,
    payload_hash: str,
    *,
    success: bool,
    error_code: str,
    message: str,
    executed_step_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "result",
        "task_id": task_id,
        "payload_hash": payload_hash,
        "data": {
            "success": success,
            "error_code": error_code,
            "message": message,
            "executed_step_count": executed_step_count,
        },
    }
