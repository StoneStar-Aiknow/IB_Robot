"""Helpers for carrying timeout context across the embodied task pipeline."""

from __future__ import annotations

import json
import time
from typing import Any

TIMEOUT_CONTEXT_KEY = "timeout_context"


def load_task_context(raw_context_json: str) -> dict[str, Any]:
    if not raw_context_json:
        return {}
    loaded = json.loads(raw_context_json)
    if not isinstance(loaded, dict):
        raise ValueError("task context_json must decode to a JSON object")
    return loaded


def dump_task_context(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False)


def build_timeout_context(task_timeout_sec: float, now_unix_sec: float | None = None) -> dict[str, float]:
    timeout_sec = float(task_timeout_sec)
    if timeout_sec <= 0.0:
        return {}
    current_time = float(now_unix_sec if now_unix_sec is not None else time.time())
    return {
        "task_timeout_sec": timeout_sec,
        "created_at_unix_sec": current_time,
        "deadline_unix_sec": current_time + timeout_sec,
    }


def ensure_timeout_context(
    raw_context_json: str,
    task_timeout_sec: float,
    now_unix_sec: float | None = None,
) -> dict[str, Any]:
    context = load_task_context(raw_context_json)
    if TIMEOUT_CONTEXT_KEY not in context:
        timeout_context = build_timeout_context(task_timeout_sec, now_unix_sec=now_unix_sec)
        if timeout_context:
            context[TIMEOUT_CONTEXT_KEY] = timeout_context
    return context


def remaining_task_budget_sec(
    context_or_json: dict[str, Any] | str,
    now_unix_sec: float | None = None,
) -> float | None:
    context = context_or_json if isinstance(context_or_json, dict) else load_task_context(context_or_json)
    timeout_context = context.get(TIMEOUT_CONTEXT_KEY, {})
    if not isinstance(timeout_context, dict):
        return None
    deadline_unix_sec = timeout_context.get("deadline_unix_sec")
    if deadline_unix_sec is None:
        return None
    current_time = float(now_unix_sec if now_unix_sec is not None else time.time())
    return float(deadline_unix_sec) - current_time
