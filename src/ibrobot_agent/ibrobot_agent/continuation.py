"""Resolve "继续" against the Agent and Gateway ledgers (ROS-free).

A continuation is never a Gateway resume: the Agent re-plans the interrupted
request's original text from the current robot state and submits a fresh
Program under the new request id.  The ledgers only decide whether that is
allowed and tell the planner how much already ran:

* Agent ledger: the session's latest actionable request, its terminal state and
  the original text from its bound proposal.
* Gateway ledger: whether that request's Program really reached CANCELED, and
  how many canonical steps completed before the stop.

Only a confirmed cancellation (or a stop before anything was submitted) is
continued; every unknown, failed or inconsistent record is refused without
motion.  ``completed_step_count`` is a lower bound: the Gateway does not record
progress inside a step, so the interrupted step is planned again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_common.text_normalization import normalize_trigger_text
from robot_skill_cli.interactive_control import CONTINUE_PHRASES

CONTINUE_REFUSED = "CONTINUE_REFUSED"
_CONTINUE_WORDS = frozenset(normalize_trigger_text(phrase) for phrase in CONTINUE_PHRASES)
# Admission rejections never ran anything and are not the task the user means.
_ADMISSION_REJECTIONS = frozenset({"BUSY", "ROBOT_QUARANTINED"})


@dataclass(frozen=True)
class ContinuationDecision:
    """``text`` set: re-plan it with ``hint``; otherwise answer ``message`` without motion."""

    message: str
    text: str | None = None
    error_code: str = ""
    hint: Mapping[str, Any] = field(default_factory=dict)


def is_continue_request(text: str) -> bool:
    return normalize_trigger_text(str(text)) in _CONTINUE_WORDS


def select_anchor(records: Sequence[Any]) -> Any | None:
    """Return the newest request (records are newest first) that "继续" can refer to."""
    for record in records:
        if record.state == "ANSWERED":
            # Conversations, clarifications and earlier refused continuations.
            continue
        terminal = record.terminal
        if record.task_ref is None and terminal is not None and terminal.error_code in _ADMISSION_REJECTIONS:
            continue
        return record
    return None


def needs_gateway(anchor: Any | None) -> bool:
    return anchor is not None and anchor.state == "CANCELLED"


def decide_continuation(
    anchor: Any | None,
    *,
    gateway: Mapping[str, Any] | None = None,
    gateway_error: str = "",
    previous_hint: Mapping[str, Any] | None = None,
) -> ContinuationDecision:
    """Decide from the anchor record and, for cancellations, its Gateway execution.

    ``previous_hint`` is the continuation hint the anchor itself was planned
    with, so steps finished before an earlier stop stay counted across a chain
    of stop/continue.
    """
    if anchor is None:
        return ContinuationDecision("没有可以继续的任务，请直接说出要执行的指令。")
    state = anchor.state
    if state == "SUCCEEDED":
        return ContinuationDecision("上一个任务已经完成，没有需要继续的动作。")
    if state == "FAILED":
        return _refuse("上一个任务执行失败，不能自动继续，请重新下达指令。")
    if state == "UNKNOWN":
        return _refuse("上一个任务的执行结果未知，请人工检查机器人状态后重新下达指令。")
    if state == "CANCELLED_BEFORE_EXECUTION":
        return _refuse("上一条指令在执行前就取消了，请再说一遍完整指令。")
    if state != "CANCELLED":
        return _refuse("上一个任务还没有结束，不能继续。")

    proposal = _proposal(anchor)
    original_text = _original_text(proposal, previous_hint)
    if not original_text:
        return _refuse("上一条指令没有可继续的计划，请再说一遍完整指令。")
    if gateway_error or gateway is None:
        return _refuse("无法确认上一个任务已经安全停止，不能继续，请重新下达指令。")

    terminal_detail = dict(anchor.terminal.detail) if anchor.terminal is not None else {}
    agent_submitted = terminal_detail.get("submitted") is True
    planned_steps = _proposal_steps(proposal)
    if not gateway.get("found"):
        if agent_submitted:
            # The Agent recorded a submitted Program the Gateway cannot find.
            return _refuse("两个账本记录不一致，无法确认上一个任务已经安全停止，不能继续。")
        completed: list[Any] = []
        steps = planned_steps
    else:
        if str(gateway.get("state", "")) != "CANCELED":
            return _refuse("无法确认上一个任务已经安全停止，不能继续，请重新下达指令。")
        recorded_execution = str(terminal_detail.get("execution_id", ""))
        if recorded_execution and recorded_execution != str(gateway.get("execution_id", "")):
            return _refuse("两个账本记录不一致，无法确认上一个任务已经安全停止，不能继续。")
        steps = list(gateway.get("canonical_steps") or planned_steps)
        count = int(gateway.get("completed_step_count", 0) or 0)
        if count < 0 or count > len(steps):
            return _refuse("两个账本记录不一致，无法确认上一个任务已经安全停止，不能继续。")
        completed = steps[:count]

    earlier = list((previous_hint or {}).get("completed_steps", ()))
    completed_total = earlier + [_step_summary(step) for step in completed]
    remaining = steps[len(completed) :]
    total = len(earlier) + len(steps)
    if not remaining:
        return ContinuationDecision("上一个任务的步骤已经全部完成，没有需要继续的动作。")
    hint = {
        "original_text": original_text,
        "continues_from_request_id": anchor.key.request_id,
        "continues_from_execution_id": str((gateway or {}).get("execution_id", "")),
        "completed_step_count": len(completed_total),
        "step_count": total,
        "completed_steps": completed_total,
        "interrupted_step": _step_summary(remaining[0]),
        "instruction": (
            "The user asked to continue an interrupted task. completed_steps finished fully; do not repeat "
            "them. interrupted_step was stopped and may be partially done, so plan it again. Plan only the "
            "remaining part of original_text from the current robot state."
        ),
    }
    if completed_total:
        message = f"上次任务完整完成了 {len(completed_total)}/{total} 步后停止，现在继续剩余部分。"
    else:
        message = "上次任务还没有完成任何步骤，现在重新执行。"
    return ContinuationDecision(message, text=original_text, hint=hint)


def _refuse(message: str) -> ContinuationDecision:
    return ContinuationDecision(message, error_code=CONTINUE_REFUSED)


def _proposal(anchor: Any) -> Mapping[str, Any]:
    try:
        proposal = json.loads(anchor.proposal_json) if anchor.proposal_json else {}
    except (TypeError, ValueError):
        return {}
    return proposal if isinstance(proposal, Mapping) else {}


def _original_text(proposal: Mapping[str, Any], previous_hint: Mapping[str, Any] | None) -> str:
    if previous_hint and previous_hint.get("original_text"):
        return str(previous_hint["original_text"])
    request = proposal.get("request")
    text = request.get("text") if isinstance(request, Mapping) else None
    return text.strip() if isinstance(text, str) else ""


def _proposal_steps(proposal: Mapping[str, Any]) -> list[Any]:
    outcome = proposal.get("outcome")
    steps = outcome.get("steps") if isinstance(outcome, Mapping) else None
    return list(steps) if isinstance(steps, list) else []


def _step_summary(step: Any) -> dict[str, Any]:
    """Keep the identifying, non-default fields so the hint stays short."""
    if not isinstance(step, Mapping):
        return {"skill_name": str(step)}
    return {
        key: value for key, value in step.items() if key != "schema_version" and value not in ("", 0, 0.0, None, [], {})
    }


__all__ = [
    "CONTINUE_REFUSED",
    "ContinuationDecision",
    "decide_continuation",
    "is_continue_request",
    "needs_gateway",
    "select_anchor",
]
