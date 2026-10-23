"""Deterministic replies from a managed, identity-checked workflow result."""

from __future__ import annotations

import json

from .hermes_response import HermesResponse, load_json, parse_response
from .hermes_robot_runner import terminal_state, workflow_arguments

# Presentation text only. Motion parameters and terminal classification remain
# owned by the skill catalog and shared Agent terminal contract.
_SUCCESS_SPEECH = {
    "open_gripper_skill": "夹爪已经打开了。",
    "close_gripper_skill": "夹爪已经关闭了。",
    "recover_safe_pose": "机械臂已经回到安全位置了。",
    "nod_yes": "点头动作已完成。",
    "shake_no": "摇头动作已完成。",
    "wave_hello": "挥手动作已完成。",
}


def completion_requested(content: str, tool_calls: list[dict], executable: str, environment: dict) -> bool:
    """Only an explicit response policy and one managed workflow can finish a turn."""
    try:
        response = parse_response(content)
        if response.completion != "after_workflow" or len(tool_calls) != 1:
            return False
        function = tool_calls[0]["function"]
        if function["name"] != "terminal":
            return False
        args = function["arguments"]
        if isinstance(args, str):
            args = load_json(args)
        if not isinstance(args, dict) or args.get("background") or args.get("pty"):
            return False
        if not isinstance(args.get("command"), str):
            return False
        workflow = workflow_arguments(args["command"], executable, environment)
        if workflow is None:
            return False
        steps = load_json(workflow[workflow.index("--workflow-json") + 1])
        return (
            isinstance(steps, list)
            and len(steps) == 1
            and isinstance(steps[0], dict)
            and steps[0].get("skill_name") in _SUCCESS_SPEECH
        )
    except (KeyError, TypeError, ValueError):
        return False


def workflow_response(presentation: dict, terminal: dict) -> str | None:
    """Do not trust free-form tool stdout or a model's predicted success wording."""
    try:
        if not isinstance(presentation, dict) or not isinstance(terminal, dict):
            return None
        steps = presentation["steps"]
        if (
            not isinstance(steps, list)
            or len(steps) != 1
            or not isinstance(steps[0], dict)
            or steps[0]["skill_name"] not in _SUCCESS_SPEECH
            or not isinstance(presentation.get("registry_identity"), dict)
        ):
            return None
        result = terminal["result"]
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            return None
        state = terminal_state(presentation, terminal)
        if state == "succeeded":
            speech = _SUCCESS_SPEECH[steps[0]["skill_name"]]
        elif state == "stopped":
            speech = "任务已取消，机器人已确认停止。"
        elif state == "not_submitted":
            speech = "这次动作在执行前已取消。"
        elif state == "failed":
            speech = "这次动作没有成功完成。"
        else:
            return None
        details = json.dumps(terminal, ensure_ascii=False, separators=(",", ":"))
        response = parse_response(HermesResponse(speech, details).to_dict())
        return json.dumps(response.to_dict(), ensure_ascii=False)
    except (KeyError, TypeError, ValueError):
        return None
