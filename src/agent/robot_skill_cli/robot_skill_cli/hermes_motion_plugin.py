"""Native Hermes hooks for out-of-band robot stopping and managed execution."""

from __future__ import annotations

import json
import logging
import threading

from .hermes_robot_runner import RobotRunner, workflow_arguments
from .hermes_stop_state import StopStateError, StopStore
from .hermes_workflow_response import completion_requested, workflow_response
from .stop_contract import is_stop

logger = logging.getLogger(__name__)


class MotionAdapter:
    def __init__(
        self, store: StopStore, executable: str, *, environment: dict | None = None, cancel_timeout: float = 75.0
    ):
        self.store = store
        self.runner = RobotRunner(store, executable, environment=environment, cancel_timeout=cancel_timeout)

    def user_input(self, text="", session_id="", busy=False, agent=None, **_kwargs):
        if not session_id:
            return None
        stop = is_stop(text)
        try:
            result = self.store.user_input(session_id, stop=stop, busy=busy)
        except (OSError, ValueError, StopStateError):
            if stop:
                logger.exception("IBROBOT_STOP_DELIVERY_FAILED session=%s", session_id)
                return {"handled": True, "message": "停止请求未确认送达，机器人停止状态未知。"}
            return None
        if not stop:
            return None
        logger.info("IBROBOT_STOP_RECEIVED session=%s state=%s", session_id, result["state"])
        owner = self.store.snapshot(session_id)["owner"]
        if (
            owner
            and owner["state"] in {"starting", "running", "unknown"}
            and not self.store.owner_alive(owner["token"])
        ):
            threading.Thread(target=self.runner.recover, args=(session_id,), daemon=True).start()
            result["state"] = "unknown"
        if agent is not None and callable(getattr(agent, "steer", None)):
            # Context notification only. The shared stop ledger has already
            # stopped admission and signaled the running CLI independently.
            agent.steer("用户已要求停止。停止由控制入口处理，等待当前任务终态；本回合不要发起新的机器人动作。")
        message = {
            "stopping": "正在停止当前动作，等待执行结果确认。",
            "unknown": "已记录停止要求，但还无法确认机器人是否停下。",
            "no_active_task": "已阻止本回合的新动作；当前没有本会话正在执行的机器人任务。",
        }[result["state"]]
        return {"handled": True, "message": message}

    def begin_turn(self, session_id="", turn_id="", started_at=0.0, **_kwargs):
        self.store.begin_turn(session_id, turn_id, started_at)

    def end_turn(self, session_id="", turn_id="", **_kwargs):
        self.store.end_turn(session_id, turn_id)

    def execute(self, tool_name="", args=None, session_id="", turn_id="", **_kwargs):
        if tool_name != "terminal" or not isinstance(args, dict):
            return None
        command = args.get("command", "")
        try:
            workflow = workflow_arguments(command, self.runner.executable, self.runner.environment)
            if workflow is None:
                return None
            if args.get("background") or args.get("pty"):
                raise StopStateError("robot workflow requires managed foreground execution")
            from tools.interrupt import is_interrupted

            direct = []
            result = self.runner.run(
                workflow,
                session_id,
                turn_id,
                interrupted=is_interrupted,
                timeout=args.get("timeout", 600),
                terminal_callback=lambda presentation, terminal: direct.append(
                    workflow_response(presentation, terminal)
                ),
                presentation_callback=_kwargs.get("presentation_callback"),
            )
            reply = {"handled": True, "result": result}
            if len(direct) == 1 and direct[0]:
                reply["final_response"] = direct[0]
            return reply
        except (OSError, ValueError, StopStateError) as exc:
            result = json.dumps({"error": str(exc), "exit_code": 15}, ensure_ascii=False)
            return {"handled": True, "result": result}

    def finish_tools(self, content="", tool_calls=(), managed_results=(), **_kwargs):
        if not completion_requested(content, tool_calls, self.runner.executable, self.runner.environment):
            return None
        if len(managed_results) != 1:
            return None
        result = managed_results[0]
        if result.get("tool_call_id") != tool_calls[0].get("id"):
            return None
        final = result.get("final_response")
        return {"response": final} if isinstance(final, str) and final else None


def register(ctx):
    from hermes_cli.robot_control import ROBOT_CONTROL_API
    from hermes_constants import get_hermes_home

    if ROBOT_CONTROL_API != 1:
        raise RuntimeError("unsupported Hermes robot control adapter")
    adapter = MotionAdapter(
        StopStore(ctx.get_config("state_dir", str(get_hermes_home() / "ibrobot/control-state"))),
        ctx.get_config("executable", str(get_hermes_home() / "ibrobot/bin/robot-skill")),
        environment=ctx.get_config("environment", {}),
        cancel_timeout=float(ctx.get_config("cancel_timeout_sec", 75.0)),
    )
    ctx.register_hook("on_control_input", adapter.user_input)
    ctx.register_hook("is_priority_control_text", lambda text="", **_kwargs: is_stop(text))
    ctx.register_hook("on_control_turn_start", adapter.begin_turn)
    ctx.register_hook("on_control_turn_end", adapter.end_turn)
    ctx.register_hook("execute_managed_tool", adapter.execute)
    ctx.register_hook("finalize_tool_batch", adapter.finish_tools)
