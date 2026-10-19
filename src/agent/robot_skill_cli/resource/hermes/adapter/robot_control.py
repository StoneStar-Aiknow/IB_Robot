"""Optional native control hooks, installed by hermes-robot-configure.

No robot knowledge lives here: authenticated input and authorized tool dispatch
are exposed to the profile's plugin, ahead of conversational queueing.
"""

from __future__ import annotations

import asyncio
import threading
import time

ROBOT_CONTROL_API = 1


def priority_control_text(event):
    """Classify only; authorization and cancellation still happen in Gateway."""
    if not event.allow_gateway_control or getattr(event, "internal", False):
        return False
    if getattr(event, "media_urls", None) or getattr(event, "media_types", None):
        return False
    from hermes_cli.lifecycle import invoke_hook

    return any(value is True for value in invoke_hook("is_priority_control_text", text=event.text or ""))


def control_input(text, session_id, *, busy=False, agent=None):
    from hermes_cli.lifecycle import invoke_hook

    for result in invoke_hook(
        "on_control_input",
        text=text,
        session_id=session_id,
        busy=busy,
        agent=agent,
    ):
        if isinstance(result, dict) and result.get("handled") is True:
            return result
    return None


def begin_turn(agent, session_id, turn_id):
    from hermes_cli.lifecycle import invoke_hook

    agent._robot_control_session = session_id
    agent._robot_control_turn = turn_id
    invoke_hook("on_control_turn_start", session_id=session_id, turn_id=turn_id, started_at=time.time())


def end_turn(agent):
    from hermes_cli.lifecycle import invoke_hook

    invoke_hook(
        "on_control_turn_end",
        session_id=getattr(agent, "_robot_control_session", ""),
        turn_id=getattr(agent, "_robot_control_turn", ""),
    )


def begin_tool_batch(agent, assistant_message):
    """Bind optional direct replies to this exact tool batch, not arbitrary stdout."""
    agent._managed_tool_batch = {
        "lock": threading.Lock(),
        "content": assistant_message.content or "",
        "calls": [
            {"id": call.id, "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in assistant_message.tool_calls
        ],
        "results": [],
    }


def execute_managed(agent, tool_name, args, tool_call_id=""):
    from hermes_cli.lifecycle import invoke_hook

    batch = getattr(agent, "_managed_tool_batch", None)
    for result in invoke_hook(
        "execute_managed_tool",
        tool_name=tool_name,
        args=args,
        session_id=getattr(agent, "_robot_control_session", ""),
        turn_id=getattr(agent, "_robot_control_turn", ""),
        tool_call_id=tool_call_id,
        presentation_callback=getattr(agent, "_emit_status", None),
    ):
        if isinstance(result, dict) and result.get("handled") is True:
            if batch is not None and isinstance(result.get("final_response"), str):
                with batch["lock"]:
                    batch["results"].append(
                        {
                            "tool_call_id": tool_call_id,
                            "result": result["result"],
                            "final_response": result["final_response"],
                        }
                    )
            return result
    return None


def finish_tool_batch(agent, messages):
    from hermes_cli.lifecycle import invoke_hook

    batch = getattr(agent, "_managed_tool_batch", None)
    agent._managed_tool_batch = None
    if batch is None or getattr(agent, "_interrupt_requested", False) or getattr(agent, "_pending_steer", None):
        return None
    with batch["lock"]:
        results = list(batch["results"])
    if len(batch["calls"]) != 1 or len(results) != 1 or not messages:
        return None
    last = messages[-1]
    result = results[0]
    # A timeout, guardrail transform or user steering changes the posted result.
    # Normal conversation processing must see that new observation.
    if not (
        last.get("role") == "tool"
        and last.get("tool_call_id") == result["tool_call_id"] == batch["calls"][0]["id"]
        and last.get("content") == result["result"]
    ):
        return None
    for decision in invoke_hook(
        "finalize_tool_batch",
        content=batch["content"],
        tool_calls=batch["calls"],
        managed_results=results,
        session_id=getattr(agent, "_robot_control_session", ""),
        turn_id=getattr(agent, "_robot_control_turn", ""),
    ):
        if isinstance(decision, dict) and isinstance(decision.get("response"), str) and decision["response"]:
            return decision["response"]
    return None


async def gateway_input(runner, event, session_key, *, busy):
    if not event.allow_gateway_control or getattr(event, "internal", False):
        return None
    if getattr(event, "media_urls", None) or getattr(event, "media_types", None):
        return None
    state = runner._peek_session_state(session_key)
    agent = state.turn.agent if state else None
    session_id = getattr(agent, "session_id", "")
    if not session_id:
        session_id = await asyncio.to_thread(
            runner._lookup_session_id_under_store_lock,
            runner.session_store,
            session_key,
        )
    if not session_id:
        return None
    # Busy handlers do not otherwise enter the multiplex profile scope.
    profile = getattr(event.source, "profile", "")
    if profile:
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        with _profile_runtime_scope(get_profile_dir(profile)):
            return control_input(event.text or "", session_id, busy=busy, agent=agent)
    return control_input(event.text or "", session_id, busy=busy, agent=agent)
