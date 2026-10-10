"""Check the probe's approved step against the Agent presentation.

Typed Programs display asynchronously: the robot may already be moving when
the presentation arrives, no receipt is sent, and ``presentation_rendered``
is not awaited.  A plan that differs from the approved step is unapproved
motion, so the probe requests a stop.  Legacy receipt-gated presentations are
still acknowledged only after the exact plan was flushed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from std_msgs.msg import String

from ibrobot_agent.chat_tui import (
    AgentChatNode,
    ChatState,
    _event_matches_session,
    _is_async_presentation,
    _render_output_item,
)


def _expected_steps(skill_names) -> list[dict]:
    return [
        {
            "schema_version": 1,
            "skill_name": name,
            "target_name": "",
            "container_name": "",
            "place_name": "",
            "motion_direction": "",
            "motion_distance": 0.0,
            "timeout_sec": 0.0,
        }
        for name in skill_names
    ]


def _for_request(events, request_id: str, event_type: str):
    return next(
        (
            e
            for e in events
            if e.get("event_type") == event_type and e.get("request_key", {}).get("request_id") == request_id
        ),
        None,
    )


def display_probe_plan(node, events, request_id: str, skill_name, spin_until, timeout_sec: float) -> dict:
    """Verify and render the plan; return the presentation event.

    ``skill_name`` may be one name or an ordered sequence of approved names.
    """
    skill_names = [skill_name] if isinstance(skill_name, str) else list(skill_name)
    state = ChatState("probe")
    state.register(request_id)
    spin_until(node, lambda: _for_request(events, request_id, "presentation") is not None, timeout_sec)
    event = _for_request(events, request_id, "presentation")
    publisher = node.create_publisher(String, "/agent/control", 10)
    try:
        if not _event_matches_session(event, state):
            raise RuntimeError("presentation does not belong to the probe identity")
        if event["detail"]["presentation"]["steps"] != _expected_steps(skill_names):
            raise RuntimeError("presentation differs from the approved probe steps")
        adapter = SimpleNamespace(state=state, _control_publisher=publisher)
        adapter.acknowledge_presentation = lambda payload: AgentChatNode.acknowledge_presentation(adapter, payload)
        if _is_async_presentation(event):
            # Display only; execution never waited for this probe.
            _render_output_item(adapter, "event", event, writer=lambda line: print(line, flush=True))
            return event
        spin_until(node, lambda: publisher.get_subscription_count() > 0, 5.0)
        _render_output_item(adapter, "event", event, writer=lambda line: print(line, flush=True))
        spin_until(node, lambda: _for_request(events, request_id, "presentation_rendered") is not None, 5.0)
        return event
    except Exception:
        publisher.publish(String(data=json.dumps({"operation": "stop", "request_id": request_id})))
        raise
    finally:
        node.destroy_publisher(publisher)
