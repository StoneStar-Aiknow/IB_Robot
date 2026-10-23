"""Execute an approved SO-101 workflow through Kimi and the Agent node."""

from __future__ import annotations

import argparse
import json
import time
import uuid

import rclpy
from presentation_probe import display_probe_plan
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


def _spin_until(node: Node, predicate, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise RuntimeError("timed out waiting for Kimi SO-101 execution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="只执行一次点头动作，不要执行其他动作。")
    parser.add_argument("--steps", nargs="+", default=["nod_yes"], help="approved ordered skill names")
    args = parser.parse_args()
    rclpy.init()
    node = Node("agent_kimi_hardware_probe")
    responses: list[dict] = []
    events: list[dict] = []
    arrivals: dict[tuple[str, str], float] = {}

    def on_event(msg: String) -> None:
        payload = json.loads(msg.data)
        events.append(payload)
        key = (payload.get("request_key", {}).get("request_id", ""), payload.get("event_type", ""))
        arrivals.setdefault(key, time.monotonic())

    subscriptions = [
        node.create_subscription(String, "/agent/response", lambda msg: responses.append(json.loads(msg.data)), 10),
        node.create_subscription(String, "/agent/event", on_event, 10),
    ]
    publisher = node.create_publisher(String, "/agent/request", 10)
    ready_client = node.create_client(Trigger, "/ibrobot_agent_node/ready")
    try:
        if not ready_client.wait_for_service(timeout_sec=90.0):
            raise RuntimeError("Agent ready service is unavailable")
        ready_response = None
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            future = ready_client.call_async(Trigger.Request())
            _spin_until(node, future.done, 5.0)
            ready_response = future.result()
            if ready_response is not None and ready_response.success:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Agent did not become ready: {getattr(ready_response, 'message', '')}")

        request_id = f"kimi-hardware-{uuid.uuid4().hex[:12]}"
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "session_id": "kimi-hardware-session",
                "text": args.text,
            },
            ensure_ascii=False,
        )
        start = time.monotonic()
        publisher.publish(message)
        _spin_until(node, lambda: any(item.get("request_id") == request_id for item in responses), 10.0)
        presentation = display_probe_plan(node, events, request_id, args.steps, _spin_until, 150.0)
        _spin_until(
            node,
            lambda: any(
                item.get("request_key", {}).get("request_id") == request_id and item.get("event_type") == "terminal"
                for item in events
            ),
            150.0,
        )
        relevant = [item for item in events if item.get("request_key", {}).get("request_id") == request_id]
        terminal = next(item for item in relevant if item.get("event_type") == "terminal")
        assert terminal["state"] == "SUCCEEDED"
        event_types = {item["event_type"] for item in relevant}
        assert event_types >= {"proposal_ready", "presentation", "terminal"}
        async_display = presentation["detail"].get("async_display") is True
        if async_display:
            # Direct execution: no display receipt was required or recorded.
            assert "presentation_rendered" not in event_types

        def since_start(event_type: str) -> float | None:
            arrived = arrivals.get((request_id, event_type))
            return None if arrived is None else round((arrived - start) * 1000.0, 3)

        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "async_display": async_display,
                    "steps": args.steps,
                    "elapsed_ms": round((time.monotonic() - start) * 1000.0, 3),
                    "proposal_ready_ms": since_start("proposal_ready"),
                    "presentation_ms": since_start("presentation"),
                    "terminal_ms": since_start("terminal"),
                    "events": relevant,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
