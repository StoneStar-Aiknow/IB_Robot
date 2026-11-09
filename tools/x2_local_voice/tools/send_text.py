"""Send one typed sentence to the voice bridge, as an ASR transcript, and wait for its terminal.

    rig_env.sh python3 tools/send_text.py "欧拉欧拉，自我介绍" [--interrupt 别说了 --after-progress-s 3]

Prints REQUEST, EVENT and a final SUMMARY line (JSON). Exit code 2 when no terminal arrived.
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from std_msgs.msg import String

from ibrobot_msgs.msg import SpeechTranscript

TERMINAL = {"SUCCEEDED", "FAILED", "UNKNOWN", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "REJECTED", "ANSWERED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--interrupt", default="", help="second sentence sent while the request runs")
    parser.add_argument("--after-progress-s", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("x2_local_voice_send_text")
    state = {"request_id": None, "terminal": None, "progress_at": None, "interrupted": False}
    publisher = node.create_publisher(SpeechTranscript, "/voice/transcript", 10)
    sequence = int(time.time() * 1000) % 1_000_000

    def say(text: str) -> None:
        nonlocal sequence
        sequence += 1
        publisher.publish(SpeechTranscript(source_id="x2_local_voice_text", segment_sequence=sequence, text=text))
        print("INPUT", text, flush=True)

    def on_request(message: String) -> None:
        payload = json.loads(message.data)
        print("REQUEST", json.dumps(payload, ensure_ascii=False), flush=True)
        state["request_id"] = state["request_id"] or payload["request_id"]

    def on_event(message: String) -> None:
        payload = json.loads(message.data)
        if not state["request_id"] or payload.get("request_key", {}).get("request_id") != state["request_id"]:
            return
        print("EVENT", json.dumps(payload, ensure_ascii=False), flush=True)
        if payload.get("event_type") == "execution_progress" and state["progress_at"] is None:
            state["progress_at"] = time.monotonic()
        if payload.get("state") in TERMINAL:
            state["terminal"] = payload

    node.create_subscription(String, "/agent/request", on_request, 30)
    node.create_subscription(String, "/agent/event", on_event, 100)
    # The sentence is published once: wait until the voice bridge has matched this publisher.
    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    settle = time.monotonic() + 1.0
    while time.monotonic() < settle:
        rclpy.spin_once(node, timeout_sec=0.1)
    say(args.text)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not state["terminal"]:
        rclpy.spin_once(node, timeout_sec=0.1)
        progress_at = state["progress_at"]
        due = progress_at is not None and time.monotonic() - progress_at >= args.after_progress_s
        if args.interrupt and not state["interrupted"] and due:
            state["interrupted"] = True
            say(args.interrupt)
    terminal = state["terminal"] or {}
    summary = {
        "request_id": state["request_id"],
        "interrupted": state["interrupted"],
        "state": terminal.get("state"),
        "message": terminal.get("user_message"),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if state["terminal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
