"""Send a parameterized request to the resident interaction demo node."""

import argparse
import json
import math
import sys
import uuid

import rclpy

from ibrobot_msgs.srv import ExecuteInteractionDemo
from robot_interaction_demo.node import DEFAULT_SERVICE


def build_request(args) -> ExecuteInteractionDemo.Request:
    request = ExecuteInteractionDemo.Request()
    if args.operation == "speak":
        request.operation = request.SPEAK
        request.text = args.text
        request.language = args.language
        request.priority = args.priority
        request.interrupt = args.interrupt
    else:
        request.operation = request.NAMED_MOTION
        request.motion_name = args.name
        request.target = args.target
        request.interrupt = args.interrupt
        request.allow_motion = args.allow_motion
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--timeout", type=float, default=45.0, help="Client wait budget in seconds")
    commands = parser.add_subparsers(dest="operation", required=True)
    speech = commands.add_parser("speak", help="Request native robot speech")
    speech.add_argument("--text", required=True)
    speech.add_argument("--language", default="", help="Empty selects the runtime default")
    speech.add_argument("--priority", type=int, choices=range(101), default=50)
    speech.add_argument("--interrupt", action="store_true")
    motion = commands.add_parser("motion", help="Execute one semantic named motion")
    motion.add_argument("--name", required=True)
    motion.add_argument("--target", choices=("", "left", "right", "both"), default="")
    motion.add_argument("--interrupt", action="store_true")
    motion.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    rclpy.init(args=[])
    node = rclpy.create_node(f"interaction_demo_client_{uuid.uuid4().hex[:8]}")
    try:
        client = node.create_client(ExecuteInteractionDemo, args.service)
        if not client.wait_for_service(timeout_sec=args.timeout):
            raise TimeoutError("Demo service unavailable; no request sent. Start the unified robot launch first.")
        future = client.call_async(build_request(args))
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
        if not future.done():
            raise TimeoutError("Demo response timed out; the request may still execute. No retry or cancellation sent.")
        response = future.result()
        print(json.dumps(json.loads(response.result_json), ensure_ascii=False, indent=2))
        return 0 if response.success else 1
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; the request may still execute. No cancellation sent.", file=sys.stderr)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
