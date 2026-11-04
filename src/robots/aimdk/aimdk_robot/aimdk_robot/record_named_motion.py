"""Record a preset motion's arm trajectory for replay in the upper-body stream mode.

Run against a live runtime that is in ``idle`` (stable stand), so the platform
plays its own preset::

    ros2 run aimdk_robot aimdk_record_named_motion wave --target right -o profiles/trajectories/wave.json

The tool sends ``/motion/execute_named`` for the name, records ``/joint_states``
until the motion completes, trims the still segments at both ends and writes
the trajectory file that ``vendor.named_motion.trajectories`` points at. It
never changes the runtime mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from aimdk_robot.named_motion import NAMED_MOTION_ACTION
from aimdk_robot.projection import VENDOR_ARM_ORDER
from aimdk_robot.trajectory import TARGETS
from ibrobot_msgs.action import ExecuteNamedMotion


def trim_still(times: list[float], rows: list[list[float]], *, threshold: float, margin_s: float):
    """Drop the samples before the motion leaves its start and after it reaches its end pose."""

    def moved(row, reference):
        return max(abs(a - b) for a, b in zip(row, reference, strict=True)) > threshold

    first = next((i for i, row in enumerate(rows) if moved(row, rows[0])), 0)
    last = next((i for i in range(len(rows) - 1, -1, -1) if moved(rows[i], rows[-1])), len(rows) - 1)
    begin = next((i for i in range(first, -1, -1) if times[first] - times[i] >= margin_s), 0)
    end = next((i for i in range(last, len(rows)) if times[i] - times[last] >= margin_s), len(rows) - 1)
    t0 = times[begin]
    return [round(t - t0, 4) for t in times[begin : end + 1]], rows[begin : end + 1]


def dump_trajectory(data: dict) -> str:
    """JSON with one sample per line, so a re-recording reads as a reviewable diff."""
    fields = [f" {json.dumps(key)}: {json.dumps(value)}" for key, value in data.items() if key != "positions"]
    rows = ",\n".join("  " + json.dumps(row) for row in data["positions"])
    return "{\n" + ",\n".join(fields) + ',\n "positions": [\n' + rows + "\n ]\n}\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="named motion to play and record, e.g. wave")
    parser.add_argument("-o", "--output", required=True, type=Path)
    # Required, not defaulted to the profile's: the side is sent with the goal,
    # so the file records the side the platform actually played, which the
    # replay checks requests against.
    parser.add_argument("--target", required=True, choices=TARGETS, help="side the preset plays (clap: body)")
    parser.add_argument("--threshold", type=float, default=0.01, help="rad; motion below this counts as still")
    parser.add_argument("--margin", type=float, default=0.2, help="s of still pose kept at each end")
    parser.add_argument("--timeout", type=float, default=40.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = rclpy.create_node("aimdk_record_named_motion")
    samples: list[tuple[float, list[float]]] = []
    recording = threading.Event()

    def on_joint_state(message: JointState) -> None:
        if not recording.is_set():
            return
        by_name = dict(zip(message.name, message.position, strict=False))
        if all(joint in by_name for joint in VENDOR_ARM_ORDER):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if not samples or stamp > samples[-1][0]:
                samples.append((stamp, [float(by_name[joint]) for joint in VENDOR_ARM_ORDER]))

    node.create_subscription(JointState, "/joint_states", on_joint_state, 50)
    client = ActionClient(node, ExecuteNamedMotion, NAMED_MOTION_ACTION)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            print(f"{NAMED_MOTION_ACTION} is not available", file=sys.stderr)
            return 1
        goal = ExecuteNamedMotion.Goal()
        goal.name = args.name
        goal.target = args.target
        recording.set()
        time.sleep(0.5)
        handle_future = client.send_goal_async(goal)
        deadline = time.monotonic() + args.timeout
        while not handle_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        handle = handle_future.result() if handle_future.done() else None
        if handle is None or not handle.accepted:
            print(f"goal {args.name!r} was not accepted", file=sys.stderr)
            return 1
        result_future = handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        recording.clear()
        if not result_future.done():
            print(f"{args.name} did not complete within {args.timeout}s", file=sys.stderr)
            return 1
        result = result_future.result().result
        if not result.success:
            print(f"{args.name} failed ({result.error_code}): {result.message}", file=sys.stderr)
            return 1
        if len(samples) < 2:
            print("no arm joint states were recorded", file=sys.stderr)
            return 1
        times = [stamp - samples[0][0] for stamp, _ in samples]
        rows = [row for _, row in samples]
        times, rows = trim_still(times, rows, threshold=args.threshold, margin_s=args.margin)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            dump_trajectory(
                {
                    "name": args.name,
                    "target": args.target,
                    "source": f"recorded from the platform preset via {NAMED_MOTION_ACTION}",
                    "joints": list(VENDOR_ARM_ORDER),
                    "time_from_start_s": times,
                    "positions": [[round(value, 5) for value in row] for row in rows],
                }
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output}: {len(times)} samples, {times[-1]:.2f}s")
        return 0
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
