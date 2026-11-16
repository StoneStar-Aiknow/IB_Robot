"""Smoke of a running direct-control runtime: idle, one move, the simulated clock, finish then stop.

    python3 src/robot_runtime_sim/scripts/direct_control_smoke.py --out smoke.json --group <motion group> \
        [--idle 3.0] [--finish]

A developer tool kept in the source tree and not installed with the runtime; run it from the
worktree in a sourced environment (it imports ``robot_runtime_sim`` and ``ibrobot_msgs``).

Needs a runtime with a session in progress (the platform connected and reset). Talks only to
the public runtime contract, ``/clock`` and the /simulation surface; it checks no evaluation
concept (an evaluation layer brings its own checks). Exit code 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_msgs.srv import MoveToPose, SetRuntimeMode, StopRuntime
from robot_runtime_sim.direct_control.node import CLOCK_TOPIC, FINISH_SESSION_SERVICE, ROS_CLOCK_TOPIC


class Smoke(Node):
    def __init__(self, group: str):
        super().__init__("simulation_smoke")
        self._lock = threading.Lock()
        self.clock: dict[str, Any] = {}
        self.ros_clock_s = -1.0
        self.ee: PoseStamped | None = None
        self.create_subscription(String, CLOCK_TOPIC, self._on_clock, 10)
        self.create_subscription(Clock, ROS_CLOCK_TOPIC, self._on_ros_clock, 10)
        self.create_subscription(PoseStamped, f"/robot_status/{group}/ee_pose", self._on_ee, 10)
        self.mode = self.create_client(SetRuntimeMode, "/runtime/set_mode")
        self.stop = self.create_client(StopRuntime, "/runtime/stop")
        self.move = self.create_client(MoveToPose, "/motion/move_to_pose")
        self.finish = self.create_client(Trigger, FINISH_SESSION_SERVICE)

    def _on_clock(self, msg: String) -> None:
        with self._lock:
            self.clock = json.loads(msg.data)

    def _on_ros_clock(self, msg: Clock) -> None:
        with self._lock:
            self.ros_clock_s = msg.clock.sec + msg.clock.nanosec * 1e-9

    def sim_s(self) -> float:
        with self._lock:
            return self.ros_clock_s

    def _on_ee(self, msg: PoseStamped) -> None:
        self.ee = msg

    def tick(self) -> int:
        with self._lock:
            return int(self.clock.get("tick", -1))

    def call(self, client, request, timeout_s: float = 90.0):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f"{client.srv_name} timed out")
        return future.result()


def _move_up(node: Smoke, group: str, dz: float):
    p = node.ee.pose
    request = MoveToPose.Request(target_group=group, velocity_scaling=0.5)
    request.target_pose.position.x, request.target_pose.position.y = p.position.x, p.position.y
    request.target_pose.position.z = p.position.z + dz
    request.target_pose.orientation = p.orientation
    return node.call(node.move, request)


def run(node: Smoke, args) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline and node.clock.get("state") not in ("READY", "RUNNING"):
        time.sleep(0.2)
    checks["session"] = dict(node.clock)
    if node.clock.get("state") not in ("READY", "RUNNING"):
        raise RuntimeError(f"no running session within {args.wait} s: {node.clock}")
    checks["set_mode"] = node.call(node.mode, SetRuntimeMode.Request(mode="trajectory")).success
    deadline = time.monotonic() + 30.0
    while node.ee is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if node.ee is None:
        raise RuntimeError(f"no ee pose for {args.group}")
    t0, s0, w0 = node.tick(), node.sim_s(), time.time()
    time.sleep(args.idle)
    t1, s1 = node.tick(), node.sim_s()
    checks["idle"] = {
        "wall_s": round(time.time() - w0, 2),
        "tick_before": t0,
        "tick_after": t1,
        "clock_before_s": s0,
        "clock_after_s": s1,
        "ok": t0 == t1 and s0 == s1 and s0 >= 0.0,  # no step, and /clock stands still
    }

    moved = _move_up(node, args.group, args.move_dz)
    checks["move_to_pose"] = {
        "group": args.group,
        "outcome": moved.outcome,
        "message": moved.message,
        "position_error_m": moved.position_error_m,
        "sim_time_s": moved.execution_time_s,
        "clock_advanced_s": round(node.sim_s() - s1, 6),
        "ok": moved.success and node.sim_s() > s1,
    }
    if args.finish:
        # finish -> explicit stop must stop ticking (not just cancel), then finish again to end the session
        node.call(node.finish, Trigger.Request())
        time.sleep(1.0)
        running = node.tick()
        node.call(node.stop, StopRuntime.Request(policy="HOLD"))
        time.sleep(0.5)
        stopped_at = node.tick()
        time.sleep(2.0)
        checks["finish_then_stop"] = {
            "ticks_while_finishing_1s": running,
            "ticks_after_stop_2s": node.tick() - stopped_at,
            "ok": node.tick() - stopped_at == 0,
        }
        node.call(node.mode, SetRuntimeMode.Request(mode="idle"))
        node.call(node.mode, SetRuntimeMode.Request(mode="trajectory"))
        checks["finish_session"] = node.call(node.finish, Trigger.Request()).message
    checks["all_ok"] = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict))
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", required=True, help="motion group for the move check")
    parser.add_argument("--idle", type=float, default=3.0)
    parser.add_argument("--move-dz", type=float, default=0.03)
    parser.add_argument("--wait", type=float, default=600.0)
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args(argv)
    rclpy.init()
    node = Smoke(args.group)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        checks = run(node, args)
    except Exception as exc:
        checks = {"error": repr(exc), "all_ok": False}
    finally:
        executor.shutdown()
        spinner.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=1, default=str)
    print(json.dumps(checks, indent=1, default=str))
    return 0 if checks.get("all_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
