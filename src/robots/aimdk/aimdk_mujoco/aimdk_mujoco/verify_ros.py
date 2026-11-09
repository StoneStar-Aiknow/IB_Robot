"""Bounded ROS translation/stop check; does not certify the foot gait."""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from aimdk_interfaces.action import ExecuteSimulationPolicy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from std_srvs.srv import Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Directory containing prepared assets")
    parser.add_argument("--config-dir", type=Path, help="Controller catalog directory; defaults to installed configs")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--check-pause-resume", action="store_true", help="Also check physics pause/resume before the phases"
    )
    args = parser.parse_args()
    if not args.assets.is_dir():
        parser.error(f"Assets directory does not exist: {args.assets}")
    if args.config_dir is not None and not args.config_dir.is_dir():
        parser.error(f"Controller catalog directory does not exist: {args.config_dir}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"translation_passed": False, "gait_evaluated": False, "phases": []}
    log_path = args.report.with_suffix(".log")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "aimdk_mujoco",
                "simulation.launch.py",
                f"assets:={args.assets.resolve()}",
                f"headless:={str(not args.gui).lower()}",
                *([f"config_dir:={args.config_dir.resolve()}"] if args.config_dir else []),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        node = None
        action = None
        goal = None
        result_future = None
        states = []
        faults = []

        def stamp(state):
            return state.header.stamp.sec + state.header.stamp.nanosec / 1e9

        def spin_once(*, monitor_policy=True):
            if process.poll() is not None:
                raise RuntimeError(f"Simulation launch exited ({process.returncode}); see {log_path}")
            if monitor_policy:
                if faults:
                    raise RuntimeError(f"Simulation fault: {faults}; see {log_path}")
                if result_future is not None and result_future.done():
                    response = result_future.result()
                    raise RuntimeError(
                        f"Policy ended before verification finished: status={response.status}, "
                        f"success={response.result.success}, {response.result.message}"
                    )
            rclpy.spin_once(node, timeout_sec=0.02)

        def wait_for(predicate, timeout, description, *, monitor_policy=True):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out waiting for {description}; see {log_path}")
                spin_once(monitor_policy=monitor_policy)

        try:
            rclpy.init()
            node = rclpy.create_node("x2_simulation_acceptance")
            node.create_subscription(Odometry, "/simulation/ground_truth/odom", states.append, 100)
            node.create_subscription(
                DiagnosticArray,
                "/simulation/diagnostics",
                lambda msg: faults.extend(
                    s.message
                    for s in msg.status
                    if s.name == "aimdk_mujoco/physics" and s.level == DiagnosticStatus.ERROR
                ),
                10,
            )
            publisher = node.create_publisher(Twist, "/simulation/cmd_vel", 1)
            action = ActionClient(node, ExecuteSimulationPolicy, "/simulation/execute_policy")
            wait_for(
                lambda: action.server_is_ready() and bool(states) and publisher.get_subscription_count() > 0,
                60,
                "resident action server, odometry and /simulation/cmd_vel subscriber",
            )
            request = ExecuteSimulationPolicy.Goal(
                controller="mc_rl_velocity", motion="", reset_before_start=True, duration_s=30.0
            )
            goal_future = action.send_goal_async(request)
            wait_for(goal_future.done, 10, "policy goal acknowledgement")
            goal = goal_future.result()
            if not goal.accepted:
                raise RuntimeError("mc_rl_velocity goal rejected; check controller catalog and server log")
            result_future = goal.get_result_async()
            begin = len(states)
            wait_for(
                lambda: len(states) >= begin + 2 and stamp(states[-1]) > stamp(states[begin]),
                60,
                "physics to advance after policy preparation",
            )
            if args.check_pause_resume:
                stop = node.create_client(Trigger, "/simulation/stop")
                start = node.create_client(Trigger, "/simulation/start")
                wait_for(lambda: stop.service_is_ready() and start.service_is_ready(), 5, "pause/resume services")
                future = stop.call_async(Trigger.Request())
                wait_for(future.done, 5, "pause response")
                if not future.result().success:
                    raise RuntimeError(f"Pause failed: {future.result().message}")
                # Drain in-flight pre-pause odometry before measuring the paused interval.
                settle = time.monotonic() + 0.2
                wait_for(lambda: time.monotonic() >= settle, 1, "pause telemetry to settle")
                paused_time = stamp(states[-1])
                begin = len(states)
                deadline = time.monotonic() + 1.0
                wait_for(lambda: time.monotonic() >= deadline, 2, "one wall second of paused physics")
                if len(states) < begin + 2 or any(stamp(s) != paused_time for s in states[begin:]):
                    raise RuntimeError("Pause did not keep simulation time fixed with live odometry")
                future = start.call_async(Trigger.Request())
                wait_for(future.done, 5, "resume response")
                if not future.result().success:
                    raise RuntimeError(f"Resume failed: {future.result().message}")
                wait_for(lambda: stamp(states[-1]) > paused_time, 5, "resumed physics to advance")
                report["pause_resume"] = {"passed": True, "paused_wall_duration_s": 1.0, "paused_sim_duration_s": 0.0}
            for phase, duration, speed in (("stand", 5, 0.0), ("forward", 10, 0.2), ("stop", 5, 0.0)):
                start = states[-1]
                begin = len(states)
                if phase == "stop":
                    # This is a velocity stop, not /simulation/stop (which pauses physics).
                    publisher.publish(Twist())
                wall_begin = time.monotonic()
                deadline = wall_begin + duration
                next_publish = 0
                while time.monotonic() < deadline:
                    # After the one zero command, silence also exercises the wall timeout.
                    if phase != "stop" and time.monotonic() >= next_publish:
                        msg = Twist()
                        msg.linear.x = speed
                        publisher.publish(msg)
                        next_publish = time.monotonic() + 0.1
                    spin_once()
                samples = states[begin:]
                if len(samples) < duration * 20:
                    raise RuntimeError(f"{phase}: insufficient live state updates")
                end = samples[-1]
                sim_elapsed = stamp(end) - stamp(start)
                if any(
                    not all(
                        math.isfinite(value)
                        for value in (
                            s.pose.pose.position.x,
                            s.pose.pose.position.z,
                            s.twist.twist.linear.x,
                            s.twist.twist.linear.y,
                        )
                    )
                    for s in [start, *samples]
                ):
                    raise RuntimeError(f"{phase}: non-finite ground truth")
                last_second = [s for s in samples if stamp(s) >= stamp(end) - 1.0]
                result = {
                    "name": phase,
                    "wall_duration_s": time.monotonic() - wall_begin,
                    "sim_duration_s": sim_elapsed,
                    "forward_m": end.pose.pose.position.x - start.pose.pose.position.x,
                    "min_height_m": min(s.pose.pose.position.z for s in samples),
                    "last_second_speed_m_s": sum(
                        math.hypot(s.twist.twist.linear.x, s.twist.twist.linear.y) for s in last_second
                    )
                    / len(last_second),
                }
                report["phases"].append(result)
                if sim_elapsed < duration * 0.8 or result["min_height_m"] < 0.5:
                    raise RuntimeError(f"{phase}: slow or unstable physics")
                if phase == "forward" and result["forward_m"] < 0.8:
                    raise RuntimeError("ROS forward command did not produce translation")
                if phase != "forward" and result["last_second_speed_m_s"] > 0.1:
                    raise RuntimeError(f"{phase}: failed to settle")
            report["translation_passed"] = True
        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if goal is not None and goal.accepted and result_future is not None:
                try:
                    cancel_accepted = False
                    if not result_future.done():
                        future = goal.cancel_goal_async()
                        wait_for(future.done, 5, "cancel acknowledgement", monitor_policy=False)
                        cancel_accepted = any(
                            bytes(info.goal_id.uuid) == bytes(goal.goal_id.uuid)
                            for info in future.result().goals_canceling
                        )
                    wait_for(result_future.done, 10, "terminal policy result", monitor_policy=False)
                    response = result_future.result()
                    report["policy_result"] = {
                        "status": response.status,
                        "success": response.result.success,
                        "message": response.result.message,
                        "completed_steps": response.result.completed_steps,
                        "cancel_accepted": cancel_accepted,
                    }
                    if cancel_accepted:
                        if response.status != GoalStatus.STATUS_CANCELED or response.result.success:
                            raise RuntimeError("Accepted cancellation must finish CANCELED with success=false")
                    elif response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
                        raise RuntimeError("Policy neither completed successfully nor acknowledged cancellation")
                    begin = len(states)
                    deadline = time.monotonic() + 0.5
                    wait_for(
                        lambda: time.monotonic() >= deadline and len(states) >= begin + 2,
                        3,
                        "resident server telemetry after the terminal result",
                        monitor_policy=False,
                    )
                    if process.poll() is not None or faults or not action.server_is_ready():
                        raise RuntimeError(f"Resident simulation exited or faulted: {faults}")
                    report["server_alive_after_result"] = True
                except (Exception, KeyboardInterrupt) as exc:
                    report["translation_passed"] = False
                    report.setdefault("cleanup_errors", []).append(f"Policy cleanup: {type(exc).__name__}: {exc}")
            try:
                if action is not None:
                    action.destroy()
                if node is not None:
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except (Exception, KeyboardInterrupt) as exc:
                report["translation_passed"] = False
                report.setdefault("cleanup_errors", []).append(f"ROS cleanup: {type(exc).__name__}: {exc}")
            if process.poll() is None:
                # launch forwards SIGINT to its child; do not signal the child twice.
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                report["translation_passed"] = False
                report.setdefault("cleanup_errors", []).append("Simulation did not exit after SIGINT")
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    if "[ERROR]" in log_path.read_text() or process.returncode not in (0, 130, -signal.SIGINT):
        report["translation_passed"] = False
        report.setdefault("error", f"Launch or child process failed; see {log_path}")
    text = json.dumps(report, indent=2)
    args.report.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["translation_passed"] else 1)


if __name__ == "__main__":
    main()
