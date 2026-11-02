"""Bounded real-weight ROS sequence: one initial reset, then experimental no-reset handoffs."""

import argparse
import json
import math
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from aimdk_interfaces.action import ExecuteSimulationPolicy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState


class Telemetry:
    """Keep bounded summaries, checking each stream independently of DDS arrival order."""

    def __init__(self):
        self.streams = {}
        self.faults = []
        self.position = None

    def observe(self, name, stamp, values=()):
        now = time.monotonic()
        seconds = stamp.sec + stamp.nanosec / 1e9
        if not 0 <= stamp.nanosec < 1_000_000_000 or not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"{name}: invalid simulation timestamp")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{name}: non-finite telemetry")
        previous = self.streams.get(name)
        if previous is not None:
            delta = seconds - previous["sim_end_s"]
            if delta < -1e-9 or (name != "diagnostics" and delta > 0.5):
                raise RuntimeError(f"{name}: simulation clock discontinuity ({delta:.6f} s)")
            if name == "odom" and self.position is not None:
                distance = math.dist(values[:3], self.position)
                if distance > 0.15 + 3 * max(0, delta):
                    raise RuntimeError("odom: position discontinuity across handoff")
        else:
            self.streams[name] = {"samples": 0, "sim_start_s": seconds}
        self.streams[name].update(samples=self.streams[name]["samples"] + 1, sim_end_s=seconds, last_wall_s=now)
        if name == "odom":
            self.position = tuple(values[:3])

    def odom(self, msg):
        pose, twist = msg.pose.pose, msg.twist.twist
        self.observe(
            "odom",
            msg.header.stamp,
            [
                getattr(vector, field)
                for vector, fields in (
                    (pose.position, "xyz"),
                    (pose.orientation, "xyzw"),
                    (twist.linear, "xyz"),
                    (twist.angular, "xyz"),
                )
                for field in fields
            ],
        )

    def diagnostics(self, msg):
        for status in msg.status:
            if status.name == "aimdk_mujoco/physics":
                self.observe("diagnostics", msg.header.stamp)
                if status.level >= DiagnosticStatus.ERROR:
                    self.faults.append(status.message)

    def check(self):
        if self.faults:
            raise RuntimeError(f"Simulation diagnostic fault: {self.faults}")
        for name, stream in self.streams.items():
            if time.monotonic() - stream["last_wall_s"] > 3:
                raise RuntimeError(f"{name}: telemetry stopped")


def validate_result(response, controller, total_steps, *, canceled=False):
    result = response.result
    if canceled:
        valid = response.status == GoalStatus.STATUS_CANCELED and not result.success
        valid = valid and 500 <= result.completed_steps < total_steps
    else:
        valid = response.status == GoalStatus.STATUS_SUCCEEDED and result.success
        # Recovery may settle early, but requires 50 consecutive standing steps.
        minimum = 50 if controller == "recover_stand" else total_steps
        valid = valid and 0 < minimum <= result.completed_steps <= total_steps
    if not valid:
        raise RuntimeError(
            f"Premature or invalid result: status={response.status}, success={result.success}, "
            f"steps={result.completed_steps}/{total_steps}, {result.message}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    for directory in (args.assets, args.config_dir):
        if directory is not None and not directory.is_dir():
            parser.error(f"Directory does not exist: {directory}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_DOMAIN_ID", str(100 + secrets.randbelow(100)))
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    report = {"passed": False, "ros_domain_id": os.environ["ROS_DOMAIN_ID"], "phases": []}
    telemetry = Telemetry()
    log_path = args.report.with_suffix(".log")
    process = node = action = publisher = goal = result_future = None

    def spin():
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owned launch exited ({process.returncode}); see {log_path}")
        rclpy.spin_once(node, timeout_sec=0.02)
        telemetry.check()

    def wait_for(predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while True:
            telemetry.check()
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Owned launch exited ({process.returncode}); see {log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {description}")
            if predicate():
                return
            spin()

    def settle():
        before = {key: value["samples"] for key, value in telemetry.streams.items()}
        until = time.monotonic() + 0.3
        wait_for(
            lambda: (
                time.monotonic() >= until
                and all(
                    value["samples"] >= before[key] + 2
                    for key, value in telemetry.streams.items()
                    if key != "diagnostics"
                )
            ),
            3,
            "resident telemetry after result",
        )
        if not action.server_is_ready():
            raise RuntimeError("Resident action server disappeared")

    with log_path.open("w") as log:
        try:
            config_dir = args.config_dir or Path(get_package_share_directory("aimdk_mujoco")) / "config"
            motions = yaml.safe_load((config_dir / "bmimic.yaml").read_text())["motions"]
            rclpy.init()
            node = rclpy.create_node("x2_handoff_acceptance")
            action = ActionClient(node, ExecuteSimulationPolicy, "/simulation/execute_policy")
            # Refuse a shared simulation graph rather than send goals to someone else's node.
            until = time.monotonic() + 1
            wait_for(lambda: time.monotonic() >= until, 3, "domain discovery")
            if action.server_is_ready() or any(
                name in dict(node.get_topic_names_and_types())
                for name in ("/simulation/ground_truth/odom", "/simulation/clock")
            ):
                raise RuntimeError("ROS domain already contains a simulation; choose an unused ROS_DOMAIN_ID")
            node.create_subscription(Odometry, "/simulation/ground_truth/odom", telemetry.odom, 100)
            node.create_subscription(
                Clock, "/simulation/clock", lambda msg: telemetry.observe("clock", msg.clock), qos_profile_sensor_data
            )
            node.create_subscription(
                JointState,
                "/simulation/joint_states",
                lambda msg: telemetry.observe("joints", msg.header.stamp, [*msg.position, *msg.velocity]),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                Imu,
                "/simulation/imu/data",
                lambda msg: telemetry.observe(
                    "imu",
                    msg.header.stamp,
                    [
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ],
                ),
                qos_profile_sensor_data,
            )
            node.create_subscription(DiagnosticArray, "/simulation/diagnostics", telemetry.diagnostics, 10)
            publisher = node.create_publisher(Twist, "/simulation/cmd_vel", 1)
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
            wait_for(
                lambda: (
                    action.server_is_ready() and len(telemetry.streams) == 5 and publisher.get_subscription_count() == 1
                ),
                60,
                "owned simulation and all telemetry streams",
            )
            for name, controller, motion, duration in (
                ("dance_10s", "bmimic", "kuailechongbai", 0.0),
                ("recover_after_dance", "recover_stand", "", 10.0),
                ("velocity_0.4_5s", "mc_rl_velocity", "", 5.0),
                ("recover_after_velocity", "recover_stand", "", 10.0),
                ("fan_dance", "bmimic", "fan_dance", 0.0),
            ):
                phase = {
                    "name": name,
                    "passed": False,
                    "feedback": [],
                    "reset_before_start": name == "dance_10s",
                    "transition_from_current": name != "dance_10s",
                    "controller": controller,
                    "motion": motion,
                    "duration_s": duration,
                }
                report["phases"].append(phase)
                before = {key: value.copy() for key, value in telemetry.streams.items()}
                wall_start = time.monotonic()

                def feedback(msg, phase=phase):
                    value = msg.feedback
                    previous = phase["feedback"]
                    if previous and value.completed_steps < previous[-1]["completed_steps"]:
                        raise RuntimeError("Policy feedback step count went backwards")
                    phase["feedback"].append(
                        {
                            "state": value.state,
                            "completed_steps": value.completed_steps,
                            "total_steps": value.total_steps,
                        }
                    )

                request = ExecuteSimulationPolicy.Goal(
                    controller=controller,
                    motion=motion,
                    duration_s=duration,
                    reset_before_start=phase["reset_before_start"],
                    transition_from_current=phase["transition_from_current"],
                )
                future = action.send_goal_async(request, feedback_callback=feedback)
                wait_for(future.done, 10, f"{name} acknowledgement")
                goal = future.result()
                if not goal.accepted:
                    raise RuntimeError(f"{name}: goal rejected")
                result_future = goal.get_result_async()
                next_publish = 0
                deadline = time.monotonic() + 90
                while not result_future.done():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name}: execution deadline exceeded")
                    running = [f for f in phase["feedback"] if f["state"] == "running"]
                    if running and name == "dance_10s" and running[-1]["completed_steps"] >= 500:
                        phase["cancel_feedback_steps"] = running[-1]["completed_steps"]
                        future = goal.cancel_goal_async()
                        wait_for(future.done, 5, "dance cancellation acknowledgement")
                        if not any(
                            bytes(info.goal_id.uuid) == bytes(goal.goal_id.uuid)
                            for info in future.result().goals_canceling
                        ):
                            raise RuntimeError("Dance cancellation rejected")
                        wait_for(result_future.done, 5, "canceled dance result")
                        break
                    if running and controller == "mc_rl_velocity" and time.monotonic() >= next_publish:
                        command = Twist()
                        command.linear.x = 0.4
                        publisher.publish(command)
                        phase["cmd_vel_publishes"] = phase.get("cmd_vel_publishes", 0) + 1
                        next_publish = time.monotonic() + 0.1
                    spin()
                response = result_future.result()
                phase["result"] = {
                    "status": response.status,
                    "success": response.result.success,
                    "message": response.result.message,
                    "completed_steps": response.result.completed_steps,
                }
                running = [f for f in phase["feedback"] if f["state"] == "running"]
                total = running[-1]["total_steps"] if running else 0
                expected = (
                    motions[motion]["end_count"] - motions[motion]["start_count"] + 1
                    if controller == "bmimic"
                    else math.ceil(duration / 0.02)
                )
                if total != expected:
                    raise RuntimeError(f"{name}: incorrect feedback duration budget ({total} steps)")
                validate_result(response, controller, total, canceled=name == "dance_10s")
                if name == "dance_10s" and response.result.completed_steps < phase.get("cancel_feedback_steps", 500):
                    raise RuntimeError("Canceled result precedes the cancellation feedback gate")
                if name == "dance_10s" and "cancel_feedback_steps" not in phase:
                    raise RuntimeError("Dance finished before the feedback cancellation gate")
                if controller == "mc_rl_velocity" and not phase.get("cmd_vel_publishes"):
                    raise RuntimeError("Velocity goal ended before running feedback allowed commands")
                goal = None
                settle()
                phase["wall_duration_s"] = time.monotonic() - wall_start
                phase["telemetry"] = {}
                for stream, current in telemetry.streams.items():
                    elapsed = current["sim_end_s"] - before[stream]["sim_end_s"]
                    samples = current["samples"] - before[stream]["samples"]
                    phase["telemetry"][stream] = {"samples": samples, "sim_duration_s": elapsed}
                    if samples < (1 if stream == "diagnostics" else 2) or (
                        stream != "diagnostics" and abs(elapsed - response.result.completed_steps * 0.02) > 0.06
                    ):
                        raise RuntimeError(f"{name}: {stream} missing telemetry or reset/step timing discontinuity")
                phase["server_alive_after_result"] = True
                phase["passed"] = True
            report["passed"] = True
        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            if report["phases"] and not report["phases"][-1]["passed"]:
                report["phases"][-1]["error"] = report["error"]
        finally:
            # Only the launch we own is signaled; launch forwards SIGINT once to its child.
            try:
                if publisher is not None and process is not None:
                    publisher.publish(Twist())
                if goal is not None and goal.accepted and result_future is not None and not result_future.done():
                    future = goal.cancel_goal_async()
                    wait_for(future.done, 5, "cleanup cancellation")
                    wait_for(result_future.done, 5, "cleanup result")
            except (Exception, KeyboardInterrupt) as exc:
                report.setdefault("cleanup_errors", []).append(f"Policy cleanup: {exc}")
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    report.setdefault("cleanup_errors", []).append("Owned launch did not stop gracefully")
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                report["launch_returncode"] = process.returncode
            try:
                if action is not None:
                    action.destroy()
                if node is not None:
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except (Exception, KeyboardInterrupt) as exc:
                report.setdefault("cleanup_errors", []).append(f"ROS cleanup: {exc}")
    report["telemetry"] = telemetry.streams
    report["diagnostic_faults"] = telemetry.faults
    if report.get("cleanup_errors") or (
        process is not None
        and (process.returncode not in (0, 130, -signal.SIGINT) or "[ERROR]" in log_path.read_text())
    ):
        report["passed"] = False
        report.setdefault("error", f"Launch or cleanup failed; see {log_path}")
    text = json.dumps(report, indent=2)
    args.report.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
