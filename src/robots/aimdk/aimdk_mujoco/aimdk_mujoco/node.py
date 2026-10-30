"""Small, main-thread ROS wrapper around the standalone physics implementation."""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.task import Future
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Trigger

from aimdk_mujoco.policy_loader import PolicyLoader
from ibrobot_msgs.action import ExecuteSimulationPolicy


class X2Node(Node):
    def __init__(self, sim, config_dir=None):
        super().__init__("aimdk_mujoco")
        self.sim = sim
        self.config_dir = (
            Path(config_dir) if config_dir else Path(get_package_share_directory("aimdk_mujoco")) / "config"
        )
        self._reserved = False
        self._closing = False
        self._pause_requested = False
        self._goal = None
        self._done = None
        self._prepared = False
        self._total_steps = 0
        self._stable_steps = 0
        self._start_steps = 0
        self._feedback_time = -math.inf
        self._loader = PolicyLoader()
        self._loading = None
        self._command = (0.0, 0.0, 0.0)
        self._command_time = None
        self._fault = None
        self._diagnostic_time = -math.inf
        self.create_subscription(Twist, "/cmd_vel", self._on_command, 1)
        self.create_service(Trigger, "/simulation/stand", self._on_stand)
        self.create_service(Trigger, "/simulation/stop", self._on_stand)
        self.create_service(Trigger, "/simulation/start", self._on_start)
        self._odom_pub = self.create_publisher(Odometry, "/simulation/ground_truth/odom", 10)
        self._joints_pub = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        self._imu_pub = self.create_publisher(Imu, "/imu/data", qos_profile_sensor_data)
        self._clock_pub = self.create_publisher(
            Clock, "/clock", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self._diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._action = ActionServer(
            self,
            ExecuteSimulationPolicy,
            "/simulation/execute_policy",
            self._execute,
            goal_callback=self._on_goal,
            handle_accepted_callback=self._on_accepted,
            cancel_callback=self._on_cancel,
        )

    def _on_goal(self, request):
        valid = request.controller in ("bmimic", "mc_rl_velocity", "recover_stand") and math.isfinite(
            request.duration_s
        )
        valid = valid and not (request.reset_before_start and request.transition_from_current)
        if request.controller == "bmimic":
            valid = valid and bool(request.motion) and request.duration_s == 0
        else:
            valid = valid and not request.motion and 0 < request.duration_s <= 3600
        if request.controller == "recover_stand":
            valid = valid and request.transition_from_current and 1 <= request.duration_s <= 30
        if self._closing or self._reserved or self._fault is not None or not valid or self._loader.busy:
            return GoalResponse.REJECT
        self._reserved = True
        return GoalResponse.ACCEPT

    def _on_accepted(self, goal):
        if self._loading is not None and self._loading.done():
            self._loading = None
        self._goal = goal
        self._done = Future()
        self._prepared = False
        self._pause_requested = False
        self._stable_steps = 0
        self._start_steps = self.sim.policy_steps
        self._feedback_time = -math.inf
        goal.execute()

    def _on_cancel(self, goal):
        return CancelResponse.ACCEPT if goal is self._goal else CancelResponse.REJECT

    async def _execute(self, goal):
        done = self._done
        try:
            return await done
        finally:
            if self._goal is goal:
                self._goal = None
                self._done = None
                self._reserved = False

    def _finish(self, success, message, canceled=False):
        if self._goal is None or self._done.done():
            return
        if self._loading is not None:
            # Cancellation cannot stop a running load; admission tracks the thread.
            self._loading.cancel()
        self.sim.pause_execution()
        if success:
            self.sim.selection_finished = True
        result = ExecuteSimulationPolicy.Result(
            success=success,
            message=message,
            completed_steps=max(0, self.sim.policy_steps - self._start_steps) if self._prepared else 0,
        )
        if canceled:
            self._goal.canceled()
        elif success:
            self._goal.succeed()
        else:
            self._goal.abort()
        self._done.set_result(result)

    def _tick_goal(self):
        goal = self._goal
        if self._closing or goal is None or self._done.done():
            return
        if goal.is_cancel_requested:
            self._finish(False, "Canceled; physics paused, not active balance", canceled=True)
            return
        if not self._prepared:
            request = goal.request
            config = (
                self.config_dir
                / f"{'mc_rl_velocity' if request.controller == 'recover_stand' else request.controller}.yaml"
            )
            try:
                if request.reset_before_start or request.transition_from_current:
                    if self._loading is None:
                        self._loading = self._loader.submit(self.sim.prepare_policy, config, request.motion or None)
                        goal.publish_feedback(ExecuteSimulationPolicy.Feedback(state="preparing"))
                        return
                    if not self._loading.done():
                        return
                    prepared = self._loading.result()
                    self._loading = None
                    if request.transition_from_current:
                        self.sim.activate_policy(prepared, transition=True)
                    else:
                        self.sim.activate_policy(prepared)
                else:
                    self.sim.select_policy(config, request.motion or None, False)
            except Exception as exc:
                self._loading = None
                # Preparation errors must leave the currently paused state intact.
                self._finish(False, f"Policy preparation failed: {exc}")
                return
            self._command = (0.0, 0.0, 0.0)
            self._command_time = None
            self._start_steps = self.sim.policy_steps
            self._total_steps = (
                self.sim.motion.end - self.sim.motion.frame + 1
                if request.controller == "bmimic"
                else math.ceil(request.duration_s / 0.02)
            )
            self._prepared = True
            if self._pause_requested:
                self.sim.pause_execution()
        if time.monotonic() - self._feedback_time >= 0.2:
            goal.publish_feedback(
                ExecuteSimulationPolicy.Feedback(
                    state="running" if self.sim.execution_running else "paused",
                    completed_steps=self.sim.policy_steps - self._start_steps,
                    total_steps=self._total_steps,
                )
            )
            self._feedback_time = time.monotonic()

    def _on_command(self, msg):
        if self._fault is not None:
            return
        command = (msg.linear.x, msg.linear.y, msg.angular.z)
        components = (*command, msg.linear.z, msg.angular.x, msg.angular.y)
        bounds = ((-0.3, 0.5), (-0.3, 0.3), (-0.2, 0.2))
        if (
            not all(math.isfinite(value) for value in components)
            or any(not low <= value <= high for value, (low, high) in zip(command, bounds, strict=True))
            or any(value != 0.0 for value in (msg.linear.z, msg.angular.x, msg.angular.y))
        ):
            self._command = (0.0, 0.0, 0.0)
            self._command_time = None
            self.get_logger().warning("Rejected invalid /cmd_vel; requesting zero velocity")
            return
        self._command = command
        if (
            self.sim.controller != "mc_rl_velocity"
            or (self._goal is not None and self._goal.request.controller == "recover_stand")
            or not self._reserved
            or not self._prepared
            or not self.sim.execution_running
        ) and any(command):
            self._command = (0.0, 0.0, 0.0)
            self._command_time = None
            self.get_logger().warning("/cmd_vel requires an executing mc_rl_velocity goal")
            return
        if (
            any(command)
            and self.sim.session is None
            and (self.sim.policy_path is None or not self.sim.policy_path.is_file())
        ):
            self._command = (0.0, 0.0, 0.0)
            self._command_time = None
            self.get_logger().warning(f"{self.sim.preview_message}; command rejected")
            return
        self._command_time = time.monotonic()

    def _current_command(self):
        if self._fault is not None or self._command_time is None or time.monotonic() - self._command_time >= 0.5:
            return (0.0, 0.0, 0.0)
        return self._command

    def _on_stand(self, _request, response):
        self._pause_requested = True
        self._stable_steps = 0
        self._command = (0.0, 0.0, 0.0)
        self._command_time = None
        self.sim.pause_execution()
        response.success = self._fault is None
        response.message = (
            self._fault
            if self._fault is not None
            else "Physics paused, not active balance; /simulation/start resumes the current active goal"
        )
        return response

    def _on_start(self, _request, response):
        if self._fault is not None:
            response.success = False
            response.message = self._fault
            return response
        try:
            if not self._reserved or not self._prepared or self._done.done():
                raise ValueError("No active goal to resume; submit /simulation/execute_policy")
            self.sim.execution_running = True
            self._pause_requested = False
            self.sim.motion_running = self.sim.controller == "bmimic"
            response.success = True
            response.message = "Current execution resumed"
        except (ValueError, RuntimeError) as exc:
            response.success = False
            response.message = str(exc)
        return response

    def tick(self):
        if self._closing:
            return
        if self._fault is None:
            try:
                self._tick_goal()
                self.sim.step(self._current_command())
                recovering = self._goal is not None and self._goal.request.controller == "recover_stand"
                if recovering and self._prepared and self.sim.execution_running:
                    self._stable_steps = self._stable_steps + 1 if self.sim.is_standing() else 0
                    if self._stable_steps >= 50:
                        self._finish(True, "Standing recovery settled for 1 second; physics paused")
                if (
                    self._goal is not None
                    and not self._done.done()
                    and self._prepared
                    and self.sim.policy_steps - self._start_steps >= self._total_steps
                ):
                    self._finish(
                        not recovering,
                        "Standing recovery timed out; physics paused"
                        if recovering
                        else "Completed; physics paused, not active balance",
                    )
                self._publish_state()
            except RuntimeError as exc:
                self._fault = f"Physics fault (latched): {exc}"
                self._command = (0.0, 0.0, 0.0)
                self._command_time = None
                self._diagnostic_time = -math.inf
                self.get_logger().error(self._fault)
                self._finish(False, self._fault)
        if time.monotonic() - self._diagnostic_time >= 1.0:
            preview = not self.sim.execution_running or self.sim.session is None
            status = DiagnosticStatus(
                name="aimdk_mujoco/physics",
                hardware_id="simulation",
                level=DiagnosticStatus.ERROR
                if self._fault is not None
                else (DiagnosticStatus.WARN if preview else DiagnosticStatus.OK),
                message=self._fault
                if self._fault is not None
                else (
                    "Physics paused; submit execute_policy or resume active goal"
                    if preview
                    else "Simulation running; no hardware runtime claim"
                ),
            )
            diagnostics = DiagnosticArray(status=[status])
            if math.isfinite(self.sim.data.time):
                sec, nanosec = divmod(int(self.sim.data.time * 1_000_000_000), 1_000_000_000)
                diagnostics.header.stamp = Time(sec=sec, nanosec=nanosec)
            self._diagnostics_pub.publish(diagnostics)
            self._diagnostic_time = time.monotonic()

    def close(self):
        if self._action is None:
            return
        self._closing = True
        if rclpy.ok() and self._goal is not None:
            self._finish(False, "Simulation shutting down")
            # Completing the Future schedules execute/result delivery on the executor.
            executor = self.executor or rclpy.get_global_executor()
            added = executor.add_node(self)
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline:
                executor.spin_once(timeout_sec=0.01)
            if added:
                executor.remove_node(self)
        else:
            self.sim.pause_execution()
        if self._loading is not None:
            self._loading.cancel()
        self._loading = None
        if not self._loader.close():
            self.get_logger().warning(
                "Policy loader unfinished after 0.5s; abandoning daemon preparation without activation"
            )
        self._action.destroy()
        self._action = None

    def _publish_state(self):
        truth = self.sim.ground_truth()
        sec, nanosec = divmod(int(truth["time"] * 1_000_000_000), 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "pelvis"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, truth["position"])
        orientation = odom.pose.pose.orientation
        orientation.w, orientation.x, orientation.y, orientation.z = map(float, truth["quaternion_wxyz"])
        # MuJoCo xmat maps body to world; transpose it for Odometry's child-frame twist.
        rotation = self.sim.data.xmat[self.sim.pelvis_id]
        body_velocity = [
            sum(float(rotation[3 * row + col]) * float(truth["linear_velocity_world"][row]) for row in range(3))
            for col in range(3)
        ]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = body_velocity
        angular = odom.twist.twist.angular
        angular.x, angular.y, angular.z = map(float, truth["angular_velocity_body"])
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(self.sim.joint_names)
        joints.position = [float(self.sim.data.qpos[index]) for index in self.sim.qpos_indices]
        joints.velocity = [float(self.sim.data.qvel[index]) for index in self.sim.qvel_indices]
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "pelvis"
        imu.orientation = orientation
        imu.angular_velocity = angular
        imu.linear_acceleration_covariance[0] = -1.0  # No accelerometer measurement is provided.
        self._odom_pub.publish(odom)
        self._joints_pub.publish(joints)
        self._imu_pub.publish(imu)
        self._clock_pub.publish(Clock(clock=stamp))


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description="Standalone X2 MuJoCo standing/walking simulation")
    parser.add_argument("--assets", type=Path, required=True, help="Directory containing prepared assets")
    parser.add_argument("--config-dir", type=Path, help="Controller catalog directory; defaults to installed configs")
    parser.add_argument("--headless", action="store_true")
    options = parser.parse_args(remove_ros_args(args=["aimdk_mujoco", *argv])[1:])
    from aimdk_mujoco.physics import X2Simulation

    sim = X2Simulation(options.assets)
    sim.pause_execution()
    rclpy.init(args=argv)
    node = X2Node(sim, options.config_dir)
    viewer = None
    viewer_threads = set()
    try:
        if not options.headless:
            import mujoco.viewer

            before_viewer = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            # launch_passive owns daemon render threads but exposes no join handle.
            viewer_threads = set(threading.enumerate()) - before_viewer
        deadline = time.monotonic()
        while rclpy.ok() and (viewer is None or viewer.is_running()):
            rclpy.spin_once(node, timeout_sec=0)
            node.tick()
            if viewer is not None:
                # Display physics only: GUI reset/control widgets cannot change clock/state.
                viewer.sync(state_only=True)
            deadline += 0.02
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("MuJoCo render thread did not stop")
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
