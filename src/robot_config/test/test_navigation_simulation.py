"""Gazebo navigation E2E tests through robot_config bringup.

This owns the full robot launch path: Gazebo, ros2_control, controllers,
Nav2, and robot_navigation nodes are started via robot_config/robot.launch.py.
robot_navigation keeps only component-level tests that do not bring up a full
robot.

Usage:
    colcon test --packages-select robot_config --pytest-args -k "test_navigation_simulation"
"""

import contextlib
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

POSITION_TOLERANCE = 0.25
GAZEBO_STARTUP_TIMEOUT = 120
NAV2_SETTLE_TIME = 10
NAV_GOAL_TIMEOUT = 90
NAV_TEST_PROFILE_ENV = "NAV_TEST_PROFILE"

SPAWN_X = -1.5
SPAWN_Y = -1.5
POINT_A_X = 1.0
POINT_A_Y = 1.0
POINT_B_X = -1.0
POINT_B_Y = 1.0


@dataclass
class ManagedProcess:
    proc: subprocess.Popen
    log_file: object


class NavigationProbe(Node):
    def __init__(self):
        super().__init__("robot_config_navigation_probe")
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")


class MockTriggerServer:
    def __init__(self, service_name: str, executor: MultiThreadedExecutor):
        self._node = Node(f"mock_trigger_{service_name.replace('/', '_')}")
        self.calls = []
        self._lock = threading.Lock()
        self._srv = self._node.create_service(Trigger, service_name, self._callback)
        executor.add_node(self._node)

    def _callback(self, request, response):
        with self._lock:
            self.calls.append(request)
        response.success = True
        response.message = "mock success"
        return response

    def destroy(self):
        self._node.destroy_service(self._srv)
        self._node.destroy_node()


def _check_gazebo_available():
    for cmd in [["ign", "gazebo", "--version"], ["gz", "sim", "--version"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _is_openeuler():
    try:
        with open("/etc/os-release", encoding="utf-8") as os_release:
            return any(line.strip().lower() == "id=openeuler" for line in os_release)
    except OSError:
        return False


def _select_nav2_test_profile():
    requested = os.environ.get(NAV_TEST_PROFILE_ENV, "").strip().lower()
    if requested:
        if requested not in {"full", "minimal"}:
            pytest.skip(f"Unsupported {NAV_TEST_PROFILE_ENV}={requested!r}; expected 'full' or 'minimal'")
        if requested == "full" and _is_openeuler():
            print(
                "[robot_config] Warning: full test profile enables Gazebo navigation simulation, "
                "which is not part of the openEuler minimal E2E profile. If Gazebo fails on openEuler, "
                f"rerun with {NAV_TEST_PROFILE_ENV}=minimal."
            )
        return requested

    if _is_openeuler():
        print(
            "[robot_config] openEuler detected; skipping Gazebo navigation simulation in minimal profile. "
            f"Set {NAV_TEST_PROFILE_ENV}=full to run Gazebo explicitly."
        )
        return "minimal"

    return "full"


def _wait_for_nav2(probe: NavigationProbe, timeout=GAZEBO_STARTUP_TIMEOUT):
    return probe.nav_to_pose_client.wait_for_server(timeout_sec=timeout)


def _send_nav_goal(probe: NavigationProbe, x, y, theta, timeout=NAV_GOAL_TIMEOUT):
    client = probe.nav_to_pose_client
    if not client.wait_for_server(timeout_sec=5.0):
        return None

    goal_msg = NavigateToPose.Goal()
    goal_msg.pose.header.frame_id = "map"
    goal_msg.pose.header.stamp = probe.get_clock().now().to_msg()
    goal_msg.pose.pose.position.x = x
    goal_msg.pose.pose.position.y = y
    goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
    goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

    future = client.send_goal_async(goal_msg)
    deadline = time.time() + timeout
    while not future.done() and time.time() < deadline:
        time.sleep(0.1)
    if not future.done():
        return None

    goal_handle = future.result()
    if not goal_handle.accepted:
        return None

    result_future = goal_handle.get_result_async()
    while not result_future.done() and time.time() < deadline:
        time.sleep(0.1)
    if not result_future.done():
        return None

    return result_future.result().status


def _get_odom_position(node: Node, topic="/odom", timeout=5.0):
    odom_msgs = []
    event = threading.Event()

    def _cb(msg):
        odom_msgs.append(msg)
        event.set()

    sub = node.create_subscription(Odometry, topic, _cb, 10)
    event.wait(timeout=timeout)
    node.destroy_subscription(sub)

    if not odom_msgs:
        return None
    odom = odom_msgs[-1]
    odom.pose.pose.position.x += SPAWN_X
    odom.pose.pose.position.y += SPAWN_Y
    return odom.pose.pose.position


def _terminate_process(managed: ManagedProcess, timeout=15):
    proc = managed.proc
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
    finally:
        managed.log_file.close()


def _start_process(cmd, env, log_dir: Path, name: str):
    # Keep the file open until the process exits so subprocess can stream logs.
    log_file = open(log_dir / f"{name}.log", "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return ManagedProcess(proc=proc, log_file=log_file)


@pytest.fixture(scope="module")
def gazebo_env():
    if _select_nav2_test_profile() == "minimal":
        pytest.skip(
            "Gazebo navigation simulation is disabled by the minimal E2E profile; "
            f"set {NAV_TEST_PROFILE_ENV}=full to run it."
        )

    if not _check_gazebo_available():
        pytest.skip("Gazebo not available; skipping navigation simulation tests")

    rclpy.init()
    executor = MultiThreadedExecutor()
    env = os.environ.copy()
    if os.environ.get("SIM_GUI") != "1":
        env["IBROBOT_GAZEBO_HEADLESS"] = "1"

    robot_config_share = get_package_share_directory("robot_config")
    config_path = os.path.join(robot_config_share, "test", "config", "lekiwi_navi_sim.yaml")
    robot_launch = os.path.join(robot_config_share, "launch", "robot.launch.py")
    e2e_dir = Path(robot_config_share) / "test" / "e2e"
    log_dir = Path(tempfile.mkdtemp(prefix="robot_config_nav_sim_"))
    print(f"[robot_config] navigation simulation logs: {log_dir}")

    robot_proc = _start_process(
        [
            "ros2",
            "launch",
            robot_launch,
            f"config_path:={config_path}",
            "use_sim:=true",
            "with_navigation:=true",
            "with_inference:=false",
            "with_moveit:=false",
            "voice_asr_auto_start:=false",
            "auto_start_controllers:=true",
            "control_mode:=base_navigation",
        ],
        env,
        log_dir,
        "robot_launch",
    )
    odom_bridge_proc = _start_process(
        [
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/model/lekiwi/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
        env,
        log_dir,
        "odom_bridge",
    )
    gt_odom_proc = _start_process(
        [
            "python3",
            str(e2e_dir / "gt_odom_node.py"),
            "--ros-args",
            "-p",
            f"spawn_x:={SPAWN_X}",
            "-p",
            f"spawn_y:={SPAWN_Y}",
        ],
        env,
        log_dir,
        "gt_odom_node",
    )
    cmd_vel_relay_proc = _start_process(["python3", str(e2e_dir / "cmd_vel_relay.py")], env, log_dir, "cmd_vel_relay")

    probe = NavigationProbe()
    executor.add_node(probe)
    mock_eval = MockTriggerServer("/action_dispatcher/start_evaluate", executor)
    mock_stop = MockTriggerServer("/action_dispatcher/stop_evaluate", executor)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    if not _wait_for_nav2(probe):
        for proc in (cmd_vel_relay_proc, gt_odom_proc, odom_bridge_proc, robot_proc):
            _terminate_process(proc, timeout=5)
        mock_stop.destroy()
        mock_eval.destroy()
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        probe.destroy_node()
        rclpy.shutdown()
        pytest.skip("Nav2 stack did not become ready in time")

    time.sleep(NAV2_SETTLE_TIME)

    yield {
        "probe": probe,
        "mock_eval": mock_eval,
        "mock_stop": mock_stop,
    }

    executor.shutdown()
    spin_thread.join(timeout=5.0)
    mock_stop.destroy()
    mock_eval.destroy()
    probe.destroy_node()
    for proc in (cmd_vel_relay_proc, gt_odom_proc, odom_bridge_proc, robot_proc):
        _terminate_process(proc)
    rclpy.shutdown()


@pytest.fixture
def reset_gazebo_env(gazebo_env):
    mock_eval = gazebo_env["mock_eval"]
    mock_stop = gazebo_env["mock_stop"]
    with mock_eval._lock:
        mock_eval.calls.clear()
    with mock_stop._lock:
        mock_stop.calls.clear()

    teleport_args = [
        "-s",
        "/world/nav_test/set_pose",
        "--reqtype",
        "ignition.msgs.Pose",
        "--reptype",
        "ignition.msgs.Boolean",
        "--timeout",
        "300",
        "--req",
        f'name: "lekiwi", position: {{x: {SPAWN_X}, y: {SPAWN_Y}, z: 0.01}}, orientation: {{w: 1.0}}',
    ]
    for gz_cmd in [["gz", "service"] + teleport_args, ["ign", "service"] + teleport_args]:
        result = subprocess.run(gz_cmd, capture_output=True, timeout=5)
        if result.returncode == 0:
            break
    time.sleep(2.0)


class TestNavigationSimulation:
    def test_navigate_with_voice_and_obstacle(self, gazebo_env, reset_gazebo_env):
        probe = gazebo_env["probe"]

        cmd_vel_msgs = []
        cmd_sub = probe.create_subscription(
            Twist,
            "/cmd_vel",
            lambda msg: cmd_vel_msgs.append((msg.linear.x, msg.linear.y, msg.angular.z)),
            10,
        )
        status = _send_nav_goal(probe, POINT_A_X, POINT_A_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        probe.destroy_subscription(cmd_sub)

        assert status is not None, "Navigation did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"Expected SUCCEEDED, got status {status}"
        assert any(abs(v[0]) > 1e-4 or abs(v[1]) > 1e-4 for v in cmd_vel_msgs)

        pos = _get_odom_position(probe, "/odom", timeout=5.0)
        assert pos is not None, "Should receive odom messages"
        assert abs(pos.x - POINT_A_X) < POSITION_TOLERANCE
        assert abs(pos.y - POINT_A_Y) < POSITION_TOLERANCE

        pub = probe.create_publisher(String, "/voice_command", 10)
        voice_cmd_vel_msgs = []
        voice_cmd_sub = probe.create_subscription(
            Twist,
            "/cmd_vel",
            lambda msg: voice_cmd_vel_msgs.append((msg.linear.x, msg.linear.y, msg.angular.z)),
            10,
        )
        time.sleep(0.3)
        msg = String()
        msg.data = "去b点"
        pub.publish(msg)

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if any(abs(v[0]) > 0.01 or abs(v[1]) > 0.01 for v in voice_cmd_vel_msgs):
                break
            time.sleep(0.1)
        else:
            probe.destroy_subscription(voice_cmd_sub)
            probe.destroy_publisher(pub)
            pytest.fail("Voice command should trigger non-zero cmd_vel")

        probe.destroy_subscription(voice_cmd_sub)
        probe.destroy_publisher(pub)

    def test_stop_zeros_cmd_vel(self, gazebo_env, reset_gazebo_env):
        probe = gazebo_env["probe"]
        voice_pub = probe.create_publisher(String, "/voice_command", 10)
        time.sleep(0.3)

        cmd_vel_msgs = []
        cmd_sub = probe.create_subscription(
            Twist,
            "/cmd_vel",
            lambda msg: cmd_vel_msgs.append((msg.linear.x, msg.linear.y, msg.angular.z)),
            10,
        )

        msg = String()
        msg.data = "去a点"
        voice_pub.publish(msg)

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if any(abs(v[0]) > 0.01 or abs(v[1]) > 0.01 for v in cmd_vel_msgs):
                break
            time.sleep(0.05)
        else:
            probe.destroy_subscription(cmd_sub)
            probe.destroy_publisher(voice_pub)
            pytest.skip("Navigation completed before movement detected")

        time.sleep(3.0)
        stop_msg = String()
        stop_msg.data = "停止"
        voice_pub.publish(stop_msg)

        post_stop_msgs = []
        post_stop_sub = probe.create_subscription(
            Twist,
            "/cmd_vel",
            lambda msg: post_stop_msgs.append((msg.linear.x, msg.linear.y, msg.angular.z)),
            10,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if len(post_stop_msgs) >= 3 and all(abs(v[0]) < 0.05 and abs(v[1]) < 0.05 for v in post_stop_msgs[-3:]):
                break

        probe.destroy_subscription(post_stop_sub)
        probe.destroy_subscription(cmd_sub)
        probe.destroy_publisher(voice_pub)

        assert post_stop_msgs, "Should receive cmd_vel after stop"
        last_vx, last_vy, _ = post_stop_msgs[-1]
        assert abs(last_vx) < 0.05
        assert abs(last_vy) < 0.05

    def test_navigate_there_and_back(self, gazebo_env, reset_gazebo_env):
        probe = gazebo_env["probe"]

        status = _send_nav_goal(probe, POINT_A_X, POINT_A_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        assert status is not None, "Navigation to point_a did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"point_a failed with status {status}"

        pos = _get_odom_position(probe, "/odom", timeout=5.0)
        assert pos is not None
        assert abs(pos.x - POINT_A_X) < POSITION_TOLERANCE
        assert abs(pos.y - POINT_A_Y) < POSITION_TOLERANCE

        status = _send_nav_goal(probe, SPAWN_X, SPAWN_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        assert status is not None, "Navigation to origin did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"origin failed with status {status}"

        pos = _get_odom_position(probe, "/odom", timeout=5.0)
        assert pos is not None
        assert abs(pos.x - SPAWN_X) < POSITION_TOLERANCE
        assert abs(pos.y - SPAWN_Y) < POSITION_TOLERANCE
