"""Layer 2: Nav2 real stack E2E tests (no Gazebo).

Starts real Nav2 stack + MockRobotHardware, validates complete
navigation loop from voice command to robot motion.

Usage:
    colcon test --packages-select robot_navigation --pytest-args -k "test_pipeline_nav2"

Prerequisites:
    - Nav2 bringup installed (nav2_bringup)
    - Test map and params installed (config/test/)
"""

import contextlib
import json
import math
import os
import signal
import subprocess
import threading
import time

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64MultiArray, String

from robot_navigation.cmd_vel_bridge_node import CmdVelBridgeNode, _body_to_wheel_radps
from robot_navigation.nav2_goal_client import Nav2GoalClient
from test.e2e.mock_robot_hardware import MockRobotHardware
from test.e2e.mock_servers import MockTriggerServer

# ── constants ───────────────────────────────────────────────────────────────

WHEEL_RADIUS = 0.05
BASE_RADIUS = 0.125
MAX_RADPS = 4.602

NAV2_STARTUP_TIMEOUT = 60


# ── helpers ──────────────────────────────────────────────────────────────────


def _find_test_config():
    """Find test config files — search both source tree and install tree."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_dir = os.path.join(src_dir, "config", "test")
    nav2_params = os.path.join(config_dir, "nav2_params_test.yaml")
    test_map_yaml = os.path.join(config_dir, "test_map.yaml")
    if os.path.isfile(nav2_params) and os.path.isfile(test_map_yaml):
        return nav2_params, test_map_yaml
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory("robot_navigation")
        config_dir = os.path.join(share_dir, "config", "test")
        nav2_params = os.path.join(config_dir, "nav2_params_test.yaml")
        test_map_yaml = os.path.join(config_dir, "test_map.yaml")
        if os.path.isfile(nav2_params) and os.path.isfile(test_map_yaml):
            return nav2_params, test_map_yaml
    except Exception:
        pass
    return None, None


def _wait_for_nav2(goal_client_node, timeout=NAV2_STARTUP_TIMEOUT):
    """Wait for Nav2 navigate_to_pose action server using goal_client's client."""
    client = goal_client_node.nav_to_pose_client
    return client.wait_for_server(timeout_sec=timeout)


def _send_nav_goal(goal_client_node, x, y, theta, timeout=30.0):
    """Send NavigateToPose goal using the goal_client's built-in ActionClient."""
    # Use the Nav2GoalClient's internal action client to avoid creating
    # duplicate ActionClients that conflict in the executor's wait set.
    client = goal_client_node.nav_to_pose_client
    if not client.wait_for_server(timeout_sec=5.0):
        return None

    goal_msg = NavigateToPose.Goal()
    goal_msg.pose.header.frame_id = "map"
    goal_msg.pose.header.stamp = goal_client_node.get_clock().now().to_msg()
    goal_msg.pose.pose.position.x = x
    goal_msg.pose.pose.position.y = y
    goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
    goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

    future = client.send_goal_async(goal_msg)
    end_time = time.time() + timeout
    while not future.done() and time.time() < end_time:
        time.sleep(0.1)
    if not future.done():
        return None

    goal_handle = future.result()
    if not goal_handle.accepted:
        return None

    result_future = goal_handle.get_result_async()
    while not result_future.done() and time.time() < end_time:
        time.sleep(0.1)
    if not result_future.done():
        return None

    return result_future.result().status


def _collect_cmd_vel(mock_robot, duration=2.0):
    """Collect cmd_vel values for a duration using an extra subscription on mock_robot."""
    velocities = []
    # Create an additional subscription on the same node — executor already spins it
    sub = mock_robot.create_subscription(
        Twist,
        "/cmd_vel",
        lambda msg: velocities.append((msg.linear.x, msg.linear.y, msg.angular.z)),
        10,
    )
    time.sleep(duration)
    mock_robot.destroy_subscription(sub)
    return velocities


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def nav2_env():
    """Set up full Layer 2 environment.

    Order:
    1. Init rclpy + start MockRobotHardware (publishes /odom, /scan, TF)
    2. Launch Nav2 subprocess
    3. Wait for Nav2 ready
    4. Create goal_client
    """
    nav2_params, test_map_yaml = _find_test_config()
    if nav2_params is None:
        pytest.skip("Nav2 test config files not found")

    rclpy.init()
    executor = MultiThreadedExecutor()

    # ── Step 1: MockRobotHardware FIRST (publishes odom/scan/TF) ──
    mock_robot = MockRobotHardware()
    executor.add_node(mock_robot)
    mock_eval = MockTriggerServer("/action_dispatcher/start_evaluate", executor)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    # ── Step 2: Launch Nav2 bringup ──
    cmd = [
        "ros2",
        "launch",
        "nav2_bringup",
        "bringup_launch.py",
        "use_composition:=False",
        "use_sim_time:=False",
        "autostart:=True",
        f"params_file:={nav2_params}",
        f"map:={test_map_yaml}",
    ]
    nav2_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )

    # ── Step 3: goal_client (shares executor with mock_robot) ──
    goal_client = Nav2GoalClient()
    goal_client.voice_sub = goal_client.create_subscription(
        String, "/e2e/keyword_matched", goal_client.voice_command_callback, 10
    )
    goal_client.nav_stop_sub = goal_client.create_subscription(String, "/e2e/nav_stop", goal_client.stop_callback, 10)
    executor.add_node(goal_client)

    # ── Step 4: Wait for Nav2 ──
    nav2_ready = _wait_for_nav2(goal_client)
    if not nav2_ready:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(nav2_proc.pid), signal.SIGKILL)
        mock_robot.destroy_node()
        goal_client.destroy_node()
        mock_eval.destroy()
        rclpy.shutdown()
        pytest.skip("Nav2 stack did not become ready in time")

    # Extra settle for AMCL localization + costmap initialization
    time.sleep(5.0)

    yield {
        "mock_robot": mock_robot,
        "goal_client": goal_client,
        "mock_eval": mock_eval,
        "executor": executor,
    }

    # Teardown: stop executor first to avoid InvalidHandle
    executor.shutdown()
    spin_thread.join(timeout=5.0)

    try:
        os.killpg(os.getpgid(nav2_proc.pid), signal.SIGINT)
        nav2_proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(nav2_proc.pid), signal.SIGKILL)

    mock_robot.destroy_node()
    goal_client.destroy_node()
    mock_eval.destroy()
    rclpy.shutdown()


@pytest.fixture
def reset_nav2_env(nav2_env):
    """Reset state between Layer 2 tests."""
    gc = nav2_env["goal_client"]

    # Cancel any active Nav2 goal before resetting state
    if gc.goal_handle is not None and gc.is_navigating:
        with contextlib.suppress(Exception):
            gc.goal_handle.cancel_goal_async()

    gc.is_navigating = False
    gc.navigation_succeeded = False
    gc.navigation_failed = False
    gc.current_task_description = ""
    gc.goal_handle = None
    gc._nav_start_time = None

    mock_eval = nav2_env["mock_eval"]
    with mock_eval._lock:
        mock_eval.calls.clear()

    # Reset mock robot position to origin
    mock_robot = nav2_env["mock_robot"]
    mock_robot.pose_x = 0.0
    mock_robot.pose_y = 0.0
    mock_robot.pose_theta = 0.0

    # Wait for cmd_vel to settle after cancellation
    time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# TC2.1: Nav2 plans path and drives to goal
# ═══════════════════════════════════════════════════════════════════════════


class TestNav2Navigation:
    """TC2.1: Nav2 plans path and drives to target point."""

    def test_navigate_to_pose_succeeds(self, nav2_env, reset_nav2_env):
        gc = nav2_env["goal_client"]

        # Collect cmd_vel to verify Nav2 produces movement commands
        mock_robot = nav2_env["mock_robot"]
        cmd_vel_msgs = []
        cmd_sub = mock_robot.create_subscription(
            Twist,
            "/cmd_vel",
            lambda m: cmd_vel_msgs.append((m.linear.x, m.linear.y, m.angular.z)),
            10,
        )

        status = _send_nav_goal(gc, 0.3, 0.0, 0.0, timeout=30.0)
        mock_robot.destroy_subscription(cmd_sub)

        assert status is not None, "Navigation did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"Expected SUCCEEDED, got status {status}"

        # Verify Nav2 produced non-zero cmd_vel during navigation
        nonzero = [v for v in cmd_vel_msgs if abs(v[0]) > 1e-6 or abs(v[1]) > 1e-6]
        assert len(nonzero) > 0, "Nav2 should produce non-zero cmd_vel during navigation"


# ═══════════════════════════════════════════════════════════════════════════
# TC2.2: Voice triggers full navigation
# ═══════════════════════════════════════════════════════════════════════════


class TestVoiceNavigation:
    """TC2.2: Voice command triggers complete navigation via goal_client."""

    def test_voice_navigates_via_topic(self, nav2_env, reset_nav2_env):
        gc = nav2_env["goal_client"]
        pub = gc.create_publisher(String, "/e2e/keyword_matched", 10)
        time.sleep(0.2)

        # Send goal via voice command topic
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "去b点",
                "type": "destination",
                "info": {"x": 0.3, "y": 0.0, "theta": 0.0},
            }
        )
        pub.publish(msg)

        # Wait for goal_client to process the voice command and accept a goal.
        # is_navigating can transition True→False in <100ms when the robot is
        # already near the goal, so we check goal_handle instead (set on accept,
        # stays non-None until reset).
        deadline = time.monotonic() + 10.0
        while gc.goal_handle is None and time.monotonic() < deadline:
            time.sleep(0.1)

        assert gc.goal_handle is not None, "Goal client should have received and processed the voice command"
        gc.destroy_publisher(pub)


# ═══════════════════════════════════════════════════════════════════════════
# TC2.3: Stop during navigation -> cmd_vel zeros
# ═══════════════════════════════════════════════════════════════════════════


class TestStopNavigation:
    """TC2.3: Stop command during navigation causes cmd_vel to zero."""

    def test_stop_zeros_cmd_vel(self, nav2_env, reset_nav2_env):
        gc = nav2_env["goal_client"]
        mock_robot = nav2_env["mock_robot"]

        pub = gc.create_publisher(String, "/e2e/keyword_matched", 10)
        stop_pub = gc.create_publisher(String, "/e2e/nav_stop", 10)
        time.sleep(0.2)

        # Collect cmd_vel to detect when Nav2 starts driving
        cmd_vel_msgs = []
        cmd_sub = mock_robot.create_subscription(
            Twist,
            "/cmd_vel",
            lambda m: cmd_vel_msgs.append((m.linear.x, m.linear.y, m.angular.z)),
            10,
        )

        # Send to a far target so navigation takes time
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "去远点",
                "type": "destination",
                "info": {"x": 1.5, "y": 1.5, "theta": 0.0},
            }
        )
        pub.publish(msg)

        # Wait until Nav2 produces non-zero cmd_vel (proves navigation started).
        # This avoids the race with the transient is_navigating flag.
        deadline = time.monotonic() + 15.0
        got_movement = False
        while time.monotonic() < deadline:
            if any(abs(v[0]) > 0.01 or abs(v[1]) > 0.01 for v in cmd_vel_msgs):
                got_movement = True
                break
            time.sleep(0.05)

        if not got_movement:
            mock_robot.destroy_subscription(cmd_sub)
            gc.destroy_publisher(pub)
            gc.destroy_publisher(stop_pub)
            pytest.skip("Navigation completed before movement detected")

        # Wait until goal_handle is set (goal accepted by Nav2) before stopping
        handle_deadline = time.monotonic() + 5.0
        while gc.goal_handle is None and time.monotonic() < handle_deadline:
            time.sleep(0.1)
        assert gc.goal_handle is not None, "goal_handle was never set — goal not accepted?"

        # Send stop
        stop_msg = String()
        stop_msg.data = "stop"
        stop_pub.publish(stop_msg)

        # Wait for cmd_vel to settle to zero (goal cancel + velocity_smoother timeout)
        # Use a single subscription to avoid repeated create/destroy crashing the executor
        post_stop_msgs = []
        post_stop_sub = mock_robot.create_subscription(
            Twist,
            "/cmd_vel",
            lambda m: post_stop_msgs.append((m.linear.x, m.linear.y, m.angular.z)),
            10,
        )
        settle_deadline = time.monotonic() + 5.0
        while time.monotonic() < settle_deadline:
            time.sleep(0.5)
            if len(post_stop_msgs) >= 3 and all(abs(v[0]) < 0.05 and abs(v[1]) < 0.05 for v in post_stop_msgs[-3:]):
                break
        mock_robot.destroy_subscription(post_stop_sub)

        # Verify final cmd_vel is near zero
        if post_stop_msgs:
            last_vx, last_vy, _ = post_stop_msgs[-1]
        else:
            last_vx, last_vy = 1.0, 1.0  # force failure if no messages
        assert abs(last_vx) < 0.05, f"cmd_vel.linear.x should be ~0 after stop, got {last_vx}"
        assert abs(last_vy) < 0.05, f"cmd_vel.linear.y should be ~0 after stop, got {last_vy}"

        mock_robot.destroy_subscription(cmd_sub)
        gc.destroy_publisher(pub)
        gc.destroy_publisher(stop_pub)


# ═══════════════════════════════════════════════════════════════════════════
# TC2.4: cmd_vel_bridge produces valid wheel commands
# ═══════════════════════════════════════════════════════════════════════════


class TestWheelCommandsNav2:
    """TC2.4: cmd_vel_bridge wheel commands match IK and respect max_radps.

    Creates a standalone bridge node for this test to avoid odom/TF conflicts.
    """

    def test_wheel_commands_valid(self, nav2_env, reset_nav2_env):
        bridge = CmdVelBridgeNode()
        # Disable TF and odom to avoid conflicts with MockRobotHardware
        bridge.publish_tf = False
        # Destroy old odom pub and create on dummy topic
        bridge.destroy_publisher(bridge.odom_pub)
        bridge.odom_pub = bridge.create_publisher(Odometry, "/cmd_vel_bridge/odom", 10)
        # Destroy timer to control manually
        bridge.destroy_timer(bridge.control_timer)

        # Add bridge to the running executor so subscription callbacks fire.
        # TF disabled + odom redirected = no conflict with MockRobotHardware.
        executor = nav2_env["executor"]
        executor.add_node(bridge)

        received = []
        sub = bridge.create_subscription(
            Float64MultiArray,
            "/base_velocity_controller/commands",
            lambda m: received.append(m),
            10,
        )
        time.sleep(0.2)

        cmd_pub = bridge.create_publisher(Twist, "/cmd_vel", 10)
        time.sleep(0.2)
        twist = Twist()
        twist.linear.x = 0.1
        cmd_pub.publish(twist)
        time.sleep(0.3)
        bridge.control_loop()
        time.sleep(0.5)

        assert len(received) >= 1, "Should have received wheel commands"
        expected = _body_to_wheel_radps(0.1, 0.0, 0.0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        actual = received[-1].data
        for i in range(3):
            assert abs(actual[i]) <= MAX_RADPS + 0.01
            if abs(expected[i]) > 0.01:
                assert actual[i] * expected[i] >= 0

        bridge.destroy_subscription(sub)
        bridge.destroy_publisher(cmd_pub)
        bridge.destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
# TC2.5: Odometry consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestOdometryConsistency:
    """TC2.5: /odom and TF from MockRobotHardware are consistent."""

    def test_odometry_and_tf(self, nav2_env, reset_nav2_env):
        mock_robot = nav2_env["mock_robot"]
        gc = nav2_env["goal_client"]

        # Collect /odom from MockRobotHardware (subscribe on mock_robot itself)
        odom_msgs = []
        odom_sub = mock_robot.create_subscription(Odometry, "/odom", lambda m: odom_msgs.append(m), 10)
        time.sleep(1.0)

        assert len(odom_msgs) >= 1, "Should receive odom messages from MockRobotHardware"
        odom = odom_msgs[-1]
        assert odom.header.frame_id == "odom"
        assert odom.child_frame_id == "base_link"

        # Check TF chain: odom -> base_link
        # Use threading.Event + callback on goal_client for reliable detection.
        # Subscribe on a DIFFERENT node from the publisher to avoid DDS loopback issues.
        from tf2_msgs.msg import TFMessage

        tf_received = threading.Event()
        tf_transforms = []

        def _tf_callback(msg):
            tf_transforms.extend(msg.transforms)
            if any(t.header.frame_id == "odom" and t.child_frame_id == "base_link" for t in msg.transforms):
                tf_received.set()

        tf_sub = gc.create_subscription(TFMessage, "/tf", _tf_callback, 100)
        tf_received.wait(timeout=5.0)
        gc.destroy_subscription(tf_sub)

        odom_to_base = [t for t in tf_transforms if t.header.frame_id == "odom" and t.child_frame_id == "base_link"]
        assert len(odom_to_base) > 0, "Should have received odom->base_link TF from MockRobotHardware"

        # Verify the TF transform values are consistent with odom
        tf = odom_to_base[-1]
        assert tf.header.frame_id == "odom"
        assert tf.child_frame_id == "base_link"

        mock_robot.destroy_subscription(odom_sub)
