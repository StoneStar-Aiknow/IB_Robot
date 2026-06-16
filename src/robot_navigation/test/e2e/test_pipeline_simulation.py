"""Layer 3: Gazebo simulation E2E tests.

Starts Gazebo Fortress + Nav2 + OmniWheelController, validates complete
navigation loop with DART physics, CPU lidar, and static TF localization.

Localization: static map->odom TF at spawn position (no AMCL).
Odometry: Gazebo OdometryPublisher ground truth (not wheel odometry).
Lidar: CPU lidar (avoids gpu_lidar EGL failure on hybrid GPU).

Usage:
    colcon test --packages-select robot_navigation --pytest-args -k "test_pipeline_simulation"

Prerequisites:
    - Gazebo Fortress + ros_gz_sim + ros_gz_bridge installed
    - nav_test.world installed (robot_config)
    - lekiwi_description URDF installed
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
from std_msgs.msg import String

from robot_navigation.nav2_goal_client import Nav2GoalClient
from test.e2e.mock_servers import MockTriggerServer

# ── constants ───────────────────────────────────────────────────────────────

POSITION_TOLERANCE = 0.25  # 25cm tolerance for Gazebo physics simulation
GAZEBO_STARTUP_TIMEOUT = 120  # seconds — Gazebo + spawn + controllers + Nav2
NAV2_SETTLE_TIME = 10  # seconds — let Nav2 stack settle after startup
NAV_GOAL_TIMEOUT = 90  # seconds — per-goal navigation timeout (Gazebo physics is slow)
NAV_TEST_PROFILE_ENV = "NAV_TEST_PROFILE"

SPAWN_X = -1.5
SPAWN_Y = -1.5
POINT_A_X = 1.0
POINT_A_Y = 1.0
POINT_B_X = -1.0
POINT_B_Y = 1.0


# ── helpers ──────────────────────────────────────────────────────────────────


def _check_gazebo_available():
    """Skip tests if Gazebo is not available."""
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
        with open("/etc/os-release") as os_release:
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
                "[robot_navigation] Warning: full test profile enables Gazebo Layer 3 simulation, "
                "which is not part of the openEuler minimal E2E profile. If Gazebo fails on openEuler, "
                f"rerun with {NAV_TEST_PROFILE_ENV}=minimal."
            )
        return requested

    if _is_openeuler():
        print(
            "[robot_navigation] openEuler detected; skipping Gazebo Layer 3 simulation in minimal profile. "
            f"Set {NAV_TEST_PROFILE_ENV}=full to run Gazebo explicitly."
        )
        return "minimal"

    return "full"


def _wait_for_nav2(goal_client_node, timeout=GAZEBO_STARTUP_TIMEOUT):
    """Wait for Nav2 navigate_to_pose action server."""
    client = goal_client_node.nav_to_pose_client
    return client.wait_for_server(timeout_sec=timeout)


def _send_nav_goal(goal_client_node, x, y, theta, timeout=NAV_GOAL_TIMEOUT):
    """Send NavigateToPose goal and wait for result."""
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


def _get_odom_position(node, topic="/odom", timeout=5.0):
    """Get the latest position in map frame from ground truth odometry.

    gt_odom_node publishes odom-relative coordinates (spawn = 0,0).
    Convert to map frame: map_pos = odom_pos + spawn_offset.
    """
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
    # Convert odom frame to map frame
    odom.pose.pose.position.x += SPAWN_X
    odom.pose.pose.position.y += SPAWN_Y
    return odom.pose.pose.position


def _collect_cmd_vel(node, topic="/cmd_vel", duration=3.0):
    """Collect cmd_vel messages for a duration."""
    velocities = []
    sub = node.create_subscription(
        Twist,
        topic,
        lambda msg: velocities.append((msg.linear.x, msg.linear.y, msg.angular.z)),
        10,
    )
    time.sleep(duration)
    node.destroy_subscription(sub)
    return velocities


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gazebo_env():
    """Layer 3 environment: Gazebo + Nav2 + in-process goal_client."""
    if _select_nav2_test_profile() == "minimal":
        pytest.skip(
            "Gazebo Layer 3 simulation is disabled by the minimal E2E profile; "
            f"set {NAV_TEST_PROFILE_ENV}=full to run it."
        )

    if not _check_gazebo_available():
        pytest.skip("Gazebo not available — skipping Layer 3 tests")

    rclpy.init()
    executor = MultiThreadedExecutor()

    # ── Step 1: Launch Gazebo + Nav2 via sim_e2e.launch.py ──
    gui_args = ["gui:=true"] if os.environ.get("SIM_GUI") == "1" else []
    cmd = ["ros2", "launch", "robot_navigation", "sim_e2e.launch.py"] + gui_args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )

    # ── Step 2: Create Nav2GoalClient + MockTriggerServer ──
    goal_client = Nav2GoalClient()
    goal_client.voice_sub = goal_client.create_subscription(
        String, "/e2e/keyword_matched", goal_client.voice_command_callback, 10
    )
    goal_client.nav_stop_sub = goal_client.create_subscription(String, "/e2e/nav_stop", goal_client.stop_callback, 10)
    executor.add_node(goal_client)

    mock_eval = MockTriggerServer("/action_dispatcher/start_evaluate", executor)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    # ── Step 3: Wait for Nav2 ready ──
    nav2_ready = _wait_for_nav2(goal_client)
    if not nav2_ready:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        goal_client.destroy_node()
        mock_eval.destroy()
        rclpy.shutdown()
        pytest.skip("Nav2 stack did not become ready in time (Gazebo may have failed to start)")

    # ── Step 4: Let Nav2 settle ──
    time.sleep(NAV2_SETTLE_TIME)

    yield {
        "goal_client": goal_client,
        "mock_eval": mock_eval,
        "executor": executor,
        "proc": proc,
    }

    # Teardown
    executor.shutdown()
    spin_thread.join(timeout=5.0)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    goal_client.destroy_node()
    mock_eval.destroy()
    rclpy.shutdown()


@pytest.fixture
def reset_gazebo_env(gazebo_env):
    """Reset goal_client + mock_eval state and teleport robot back to spawn."""
    gc = gazebo_env["goal_client"]

    # Reset software state
    gc.is_navigating = False
    gc.navigation_succeeded = False
    gc.navigation_failed = False
    gc.current_task_description = ""
    gc.goal_handle = None
    gc._nav_start_time = None

    mock_eval = gazebo_env["mock_eval"]
    with mock_eval._lock:
        mock_eval.calls.clear()

    # Teleport robot back to spawn position via Gazebo set_pose service
    # Try 'gz service' first (Gazebo Sim >= 7), fall back to 'ign service' (Fortress)
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


# ═══════════════════════════════════════════════════════════════════════════
# TC3.1+3.2+3.3 (merged): Navigate + voice command + obstacle avoidance
# ═══════════════════════════════════════════════════════════════════════════


class TestNavigationE2E:
    """Merged TC3.1+3.2+3.3: Navigate to point_a (via obstacle), verify
    cmd_vel + position, then verify voice command triggers navigation.

    Merging avoids 2 extra teleport + 2 extra navigation cycles (~2 min saved).
    Obstacle avoidance is implicitly verified: the robot spawns at (-1.5, -1.5)
    and must navigate around the mapped obstacle at (0, 0) to reach point_a.
    """

    def test_navigate_with_voice_and_obstacle(self, gazebo_env, reset_gazebo_env):
        gc = gazebo_env["goal_client"]

        # ── Phase 1: NavigateToPose → point_a (TC3.1 + TC3.3) ──

        # Subscribe to cmd_vel BEFORE navigation to capture all commands
        cmd_vel_msgs = []
        cmd_sub = gc.create_subscription(
            Twist,
            "/cmd_vel",
            lambda m: cmd_vel_msgs.append((m.linear.x, m.linear.y, m.angular.z)),
            10,
        )

        # Send navigation goal to point_a (path crosses obstacle at (0,0))
        status = _send_nav_goal(gc, POINT_A_X, POINT_A_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        gc.destroy_subscription(cmd_sub)

        assert status is not None, "Navigation did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"Expected SUCCEEDED, got status {status}"

        # Verify that non-zero cmd_vel was produced during navigation
        nonzero = [v for v in cmd_vel_msgs if abs(v[0]) > 1e-4 or abs(v[1]) > 1e-4]
        assert len(nonzero) > 0, "Nav2 should produce non-zero cmd_vel during navigation"

        # Verify final position via ground truth odometry
        pos = _get_odom_position(gc, "/odom", timeout=5.0)
        assert pos is not None, "Should receive odom messages"
        assert abs(pos.x - POINT_A_X) < POSITION_TOLERANCE, (
            f"Final x={pos.x:.3f}, expected {POINT_A_X} ± {POSITION_TOLERANCE}"
        )
        assert abs(pos.y - POINT_A_Y) < POSITION_TOLERANCE, (
            f"Final y={pos.y:.3f}, expected {POINT_A_Y} ± {POSITION_TOLERANCE}"
        )

        # ── Phase 2: Voice command triggers navigation (TC3.2) ──

        # Reset goal_handle so we can detect the new voice-triggered goal
        gc.goal_handle = None

        pub = gc.create_publisher(String, "/e2e/keyword_matched", 10)
        time.sleep(0.3)

        # Send voice command to navigate to point_b
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "去b点",
                "type": "destination",
                "info": {"x": POINT_B_X, "y": POINT_B_Y, "theta": 0.0},
            }
        )
        pub.publish(msg)

        # Wait for goal to be accepted
        deadline = time.monotonic() + 15.0
        while gc.goal_handle is None and time.monotonic() < deadline:
            time.sleep(0.1)

        assert gc.goal_handle is not None, "Goal client should accept the voice command"

        # Cancel the voice-triggered navigation — we only needed to verify acceptance
        gc.goal_handle.cancel_goal_async()
        time.sleep(0.5)

        gc.destroy_publisher(pub)


# ═══════════════════════════════════════════════════════════════════════════
# TC3.4: Stop during navigation
# ═══════════════════════════════════════════════════════════════════════════


class TestStopNavigation:
    """TC3.4: Stop command during navigation causes cmd_vel to zero."""

    def test_stop_zeros_cmd_vel(self, gazebo_env, reset_gazebo_env):
        gc = gazebo_env["goal_client"]
        pub = gc.create_publisher(String, "/e2e/keyword_matched", 10)
        stop_pub = gc.create_publisher(String, "/e2e/nav_stop", 10)
        time.sleep(0.3)

        # Collect cmd_vel to detect when navigation starts
        cmd_vel_msgs = []
        cmd_sub = gc.create_subscription(
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
                "info": {"x": 1.0, "y": -1.0, "theta": 0.0},
            }
        )
        pub.publish(msg)

        # Wait until Nav2 produces non-zero cmd_vel
        deadline = time.monotonic() + 15.0
        got_movement = False
        while time.monotonic() < deadline:
            if any(abs(v[0]) > 0.01 or abs(v[1]) > 0.01 for v in cmd_vel_msgs):
                got_movement = True
                break
            time.sleep(0.05)

        if not got_movement:
            gc.destroy_subscription(cmd_sub)
            gc.destroy_publisher(pub)
            gc.destroy_publisher(stop_pub)
            pytest.skip("Navigation completed before movement detected")

        # Let navigation run for a bit before stopping
        time.sleep(3.0)

        # Send stop
        stop_msg = String()
        stop_msg.data = "stop"
        stop_pub.publish(stop_msg)

        # Wait for cmd_vel to settle to zero (goal cancel + velocity_smoother timeout)
        post_stop_msgs = []
        post_stop_sub = gc.create_subscription(
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
        gc.destroy_subscription(post_stop_sub)

        # Verify final cmd_vel is near zero
        if post_stop_msgs:
            last_vx, last_vy, _ = post_stop_msgs[-1]
        else:
            last_vx, last_vy = 1.0, 1.0  # force failure if no messages
        assert abs(last_vx) < 0.05, f"cmd_vel.linear.x should be ~0 after stop, got {last_vx}"
        assert abs(last_vy) < 0.05, f"cmd_vel.linear.y should be ~0 after stop, got {last_vy}"

        assert gc.is_navigating is False, "is_navigating should be False after stop"

        gc.destroy_subscription(cmd_sub)
        gc.destroy_publisher(pub)
        gc.destroy_publisher(stop_pub)


# ═══════════════════════════════════════════════════════════════════════════
# TC3.5: Sequential multi-point navigation
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiPointNavigation:
    """TC3.5: Navigate to a target, then navigate back to origin.

    Tests that the robot can accept a second navigation goal after
    completing the first. Multi-point coverage is achieved by the
    teleport-reset between all test cases.
    """

    def test_navigate_there_and_back(self, gazebo_env, reset_gazebo_env):
        gc = gazebo_env["goal_client"]

        # Leg 1: spawn → point_a
        status = _send_nav_goal(gc, POINT_A_X, POINT_A_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        assert status is not None, "Navigation to point_a did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"point_a failed with status {status}"

        pos = _get_odom_position(gc, "/odom", timeout=5.0)
        assert pos is not None
        assert abs(pos.x - POINT_A_X) < POSITION_TOLERANCE, (
            f"After point_a: x={pos.x:.3f}, expected {POINT_A_X} ± {POSITION_TOLERANCE}"
        )
        assert abs(pos.y - POINT_A_Y) < POSITION_TOLERANCE, (
            f"After point_a: y={pos.y:.3f}, expected {POINT_A_Y} ± {POSITION_TOLERANCE}"
        )

        # Leg 2: point_a → origin
        status = _send_nav_goal(gc, SPAWN_X, SPAWN_Y, 0.0, timeout=NAV_GOAL_TIMEOUT)
        assert status is not None, "Navigation to origin did not complete"
        assert status == GoalStatus.STATUS_SUCCEEDED, f"origin failed with status {status}"

        pos = _get_odom_position(gc, "/odom", timeout=5.0)
        assert pos is not None
        assert abs(pos.x - SPAWN_X) < POSITION_TOLERANCE, (
            f"After origin: x={pos.x:.3f}, expected {SPAWN_X} ± {POSITION_TOLERANCE}"
        )
        assert abs(pos.y - SPAWN_Y) < POSITION_TOLERANCE, (
            f"After origin: y={pos.y:.3f}, expected {SPAWN_Y} ± {POSITION_TOLERANCE}"
        )
