"""D5 arm/base cross gate for the mobile manipulator (lekiwi-nav-grasp-sim-platforms 4.1).

Both physics platforms, ground truth for the base, the shipped
``lekiwi_nav_grasp`` profile:

- moving base + HOLD / TORQUE_OFF / idle request -> the base stops within the
  #193 thresholds (planar <= 0.02 m/s, |yaw rate| <= 0.05 rad/s within 1 s of
  simulation time) and the arm stays within 0.05 rad of its pose at the stop;
- ``stream`` with the arm still converging to a target more than 0.1 rad
  away, then HOLD and (separately) an idle request -> stream commands sent
  after the stop do not move the arm, and the arm's settling follows the
  platform semantics (D5 decision; Gazebo part adjudicated 2026-10-29):
  MuJoCo settles within 0.05 rad of the LAST ACCEPTED target within 1 s of
  simulation time (a released position actuator keeps its ctrl); on Gazebo
  the settle point is NOT deterministic — gz_ros2_control 0.7.19 implements
  position control as a proportional-velocity servo whose last
  JointVelocityCmd persists in the entity-component system until the next
  controller-manager ``write()`` cycle after the release lands
  (``perform_command_mode_switch`` clears the whole control method with
  ``joint_control_method &= (VELOCITY & EFFORT)`` = 0), so under CPU
  contention the joint coasts past the command by a load-dependent amount —
  so the Gazebo leg asserts what actually holds: the arm ends STATIC (max
  joint velocity over the final second ~ 0), later stream commands never
  move it, and after the stop the arm controllers are inactive claiming no
  interface (no landing-point number is asserted; the pre-existing platform
  behaviour reproduces on so101_single_arm untouched by #205);
- ``trajectory`` mode executing an arm trajectory -> base ground-truth planar
  speed <= 0.02 m/s (the always-active MuJoCo base controller cannot move
  the base in an arm mode);
- the navigation-envelope gap (D7): with the arm at a non-stowed pose the
  runtime still admits ``base_navigation`` (no pose admission exists today)
  — asserted so a later admission check changes it deliberately — and the
  arm is then STOWED before any base motion (the D5 tests that move the arm
  return it to the reset pose first), after which ``base_navigation`` ->
  the arm holds within 0.05 rad of its end pose while the base drives.

Skips follow the shared rules (``IBROBOT_REQUIRE_SIM_PLATFORMS`` fails
instead of skipping).
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode, StopRuntime
from robot_runtime.profile import PHYSICS_SIM_PLATFORMS, load_profile
from robot_runtime.testing.arm_envelope import envelope_verdict, rest_verdict
from robot_runtime.testing.sim_environment import environment_missing, skip_or_fail
from robot_runtime.testing.sim_ground_truth import (
    RUN_MARKER_ENV,
    open_ground_truth,
    terminate_marked_processes,
)

PACKAGE = "lekiwi_robot"
MODEL_NAME = "lekiwi_robot"
WORLD_NAME = "empty"
PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_nav_grasp.yaml"
ARM_JOINTS = ["1", "2", "3", "4", "5", "6"]
#: The stowed pose = ``hardware.reset_positions`` (joints 1-5; D4).
STOWED_TARGET = [0.0, -1.5854, 1.5708, 0.7854, 0.0]
#: A stream target more than 0.1 rad from the stowed pose on joint 2.
STREAM_TARGET = [0.0, -0.5, 1.5708, 0.7854, 0.0]
FURTHER_TARGET = [0.0, 0.3, 1.5708, 0.7854, 0.0]
#: The (e) trajectory target: joints 1-2 +0.25 rad from the stowed pose (a
#: real executed trajectory, displacement > 0.1 rad; the direct JTC path has
#: no FCL, so the MoveIt-invalid stream region is irrelevant here).
TRAJECTORY_TARGET = [0.25, -1.3354, 1.5708, 0.7854, 0.0]
#: The arm JTC owns joints 1-5 only — the gripper joint "6" has its own
#: gripper controllers, and a goal naming a joint the controller does not
#: own (or with more names than positions) is rejected.
TRAJECTORY_JOINTS = ["1", "2", "3", "4", "5"]
#: The arm controllers that command the arm in ``stream`` mode (the Gazebo
#: post-stop check: both inactive, claiming nothing).
ARM_COMMAND_CONTROLLERS = ("arm_position_controller", "gripper_position_controller")
#: Gazebo post-stop rest bound: the measured resting joint velocity is
#: ~1e-11 rad/s (every run); six orders of margin still means "static".
GAZEBO_REST_RAD_S = 1e-6
#: The arm stream / rest freshness bound (D7): the arm moves on 100 ms
#: timescales at most, so a longer silence means the stream went stale.
ARM_MAX_SAMPLE_GAP_SEC = 0.5
#: Gazebo post-stop "later commands never move the arm": measured <= 1e-9.
GAZEBO_LATER_MOTION_RAD = 1e-6
DRIVE_VX = 0.26 / 2
LAUNCH_STARTUP_TIMEOUT = 180.0
COMMAND_RATE_HZ = 20.0


def _environment_missing(sim_platform: str) -> str | None:
    return environment_missing(sim_platform)


def _gate_skip_or_fail(sim_platform: str, reason: str) -> None:
    skip_or_fail(sim_platform, reason, context="nav_grasp stop cross gate")


def _platform_capabilities_missing(sim_platform: str) -> list[str]:
    from robot_runtime.testing.sim_ground_truth import UnsupportedGroundTruth

    missing = []
    try:
        open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME)
    except UnsupportedGroundTruth as error:
        missing.append(f"ground truth: {error}")
    supported = [str(entry) for entry in load_profile(PROFILE).get("sim_platforms") or []]
    if sim_platform not in supported:
        missing.append(f"runtime profile: {sim_platform!r} is not in sim_platforms {supported}")
    return missing


class CrossGateNode(Node):
    def __init__(self):
        super().__init__("navgrasp_stop_cross_gate")
        self.status: RuntimeStatus | None = None
        self.joint_samples: list[tuple[float, dict[str, float]]] = []
        self.joint_velocity_samples: list[tuple[float, dict[str, float]]] = []
        self.odom: list[Odometry] = []
        self.create_subscription(RuntimeStatus, "/runtime_status", self._on_status, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        # best-effort: the simulator process is load-sensitive (a reliable
        # 100 Hz subscription measurably degrades its control delivery)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joints,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=20),
        )
        self._cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self._arm_stream = self.create_publisher(Float64MultiArray, "/arm_position_controller/commands", 10)

    def _on_status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _on_odom(self, msg: Odometry) -> None:
        self.odom.append(msg)

    def _on_joints(self, msg) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.joint_samples.append((stamp, dict(zip(msg.name, msg.position, strict=False))))
        if len(msg.velocity) == len(msg.name):
            self.joint_velocity_samples.append((stamp, dict(zip(msg.name, msg.velocity, strict=False))))

    def arm_positions(self) -> dict[str, float]:
        for _stamp, positions in reversed(self.joint_samples):
            if all(joint in positions for joint in ARM_JOINTS):
                return {joint: positions[joint] for joint in ARM_JOINTS}
        return {}

    def stream_error(self, target: list[float]) -> float:
        positions = self.arm_positions()
        return max(
            abs(positions[f"{index + 1}"] - value) for index, value in enumerate(target) if f"{index + 1}" in positions
        )

    def publish_cmd_vel(self, vx: float) -> None:
        self._cmd_vel.publish(Twist(linear=Vector3(x=vx)))

    def publish_stream(self, target: list[float]) -> None:
        self._arm_stream.publish(Float64MultiArray(data=[float(v) for v in target]))


def _measure(window: list) -> dict:
    speeds, yaw_rates = [], []
    for previous, current in zip(window, window[1:], strict=False):
        dt = current.stamp_sec - previous.stamp_sec
        if dt <= 1e-6:
            continue
        speeds.append(math.hypot(current.x - previous.x, current.y - previous.y) / dt)
        dyaw = math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw))
        yaw_rates.append(dyaw / dt)
    return {
        "mean_planar_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "mean_abs_yaw_rate": sum(abs(rate) for rate in yaw_rates) / len(yaw_rates) if yaw_rates else 0.0,
        "pairs": len(speeds),
    }


#: Ground-truth coverage bound: at ~95 Hz a window whose edges are further
#: than this from the first/last sample is not measuring the interval it
#: claims (review round 2 — an empty window must fail, never read as 0.0).
GT_MAX_EDGE_GAP_SEC = 0.3


def _measure_gt(window: list, begin: float, end: float, label: str, *, min_pairs: int = 5) -> dict:
    """Ground-truth speeds over ``[begin, end]`` with window validity (review round 2).

    The plain :func:`_measure` returns two clean zeros for an empty window —
    a stopped base would then "pass" on no evidence. This wrapper requires
    enough samples, finite values and coverage of the whole interval before
    any number is trusted.
    """
    assert len(window) >= min_pairs + 1, (
        f"{label}: only {len(window)} ground-truth samples over [{begin:.2f}, {end:.2f}] s"
    )
    for sample in window:
        assert all(math.isfinite(value) for value in (sample.x, sample.y, sample.yaw)), (
            f"{label}: non-finite ground-truth sample at {sample.stamp_sec:.2f} s"
        )
    assert window[0].stamp_sec - begin <= GT_MAX_EDGE_GAP_SEC, (
        f"{label}: first ground-truth sample {window[0].stamp_sec - begin:.2f} s after the window start"
    )
    assert end - window[-1].stamp_sec <= GT_MAX_EDGE_GAP_SEC, (
        f"{label}: last ground-truth sample {end - window[-1].stamp_sec:.2f} s before the window end"
    )
    result = _measure(window)
    assert result["pairs"] >= min_pairs, (
        f"{label}: only {result['pairs']} valid ground-truth differentials over [{begin:.2f}, {end:.2f}] s"
    )
    return result


def _arm_hold(
    samples: list[tuple[float, dict[str, float]]],
    begin: float,
    end: float,
    reference: dict[str, float],
    tolerance: float,
) -> dict:
    """Whole-window arm-hold verdict over ``[begin, end]`` (review round 2).

    The shared envelope rule against ``reference``: fresh coverage of the
    whole interval (leading edge, internal gaps, trailing edge), every joint
    present with finite values, and the worst deviation ACCUMULATED over the
    window — an arm that deviates mid-window and returns, or a joint stream
    that goes stale, both fail; comparing only the final frame let both pass.
    """
    return envelope_verdict(
        samples,
        begin,
        end,
        reset={joint: float(reference[joint]) for joint in ARM_JOINTS if joint in reference},
        tolerance=tolerance,
        max_gap=ARM_MAX_SAMPLE_GAP_SEC,
    )


def _launch(sim_platform: str, environment: dict, log_path: Path):
    arguments = [
        "ros2",
        "launch",
        PACKAGE,
        "runtime.launch.py",
        f"profile:={PROFILE}",
        f"sim_platform:={sim_platform}",
    ]
    if sim_platform == "mujoco":
        arguments.append(f"mujoco_scene:={WORLD_NAME}")
    return subprocess.Popen(
        arguments, stdout=log_path.open("w"), stderr=subprocess.STDOUT, env=environment, start_new_session=True
    )


def _terminate(launch) -> None:
    try:
        if launch.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(launch.pid), signal.SIGINT)
            launch.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            launch.wait(timeout=10)


def _arm_controller_states(node) -> dict[str, tuple[str, list[str]]]:
    """The arm command controllers' (state, claimed interfaces) right now."""
    from controller_manager_msgs.srv import ListControllers

    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    assert client.wait_for_service(timeout_sec=10.0), "the controller manager is not answering"
    future = client.call_async(ListControllers.Request())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)
    assert future.done() and future.result() is not None
    states = {
        str(entry.name): (str(entry.state), [str(name) for name in entry.claimed_interfaces])
        for entry in future.result().controller
    }
    node.destroy_client(client)
    return states


def _wait_sim_seconds(node, ground_truth, seconds: float, drive: float | None = None, stream=None) -> float:
    """Advance ``seconds`` of simulation time, optionally driving/streaming."""
    start = ground_truth.latest_time()
    assert start is not None
    deadline = time.monotonic() + seconds * 20.0 + 60.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)
        if drive is not None:
            node.publish_cmd_vel(drive)
        if stream is not None:
            node.publish_stream(stream)
        now = ground_truth.latest_time()
        if now is not None and now - start >= seconds:
            return now
        time.sleep(1.0 / COMMAND_RATE_HZ)
    raise AssertionError(f"simulation time did not advance {seconds} s")


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_nav_grasp_stop_cross_assertions(tmp_path, sim_platform):
    missing = _platform_capabilities_missing(sim_platform)
    if missing:
        _gate_skip_or_fail(sim_platform, f"nav_grasp stop cross gate: platform {sim_platform!r} lacks: {missing}")
    environment_missing_reason = _environment_missing(sim_platform)
    if environment_missing_reason:
        _gate_skip_or_fail(
            sim_platform,
            f"nav_grasp stop cross gate: platform {sim_platform!r} cannot run: {environment_missing_reason}",
        )
    partition = f"navgrasp_cross_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    environment.update({RUN_MARKER_ENV: partition, "RCUTILS_COLORIZED_OUTPUT": "0"})
    if sim_platform == "gazebo":
        environment.update({"GZ_PARTITION": partition, "IGN_PARTITION": partition, "IBROBOT_GAZEBO_HEADLESS": "1"})
    else:
        environment["IBROBOT_MUJOCO_HEADLESS"] = "1"
    launch_log = tmp_path / "launch.log"
    launch = _launch(sim_platform, environment, launch_log)
    measurements: dict[str, dict] = {}
    ground_truth = None
    node = None
    try:
        rclpy.init()
        node = CrossGateNode()
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (node.status is None or node.status.lifecycle != "ACTIVE"):
            rclpy.spin_once(node, timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the launch exited early ({launch.returncode}); see {launch_log}")
        assert node.status is not None and node.status.lifecycle == "ACTIVE", f"see {launch_log}"
        ground_truth = open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME, env=environment).start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and ground_truth.latest_time() is None:
            time.sleep(0.1)
        assert ground_truth.latest_time() is not None, "no ground-truth samples"

        mode_client = node.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=15.0)
        stop_client = node.create_client(StopRuntime, "/runtime/stop")
        assert stop_client.wait_for_service(timeout_sec=15.0)
        from std_srvs.srv import SetBool

        gate_client = node.create_client(SetBool, "/motion_mode/set_navigation_enabled")
        assert gate_client.wait_for_service(timeout_sec=15.0)

        def call(client, request):
            future = client.call_async(request)
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not future.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            assert future.done() and future.result() is not None
            return future.result()

        def admit_base():
            request = SetRuntimeMode.Request()
            request.mode = "base_navigation"
            for _ in range(10):
                response = call(mode_client, request)
                if response.success:
                    break
                time.sleep(1.0)
                rclpy.spin_once(node, timeout_sec=0.1)
            assert response.success, response.message
            assert call(gate_client, SetBool.Request(data=True)).success

        def clear_stop():
            request = SetRuntimeMode.Request()
            request.mode = "idle"
            for _ in range(10):
                response = call(mode_client, request)
                if response.success:
                    return
                time.sleep(1.0)
                rclpy.spin_once(node, timeout_sec=0.1)
            pytest.fail("the idle request did not clear the stop latch")

        def assert_base_stopped_and_arm_holds(label: str, stop_stamp: float, arm_at_stop: dict[str, float]) -> None:
            # base: the #193 thresholds within 1 s of simulation time
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
                if ground_truth.latest_time() - stop_stamp >= 1.0:
                    break
            window = ground_truth.window(stop_stamp + 0.7, stop_stamp + 1.0)
            result = _measure_gt(window, stop_stamp + 0.7, stop_stamp + 1.0, label)
            # arm: the WHOLE post-stop phase against the pose at the stop
            # (review round 2 — a mid-window excursion that returns must fail)
            end = ground_truth.latest_time()
            hold = _arm_hold(node.joint_samples, stop_stamp, end, arm_at_stop, 0.05)
            measurements[label] = {
                **result,
                "arm_hold": {
                    "worst_rad": hold["worst_rad"],
                    "samples": hold["samples"],
                    "violations": hold["violations"],
                },
                "samples": len(window),
            }
            print(f"[cross-gate] {label}: {measurements[label]}", flush=True)
            assert result["mean_planar_speed"] <= 0.02, f"{label}: base planar speed {result['mean_planar_speed']:.4f}"
            assert result["mean_abs_yaw_rate"] <= 0.05, f"{label}: |yaw rate| {result['mean_abs_yaw_rate']:.4f}"
            assert hold["within"], f"{label}: the arm left its stop pose over the window: {hold['violations']}"

        # --- (a) moving base + HOLD ------------------------------------------
        admit_base()
        _wait_sim_seconds(node, ground_truth, 1.5, drive=DRIVE_VX)
        arm_at_stop = node.arm_positions()
        stop_stamp = ground_truth.latest_time()
        response = call(stop_client, StopRuntime.Request(policy="HOLD"))
        assert response.success, response.message
        _wait_sim_seconds(node, ground_truth, 1.5)
        assert_base_stopped_and_arm_holds("hold", stop_stamp, arm_at_stop)
        clear_stop()

        # --- (b) moving base + TORQUE_OFF --------------------------------------
        admit_base()
        _wait_sim_seconds(node, ground_truth, 1.5, drive=DRIVE_VX)
        arm_at_stop = node.arm_positions()
        stop_stamp = ground_truth.latest_time()
        response = call(stop_client, StopRuntime.Request(policy="TORQUE_OFF"))
        assert response.success, response.message
        _wait_sim_seconds(node, ground_truth, 1.5)
        assert_base_stopped_and_arm_holds("torque_off", stop_stamp, arm_at_stop)
        clear_stop()

        # --- (c) moving base + an idle mode request ------------------------------
        admit_base()
        _wait_sim_seconds(node, ground_truth, 1.5, drive=DRIVE_VX)
        arm_at_stop = node.arm_positions()
        stop_stamp = ground_truth.latest_time()
        request = SetRuntimeMode.Request()
        request.mode = "idle"
        assert call(mode_client, request).success
        _wait_sim_seconds(node, ground_truth, 1.5)
        assert_base_stopped_and_arm_holds("idle_request", stop_stamp, arm_at_stop)

        # --- (d) stream convergence + HOLD / idle, per-platform settling ----------
        for variant, engage in (("stream_hold", "stop"), ("stream_idle", "idle")):
            # park the arm at the stowed pose first (fresh start each variant)
            request = SetRuntimeMode.Request()
            request.mode = "stream"
            assert call(mode_client, request).success
            stowed = STOWED_TARGET
            _wait_sim_seconds(node, ground_truth, 2.0, stream=stowed)
            # stream toward the target and engage mid-convergence (the joint-2
            # error is still > 0.1 rad)
            start = time.monotonic()
            deadline = start + 30.0
            error_at_stop = None
            stop_stamp = None
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
                node.publish_stream(STREAM_TARGET)
                error = node.stream_error(STREAM_TARGET)
                if error < 0.9 and error > 0.1 and stop_stamp is None:
                    error_at_stop = error
                    stop_stamp = ground_truth.latest_time()
                    if engage == "stop":
                        response = call(stop_client, StopRuntime.Request(policy="HOLD"))
                        assert response.success, response.message
                    else:
                        request = SetRuntimeMode.Request()
                        request.mode = "idle"
                        assert call(mode_client, request).success
                    break
                time.sleep(1.0 / COMMAND_RATE_HZ)
            assert stop_stamp is not None and error_at_stop > 0.1, "the arm never entered the mid-convergence window"
            # the settle deadline: 1 s of simulation time from the stop request
            _wait_sim_seconds(node, ground_truth, max(0.0, stop_stamp + 1.0 - ground_truth.latest_time()))
            settled_error = node.stream_error(STREAM_TARGET)
            # then hold: 2 more seconds with no stream commands
            _wait_sim_seconds(node, ground_truth, 2.0)
            final_error = node.stream_error(STREAM_TARGET)
            later_start = node.arm_positions()
            later_begin = ground_truth.latest_time()
            # later stream commands must not move the arm — over the WHOLE
            # later phase, not just its final frame (review round 2)
            _wait_sim_seconds(node, ground_truth, 2.0, stream=FURTHER_TARGET)
            _wait_sim_seconds(node, ground_truth, 1.0)
            later_end = ground_truth.latest_time()
            later_bound = 0.02 if sim_platform == "mujoco" else GAZEBO_LATER_MOTION_RAD
            later = _arm_hold(node.joint_samples, later_begin, later_end, later_start, later_bound)
            later_motion = later["worst_rad"]
            measurements[variant] = {
                "error_at_stop_rad": error_at_stop,
                "settled_error_rad": settled_error,
                "final_error_rad": final_error,
                "later_motion_rad": later_motion,
                "later_violations": later["violations"],
            }
            print(f"[cross-gate] {variant}: {measurements[variant]}", flush=True)
            if sim_platform == "mujoco":
                # MuJoCo (like the physical servos, the SDK and
                # so101_single_arm): the released position actuator keeps its
                # ctrl, so the arm settles at the LAST ACCEPTED target
                # within 1 s and stays there
                assert settled_error <= 0.05, (
                    f"{variant}: MuJoCo arm was {settled_error:.4f} rad from the last accepted target "
                    f"1 s after the stop"
                )
                assert final_error <= 0.05, (
                    f"{variant}: MuJoCo arm settled {final_error:.4f} rad from the last accepted target"
                )
                assert later["within"], (
                    f"{variant}: later stream commands moved the arm (whole-window worst "
                    f"{later_motion:.4f} rad, bound {later_bound}): {later['violations']}"
                )
            else:
                # Gazebo (adjudicated 2026-10-29): the settle point races the
                # P-servo against the zero-velocity write that lands only on
                # the next controller-manager cycle after the release, so NO
                # landing point is asserted. What must hold: the arm ends
                # static, later commands never move it, and the arm
                # controllers are inactive claiming nothing after the stop.
                # The rest verdict is the shared window rule: the worst
                # |joint velocity| over the final second, all joints, finite
                # values, the window freshly covered.
                end = ground_truth.latest_time()
                rest = rest_verdict(
                    node.joint_velocity_samples,
                    end - 1.0,
                    end,
                    joints=ARM_JOINTS,
                    max_velocity=GAZEBO_REST_RAD_S,
                    max_gap=ARM_MAX_SAMPLE_GAP_SEC,
                )
                measurements[variant]["rest"] = rest
                assert rest["at_rest"], (
                    f"{variant}: the arm is not provably at rest after the stop: {rest['violations']}"
                )
                assert later["within"], (
                    f"{variant}: later stream commands moved the arm (whole-window worst "
                    f"{later_motion:.4f} rad, bound {later_bound}): {later['violations']}"
                )
                controller_states = _arm_controller_states(node)
                measurements[variant]["arm_controllers"] = controller_states
                for controller in ARM_COMMAND_CONTROLLERS:
                    state, claimed = controller_states.get(controller, ("missing", ["?"]))
                    assert state == "inactive" and not claimed, (
                        f"{variant}: {controller} is {state!r} claiming {claimed} after the stop"
                    )
            if engage == "stop":
                clear_stop()

        # --- (e) trajectory mode: a REAL arm trajectory keeps the base still --
        request = SetRuntimeMode.Request()
        request.mode = "trajectory"
        assert call(mode_client, request).success
        traj_client = ActionClient(node, FollowJointTrajectory, "/arm_trajectory_controller/follow_joint_trajectory")
        assert traj_client.wait_for_server(timeout_sec=15.0), "the arm trajectory action server never came up"
        # execute a non-zero arm trajectory through the JTC (review rework:
        # the section must exercise real motion, not just dwell in the mode).
        # TRAJECTORY_TARGET is > 0.1 rad from every pose (d) can leave the
        # arm at (joint 2 in [-1.2, 0.2] there; the target is -1.3354), so
        # the executed displacement is deterministic.
        before = node.arm_positions()
        traj_stamp = ground_truth.latest_time()
        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory.joint_names = list(TRAJECTORY_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in TRAJECTORY_TARGET]
        point.time_from_start.sec = 2
        traj_goal.trajectory.points = [point]
        traj_future = traj_client.send_goal_async(traj_goal)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not traj_future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        assert traj_future.done() and traj_future.result() is not None, "no trajectory goal response"
        assert traj_future.result().accepted, "the trajectory goal was rejected"
        result_future = traj_future.result().get_result_async()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        assert result_future.done(), "the trajectory never completed"
        assert result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, (
            f"the trajectory failed: {result_future.result().result.error_string}"
        )
        _wait_sim_seconds(node, ground_truth, 0.5)
        after = node.arm_positions()
        displacement = max(
            (abs(after[f"{index + 1}"] - before[f"{index + 1}"]) for index in range(len(TRAJECTORY_TARGET))),
            default=0.0,
        )
        target_error = max(
            (abs(after[f"{index + 1}"] - value) for index, value in enumerate(TRAJECTORY_TARGET)),
            default=float("inf"),
        )
        traj_end = ground_truth.latest_time()
        window = ground_truth.window(traj_stamp, traj_end)
        result = _measure_gt(window, traj_stamp, traj_end, "trajectory_base")
        measurements["trajectory_base"] = {
            **result,
            "arm_displacement_rad": displacement,
            "arm_target_error_rad": target_error,
        }
        print(f"[cross-gate] trajectory_base: {measurements['trajectory_base']}", flush=True)
        assert displacement > 0.1, (
            f"the executed trajectory moved the arm only {displacement:.4f} rad (a real motion is required)"
        )
        assert target_error < 0.05, f"the arm did not reach the trajectory target ({target_error:.4f} rad off)"
        assert result["mean_planar_speed"] <= 0.02, (
            f"trajectory mode moved the base {result['mean_planar_speed']:.4f} m/s"
        )
        traj_client.destroy()

        # --- (f) the navigation-envelope gap (D7) + base_navigation ---------------
        # Bring the arm to a KNOWN non-stowed pose first (the (d) settle point
        # is not deterministic on Gazebo): stream STREAM_TARGET, then hold it
        # in trajectory mode.
        request = SetRuntimeMode.Request()
        request.mode = "stream"
        assert call(mode_client, request).success
        unstash_deadline = time.monotonic() + 120.0
        unstash_error = None
        while time.monotonic() < unstash_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.publish_stream(STREAM_TARGET)
            unstash_error = node.stream_error(STREAM_TARGET)
            if unstash_error <= 0.05:
                break
            time.sleep(1.0 / COMMAND_RATE_HZ)
        assert unstash_error is not None and unstash_error <= 0.05, (
            f"the arm did not reach the non-stowed pose ({unstash_error} rad)"
        )
        request.mode = "trajectory"
        assert call(mode_client, request).success
        # The envelope gap (D7): with the arm at a non-stowed pose the runtime
        # admits base_navigation anyway — nothing between the mode table and
        # base_node checks the arm pose before base motion. Asserted so a
        # later admission check changes it deliberately (navigating with a
        # non-stowed arm is outside the validated envelope; enforcement is a
        # grasp-stage decision).
        non_stowed = node.stream_error(STOWED_TARGET)
        assert non_stowed > 0.1, f"expected a non-stowed arm here, error {non_stowed:.3f} rad"
        request.mode = "base_navigation"
        gap_response = call(mode_client, request)
        measurements["envelope_gap"] = {
            "arm_error_from_stowed_rad": non_stowed,
            "base_navigation_admitted": bool(gap_response.success),
        }
        print(f"[cross-gate] envelope_gap: {measurements['envelope_gap']}", flush=True)
        assert gap_response.success, f"base_navigation was rejected with a non-stowed arm: {gap_response.message}"
        # D7: the D5 tests that move the arm stow it before any base motion —
        # navigation is validated only with the arm at its reset pose
        request.mode = "trajectory"
        assert call(mode_client, request).success
        request.mode = "stream"
        assert call(mode_client, request).success
        stow_deadline = time.monotonic() + 120.0
        stow_error = None
        while time.monotonic() < stow_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.publish_stream(STOWED_TARGET)
            stow_error = node.stream_error(STOWED_TARGET)
            if stow_error <= 0.01:
                break
            time.sleep(1.0 / COMMAND_RATE_HZ)
        measurements["stow_before_base"] = {"error_rad": stow_error}
        print(f"[cross-gate] stow_before_base: {measurements['stow_before_base']}", flush=True)
        assert stow_error is not None and stow_error <= 0.01, (
            f"the arm did not stow before base motion ({stow_error} rad)"
        )
        # base_navigation after trajectory -> the arm holds while the base drives
        request.mode = "trajectory"
        assert call(mode_client, request).success
        arm_at_drive = node.arm_positions()
        drive_begin = ground_truth.latest_time()
        admit_base()
        _wait_sim_seconds(node, ground_truth, 3.0, drive=DRIVE_VX)
        drive_end = ground_truth.latest_time()
        # the whole drive phase, not just its final frame (review round 2)
        hold = _arm_hold(node.joint_samples, drive_begin, drive_end, arm_at_drive, 0.05)
        measurements["base_after_trajectory"] = {
            "arm_hold": {"worst_rad": hold["worst_rad"], "samples": hold["samples"], "violations": hold["violations"]},
        }
        print(f"[cross-gate] base_after_trajectory: {measurements['base_after_trajectory']}", flush=True)
        assert hold["within"], f"the arm left its pose while the base drove: {hold['violations']}"
        measurements["summary"] = "all cross assertions held"
    finally:
        if ground_truth is not None:
            ground_truth.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _terminate(launch)
        terminate_marked_processes({RUN_MARKER_ENV: partition})
        print(f"[cross-gate] measurements: {measurements}", flush=True)


# ---------------------------------------------------------------------------
# Counter-examples for the hold / measurement rules (review round 2, no
# simulator needed — the helpers are pure functions over sample lists)
# ---------------------------------------------------------------------------


def _arm_sample(stamp: float, joint2: float) -> tuple[float, dict[str, float]]:
    return (stamp, {joint: joint2 if joint == "2" else 0.0 for joint in ARM_JOINTS})


def _hold_reference() -> dict[str, float]:
    return {joint: 0.0 for joint in ARM_JOINTS}


def test_a_mid_window_deviation_that_returns_fails_the_hold():
    # the arm dips 0.3 rad mid-window and comes back: the final frame is
    # clean, the whole-window rule must not be
    samples = [_arm_sample(0.0 + 0.1 * i, 0.0) for i in range(11)]
    samples[5] = _arm_sample(0.5, 0.3)
    verdict = _arm_hold(samples, 0.0, 1.0, _hold_reference(), 0.05)
    assert not verdict["within"], "a mid-window excursion that returns must fail"
    assert verdict["worst_rad"] == pytest.approx(0.3)


def test_a_stale_trailing_stream_fails_the_hold():
    # the joint stream stops 0.8 s before the window ends (gap > 0.5): the
    # last complete frame still says 0.0 — freshness must fail it
    samples = [_arm_sample(0.1 * i, 0.0) for i in range(3)]
    verdict = _arm_hold(samples, 0.0, 1.0, _hold_reference(), 0.05)
    assert not verdict["within"], "a stale trailing edge must fail the hold"
    assert any("before the window end" in violation for violation in verdict["violations"])


def _gt_sample(stamp: float, x: float = 0.0) -> object:
    from collections import namedtuple

    return namedtuple("GtSample", "stamp_sec x y yaw")(stamp, x, 0.0, 0.0)


def test_an_empty_ground_truth_window_fails_instead_of_reading_zero():
    # _measure alone returns two clean zeros for no samples; the checked
    # wrapper must refuse to produce a verdict from nothing
    with pytest.raises(AssertionError, match="ground-truth samples"):
        _measure_gt([], 0.0, 1.0, "counterexample")


def test_a_sparse_ground_truth_window_fails_instead_of_averaging_two_samples():
    # two samples straddling a 1 s gap produce one huge-or-tiny differential:
    # not enough pairs to trust a mean speed
    with pytest.raises(AssertionError, match="ground-truth samples"):
        _measure_gt([_gt_sample(0.05), _gt_sample(0.95, 0.5)], 0.0, 1.0, "counterexample")
