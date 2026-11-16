"""The nav-grasp reach-accuracy test (lekiwi-nav-grasp D4).

Both physics platforms, the shipped ``lekiwi_nav_grasp`` profile launched
provider-alone in the ``empty`` scene, mode ``trajectory``. N = 4 ``gripper``
targets — the FK poses of known CONTACT-FREE configurations (two
fingers-down in the grasp height band, one forward-tilted, one lateral) —
each commanded through ``MoveToPose``; after a settle wait of
:data:`SETTLE_SIM_SEC` of simulation time the GROUND-TRUTH ``gripper`` link
pose (the entity ground truth: Gazebo's pose-info model∘link composed with
the GT base pose and the static ``base_link → base`` mount transform from
TF; MuJoCo's FK of the provider's own model, relative to ``base`` — read
from the provider's ``mujoco_model`` parameter) is compared with the target
FK: position error <= 0.015 m and full orientation error <= 10 deg for
EVERY target. The service's own ``position_error_m`` is recorded alongside
the ground-truth error and their difference (the URDF-vs-simulator model
mismatch).
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import rclpy
from rclpy.node import Node

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import ComputeFk, MoveToPose, SetRuntimeMode
from robot_runtime_sim.testing.entity_ground_truth import open_entity_ground_truth
from robot_runtime_sim.testing.sim_ground_truth import RUN_MARKER_ENV, terminate_marked_processes

PACKAGE = "lekiwi_robot"
PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_nav_grasp.yaml"
LAUNCH_STARTUP_TIMEOUT = 240.0
ARM_JOINTS = ("1", "2", "3", "4", "5")

#: D4 thresholds: every target is the FK pose of a known configuration, so
#: its orientation is exactly reachable by the 5-DOF arm.
POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_DEG = 10.0

#: The settle wait: simulation time that must pass after the MoveToPose
#: acceptance before the ground-truth sample is read (the consolidation's
#: simple sim-time settle — a non-settled arm shows up as a threshold
#: failure, not as a hidden pass).
SETTLE_SIM_SEC = 2.0

#: The 4 targets (D4), DERIVED at run time through the PUBLIC
#: /motion/compute_fk (the provider's own model): each MoveToPose target is
#: the GRIPPER-link pose of its configuration. The first version of this
#: table carried hard-coded tcp positions, which commanded the gripper ~9 cm
#: below the intended configurations — the IK then jammed the jaws onto the
#: support plane and tipped the chassis (measured 2026-10-30). The gripper
#: poses ARE what MoveToPose commands; the orientations equal the tcp FK
#: values because the gripper->tcp offset is a pure translation.
REACH_TARGETS: dict[str, list[float]] = {
    "fingers_down_low": [0.0273, -0.1949, 1.4249, 0.1154, 0.0],
    "fingers_down_high": [-0.1585, 0.1118, 0.8193, 0.2856, 0.0],
    "forward_tilted": [-0.1434, -0.3161, 1.0963, 0.2938, 0.0],
    "lateral": [-0.5297, -0.6548, 1.5359, 0.3328, 0.0],
}

_ENTITY = "lekiwi_robot"
_LINK = "gripper"


def _quat_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _orientation_error_deg(achieved, target) -> float:
    """The quaternion angle between two rotation matrices."""
    relative = achieved.T @ target
    trace = float(np.trace(relative))
    return math.degrees(math.acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0))))


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
        arguments.append("mujoco_scene:=empty")
    return subprocess.Popen(
        arguments, stdout=log_path.open("w"), stderr=subprocess.STDOUT, env=environment, start_new_session=True
    )


def _terminate(launch) -> None:
    import contextlib

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


class ReachNode(Node):
    """The runtime status and the sim clock."""

    def __init__(self):
        super().__init__("navgrasp_reach")
        self.status = None
        self.clock_sim = 0.0
        self.create_subscription(RuntimeStatus, "/runtime_status", lambda message: setattr(self, "status", message), 10)
        from rosgraph_msgs.msg import Clock

        self.create_subscription(
            Clock,
            "/clock",
            lambda message: setattr(self, "clock_sim", message.clock.sec + message.clock.nanosec / 1e9),
            200,
        )

    def spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_sim_seconds(self, seconds: float) -> None:
        """Let ``seconds`` of SIMULATION time pass (wall bound absorbs drift)."""
        target = self.clock_sim + seconds
        deadline = time.monotonic() + max(30.0, seconds * 20.0)
        while time.monotonic() < deadline and self.clock_sim < target:
            rclpy.spin_once(self, timeout_sec=0.05)


def _mujoco_model_path(node: ReachNode, launch_log: Path) -> str:
    """The MJCF the provider ACTUALLY loaded: the ``mujoco_model`` hardware
    parameter of the URDF it received (/robot_description — the composed
    path is substituted into it at render time). No fallback — the URDF is
    the provider's own statement of what it loaded, and its absence is an
    error."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    transient = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    descriptions: list[String] = []
    subscription = node.create_subscription(String, "/robot_description", descriptions.append, transient)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not descriptions:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(subscription)
    assert descriptions, f"no /robot_description; see {launch_log}"
    for param in ElementTree.fromstring(descriptions[-1].data).findall(".//ros2_control/hardware/param"):
        if param.get("name") == "mujoco_model":
            mjcf_path = (param.text or "").strip()
            if mjcf_path:
                return mjcf_path
    pytest.fail(f"no mujoco_model hardware parameter in /robot_description; see {launch_log}")


@pytest.mark.parametrize("sim_platform", ("gazebo", "mujoco"))
def test_navgrasp_reach_accuracy(tmp_path, sim_platform):
    from robot_runtime.testing.sim_environment import environment_missing, skip_or_fail

    environment_reason = environment_missing(sim_platform)
    if environment_reason:
        skip_or_fail(sim_platform, environment_reason, context="nav_grasp reach accuracy")

    partition = f"navgrasp_reach_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    environment.update({RUN_MARKER_ENV: partition, "RCUTILS_COLORIZED_OUTPUT": "0"})
    if sim_platform == "gazebo":
        environment.update({"GZ_PARTITION": partition, "IGN_PARTITION": partition, "IBROBOT_GAZEBO_HEADLESS": "1"})
    else:
        environment["IBROBOT_MUJOCO_HEADLESS"] = "1"
    launch_log = tmp_path / "launch.log"
    launch = _launch(sim_platform, environment, launch_log)
    node = None
    streams: list = []
    measurements: dict[str, dict] = {}
    teardown_errors: list[str] = []
    try:
        rclpy.init()
        node = ReachNode()
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (node.status is None or node.status.lifecycle != "ACTIVE"):
            rclpy.spin_once(node, timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the launch exited early ({launch.returncode}); see {launch_log}")
        assert node.status is not None and node.status.lifecycle == "ACTIVE", f"see {launch_log}"

        mode_client = node.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=15.0)
        pose_client = node.create_client(MoveToPose, "/motion/move_to_pose")
        assert pose_client.wait_for_service(timeout_sec=120.0), f"see {launch_log}"
        fk_client = node.create_client(ComputeFk, "/motion/compute_fk")
        assert fk_client.wait_for_service(timeout_sec=120.0), f"see {launch_log}"

        def call(client, request, timeout=120.0):
            future = client.call_async(request)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not future.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            assert future.done() and future.result() is not None
            return future.result()

        request = SetRuntimeMode.Request()
        request.mode = "trajectory"
        # the mode switch races the controller loading under host load
        # (measured 2026-10-30: "switch_controller rejected 'idle' ->
        # 'trajectory'" with the runtime already ACTIVE) — the switch is
        # not what D4 measures, so it gets a few retries
        for attempt in range(4):
            response = call(mode_client, request)
            if response.success:
                break
            print(f"[reach] set_mode attempt {attempt + 1} failed: {response.message}", flush=True)
            node.spin(2.0)
        assert response.success, f"the runtime never accepted mode trajectory; see {launch_log}"

        # the entity ground truth of the gripper link (D3) and of the base
        mjcf_path = _mujoco_model_path(node, launch_log) if sim_platform == "mujoco" else None
        gripper_stream = open_entity_ground_truth(
            sim_platform,
            world="empty",
            scene="empty",
            entity=_ENTITY,
            link=_LINK,
            env=environment,
            mjcf_path=mjcf_path,
            reference_body="base" if sim_platform == "mujoco" else None,
        )
        base_stream = open_entity_ground_truth(
            sim_platform, world="empty", scene="empty", entity=_ENTITY, env=environment, mjcf_path=mjcf_path
        )
        streams = [gripper_stream, base_stream]
        for stream in streams:
            stream.start()
        assert gripper_stream.wait_for(5, 30.0), f"no ground-truth gripper samples; see {launch_log}"

        # the static base_link -> base transform (the arm mount), from TF
        import tf2_ros

        buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(buffer, node)
        node.spin(3.0)
        transform = buffer.lookup_transform("base", "base_link", rclpy.time.Time())
        base_to_baselink = np.eye(4)
        base_to_baselink[:3, :3] = _quat_to_matrix(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        base_to_baselink[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]

        # every MoveToPose target is DERIVED from the shared configuration
        # through the PUBLIC /motion/compute_fk (the provider's own model) at
        # run time — the gripper poses the FK reports for the pinned,
        # contact-free configurations
        target_poses = {}
        for label, configuration in REACH_TARGETS.items():
            fk_request = ComputeFk.Request()
            fk_request.joint_state.name = list(ARM_JOINTS)
            fk_request.joint_state.position = [float(value) for value in configuration]
            fk_request.link_names = [_LINK]
            fk_response = call(fk_client, fk_request)
            assert fk_response.success, f"{label}: compute_fk failed: {fk_response.message}"
            pose = fk_response.poses[0].pose
            target_poses[label] = (
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )

        for label, (position, quaternion) in target_poses.items():
            request = MoveToPose.Request()
            request.target_pose.position.x, request.target_pose.position.y, request.target_pose.position.z = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
            (
                request.target_pose.orientation.x,
                request.target_pose.orientation.y,
                request.target_pose.orientation.z,
                request.target_pose.orientation.w,
            ) = (float(value) for value in quaternion)
            request.velocity_scaling = 0.2
            target_rotation = _quat_to_matrix(*quaternion)
            response = call(pose_client, request, 180.0)
            assert response.success and response.outcome == "SUCCEEDED", (
                f"{label}: MoveToPose {response.outcome}: {response.message}; see {launch_log}"
            )
            # the simple sim-time settle: SETTLE_SIM_SEC of simulation time
            # after the goal acceptance, then the freshest GT sample
            node.wait_sim_seconds(SETTLE_SIM_SEC)
            sample = gripper_stream.latest()
            assert sample is not None, f"{label}: no ground-truth sample after the settle"

            # the gripper link pose in `base`: on Gazebo the stream's world
            # pose is composed through the GT base pose and the static
            # base_link -> base mount transform. On MuJoCo the stream returns
            # the pose RELATIVE to the arm `base` body (reference_body="base"
            # — the free-joint qpos IS the root body's world pose, so the
            # relative return needs no second base reader: the base cancels).
            if sim_platform == "mujoco":
                achieved_in_base = np.eye(4)
                achieved_in_base[:3, :3] = _quat_to_matrix(*sample.quat_xyzw)
                achieved_in_base[:3, 3] = sample.position
            else:
                achieved_world = np.eye(4)
                achieved_world[:3, :3] = _quat_to_matrix(*sample.quat_xyzw)
                achieved_world[:3, 3] = sample.position
                base_sample = base_stream.latest()
                assert base_sample is not None, f"{label}: no ground-truth base sample after the settle"
                base_world = np.eye(4)
                base_world[:3, :3] = _quat_to_matrix(*base_sample.quat_xyzw)
                base_world[:3, 3] = base_sample.position
                achieved_in_base = base_to_baselink @ np.linalg.inv(base_world) @ achieved_world

            position_error = float(np.linalg.norm(achieved_in_base[:3, 3] - np.array(position)))
            orientation_error = _orientation_error_deg(achieved_in_base[:3, :3], target_rotation)
            measurements[label] = {
                "position_error_m": round(position_error, 4),
                "orientation_error_deg": round(orientation_error, 2),
                "service_position_error_m": round(response.position_error_m, 4),
                "service_vs_gt_difference_m": round(response.position_error_m - position_error, 4),
            }
            print(f"[reach] {label}: {measurements[label]}", flush=True)
            assert position_error <= POSITION_TOLERANCE_M, (
                f"{label}: ground-truth position error {position_error * 1000:.1f} mm > "
                f"{POSITION_TOLERANCE_M * 1000:.0f} mm; see {launch_log}"
            )
            assert orientation_error <= ORIENTATION_TOLERANCE_DEG, (
                f"{label}: ground-truth orientation error {orientation_error:.1f} deg > "
                f"{ORIENTATION_TOLERANCE_DEG:.0f} deg; see {launch_log}"
            )
        print(f"[reach] {sim_platform} measurements: {measurements}", flush=True)
    finally:
        # review r2 finding 3: every teardown step runs INDEPENDENTLY (a
        # failure in an earlier step must not skip the later ones) and the
        # collected failures are hard failures once the finally completes —
        # never swallowed behind a passing body or another step's error.
        for stream in streams:
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - one failing step must not skip the rest
                teardown_errors.append(f"ground-truth stream close: {type(exc).__name__}: {exc}")
        for label, action in (
            ("node destroy", lambda: node.destroy_node() if node is not None else None),
            ("rclpy shutdown", rclpy.shutdown),
            ("launch termination", lambda: _terminate(launch)),
            ("run-marker sweep", lambda: terminate_marked_processes({RUN_MARKER_ENV: environment[RUN_MARKER_ENV]})),
            (
                "ros2 daemon stop",
                lambda: subprocess.run(
                    ["ros2", "daemon", "stop"], env=environment, capture_output=True, timeout=60, check=False
                ),
            ),
        ):
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - one failing step must not skip the rest
                teardown_errors.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not teardown_errors, "the reach test's teardown steps failed:\n- " + "\n- ".join(teardown_errors)
