"""MoveIt smoke + wrist camera checks for nav_grasp (D6, task 4.4).

Both physics platforms, the shipped ``lekiwi_nav_grasp`` profile launched
provider-alone (the profile's ``motion`` section starts the so101_motion
stack: move_group, the motion server, the IK worker — on simulation time):

- **MoveIt smoke**: ``MoveToConfiguration`` to a non-stowed configuration
  succeeds (outcome SUCCEEDED) and ``/joint_states`` reaches it within
  0.05 rad; FK of the reached state matches ``compute_fk`` of the target
  within 1 cm (D6). Reach accuracy against simulator ground truth is the
  grasp stage, not this one.
- **Wrist camera**: the wrist RealSense's color and aligned depth publish
  on the advertised topics (``/camera/wrist/image_raw`` and
  ``/camera/wrist/aligned_depth_to_color/image_raw``); the image and
  CameraInfo headers carry the optical frame
  ``camera_wrist_optical_frame`` (the advertised optical frame, both
  platforms); the static TF chain
  ``gripper -> camera_wrist_link -> camera_wrist_optical_frame`` exists;
  the stamps are on simulation time; the images carry real rendered
  structure — decoded per encoding with row padding dropped, the color frame
  varies in every channel and the depth frame has a real fraction of valid
  (finite, non-zero) returns with depth structure (a failed, flat, constant
  or all-invalid render fails; the scene itself is static, so identical
  content across frames is expected).

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
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import ComputeFk, MoveToConfiguration, SetRuntimeMode
from robot_runtime.profile import PHYSICS_SIM_PLATFORMS, load_profile
from robot_runtime.testing.sim_environment import environment_missing, skip_or_fail

PACKAGE = "lekiwi_robot"
MODEL_NAME = "lekiwi_robot"
WORLD_NAME = "empty"
PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_nav_grasp.yaml"
ARM_JOINTS = ["1", "2", "3", "4", "5"]
#: The stowed pose = hardware.reset_positions (joints 1-5; D4).
STOWED_TARGET = [0.0, -1.5854, 1.5708, 0.7854, 0.0]
#: A non-stowed configuration for the MoveToConfiguration smoke — joints 1-2
#: +0.2 rad from the stowed pose, the conformance suite's proven motion
#: target family. NOTE: the cross gate's stream target [0.0, -0.5, ...] is
#: NOT plannable: MoveIt's arm-only FCL model finds a gripper-base contact
#: there (measured 2026-10-29: "Found a contact between 'gripper' and
#: 'base'"), although the configuration is physically reachable — the
#: cross gate streams to it on both platforms and MuJoCo excludes the
#: parent-root contact pair. Streaming and planning have different valid
#: regions; the smoke must use a plannable one.
SMOKE_TARGET = [0.2, -1.3854, 1.5708, 0.7854, 0.0]
#: D6 tolerances: /joint_states reaches the target within 0.05 rad; FK of the
#: reached state matches FK of the target within 1 cm.
REACH_TOLERANCE_RAD = 0.05
FK_TOLERANCE_M = 0.01
LAUNCH_STARTUP_TIMEOUT = 240.0  # the motion stack (move_group et al.) is slow to boot
COLOR_TOPIC = "/camera/wrist/image_raw"
COLOR_INFO_TOPIC = "/camera/wrist/camera_info"
DEPTH_TOPIC = "/camera/wrist/aligned_depth_to_color/image_raw"
OPTICAL_FRAME = "camera_wrist_optical_frame"
#: The minimum fraction of valid (finite, non-zero) aligned-depth pixels a
#: real render must carry (review round 2; an all-NaN/zero depth is invalid).
#: The wrist camera in `empty` sees the ground plane and the arm up close —
#: measured fractions live far above this floor (recorded in the smoke run).
DEPTH_VALID_FRACTION_FLOOR = 0.3
#: The static TF chain the wrist camera must contribute (D6).
CAMERA_TF_CHAIN = {("gripper", "camera_wrist_link"), ("camera_wrist_link", OPTICAL_FRAME)}


def _environment_missing(sim_platform: str) -> str | None:
    return environment_missing(sim_platform)


def _smoke_skip_or_fail(sim_platform: str, reason: str) -> None:
    skip_or_fail(sim_platform, reason, context="nav_grasp motion smoke")


def _platform_capabilities_missing(sim_platform: str) -> list[str]:
    from robot_runtime.testing.sim_ground_truth import UnsupportedGroundTruth

    missing = []
    try:
        from robot_runtime.testing.sim_ground_truth import open_ground_truth

        open_ground_truth(sim_platform, world=WORLD_NAME, model=MODEL_NAME)
    except UnsupportedGroundTruth as error:
        missing.append(f"ground truth: {error}")
    supported = [str(entry) for entry in load_profile(PROFILE).get("sim_platforms") or []]
    if sim_platform not in supported:
        missing.append(f"runtime profile: {sim_platform!r} is not in sim_platforms {supported}")
    return missing


class SmokeNode(Node):
    def __init__(self):
        super().__init__("navgrasp_motion_smoke")
        self.status: RuntimeStatus | None = None
        self.joint_samples: list[tuple[float, dict[str, float]]] = []
        self.color_frames: list[Image] = []
        self.color_info: list[CameraInfo] = []
        self.depth_frames: list[Image] = []
        self.clock_sim: float | None = None
        self.tf_static_pairs: set[tuple[str, str]] = set()
        self.create_subscription(RuntimeStatus, "/runtime_status", self._on_status, 10)
        # best-effort with a small depth — the simulator process is
        # load-sensitive (the same rule as the gates)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        best_effort = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
        )
        self.create_subscription(JointState, "/joint_states", self._on_joints, best_effort)
        self.create_subscription(Image, COLOR_TOPIC, self.color_frames.append, best_effort)
        self.create_subscription(CameraInfo, COLOR_INFO_TOPIC, self.color_info.append, best_effort)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_frames.append, best_effort)
        from rosgraph_msgs.msg import Clock

        self.create_subscription(Clock, "/clock", self._on_clock, 50)
        from tf2_msgs.msg import TFMessage

        self.create_subscription(
            TFMessage,
            "/tf_static",
            self._on_tf_static,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def _on_status(self, msg: RuntimeStatus) -> None:
        self.status = msg

    def _on_joints(self, msg: JointState) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.joint_samples.append((stamp, dict(zip(msg.name, msg.position, strict=False))))

    def _on_clock(self, msg) -> None:
        sim = msg.clock.sec + msg.clock.nanosec / 1e9
        if self.clock_sim is None or sim > self.clock_sim:
            self.clock_sim = sim

    def _on_tf_static(self, msg) -> None:
        for transform in msg.transforms:
            self.tf_static_pairs.add((transform.header.frame_id, transform.child_frame_id))

    def arm_positions(self) -> dict[str, float]:
        for _stamp, positions in reversed(self.joint_samples):
            if all(joint in positions for joint in ARM_JOINTS):
                return {joint: positions[joint] for joint in ARM_JOINTS}
        return {}

    def arm_error(self, target: list[float]) -> float:
        positions = self.arm_positions()
        return max(
            (abs(positions[f"{index + 1}"] - value) for index, value in enumerate(target)),
            default=float("inf"),
        )


def _rows(frame: Image):
    """``step``-wide byte rows of the image (row padding preserved)."""
    import numpy as np

    return np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.step)


def _color_channel_variances(frame: Image) -> list[float]:
    """Spatial per-channel variance of an rgb8/bgr8 frame, padding dropped.

    Decoded per the declared encoding with ``step`` respected (review round
    2): the old raw-byte variance passed padding noise — a PURE-COLOR frame
    whose row padding varies read as "structure". A rendered scene varies in
    every channel; a flat or failed render does not.
    """
    import numpy as np

    encoding = str(frame.encoding).lower()
    if encoding not in ("rgb8", "bgr8"):
        raise AssertionError(f"the color image encoding {encoding!r} is not one this check decodes (rgb8/bgr8)")
    pixels = _rows(frame)[:, : frame.width * 3].reshape(frame.height, frame.width, 3).astype(np.float64)
    return [float(pixels[:, :, channel].var()) for channel in range(3)]


def _depth_valid_variance(frame: Image) -> tuple[float, float]:
    """(valid-pixel fraction, spatial variance of the valid pixels) for 32FC1.

    Decoded per the declared encoding with ``step`` respected and explicit
    little-endian float32. Valid = finite AND non-zero (0 is the no-return
    value). The old raw-byte variance passed an all-NaN depth (NaN bytes and
    padding vary) and a constant-1.0 depth; the review round 2 rule requires
    a real fraction of valid returns first, then actual depth structure.
    """
    import numpy as np

    if str(frame.encoding).lower() != "32fc1":
        raise AssertionError(f"the depth image encoding {frame.encoding!r} is not one this check decodes (32fc1)")
    packed = _rows(frame)[:, : frame.width * 4].tobytes()
    pixels = np.frombuffer(packed, dtype="<f4").reshape(frame.height, frame.width).astype(np.float64)
    valid = np.isfinite(pixels) & (pixels > 0.0)
    fraction = float(valid.sum()) / float(valid.size)
    if fraction <= 0.0:
        return fraction, 0.0
    return fraction, float(pixels[valid].var())


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


@pytest.mark.parametrize("sim_platform", PHYSICS_SIM_PLATFORMS)
def test_navgrasp_motion_smoke_and_wrist_camera(tmp_path, sim_platform):
    missing = _platform_capabilities_missing(sim_platform)
    if missing:
        _smoke_skip_or_fail(sim_platform, f"nav_grasp motion smoke: platform {sim_platform!r} lacks: {missing}")
    environment_missing_reason = _environment_missing(sim_platform)
    if environment_missing_reason:
        _smoke_skip_or_fail(
            sim_platform,
            f"nav_grasp motion smoke: platform {sim_platform!r} cannot run: {environment_missing_reason}",
        )
    from robot_runtime.testing.sim_ground_truth import RUN_MARKER_ENV, terminate_marked_processes

    partition = f"navgrasp_smoke_{uuid.uuid4().hex[:8]}"
    environment = dict(os.environ)
    environment.update({RUN_MARKER_ENV: partition, "RCUTILS_COLORIZED_OUTPUT": "0"})
    if sim_platform == "gazebo":
        environment.update({"GZ_PARTITION": partition, "IGN_PARTITION": partition, "IBROBOT_GAZEBO_HEADLESS": "1"})
    else:
        environment["IBROBOT_MUJOCO_HEADLESS"] = "1"
    launch_log = tmp_path / "launch.log"
    launch = _launch(sim_platform, environment, launch_log)
    measurements: dict[str, dict] = {}
    node = None
    try:
        rclpy.init()
        node = SmokeNode()
        deadline = time.monotonic() + LAUNCH_STARTUP_TIMEOUT
        while time.monotonic() < deadline and (node.status is None or node.status.lifecycle != "ACTIVE"):
            rclpy.spin_once(node, timeout_sec=0.2)
            if launch.poll() is not None:
                pytest.fail(f"the launch exited early ({launch.returncode}); see {launch_log}")
        assert node.status is not None and node.status.lifecycle == "ACTIVE", f"see {launch_log}"

        mode_client = node.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=15.0)
        move_client = node.create_client(MoveToConfiguration, "/motion/move_to_joint")
        assert move_client.wait_for_service(timeout_sec=120.0), f"the motion server never came up; see {launch_log}"
        fk_client = node.create_client(ComputeFk, "/motion/compute_fk")
        assert fk_client.wait_for_service(timeout_sec=120.0), f"the FK service never came up; see {launch_log}"

        def call(client, request, timeout=60.0):
            future = client.call_async(request)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not future.done():
                rclpy.spin_once(node, timeout_sec=0.05)
            assert future.done() and future.result() is not None
            return future.result()

        def _spin(seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

        # --- the wrist camera checks (D6) ------------------------------------
        _spin(20.0)
        assert len(node.color_frames) >= 3, (
            f"too few color frames on {COLOR_TOPIC} ({len(node.color_frames)}); see {launch_log}"
        )
        assert len(node.color_info) >= 3, f"too few CameraInfo messages on {COLOR_INFO_TOPIC}"
        assert len(node.depth_frames) >= 3, (
            f"too few aligned-depth frames on {DEPTH_TOPIC} ({len(node.depth_frames)}); see {launch_log}"
        )
        for label, key, frames in (
            ("color", "color_variance", node.color_frames),
            ("aligned depth", "aligned_depth_variance", node.depth_frames),
        ):
            for frame in frames[-3:]:
                assert frame.header.frame_id == OPTICAL_FRAME, (
                    f"the {label} image header carries {frame.header.frame_id!r}, not {OPTICAL_FRAME!r}"
                )
                stamp = frame.header.stamp.sec + frame.header.stamp.nanosec / 1e9
                assert node.clock_sim is not None and 0.0 < stamp <= node.clock_sim + 0.1, (
                    f"the {label} image stamp {stamp} is not simulation time (clock {node.clock_sim})"
                )
            if label == "color":
                channel_variances = _color_channel_variances(frames[-1])
                measurements[key] = {"channel_variances": channel_variances, "frames": len(frames)}
                print(f"[smoke] {label}: {measurements[key]}", flush=True)
                # D6: "images are not constant" = the decoded frame carries
                # structure in EVERY channel (a flat/failed render does not).
                # The SCENE here is static (robot idle in `empty`), so
                # identical content ACROSS frames is expected, not a fault.
                assert min(channel_variances) > 0.0, (
                    f"the {label} image is constant in a channel (per-channel variances {channel_variances})"
                )
            else:
                valid_fraction, depth_variance = _depth_valid_variance(frames[-1])
                measurements[key] = {
                    "valid_fraction": valid_fraction,
                    "depth_variance": depth_variance,
                    "frames": len(frames),
                }
                print(f"[smoke] {label}: {measurements[key]}", flush=True)
                # a real fraction of valid returns first (an all-NaN or
                # all-zero depth renders "valid bytes" but no depth), then
                # actual depth structure (a constant-1.0 plane is not a
                # rendered scene)
                assert valid_fraction >= DEPTH_VALID_FRACTION_FLOOR, (
                    f"the {label} image has only {valid_fraction:.0%} valid (finite, non-zero) pixels"
                )
                assert depth_variance > 0.0, (
                    f"the {label} image is constant over its valid pixels (variance {depth_variance:.3g})"
                )
        for info in node.color_info[-3:]:
            assert info.header.frame_id == OPTICAL_FRAME, (
                f"the CameraInfo header carries {info.header.frame_id!r}, not {OPTICAL_FRAME!r}"
            )
        missing_chain = CAMERA_TF_CHAIN - node.tf_static_pairs
        assert not missing_chain, f"the static TF chain is missing {sorted(missing_chain)}"

        # --- the MoveIt smoke (D6) -------------------------------------------
        request = SetRuntimeMode.Request()
        request.mode = "trajectory"
        assert call(mode_client, request).success
        # a non-stowed target: differs from the stowed pose by > 0.1 rad
        assert max(abs(a - b) for a, b in zip(SMOKE_TARGET, STOWED_TARGET, strict=True)) > 0.1
        move_request = MoveToConfiguration.Request()
        move_request.target_joint_state.name = list(ARM_JOINTS)
        move_request.target_joint_state.position = [float(value) for value in SMOKE_TARGET]
        response = call(move_client, move_request, timeout=90.0)
        assert response.success and response.outcome == "SUCCEEDED", (
            f"MoveToConfiguration failed: outcome={response.outcome}: {response.message}"
        )
        # /joint_states reaches the target within 0.05 rad
        deadline = time.monotonic() + 30.0
        reach_error = node.arm_error(SMOKE_TARGET)
        while time.monotonic() < deadline and reach_error > REACH_TOLERANCE_RAD:
            rclpy.spin_once(node, timeout_sec=0.05)
            reach_error = node.arm_error(SMOKE_TARGET)
        measurements["move_to_configuration"] = {"reach_error_rad": reach_error}
        print(f"[smoke] move_to_configuration: {measurements['move_to_configuration']}", flush=True)
        assert reach_error <= REACH_TOLERANCE_RAD, (
            f"/joint_states is {reach_error:.4f} rad from the commanded configuration"
        )
        # FK of the reached state matches FK of the target within 1 cm
        reached = node.arm_positions()

        def fk_of(positions: dict[str, float]):
            request = ComputeFk.Request()
            request.joint_state.name = list(ARM_JOINTS)
            request.joint_state.position = [float(positions[joint]) for joint in ARM_JOINTS]
            request.link_names = ["gripper"]
            response = call(fk_client, request, timeout=30.0)
            assert response.success, f"compute_fk failed: {response.message}"
            assert len(response.poses) == 1, f"compute_fk returned {len(response.poses)} poses"
            return response.poses[0]

        target_stamped = fk_of({f"{index + 1}": value for index, value in enumerate(SMOKE_TARGET)})
        reached_stamped = fk_of(reached)
        target_pose, reached_pose = target_stamped.pose, reached_stamped.pose
        fk_error = math.dist(
            (target_pose.position.x, target_pose.position.y, target_pose.position.z),
            (reached_pose.position.x, reached_pose.position.y, reached_pose.position.z),
        )
        measurements["fk_consistency"] = {
            "fk_frame": target_stamped.header.frame_id,
            "position_error_m": fk_error,
        }
        print(f"[smoke] fk_consistency: {measurements['fk_consistency']}", flush=True)
        assert fk_error <= FK_TOLERANCE_M, f"FK of the reached state differs from the target by {fk_error:.4f} m"
        measurements["summary"] = "smoke and camera checks held"
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _terminate(launch)
        terminate_marked_processes({RUN_MARKER_ENV: partition})
        print(f"[smoke] measurements: {measurements}", flush=True)


# ---------------------------------------------------------------------------
# Counter-examples for the camera structure checks (review round 2, no
# simulator — plain Image messages, padding included)
# ---------------------------------------------------------------------------


def _image(encoding: str, width: int, height: int, step: int, data: bytes) -> Image:
    frame = Image()
    frame.encoding = encoding
    frame.width = width
    frame.height = height
    frame.step = step
    frame.data = data
    return frame


def test_a_pure_color_frame_fails_even_with_varying_row_padding():
    # every pixel is (30, 144, 255); the 2 pad bytes per row vary — the old
    # raw-byte variance read that padding noise as "structure"
    width, height, pad = 4, 3, 2
    rows = [
        bytes((30, 144, 255)) * width + bytes((17 * (row + 1) & 0xFF, 231 * (row + 1) & 0xFF)) for row in range(height)
    ]
    frame = _image("rgb8", width, height, width * 3 + pad, b"".join(rows))
    assert min(_color_channel_variances(frame)) == 0.0


def test_a_gradient_color_frame_passes():
    width, height = 6, 4
    row = b"".join(bytes((10 * x, 200 - 10 * x, x)) for x in range(width))
    frame = _image("rgb8", width, height, width * 3, row * height)
    assert min(_color_channel_variances(frame)) > 0.0


def test_a_constant_depth_plane_fails_the_structure_check():
    # every pixel 1.0 m: valid fraction 100 %, but zero depth variance
    import struct

    width, height = 4, 3
    frame = _image("32FC1", width, height, width * 4, struct.pack("<f", 1.0) * width * height)
    fraction, variance = _depth_valid_variance(frame)
    assert fraction == 1.0
    assert variance == 0.0


def test_an_all_nan_depth_fails_the_valid_fraction_check():
    import math
    import struct

    width, height = 4, 3
    frame = _image("32FC1", width, height, width * 4, struct.pack("<f", math.nan) * width * height)
    fraction, variance = _depth_valid_variance(frame)
    assert fraction == 0.0
    assert variance == 0.0


def test_a_ranged_depth_frame_passes_and_ignores_padding():
    import struct

    width, height, pad = 4, 3, 4
    values = [0.4, 0.8, 1.2, 2.0] * height
    rows = [
        b"".join(struct.pack("<f", value) for value in values[row * width : (row + 1) * width]) + b"\xde\xad\xbe\xef"
        for row in range(height)
    ]
    frame = _image("32FC1", width, height, width * 4 + pad, b"".join(rows))
    fraction, variance = _depth_valid_variance(frame)
    assert fraction == 1.0
    assert variance > 0.0
