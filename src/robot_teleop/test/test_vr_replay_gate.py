"""Recorded Quest input through the full SDK-transport SO-101 stack.

A real launch: ``vr_teleop``, placo and the controllers run on the simulated
transport, and the recording enters through the TCP socket. The runtime profile
is rewritten so no serial port can be opened. Dependencies are mandatory: this
gate must never silently skip when placo or ROS is missing.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode
from robot_runtime import contract as C
from robot_teleop.vr_replay import RecordedPacket, load_recording, replay

# 10-29 vr_scenes_1, 85-95 s: trigger pressed at about 2.5 s, released at about 7.5 s.
FIXTURE = Path(__file__).parent / "fixtures/vr_replay/scenes_press_release_10s.ndjson.gz"
STARTUP_TIMEOUT_S = 90.0
# Runtime log: max_speed=4.0 rad/s at 50 Hz => 0.08 rad/step.
# The budget adds approximately 10% tolerance, rounded up.
# Do not derive this from the node's unrelated 2.0 rad/s default.
MAX_JOINT_STEP_RAD = 0.09
MAX_POSE_STEP_M = 0.05  # Regression budget, not a runtime rejection contract.
# vr_teleop's 0.2 s stale limit plus scheduling margin, below the runtime's 0.5 s bound,
# so losing the vr_teleop watchdog fails even though the runtime would still stop the arm.
MAX_HOLD_LATENCY_S = 0.4
GATE_BUDGET_S = 300.0


def _require_placo() -> None:
    """Fail the gate tests only: an import error at module level would abort the whole package's collection."""
    try:
        import placo  # noqa: F401
    except ImportError as error:
        pytest.fail(
            f"the VR replay gate requires placo ({error}); only the 'full' setup profile installs it. "
            "Run ./scripts/setup.sh --profile full.",
            pytrace=False,
        )


def _stop_process_group(process: subprocess.Popen) -> None:
    """Stop the launch and every child; children can outlive a launch leader that already exited."""
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=20.0)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _write_profile(directory: Path) -> Path:
    """SDK-transport runtime profile that cannot reach hardware.

    The production profile always launches the read-only leader source on a real
    serial port; the gate drops it and points the follower at a missing port with
    a neutral calibration, so an accidental hardware launch fails instead.
    """
    share = Path(get_package_share_directory("so101_robot"))
    profile = yaml.safe_load((share / "profiles/so101_single_arm.yaml").read_text())
    profile["teleoperation"].pop("leader_source", None)
    profile.update(use_sim=True, sim_platform="sdk")
    calibration = {
        str(motor): {"id": motor, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for motor in range(1, 7)
    }
    calibration_path = directory / "sim_calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    profile["hardware"].update(port="/dev/ibrobot_vr_replay_gate_no_hardware", calib_file=str(calibration_path))
    assert not Path(profile["hardware"]["port"]).exists()
    profile_path = directory / "so101_runtime_sim.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    return profile_path


def _write_config(directory: Path, port: int) -> Path:
    share = Path(get_package_share_directory("robot_config"))
    inputs = yaml.safe_load((share / "config/teleop/so101_leader_inputs.yaml").read_text())
    device = next(item for item in inputs["devices"] if item["name"] == "vr_teleop")
    device["vr_config"].update(host="127.0.0.1", port=port)
    input_path = directory / "inputs.yaml"
    input_path.write_text(yaml.safe_dump(inputs))
    config = yaml.safe_load((share / "config/robots/so101_single_arm.yaml").read_text())
    config["robot"]["runtime"]["profile"] = str(_write_profile(directory))
    config["robot"]["teleoperation"].update(active_device="vr_teleop", input_config=str(input_path))
    config_path = directory / "so101_vr_replay.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class Observer(Node):
    def __init__(self):
        super().__init__("vr_replay_gate_observer")
        self.commands: list[list[float]] = []
        self.poses: list[tuple[float, float, float]] = []
        self.gripper_commands: list[list[float]] = []
        self.joint_positions: dict[str, float] = {}
        self.first_command_start: dict[str, float] = {}
        self.get_status = self.create_client(GetRuntimeStatus, C.GET_STATUS_SERVICE)
        self.set_mode = self.create_client(SetRuntimeMode, C.SET_MODE_SERVICE)

    def bind_description(self):
        self.description = json.loads(self.runtime_status().interface_description_json)
        assert self.description["execution"] == "simulated"
        assert self.description["sim_platform"] == "sdk"
        interfaces = self.description["interfaces"]
        self.arm_command_topic = interfaces["joint.arm_stream"]["endpoint"]
        self.create_subscription(Float64MultiArray, self.arm_command_topic, self._on_command, 100)
        self.create_subscription(JointState, interfaces["joint.state"]["endpoint"], self._on_joint_state, 100)
        self.create_subscription(PoseStamped, interfaces["motion.arm.pose"]["endpoint"], self._on_pose, 100)
        self.create_subscription(
            Float64MultiArray,
            interfaces["joint.gripper_stream"]["endpoint"],
            lambda msg: self.gripper_commands.append(list(msg.data)),
            100,
        )

    def _on_pose(self, message):
        p = message.pose.position
        self.poses.append((p.x, p.y, p.z))

    def arm_simulation(self):
        # Only this test may clear the startup/release HOLD; verify the actual
        # runtime identity again before making the request.
        status = self.runtime_status()
        description = json.loads(status.interface_description_json)
        assert description["execution"] == "simulated" and description["sim_platform"] == "sdk"
        request = SetRuntimeMode.Request(mode=C.IDLE_MODE)
        future = self.set_mode.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        assert future.done() and future.result().success, "simulation idle unlock failed"
        self.clear_outputs()

    def clear_outputs(self):
        self.commands.clear()
        self.poses.clear()
        self.gripper_commands.clear()
        self.first_command_start = {}

    def output_counts(self) -> tuple[int, int, int]:
        return len(self.commands), len(self.poses), len(self.gripper_commands)

    def _on_command(self, message):
        if not self.commands:
            self.first_command_start = dict(self.joint_positions)
        self.commands.append(list(message.data))

    def _on_joint_state(self, message):
        self.joint_positions.update(zip(message.name, message.position, strict=True))

    def spin_for(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def runtime_status(self):
        if not self.get_status.service_is_ready():
            return None
        future = self.get_status.call_async(GetRuntimeStatus.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return future.result().status if future.done() else None

    def ready(self) -> bool:
        """Runtime connected (queried; the status topic only publishes on change), placo and vr_teleop up.

        vr_teleop latches HOLD as soon as it starts without input, so STOPPED is a ready state too.
        """
        if "vr_teleop" not in self.get_node_names():
            return False
        status = self.runtime_status()
        if status is None or status.lifecycle not in (C.LIFECYCLE_ACTIVE, C.LIFECYCLE_STOPPED):
            return False
        description = json.loads(status.interface_description_json or "{}")
        start = description.get("interfaces", {}).get("motion.arm.start")
        services = {name for name, _ in self.get_service_names_and_types()}
        return start is not None and start["endpoint"] in services


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    _require_placo()
    started = time.monotonic()
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "root conftest ROS domain isolation is required"
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    directory = tmp_path_factory.mktemp("vr_replay_gate")
    port = _free_port()
    config_path = _write_config(directory, port)
    log_path = directory / "launch.log"
    log = log_path.open("w")
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "robot_config",
            "robot.launch.py",
            "robot_config:=so101_single_arm",
            f"config_path:={config_path}",
            "control_mode:=teleop",
            "record:=false",
            "use_sim:=true",
            "sim_platform:=sdk",
            "moveit_display:=false",
        ],
        start_new_session=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    node = None
    try:
        rclpy.init()
        node = Observer()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not node.ready():
            assert process.poll() is None, f"launch exited; see {log_path}"
            assert time.monotonic() < deadline, f"stack not ready; see {log_path}"
            node.spin_for(0.5)
        # vr_teleop retries its startup stop until placo answers; let that settle.
        node.spin_for(2.0)
        node.bind_description()
        yield node, port, log_path
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_process_group(process)
        log.close()
        elapsed = time.monotonic() - started
        print(f"VR replay gate wall time (launch + replay + teardown): {elapsed:.3f}s; budget={GATE_BUDGET_S}s")
        assert elapsed < GATE_BUDGET_S


def _run_replay(node, port, packets, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(replay, packets, "127.0.0.1", port, **kwargs)
        while not future.done():
            node.spin_for(0.05)
        assert future.result() == len(packets)  # Propagate socket/thread failures.
    node.spin_for(0.6)


def _assert_steps(samples, threshold, label):
    assert len(samples) >= 10, f"{label}: too few samples; replay did not drive the stack"
    assert all(math.isfinite(v) for sample in samples for v in sample), f"{label}: non-finite sample"
    steps = [
        math.dist(left, right) if label == "pose" else max(abs(a - b) for a, b in zip(left, right, strict=True))
        for left, right in zip(samples, samples[1:], strict=False)
    ]
    maximum = max(steps)
    print(f"VR replay gate {label}: samples={len(samples)} max_step={maximum:.9f} margin={threshold - maximum:.9f}")
    assert maximum <= threshold, f"{label} step {maximum:.9f} > {threshold}"
    assert maximum > 1e-5, f"{label}: no meaningful motion"


def _assert_first_steps(poses, commands, start, names):
    """Adjacent-sample checks never see the first output, where a stale clutch baseline would jump."""
    assert poses and commands, "no output to check"
    assert start, "no joint feedback before the first command"
    pose_step = math.dist(poses[0], (0.0, 0.0, 0.0))  # clutch_relative: a session starts at zero displacement.
    joint_step = max(abs(value - start[name]) for name, value in zip(names, commands[0], strict=True))
    print(f"VR replay gate first step: pose={pose_step:.9f} joint={joint_step:.9f}")
    assert pose_step <= MAX_POSE_STEP_M, f"first pose step {pose_step:.9f} > {MAX_POSE_STEP_M}"
    assert joint_step <= MAX_JOINT_STEP_RAD, f"first joint step {joint_step:.9f} > {MAX_JOINT_STEP_RAD}"


def _assert_limits(commands, names, limits):
    assert commands, "no joint commands"
    for command in commands:
        assert len(command) == len(names)
        for name, value in zip(names, command, strict=True):
            assert math.isfinite(value) and limits[name]["min"] - 1e-6 <= value <= limits[name]["max"] + 1e-6, (
                f"joint {name} outside limits: {value} not in {limits[name]}"
            )


def _assert_hold(status, before, after):
    """``before`` and ``after`` are ``Observer.output_counts()``: arm, pose and gripper outputs must all be quiet."""
    assert status is not None and status.stop_latched and status.stop_policy == C.STOP_HOLD, (
        "expired stream did not HOLD"
    )
    assert status.lifecycle == C.LIFECYCLE_STOPPED
    assert after == before, f"commands continued after HOLD: {before} -> {after}"


def test_latched_runtime_is_not_moved_by_a_replayed_press(stack):
    """A latched runtime stays latched when a trigger press arrives.

    The stack latches HOLD at startup without input and at the end of every other test, so this test does not
    depend on running first. Re-arming after a stop is an explicit operator step (SetRuntimeMode idle), never a
    side effect of input.
    """
    node, port, log_path = stack
    before = node.runtime_status()
    assert before is not None and before.stop_latched, "precondition: the runtime should be latched in HOLD"
    node.clear_outputs()
    log_offset = log_path.stat().st_size

    _run_replay(node, port, load_recording(FIXTURE))

    after = node.runtime_status()
    assert after is not None and after.stop_latched, "the replayed press cleared the stop latch"
    assert node.output_counts() == (0, 0, 0), f"output while latched: {node.output_counts()}; see {log_path}"
    with log_path.open() as log:
        log.seek(log_offset)
        assert "enable refused" in log.read(), "the press never reached vr_teleop; the check is vacuous"


@pytest.mark.parametrize("fixture", [FIXTURE.name, "fast_press_8s.ndjson.gz"])
def test_unlocked_replay_steps_limits_and_hold(stack, fixture):
    node, port, _ = stack
    node.arm_simulation()
    _run_replay(node, port, load_recording(FIXTURE.with_name(fixture)))
    _assert_steps(node.commands, MAX_JOINT_STEP_RAD, "joint")
    _assert_steps(node.poses, MAX_POSE_STEP_M, "pose")
    model = node.description["model"]
    _assert_first_steps(node.poses, node.commands, node.first_command_start, model["joint_groups"]["arm"])
    _assert_limits(node.commands, model["joint_groups"]["arm"], model["joint_limits"])
    _assert_limits(node.gripper_commands, model["joint_groups"]["gripper"], model["joint_limits"])
    before = node.output_counts()
    node.spin_for(0.4)
    _assert_hold(node.runtime_status(), before, node.output_counts())


def test_first_step_assertion_rejects_a_jump_on_the_first_output():
    names = ["a"]
    _assert_first_steps([(0.0, 0.0, 0.0)], [[0.0]], {"a": 0.0}, names)
    with pytest.raises(AssertionError, match="first pose step"):
        _assert_first_steps([(0.3, 0.0, 0.0)], [[0.0]], {"a": 0.0}, names)
    with pytest.raises(AssertionError, match="first joint step"):
        _assert_first_steps([(0.0, 0.0, 0.0)], [[1.0]], {"a": 0.0}, names)


def test_connected_stream_stall_enters_hold_and_rejects_continued_commands(stack):
    node, port, _ = stack
    node.arm_simulation()
    packets = load_recording(FIXTURE)
    with ThreadPoolExecutor(max_workers=1) as pool:
        started = time.monotonic()
        future = pool.submit(replay, packets, "127.0.0.1", port, stall_at_s=5.0, stall_for_s=2.0)
        while time.monotonic() - started < 4.5:
            node.spin_for(0.02)
        active = node.runtime_status()
        assert active is not None and not active.stop_latched and active.active_mode == "stream", (
            "stall must interrupt an active stream"
        )
        assert len(node.commands) >= 10, "stall happened before any motion"
        while time.monotonic() - started < 5.0:
            node.spin_for(0.02)
        status = node.runtime_status()
        while not status.stop_latched and time.monotonic() - started < 5.0 + MAX_HOLD_LATENCY_S:
            node.spin_for(0.02)
            status = node.runtime_status()
        latency = time.monotonic() - started - 5.0
        print(f"VR replay gate stall: HOLD observed {latency:.3f}s after pause; limit={MAX_HOLD_LATENCY_S}s")
        assert not future.done(), "stall must be observed while TCP is still connected"
        assert status.stop_latched, f"no HOLD within {MAX_HOLD_LATENCY_S}s of the stream stalling"
        node.spin_for(0.1)  # Outputs already in flight when the latch landed.
        before = node.output_counts()
        node.spin_for(0.3)
        status = node.runtime_status()
        _assert_hold(status, before, node.output_counts())
        missing_hold = deepcopy(status)
        missing_hold.stop_latched = False
        with pytest.raises(AssertionError, match="expired stream did not HOLD"):
            _assert_hold(missing_hold, before, node.output_counts())
        # Mutation: a regressed producer continues commanding after HOLD.
        publisher = node.create_publisher(Float64MultiArray, node.arm_command_topic, 10)
        node.spin_for(0.2)
        publisher.publish(Float64MultiArray(data=node.commands[-1]))
        node.spin_for(0.2)
        with pytest.raises(AssertionError, match="commands continued"):
            _assert_hold(status, before, node.output_counts())
        node.destroy_publisher(publisher)
        after_injection = node.output_counts()
        while not future.done():
            node.spin_for(0.05)
        assert future.result() == len(packets)
        _assert_hold(node.runtime_status(), after_injection, node.output_counts())
    node.spin_for(0.6)


def test_raw_position_jump_is_caught_on_pose_stream(stack):
    node, port, _ = stack
    node.arm_simulation()
    packets = load_recording(FIXTURE)
    changed = []
    for packet in packets:
        raw = json.loads(packet.raw)
        if 4.0 <= packet.offset_s < 4.3:
            raw["right_controller"]["position"][0] += 1.0
        changed.append(RecordedPacket(packet.offset_s, json.dumps(raw).encode() + b"\n"))
    _run_replay(node, port, changed)
    with pytest.raises(AssertionError, match="pose step"):
        _assert_steps(node.poses, MAX_POSE_STEP_M, "pose")
    # The jump must stop at the target: the joints stay bounded through the real stack.
    model = node.description["model"]
    _assert_steps(node.commands, MAX_JOINT_STEP_RAD, "joint")
    _assert_limits(node.commands, model["joint_groups"]["arm"], model["joint_limits"])


def test_injected_joint_jump_and_limit_violation_are_caught(stack):
    node, port, _ = stack
    node.arm_simulation()
    _run_replay(node, port, load_recording(FIXTURE))
    model = node.description["model"]
    names = model["joint_groups"]["arm"]
    _assert_steps(node.commands, MAX_JOINT_STEP_RAD, "joint")
    _assert_limits(node.commands, names, model["joint_limits"])
    settled = node.output_counts()
    node.spin_for(0.2)
    _assert_hold(node.runtime_status(), settled, node.output_counts())
    # Inject at the observed controller boundary in the isolated SDK graph.
    publisher = node.create_publisher(Float64MultiArray, node.arm_command_topic, 10)
    node.spin_for(0.2)
    command = node.commands[-1].copy()
    command[0] = model["joint_limits"][names[0]]["max"] + 1.0
    before = len(node.commands)
    publisher.publish(Float64MultiArray(data=command))
    node.spin_for(0.2)
    node.destroy_publisher(publisher)
    assert len(node.commands) == before + 1 and node.commands[-1] == command, "injected command not received"
    with pytest.raises(AssertionError, match="joint step"):
        _assert_steps(node.commands, MAX_JOINT_STEP_RAD, "joint")
    with pytest.raises(AssertionError, match="outside limits"):
        _assert_limits(node.commands, names, model["joint_limits"])
