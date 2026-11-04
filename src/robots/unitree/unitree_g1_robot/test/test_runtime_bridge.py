"""End-to-end tests: the G1 runtime bridge against the in-repo vendor mock.

The mock and the bridge run as real nodes on one DDS graph, so these exercise
the whole wrapper — startup generation check (D10), stream entry and the
shared-weight latch (D4), mode/FSM confirmation and stop ordering (D5), the
action-table check (D7), RPC timeout handling, and the named-motion surface —
without a robot. Observation happens where the design pins behaviour: on the
vendor wire (rt/arm_sdk frames, the mock's recorded RPC payloads) and on the
contract surface (RuntimeStatus, /joint_states).

The ROS context and the vendor tier both use the domain the workspace conftest
reserved for THIS pytest process (``IBROBOT_TEST_ROS_DOMAIN_ID``), so a parallel
suite cannot meet these publishers, subscriptions or vendor participants on a
shared graph — and no test carries a domain literal of its own.
Every wait is bounded: a missing publisher, a silent service or a stuck stream
fails the test on its timeout instead of hanging it. A fixed short window is
used only to prove the *absence* of traffic, which cannot be polled for.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import pytest as _pytest
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from unitree_g1_robot import vendor_gateway as _vg

if not _vg.VENDOR_AVAILABLE:
    # The vendor frames only exist with the official SDK installed (setup
    # full profile). Skip the module at collection instead of failing it —
    # mirrors the adapter suite's skip convention.
    _pytest.skip("unitree_sdk2py is not installed", allow_module_level=True)

from mock_process import (  # noqa: E402 - the module-level vendor-defect skip must run first
    MockProcess,
    vendor_test_domain,
)
from unitree_g1_adapter.runtime import UnitreeSdk  # noqa: E402 - the module-level vendor-defect skip must run first
from unitree_g1_robot import projection  # noqa: E402 - the module-level vendor-defect skip must run first
from unitree_g1_robot.runtime_node import (  # noqa: E402 - the module-level vendor-defect skip must run first
    UnitreeG1RuntimeNode,
)
from unitree_g1_robot.vendor_gateway import (  # noqa: E402 - the module-level vendor-defect skip must run first
    handstate_q,
)

from ibrobot_msgs.action import ExecuteNamedMotion  # noqa: E402 - the module-level vendor-defect skip must run first
from ibrobot_msgs.srv import (  # noqa: E402 - the module-level vendor-defect skip must run first
    GetRuntimeStatus,
    SetLedPattern,
    SetRuntimeMode,
    SpeakText,
    StopRuntime,
)

#: The vendor DDS domain this process owns (see the workspace conftest): the
#: ROS context and the vendor tier share it so both are isolated together.
TEST_DOMAIN_ID = vendor_test_domain()

TIMEOUT_S = 15.0

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
with open(PROFILE) as _handle:
    _PROFILE: dict[str, Any] = yaml.safe_load(_handle)
_VENDOR: dict[str, Any] = _PROFILE["vendor"]
_TOPICS: dict[str, str] = _VENDOR["topics"]
_CHANNELS = {str(entry["channel"]): entry for entry in _PROFILE["command_channels"]}
JOINTS = [str(joint) for joint in _PROFILE["joints"]]

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _named_client(harness: RuntimeHarness):
    """One action client on the contract's named-motion endpoint."""
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S), "the named motion server never appeared"
    return client


def _send_goal(harness: RuntimeHarness, client, name: str, *, interrupt: bool = False):
    goal = ExecuteNamedMotion.Goal()
    goal.name = name
    goal.interrupt = interrupt
    handle_future = client.send_goal_async(goal)
    harness.wait_until(lambda: handle_future.done(), f"the {name!r} goal decision")
    return handle_future.result()


class _MockFacade:
    """Watermark-scoped view of the shared mock (records of THIS test only)."""

    def __init__(self, mock: MockProcess, watermark: int) -> None:
        self._mock = mock
        self._watermark = watermark
        self.records_file = mock.records_file

    def record(self, name: str) -> Any:
        return self._mock.record(name, after_seq=self._watermark)

    def apply(self, overrides: dict[str, Any]) -> None:
        self._mock.apply(overrides)

    def set_parameters(self, parameters: list) -> None:
        self._mock.set_parameters(parameters)


class RuntimeHarness:
    """One vendor mock plus the runtime bridge and a test client, with bounded-wait helpers."""

    def __init__(
        self,
        mock: MockProcess,
        *,
        startup_timeout_s: float = 8.0,
        runtime_profile: Path = PROFILE,
        preset_test_transport: bool = False,
        **mock_params: Any,
    ) -> None:
        # The mock runs as a SUBPROCESS shared module-wide (the launch
        # composition; a per-test subprocess would leave undisposed vendor
        # participants whose stale reader endpoints poison the next test's
        # reliable writes for the whole discovery-lease window). Per-test
        # fault injection goes through the parameter services instead, from
        # the mock's profile baseline; records are isolated per test by a
        # sequence watermark.
        self._mock = mock
        self._mock.ensure_alive(node=mock.spawner_node, profile=PROFILE, domain_id=TEST_DOMAIN_ID)
        self._record_watermark = mock.seq_snapshot()
        self._records_file = mock.records_file
        self.mock = _MockFacade(mock, self._record_watermark)
        self.node = rclpy.create_node("g1_runtime_bridge_test_client")
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()
        # Fault injection rides the parameter services and must land BEFORE
        # the runtime node exists: its startup thread consumes the vendor
        # streams immediately, so a late mode_machine/hand_missing override
        # would race the D10 checks. The executor above services the
        # parameter responses; the runtime joins the executor afterwards.
        self.mock.apply({"initial_fsm_id": 500, **mock_params})
        self.runtime = UnitreeG1RuntimeNode(
            parameter_overrides=[
                Parameter("profile", value=str(runtime_profile)),
                Parameter("preset_test_transport", value=preset_test_transport),
                Parameter("vendor_network_interface", value="lo"),
                Parameter("startup_timeout_s", value=startup_timeout_s),
                # The subprocess mock speaks the vendor tier on the test
                # domain; the runtime's factory must bind the same one (the
                # launch entry passes the deployment's choice the same way).
                Parameter("vendor_domain_id", value=TEST_DOMAIN_ID),
            ]
        )
        self.executor.add_node(self.runtime)

        # --- observation state (guarded by _watch_lock) ----------------------
        self._watch_lock = threading.Lock()
        self._arm_sdk: list[Any] = []
        self._joint_states: list[JointState] = []
        self._imu: list[Imu] = []
        self._lowstate_sample: Any | None = None
        self._hands: dict[str, Any | None] = {"left": None, "right": None}
        self._clients: dict[tuple[type, str], Any] = {}
        self._publishers: dict[str, Any] = {}
        # Vendor-tier observation rides the SDK (rclpy cannot register the
        # SDK's IdlStruct types): one facade per harness, feeds cache the
        # newest samples and the loco-request watcher counts api ids.
        self._sdk = UnitreeSdk()
        self._sdk.initialize(TEST_DOMAIN_ID, "lo")
        self._sdk.state_feed(_TOPICS["low_state"], "low_state", handler=self._on_lowstate_sample, queue_len=1)
        for side in ("left", "right"):
            self._sdk.state_feed(
                _TOPICS[f"dex3_{side}_state"],
                "hand_state",
                handler=(lambda sample, side=side: self._on_hand_state_sample(side, sample)),
                queue_len=1,
            )
        self.node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.node.create_subscription(Imu, "/imu/data", self._on_imu, _SENSOR_QOS)

    def shutdown(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        # Stop the executor, let in-flight DDS listener callbacks drain, then
        # close the in-process runtime's channels BEFORE its rclpy node dies.
        # The shared mock subprocess is owned by the module fixture.
        time.sleep(0.2)
        self._sdk.close()
        with contextlib.suppress(Exception):  # a faulted runtime's teardown must not mask the test result
            self.runtime._sdk.close()
        for node in (self.node, self.runtime):
            node.destroy_node()

    # --- watcher callbacks ---------------------------------------------------

    def _on_lowstate_sample(self, message: Any) -> None:
        with self._watch_lock:
            self._lowstate_sample = message

    def _on_hand_state_sample(self, side: str, message: Any) -> None:
        with self._watch_lock:
            self._hands[side] = message

    def _on_joint_state(self, message: JointState) -> None:
        with self._watch_lock:
            self._joint_states.append(message)
            del self._joint_states[:-50]

    def _on_imu(self, message: Imu) -> None:
        with self._watch_lock:
            self._imu.append(message)
            del self._imu[:-20]

    # --- bounded waiting -------------------------------------------------------

    def wait_until(self, predicate, what: str, timeout_s: float = TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            if not self._mock.alive():
                # The vendor binding crashed its process mid-test (observed
                # under sustained traffic plus reader churn). Respawn so the
                # NEXT test starts clean, and fail THIS one with the reason.
                self._mock.ensure_alive(node=self._mock.spawner_node, profile=PROFILE, domain_id=TEST_DOMAIN_ID)
                raise AssertionError(
                    f"the vendor mock subprocess died mid-test while waiting for {what!r} "
                    "(vendor binding crash); it was respawned for the next test"
                )
            last = predicate()
            if last:
                return last
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what} (last: {last!r})")

    def wait_active(self):
        def _active():
            status = self.status()
            return status if status.lifecycle == "ACTIVE" else None

        return self.wait_until(_active, "the runtime to reach ACTIVE")

    # --- contract surface --------------------------------------------------------

    def _client(self, service_type, endpoint):
        key = (service_type, endpoint)
        client = self._clients.get(key)
        if client is None:
            client = self.node.create_client(service_type, endpoint)
            self._clients[key] = client
        return client

    def call(self, service_type, endpoint, request, timeout_s: float = TIMEOUT_S):
        client = self._client(service_type, endpoint)
        assert client.wait_for_service(timeout_sec=timeout_s), f"{endpoint} never appeared"
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not future.done():
            time.sleep(0.02)
        assert future.done(), f"{endpoint} did not answer"
        return future.result()

    def status(self):
        return self.call(GetRuntimeStatus, "/runtime/get_status", GetRuntimeStatus.Request()).status

    def set_mode(self, mode: str):
        request = SetRuntimeMode.Request()
        request.mode = mode
        return self.call(SetRuntimeMode, "/runtime/set_mode", request)

    def set_mode_async(self, mode: str):
        request = SetRuntimeMode.Request()
        request.mode = mode
        client = self._client(SetRuntimeMode, "/runtime/set_mode")
        assert client.wait_for_service(timeout_sec=TIMEOUT_S)
        return client.call_async(request)

    def stop(self, policy: str):
        request = StopRuntime.Request()
        request.policy = policy
        return self.call(StopRuntime, "/runtime/stop", request)

    def stop_async(self, policy: str):
        request = StopRuntime.Request()
        request.policy = policy
        client = self._client(StopRuntime, "/runtime/stop")
        assert client.wait_for_service(timeout_sec=TIMEOUT_S)
        return client.call_async(request)

    # --- commanding -----------------------------------------------------------------

    def publish_channel(self, channel: str, values: list[float], count: int = 15) -> None:
        message = Float64MultiArray()
        message.data = [float(value) for value in values]
        self.publish(str(_CHANNELS[channel]["topic"]), message, count=count)

    def publish_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float, count: int = 15) -> None:
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self.publish("/cmd_vel", twist, message_type=Twist, count=count)

    def publish(self, topic: str, message, message_type=Float64MultiArray, count: int = 15) -> None:
        key = f"{topic}:{message_type.__name__}"
        publisher = self._publishers.get(key)
        if publisher is None:
            publisher = self.node.create_publisher(message_type, topic, 10)
            self._publishers[key] = publisher
        for _ in range(count):
            publisher.publish(message)
            time.sleep(0.02)

    # --- observation --------------------------------------------------------------------

    def arm_sdk_count(self) -> int:
        """arm_sdk frames recorded since this harness started (mock readout).

        The runtime writes rt/arm_sdk from THIS process, and the vendor
        binding deadlocks when a same-process reader observes those writes —
        so the frame readout comes from the mock's records file instead of a
        local SDK feed.
        """
        entries = self._mock_records("arm_sdk_frame")
        return len(entries)

    def arm_sdk_weight(self) -> float | None:
        entries = self._mock_records("arm_sdk_frame")
        return entries[-1]["weight"] if entries else None

    def arm_sdk_targets(self) -> dict[str, float] | None:
        """The joint targets the bridge last put on rt/arm_sdk, by joint name."""
        entries = self._mock_records("arm_sdk_frame")
        return entries[-1]["targets"] if entries else None

    def latest_joint_state(self) -> JointState | None:
        with self._watch_lock:
            return self._joint_states[-1] if self._joint_states else None

    def joint_position(self, name: str) -> float | None:
        message = self.latest_joint_state()
        if message is None or name not in message.name:
            return None
        return float(message.position[list(message.name).index(name)])

    def latest_imu(self) -> Imu | None:
        with self._watch_lock:
            return self._imu[-1] if self._imu else None

    def vendor_position(self, name: str) -> float | None:
        with self._watch_lock:
            message = self._lowstate_sample
        if message is None:
            return None
        return projection.low_state_joint_positions([state.q for state in message.motor_state])[name]

    def hand_positions(self, side: str) -> dict[str, float] | None:
        with self._watch_lock:
            message = self._hands[side]
        if message is None:
            return None
        return projection.dex3_joint_positions(handstate_q(message), side=side)

    def loco_requests(self, api_id: int) -> int:
        return sum(1 for seen in self._mock_records("loco_request") if seen == api_id)

    def _mock_records(self, name: str) -> list[Any]:
        """All of this test's records for ``name`` (watermark-scoped)."""
        if not self._records_file.exists():
            return []
        found: list[Any] = []
        try:
            with open(self._records_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("name") == name and int(entry.get("seq", 0)) > self._record_watermark:
                        found.append(entry.get("value"))
        except OSError:
            return found
        return found

    # --- direct vendor RPC (to verify the vendor side of a switch) --------------------------

    def vendor_fsm_id(self) -> Any:
        """The mock's own FSM report, read through a real SDK LocoClient."""
        deadline = time.monotonic() + 5.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                result = self._sdk.loco(timeout_s=3.0).get_fsm_id()
                if not result.timed_out:
                    return result.data if result.ok else None
            except Exception as exc:  # noqa: BLE001 - the mock may not be matched yet
                last_error = exc
            time.sleep(0.05)
        raise AssertionError(f"the vendor never answered GetFsmId (last error: {last_error})")


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Own the ROS context on the pinned test domain, if nothing else already does."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init(domain_id=TEST_DOMAIN_ID)
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture(scope="module")
def mock_process():
    """One mock subprocess per module: churning subprocesses would leave
    undisposed vendor participants whose stale reader endpoints poison the
    next test's reliable writes for the discovery-lease window."""
    spawner = rclpy.create_node("g1_bridge_mock_spawner")
    # The parameter client lives on the spawner: spin it or the mock's
    # set_parameters calls never complete.
    spin_executor = rclpy.executors.SingleThreadedExecutor()
    spin_executor.add_node(spawner)
    threading.Thread(target=spin_executor.spin, daemon=True).start()
    handle = MockProcess(
        node=spawner,
        profile=PROFILE,
        domain_id=TEST_DOMAIN_ID,
        records_file=Path(tempfile.gettempdir()) / f"g1_bridge_mock_{os.getpid()}.jsonl",
    )
    handle.wait_ready()
    yield handle
    handle.terminate()
    spin_executor.shutdown()
    spawner.destroy_node()


@pytest.fixture
def make_harness(mock_process):
    """A harness factory; every harness it builds is torn down after the test."""
    running_harnesses = []

    def _make(**kwargs: Any) -> RuntimeHarness:
        running = RuntimeHarness(mock_process, **kwargs)
        running_harnesses.append(running)
        return running

    yield _make
    for running in reversed(running_harnesses):
        running.shutdown()


@pytest.fixture
def harness(make_harness):
    return make_harness()


# --- startup (design D10) --------------------------------------------------------


def test_preset_only_reaches_active_without_hands_and_dispatches_only_15(make_harness):
    harness = make_harness(
        runtime_profile=PROFILE.with_name("g1_29dof_arm_presets.yaml"),
        preset_test_transport=True,
        hand_missing="both",
        initial_fsm_id=802,
        action_list=["both_hands_up"],
    )
    status = harness.wait_active()
    assert status.active_mode == "idle"
    assert "hand.multi_joint" not in status.capabilities
    assert "base.cmd_vel" not in status.capabilities
    assert "runtime.stop" not in status.capabilities
    client = _named_client(harness)
    goal = _send_goal(harness, client, "arms_horizontal")
    assert goal.accepted
    future = goal.get_result_async()
    harness.wait_until(future.done, "preset result")
    assert future.result().result.success
    assert harness.mock.record("last_action_id") == 15
    assert harness.arm_sdk_count() == 0
    assert harness.mock.record("last_velocity") is None
    assert harness.loco_requests(projection.LOCO_API_IDS["set_fsm_id"]) == 0
    assert harness.loco_requests(projection.LOCO_API_IDS["switch_to_user_ctrl"]) == 0
    assert not _send_goal(harness, client, "release_arm").accepted
    assert not harness.set_mode("stream").success
    stopped = harness.stop("HOLD")
    assert not stopped.success and stopped.idle_latency_s == -1.0
    assert not harness.set_mode("idle").success
    assert harness.mock.record("last_action_id") == 15  # no automatic recovery
    client.destroy()


def test_startup_reaches_active_in_idle_mode(harness):
    status = harness.wait_active()
    assert status.active_mode == "idle"
    assert status.runtime_name == "unitree_g1_robot"
    assert not status.stop_latched
    assert not status.faults
    for capability in ("runtime.stop", "motion.named", "joint.state", "hand.multi_joint"):
        assert capability in status.capabilities
    names = json.loads(status.capabilities_json)["motion.named"]["names"]
    assert set(names) == set(_VENDOR["arm_actions"]), "the mock reports the full action table"


def test_commands_outside_their_mode_are_not_dispatched(harness):
    """Admission is on the output path: nothing may reach the vendor from idle."""
    harness.wait_active()
    harness.publish_channel("arm_stream", [0.1] * 14)
    harness.publish_cmd_vel(0.5, 0.0, 0.3)
    status = harness.wait_until(
        lambda: (
            harness.status() if {"arm_stream", "base_velocity"} <= set(harness.status().rejected_channels) else None
        ),
        "the out-of-mode commands to be counted",
    )
    assert "arm_stream" in status.rejected_channels
    time.sleep(0.5)  # a dispatched command would have crossed in well under this
    assert harness.mock.record("last_velocity") is None, "SetVelocity left while idle"
    assert harness.arm_sdk_count() == 0, "an arm_sdk frame left while idle"


def test_joint_states_and_imu_republish_the_vendor_frame(harness):
    harness.wait_active()
    message = harness.wait_until(
        lambda: (
            harness.latest_joint_state()
            if harness.latest_joint_state() is not None and len(harness.latest_joint_state().name) == 43
            else None
        ),
        "a 43-joint /joint_states frame",
    )
    assert list(message.name) == JOINTS
    assert message.position == pytest.approx([0.0] * 43)
    imu = harness.wait_until(lambda: harness.latest_imu(), "an /imu/data message")
    assert imu.orientation.w == pytest.approx(1.0)
    assert imu.linear_acceleration.z == pytest.approx(9.81)


def test_mode_machine_mismatch_faults_and_blocks_the_command_channels(make_harness):
    harness = make_harness(mode_machine=15)
    status = harness.wait_until(
        lambda: harness.status() if harness.status().lifecycle == "FAULTED" else None,
        "the runtime to fault on the generation mismatch",
    )
    assert any("mode_machine" in fault for fault in status.faults)
    response = harness.set_mode("stream")
    assert not response.success
    assert "FAULTED" in response.message
    harness.publish_channel("arm_stream", [0.1] * 14)
    harness.wait_until(
        lambda: harness.status() if "arm_stream" in harness.status().rejected_channels else None,
        "the command to be refused while faulted",
    )
    time.sleep(0.5)
    assert harness.arm_sdk_count() == 0, "a command channel was served from a faulted runtime"


def test_missing_hand_state_faults_at_startup(make_harness):
    harness = make_harness(hand_missing="left", startup_timeout_s=1.5)
    status = harness.wait_until(
        lambda: harness.status() if harness.status().lifecycle == "FAULTED" else None,
        "the runtime to fault on the missing hand",
        timeout_s=6.0,
    )
    assert any("left" in fault for fault in status.faults)


# --- streaming (design D4) ------------------------------------------------------------


def test_stream_entry_latches_and_streams_arm_targets(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.publish_channel("arm_stream", [0.1] * 14)
    targets = harness.wait_until(
        lambda: (
            harness.arm_sdk_targets()
            if harness.arm_sdk_targets() is not None
            and harness.arm_sdk_targets()["left_elbow_joint"] == pytest.approx(0.1)
            else None
        ),
        "the commanded arm targets on rt/arm_sdk",
    )
    # The unselected waist slots keep the entry latch: measured zeros, no drift.
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        assert targets[name] == pytest.approx(0.0), f"{name} drifted off its latched hold value"

    def converged(name: str, target: float) -> bool:
        value = harness.vendor_position(name)
        return value is not None and abs(value - target) < 1e-6

    # The weight ramp hands the frame over gradually; full convergence needs
    # the whole ramp (weight_ramp_rate 0.2/s from 0 to 1).
    harness.wait_until(
        lambda: converged("left_elbow_joint", 0.1), "the mock's arm position to converge on the target", timeout_s=10.0
    )
    assert converged("waist_yaw_joint", 0.0), "the uncommanded waist moved"


def test_hand_stream_reaches_the_vendor_hand(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    payload = [0.3, 0.3, 0.2, -0.2, -0.2, -0.2, -0.2]
    harness.publish_channel("hand_left_stream", payload)
    positions = harness.wait_until(lambda: harness.hand_positions("left"), "the left hand state to carry the command")
    assert [positions[name] for name in projection.dex3_joint_names("left")] == pytest.approx(payload)
    harness.wait_until(
        lambda: harness.joint_position("left_hand_thumb_0_joint") == pytest.approx(0.3),
        "the finger positions on /joint_states",
    )
    assert harness.joint_position("left_hand_index_1_joint") == pytest.approx(-0.2)


def test_stream_entry_latches_the_measured_waist_not_a_default(harness):
    """The entry latch comes from the measured positions, not a default.

    A commanded waist must keep its targets during release; an unacknowledged
    handover must block re-entry rather than seed a new default pose."""
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.publish_channel("waist_stream", [0.3, 0.3, 0.3])
    harness.wait_until(
        lambda: (
            harness.vendor_position("waist_yaw_joint") is not None
            and abs(harness.vendor_position("waist_yaw_joint") - 0.3) < 1e-6
        ),
        "the waist to converge on its commanded value",
        timeout_s=10.0,
    )
    response = harness.set_mode("idle")
    assert response.success, "the public mode contract holds once the weight is back to 0"
    status = harness.status()
    assert status.stop_latched, "admission stays closed until the operator clears it"
    assert any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in status.faults)
    assert not harness.set_mode("stream").success, "re-entry needs the latch cleared first"
    targets = harness.arm_sdk_targets()
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        assert targets[name] == pytest.approx(0.3), f"{name} was not latched from its measured position"


def test_a_stale_channel_keeps_its_last_commanded_value(harness):
    """Each channel expires on its own clock: a quiet waist holds its last command."""
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.publish_channel("waist_stream", [0.3, 0.3, 0.3])
    harness.wait_until(
        lambda: (
            harness.arm_sdk_targets() is not None and harness.arm_sdk_targets()["waist_yaw_joint"] == pytest.approx(0.3)
        ),
        "the waist command to reach the arm_sdk frame",
    )
    # The waist channel goes quiet; the arm channel keeps flowing well past the
    # profile's stream_staleness_s (0.5 s).
    harness.publish_channel("arm_stream", [0.1] * 14, count=60)  # ~1.2 s
    targets = harness.arm_sdk_targets()
    assert targets["waist_yaw_joint"] == pytest.approx(0.3), "the stale waist target was dropped or refilled"
    assert targets["waist_pitch_joint"] == pytest.approx(0.3)
    assert targets["left_elbow_joint"] == pytest.approx(0.1), "the fresh arm target stopped streaming"


def test_stream_entry_is_refused_without_fresh_feedback(harness):
    harness.wait_active()
    harness.mock.set_parameters([Parameter("publish_lowstate", value=False)])
    time.sleep(0.8)  # past the profile's feedback_staleness_s (0.5 s)
    response = harness.set_mode("stream")
    assert not response.success
    assert "stale" in response.message.lower()
    assert harness.status().active_mode == "idle"
    assert harness.loco_requests(projection.LOCO_API_IDS["switch_to_user_ctrl"]) == 0, (
        "SwitchToUserCtrl was sent without a fresh hold latch"
    )
    assert harness.arm_sdk_count() == 0


# --- modes and locomotion (design D5) ----------------------------------------------------


def test_locomotion_confirms_the_fsm_and_streams_cmd_vel(harness):
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    assert harness.status().active_mode == "locomotion"
    assert harness.vendor_fsm_id() == projection.FSM_IDS["start"]
    harness.publish_cmd_vel(0.5, 0.0, 0.3)
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.5, 0.0, 0.3], "duration": 1.0},
        "the SetVelocity command to reach the vendor",
    )


def test_a_changed_velocity_command_is_not_delayed_by_the_reissue_cadence(harness):
    """Only the renewal of the SAME command waits out its duration; a new
    command — above all a zero — goes out at once (publish + one 100 ms tick,
    far under the 1 s duration the old throttle would have waited out)."""
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    harness.publish_cmd_vel(0.5, 0.0, 0.0)
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.5, 0.0, 0.0], "duration": 1.0},
        "the first velocity command to reach the vendor",
    )
    started = time.monotonic()
    harness.publish_cmd_vel(0.0, 0.0, 0.0, count=3)
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.0, 0.0, 0.0], "duration": 1.0},
        "the changed command to reach the vendor",
    )
    assert time.monotonic() - started < 0.3, "a changed command waited out the previous command's duration"


def test_leaving_locomotion_zeroes_the_base_velocity(harness):
    """The last SetVelocity keeps its effect for its duration; leaving the mode
    must end it explicitly, or the robot keeps walking in a mode that no
    longer admits base commands."""
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    harness.publish_cmd_vel(0.5, 0.0, 0.0)
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.5, 0.0, 0.0], "duration": 1.0},
        "the velocity command to reach the vendor",
    )
    response = harness.set_mode("idle")
    assert response.success, response.message
    assert "zeroed" in response.message
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.0, 0.0, 0.0], "duration": 1.0},
        "the exit's zero-velocity command",
    )


def test_leaving_locomotion_fails_closed_when_the_zero_is_not_confirmed(make_harness):
    """A zero the vendor does not confirm fails the switch: the mode must not
    change while the base may still be commanded."""
    harness = make_harness(silent_api_ids=[projection.LOCO_API_IDS["set_velocity"]])
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    response = harness.set_mode("idle")
    assert not response.success
    assert "SetVelocity(0,0,0)" in response.message
    assert "refusing to leave locomotion" in response.message
    assert harness.status().active_mode == "locomotion"


def test_fsm_rejection_fails_set_mode_with_the_vendor_reason(harness):
    harness.wait_active()
    start = projection.FSM_IDS["start"]
    harness.mock.set_parameters([Parameter("reject_fsm_ids", value=[start])])
    response = harness.set_mode("locomotion")
    assert not response.success
    assert f"SetFsmId({start})" in response.message
    assert "code" in response.message
    assert harness.status().active_mode == "idle"
    assert harness.vendor_fsm_id() == 500, "the rejected switch changed the vendor FSM"
    # Once the vendor accepts it, the same switch succeeds.
    harness.mock.set_parameters([Parameter("reject_fsm_ids", value=[])])
    assert harness.set_mode("locomotion").success


def test_stream_exit_ramps_down_and_reports_the_unconfirmed_handover(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.wait_until(lambda: (harness.arm_sdk_weight() or 0.0) >= 0.2, "the shared weight to start ramping up")
    future = harness.set_mode_async("idle")
    observed_stream_while_weighted = False
    deadline = time.monotonic() + TIMEOUT_S
    while not future.done() and time.monotonic() < deadline:
        weight = harness.arm_sdk_weight()
        if weight is not None and weight > 0.0:
            assert harness.status().stop_latched, "exit must close admission during ramp-down"
            observed_stream_while_weighted = True
        time.sleep(0.05)
    assert future.done(), "the stream exit never answered"
    assert observed_stream_while_weighted, "the exit finished before the ramp-down was observable"
    response = future.result()
    assert response.success, "the exit is a settled mode switch once the weight is 0"
    status = harness.status()
    assert status.active_mode == "idle"
    assert status.stop_latched, "admission stays closed until the operator clears it"
    assert any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in status.faults), status.faults
    assert "not acknowledged" in response.message
    harness.wait_until(lambda: harness.arm_sdk_weight() == 0.0, "a final weight-0 frame", timeout_s=1.0)
    frames = harness.arm_sdk_count()
    time.sleep(0.3)  # a 50 Hz stream would have published ~15 more frames
    assert harness.arm_sdk_count() == frames, "arm_sdk frames kept leaving after the exit"


# --- stop (design D5) -----------------------------------------------------------------------


def test_arm_hold_stop_closes_admission_without_base_command(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.publish_channel("arm_stream", [0.1] * 14)
    harness.wait_until(lambda: (harness.arm_sdk_weight() or 0.0) >= 0.2, "the shared weight to start ramping up")
    future = harness.stop_async("HOLD")
    # (1) admission closes first: latched while the stop is still working.
    harness.wait_until(lambda: harness.status().stop_latched, "the stop latch to engage")
    assert not future.done(), "the stop answered before its vendor steps completed"
    # An arm-only session must not send base commands.
    assert harness.mock.record("last_velocity") is None
    assert not future.done(), "the stop answered before the weight handover completed"
    # The weight ramp continues while admission is closed.
    weight_at_zero = harness.arm_sdk_weight()
    assert weight_at_zero is not None and weight_at_zero > 0.0, "the weight was already 0 when the zero landed"
    # The response waits for local publication, without claiming remote handover.
    deadline = time.monotonic() + TIMEOUT_S
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert future.done()
    response = future.result()
    assert not response.success and "not acknowledged" in response.message
    assert response.cancel_latency_s >= 0.0 and response.idle_latency_s == -1.0
    assert harness.status().active_mode == "idle"
    harness.wait_until(lambda: harness.arm_sdk_weight() == 0.0, "a final weight-0 frame", timeout_s=1.0)
    frames = harness.arm_sdk_count()
    time.sleep(0.3)
    assert harness.arm_sdk_count() == frames, "arm_sdk frames kept leaving after the stop"
    # Re-entry stays refused until the operator clears the latch, and clearing it
    # is exactly what `idle` does — without claiming the vendor confirmed.
    assert not harness.set_mode("stream").success
    assert any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in harness.status().faults)
    assert harness.set_mode("idle").success
    status = harness.status()
    assert not status.stop_latched
    assert not any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in status.faults)
    assert not status.faults


def test_stop_after_a_cleared_stream_session_reports_success(harness):
    """N1: "entered stream once" must not read as "is in stream now".

    A stream session, its exit and the operator's idle clear leave no latched
    frame behind, so the next HOLD is an ordinary idle stop: it reports success
    and raises no handover warning, because there was no arm frame to hand back.
    """
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.wait_until(lambda: (harness.arm_sdk_weight() or 0.0) >= 0.2, "the shared weight to start ramping up")
    assert harness.set_mode("idle").success, "the exit is a settled mode switch"
    assert harness.status().stop_latched, "the exit closes admission until the operator clears it"
    cleared = harness.set_mode("idle")
    assert cleared.success, cleared.message
    status = harness.status()
    assert not status.stop_latched
    assert not any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in status.faults)

    response = harness.stop("HOLD")

    assert response.success, response.message
    assert response.idle_latency_s >= 0.0, "no arm handover is expected, so the idle latency is measured"
    assert not any("ARM_HANDOVER_UNCONFIRMED" in fault for fault in harness.status().faults)
    assert not harness.status().faults


def test_torque_off_is_refused_while_standing(harness):
    harness.wait_active()
    response = harness.stop("TORQUE_OFF")
    assert not response.success
    assert "TORQUE_OFF_UNSAFE_POSTURE" in response.message
    status = harness.status()
    # E3: admission closes before the FSM read, so the refusal covers the torque
    # release only; it must never reopen admission.
    assert status.stop_latched, "a stop request closes admission first"
    assert "stays engaged" in response.message
    assert status.active_mode == "idle"
    assert harness.vendor_fsm_id() == 500, "the refused stop switched the FSM anyway"
    assert harness.set_mode("idle").success and not harness.status().stop_latched


def test_torque_off_from_sit_releases_through_damping(make_harness):
    harness = make_harness(initial_fsm_id=projection.FSM_IDS["sit"])
    harness.wait_active()
    response = harness.stop("TORQUE_OFF")
    assert response.success, response.message
    assert "TORQUE_OFF" in response.message and "damping" in response.message
    assert response.torque_off_latency_s >= 0.0
    assert harness.vendor_fsm_id() == projection.FSM_IDS["damping"]
    assert harness.status().stop_latched
    assert harness.set_mode("idle").success
    assert not harness.status().stop_latched


def test_stop_preempts_a_mode_switch_waiting_on_the_vendor(make_harness):
    """A stop must never queue behind a set_mode that is waiting on the vendor."""
    harness = make_harness(silent_api_ids=[projection.LOCO_API_IDS["set_fsm_id"]])
    harness.wait_active()
    future = harness.set_mode_async("locomotion")
    requested = time.monotonic()
    response = harness.stop("HOLD")
    answered = time.monotonic() - requested
    assert answered < 2.0, f"the stop queued behind the mode switch for {answered:.2f}s"
    # The loco dispatcher is held by the mode switch's silent SetFsmId call,
    # so the stop's zero-velocity could not be dispatched within its bounded
    # priority window: the stop latched locally and reports the zero as
    # unconfirmed instead of blocking for the mode switch (Q7: the latch and
    # the admission close happen immediately, the vendor confirmation is
    # best-effort while another caller is mid-request).
    assert not response.success
    assert "unconfirmed" in response.message
    assert response.cancel_latency_s >= 0.0
    deadline = time.monotonic() + TIMEOUT_S
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert future.done(), "the interrupted set_mode never answered"
    mode_response = future.result()
    assert not mode_response.success
    assert "stop engaged" in mode_response.message
    status = harness.status()
    assert status.stop_latched and status.active_mode == "idle"


# --- RPC failure handling ----------------------------------------------------------------------


def test_rpc_timeout_fails_the_call_and_is_not_resent(make_harness):
    switch = projection.LOCO_API_IDS["set_fsm_id"]
    harness = make_harness(silent_api_ids=[switch])
    harness.wait_active()
    response = harness.set_mode("locomotion")
    assert not response.success
    assert "timed out" in response.message
    assert harness.status().active_mode == "idle"
    time.sleep(0.3)  # a retry would have crossed in this window
    assert harness.loco_requests(switch) == 1, "a motion-changing RPC was re-sent after its timeout"
    assert harness.arm_sdk_count() == 0


# --- named motions and postures (design D7) -------------------------------------------------------


def test_named_action_is_dispatched_and_reported(harness):
    harness.wait_active()
    client = _named_client(harness)
    handle = _send_goal(harness, client, "wave")
    assert handle.accepted
    result_future = handle.get_result_async()
    harness.wait_until(lambda: result_future.done(), "the named motion result")
    result = result_future.result().result
    assert result.success, result.message
    assert harness.mock.record("last_action_id") == _VENDOR["arm_actions"]["wave"]["action_id"]


def test_actions_missing_from_the_vendor_list_are_not_advertised(make_harness):
    harness = make_harness(action_list=[_VENDOR["arm_actions"]["wave"]["action_name"]])
    harness.wait_active()
    names = json.loads(harness.status().capabilities_json)["motion.named"]["names"]
    assert names == ["wave"], "an action the vendor does not report was advertised"
    client = _named_client(harness)
    assert not _send_goal(harness, client, "clap").accepted, "a motion outside the vendor list was accepted"
    handle = _send_goal(harness, client, "wave")
    assert handle.accepted
    result_future = handle.get_result_async()
    harness.wait_until(lambda: result_future.done(), "the confirmed action to run")
    assert result_future.result().result.success


def test_posture_is_confirmed_by_the_reported_fsm(harness):
    harness.wait_active()
    client = _named_client(harness)
    handle = _send_goal(harness, client, "sit")
    assert handle.accepted
    result_future = handle.get_result_async()
    harness.wait_until(lambda: result_future.done(), "the posture result")
    result = result_future.result().result
    assert result.success, result.message
    assert harness.vendor_fsm_id() == _VENDOR["postures"]["sit"]["fsm_id"]


def test_named_motion_cancel_is_refused(make_harness):
    """The platform cannot cancel a posture; the refusal is explicit, not silent."""
    harness = make_harness(silent_api_ids=[projection.LOCO_API_IDS["get_fsm_id"]])
    harness.wait_active()
    client = _named_client(harness)
    handle = _send_goal(harness, client, "sit")
    assert handle.accepted
    cancel_future = handle.cancel_goal_async()
    harness.wait_until(lambda: cancel_future.done(), "the cancel decision")
    assert not cancel_future.result().goals_canceling, "the cancel was accepted and then ignored"
    result_future = handle.get_result_async()
    harness.wait_until(lambda: result_future.done(), "the posture result", timeout_s=10.0)
    result = result_future.result().result
    assert not result.success
    assert result.error_code == ExecuteNamedMotion.Result.TIMEOUT


# --- interaction (design D7) -------------------------------------------------------


def test_speak_maps_onto_the_vendor_tts_call(harness):
    harness.wait_active()
    request = SpeakText.Request()
    request.text = "hello"
    response = harness.call(SpeakText, "/speech/speak", request)
    assert response.success, response.message
    sent = harness.wait_until(lambda: harness.mock.record("last_tts"), "a TTS request")
    assert sent == {"index": 0, "speaker_id": 0, "text": "hello"}
    # The vendor answers with no utterance id; inventing one would be a lie.
    assert response.utterance_id == ""


def test_speak_rejects_what_the_vendor_call_cannot_map(harness):
    """The vendor TTS parameter carries only index/speaker_id/text: language,
    priority and interrupt have no counterpart and are refused, not dropped."""
    harness.wait_active()
    rejects = []
    priority = SpeakText.Request()
    priority.text = "hi"
    priority.priority = 50
    rejects.append((priority, "INVALID_REQUEST"))
    language = SpeakText.Request()
    language.text = "hi"
    language.language = "en-US"
    rejects.append((language, "UNSUPPORTED_LANGUAGE"))
    interrupt = SpeakText.Request()
    interrupt.text = "hi"
    interrupt.interrupt = True
    rejects.append((interrupt, "INVALID_REQUEST"))
    empty = SpeakText.Request()
    empty.text = "   "
    rejects.append((empty, "INVALID_REQUEST"))
    for request, error_code in rejects:
        response = harness.call(SpeakText, "/speech/speak", request)
        assert not response.success, request.text or request.language or "interrupt"
        assert response.error_code == error_code
    assert harness.mock.record("last_tts") is None, "a rejected request reached the vendor"


def test_led_maps_onto_the_vendor_rgb_call(harness):
    harness.wait_active()
    request = SetLedPattern.Request()
    request.pattern = SetLedPattern.Request.PATTERN_SOLID
    request.r, request.g, request.b = 10, 20, 30
    response = harness.call(SetLedPattern, "/led/set_pattern", request)
    assert response.success, response.message
    sent = harness.wait_until(lambda: harness.mock.record("last_led"), "an LED request")
    assert sent == {"R": 10, "G": 20, "B": 30}
    # The vendor call reports no effective priority; echoing the request's
    # would claim one it does not hold.
    assert response.active_priority == 0


def test_led_rejects_what_the_vendor_call_cannot_map(harness):
    """SetRgbLed sets one static colour: only PATTERN_SOLID maps, and the call
    carries no priority or preemption."""
    harness.wait_active()
    pattern = SetLedPattern.Request()
    pattern.pattern = SetLedPattern.Request.PATTERN_BREATHE
    response = harness.call(SetLedPattern, "/led/set_pattern", pattern)
    assert not response.success
    assert response.error_code == "UNSUPPORTED_PATTERN"
    priority = SetLedPattern.Request()
    priority.pattern = SetLedPattern.Request.PATTERN_SOLID
    priority.priority = 30
    response = harness.call(SetLedPattern, "/led/set_pattern", priority)
    assert not response.success
    assert response.error_code == "INVALID_REQUEST"
    preempt = SetLedPattern.Request()
    preempt.pattern = SetLedPattern.Request.PATTERN_SOLID
    preempt.preempt = True
    response = harness.call(SetLedPattern, "/led/set_pattern", preempt)
    assert not response.success
    assert response.error_code == "INVALID_REQUEST"
    assert harness.mock.record("last_led") is None, "a rejected request reached the vendor"


# --- E1/E3: vendor DDS binding and stop ordering -------------------------------


def test_vendor_tier_binds_the_isolated_domain_on_loopback(harness):
    """E1: the Dex3 main path hands the domain/NIC to the SDK itself.

    The pair is a node parameter, not a ``CYCLONEDDS_URI`` side effect — the SDK
    builds its own domain configuration, so only this assertion shows where the
    vendor graph actually lives.
    """
    harness.wait_active()
    assert TEST_DOMAIN_ID != 0, "the isolated vendor domain is never the robot's production domain"
    assert harness.runtime._vendor_domain_id == TEST_DOMAIN_ID
    assert harness.runtime._vendor_network_interface == "lo"
    assert harness.runtime._sdk.initialized
    assert harness.runtime._sdk._domain_id == TEST_DOMAIN_ID
    assert harness.runtime._sdk._network_interface == "lo"


def test_stop_closes_admission_before_a_blocking_fsm_read(make_harness):
    """E3: the latch must never wait on a vendor query.

    With the FSM query unanswered, the stop still has to close admission at
    once — and the query must use its own 1 s bound, not the profile's 5 s.
    """
    harness = make_harness(
        initial_fsm_id=projection.FSM_IDS["sit"],
        silent_api_ids=[projection.LOCO_API_IDS["get_fsm_id"]],
    )
    harness.wait_active()
    requested = time.monotonic()
    response = harness.stop("TORQUE_OFF")
    answered = time.monotonic() - requested

    assert response.cancel_latency_s >= 0.0 and response.cancel_latency_s < 0.5
    assert harness.status().stop_latched, "admission must close before the vendor query"
    assert not response.success and "could not be read" in response.message
    assert "stays engaged" in response.message
    assert answered < 3.0, f"the FSM read used the profile timeout, not its own bound ({answered:.1f}s)"
