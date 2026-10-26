"""End-to-end tests: the G1 runtime bridge against the in-repo vendor mock.

The mock and the bridge run as real nodes on one DDS graph, so these exercise
the whole wrapper — startup generation check (D10), stream entry and the
shared-weight latch (D4), mode/FSM confirmation and stop ordering (D5), the
action-table check (D7), RPC timeout handling, and the named-motion surface —
without a robot. Observation happens where the design pins behaviour: on the
vendor wire (rt/arm_sdk frames, the mock's recorded RPC payloads) and on the
contract surface (RuntimeStatus, /joint_states).

The ROS context pins ``domain_id=148``: 110/111 are the other worktrees'
running ROS graphs and 147 is the vendor-mock suite's, and a fixed private
domain keeps these publishers and subscriptions from crossing any of them.
Every wait is bounded: a missing publisher, a silent service or a stuck stream
fails the test on its timeout instead of hanging it. A fixed short window is
used only to prove the *absence* of traffic, which cannot be polled for.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from unitree_api.msg import Request, Response
from unitree_g1_robot import projection
from unitree_g1_robot.runtime_node import UnitreeG1RuntimeNode
from unitree_g1_robot.vendor_gateway import RpcClient, handstate_q, qos_profile
from unitree_g1_robot.vendor_mock_node import UnitreeVendorMock
from unitree_hg.msg import HandState, LowCmd, LowState

from ibrobot_msgs.action import ExecuteNamedMotion
from ibrobot_msgs.srv import GetRuntimeStatus, SetLedPattern, SetRuntimeMode, SpeakText, StopRuntime

#: Private DDS domain for these tests; 110/111 are the other worktrees' graphs
#: and 147 is the vendor-mock suite's.
TEST_DOMAIN_ID = 148

TIMEOUT_S = 15.0

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
with open(PROFILE) as _handle:
    _PROFILE: dict[str, Any] = yaml.safe_load(_handle)
_VENDOR: dict[str, Any] = _PROFILE["vendor"]
_TOPICS: dict[str, str] = _VENDOR["topics"]
_QOS: dict[str, Any] = _VENDOR["qos"]
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


class RuntimeHarness:
    """One vendor mock plus the runtime bridge and a test client, with bounded-wait helpers."""

    def __init__(self, *, startup_timeout_s: float = 8.0, **mock_params: Any) -> None:
        mock_overrides = [Parameter("profile", value=str(PROFILE))]
        mock_overrides += [Parameter(name, value=value) for name, value in mock_params.items()]
        self.mock = UnitreeVendorMock(parameter_overrides=mock_overrides)
        self.runtime = UnitreeG1RuntimeNode(
            parameter_overrides=[
                Parameter("profile", value=str(PROFILE)),
                Parameter("startup_timeout_s", value=startup_timeout_s),
            ]
        )
        self.node = rclpy.create_node("g1_runtime_bridge_test_client")
        self.executor = MultiThreadedExecutor(num_threads=8)
        for node in (self.mock, self.runtime, self.node):
            self.executor.add_node(node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

        # --- observation state (guarded by _watch_lock) ----------------------
        self._watch_lock = threading.Lock()
        self._arm_sdk: list[LowCmd] = []
        self._joint_states: list[JointState] = []
        self._imu: list[Imu] = []
        self._lowstate: LowState | None = None
        self._hands: dict[str, HandState | None] = {"left": None, "right": None}
        self._loco_request_apis: list[int] = []
        self._clients: dict[tuple[type, str], Any] = {}
        self._publishers: dict[str, Any] = {}
        self._rpc: dict[str, RpcClient] = {}

        self.node.create_subscription(LowCmd, _TOPICS["arm_sdk"], self._on_arm_sdk, qos_profile(_QOS, "arm_sdk"))
        self.node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.node.create_subscription(Imu, "/imu/data", self._on_imu, _SENSOR_QOS)
        self.node.create_subscription(LowState, _TOPICS["low_state"], self._on_lowstate, qos_profile(_QOS, "low_state"))
        for side in ("left", "right"):
            self.node.create_subscription(
                HandState,
                _TOPICS[f"dex3_{side}_state"],
                lambda message, side=side: self._on_hand_state(side, message),
                qos_profile(_QOS, "dex3_state"),
            )
        self.node.create_subscription(
            Request,
            _VENDOR["rpc"]["request_topics"]["loco"],
            self._on_loco_request,
            qos_profile(_QOS, "rpc_request"),
        )

    def shutdown(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        for node in (self.node, self.runtime, self.mock):
            node.destroy_node()

    # --- watcher callbacks ---------------------------------------------------

    def _on_arm_sdk(self, message: LowCmd) -> None:
        with self._watch_lock:
            self._arm_sdk.append(message)
            del self._arm_sdk[:-200]

    def _on_joint_state(self, message: JointState) -> None:
        with self._watch_lock:
            self._joint_states.append(message)
            del self._joint_states[:-50]

    def _on_imu(self, message: Imu) -> None:
        with self._watch_lock:
            self._imu.append(message)
            del self._imu[:-20]

    def _on_lowstate(self, message: LowState) -> None:
        with self._watch_lock:
            self._lowstate = message

    def _on_hand_state(self, side: str, message: HandState) -> None:
        with self._watch_lock:
            self._hands[side] = message

    def _on_loco_request(self, message: Request) -> None:
        with self._watch_lock:
            self._loco_request_apis.append(int(message.header.identity.api_id))
            del self._loco_request_apis[:-500]

    # --- bounded waiting -------------------------------------------------------

    def wait_until(self, predicate, what: str, timeout_s: float = TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
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
        with self._watch_lock:
            return len(self._arm_sdk)

    def arm_sdk_weight(self) -> float | None:
        with self._watch_lock:
            message = self._arm_sdk[-1] if self._arm_sdk else None
        if message is None:
            return None
        return float(message.motor_cmd[projection.ARM_SDK_WEIGHT_INDEX].q)

    def arm_sdk_targets(self) -> dict[str, float] | None:
        """The joint targets the bridge last put on rt/arm_sdk, by joint name."""
        with self._watch_lock:
            message = self._arm_sdk[-1] if self._arm_sdk else None
        if message is None:
            return None
        return projection.low_state_joint_positions([slot.q for slot in message.motor_cmd])

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
            message = self._lowstate
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
        with self._watch_lock:
            return sum(1 for seen in self._loco_request_apis if seen == api_id)

    # --- direct vendor RPC (to verify the vendor side of a switch) --------------------------

    def vendor_fsm_id(self) -> Any:
        client = self._vendor_rpc("loco")
        identity_id = client.send(projection.LOCO_API_IDS["get_fsm_id"], {}, timeout_s=3.0)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            result = client.poll(identity_id)
            if result is not None:
                client.discard(identity_id)
                return result.data if result.ok else None
            time.sleep(0.02)
        raise AssertionError("the vendor never answered GetFsmId")

    def _vendor_rpc(self, service: str) -> RpcClient:
        if service not in self._rpc:
            request_pub = self.node.create_publisher(
                Request, _VENDOR["rpc"]["request_topics"][service], qos_profile(_QOS, "rpc_request")
            )
            client = RpcClient(publish=request_pub.publish, now=time.monotonic)
            self.node.create_subscription(
                Response,
                _VENDOR["rpc"]["response_topics"][service],
                client.on_response,
                qos_profile(_QOS, "rpc_response"),
            )
            self._rpc[service] = client
            # A request published before DDS matching completes is silently
            # dropped; the mock's subscription must be visible first.
            self.wait_until(
                lambda: request_pub.get_subscription_count() > 0,
                f"the {service} rpc subscription to appear",
            )
        return self._rpc[service]


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Own the ROS context on the pinned test domain, if nothing else already does."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init(domain_id=TEST_DOMAIN_ID)
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def make_harness():
    """A harness factory; every harness it builds is torn down after the test."""
    made: list[RuntimeHarness] = []

    def _make(**kwargs: Any) -> RuntimeHarness:
        running = RuntimeHarness(**kwargs)
        made.append(running)
        return running

    yield _make
    for running in made:
        running.shutdown()


@pytest.fixture
def harness(make_harness):
    return make_harness()


# --- startup (design D10) --------------------------------------------------------


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

    A waist that was commanded to 0.3 and left there through a stream exit
    must be held at 0.3 on re-entry — a zero- or default-filled latch would
    command a motion nobody asked for."""
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
    assert harness.set_mode("idle").success  # the weight ramps down; the mock holds 0.3
    frames = harness.arm_sdk_count()
    assert harness.set_mode("stream").success
    harness.wait_until(lambda: harness.arm_sdk_count() > frames, "fresh arm_sdk frames after re-entry")
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
    assert harness.vendor_fsm_id() == {"data": projection.FSM_IDS["start"]}
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
    assert harness.vendor_fsm_id() == {"data": 0}, "the rejected switch changed the vendor FSM"
    # Once the vendor accepts it, the same switch succeeds.
    harness.mock.set_parameters([Parameter("reject_fsm_ids", value=[])])
    assert harness.set_mode("locomotion").success


def test_stream_exit_ramps_the_weight_down_before_reporting_idle(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.wait_until(lambda: (harness.arm_sdk_weight() or 0.0) >= 0.2, "the shared weight to start ramping up")
    future = harness.set_mode_async("idle")
    observed_stream_while_weighted = False
    deadline = time.monotonic() + TIMEOUT_S
    while not future.done() and time.monotonic() < deadline:
        weight = harness.arm_sdk_weight()
        if weight is not None and weight > 0.0:
            assert harness.status().active_mode == "stream", "idle was reported before the weight reached 0"
            observed_stream_while_weighted = True
        time.sleep(0.05)
    assert future.done(), "the stream exit never answered"
    assert observed_stream_while_weighted, "the exit finished before the ramp-down was observable"
    response = future.result()
    assert response.success, response.message
    assert harness.status().active_mode == "idle"
    harness.wait_until(lambda: harness.arm_sdk_weight() == 0.0, "a final weight-0 frame", timeout_s=1.0)
    frames = harness.arm_sdk_count()
    time.sleep(0.3)  # a 50 Hz stream would have published ~15 more frames
    assert harness.arm_sdk_count() == frames, "arm_sdk frames kept leaving after the exit"


# --- stop (design D5) -----------------------------------------------------------------------


def test_hold_stop_closes_admission_then_zeroes_velocity_then_ramps_down(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    harness.publish_channel("arm_stream", [0.1] * 14)
    harness.wait_until(lambda: (harness.arm_sdk_weight() or 0.0) >= 0.2, "the shared weight to start ramping up")
    future = harness.stop_async("HOLD")
    # (1) admission closes first: latched while the stop is still working.
    harness.wait_until(lambda: harness.status().stop_latched, "the stop latch to engage")
    assert not future.done(), "the stop answered before its vendor steps completed"
    # (2) the explicit zero reaches the vendor while the stop is still running.
    harness.wait_until(
        lambda: harness.mock.record("last_velocity") == {"velocity": [0.0, 0.0, 0.0], "duration": 1.0},
        "the zero-velocity command to reach the vendor",
    )
    assert not future.done(), "the stop answered before the weight handover completed"
    # (3) the ramp-down runs inside the stop window, after the zero.
    weight_at_zero = harness.arm_sdk_weight()
    assert weight_at_zero is not None and weight_at_zero > 0.0, "the weight was already 0 when the zero landed"
    # (4) the response arrives only after the handover completed.
    deadline = time.monotonic() + TIMEOUT_S
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert future.done()
    response = future.result()
    assert response.success, response.message
    assert 0.0 <= response.cancel_latency_s <= response.idle_latency_s
    assert harness.status().active_mode == "idle"
    harness.wait_until(lambda: harness.arm_sdk_weight() == 0.0, "a final weight-0 frame", timeout_s=1.0)
    frames = harness.arm_sdk_count()
    time.sleep(0.3)
    assert harness.arm_sdk_count() == frames, "arm_sdk frames kept leaving after the stop"
    # Latched: only idle is accepted, and it clears the latch.
    assert not harness.set_mode("stream").success
    assert harness.set_mode("idle").success
    assert not harness.status().stop_latched


def test_torque_off_is_refused_while_standing(harness):
    harness.wait_active()
    response = harness.stop("TORQUE_OFF")
    assert not response.success
    assert "TORQUE_OFF_UNSAFE_POSTURE" in response.message
    status = harness.status()
    assert not status.stop_latched, "a refused stop must not latch"
    assert status.active_mode == "idle"
    assert harness.vendor_fsm_id() == {"data": 0}, "the refused stop switched the FSM anyway"


def test_torque_off_from_sit_releases_through_damping(make_harness):
    harness = make_harness(initial_fsm_id=projection.FSM_IDS["sit"])
    harness.wait_active()
    response = harness.stop("TORQUE_OFF")
    assert response.success, response.message
    assert "TORQUE_OFF" in response.message and "damping" in response.message
    assert response.torque_off_latency_s >= 0.0
    assert harness.vendor_fsm_id() == {"data": projection.FSM_IDS["damping"]}
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
    assert response.success, response.message
    assert answered < 2.0, f"the stop queued behind the mode switch for {answered:.2f}s"
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
    switch = projection.LOCO_API_IDS["switch_to_user_ctrl"]
    harness = make_harness(silent_api_ids=[switch])
    harness.wait_active()
    response = harness.set_mode("stream")
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
    assert harness.vendor_fsm_id() == {"data": _VENDOR["postures"]["sit"]["fsm_id"]}


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
