"""ROS-graph tests: the G1 vendor mock driven through the gateway's own conversions.

The mock speaks the vendored IDL on the profile's topics with the profile's
QoS; these tests drive it the way the bridge will drive a robot — LowCmd in
through ``lowcmd_from_dict``, LowState out through ``lowstate_q``, the RPC
tier through ``RpcClient`` — so a regression here is a regression against the
wire contract, not against a convenience mirror of it. The vendored message
packages are this repository's own build products, so a missing overlay is a
broken environment that fails loudly, not a skip. Fault injection is exercised
both as construction overrides and, where the task demands runtime
observability, through ``set_parameters`` on the live node.

The ROS context pins ``domain_id=147``: the environment's ``ROS_DOMAIN_ID`` is
untrusted here (unset means the shared domain 0, and the 110/111 range is used
by the other worktrees' running ROS graphs), and a fixed private domain keeps
these publishers and subscriptions from crossing any of them. Every wait is
bounded: a missing publisher, a silent service or a stuck stream fails the
test on its timeout instead of hanging it. A fixed short window is used only
to prove the *absence* of traffic, which cannot be polled for.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from unitree_api.msg import Request, Response
from unitree_g1_robot import projection
from unitree_g1_robot.vendor_gateway import (
    RpcClient,
    RpcResult,
    handcmd_from_dict,
    handstate_q,
    lowcmd_from_dict,
    lowstate_imu,
    lowstate_mode_machine,
    lowstate_q,
    qos_profile,
)
from unitree_g1_robot.vendor_mock_node import REJECT_CODE, UnitreeVendorMock
from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState

#: Private DDS domain for these tests; 110/111 are the other worktrees' graphs.
TEST_DOMAIN_ID = 147

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
with open(PROFILE) as _handle:
    _VENDOR: dict[str, Any] = yaml.safe_load(_handle)["vendor"]
_TOPICS: dict[str, str] = _VENDOR["topics"]
_QOS: dict[str, Any] = _VENDOR["qos"]
_RPC_REQUEST_TOPICS: dict[str, str] = _VENDOR["rpc"]["request_topics"]
_RPC_RESPONSE_TOPICS: dict[str, str] = _VENDOR["rpc"]["response_topics"]


class MockHarness:
    """One vendor mock plus a test client node, with bounded-wait helpers."""

    def __init__(self, **mock_params: Any) -> None:
        overrides = [Parameter("profile", value=str(PROFILE))]
        overrides += [Parameter(name, value=value) for name, value in mock_params.items()]
        self.mock = UnitreeVendorMock(parameter_overrides=overrides)
        self.node = rclpy.create_node("g1_vendor_mock_test_client")
        self._lowstate: LowState | None = None
        self._lowstate_count = 0
        self._hands: dict[str, HandState | None] = {}
        self._hand_counts: dict[str, int] = {}
        self._rpc: dict[str, RpcClient] = {}
        self._executor = MultiThreadedExecutor()
        for node in (self.mock, self.node):
            self._executor.add_node(node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._executor.shutdown(timeout_sec=2.0)
        for node in (self.node, self.mock):
            node.destroy_node()

    # --- bounded waiting ------------------------------------------------------

    def wait_until(self, predicate: Callable[[], Any], what: str, timeout_s: float = 5.0) -> Any:
        deadline = time.monotonic() + timeout_s
        last: Any = None
        while time.monotonic() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what} (last: {last!r})")

    # --- observation ------------------------------------------------------------

    def watch_lowstate(self) -> None:
        def _on_lowstate(message: LowState) -> None:
            self._lowstate = message
            self._lowstate_count += 1

        self.node.create_subscription(LowState, _TOPICS["low_state"], _on_lowstate, qos_profile(_QOS, "low_state"))

    def latest_lowstate(self) -> LowState | None:
        return self._lowstate

    def lowstate_frame_count(self) -> int:
        return self._lowstate_count

    def lowstate_positions(self) -> dict[str, float]:
        assert self._lowstate is not None, "no lowstate observed yet"
        return projection.low_state_joint_positions(lowstate_q(self._lowstate))

    def watch_hand(self, side: str) -> None:
        def _on_hand(message: HandState) -> None:
            self._hands[side] = message
            self._hand_counts[side] = self._hand_counts.get(side, 0) + 1

        self.node.create_subscription(
            HandState, _TOPICS[f"dex3_{side}_state"], _on_hand, qos_profile(_QOS, "dex3_state")
        )

    def latest_hand(self, side: str) -> HandState | None:
        return self._hands.get(side)

    def hand_count(self, side: str) -> int:
        return self._hand_counts.get(side, 0)

    # --- commanding -------------------------------------------------------------

    def send_arm_sdk(self, latch: dict[str, float], *, weight: float, count: int = 20) -> None:
        publisher = self.node.create_publisher(LowCmd, _TOPICS["arm_sdk"], qos_profile(_QOS, "arm_sdk"))
        message = lowcmd_from_dict(
            projection.arm_sdk_lowcmd(
                latch,
                weight=weight,
                mode_pr=int(_VENDOR["mode_pr"]),
                mode_machine=int(_VENDOR["mode_machine"]),
            )
        )
        for _ in range(count):
            publisher.publish(message)
            time.sleep(0.02)

    def send_hand_cmd(self, side: str, targets: dict[str, float], *, count: int = 15) -> None:
        publisher = self.node.create_publisher(HandCmd, _TOPICS[f"dex3_{side}_cmd"], qos_profile(_QOS, "dex3_cmd"))
        message = handcmd_from_dict(projection.dex3_handcmd(targets, side=side))
        for _ in range(count):
            publisher.publish(message)
            time.sleep(0.02)

    def rpc(self, service: str) -> RpcClient:
        if service not in self._rpc:
            request_pub = self.node.create_publisher(
                Request, _RPC_REQUEST_TOPICS[service], qos_profile(_QOS, "rpc_request")
            )
            client = RpcClient(publish=request_pub.publish, now=time.monotonic)
            self.node.create_subscription(
                Response, _RPC_RESPONSE_TOPICS[service], client.on_response, qos_profile(_QOS, "rpc_response")
            )
            self._rpc[service] = client
            # A request published before DDS matching completes is silently
            # dropped; the mock's subscription must be visible first.
            self.wait_until(
                lambda: request_pub.get_subscription_count() > 0,
                f"the {service} rpc subscription to appear",
            )
        return self._rpc[service]

    def rpc_call(self, service: str, api_id: int, parameter: Any, timeout_s: float = 5.0) -> RpcResult:
        client = self.rpc(service)
        identity_id = client.send(api_id, parameter, timeout_s=timeout_s)
        while True:  # poll resolves to a result or a timeout within timeout_s
            result = client.poll(identity_id)
            if result is not None:
                client.discard(identity_id)
                return result
            time.sleep(0.02)


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
    made: list[MockHarness] = []

    def _make(**mock_params: Any) -> MockHarness:
        running = MockHarness(**mock_params)
        made.append(running)
        return running

    yield _make
    for running in made:
        running.shutdown()


# --- lowstate -----------------------------------------------------------------


def test_lowstate_publishes_periodically_with_the_profile_mode_machine(make_harness):
    harness = make_harness()
    harness.watch_lowstate()
    first = harness.wait_until(lambda: harness.latest_lowstate(), "the first lowstate")

    assert lowstate_mode_machine(first) == int(_VENDOR["mode_machine"])
    imu = projection.imu_payload(lowstate_imu(first))
    assert imu["orientation"]["w"] == pytest.approx(1.0)

    harness.wait_until(lambda: harness.lowstate_frame_count() >= 6, "five more lowstate frames", timeout_s=2.0)


def test_mode_machine_override_reports_a_mismatched_generation(make_harness):
    harness = make_harness(mode_machine=15)
    harness.watch_lowstate()
    first = harness.wait_until(lambda: harness.latest_lowstate(), "the first lowstate")

    assert lowstate_mode_machine(first) == 15


# --- arm_sdk -------------------------------------------------------------------


def test_arm_sdk_weight_one_converges_and_weight_zero_holds(make_harness):
    harness = make_harness()
    harness.watch_lowstate()
    harness.wait_until(lambda: harness.latest_lowstate(), "the first lowstate")  # stream proven matched

    harness.send_arm_sdk({name: 0.5 for name in projection.ARM_SDK_FRAME}, weight=1.0)
    harness.wait_until(
        lambda: abs(harness.lowstate_positions()["left_elbow_joint"] - 0.5) < 1e-6,
        "the weight-1 frame to take effect",
    )

    # Weight 0 hands the frame back to internal control: new targets must not move it.
    held = harness.lowstate_positions()["left_elbow_joint"]
    tick_before = harness.latest_lowstate().tick
    harness.send_arm_sdk({name: 0.9 for name in projection.ARM_SDK_FRAME}, weight=0.0)
    harness.wait_until(
        lambda: (harness.latest_lowstate() or LowState()).tick >= tick_before + 10,
        "the lowstate stream to keep ticking",
    )
    assert abs(harness.lowstate_positions()["left_elbow_joint"] - held) < 1e-6


# --- dex3 hands ------------------------------------------------------------------


def test_hand_cmd_reaches_the_hand_state(make_harness):
    harness = make_harness()
    harness.watch_hand("left")
    harness.wait_until(lambda: harness.latest_hand("left"), "the first left hand state")

    targets = {"left_hand_thumb_0_joint": 0.3, "left_hand_index_1_joint": -0.2}
    harness.send_hand_cmd("left", targets)

    def finger(name: str) -> float:
        message = harness.latest_hand("left")
        assert message is not None
        return projection.dex3_joint_positions(handstate_q(message), side="left")[name]

    harness.wait_until(
        lambda: (
            abs(finger("left_hand_thumb_0_joint") - 0.3) < 1e-6 and abs(finger("left_hand_index_1_joint") + 0.2) < 1e-6
        ),
        "the hand targets to take effect",
    )


# --- RPC tier ----------------------------------------------------------------------


def test_set_fsm_id_then_get_fsm_id_round_trip(make_harness):
    harness = make_harness()

    switch = harness.rpc_call("loco", projection.LOCO_API_IDS["set_fsm_id"], projection.fsm_id_parameter(4))
    assert switch.ok and switch.code == 0

    query = harness.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {})
    assert query.ok
    assert query.data == {"data": 4}


def test_reject_fsm_ids_answers_nonzero_and_keeps_the_old_fsm(make_harness):
    harness = make_harness()
    warm = harness.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {})
    assert warm.ok  # the request/response path is proven before the fault is injected

    harness.mock.set_parameters([Parameter("reject_fsm_ids", value=[3])])
    result = harness.rpc_call("loco", projection.LOCO_API_IDS["set_fsm_id"], projection.fsm_id_parameter(3))
    assert not result.ok
    assert result.code == REJECT_CODE

    after = harness.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {})
    assert after.ok
    assert after.data == {"data": 0}  # zero_torque, the initial fsm, was kept


def test_silent_api_ids_never_answer(make_harness):
    harness = make_harness()
    warm = harness.rpc_call("loco", projection.LOCO_API_IDS["get_fsm_id"], {})
    assert warm.ok

    set_fsm_id = projection.LOCO_API_IDS["set_fsm_id"]
    harness.mock.set_parameters([Parameter("silent_api_ids", value=[set_fsm_id])])
    result = harness.rpc_call("loco", set_fsm_id, projection.fsm_id_parameter(4), timeout_s=1.5)
    assert result.timed_out and result.code == -1  # no vendor code ever arrived


def test_get_action_list_returns_the_parameter_table(make_harness):
    harness = make_harness(action_list=["wave", "clap"])

    result = harness.rpc_call("arm", projection.ARM_ACTION_API_IDS["get_action_list"], {})
    assert result.ok
    assert result.data == {"names": ["wave", "clap"]}


def test_rpc_surface_answers_and_records_vendor_side_effects(make_harness):
    harness = make_harness()

    velocity = harness.rpc_call(
        "loco", projection.LOCO_API_IDS["set_velocity"], projection.velocity_parameter(0.2, 0.0, 0.1)
    )
    assert velocity.ok
    assert harness.mock.record("last_velocity") == {"velocity": [0.2, 0.0, 0.1], "duration": 1.0}

    for api_name in ("switch_to_user_ctrl", "switch_to_internal_ctrl"):
        result = harness.rpc_call("loco", projection.LOCO_API_IDS[api_name], {})
        assert result.ok, api_name

    action = harness.rpc_call(
        "arm", projection.ARM_ACTION_API_IDS["execute_action"], projection.action_id_parameter(11)
    )
    assert action.ok
    assert harness.mock.record("last_action_id") == 11

    tts = harness.rpc_call("audio", projection.AUDIO_API_IDS["tts"], projection.tts_parameter("hello"))
    assert tts.ok
    assert harness.mock.record("last_tts") == {"index": 0, "speaker_id": 0, "text": "hello"}

    led = harness.rpc_call("audio", projection.AUDIO_API_IDS["set_rgb_led"], projection.led_parameter(255, 0, 0))
    assert led.ok
    assert harness.mock.record("last_led") == {"R": 255, "G": 0, "B": 0}


# --- runtime fault injection --------------------------------------------------------


def test_publish_lowstate_false_stops_the_feedback_stream(make_harness):
    harness = make_harness()
    harness.watch_lowstate()
    harness.wait_until(lambda: harness.latest_lowstate(), "the first lowstate")

    harness.mock.set_parameters([Parameter("publish_lowstate", value=False)])
    time.sleep(0.1)  # the toggle reads in the next timer tick; let the in-flight frame land
    frozen = harness.latest_lowstate().tick
    time.sleep(0.5)  # a live 50 Hz stream would have published ~25 frames in this window
    assert harness.latest_lowstate().tick == frozen, "lowstate kept publishing after the toggle"


def test_hand_missing_goes_quiet_on_exactly_one_side(make_harness):
    harness = make_harness()
    harness.watch_hand("left")
    harness.watch_hand("right")
    harness.wait_until(
        lambda: harness.hand_count("left") > 0 and harness.hand_count("right") > 0, "both hand state streams"
    )

    harness.mock.set_parameters([Parameter("hand_missing", value="right")])
    time.sleep(0.1)  # the toggle reads in the next timer tick; let the in-flight frame land
    left_before, right_before = harness.hand_count("left"), harness.hand_count("right")
    time.sleep(0.5)  # a live 50 Hz stream would have published ~25 frames per hand
    assert harness.hand_count("left") > left_before  # the other hand keeps reporting
    assert harness.hand_count("right") == right_before  # only the missing side went quiet
