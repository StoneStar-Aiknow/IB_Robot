"""ROS-graph tests: the G1 vendor mock speaking the official SDK's vendor wire.

The mock builds and reads the official ``unitree_sdk2py`` IDL frames through
the adapter's converters on the profile's topics with the profile's QoS; these
tests drive it the way the bridge drives a robot — arm_sdk/hand frames in
through the adapter's frame mapping, LowState out through ``lowstate_q``, and
the RPC tier through the REAL SDK clients (the adapter's ``UnitreeSdk`` typed
clients) so the request/response contract is exercised end-to-end across the
DDS wire, not against a convenience mirror of it. Fault injection is exercised
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

import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import pytest as _pytest
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from unitree_g1_robot import vendor_gateway as _vg

if not _vg.VENDOR_AVAILABLE:
    # The vendor frames only exist with the official SDK installed (setup
    # full profile). Skip the module at collection instead of failing it —
    # mirrors the adapter suite's skip convention.
    _pytest.skip("unitree_sdk2py is not installed", allow_module_level=True)

from mock_process import MockProcess, vendor_test_domain
from unitree_g1_adapter.runtime import UnitreeSdk
from unitree_g1_robot import projection
from unitree_g1_robot.vendor_gateway import (
    HandState,
    LowState,
    RpcResult,
    handstate_q,
    lowstate_imu,
    lowstate_mode_machine,
    lowstate_q,
)
from unitree_g1_robot.vendor_mock_node import REJECT_CODE

#: The vendor DDS domain this process owns (see the workspace conftest). The SDK
#: factory is process-wide and cannot rebind between pytest modules, so
#: test_runtime_bridge must read the same per-process value — both call
#: ``vendor_test_domain()``.
TEST_DOMAIN_ID = vendor_test_domain()

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
with open(PROFILE) as _handle:
    _VENDOR: dict[str, Any] = yaml.safe_load(_handle)["vendor"]
_TOPICS: dict[str, str] = _VENDOR["topics"]
_RPC_REQUEST_TOPICS: dict[str, str] = _VENDOR["rpc"]["request_topics"]
_RPC_RESPONSE_TOPICS: dict[str, str] = _VENDOR["rpc"]["response_topics"]
_LOCO_SET_FSM_ID = int(projection.LOCO_API_IDS["set_fsm_id"])


class _MockFacade:
    """Watermark-scoped view of the shared mock (records of THIS test only)."""

    def __init__(self, mock: MockProcess, watermark: int) -> None:
        self._mock = mock
        self._watermark = watermark

    def record(self, name: str) -> Any:
        return self._mock.record(name, after_seq=self._watermark)

    def apply(self, overrides: dict[str, Any]) -> None:
        self._mock.apply(overrides)

    def set_parameters(self, parameters: list) -> None:
        self._mock.set_parameters(parameters)


class MockHarness:
    """One vendor mock SUBPROCESS plus a test client node, with bounded waits.

    The mock runs in its own process exactly as the launch entry composes it:
    the vendor binding's blocking writes must never share a GIL with this
    process. Observation rides this process's SDK feeds (cross-process DDS),
    fault injection rides the ROS parameter services, and the vendor-side
    request records come back through the mock's JSON-Lines records file.
    """

    def __init__(self, mock: MockProcess, sdk_surface, **mock_params: Any) -> None:
        self.node = rclpy.create_node("g1_vendor_mock_test_client")
        self._mock = mock
        self._mock.ensure_alive(node=mock.spawner_node, profile=PROFILE, domain_id=TEST_DOMAIN_ID)
        self._record_watermark = mock.seq_snapshot()
        self.mock = _MockFacade(mock, self._record_watermark)
        self._hands: dict[str, HandState | None] = {}
        self._hand_counts: dict[str, int] = {}
        self._watchers: dict[str, Any] = {}
        self._hand_seen: dict[str, int] = {}
        # The SDK facade, its writers and the RPC clients are MODULE-scoped
        # (see the sdk_surface fixture): per-test channel churn is what the
        # vendor binding does not survive.
        surface, writers = sdk_surface
        self._sdk_io = surface
        self._sdk = surface
        self._arm_sdk_writer = writers["arm_sdk"]
        self._hand_writers = {side: writers[f"hand_{side}"] for side in ("left", "right")}
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        # Fault injection rides the parameter services: the executor above
        # must already be spinning this node for the response to be serviced.
        self.mock.apply(mock_params)

    def shutdown(self) -> None:
        self._executor.shutdown(timeout_sec=2.0)
        # Stop the executor and let in-flight DDS listener callbacks drain.
        # The SDK surface and the mock subprocess are module-scoped.
        time.sleep(0.2)
        for feed in self._watchers.values():
            feed.close()
        self.node.destroy_node()

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

    # --- observation (SDK StateFeeds: rclpy cannot register the SDK IDL) ----

    def watch_lowstate(self) -> None:
        if "low_state" not in self._watchers:
            self._watchers["low_state"] = self._sdk_io.state_feed(_TOPICS["low_state"], "low_state")

    def latest_lowstate(self) -> Any | None:
        feed = self._watchers.get("low_state")
        return feed.last_sample if feed else None

    def lowstate_frame_count(self) -> int:
        # StateFeed caches only the newest sample; the count is approximated
        # by the mock's own tick, which increments once per frame.
        sample = self.latest_lowstate()
        return int(getattr(sample, "tick", 0)) if sample is not None else 0

    def lowstate_positions(self) -> dict[str, float]:
        assert self.latest_lowstate() is not None, "no lowstate observed yet"
        return projection.low_state_joint_positions(lowstate_q(self.latest_lowstate()))

    def watch_hand(self, side: str) -> None:
        if f"dex3_{side}_state" not in self._watchers:
            self._watchers[f"dex3_{side}_state"] = self._sdk_io.state_feed(_TOPICS[f"dex3_{side}_state"], "hand_state")

    def latest_hand(self, side: str) -> Any | None:
        feed = self._watchers.get(f"dex3_{side}_state")
        return feed.last_sample if feed else None

    def hand_count(self, side: str) -> int:
        # Same tick-approximation as lowstate_frame_count; HandState_.tick? the
        # hand IDL carries no tick — use presence + sample identity counter.
        sample = self.latest_hand(side)
        if sample is None:
            return self._hand_counts.get(side, 0)
        key = id(sample)
        if key != self._hand_seen.get(side):
            self._hand_seen[side] = key
            self._hand_counts[side] = self._hand_counts.get(side, 0) + 1
        return self._hand_counts.get(side, 0)

    # --- commanding -------------------------------------------------------------

    def send_arm_sdk(self, latch: dict[str, float], *, weight: float, count: int = 20) -> None:
        frame = projection.arm_sdk_lowcmd(
            latch,
            weight=weight,
            mode_pr=int(_VENDOR["mode_pr"]),
            mode_machine=int(_VENDOR["mode_machine"]),
        )
        for _ in range(count):
            assert self._arm_sdk_writer.write(frame), "the mock refused the arm_sdk frame"
            time.sleep(0.02)

    def send_hand_cmd(self, side: str, targets: dict[str, float], *, count: int = 15) -> None:
        frame = projection.dex3_handcmd(targets, side=side)
        for _ in range(count):
            assert self._hand_writers[side].write(frame["motor_cmd"]), "the mock refused the hand frame"
            time.sleep(0.02)

    def _sdk_clients(self) -> UnitreeSdk:
        """The real SDK clients, wired once per harness (Issue #171: the RPC
        request/response contract is exercised end-to-end through them)."""
        if self._sdk is None:
            self._sdk = self._sdk_io
            # The mock's RPC subscriptions must exist before the first client
            # write (a request published before DDS matching completes is
            # silently dropped): create the clients eagerly so the stubs'
            # own 0.5 s settle applies, then warm-probe one read-only call
            # with bounded retries until the cross-process match is live.
            self._sdk.loco(timeout_s=5.0)
            self._sdk.arm_action(timeout_s=5.0)
            self._sdk.audio(timeout_s=5.0)
            deadline = time.monotonic() + 15.0
            while True:
                probe = self._sdk.loco(timeout_s=1.0).get_fsm_id()
                if not probe.timed_out or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            assert not probe.timed_out, "the mock's loco service never answered its warm probe"
        return self._sdk

    def rpc_call(self, service: str, api_name: str, parameter: Any, timeout_s: float = 5.0) -> RpcResult:
        """One call through the real SDK client (the runtime's own path)."""
        sdk = self._sdk_clients()
        client = {"loco": sdk.loco, "arm": sdk.arm_action, "audio": sdk.audio}[service](timeout_s=timeout_s)
        method = {
            "get_fsm_id": "get_fsm_id",
            "set_fsm_id": "set_fsm_id",
            "set_velocity": "set_velocity",
            "switch_to_user_ctrl": "switch_to_user_ctrl",
            "switch_to_internal_ctrl": "switch_to_internal_ctrl",
            "execute_action": "execute_action",
            "get_action_list": "get_action_list",
            "tts": "tts_maker",
            "set_rgb_led": "led_control",
        }[api_name]
        args = tuple(parameter) if parameter is not None else ()
        outcome = getattr(client, method)(*args)
        from unitree_g1_adapter.clients import SdkCallResult  # noqa: PLC0415

        if isinstance(outcome, tuple) and len(outcome) == 2:
            code, data = outcome
            outcome = SdkCallResult(ok=int(code) == 0, code=int(code), data=data)
        from unitree_g1_robot.vendor_gateway import rpc_result_from_sdk  # noqa: PLC0415

        return rpc_result_from_sdk(outcome, api_id=api_name)


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
def sdk_surface():
    """One SDK facade for the whole module.

    The vendor binding is not robust under repeated in-process channel
    create/close churn (blocking writes holding the GIL, teardown
    segfaults); the suites therefore create their writers, watchers and RPC
    clients once and share them, exactly as a long-lived process would.
    """
    from unitree_g1_adapter.runtime import UnitreeSdk

    surface = UnitreeSdk()
    surface.initialize(TEST_DOMAIN_ID, "lo")
    writers = {
        "arm_sdk": surface.command_writer(_TOPICS["arm_sdk"], "low_cmd"),
        **{
            f"hand_{side}": surface.command_writer(_TOPICS[f"dex3_{side}_cmd"], "hand_cmd")
            for side in ("left", "right")
        },
    }
    yield surface, writers
    surface.close()


@pytest.fixture(scope="module")
def mock_process():
    """One mock subprocess per module — see test_runtime_bridge's note on
    participant churn."""
    spawner = rclpy.create_node("g1_mock_suite_spawner")
    # The parameter client lives on the spawner: spin it or the mock's
    # set_parameters calls never complete.

    _spin_executor = rclpy.executors.SingleThreadedExecutor()
    _spin_executor.add_node(spawner)
    threading.Thread(target=_spin_executor.spin, daemon=True).start()
    handle = MockProcess(
        node=spawner,
        profile=PROFILE,
        domain_id=TEST_DOMAIN_ID,
        records_file=Path(tempfile.gettempdir()) / f"g1_mock_suite_{os.getpid()}.jsonl",
    )
    handle.wait_ready()
    yield handle
    handle.terminate()
    _spin_executor.shutdown()
    spawner.destroy_node()


@pytest.fixture
def make_harness(mock_process, sdk_surface):
    """A harness factory; every harness it builds is torn down after the test."""
    running = []

    def _make(**mock_params: Any) -> MockHarness:
        harness = MockHarness(mock_process, sdk_surface, **mock_params)
        running.append(harness)
        return harness

    yield _make
    for harness in reversed(running):
        harness.shutdown()


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

    switch = harness.rpc_call("loco", "set_fsm_id", (4,))
    assert switch.ok and switch.code == 0

    query = harness.rpc_call("loco", "get_fsm_id", None)
    assert query.ok
    assert query.data == 4  # the SDK wrapper unwraps {"data": id} to the bare id


def test_reject_fsm_ids_answers_nonzero_and_keeps_the_old_fsm(make_harness):
    harness = make_harness()
    warm = harness.rpc_call("loco", "get_fsm_id", None)
    assert warm.ok  # the request/response path is proven before the fault is injected

    harness.mock.set_parameters([Parameter("reject_fsm_ids", value=[3])])
    result = harness.rpc_call("loco", "set_fsm_id", (3,))
    assert not result.ok
    assert result.code == REJECT_CODE

    after = harness.rpc_call("loco", "get_fsm_id", None)
    assert after.ok
    assert after.data == 0  # zero_torque, the initial fsm, was kept (SDK unwraps {"data": id})


def test_silent_api_ids_never_answer(make_harness):
    harness = make_harness()
    warm = harness.rpc_call("loco", "get_fsm_id", None)
    assert warm.ok

    harness.mock.set_parameters([Parameter("silent_api_ids", value=[_LOCO_SET_FSM_ID])])
    result = harness.rpc_call("loco", "set_fsm_id", (4,), timeout_s=1.5)
    assert result.timed_out  # no vendor code ever arrived (the SDK's sentinel outcome)


def test_get_action_list_returns_the_parameter_table(make_harness):
    harness = make_harness(action_list=["wave", "clap"])

    result = harness.rpc_call("arm", "get_action_list", None)
    assert result.ok
    assert result.data == {"names": ["wave", "clap"]}


def test_rpc_surface_answers_and_records_vendor_side_effects(make_harness):
    harness = make_harness()

    velocity = harness.rpc_call("loco", "set_velocity", (0.2, 0.0, 0.1, 1.0))
    assert velocity.ok
    assert harness.mock.record("last_velocity") == {"velocity": [0.2, 0.0, 0.1], "duration": 1.0}

    for api_name in ("switch_to_user_ctrl",):
        result = harness.rpc_call("loco", api_name, None)
        assert result.ok, api_name
    # SwitchToInternalCtrl takes the vendor InternalFsmMode (0=last per the SDK).
    result = harness.rpc_call("loco", "switch_to_internal_ctrl", (0,))
    assert result.ok

    action = harness.rpc_call("arm", "execute_action", (11,))
    assert action.ok
    assert harness.mock.record("last_action_id") == 11

    tts = harness.rpc_call("audio", "tts", ("hello", 0))
    assert tts.ok
    # The SDK's TtsMaker sends its (stuck-at-0) index, the text and the
    # speaker id — the documented upstream quirk lives inside the vendor
    # client, so the mock records exactly that wire payload.
    assert harness.mock.record("last_tts") == {"index": 0, "text": "hello", "speaker_id": 0}

    led = harness.rpc_call("audio", "set_rgb_led", (255, 0, 0))
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
