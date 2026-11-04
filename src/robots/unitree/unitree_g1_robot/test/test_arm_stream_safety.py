"""Exercise real bridge methods with inert I/O: no ROS node, DDS or robot.

These regression tests do not replace the separate wire-level bridge suite.
"""

import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from unitree_g1_robot import projection, runtime_node
from unitree_g1_robot.runtime_node import UnitreeG1RuntimeNode

from ibrobot_msgs.srv import StopRuntime


class State:
    stop_epoch = 0
    stop_latched = False
    lifecycle = "ACTIVE"

    def __init__(self):
        self.faults = []

    def engage_stop(self, policy):
        self.stop_latched = True
        self.stop_epoch += 1

    def add_fault(self, detail):
        if detail not in self.faults:
            self.faults.append(detail)

    def set_status(self, lifecycle, faults):
        self.faults = list(dict.fromkeys(faults))

    def clear_stop(self):
        self.stop_latched = False


class Modes:
    mode = "stream"

    def commit(self, mode):
        self.mode = mode

    def spec(self):
        return SimpleNamespace(
            allows_stream=self.mode == "stream",
            allows_base=self.mode == "locomotion",
        )


@pytest.fixture
def bridge():
    node = object.__new__(UnitreeG1RuntimeNode)
    node._preset_only = False
    node._lock = threading.RLock()
    node._stop_lock = threading.Lock()
    node._state = State()
    node._modes = Modes()
    # Stream admission confirms the service ownership the profile declares, so
    # the fixture declares the same pair the shipped profile does.
    node._vendor = {
        "arm_sdk": {"allowed_fsm_ids": [4, 500, 501]},
        "arm_admission": {"stream_entry": {"ai_sport": 0, "g1_arm_example": 1}},
    }
    node._lowstate_stamp = time.monotonic()
    node._lowstate_positions = {name: 0.1 for name in projection.ARM_SDK_FRAME}
    node._feedback_window_s = 0.5
    node._latch = {}
    node._hand_latch = {"left": {}, "right": {}}
    node._weight = 0.0
    node._weight_target = 0.0
    node._arm_write_fault = None
    node._arm_write_pending = False
    node._arm_handover_unconfirmed = False
    node._arm_sdk_period_s = 0.02
    node._weight_ramp_up_rate = 0.2
    node._weight_ramp_down_rate = 0.5
    node._arm_sdk_kp = 60.0
    node._arm_sdk_kd = 1.5
    node._mode_pr = 1
    node._mode_machine_expected = 5
    node._velocity_inflight = False
    node._locomotion_requested = False
    node._startup_fault = None
    node.get_logger = Mock(return_value=Mock())
    node._publish_status = Mock()
    node._arm_sdk_writer = Mock()
    node._arm_sdk_writer.write.return_value = True
    node._stop_zero_velocity = Mock(side_effect=AssertionError("unexpected base command"))
    node.rpc_call = Mock(
        side_effect=[
            SimpleNamespace(
                ok=True,
                data=[
                    {"name": "ai_sport", "status": 0},
                    {"name": "g1_arm_example", "status": 1},
                ],
            ),
            SimpleNamespace(ok=True, data=500),
        ]
    )
    return node


def test_entry_uses_only_reads_and_holds_all_measured_joints(bridge):
    ok, _ = bridge._enter_stream(0)
    assert ok
    assert [call.args[:2] for call in bridge.rpc_call.call_args_list] == [
        ("robot_state", "service_list"),
        ("loco", "get_fsm_id"),
    ]
    assert all(not call.kwargs.get("motion") for call in bridge.rpc_call.call_args_list)
    assert bridge._latch == bridge._lowstate_positions
    assert bridge._weight == 0
    bridge._arm_sdk_writer.write.assert_not_called()


@pytest.mark.parametrize(
    "services",
    [
        [],
        [{"name": "ai_sport", "status": 1}],
        [{"name": "ai_sport", "status": 0}, {"name": "g1_arm_example", "status": 0}],
        [{"name": "ai_sport", "status": 0}, {"name": "g1_arm_example", "status": True}],
        [None],
        [{"name": "ai_sport", "status": 0}] * 2,
    ],
)
def test_unknown_or_conflicting_service_ownership_refuses_entry(bridge, services):
    bridge.rpc_call.side_effect = [SimpleNamespace(ok=True, data=services)]
    assert not bridge._enter_stream(0)[0]
    assert bridge._latch == {}
    bridge._arm_sdk_writer.write.assert_not_called()


@pytest.mark.parametrize("fsm", [0, 1, 3, 801, 802, None, True, "500"])
def test_unsupported_fsm_never_gets_changed(bridge, fsm):
    responses = list(bridge.rpc_call.side_effect)
    responses[1].data = fsm
    bridge.rpc_call.side_effect = responses
    assert not bridge._enter_stream(0)[0]
    assert bridge._latch == {}


def test_feedback_is_rechecked_after_rpc_queries(bridge):
    bridge._lowstate_stamp = time.monotonic() - 2
    assert "FEEDBACK_STALE" in bridge._enter_stream(0)[1]
    assert bridge._latch == {}


def test_stop_during_query_discards_pending_entry(bridge):
    responses = iter(bridge.rpc_call.side_effect)

    def query(*args, **kwargs):
        bridge._state.engage_stop("HOLD")
        return next(responses)

    bridge.rpc_call.side_effect = query
    assert not bridge._enter_stream(0)[0]
    assert bridge._latch == {}


@pytest.mark.parametrize("failure", [False, RuntimeError("unknown write outcome")])
def test_failed_zero_frame_is_not_a_successful_handover_or_retried(bridge, failure):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight = 0.005  # next decrement would reach zero
    bridge._weight_target = 0.0
    if isinstance(failure, Exception):
        bridge._arm_sdk_writer.write.side_effect = failure
    else:
        bridge._arm_sdk_writer.write.return_value = failure
    bridge._arm_sdk_tick()
    assert bridge._weight == 0.005
    assert bridge._state.stop_latched
    assert not bridge._await_weight_zero()[0]
    bridge._arm_sdk_tick()
    bridge._arm_sdk_writer.write.assert_called_once()
    assert not bridge._clear_stop_latch()


def test_successful_publication_does_not_claim_remote_stop(bridge):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight = 0.005
    bridge._arm_handover_unconfirmed = True
    bridge._arm_sdk_tick()
    assert bridge._await_weight_zero()[0]  # local publication only
    result = bridge._on_stop(SimpleNamespace(policy="HOLD"), StopRuntime.Response())
    assert not result.success
    assert result.idle_latency_s == -1
    assert "not acknowledged" in result.message
    bridge._stop_zero_velocity.assert_not_called()
    # The local handover is complete (weight 0, nothing pending), so the
    # operator path clears the latch — without the platform ever claiming the
    # vendor acknowledged it.
    assert bridge._clear_stop_latch()
    assert not bridge._arm_handover_unconfirmed


def test_stream_exit_succeeds_for_idle_and_keeps_the_warning_visible(bridge):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight = 0.005
    bridge._arm_handover_unconfirmed = True
    bridge._arm_sdk_tick()
    assert bridge._await_weight_zero()[0]

    ok, detail = bridge._exit_stream(target_mode="idle")

    assert ok, "the public mode contract must hold once the weight is back to 0"
    assert "not acknowledged" in detail, "the unconfirmed remote handover is still named"
    assert runtime_node.ARM_HANDOVER_WARNING in bridge._state.faults
    assert bridge._state.stop_latched, "admission stays closed until the operator clears it"
    assert bridge._arm_handover_unconfirmed, "the fact itself is not cleared by the exit"

    # Another target keeps the old refusal: only a settled handover may precede
    # a new vendor mode.
    ok, detail = bridge._exit_stream(target_mode="locomotion")
    assert not ok and "unconfirmed" in detail


def test_idle_clears_the_warning_and_reentry_still_runs_full_admission(bridge):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight = 0.005
    bridge._arm_handover_unconfirmed = True
    bridge._arm_sdk_tick()
    assert bridge._await_weight_zero()[0]
    assert bridge._exit_stream(target_mode="idle")[0]
    assert runtime_node.ARM_HANDOVER_WARNING in bridge._state.faults

    assert bridge._clear_stop_latch()
    assert bridge._state.stop_latched is False
    assert bridge._arm_handover_unconfirmed is False
    assert runtime_node.ARM_HANDOVER_WARNING not in bridge._state.faults

    # Re-entry is not a shortcut: the ownership/FSM checks run again and can
    # still refuse.
    bridge.rpc_call.side_effect = [
        SimpleNamespace(ok=True, data=[{"name": "ai_sport", "status": 1}, {"name": "g1_arm_example", "status": 1}])
    ]
    weight_target_before = bridge._weight_target
    ok, detail = bridge._enter_stream(bridge._state.stop_epoch)
    assert not ok and "ARM_OWNERSHIP_UNKNOWN" in detail
    assert bridge._weight_target == weight_target_before, "a refused re-entry must not admit the stream"


def test_stop_before_send_discards_unsent_ramp_up(bridge, monkeypatch):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight_target = 1.0
    build = projection.arm_sdk_lowcmd

    def stopped_frame(*args, **kwargs):
        frame = build(*args, **kwargs)
        bridge._state.engage_stop("HOLD")
        return frame

    monkeypatch.setattr(projection, "arm_sdk_lowcmd", stopped_frame)
    bridge._arm_sdk_tick()
    bridge._arm_sdk_writer.write.assert_not_called()


def test_write_in_flight_is_not_reported_as_zero(bridge):
    bridge._latch = dict(bridge._lowstate_positions)
    bridge._weight_target = 1.0

    def inspect_write(frame):
        assert bridge._arm_write_pending
        assert bridge._weight == 0.0
        assert bridge._arm_handover_unconfirmed
        return True

    bridge._arm_sdk_writer.write.side_effect = inspect_write
    bridge._arm_sdk_tick()
    assert bridge._weight > 0
    assert not bridge._arm_write_pending


class _FakeVendorClient:
    """The vendor client surface a LocoClient needs, recording the live timeout."""

    def __init__(self) -> None:
        self.timeout = None
        self.observed: list = []

    def Init(self) -> None:
        pass

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def GetFsmId(self):
        self.observed.append(self.timeout)
        return 0, {"data": 500}


def test_rpc_call_bounds_a_single_call_with_its_own_timeout():
    """E3: ``timeout_s`` is real — the vendor sees it for that call only."""
    from unitree_g1_adapter.clients import LocoClient

    fake = _FakeVendorClient()
    loco = LocoClient(vendor={"loco_client": lambda: fake, "rpc_ok": 0, "rpc_timeout": 3104}, timeout_s=5.0)
    node = object.__new__(UnitreeG1RuntimeNode)
    node._rpc = {"loco": loco}
    node._rpc_dispatchers = {"loco": threading.Lock()}

    result = node.rpc_call("loco", "get_fsm_id", None, timeout_s=0.5)

    assert result.ok and fake.observed == [0.5], "the short timeout must reach the vendor call"
    assert fake.timeout == 5.0 and loco.timeout_s == 5.0, "the configured default is restored"

    node.rpc_call("loco", "get_fsm_id", None)
    assert fake.observed[-1] == 5.0, "a call without timeout_s keeps the profile timeout"


class _RecordingLocoClient:
    """The loco client surface rpc_call needs, recording what actually went out."""

    timeout_s = 5.0

    def __init__(self):
        self.sent: list[tuple] = []

    @contextlib.contextmanager
    def call_timeout(self, timeout_s=None):
        yield

    def set_velocity(self, *args):
        self.sent.append(tuple(args))
        return SimpleNamespace(ok=True, code=0, timed_out=False, data=None, dropped=False)


def test_a_velocity_command_that_waits_out_a_stop_is_discarded(bridge):
    """N2: admission is re-checked under the dispatcher, not only when accepted.

    The worker starts while the test holds the loco dispatcher, so it can only
    reach the vendor after the stop below. The stale non-zero command must be
    dropped there — the stop's zero is the last velocity the vendor sees.
    """
    loco = _RecordingLocoClient()
    bridge._modes.mode = "locomotion"
    bridge._state.lifecycle = "ACTIVE"
    bridge._rpc = {"loco": loco}
    bridge._rpc_dispatchers = {"loco": threading.Lock()}
    # The fixture stubs the two vendor paths; this test needs them real.
    bridge.rpc_call = UnitreeG1RuntimeNode.rpc_call.__get__(bridge)
    bridge._stop_zero_velocity = UnitreeG1RuntimeNode._stop_zero_velocity.__get__(bridge)
    bridge._twist = (0.5, 0.0, 0.2)
    bridge._twist_stamp = time.monotonic()
    bridge._twist_staleness_s = 1.0
    bridge._velocity_duration_s = 1.0
    bridge._velocity_inflight = False
    bridge._velocity_sent = None
    bridge._last_velocity_sent_s = 0.0

    dispatcher = bridge._rpc_dispatchers["loco"]
    assert dispatcher.acquire(timeout=1.0), "the test takes the vendor first"
    bridge._velocity_tick()
    assert bridge._velocity_inflight, "the tick handed the command to a worker"
    # The stop lands while that worker is still waiting for the vendor.
    bridge._state.engage_stop("HOLD")
    bridge._modes.commit("idle")
    dispatcher.release()

    deadline = time.monotonic() + 5.0
    while bridge._velocity_inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not bridge._velocity_inflight, "the worker never finished"
    assert loco.sent == [], "a command the stop superseded must not reach the vendor"

    assert bridge._stop_zero_velocity().ok
    assert loco.sent == [(0.0, 0.0, 0.0, 1.0)], "only the stop's zero goes out"
