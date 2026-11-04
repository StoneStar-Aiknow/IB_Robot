"""Preset completion allows new requests; uncertain outcomes do not."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from unitree_g1_robot.named_motion import NamedMotionServer
from unitree_g1_robot.runtime_node import UnitreeG1RuntimeNode
from unitree_g1_robot.vendor_gateway import RpcResult

from ibrobot_msgs.action import ExecuteNamedMotion
from robot_runtime.contract import LIFECYCLE_ACTIVE


@pytest.fixture
def server():
    state = SimpleNamespace(stop_latched=False, stop_epoch=0, lifecycle=LIFECYCLE_ACTIVE, faults=[])

    def stop(policy):
        state.stop_latched = True
        state.stop_epoch += 1

    state.engage_stop = stop
    node = SimpleNamespace(
        _preset_only=True,
        runtime_state=state,
        current_mode=lambda: "idle",
        admit_preset_attempt=Mock(return_value=(True, "ready")),
        rpc_call=Mock(return_value=SimpleNamespace(ok=True, code=0, timed_out=False)),
    )
    server = object.__new__(NamedMotionServer)
    server._node = node
    server._lock = threading.Lock()
    server._preset_dispatch_lock = threading.Lock()
    server._actions = {"wave": 26}
    server._allowed_modes = ["idle"]
    server._running = None
    return server


def test_success_allows_a_new_independent_request(server):
    assert server._run_action("wave", None, 1, 0)[0]
    assert server._run_action("wave", None, 2, 0)[0]
    assert server._node.rpc_call.call_count == 2


def test_inflight_request_rejects_concurrent_dispatch(server):
    def dispatch(*args, **kwargs):
        result = server._run_action("wave", None, 2, 0)
        assert result[1] == ExecuteNamedMotion.Result.BUSY
        return SimpleNamespace(ok=True, code=0, timed_out=False)

    server._node.rpc_call.side_effect = dispatch
    assert server._run_action("wave", None, 1, 0)[0]
    assert server._node.rpc_call.call_count == 1


@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(ok=False, code=3104, timed_out=True),
        SimpleNamespace(ok=False, code=3102, timed_out=False),
        RuntimeError("connection lost"),
    ],
)
def test_unknown_result_latches_and_never_resends(server, outcome):
    if isinstance(outcome, Exception):
        server._node.rpc_call.side_effect = outcome
    else:
        server._node.rpc_call.return_value = outcome
    result = server._run_action("wave", None, 1, 0)
    assert not result[0] and "unknown" in result[2]
    assert server._node.runtime_state.stop_latched
    assert not server._run_action("wave", None, 2, 1)[0]
    assert server._node.rpc_call.call_count == 1


def test_explicit_vendor_rejection_does_not_consume_session(server):
    server._node.rpc_call.side_effect = [
        SimpleNamespace(ok=False, code=7404, timed_out=False),
        SimpleNamespace(ok=True, code=0, timed_out=False),
    ]
    assert not server._run_action("wave", None, 1, 0)[0]
    assert not server._node.runtime_state.stop_latched
    assert server._run_action("wave", None, 2, 0)[0]


def test_stop_during_preflight_prevents_dispatch(server):
    def preflight():
        server._node.runtime_state.engage_stop("HOLD")
        return True, "ready"

    server._node.admit_preset_attempt.side_effect = preflight
    assert not server._run_action("wave", None, 1, 0)[0]
    server._node.rpc_call.assert_not_called()


@pytest.mark.parametrize(
    "outcome,detail",
    [
        (RpcResult(ok=False, code=3104, timed_out=True), "timed out (SDK code 3104)"),
        (RpcResult(ok=False, code=3102), "failed (SDK code 3102)"),
        (RpcResult(ok=False, code=5201), "failed (SDK code 5201)"),
        (RpcResult(ok=True, code=0, data={}), "invalid response (expected a service list)"),
    ],
)
def test_service_query_failure_reports_cause_without_dispatch_or_retry(server, outcome, detail):
    node = server._node
    node._lock = threading.RLock()
    node._state = node.runtime_state
    node._preset_physical = True
    node.rpc_call.return_value = outcome
    node.admit_preset_attempt = lambda: UnitreeG1RuntimeNode.admit_preset_attempt(node)

    success, code, message = server._run_action("wave", None, 1, 0)

    assert not success
    assert code == ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM
    assert "robot_state.service_list" in message
    assert detail in message
    assert "preset not dispatched" in message
    node.rpc_call.assert_called_once_with("robot_state", "service_list", None)
    assert not node.runtime_state.stop_latched
