"""Uncertain object custody must leave the idle Gateway's operator recovery path available."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from ibrobot_msgs.srv import AuthorizeHeldReset
from skill_library.gateway_policy import RootExecutionLease, RuntimeSnapshot
from skill_library.skill_executor_node import SkillExecutorNode


class _ResetHarness:
    _authorize_held_reset = SkillExecutorNode._authorize_held_reset
    _manual_held_reset_is_current = SkillExecutorNode._manual_held_reset_is_current


def _reset_gateway(*, busy=False, unresolved=False):
    gateway, requests = _ResetHarness(), []
    snapshot = RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="moveit_planning",
        required_control_mode="moveit_planning",
        busy=busy,
        unresolved_unknown=unresolved,
        held_object_state="uncertain",
        held_object_reason="GRASP_UNCERTAIN",
    )
    gateway._state_guard = threading.RLock
    gateway._runtime_snapshot = lambda: snapshot
    gateway._held_object = SimpleNamespace(custody_id="custody-1", state="uncertain", reason_code="GRASP_UNCERTAIN")
    gateway._runtime_enabled = False
    gateway._gateway_lease = RootExecutionLease()
    gateway._rpc_timeout = 0.01
    gateway._manual_reset_authorization = None
    gateway._wait_for_future = lambda future, **_kwargs: future.done()
    gateway.get_logger = lambda: SimpleNamespace(info=lambda _message: None)

    def authorize(request):
        requests.append(request)
        future = Future()
        future.set_result(AuthorizeHeldReset.Response(success=True, custody_id="custody-1", remaining_sec=60.0))
        return future

    gateway._authorize_held_reset_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True, call_async=authorize
    )
    return gateway, requests


def _request(**overrides):
    return AuthorizeHeldReset.Request(
        **{
            "schema_version": 1,
            "request_id": "operator-check-1",
            "custody_id": "custody-1",
            "operator_confirmed_empty": True,
            "operator_note": "object removed and hand is empty",
            **overrides,
        }
    )


def test_idle_uncertain_custody_accepts_manual_recovery_without_retaining_the_gateway_lease():
    gateway, requests = _reset_gateway()
    response = gateway._authorize_held_reset(_request(), AuthorizeHeldReset.Response())
    assert response.success and len(requests) == 1
    assert gateway._gateway_lease.owner is None
    assert gateway._manual_held_reset_is_current()
    assert gateway._held_object.state == "uncertain"  # Only verified recovery clears custody.


@pytest.mark.parametrize("busy,unresolved", [(True, False), (False, True)])
def test_operator_reset_does_not_override_an_active_or_unconfirmed_motion(busy, unresolved):
    gateway, requests = _reset_gateway(busy=busy, unresolved=unresolved)
    response = gateway._authorize_held_reset(_request(), AuthorizeHeldReset.Response())
    assert not response.success and response.error_code == "SKILL_BUSY"
    assert not requests and gateway._gateway_lease.owner is None


def test_a_stale_custody_or_missing_operator_check_cannot_authorize_recovery():
    gateway, requests = _reset_gateway()
    for request, code in [
        (_request(custody_id="previous-custody"), "STALE_CUSTODY"),
        (_request(operator_confirmed_empty=False), "OPERATOR_CHECK_REQUIRED"),
    ]:
        response = gateway._authorize_held_reset(request, AuthorizeHeldReset.Response())
        assert not response.success and response.error_code == code
    assert not requests and not gateway._manual_held_reset_is_current()
