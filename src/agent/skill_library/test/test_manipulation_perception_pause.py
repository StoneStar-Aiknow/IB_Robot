"""Pick/place must pause before submission and release only after a definite terminal."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from embodied_common.dispatch_binding import new_binding
from ibrobot_msgs.action import PickObject, PlaceObject, SkillCommand
from ibrobot_msgs.srv import SetPerceptionPaused
from skill_library.skill_executor_node import SkillExecutorNode


class _GatewayHarness:
    _execute_pick_skill = SkillExecutorNode._execute_pick_skill
    _execute_place_skill = SkillExecutorNode._execute_place_skill
    _pause_motion_imitation = SkillExecutorNode._pause_motion_imitation
    _resume_motion_imitation = SkillExecutorNode._resume_motion_imitation
    _cancel_goal = SkillExecutorNode._cancel_goal
    _best_effort_cancel_goal = SkillExecutorNode._best_effort_cancel_goal
    _place_public_error = staticmethod(SkillExecutorNode._place_public_error)


@pytest.fixture(autouse=True)
def _running_context(monkeypatch):
    monkeypatch.setattr("skill_library.skill_executor_node.rclpy.ok", lambda: True)


def _done(value):
    future = Future()
    future.set_result(value)
    return future


def _gateway(kind, *, pause_ok=True, terminal="rejected", error_code="", recovery_state="NOT_NEEDED"):
    node = _GatewayHarness()
    calls = []
    pending = Future()

    class PauseClient:
        def wait_for_service(self, **_kwargs):
            return True

        def call_async(self, request):
            calls.append(("pause", request.owner_id, request.paused))
            return _done(SetPerceptionPaused.Response(success=pause_ok, paused=request.paused))

    class ActionClient:
        def wait_for_server(self, **_kwargs):
            return True

        def send_goal_async(self, goal, **_kwargs):
            calls.append(("send", goal.dispatch_binding.dispatch_nonce))
            if terminal == "unknown":
                return pending
            if terminal == "accepted":
                result_type = PickObject.Result if kind == "pick" else PlaceObject.Result
                result = result_type(success=not error_code, error_code=error_code)
                result.execution_detail.schema_version = 1
                result.execution_detail.recovery_state = recovery_state
                result.actual_executor = goal.expected_executor
                if kind == "place":
                    result.place_succeeded = not error_code
                pending.set_result(SimpleNamespace(result=result))
                return _done(SimpleNamespace(accepted=True, get_result_async=lambda: pending))
            if terminal == "canceling":
                return _done(
                    SimpleNamespace(
                        accepted=True,
                        get_result_async=lambda: pending,
                        cancel_goal_async=lambda: _done(SimpleNamespace(goals_canceling=[object()])),
                    )
                )
            return _done(SimpleNamespace(accepted=False))

    executor_name = "grasp_pipeline" if kind == "pick" else "placement_pipeline"
    executor = SimpleNamespace(
        name=executor_name,
        contract_version="1",
        endpoint_kind="ros_action",
        endpoint_name=f"/manipulation/{kind}",
        configuration_digest="a" * 64,
        model_deployment_name="",
        model_fingerprint="",
        model_bundle_digest="",
    )
    node._rpc_timeout = 0.01
    node._motion_imitation_pause_client = PauseClient()
    node._pick_client = node._place_client = ActionClient()
    node._active_runtime_bundle = SimpleNamespace(
        snapshot=SimpleNamespace(delegated_executors={executor_name: executor})
    )
    node._active_skill_admission = object()
    node._register_delegated_dispatch = lambda *_args: b"cleanup"
    node._confirm_delegated_terminal = lambda *_args: calls.append(("terminal",))
    node._wait_for_future = lambda future, **_kwargs: future.done()
    node._abort_skill = lambda _result, _handle, _phases, code, message: SimpleNamespace(
        success=False,
        error_code=code,
        message=message,
    )
    node._cancel_skill = lambda *_args: SimpleNamespace(success=False, error_code="SKILL_CANCELLED")
    node._set_result_catalog_identity = lambda *_args: None
    node._audit_cancel_propagated = lambda: calls.append(("cancel",))
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda message: calls.append(("warning", message)),
        error=lambda message: calls.append(("error", message)),
    )
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id="test-manipulation")
    goal.skill_name = "pick_object" if kind == "pick" else "place_in_container"
    goal.target_name, goal.container_name = "apple", "bowl"
    handle = SimpleNamespace(
        request=goal, is_cancel_requested=terminal == "canceling", succeed=lambda: calls.append(("succeeded",))
    )
    return node, handle, calls, pending


@pytest.mark.parametrize("kind", ["pick", "place"])
def test_pause_acknowledgement_precedes_action_and_terminal_releases_same_owner(kind):
    node, handle, calls, _ = _gateway(kind)
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert result.error_code == ("PICK_GOAL_REJECTED" if kind == "pick" else "PLACE_GOAL_REJECTED")
    assert [call[0] for call in calls] == ["pause", "send", "terminal", "pause"]
    assert calls[0][2] is True and calls[-1][2] is False
    assert calls[0][1] == calls[1][1] == calls[-1][1]


@pytest.mark.parametrize("kind", ["pick", "place"])
def test_a_rejected_pause_never_submits_robot_motion(kind):
    node, handle, calls, _ = _gateway(kind, pause_ok=False)
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert result.error_code == "PERCEPTION_PAUSE_FAILED"
    assert not any(call[0] == "send" for call in calls)


@pytest.mark.parametrize("kind", ["pick", "place"])
def test_unknown_action_keeps_pause_until_late_definite_rejection(kind):
    node, handle, calls, pending = _gateway(kind, terminal="unknown")
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert [call[0] for call in calls] == ["pause", "send"]
    pending.set_result(SimpleNamespace(accepted=False))
    assert [call[0] for call in calls] == ["pause", "send", "terminal", "pause"]
    assert calls[-1][2] is False


def test_pause_timeout_compensates_without_submitting_a_task():
    node, _, calls, _ = _gateway("pick")
    node._wait_for_future = lambda *_args, **_kwargs: False
    assert not node._pause_motion_imitation("timed-out-owner")
    assert calls[:2] == [("pause", "timed-out-owner", True), ("pause", "timed-out-owner", False)]


def test_disabled_pause_is_a_noop_for_other_robot_profiles():
    node = _GatewayHarness()
    assert node._pause_motion_imitation("pick")
    node._resume_motion_imitation("pick")


@pytest.mark.parametrize("kind", ["pick", "place"])
@pytest.mark.parametrize("error_code", ["", "PLANNING_FAILED", "GRASP_UNCERTAIN"])
def test_success_or_definite_failure_releases_pause_including_uncertain_custody(kind, error_code):
    node, handle, calls, _ = _gateway(kind, terminal="accepted", error_code=error_code)
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert result.success is (not error_code)
    releases = [call for call in calls if call[0] == "pause" and not call[2]]
    assert len(releases) == 1
    assert releases[0][1] == calls[0][1]
    assert ("terminal",) in calls


@pytest.mark.parametrize("kind", ["pick", "place"])
@pytest.mark.parametrize(
    "error_code", ["CANCEL_CLEANUP_TIMEOUT", "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT", "POSE_STOP_UNKNOWN"]
)
def test_an_action_reply_with_unknown_stop_does_not_resume_perception(kind, error_code):
    node, handle, calls, _ = _gateway(kind, terminal="accepted", error_code=error_code)
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert not result.success and result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert calls == [("pause", calls[0][1], True), ("send", calls[0][1])]


@pytest.mark.parametrize("kind", ["pick", "place"])
def test_motion_failure_with_unconfirmed_stop_keeps_pause_and_public_cleanup_quarantine(kind):
    node, handle, calls, _ = _gateway(kind, terminal="accepted", error_code="MOTION_FAILED", recovery_state="UNKNOWN")
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert not result.success and result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert [call[0] for call in calls] == ["pause", "send"]


@pytest.mark.parametrize("kind", ["pick", "place"])
def test_cancellation_acknowledgement_waits_for_the_late_motion_terminal(kind):
    node, handle, calls, pending = _gateway(kind, terminal="canceling")
    result = getattr(node, f"_execute_{kind}_skill")(handle, {})
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert [call[0] for call in calls] == ["pause", "send", "cancel"]
    pending.set_result(SimpleNamespace(result=SimpleNamespace(error_code="CANCELED")))
    assert [call[0] for call in calls] == ["pause", "send", "cancel", "terminal", "pause"]
    assert calls[-1] == ("pause", calls[0][1], False)
