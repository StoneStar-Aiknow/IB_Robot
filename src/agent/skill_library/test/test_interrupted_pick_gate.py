"""Gateway recovery runs before preparation; no hand or pose command on refusal."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from test_gateway_policy import _held_policy, _recovery, _request, _snapshot

from embodied_common.dispatch_binding import fill_delegated_executor_identity
from ibrobot_msgs.action import PickObject, SkillCommand
from ibrobot_msgs.msg import InterruptedPick
from ibrobot_msgs.srv import AuthorizeHeldReset
from skill_library.gateway_policy import ExecutionOwner
from skill_library.skill_executor_node import SkillExecutorNode


def node_and_goal(skill="pick_object", state="pending"):
    node = object.__new__(SkillExecutorNode)
    node._interrupted_pick = InterruptedPick(nonce="one", state=state, reason_code="PICK_INTERRUPTED")
    node._set_result_catalog_identity = lambda *a: None
    node._skill_templates = {"pick_object": {"executor": "grasp_pipeline", "timeout_sec": 20}}
    node._active_mirror_session_command = lambda skill: ""
    node._mirror_session_admission = lambda skill: (True, "", "")
    node._ensure_skill_control_mode = lambda *a: (True, "")
    node._ensure_skill_capabilities = lambda *a: (True, "")
    node._task_poses = {
        "skill_entry_poses": {"pick_object": "grasp_ready"},
        "poses": {"grasp_ready": {"timeout_sec": 5}},
    }
    goal = SkillCommand.Goal(skill_name=skill, target_name="an apple", timeout_sec=20.0)
    events = []
    handle = SimpleNamespace(
        request=goal,
        is_cancel_requested=False,
        publish_feedback=lambda *a: None,
        succeed=lambda: events.append("terminal"),
        abort=lambda: events.append("abort"),
    )
    node._request_pick_prefetch = lambda *a: events.append("prefetch")

    def pose(*args, **kwargs):
        events.append("prepare")
        return SkillCommand.Result(success=True)

    node._execute_pose_composer_skill = pose
    return node, handle, events


def test_pending_x2_recovery_precedes_prepare_prefetch_and_new_pick():
    node, handle, events = node_and_goal()

    def pick(child, template, *, effective_timeout_sec=None, recovery_nonce=""):
        events.append("recover" if recovery_nonce else "new_pick")
        child.succeed()
        return SkillCommand.Result(success=True)

    node._execute_pick_skill = pick
    result = node._execute_skill_child(handle, validation_done=True, effective_timeout_sec=20)
    assert result.success
    assert events == ["recover", "prefetch", "prepare", "new_pick", "terminal"]
    assert node._interrupted_pick is None


def test_failed_recovery_never_reaches_preparation_or_prefetch():
    node, handle, events = node_and_goal()

    def fail(child, template, **budget):
        events.append("recover")
        child.abort()
        return SkillCommand.Result(success=False, error_code="PICK_MANUAL_CHECK")

    node._execute_pick_skill = fail
    result = node._execute_skill_child(handle, validation_done=True, effective_timeout_sec=20)
    assert not result.success and events == ["recover", "abort"]
    assert node._interrupted_pick is not None


def test_manual_gate_blocks_pick_before_any_downstream_call():
    node, handle, events = node_and_goal(state="manual")
    node._execute_pick_skill = lambda *a, **k: pytest.fail("manual gate dispatched motion")
    result = node._execute_skill_child(handle, validation_done=True, effective_timeout_sec=20)
    assert not result.success and events == ["abort"]


def test_late_pending_does_not_downgrade_manual_or_resurrect_cleared_nonce():
    node, handle, events = node_and_goal(state="manual")
    node._handle_interrupted_pick(InterruptedPick(nonce="one", state="pending"))
    assert node._interrupted_pick.state == "manual"
    node._handle_interrupted_pick(InterruptedPick(nonce="old", state="clear"))
    assert node._interrupted_pick.state == "manual"
    node._handle_interrupted_pick(InterruptedPick(nonce="one", state="clear"))
    assert node._interrupted_pick is None
    node._handle_interrupted_pick(InterruptedPick(nonce="one", state="pending"))
    assert node._interrupted_pick is None
    node._handle_interrupted_pick(InterruptedPick(nonce="two", state="pending"))
    node._handle_interrupted_pick(InterruptedPick(nonce="two", state="clear"))
    node._handle_interrupted_pick(InterruptedPick(nonce="one", state="manual"))
    assert node._interrupted_pick is None


def test_manual_update_during_recovery_cannot_be_cleared_by_success():
    node, handle, events = node_and_goal()

    def contradictory(child, template, **budget):
        node._handle_interrupted_pick(InterruptedPick(nonce="one", state="manual"))
        child.succeed()
        return SkillCommand.Result(success=True)

    node._execute_pick_skill = contradictory
    failure, _ = node._recover_interrupted_pick(handle, 20)
    assert failure is not None and not failure.success
    assert node._interrupted_pick.state == "manual"
    assert events == ["abort"]


@pytest.mark.parametrize("state", ["pending", "manual"])
def test_gate_blocks_other_skills_workflows_and_external_primitives(state):
    policy = _held_policy()
    snapshot = _snapshot(interrupted_pick_state=state)
    assert not policy.evaluate(_request(), snapshot, owner=ExecutionOwner.skill_command("x")).admitted
    assert policy.admit_external_primitive("x", "digest", snapshot)[1] is None
    assert (
        policy.admit_workflow(
            ExecutionOwner.workflow("x"), snapshot, timeout_sec=5, skill_names=["relative", "pick_object"]
        )[1]
        is None
    )
    admitted = policy.evaluate(_recovery("x"), snapshot, owner=ExecutionOwner.skill_command("x"))
    assert admitted.admitted is (state == "pending")
    error, token = policy.admit_workflow(
        ExecutionOwner.workflow("x"), snapshot, timeout_sec=5, skill_names=["pick_object", "place_in_container"]
    )
    assert (token is not None) is (state == "pending")


def test_operator_check_forwards_matching_nonce_and_never_dispatches_motion():
    node, handle, events = node_and_goal(state="manual")
    from skill_library.gateway_policy import RootExecutionLease

    node._runtime_snapshot = lambda: _snapshot()
    node._held_object = None
    node._runtime_enabled = False
    node._gateway_lease = RootExecutionLease()
    node._rpc_timeout = 1.0
    calls = []

    def request(message):
        calls.append(message.custody_id)
        future = Future()
        future.set_result(AuthorizeHeldReset.Response(success=True, custody_id="one"))
        return future

    node._pick_reset_client = SimpleNamespace(wait_for_service=lambda **kw: True, call_async=request)
    node._wait_for_future = lambda *a, **kw: True
    response = node._authorize_held_reset(
        AuthorizeHeldReset.Request(
            schema_version=1,
            request_id="operator-1",
            custody_id="one",
            operator_confirmed_empty=True,
            operator_note="empty hand and original home inspected",
        ),
        AuthorizeHeldReset.Response(),
    )
    assert response.success and response.state == "clear" and calls == ["one"]
    assert node._interrupted_pick is None and events == []


def test_late_matching_pick_result_records_interruption_before_releasing_root_and_perception(monkeypatch):
    from test_manipulation_perception_pause import _gateway

    monkeypatch.setattr("skill_library.skill_executor_node.rclpy.ok", lambda: True)
    node, handle, calls, pending = _gateway("pick", terminal="canceling")
    node._handle_interrupted_pick = lambda message: calls.append(("interruption", message.state))
    result = node._execute_pick_skill(handle, {})
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    terminal = PickObject.Result(error_code="CANCELED", interrupted_pick=InterruptedPick(nonce="one", state="pending"))
    descriptor = node._active_runtime_bundle.snapshot.delegated_executors["grasp_pipeline"]
    fill_delegated_executor_identity(terminal.actual_executor, vars(descriptor))
    pending.set_result(SimpleNamespace(result=terminal))
    assert [call[0] for call in calls] == ["pause", "send", "cancel", "interruption", "terminal", "pause"]


def test_late_mismatched_executor_cannot_clear_interruption(monkeypatch):
    from test_manipulation_perception_pause import _gateway

    monkeypatch.setattr("skill_library.skill_executor_node.rclpy.ok", lambda: True)
    node, handle, calls, pending = _gateway("pick", terminal="canceling")
    node._handle_interrupted_pick = lambda message: pytest.fail("unverified executor changed recovery gate")
    node._execute_pick_skill(handle, {})
    terminal = PickObject.Result(error_code="CANCELED", interrupted_pick=InterruptedPick(nonce="one", state="clear"))
    pending.set_result(SimpleNamespace(result=terminal))
    assert [call[0] for call in calls] == ["pause", "send", "cancel", "terminal", "pause"]


def test_recovery_success_without_matching_clear_receipt_is_refused(monkeypatch):
    from test_manipulation_perception_pause import _gateway

    monkeypatch.setattr("skill_library.skill_executor_node.rclpy.ok", lambda: True)
    node, handle, calls, pending = _gateway("pick", terminal="accepted")
    result = node._execute_pick_skill(handle, {}, recovery_nonce="one")
    assert not result.success and result.error_code == "PICK_MANUAL_CHECK"
