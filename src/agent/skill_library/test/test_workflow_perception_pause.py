"""A root holds mirror perception paused across its motion steps and their gaps.

Return, pick and place require the pause; with the ``robot_motion`` scope every other
``moves_robot`` skill takes it best effort, and a mirror step gives it back.
"""

import threading
import uuid
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from embodied_common.dispatch_binding import copy_binding, new_binding
from embodied_common.skill_request import derive_skill_task_id
from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_msgs.action import SkillCommand
from ibrobot_msgs.srv import SetPerceptionPaused
from skill_library.gateway_policy import ExecutionOwner
from skill_library.skill_executor_node import (
    WORKFLOW_STEP_PENDING,
    SkillExecutorNode,
    _WorkflowChildAdmission,
    _WorkflowExecution,
)


def _done(value):
    future = Future()
    future.set_result(value)
    return future


def _gateway(*, pause_ok=True, enabled=True, steps=None, scope=None, ready=True, mirror_state=None):
    """Exercise the real child lifecycle and root finalizers without hardware."""
    node = object.__new__(SkillExecutorNode)
    events, owners, waits = [], set(), []

    class PauseClient:
        def wait_for_service(self, **_kwargs):
            waits.append(True)
            return ready

        def service_is_ready(self):
            return ready

        def call_async(self, request):
            events.append(("pause", request.owner_id, request.paused))
            if pause_ok:
                if request.paused:
                    owners.add(request.owner_id)
                else:
                    owners.discard(request.owner_id)
            return _done(SetPerceptionPaused.Response(success=pause_ok, paused=bool(owners)))

    node._rpc_timeout = 0.01
    if scope is not None:
        node._motion_imitation_pause_scope = scope
    if mirror_state is not None:
        node._mirror_session_enabled = True
        node._mirror_session_state_lock = threading.Lock()
        node._mirror_session_state = mirror_state
    node._motion_imitation_pause_client = PauseClient() if enabled else None
    node._wait_for_future = lambda future, **_kwargs: future.done()
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda message: events.append(("warning", message)),
        error=lambda message: events.append(("error", message)),
    )
    node._validate_skill_goal_schema = lambda *_args: ""
    node._validate_skill = lambda *_args, **_kwargs: (True, "")
    node._runtime_snapshot = lambda: None
    node._audit = lambda *_args, **_kwargs: None
    node._set_result_catalog_identity = lambda *_args: None
    node._ros_time_sec = lambda: 100.0
    node._retained_admission_cleanup = {}
    node._active_delegated_dispatches = {}
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._active_skill_admission = None
    node._runtime_coordinator = SimpleNamespace(release=lambda generation: events.append(("generation", generation)))
    binding = new_binding(task_id="workflow-root")
    binding.workflow_digest = "workflow-digest"
    binding.root_lease_nonce = uuid.uuid4().hex
    binding.expected_registry_epoch = "epoch"
    binding.expected_registry_generation = 1
    binding.expected_registry_digest = "registry-digest"
    steps = steps or [
        CanonicalWorkflowStep(schema_version=1, skill_name="return_to_task_pose", target_name="grasp_ready"),
        CanonicalWorkflowStep(schema_version=1, skill_name="pick_object", target_name="apple"),
        CanonicalWorkflowStep(
            schema_version=1, skill_name="place_in_container", target_name="apple", container_name="bowl"
        ),
    ]
    policy = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: SimpleNamespace(admitted=True, effective_timeout_sec=20.0),
        borrow_workflow_internal=lambda *_args: object(),
        release_workflow=lambda *_args: events.append(("lease",)) or True,
    )
    node._gateway_policy = policy
    workflow = _WorkflowExecution(
        root_task_id=binding.root_task_id,
        workflow_digest=binding.workflow_digest,
        root_lease_nonce=binding.root_lease_nonce,
        lease_token=object(),
        owner=ExecutionOwner.workflow(binding.root_task_id),
        policy=policy,
        bundle=SimpleNamespace(
            registry_epoch="epoch",
            generation=1,
            snapshot=SimpleNamespace(
                registry_digest="registry-digest",
                templates={
                    "pick_object": {"executor": "grasp_pipeline"},
                    "place_in_container": {"executor": "placement_pipeline"},
                    "wave_hand": {"executor": "runtime_named_motion"},
                    "offer_fruit_menu": {"executor": "runtime_play_audio"},
                    "mirror_start": {"executor": "mirror_session"},
                    "mirror_stop": {"executor": "mirror_session"},
                },
                capability_view={
                    name: {"moves_robot": name != "offer_fruit_menu"}
                    for name in (
                        "return_to_task_pose",
                        "pick_object",
                        "place_in_container",
                        "wave_hand",
                        "offer_fruit_menu",
                        "mirror_start",
                        "mirror_stop",
                    )
                },
            ),
        ),
        workflow_steps=tuple(steps),
        task_budget_key=node._task_budget_key(binding),
        deadline_unix_sec=200.0,
        step_terminal_states=[WORKFLOW_STEP_PENDING] * len(steps),
    )
    node._active_workflow = workflow
    root_admission = SimpleNamespace(lease_token=workflow.lease_token)
    workflow.late_cleanup_callback = lambda: node._finalize_program_admission(root_admission, workflow.owner)

    def execute_child(handle, **_kwargs):
        events.append(("motion", handle.request.skill_name))
        if enabled:
            assert workflow.perception_pause_owner in owners
        if handle.request.skill_name == "pick_object":
            events.append(("preparation",))
        # Keep the existing nested pick/place owner behavior: releasing it must
        # never empty the root's owner set, including between child callbacks.
        if handle.request.skill_name in {"pick_object", "place_in_container"}:
            delegated_owner = uuid.uuid4().hex
            assert node._pause_motion_imitation(delegated_owner)
            node._resume_motion_imitation(delegated_owner)
            if enabled:
                assert owners == {workflow.perception_pause_owner}
        handle.succeed()
        return SkillCommand.Result(success=True)

    node._execute_skill_child = execute_child

    def send(index, *, handle_ready=None):
        step = steps[index]
        goal = SkillCommand.Goal()
        goal.dispatch_binding = copy_binding(binding)
        goal.dispatch_binding.task_id = derive_skill_task_id(binding.root_task_id, index)
        goal.dispatch_binding.workflow_step_index = index
        goal.skill_name = step.skill_name
        goal.target_name = step.target_name
        goal.container_name = step.container_name
        handle = SimpleNamespace(
            request=goal,
            is_cancel_requested=False,
            succeed=lambda: None,
            abort=lambda: None,
            canceled=lambda: None,
        )
        if handle_ready is not None:
            handle_ready(handle)
        return node._execute_workflow_skill_gateway(handle)

    return SimpleNamespace(
        node=node,
        workflow=workflow,
        admission=root_admission,
        events=events,
        owners=owners,
        waits=waits,
        send=send,
        finish=lambda: node._finalize_program_admission(root_admission, workflow.owner),
    )


def test_root_pause_precedes_return_and_survives_preparation_and_every_child_gap():
    rig = _gateway()
    for index in range(3):
        assert rig.send(index).success
        assert rig.owners == {rig.workflow.perception_pause_owner}
    assert rig.events[0] == ("pause", rig.workflow.perception_pause_owner, True)
    assert rig.events[1] == ("motion", "return_to_task_pose")
    assert ("preparation",) in rig.events
    # Replaying a child neither reacquires a retired owner nor dispatches motion.
    event_count = len(rig.events)
    assert rig.send(0).success
    assert len(rig.events) == event_count
    root_owner = rig.workflow.perception_pause_owner
    rig.finish()
    assert not rig.owners
    assert rig.events[-1] == ("pause", root_owner, False)
    assert rig.node._active_workflow is None
    rig.finish()
    assert rig.events.count(("pause", root_owner, False)) == 1


def test_pause_rejection_prevents_even_the_first_return_motion():
    rig = _gateway(pause_ok=False)
    result = rig.send(0)
    assert not result.success and result.error_code == "PERCEPTION_PAUSE_FAILED"
    assert not any(event[0] == "motion" for event in rig.events)
    rig.finish()
    assert rig.node._active_workflow is None


@pytest.mark.parametrize("failed_index", range(3))
@pytest.mark.parametrize("error_code", ["PLANNING_FAILED", "GRASP_UNCERTAIN", "SKILL_CANCELLED"])
def test_definite_return_pick_or_place_failure_holds_until_root_finalization(failed_index, error_code):
    rig = _gateway()
    for index in range(failed_index):
        assert rig.send(index).success

    def failed(handle, **_kwargs):
        assert rig.workflow.perception_pause_owner in rig.owners
        handle.abort()
        return SkillCommand.Result(success=False, error_code=error_code)

    rig.node._execute_skill_child = failed
    assert rig.send(failed_index).error_code == error_code
    assert rig.owners
    rig.finish()
    assert not rig.owners


@pytest.mark.parametrize("unknown_index", range(3))
@pytest.mark.parametrize("error_code", ["CANCEL_CLEANUP_TIMEOUT", "POSE_STOP_UNKNOWN", "MOTION_FAILED"])
def test_unknown_stop_retains_root_until_late_child_confirmation(unknown_index, error_code):
    rig = _gateway()
    for index in range(unknown_index):
        assert rig.send(index).success
    pending = {}

    def unknown(handle, **_kwargs):
        admission = rig.node._active_skill_admission
        nonce = uuid.uuid4().hex
        cleanup_key = rig.node._register_delegated_dispatch(nonce, admission, handle.request.dispatch_binding)
        pending.update(admission=admission, nonce=nonce, cleanup_key=cleanup_key)
        result = SkillCommand.Result(success=False, error_code=error_code)
        result.execution_detail.schema_version = 1
        result.execution_detail.recovery_state = "UNKNOWN"
        handle.abort()
        return result

    rig.node._execute_skill_child = unknown
    assert rig.send(unknown_index).error_code == "SKILL_CANCEL_TIMEOUT"
    assert isinstance(rig.node._active_skill_admission, _WorkflowChildAdmission)
    root_owner = rig.workflow.perception_pause_owner
    rig.finish()  # Even a stray wrapper finalizer cannot clear unknown motion.
    rig.node._ros_time_sec = lambda: 300.0
    rig.node._reap_expired_workflow()  # Nor can expiry while a child is retained.
    assert rig.owners and rig.node._active_workflow is rig.workflow
    rig.node._confirm_delegated_terminal(pending["admission"], pending["nonce"], pending["cleanup_key"])
    assert not rig.owners and rig.node._active_workflow is None
    rig.node._confirm_delegated_terminal(pending["admission"], pending["nonce"], pending["cleanup_key"])
    assert rig.events.count(("pause", root_owner, False)) == 1


def test_expired_typed_workflow_releases_root_after_definite_child_cleanup():
    rig = _gateway()
    rig.workflow.late_cleanup_callback = None
    assert rig.send(0).success
    rig.node._ros_time_sec = lambda: 300.0
    rig.node._record_workflow_terminal = lambda *_args, **_kwargs: rig.events.append(("terminal",))
    rig.node._reap_expired_workflow()
    assert not rig.owners and rig.node._active_workflow is None
    assert rig.events.index(("terminal",)) < len(rig.events) - 1


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_cancel_or_expired_budget_during_pause_ack_does_not_start_return(interruption):
    rig = _gateway()
    original = rig.node._pause_workflow_motion_imitation
    handles = []

    def pause(workflow, skill_name):
        result = original(workflow, skill_name)
        if interruption == "deadline":
            rig.node._ros_time_sec = lambda: 300.0
        else:
            handles[0].is_cancel_requested = True
        return result

    rig.node._pause_workflow_motion_imitation = pause
    expected = "SKILL_CANCELLED" if interruption == "cancel" else "SKILL_TASK_DEADLINE_EXPIRED"
    assert rig.send(0, handle_ready=handles.append).error_code == expected
    assert not any(event[0] == "motion" for event in rig.events)
    rig.finish()
    assert not rig.owners


def test_disabled_profile_and_non_manipulation_workflows_do_not_pause():
    for rig in (
        _gateway(enabled=False),
        _gateway(steps=[CanonicalWorkflowStep(schema_version=1, skill_name="look_around")]),
    ):
        # This dummy non-manipulation executor does not require a pause.
        def succeeded(handle, **_kwargs):
            handle.succeed()
            return SkillCommand.Result(success=True)

        rig.node._execute_skill_child = succeeded
        assert rig.send(0).success
        rig.finish()
        assert not any(event[0] == "pause" for event in rig.events)


def test_old_program_finalizer_does_not_release_or_clear_a_newer_root():
    old = _gateway()
    assert old.send(0).success
    old.finish()
    newer = _gateway()
    assert newer.send(0).success
    newer.node._finalize_program_admission(old.admission, old.workflow.owner)
    assert newer.node._active_workflow is newer.workflow
    assert newer.owners == {newer.workflow.perception_pause_owner}
    newer.finish()


def test_root_cleanup_failure_can_retry_without_releasing_pause_or_generation_twice():
    rig = _gateway()
    assert rig.send(0).success
    root_owner = rig.workflow.perception_pause_owner
    rig.workflow.policy.release_workflow = lambda *_args: False
    rig.finish()
    assert rig.node._active_workflow is rig.workflow
    assert rig.owners == {root_owner}
    rig.workflow.policy.release_workflow = lambda *_args: True
    rig.finish()
    assert not rig.owners
    assert rig.events.count(("generation", 1)) == 1
    assert rig.events.count(("pause", root_owner, False)) == 1


def test_failure_before_workflow_construction_releases_only_its_admitted_root_lease():
    rig = _gateway()
    rig.node._active_workflow = None
    rig.finish()
    assert rig.events == [("lease",)]


def test_pick_only_program_pauses_before_its_preparation_without_an_explicit_return():
    rig = _gateway(steps=[CanonicalWorkflowStep(schema_version=1, skill_name="pick_object", target_name="apple")])
    assert rig.send(0).success
    assert rig.events[0][0] == "pause" and rig.events[0][2] is True
    assert rig.events.index(("preparation",)) > 0
    assert rig.owners == {rig.workflow.perception_pause_owner}
    rig.finish()
    assert not rig.owners


def _step(skill_name, **fields):
    return CanonicalWorkflowStep(schema_version=1, skill_name=skill_name, **fields)


def _succeed_and_record(rig):
    def execute(handle, **_kwargs):
        rig.events.append(("motion", handle.request.skill_name, frozenset(rig.owners)))
        handle.succeed()
        return SkillCommand.Result(success=True)

    rig.node._execute_skill_child = execute


def test_manipulation_scope_leaves_gestures_unpaused():
    rig = _gateway(steps=[_step("wave_hand")])
    _succeed_and_record(rig)
    assert rig.send(0).success
    rig.finish()
    assert not any(event[0] == "pause" for event in rig.events)


def test_robot_motion_scope_pauses_a_gesture_until_the_program_ends():
    rig = _gateway(steps=[_step("wave_hand")], scope="robot_motion")
    _succeed_and_record(rig)
    assert rig.send(0).success
    owner = rig.workflow.perception_pause_owner
    assert rig.events[0][0] == "pause" and rig.events[0][1:] == (owner, True)
    assert ("motion", "wave_hand", frozenset({owner})) in rig.events
    assert rig.owners == {owner}
    rig.finish()
    assert not rig.owners and rig.events[-1] == ("pause", owner, False)


def test_robot_motion_scope_skips_skills_that_do_not_move_the_robot():
    rig = _gateway(steps=[_step("offer_fruit_menu")], scope="robot_motion")
    _succeed_and_record(rig)
    assert rig.send(0).success
    rig.finish()
    assert not any(event[0] == "pause" for event in rig.events)


def test_unacknowledged_gesture_pause_warns_and_moves_without_an_owner():
    rig = _gateway(steps=[_step("wave_hand")], scope="robot_motion", pause_ok=False)
    _succeed_and_record(rig)
    assert rig.send(0).success
    assert ("motion", "wave_hand", frozenset()) in rig.events
    assert any(event[0] == "warning" and "wave_hand" in event[1] for event in rig.events)
    assert not rig.workflow.perception_pause_owner
    rig.finish()


def test_gesture_does_not_wait_for_a_stream_that_is_not_up():
    rig = _gateway(steps=[_step("wave_hand")], scope="robot_motion", ready=False)
    _succeed_and_record(rig)
    assert rig.send(0).success
    assert not rig.waits
    assert not any(event[0] == "pause" for event in rig.events)
    assert ("motion", "wave_hand", frozenset()) in rig.events
    rig.finish()


def test_unanswered_gesture_pause_costs_a_short_wait_and_is_compensated():
    rig = _gateway(steps=[_step("wave_hand")], scope="robot_motion")
    rig.node._rpc_timeout = 5.0
    waited = []

    def never_answered(_future, *, timeout_sec):
        waited.append(timeout_sec)
        return False

    rig.node._wait_for_future = never_answered
    _succeed_and_record(rig)
    assert rig.send(0).success
    assert waited == [1.0]
    pause, release = [event for event in rig.events if event[0] == "pause"]
    assert pause[1] == release[1] and pause[2] is True and release[2] is False
    assert ("motion", "wave_hand", frozenset()) in rig.events
    rig.finish()


def test_pick_still_refuses_to_move_without_an_acknowledged_pause_in_robot_motion_scope():
    rig = _gateway(steps=[_step("pick_object", target_name="apple")], scope="robot_motion", ready=False)
    _succeed_and_record(rig)
    result = rig.send(0)
    assert not result.success and result.error_code == "PERCEPTION_PAUSE_FAILED"
    assert not any(event[0] == "motion" for event in rig.events)
    rig.finish()


def test_mirror_step_gives_the_pause_back_and_a_later_gesture_takes_a_fresh_owner():
    rig = _gateway(steps=[_step("wave_hand"), _step("mirror_start"), _step("wave_hand")], scope="robot_motion")
    _succeed_and_record(rig)
    assert rig.send(0).success
    first = rig.workflow.perception_pause_owner
    assert rig.send(1).success
    assert ("motion", "mirror_start", frozenset()) in rig.events
    assert ("pause", first, False) in rig.events and not rig.workflow.perception_pause_owner
    assert rig.send(2).success
    second = rig.workflow.perception_pause_owner
    assert second and second != first and rig.owners == {second}
    rig.finish()
    assert not rig.owners
    assert rig.events.count(("pause", first, False)) == 1


def test_mirror_stop_first_runs_with_the_stream_and_the_following_gesture_pauses():
    rig = _gateway(steps=[_step("mirror_stop"), _step("wave_hand")], scope="robot_motion")
    _succeed_and_record(rig)
    assert rig.send(0).success
    assert not any(event[0] == "pause" for event in rig.events)
    assert rig.send(1).success
    assert rig.owners == {rig.workflow.perception_pause_owner}
    rig.finish()


def test_running_mirror_is_not_paused_for_a_skill_its_admission_will_reject():
    mirroring = {"state": "MIRRORING", "holding": True, "releasing": False, "stamp": 100.0}
    rig = _gateway(steps=[_step("wave_hand")], scope="robot_motion", mirror_state=mirroring)
    _succeed_and_record(rig)
    rig.send(0)
    assert not any(event[0] == "pause" for event in rig.events)
    rig.finish()


def test_directly_submitted_skills_follow_the_same_rule():
    rig = _gateway(scope="robot_motion")
    rig.node._active_runtime_bundle = rig.workflow.bundle
    owner, ready = rig.node._pause_skill_motion_imitation("wave_hand")
    assert ready and owner.startswith("skill:") and rig.owners == {owner}
    assert rig.node._pause_skill_motion_imitation("offer_fruit_menu") == ("", True)
    assert rig.node._pause_skill_motion_imitation("mirror_start") == ("", True)
    rig.node._resume_motion_imitation(owner)
    assert not rig.owners

    refused = _gateway(scope="robot_motion", pause_ok=False)
    refused.node._active_runtime_bundle = refused.workflow.bundle
    assert refused.node._pause_skill_motion_imitation("pick_object") == ("", False)
    assert refused.node._pause_skill_motion_imitation("wave_hand") == ("", True)
    assert any(event[0] == "warning" for event in refused.events)
