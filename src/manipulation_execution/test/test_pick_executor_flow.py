import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from rclpy.clock import Clock
from sensor_msgs.msg import JointState

from embodied_common.dispatch_binding import delegated_executor_identity, new_binding
from ibrobot_msgs.action import PickObject
from ibrobot_msgs.msg import GraspCandidate
from manipulation_execution.grasp_geometry import CandidatePlan, FixedFingerEnvelope
from manipulation_execution.phases.execution import ExecutionPhase
from manipulation_execution.phases.flow import _is_execution_retryable
from manipulation_execution.pick_executor_helpers import PickExecutorHelpers
from manipulation_execution.pick_executor_models import FlowState, PreparedCandidate, RankedCandidate
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)


class _GoalHandle:
    def __init__(self, *, mode: int, release_after_success: bool = False):
        self.request = PickObject.Goal()
        self.request.dispatch_binding = new_binding(task_id="test-pick")
        self.request.dispatch_binding.task_budget.schema_version = 1
        self.request.dispatch_binding.task_budget.started_at.sec = 1
        self.request.dispatch_binding.task_budget.deadline.sec = 2_000_000_000
        self.request.target_query = "banana"
        self.request.timeout_sec = 5.0
        self.request.mode = mode
        self.request.release_after_success = release_after_success
        self.is_cancel_requested = False
        self.feedback: list[tuple[str, str]] = []
        self.terminal_state = ""

    def succeed(self) -> None:
        self.terminal_state = "succeeded"

    def abort(self) -> None:
        self.terminal_state = "aborted"

    def canceled(self) -> None:
        self.terminal_state = "canceled"


def _prepared_candidate(index: int = 7):
    ranked = SimpleNamespace(
        index=index,
        score=0.8,
        fixed_finger_base_side=None,
        candidate=object(),
    )
    plan = SimpleNamespace(
        approach=(0.1, -0.2, 0.12),
        grasp=(0.1, -0.2, 0.02),
        quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    return SimpleNamespace(
        ranked=ranked,
        plan=plan,
        final_joint_state=object(),
        selection_score=0.75,
        contact_z_error_m=0.001,
        contact_residual_xy_m=0.002,
        approach_axis_error_deg=1.0,
        closing_axis_error_deg=2.0,
        fixed_finger_envelope=None,
        fk_fixed_finger_base_side=None,
        predicted_robust_gap_headroom_m=None,
    )


def _full_prepared_candidate(index: int = 7) -> PreparedCandidate:
    candidate = GraspCandidate()
    candidate.confidence = 0.8
    plan = CandidatePlan(
        approach=(0.10, -0.20, 0.12),
        grasp=(0.10, -0.20, 0.04),
        quaternion=(0.0, 0.0, 0.0, 1.0),
        approach_axis=(0.0, 0.0, 1.0),
        target_contact_ee=(0.0, 0.0, 0.0),
        target_contact_base=(0.10, -0.20, 0.04),
        target_width_m=0.013,
        width_reason="test",
        fixed_finger_target_gap_m=0.013,
        target_width_min_base=(0.09, -0.20, 0.04),
        target_width_max_base=(0.11, -0.20, 0.04),
        topdown_score=1.0,
    )
    ranked = RankedCandidate(index=index, candidate=candidate, plan=plan, score=0.9)
    return PreparedCandidate(
        ranked=ranked,
        plan=plan,
        final_joint_state=JointState(),
        actual_ee_xyz=plan.grasp,
        actual_ee_quaternion=plan.quaternion,
        contact_residual_xy_m=0.0,
        contact_z_error_m=0.0,
        approach_axis_error_deg=0.0,
        closing_axis_error_deg=0.0,
        tabletop_clearance_m=0.02,
        mesh_min_z=0.04,
        fixed_finger_envelope=FixedFingerEnvelope(
            fixed_gap_m=0.0085,
            moving_gap_m=0.01,
            target_gap_m=0.013,
            fixed_score=0.5,
            moving_score=0.5,
            score=0.5,
        ),
        fk_fixed_finger_base_side=None,
        predicted_robust_gap_headroom_m=0.002,
        selection_score=0.8,
    )


def _flow_harness(*, config: dict | None = None) -> SimpleNamespace:
    logger = _Logger()
    harness = SimpleNamespace(
        _config={"timeout_sec": 5.0, "max_execution_attempts": 1, **(config or {})},
        _goal_lock=threading.Lock(),
        _goal_active=True,
        _dispatch_nonce="test-nonce",
        _dispatch_binding=new_binding(task_id="test-pick"),
        _executor_identity=delegated_executor_identity(
            name="grasp_pipeline", endpoint_name="/manipulation/execute_pick"
        ),
        get_clock=lambda: Clock(),
        get_logger=lambda: logger,
    )
    harness._result_from_state = PickExecutorHelpers._result_from_state

    def publish_feedback(goal_handle, state, phase, detail):
        state.completed_phases.append(phase)
        goal_handle.feedback.append((phase, detail))

    harness._publish_feedback = publish_feedback
    harness._record_prepared_ranking = lambda *_args: None
    harness._order_prepared_candidates = lambda state, candidates: PickExecutorNode._order_prepared_candidates(
        cast(Any, harness), state, candidates
    )
    return harness


def test_kinematics_unhealthy_snapshot_returns_copy():
    harness = SimpleNamespace(
        _kinematics_health_lock=threading.Lock(),
        _kinematics_unhealthy_workers={1, 3},
    )

    snapshot = PickExecutorNode._kinematics_unhealthy_snapshot(cast(Any, harness))

    assert snapshot == {1, 3}
    snapshot.add(7)
    assert harness._kinematics_unhealthy_workers == {1, 3}


@pytest.mark.parametrize("retryable", [False, True])
def test_wait_future_propagates_retryable_for_missing_response(retryable: bool):
    future = SimpleNamespace(done=lambda: True, result=lambda: None)
    harness = SimpleNamespace(_check_cancel=lambda _goal_handle: None)

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._wait_future(
            cast(Any, harness),
            future,
            None,
            time.monotonic() + 1.0,
            1.0,
            "IK",
            retryable=retryable,
        )

    assert exc_info.value.code == "RPC_FAILED"
    assert exc_info.value.retryable is retryable


def test_target_not_visible_does_not_retry_planning():
    logger = _Logger()
    harness = SimpleNamespace(
        _config={
            "candidate_selection": {"selection_attempts": 3, "retry_settle_sec": 0.0},
            "observe_settle_sec": 0.0,
        },
        _joint_state_topic="/joint_states",
        get_logger=lambda: logger,
    )
    requests = []

    def request(*_args):
        requests.append(True)
        raise PickFlowError("TARGET_NOT_VISIBLE", "banana not detected")

    harness._request_grasps = request
    harness._lookup_base_transform = lambda *_args: "transform"
    harness._transform_to_matrix = lambda _transform: np.eye(4, dtype=np.float64)
    harness._record_frame_diagnostic = lambda *_args, **_kwargs: None
    harness._scene_geometry_base = lambda _transform, _scene: "scene_base"
    harness._publish_feedback = lambda *_args, **_kwargs: None
    harness._rank_candidates = lambda *_args, **_kwargs: ["ranked"]
    harness._snapshot_joint_state = lambda: "joint_seed"
    harness._prepare_ranked_candidates = lambda *_args, **_kwargs: (["prepared"], None)
    harness._sleep_with_cancel = lambda *_args: None

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._plan_and_prepare_candidates(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "banana",
        )

    assert exc_info.value.code == "TARGET_NOT_VISIBLE"
    assert len(requests) == 1
    assert not any("grasp selection retry" in message for message in logger.messages)


def test_target_outside_workspace_does_not_retry_planning():
    logger = _Logger()
    harness = SimpleNamespace(
        _config={"candidate_selection": {"selection_attempts": 3, "retry_settle_sec": 0.0}},
        get_logger=lambda: logger,
    )
    requests = 0

    def request(*_args):
        nonlocal requests
        requests += 1
        return SimpleNamespace(frame_id="camera", stamp=SimpleNamespace(sec=1, nanosec=2)), ["candidate"], "scene"

    harness._request_grasps = request
    harness._lookup_base_transform = lambda *_args: "transform"
    harness._transform_to_matrix = lambda _transform: np.eye(4, dtype=np.float64)
    harness._record_frame_diagnostic = lambda *_args, **_kwargs: None
    harness._scene_geometry_base = lambda _transform, _scene: "scene_base"
    harness._publish_feedback = lambda *_args, **_kwargs: None
    harness._rank_candidates = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        PickFlowError("TARGET_OUTSIDE_WORKSPACE", "all candidates are outside workspace")
    )

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._plan_and_prepare_candidates(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "banana",
        )

    assert exc_info.value.code == "TARGET_OUTSIDE_WORKSPACE"
    assert requests == 1
    assert not any("grasp selection retry" in message for message in logger.messages)


def test_selection_attempts_do_not_retry_configuration_failures():
    harness = SimpleNamespace(
        _config={"candidate_selection": {"selection_attempts": 3}},
        get_logger=lambda: _Logger(),
    )
    harness._request_grasps = lambda *_args: (_ for _ in ()).throw(
        PickFlowError("TARGET_TABLETOP_UNAVAILABLE", "missing mesh")
    )

    try:
        PickExecutorNode._plan_and_prepare_candidates(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "banana",
        )
    except PickFlowError as exc:
        assert exc.code == "TARGET_TABLETOP_UNAVAILABLE"
    else:
        raise AssertionError("configuration failure must propagate")


@pytest.mark.parametrize("error_code", ["RPC_TIMEOUT", "RPC_FAILED", "TF_UNAVAILABLE"])
def test_selection_attempts_retry_transient_pipeline_failures(error_code: str):
    harness = SimpleNamespace(
        _config={"candidate_selection": {"selection_attempts": 2, "retry_settle_sec": 0.0}},
        get_logger=lambda: _Logger(),
    )
    requests = 0

    def request(*_args):
        nonlocal requests
        requests += 1
        if requests == 1:
            raise PickFlowError(error_code, "transient failure")
        return SimpleNamespace(frame_id="camera", stamp=SimpleNamespace(sec=1, nanosec=2)), ["candidate"], "scene"

    harness._request_grasps = request
    harness._lookup_base_transform = lambda *_args: "transform"
    harness._transform_to_matrix = lambda _transform: np.eye(4, dtype=np.float64)
    harness._record_frame_diagnostic = lambda *_args, **_kwargs: None
    harness._scene_geometry_base = lambda _transform, _scene: "scene_base"
    harness._publish_feedback = lambda *_args, **_kwargs: None
    harness._rank_candidates = lambda *_args, **_kwargs: ["ranked"]
    harness._snapshot_joint_state = lambda: "joint_seed"
    harness._prepare_ranked_candidates = lambda *_args, **_kwargs: (["prepared"], None)
    harness._sleep_with_cancel = lambda *_args: None

    prepared, _scene_base = PickExecutorNode._plan_and_prepare_candidates(
        cast(Any, harness),
        None,
        time.monotonic() + 5.0,
        FlowState(completed_phases=[]),
        "banana",
    )

    assert prepared == ["prepared"]
    assert requests == 2


def test_observe_only_uses_the_canonical_flow_without_planning():
    harness = _flow_harness()
    calls: list[tuple[str, object]] = []
    harness._preflight = lambda _goal, _deadline, _state, mode: calls.append(("preflight", mode))
    harness._move_to_observe = lambda _goal, _deadline, _state, task_id: calls.append(("observe", task_id))
    harness._plan_and_prepare_candidates = lambda *_args: pytest.fail("observe-only must not plan")
    goal_handle = _GoalHandle(mode=PickObject.Goal.MODE_OBSERVE_ONLY)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success is True
    assert result.message == "observation pose reached"
    assert result.candidate_index == -1
    assert result.released_after_success is False
    assert json.loads(result.pipeline_timings_json) == {}
    assert goal_handle.terminal_state == "succeeded"
    assert calls == [("preflight", PickObject.Goal.MODE_OBSERVE_ONLY), ("observe", "test-pick")]
    assert harness._goal_active is False


def test_plan_only_prepares_and_selects_without_physical_execution():
    harness = _flow_harness()
    prepared = _prepared_candidate(index=11)
    harness._preflight = lambda *_args: None
    harness._move_to_observe = lambda *_args: None

    def plan(_goal, _deadline, state, _target_query):
        state.pipeline_timings["candidate_selection_total"] = 1.25
        return [prepared], "scene_base"

    harness._plan_and_prepare_candidates = plan
    harness._execute_candidate = lambda *_args, **_kwargs: pytest.fail("plan-only must not move the robot")
    goal_handle = _GoalHandle(mode=PickObject.Goal.MODE_PLAN_ONLY)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success is True
    assert result.attempts == 0
    assert result.candidate_index == 11
    assert result.released_after_success is False
    assert json.loads(result.pipeline_timings_json) == {"candidate_selection_total": 1.25}
    assert goal_handle.terminal_state == "succeeded"


def test_execute_mode_passes_release_goal_to_the_only_executor_path():
    harness = _flow_harness()
    prepared = _prepared_candidate(index=5)
    captured: dict[str, object] = {}
    harness._preflight = lambda *_args: None
    harness._move_to_observe = lambda *_args: None
    harness._plan_and_prepare_candidates = lambda *_args: ([prepared], "scene_base")

    def execute(_goal, _deadline, state, _task_id, _target_query, selected, _scene, **kwargs):
        captured["selected"] = selected
        captured.update(kwargs)
        state.verification_status = PickObject.Result.VERIFICATION_SUCCESS
        state.verification_confidence = 0.95
        state.released_after_success = bool(kwargs["release_after_success"])

    harness._execute_candidate = execute
    goal_handle = _GoalHandle(
        mode=PickObject.Goal.MODE_EXECUTE,
        release_after_success=True,
    )

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert captured == {
        "selected": prepared,
        "release_after_success": True,
    }
    assert result.success is True
    assert result.attempts == 1
    assert result.candidate_index == 5
    assert result.verification_status == PickObject.Result.VERIFICATION_SUCCESS
    assert result.released_after_success is True
    assert goal_handle.terminal_state == "succeeded"


def test_grasp_verification_failure_replans_before_retrying_execution():
    harness = _flow_harness(config={"max_execution_attempts": 3})
    plans = [_prepared_candidate(index=5), _prepared_candidate(index=9)]
    plan_calls = 0
    executed_candidates: list[int] = []
    observe_calls = 0
    harness._preflight = lambda *_args: None

    def observe(*_args):
        nonlocal observe_calls
        observe_calls += 1

    def plan(*_args):
        nonlocal plan_calls
        selected = plans[plan_calls]
        plan_calls += 1
        return [selected], f"scene-{plan_calls}"

    def execute(_goal, _deadline, state, _task_id, _target_query, selected, _scene, **_kwargs):
        executed_candidates.append(selected.ranked.index)
        if len(executed_candidates) == 1:
            state.recovery_completed = True
            raise PickFlowError("GRASP_VERIFICATION_FAILED", "gripper is empty")
        state.verification_status = PickObject.Result.VERIFICATION_SUCCESS
        state.verification_confidence = 0.91

    harness._move_to_observe = observe
    harness._plan_and_prepare_candidates = plan
    harness._execute_candidate = execute

    result = PickExecutorNode._execute_pick(
        cast(Any, harness),
        _GoalHandle(mode=PickObject.Goal.MODE_EXECUTE),
    )

    assert result.success is True
    assert result.attempts == 2
    assert plan_calls == 2
    assert executed_candidates == [5, 9]
    # The execution phase owns the recovery move after a real verification
    # failure, so the outer flow only performs the initial observation move.
    assert observe_calls == 1


def test_ik_orientation_rejection_returns_to_observe_and_replans():
    harness = _flow_harness(config={"max_execution_attempts": 3})
    plans = [_prepared_candidate(index=5), _prepared_candidate(index=9)]
    plan_calls = 0
    executed_candidates: list[int] = []
    observe_calls = 0
    harness._preflight = lambda *_args: None

    def observe(*_args):
        nonlocal observe_calls
        observe_calls += 1

    def plan(*_args):
        nonlocal plan_calls
        selected = plans[plan_calls]
        plan_calls += 1
        return [selected], f"scene-{plan_calls}"

    def execute(_goal, _deadline, state, _task_id, _target_query, selected, _scene, **_kwargs):
        executed_candidates.append(selected.ranked.index)
        if len(executed_candidates) == 1:
            raise PickFlowError(
                "IK_ORIENTATION_REJECTED",
                "FK closing-axis orientation exceeds the configured limit",
                retryable=True,
            )
        state.verification_status = PickObject.Result.VERIFICATION_SUCCESS
        state.verification_confidence = 0.92

    harness._move_to_observe = observe
    harness._plan_and_prepare_candidates = plan
    harness._execute_candidate = execute

    result = PickExecutorNode._execute_pick(
        cast(Any, harness),
        _GoalHandle(mode=PickObject.Goal.MODE_EXECUTE),
    )

    assert result.success is True
    assert result.attempts == 2
    assert plan_calls == 2
    assert executed_candidates == [5, 9]
    assert observe_calls == 2
    assert result.pipeline_timings_json


def test_final_ik_orientation_rejection_still_returns_to_observe():
    harness = _flow_harness(config={"max_execution_attempts": 2})
    candidate = _prepared_candidate(index=5)
    observe_calls = 0
    plan_calls = 0
    harness._preflight = lambda *_args: None

    def observe(*_args):
        nonlocal observe_calls
        observe_calls += 1

    def plan(*_args):
        nonlocal plan_calls
        plan_calls += 1
        return [candidate], f"scene-{plan_calls}"

    def reject_orientation(*_args, **_kwargs):
        raise PickFlowError("IK_ORIENTATION_REJECTED", "closing-axis mismatch", retryable=True)

    harness._move_to_observe = observe
    harness._plan_and_prepare_candidates = plan
    harness._execute_candidate = reject_orientation
    goal_handle = _GoalHandle(mode=PickObject.Goal.MODE_EXECUTE)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success is False
    assert result.error_code == "IK_ORIENTATION_REJECTED"
    assert result.attempts == 2
    assert plan_calls == 2
    assert observe_calls == 3
    assert goal_handle.terminal_state == "aborted"


@pytest.mark.parametrize(
    "error_code",
    [
        "IK_FAILED",
        "FK_FAILED",
        "IK_JOINT5_LIMIT",
        "IK_JOINT5_RETRY_FAILED",
        "IK_JOINT5_BRANCH_CHANGED",
        "IK_JOINT5_MISSING",
        "CONTACT_REALIGN_FAILED",
        "CONTACT_COMPENSATION_FAILED",
        "CONTACT_Z_ERROR",
        "IK_FK_PREDICTED_CONTACT_Z",
        "WORKSPACE_REJECTED",
        "FK_FIXED_FINGER_BASE_SIDE_REJECTED",
        "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
        "TARGET_TABLETOP_COLLISION",
        "TF_UNAVAILABLE",
    ],
)
def test_execution_recovery_error_codes_are_retryable(error_code: str):
    assert _is_execution_retryable(PickFlowError(error_code, "transient execution failure", retryable=True)) is True


@pytest.mark.parametrize("error_code", ["RPC_TIMEOUT", "RPC_FAILED"])
def test_only_retryable_ik_fk_rpc_errors_are_retried(error_code: str):
    assert _is_execution_retryable(PickFlowError(error_code, "IK/FK transient failure", retryable=True)) is True
    assert _is_execution_retryable(PickFlowError(error_code, "primitive state unknown")) is False


def test_missing_target_after_grasp_failure_stops_without_another_execution():
    harness = _flow_harness(config={"max_execution_attempts": 3})
    first_candidate = _prepared_candidate(index=5)
    plan_calls = 0
    execution_calls = 0
    harness._preflight = lambda *_args: None
    harness._move_to_observe = lambda *_args: None

    def plan(*_args):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls == 1:
            return [first_candidate], "scene-1"
        raise PickFlowError("TARGET_NOT_VISIBLE", "target is not visible from the observation pose")

    def execute(*_args, **_kwargs):
        nonlocal execution_calls
        execution_calls += 1
        raise PickFlowError("GRASP_VERIFICATION_FAILED", "gripper is empty")

    harness._plan_and_prepare_candidates = plan
    harness._execute_candidate = execute
    goal_handle = _GoalHandle(mode=PickObject.Goal.MODE_EXECUTE)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success is False
    assert result.error_code == "TARGET_NOT_VISIBLE"
    assert plan_calls == 2
    assert execution_calls == 1
    assert goal_handle.terminal_state == "aborted"


def test_post_grasp_motion_target_uses_configured_container_joints():
    harness = SimpleNamespace(
        _arm_joint_names=["1", "2", "3", "4", "5"],
        _config={
            "post_grasp_motion": {
                "pose_name": "place_container",
                "joint_names": ["1", "2", "3", "4", "5"],
                "joint_positions": {
                    "1": -0.047553,
                    "2": -0.073631,
                    "3": -0.840621,
                    "4": 1.497165,
                    "5": -1.570790,
                },
                "duration_sec": 5.0,
            }
        },
    )

    config, target = ExecutionPhase._post_grasp_motion_target(cast(Any, harness))

    assert config["pose_name"] == "place_container"
    assert target.name == ["1", "2", "3", "4", "5"]
    assert target.position == pytest.approx([-0.047553, -0.073631, -0.840621, 1.497165, -1.570790])


def test_post_grasp_motion_target_rejects_incomplete_arm_target():
    harness = SimpleNamespace(
        _arm_joint_names=["1", "2", "3", "4", "5"],
        _config={
            "post_grasp_motion": {
                "joint_names": ["1", "2"],
                "joint_positions": {"1": 0.0, "2": 0.0},
            }
        },
    )

    with pytest.raises(PickFlowError, match="exactly the configured arm joints"):
        ExecutionPhase._post_grasp_motion_target(cast(Any, harness))


def test_release_at_transport_pose_opens_gripper_without_arm_motion():
    calls: list[tuple[str, dict]] = []
    harness = SimpleNamespace(
        _config={"release_settle_sec": 0.2},
        _gripper_open=1.0,
        _publish_feedback=lambda *_args: None,
        _run_primitive=lambda _goal, _deadline, _task_id, name, **kwargs: calls.append((name, kwargs)),
        _sleep_with_cancel=lambda *_args: None,
    )
    state = FlowState(completed_phases=[])

    ExecutionPhase._release_at_transport_pose(cast(Any, harness), None, time.monotonic() + 5.0, state, "task")

    assert calls == [("open_gripper", {"gripper_position": 1.0})]
    assert state.released_after_success is True


def test_release_at_transport_pose_reports_explicit_failure():
    harness = SimpleNamespace(
        _config={},
        _gripper_open=1.0,
        _publish_feedback=lambda *_args: None,
        _run_primitive=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PickFlowError("PRIMITIVE_FAILED", "open failed")
        ),
    )

    with pytest.raises(PickFlowError, match="open failed") as exc_info:
        ExecutionPhase._release_at_transport_pose(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "task",
        )

    assert exc_info.value.code == "RELEASE_FAILED"


def test_close_gripper_is_immediate_after_descent_before_diagnostics():
    class _ReachedClose(RuntimeError):
        pass

    logger = _Logger()
    primitives: list[str] = []
    feedback: list[str] = []
    prepared = _full_prepared_candidate()
    joint_state = prepared.final_joint_state
    live_joint_state = JointState()
    harness = SimpleNamespace(
        _config={
            "open_settle_sec": 0.0,
            "approach_velocity_scaling": 0.05,
            "descend_velocity_scaling": 0.03,
            "descend_duration_sec": 2.0,
            "target_gripper": {
                "closing_axis_ee": [1.0, 0.0, 0.0],
                "fixed_finger_robust_gap": {
                    "enabled": True,
                    "max_target_gap_deficit_m": 0.003,
                    "measurement_tolerance_m": 0.001,
                },
            },
        },
        _gripper_open=1.0,
        _gripper_closed=0.0,
        _joint_state_topic="/joint_states",
        get_logger=lambda: logger,
    )
    harness._snapshot_joint_state = lambda: live_joint_state
    harness._validate_joint5_branch_continuity = lambda *_args: None
    prepare_calls = []

    def prepare_candidate(selected, *_args, **kwargs):
        prepare_calls.append(kwargs)
        return replace(
            prepared,
            ranked=selected,
            plan=selected.plan,
        )

    harness._prepare_candidate = prepare_candidate
    harness._publish_feedback = lambda _goal, _state, phase, _detail: feedback.append(phase)
    harness._sleep_with_cancel = lambda *_args: None
    harness._pregrasp_pose = lambda *_args: (0.10, -0.20, 0.07)
    harness._move_branch_locked_pose = lambda *_args, **_kwargs: SimpleNamespace(joint_state=joint_state)
    harness._realign_contact = lambda _goal, _deadline, _task_id, _phase, xyz, _quaternion, *_args: (
        xyz,
        joint_state,
    )

    def record_pose_diagnostic(_goal, _deadline, _state, label, *_args, **_kwargs):
        if label == "grasp":
            raise AssertionError("grasp diagnostics must run only after close_gripper is dispatched")
        return None

    harness._record_pose_diagnostic = record_pose_diagnostic

    def run_primitive(_goal, _deadline, _task_id, primitive_name, **_kwargs):
        primitives.append(primitive_name)
        if primitive_name == "close_gripper":
            raise _ReachedClose

    harness._run_primitive = run_primitive

    with pytest.raises(_ReachedClose):
        PickExecutorNode._execute_candidate(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "commit-test",
            "marker",
            prepared,
            SimpleNamespace(),
        )

    assert feedback[-2:] == ["descend", "close"]
    assert "pregrasp" not in feedback
    assert primitives == ["open_gripper", "move_to_configuration", "close_gripper"]
    assert len(prepare_calls) == 2
    assert prepare_calls[0]["apply_compensation"] is True
    assert prepare_calls[0]["initial_seed"] is joint_state
    assert prepare_calls[0]["initial_seed"] is not live_joint_state
    assert "apply_compensation" not in prepare_calls[1]
    assert prepare_calls[1]["enforce_contact_error"] is False
    assert prepare_calls[1]["enforce_fixed_finger_robust_gap"] is False


def test_descend_uses_compensated_safe_waypoint_without_pregrasp_realign():
    class _ReachedClose(RuntimeError):
        pass

    logger = _Logger()
    primitives: list[str] = []
    branch_locked_calls: list[tuple] = []
    feedback: list[str] = []
    events: list[tuple] = []
    realign_phases: list[str] = []
    prepared = _full_prepared_candidate(index=3)
    joint_state = prepared.final_joint_state
    harness = SimpleNamespace(
        _config={
            "open_settle_sec": 0.0,
            "approach_velocity_scaling": 0.05,
            "descend_velocity_scaling": 0.03,
            "descend_duration_sec": 2.0,
            "target_gripper": {
                "closing_axis_ee": [1.0, 0.0, 0.0],
                "fixed_finger_robust_gap": {
                    "enabled": False,
                },
            },
        },
        _gripper_open=1.0,
        _gripper_closed=0.0,
        _joint_state_topic="/joint_states",
        get_logger=lambda: logger,
    )
    harness._snapshot_joint_state = lambda: joint_state
    harness._validate_joint5_branch_continuity = lambda *_args: None

    def prepare_candidate(selected, *_args, **kwargs):
        events.append(("prepare", len([event for event in events if event[0] == "prepare"]) + 1))
        return replace(prepared, ranked=selected, plan=selected.plan, actual_ee_xyz=selected.plan.grasp)

    harness._prepare_candidate = prepare_candidate
    harness._publish_feedback = lambda _goal, _state, phase, _detail: feedback.append(phase)
    harness._sleep_with_cancel = lambda *_args: None
    harness._pregrasp_pose = lambda selected, _scene: (
        selected.plan.grasp[0],
        selected.plan.grasp[1],
        selected.plan.grasp[2] + 0.03,
    )

    def realign_contact(_goal, _deadline, _task_id, phase, xyz, _quaternion, *_args):
        realign_phases.append(phase)
        return (xyz[0] + 0.03, xyz[1], xyz[2] + 0.01), joint_state

    harness._realign_contact = realign_contact
    harness._record_pose_diagnostic = lambda *_args, **_kwargs: None

    def move_branch_locked_pose(_goal, _deadline, _task_id, xyz, _quaternion, _scaling, _seed):
        branch_locked_calls.append(xyz)
        events.append(("move", xyz))
        return SimpleNamespace(joint_state=joint_state)

    harness._move_branch_locked_pose = move_branch_locked_pose

    def run_primitive(_goal, _deadline, _task_id, primitive_name, **_kwargs):
        primitives.append(primitive_name)
        if primitive_name == "close_gripper":
            raise _ReachedClose()

    harness._run_primitive = run_primitive

    with pytest.raises(_ReachedClose):
        PickExecutorNode._execute_candidate(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "align-test",
            "marker",
            prepared,
            SimpleNamespace(),
        )

    assert len(branch_locked_calls) == 2
    assert branch_locked_calls[0] == (0.10, -0.20, 0.12)
    assert branch_locked_calls[1] == pytest.approx((0.13, -0.20, 0.08))
    assert events.index(("prepare", 2)) < events.index(("move", pytest.approx((0.13, -0.20, 0.08))))
    assert realign_phases == ["approach"]
    assert "pregrasp" not in feedback
    assert primitives == ["open_gripper", "move_to_configuration", "close_gripper"]


def test_descend_failure_returns_to_observe_before_propagating_error():
    logger = _Logger()
    prepared = _full_prepared_candidate()
    joint_state = prepared.final_joint_state
    state = FlowState(completed_phases=[])
    observe_calls: list[str] = []
    harness = SimpleNamespace(
        _config={
            "open_settle_sec": 0.0,
            "approach_velocity_scaling": 0.05,
            "descend_velocity_scaling": 0.03,
            "descend_duration_sec": 2.0,
        },
        _gripper_open=1.0,
        _gripper_closed=0.0,
        _joint_state_topic="/joint_states",
        get_logger=lambda: logger,
    )
    harness._snapshot_joint_state = lambda: joint_state
    harness._validate_joint5_branch_continuity = lambda *_args: None
    harness._prepare_candidate = lambda selected, *_args, **_kwargs: replace(
        prepared,
        ranked=selected,
        plan=selected.plan,
    )
    harness._publish_feedback = lambda *_args: None
    harness._sleep_with_cancel = lambda *_args: None
    harness._pregrasp_pose = lambda *_args: (0.10, -0.20, 0.07)
    harness._move_branch_locked_pose = lambda *_args, **_kwargs: SimpleNamespace(joint_state=joint_state)
    harness._realign_contact = lambda _goal, _deadline, _task_id, _phase, xyz, _quaternion, *_args: (
        xyz,
        joint_state,
    )
    harness._record_pose_diagnostic = lambda *_args, **_kwargs: None
    harness._move_to_observe = lambda _goal, _deadline, _state, task_id: observe_calls.append(task_id)

    def run_primitive(_goal, _deadline, _task_id, primitive_name, **_kwargs):
        if primitive_name == "move_to_configuration":
            raise PickFlowError("PRIMITIVE_FAILED", "descent failed")

    harness._run_primitive = run_primitive

    with pytest.raises(PickFlowError, match="descent failed"):
        PickExecutorNode._execute_candidate(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            state,
            "descent-failure-test",
            "marker",
            prepared,
            SimpleNamespace(),
        )

    assert observe_calls == ["descent-failure-test"]
    assert state.recovery_completed is True
    assert state.pipeline_timings["subphase_recovery"] >= 0.0


def test_cancel_after_close_does_not_enter_failure_recovery():
    from manipulation_execution.pick_executor_node import PickCancelled

    logger = _Logger()
    prepared = _full_prepared_candidate()
    joint_state = prepared.final_joint_state
    recovery_calls: list[str] = []
    sleep_calls = 0
    harness = SimpleNamespace(
        _config={
            "open_settle_sec": 0.0,
            "hold_sec": 0.8,
            "approach_velocity_scaling": 0.05,
            "descend_velocity_scaling": 0.03,
            "descend_duration_sec": 2.0,
            "recover_after_close_failure": True,
        },
        _gripper_open=1.0,
        _gripper_closed=0.0,
        _joint_state_topic="/joint_states",
        get_logger=lambda: logger,
    )
    harness._snapshot_joint_state = lambda: joint_state
    harness._validate_joint5_branch_continuity = lambda *_args: None
    harness._prepare_candidate = lambda selected, *_args, **_kwargs: replace(
        prepared,
        ranked=selected,
        plan=selected.plan,
    )
    harness._publish_feedback = lambda *_args: None

    def sleep_with_cancel(*_args):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            raise PickCancelled()

    harness._sleep_with_cancel = sleep_with_cancel
    harness._pregrasp_pose = lambda *_args: (0.10, -0.20, 0.07)
    harness._move_branch_locked_pose = lambda *_args, **_kwargs: SimpleNamespace(joint_state=joint_state)
    harness._realign_contact = lambda _goal, _deadline, _task_id, _phase, xyz, _quaternion, *_args: (
        xyz,
        joint_state,
    )
    harness._record_pose_diagnostic = lambda *_args, **_kwargs: None
    harness._run_primitive = lambda *_args, **_kwargs: None
    harness._recover_after_close_failure = lambda *_args: recovery_calls.append("recover")

    with pytest.raises(PickCancelled):
        PickExecutorNode._execute_candidate(
            cast(Any, harness),
            None,
            time.monotonic() + 5.0,
            FlowState(completed_phases=[]),
            "cancel-test",
            "marker",
            prepared,
            SimpleNamespace(),
        )

    assert recovery_calls == []


def test_pick_result_serializes_pipeline_timings():
    state = FlowState(
        completed_phases=["planning", "selecting"],
        pipeline_timings={"graspgen_request": 1.5, "candidate_ik_fk": 0.25},
    )

    result = PickExecutorHelpers._result_from_state(state)

    assert json.loads(result.pipeline_timings_json) == {
        "candidate_ik_fk": 0.25,
        "graspgen_request": 1.5,
    }


def test_flow_state_records_feedback_phase_timings_and_accumulates_repeated_phases():
    state = FlowState(completed_phases=[])

    assert state.enter_phase("observe", now=10.0) is None
    assert state.enter_phase("planning", now=11.25) == ("observe", pytest.approx(1.25))
    assert state.enter_phase("selecting", now=12.0) == ("planning", pytest.approx(0.75))
    assert state.enter_phase("planning", now=12.5) == ("selecting", pytest.approx(0.5))
    assert state.enter_phase("completed", now=13.0) == ("planning", pytest.approx(0.5))
    state.finish_active_phase(now=13.2)

    assert state.pipeline_timings == {
        "phase_observe": pytest.approx(1.25),
        "phase_planning": pytest.approx(1.25),
        "phase_selecting": pytest.approx(0.5),
    }
