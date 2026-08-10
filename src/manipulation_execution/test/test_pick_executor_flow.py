import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from manipulation_execution.grasp_geometry import CandidatePlan, FixedFingerEnvelope
from manipulation_execution.pick_executor_helpers import PickExecutorHelpers
from manipulation_execution.pick_executor_models import FlowState, PreparedCandidate, RankedCandidate
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError
from rclpy.clock import Clock
from sensor_msgs.msg import JointState

from embodied_common.dispatch_binding import delegated_executor_identity, new_binding
from ibrobot_msgs.action import PickObject
from ibrobot_msgs.msg import GraspCandidate


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


class _GoalHandle:
    def __init__(self, *, mode: int, release_after_success: bool = False, release_drop_height_m: float = -1.0):
        self.request = PickObject.Goal()
        self.request.dispatch_binding = new_binding(task_id="test-pick")
        self.request.dispatch_binding.task_budget.schema_version = 1
        self.request.dispatch_binding.task_budget.started_at.sec = 1
        self.request.dispatch_binding.task_budget.deadline.sec = 2_000_000_000
        self.request.target_query = "banana"
        self.request.timeout_sec = 5.0
        self.request.mode = mode
        self.request.release_after_success = release_after_success
        self.request.release_drop_height_m = release_drop_height_m
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
        lift=(0.1, -0.2, 0.10),
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
    return harness


def test_selection_attempts_retry_the_canonical_planning_path():
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
        if len(requests) == 1:
            raise PickFlowError("NO_GRASP_CANDIDATES", "empty first frame")
        return SimpleNamespace(frame_id="camera", stamp=SimpleNamespace(sec=1, nanosec=2)), ["candidate"], "scene"

    harness._request_grasps = request
    harness._lookup_base_transform = lambda *_args: "transform"
    harness._transform_to_matrix = lambda _transform: np.eye(4, dtype=np.float64)
    harness._record_frame_diagnostic = lambda *_args, **_kwargs: None
    harness._scene_geometry_base = lambda _transform, _scene: "scene_base"
    harness._publish_feedback = lambda *_args, **_kwargs: None
    harness._rank_candidates = lambda *_args: ["ranked"]
    harness._snapshot_joint_state = lambda: "joint_seed"
    harness._prepare_ranked_candidates = lambda *_args: (["prepared"], None)
    harness._sleep_with_cancel = lambda *_args: None

    prepared, scene_base = PickExecutorNode._plan_and_prepare_candidates(
        cast(Any, harness),
        None,
        time.monotonic() + 5.0,
        FlowState(completed_phases=[]),
        "banana",
    )

    assert prepared == ["prepared"]
    assert scene_base == "scene_base"
    assert len(requests) == 2
    assert any("grasp selection retry" in message for message in logger.messages)


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
    harness._rank_candidates = lambda *_args: ["ranked"]
    harness._snapshot_joint_state = lambda: "joint_seed"
    harness._prepare_ranked_candidates = lambda *_args: (["prepared"], None)
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
        release_drop_height_m=0.015,
    )

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert captured == {
        "selected": prepared,
        "release_after_success": True,
        "release_drop_height_m": pytest.approx(0.015),
    }
    assert result.success is True
    assert result.attempts == 1
    assert result.candidate_index == 5
    assert result.verification_status == PickObject.Result.VERIFICATION_SUCCESS
    assert result.released_after_success is True
    assert goal_handle.terminal_state == "succeeded"


def test_post_success_release_descends_then_opens_gripper():
    logger = _Logger()
    calls: list[tuple] = []
    harness = SimpleNamespace(
        _config={"release_velocity_scaling": 0.04, "release_settle_sec": 0.2},
        _gripper_open=1.0,
        get_logger=lambda: logger,
    )
    harness._publish_feedback = lambda _goal, _state, phase, detail: calls.append(("feedback", phase, detail))

    def move(_goal, _deadline, task_id, xyz, quaternion, velocity, seed, **kwargs):
        calls.append(("move", task_id, xyz, quaternion, velocity, seed, kwargs))
        return SimpleNamespace(joint_state="release-seed")

    harness._move_branch_locked_pose = move
    harness._run_primitive = lambda _goal, _deadline, task_id, primitive, **kwargs: calls.append(
        ("primitive", task_id, primitive, kwargs)
    )
    harness._sleep_with_cancel = lambda _goal, _deadline, duration: calls.append(("sleep", duration))
    state = FlowState(completed_phases=[])
    prepared = _prepared_candidate(index=3)

    PickExecutorNode._release_verified_pick(
        cast(Any, harness),
        None,
        time.monotonic() + 5.0,
        state,
        "release-test",
        prepared,
        "lift-seed",
        0.015,
    )

    move_call = next(call for call in calls if call[0] == "move")
    assert move_call[2] == pytest.approx((0.1, -0.2, 0.035))
    assert move_call[4] == pytest.approx(0.04)
    assert move_call[5] == "lift-seed"
    assert move_call[6] == {"validate_orientation": False}
    primitive_call = next(call for call in calls if call[0] == "primitive")
    assert primitive_call[2] == "open_gripper"
    assert primitive_call[3] == {"gripper_position": pytest.approx(1.0)}
    assert calls[-1] == ("sleep", 0.2)
    assert state.released_after_success is True


def test_close_gripper_is_immediate_after_descent_before_diagnostics():
    class _ReachedClose(RuntimeError):
        pass

    logger = _Logger()
    primitives: list[str] = []
    feedback: list[str] = []
    candidate = GraspCandidate()
    candidate.confidence = 0.8
    plan = CandidatePlan(
        approach=(0.10, -0.20, 0.12),
        grasp=(0.10, -0.20, 0.04),
        lift=(0.10, -0.20, 0.10),
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
    ranked = RankedCandidate(index=7, candidate=candidate, plan=plan, score=0.9)
    joint_state = JointState()
    prepared = PreparedCandidate(
        ranked=ranked,
        plan=plan,
        final_joint_state=joint_state,
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
        get_logger=lambda: logger,
    )
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
    assert primitives == ["open_gripper", "move_to_configuration", "close_gripper"]
    assert len(prepare_calls) == 2
    assert "enforce_fixed_finger_robust_gap" not in prepare_calls[0]
    assert prepare_calls[1]["enforce_fixed_finger_robust_gap"] is False


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
