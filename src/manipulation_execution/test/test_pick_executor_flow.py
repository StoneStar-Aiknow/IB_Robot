import json
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from ibrobot_msgs.action import PickObject
from manipulation_execution.pick_executor_models import CandidateSelectionDiagnostics, FlowState
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


class _ServiceClient:
    def __init__(self, ready: bool):
        self.ready = ready
        self.wait_calls: list[float] = []

    def service_is_ready(self) -> bool:
        return self.ready

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        self.wait_calls.append(timeout_sec)
        return self.ready


class _ActionClient:
    def wait_for_server(self, *, timeout_sec: float) -> bool:
        del timeout_sec
        return True


class _GoalHandle:
    def __init__(self, mode: int):
        self.request = SimpleNamespace(
            task_id="test-pick",
            target_query="banana",
            timeout_sec=5.0,
            mode=mode,
            release_after_success=False,
            release_drop_height_m=-1.0,
        )
        self.is_cancel_requested = False
        self.terminal_state = ""

    def succeed(self) -> None:
        self.terminal_state = "succeeded"

    def abort(self) -> None:
        self.terminal_state = "aborted"

    def canceled(self) -> None:
        self.terminal_state = "canceled"


def _prepared_candidate(index: int):
    return SimpleNamespace(
        ranked=SimpleNamespace(index=index, score=0.8, fixed_finger_base_side=None, candidate=object()),
        selection_score=0.75,
        contact_z_error_m=0.001,
        contact_residual_xy_m=0.002,
        approach_axis_error_deg=1.0,
        closing_axis_error_deg=2.0,
        fixed_finger_envelope=None,
        fk_fixed_finger_base_side=None,
        predicted_robust_gap_headroom_m=None,
    )


def _flow_harness() -> SimpleNamespace:
    logger = _Logger()
    harness = SimpleNamespace(
        _config={"timeout_sec": 5.0, "max_execution_attempts": 1},
        _goal_lock=threading.Lock(),
        _goal_active=True,
        get_logger=lambda: logger,
    )
    harness._result_from_state = PickExecutorNode._result_from_state
    harness._publish_feedback = lambda _goal, state, phase, _detail: state.completed_phases.append(phase)
    harness._record_prepared_ranking = lambda *_args: None
    return harness


def test_observe_only_stops_before_planning_or_grasp_execution():
    harness = _flow_harness()
    calls = []
    harness._preflight = lambda _goal, _deadline, _state, mode: calls.append(("preflight", mode))
    harness._move_to_observe = lambda *_args: calls.append(("observe", None))
    harness._plan_and_prepare_candidates = lambda *_args: pytest.fail("observe-only must not plan")
    harness._execute_candidate = lambda *_args: pytest.fail("observe-only must not execute")
    goal_handle = _GoalHandle(PickObject.Goal.MODE_OBSERVE_ONLY)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success
    assert result.candidate_index == -1
    assert goal_handle.terminal_state == "succeeded"
    assert calls == [("preflight", PickObject.Goal.MODE_OBSERVE_ONLY), ("observe", None)]


def test_plan_only_reports_candidate_without_grasp_execution():
    harness = _flow_harness()
    prepared = _prepared_candidate(11)
    harness._preflight = lambda *_args: None
    harness._move_to_observe = lambda *_args: None
    harness._plan_and_prepare_candidates = lambda *_args: ([prepared], "scene")
    harness._execute_candidate = lambda *_args: pytest.fail("plan-only must not execute")
    goal_handle = _GoalHandle(PickObject.Goal.MODE_PLAN_ONLY)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success
    assert result.attempts == 0
    assert result.candidate_index == 11
    assert goal_handle.terminal_state == "succeeded"


def test_execute_mode_reports_the_attempted_candidate_index():
    harness = _flow_harness()
    prepared = _prepared_candidate(5)
    harness._preflight = lambda *_args: None
    harness._move_to_observe = lambda *_args: None
    harness._plan_and_prepare_candidates = lambda *_args: ([prepared], "scene")
    harness._execute_candidate = lambda *_args: None
    goal_handle = _GoalHandle(PickObject.Goal.MODE_EXECUTE)

    result = PickExecutorNode._execute_pick(cast(Any, harness), goal_handle)

    assert result.success
    assert result.attempts == 1
    assert result.candidate_index == 5


@pytest.mark.parametrize("retry_code", ["RPC_TIMEOUT", "IK_FAILED"])
def test_selection_attempts_retry_transient_failure_and_record_diagnostics(retry_code):
    harness = SimpleNamespace(
        _config={"candidate_selection": {"selection_attempts": 2, "retry_settle_sec": 0.0}},
        _joint_state_topic="/joint_states",
        get_logger=lambda: _Logger(),
    )
    requests = 0

    def request(*_args):
        nonlocal requests
        requests += 1
        if requests == 1:
            raise PickFlowError(retry_code, "transient selection failure")
        header = SimpleNamespace(frame_id="camera", stamp=SimpleNamespace(sec=1, nanosec=2))
        return header, ["candidate"], "scene"

    harness._request_grasps = request
    harness._lookup_base_transform = lambda *_args: "transform"
    harness._transform_to_matrix = lambda _transform: np.eye(4, dtype=np.float64)
    harness._record_frame_diagnostic = lambda *_args, **_kwargs: None
    harness._scene_geometry_base = lambda *_args: "scene_base"
    harness._publish_feedback = lambda *_args: None
    harness._rank_candidates = lambda *_args, **_kwargs: ["ranked"]
    harness._snapshot_joint_state = lambda: "seed"

    def prepare(*_args, diagnostics, **_kwargs):
        diagnostics.prepared_candidates = 1
        return ["prepared"], None

    harness._prepare_ranked_candidates = prepare
    harness._sleep_with_cancel = lambda *_args: None
    harness._record_candidate_selection_diagnostics = PickExecutorNode._record_candidate_selection_diagnostics.__get__(
        harness
    )
    state = FlowState(completed_phases=[])

    prepared, scene = PickExecutorNode._plan_and_prepare_candidates(
        cast(Any, harness), None, time.monotonic() + 5.0, state, "banana"
    )

    assert prepared == ["prepared"]
    assert scene == "scene_base"
    assert requests == 2
    assert [item["terminal_code"] for item in state.candidate_selection_diagnostics] == [retry_code, "SUCCESS"]


def test_selection_attempts_do_not_retry_non_transient_failure():
    harness = SimpleNamespace(
        _config={"candidate_selection": {"selection_attempts": 3}},
        get_logger=lambda: _Logger(),
    )
    harness._request_grasps = lambda *_args: (_ for _ in ()).throw(
        PickFlowError("TARGET_TABLETOP_UNAVAILABLE", "missing table plane")
    )
    harness._record_candidate_selection_diagnostics = lambda *_args: None

    with pytest.raises(PickFlowError, match="missing table plane"):
        PickExecutorNode._plan_and_prepare_candidates(
            cast(Any, harness), None, time.monotonic() + 5.0, FlowState(completed_phases=[]), "banana"
        )


def test_preflight_allows_optional_fallback_detection_service_to_be_missing():
    fallback = _ServiceClient(ready=False)
    required = _ServiceClient(ready=True)
    harness = SimpleNamespace(
        _primitive_client=_ActionClient(),
        _planner_client=required,
        _detect_client=fallback,
        _move_configuration_client=required,
        _verifier_client=required,
        _ik_client=required,
        _fk_client=required,
        _ik_worker_clients=[],
        _fk_worker_clients=[],
        _ready_timeout=0.1,
        _planner_service="/planner",
        _detect_service="/optional/detect",
        _move_configuration_service="/move",
        _verifier_service="/verify",
        _ik_service="/ik",
        _fk_service="/fk",
        _ik_worker_prefix="/worker",
        _joint_state_topic="/joint_states",
        _config={},
        _verification_policy="required",
        _primitive_action_name="/primitive",
        _remaining=lambda _deadline: 1.0,
        _snapshot_joint_state=lambda: object(),
        _publish_feedback=lambda *_args: None,
        get_logger=lambda: _Logger(),
    )
    harness._wait_for_service = PickExecutorNode._wait_for_service.__get__(harness)

    PickExecutorNode._preflight(
        cast(Any, harness),
        None,
        time.monotonic() + 1.0,
        FlowState(completed_phases=[]),
        PickObject.Goal.MODE_EXECUTE,
    )

    assert fallback.wait_calls


def test_pick_result_serializes_pipeline_timings():
    state = FlowState(
        completed_phases=["planning", "selecting"],
        pipeline_timings={"graspgen_request": 1.5, "candidate_ik_fk": 0.25},
    )

    result = PickExecutorNode._result_from_state(state)

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


def test_publish_feedback_connects_runtime_phase_timing(monkeypatch):
    logger = _Logger()
    feedback = []
    harness = SimpleNamespace(
        _PHASE_PROGRESS={"preflight": 0.1, "observe": 0.2},
        _check_cancel=lambda _goal: None,
        get_logger=lambda: logger,
    )
    goal_handle = SimpleNamespace(publish_feedback=feedback.append)
    state = FlowState(completed_phases=[])
    timestamps = iter([10.0, 11.5])
    monkeypatch.setattr(time, "monotonic", lambda: next(timestamps))

    PickExecutorNode._publish_feedback(cast(Any, harness), goal_handle, state, "preflight", "checking")
    PickExecutorNode._publish_feedback(cast(Any, harness), goal_handle, state, "observe", "moving")

    assert state.pipeline_timings == {"phase_preflight": pytest.approx(1.5)}
    assert any("stage=phase_preflight" in message for message in logger.messages)
    assert [item.phase for item in feedback] == ["preflight", "observe"]


def test_candidate_selection_diagnostics_are_logged_and_written(tmp_path):
    logger = _Logger()
    harness = SimpleNamespace(get_logger=lambda: logger)
    state = FlowState(completed_phases=[], debug_output_dir=str(tmp_path))
    diagnostics = CandidateSelectionDiagnostics(
        selection_attempt=2,
        raw_candidates=5,
        geometry_surviving_candidates=3,
        ranked_candidates=3,
        truncated_by_candidate_budget=1,
        prepared_candidates=1,
        geometry_rejections={"WORKSPACE_REJECTED": 2},
        terminal_code="SUCCESS",
        duration_s=0.25,
    )

    PickExecutorNode._record_candidate_selection_diagnostics(cast(Any, harness), state, diagnostics)

    record = json.loads((tmp_path / "pick_candidate_rejections.json").read_text(encoding="utf-8"))[0]
    assert record["selection_attempt"] == 2
    assert record["geometry_rejections"] == {"WORKSPACE_REJECTED": 2}
    assert record["prepared_candidates"] == 1
    assert any(message.startswith("CANDIDATE_SELECTION_STATS ") for message in logger.messages)
