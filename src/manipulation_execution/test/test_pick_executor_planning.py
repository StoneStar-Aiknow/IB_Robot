from types import SimpleNamespace
from typing import Any, cast

import pytest

import manipulation_execution.phases.planning as planning_module
from manipulation_execution.grasp_geometry import CandidatePlan
from manipulation_execution.pick_executor_models import (
    BaseSceneGeometry,
    CandidateSelectionDiagnostics,
    PickFlowError,
    PlannerSceneGeometry,
)
from manipulation_execution.pick_executor_node import PickExecutorNode


def _candidate(index: int):
    return SimpleNamespace(
        confidence=0.9 - 0.01 * index,
        collision_free=True,
        header=SimpleNamespace(frame_id="", stamp=SimpleNamespace(sec=0, nanosec=0)),
        pose_matrix=[float(index)],
        target_width_m=0.02,
        target_width_quality=1.0,
        width_axis_camera=SimpleNamespace(x=1.0, y=0.0, z=0.0),
        target_width_min_offset_m=-0.01,
        target_width_max_offset_m=0.01,
    )


def test_candidate_budget_is_applied_after_cheap_geometry_filters(monkeypatch):
    built_indices: list[int] = []

    def build_plan(pose_matrix, *_args, **_kwargs):
        index = int(pose_matrix[0])
        built_indices.append(index)
        contact_z = -0.1 if index < 3 else 0.05
        return CandidatePlan(
            approach=(float(index), 0.0, 0.12),
            grasp=(float(index), 0.0, 0.05),
            quaternion=(0.0, 0.0, 0.0, 1.0),
            approach_axis=(0.0, 0.0, -1.0),
            target_contact_ee=(0.0, 0.0, 0.0),
            target_contact_base=(float(index), 0.0, contact_z),
            target_width_m=0.02,
            width_reason="test",
            fixed_finger_target_gap_m=0.01,
            target_width_min_base=None,
            target_width_max_base=None,
            topdown_score=1.0,
        )

    monkeypatch.setattr(planning_module, "build_candidate_plan", build_plan)
    monkeypatch.setattr(planning_module, "xyz_within_workspace", lambda *_args: (True, ""))
    harness = SimpleNamespace(
        _config={
            "candidate_selection": {
                "min_contact_z": 0.0,
                "min_approach_z": 0.04,
                "max_candidates": 2,
            },
            "execution_scoring": {},
            "target_gripper": {},
        },
        _workspace={},
        _target_geometry={},
        _stamp_to_ns=PickExecutorNode._stamp_to_ns,
    )
    diagnostics = CandidateSelectionDiagnostics(selection_attempt=1)

    ranked = PickExecutorNode._rank_candidates(
        cast(Any, harness),
        "camera",
        SimpleNamespace(sec=0, nanosec=0),
        cast(Any, None),
        [_candidate(index) for index in range(6)],
        PlannerSceneGeometry(),
        BaseSceneGeometry(),
        diagnostics=diagnostics,
    )

    assert built_indices == list(range(6))
    assert [candidate.index for candidate in ranked] == [3, 4]
    assert diagnostics.geometry_rejections == {"HEIGHT_OR_APPROACH_REJECTED": 3}
    assert diagnostics.geometry_surviving_candidates == 3
    assert diagnostics.ranked_candidates == 3
    assert diagnostics.truncated_by_candidate_budget == 1


def test_fallback_detection_returns_none_without_call_when_service_is_unavailable():
    client = SimpleNamespace(
        service_is_ready=lambda: False,
        call_async=lambda _request: pytest.fail("unavailable fallback service must not be called"),
    )
    harness = SimpleNamespace(
        _detect_client=client,
        _detect_service="/optional/detect",
        get_logger=lambda: SimpleNamespace(warning=lambda *_args: None),
    )

    result = PickExecutorNode._request_fallback_detection_centroid(
        cast(Any, harness),
        None,
        10.0,
        "banana",
        "volume",
        0.1,
    )

    assert result is None


@pytest.mark.parametrize(
    ("message", "diagnostics"),
    [
        ("Detection failed: no matching detection", ["failure_stage: detection"]),
        ("No grasps generated: empty candidate set", ["detection_status: ok", "final_grasp_count: 0"]),
    ],
)
def test_planner_failure_classifies_missing_target_as_fail_fast(message: str, diagnostics: list[str]):
    response = SimpleNamespace(message=message, diagnostic_details=diagnostics)

    failure = PickExecutorNode._planner_failure(response)

    assert failure.code == "TARGET_NOT_VISIBLE"
    assert failure.retryable is False


@pytest.mark.parametrize("reason", ["detect_service_unavailable", "segment_service_unavailable"])
def test_planner_failure_preserves_perception_service_unavailable(reason: str):
    response = SimpleNamespace(
        message=f"Detection failed: {reason}",
        diagnostic_details=["failure_stage: detection", f"failure_reason: {reason}"],
    )

    failure = PickExecutorNode._planner_failure(response)

    assert failure.code == "SERVICE_UNAVAILABLE"
    assert "target is not visible" not in str(failure)
    assert failure.retryable is False


def test_all_workspace_rejections_return_fail_fast_error(monkeypatch):
    plan = CandidatePlan(
        approach=(0.8, 0.0, 0.12),
        grasp=(0.8, 0.0, 0.05),
        quaternion=(0.0, 0.0, 0.0, 1.0),
        approach_axis=(0.0, 0.0, -1.0),
        target_contact_ee=(0.0, 0.0, 0.0),
        target_contact_base=(0.8, 0.0, 0.05),
        target_width_m=0.02,
        width_reason="test",
        fixed_finger_target_gap_m=0.01,
        target_width_min_base=None,
        target_width_max_base=None,
        topdown_score=1.0,
    )
    monkeypatch.setattr(planning_module, "build_candidate_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(planning_module, "xyz_within_workspace", lambda *_args: (False, "x outside workspace"))
    harness = SimpleNamespace(
        _config={"candidate_selection": {}, "execution_scoring": {}, "target_gripper": {}},
        _workspace={},
        _target_geometry={},
        _stamp_to_ns=PickExecutorNode._stamp_to_ns,
    )
    diagnostics = CandidateSelectionDiagnostics(selection_attempt=1)

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._rank_candidates(
            cast(Any, harness),
            "camera",
            SimpleNamespace(sec=0, nanosec=0),
            cast(Any, None),
            [_candidate(0), _candidate(1)],
            PlannerSceneGeometry(),
            BaseSceneGeometry(),
            diagnostics=diagnostics,
        )

    assert exc_info.value.code == "TARGET_OUTSIDE_WORKSPACE"
    assert exc_info.value.retryable is False
    assert diagnostics.geometry_rejections == {"WORKSPACE_REJECTED": 2}
