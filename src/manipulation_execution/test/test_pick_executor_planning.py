from types import SimpleNamespace
from typing import Any, cast

import manipulation_execution.phases.planning as planning_module
from manipulation_execution.grasp_geometry import CandidatePlan
from manipulation_execution.pick_executor_models import BaseSceneGeometry, PlannerSceneGeometry
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
            lift=(float(index), 0.0, 0.10),
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

    ranked = PickExecutorNode._rank_candidates(
        cast(Any, harness),
        "camera",
        SimpleNamespace(sec=0, nanosec=0),
        cast(Any, None),
        [_candidate(index) for index in range(6)],
        PlannerSceneGeometry(),
        BaseSceneGeometry(),
    )

    assert built_indices == list(range(6))
    assert [candidate.index for candidate in ranked] == [3, 4]
