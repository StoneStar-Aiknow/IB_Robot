"""Only recoverable placement searches may request another observation, never device/model faults."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import place_request
from aimdk_grasp.place_request import X2PlacePlanner
from aimdk_grasp.planner import LeftArmPlanner, PlacementSearchExhausted, PlannerFault
from aimdk_grasp.revalidation import ReobserveRequired

from manipulation_execution.grasp_pipeline.contracts import GraspFailure


@pytest.mark.parametrize(
    "failure,retryable",
    [
        (PlacementSearchExhausted("no complete path in this snapshot"), True),
        (TimeoutError("planning deadline exceeded"), True),
        (ReobserveRequired("scene stale"), True),
        (RuntimeError("invalid start configuration"), False),
        (ValueError("bad hand cover"), False),
        (PlannerFault("CUDA context failed"), False),
    ],
)
def test_place_request_distinguishes_retryable_searches_from_faults(monkeypatch, failure, retryable):
    joints = tuple(f"j{i}" for i in range(7))
    feedback = SimpleNamespace(vector=lambda names: np.zeros(7))
    admitted = SimpleNamespace(live=SimpleNamespace(feedback=feedback), admitted_at=0.0)
    observation = SimpleNamespace(robot_state=admitted)

    class Service:
        @contextmanager
        def session(self, *args, **kwargs):
            yield self

        def plan_place(self, *args, **kwargs):
            raise failure

    spheres = SimpleNamespace(arm=joints, last_link_frames=np.eye(4)[:3][None, None], link_indices={"R_palm": 0})
    arm = SimpleNamespace(spheres=spheres, gpu=nullcontext(), service=Service())
    planner = X2PlacePlanner(SimpleNamespace(arms={"right": arm}), {"right": SimpleNamespace(place=None)})
    held = SimpleNamespace(
        arm_side="right", joint_names=joints, joint_positions=np.zeros(7), home_positions=np.zeros(7), return_times=()
    )
    monkeypatch.setattr(place_request, "measured_spheres", lambda *args: None)
    monkeypatch.setattr(place_request, "place_targets", lambda *args: [(np.eye(4), np.eye(4))])
    monkeypatch.setattr(place_request, "placement_cover", lambda *args: np.zeros((1, 4)))
    monkeypatch.setattr(place_request, "opening_cover", lambda *args: np.zeros((1, 4)))

    with pytest.raises(GraspFailure) as raised:
        planner.plan(
            observation, SimpleNamespace(scene="snapshot"), held, deadline=float("inf"), cancelled=lambda: False
        )
    assert raised.value.code == "PLACE_PLAN_FAILED" and raised.value.retryable is retryable
    assert str(raised.value) == str(failure)


def test_exhausted_placement_candidates_have_a_typed_recoverable_failure():
    calls = []

    def solve(*args, **kwargs):
        calls.append(None)
        return None  # No reachable release pose in this observation.

    planner = SimpleNamespace(
        snapshot_matches=lambda *args: True,
        carrying=True,
        _measured_start=lambda feedback: np.zeros(7),
        _require_valid_start=lambda q: None,
        max_goalset=10,
        _goal_pose=lambda poses: poses,
        _state=lambda q: q,
        motion=SimpleNamespace(ik_solver=SimpleNamespace(solve_pose=solve)),
    )
    with pytest.raises(PlacementSearchExhausted, match="no complete placement path"):
        LeftArmPlanner.plan_place(
            planner,
            "feedback",
            [(np.eye(4), np.eye(4))],
            np.zeros(7),
            0.0,
            check_deadline=lambda: None,
            deadline=float("inf"),
        )
    assert len(calls) == 1


def test_place_request_logs_its_timings_also_for_a_failed_search(monkeypatch, capsys):
    import json

    joints = tuple(f"j{i}" for i in range(7))
    feedback = SimpleNamespace(vector=lambda names: np.zeros(7))
    admitted = SimpleNamespace(live=SimpleNamespace(feedback=feedback), admitted_at=0.0)

    class Service:
        @contextmanager
        def session(self, *args, **kwargs):
            yield self

        def plan_place(self, *args, **kwargs):
            self.place_report = {"rounds": 3, "failed_stages": {"opening": 3}, "timings_s": {"total": 0.5}}
            raise PlacementSearchExhausted("no complete placement path after 3 rounds")

    spheres = SimpleNamespace(arm=joints, last_link_frames=np.eye(4)[:3][None, None], link_indices={"R_palm": 0})
    arm = SimpleNamespace(spheres=spheres, gpu=nullcontext(), service=Service())
    planner = X2PlacePlanner(SimpleNamespace(arms={"right": arm}), {"right": SimpleNamespace(place=None)})
    held = SimpleNamespace(
        arm_side="right", joint_names=joints, joint_positions=np.zeros(7), home_positions=np.zeros(7), return_times=()
    )
    monkeypatch.setattr(place_request, "measured_spheres", lambda *args: None)
    monkeypatch.setattr(place_request, "place_targets", lambda *args: [(np.eye(4), np.eye(4))])
    monkeypatch.setattr(place_request, "placement_cover", lambda *args: np.zeros((1, 4)))
    monkeypatch.setattr(place_request, "opening_cover", lambda *args: np.zeros((1, 4)))

    with pytest.raises(GraspFailure):
        planner.plan(
            SimpleNamespace(robot_state=admitted),
            SimpleNamespace(scene="snapshot"),
            held,
            deadline=float("inf"),
            cancelled=lambda: False,
        )
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith('{"place_plan"')]
    assert len(lines) == 1
    trace = lines[0]["place_plan"]
    assert trace["arm_side"] == "right" and trace["outcome"] == "PlacementSearchExhausted"
    assert trace["planner"]["failed_stages"] == {"opening": 3}
    assert set(trace["timings_s"]) >= {"targets", "gpu_wait", "session", "plan_place", "total"}
    assert "recheck" not in trace["timings_s"]  # the search failed before the recheck
