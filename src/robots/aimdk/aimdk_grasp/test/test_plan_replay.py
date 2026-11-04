"""Replaying recorded plan calls: clocks shifted together, the leg as coarse boxes, live beside replayed."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.plan_replay import recorded_calls, replay, shifted, summary, with_leg_boxes
from aimdk_grasp.scene_estimation import LEG_OBSTACLE, SceneEstimate, leg_boxes


def recorded(received_at, *, success=True, snapshot_now=None):
    feedback = Feedback(received_at - 0.02, {"left_elbow_joint": -1.2}, np.zeros(10))
    scene = {"cuboid": {"table": {"dims": [1.0, 1.0, 0.1]}}, "mesh": {}}
    estimate = SceneEstimate(scene, received_at, 1, "camera", True, None, {}, dict(scene))
    options = {"screen_check": True} if snapshot_now is None else {"screen_check": True, "snapshot_now": snapshot_now}
    outcome = dict(success=success, wall_s=1.0) if success else dict(success=False, error="RuntimeError: x")
    return dict(
        feedback=feedback,
        estimate=estimate,
        camera_from_grasps=np.stack([np.eye(4)] * 3),
        confidence=np.array([0.9, 0.8, 0.7]),
        base_from_camera=np.eye(4),
        options=options,
        budget_s=4.0,
        outcome=outcome,
    )


def test_every_clock_of_a_call_moves_together():
    call = recorded(50.0, snapshot_now=50.3)
    feedback, estimate, options = shifted(call, 1000.0)
    assert (estimate.received_at, feedback.received_at, options["snapshot_now"]) == (1050.0, 1049.98, 1050.3)
    assert options["screen_check"] and call["options"]["snapshot_now"] == 50.3  # the recording is untouched


def test_the_leg_boxes_join_the_coarse_scene_only():
    estimate = with_leg_boxes(recorded(50.0)["estimate"], 0.025)
    assert set(estimate.scene["cuboid"]) == {"table", *leg_boxes(0.025)}
    assert set(estimate.validation_scene["cuboid"]) == {"table"}
    assert all(name.startswith(LEG_OBSTACLE) for name in set(estimate.scene["cuboid"]) - {"table"})


class FakeService:
    """Plans while the coarse scene has the leg boxes; refuses without them."""

    def __init__(self):
        self._planner, self.calls, self.seen = None, [], []

    def prepare(self, feedback, estimate):
        self._planner = SimpleNamespace(locked_joints_match=lambda feedback: True)
        self.calls.append("prepare")

    def plan(self, feedback, estimate, grasps, confidence, matrix, *, deadline, **options):
        self.calls.append("plan")
        self.seen.append((estimate.received_at, options.get("snapshot_now"), deadline))
        if not any(name.startswith(LEG_OBSTACLE) for name in estimate.scene["cuboid"]):
            raise RuntimeError("Interpolated approach segment violates fine collision constraints: world 50 mm")
        return None, {"candidate_index": 2, "planning_attempts": [{}, {}]}


def test_a_replay_runs_each_call_in_order_on_increasing_clocks_beside_its_live_outcome(tmp_path):
    for name, call in (
        ("plan_1.pkl", recorded(50.0, snapshot_now=50.3)),
        ("plan_2.pkl", recorded(20.0, success=False)),
    ):
        (tmp_path / name).write_bytes(pickle.dumps(call))
    files = recorded_calls([tmp_path])
    service = FakeService()
    rows = replay(files, service, repeats=2)
    # Built once, before the first timed call; the second recording is older yet replays later.
    assert service.calls == ["prepare"] + ["plan"] * 4 and [row["rebuilt"] for row in rows] == [True] + [False] * 3
    received = [seen[0] for seen in service.seen]
    assert received == sorted(received) and abs(service.seen[0][1] - received[0] - 0.3) < 1e-9
    assert [row["success"] for row in rows] == [False] * 4 and all(row["budget_s"] == 4.0 for row in rows)
    assert [row["live"]["success"] for row in rows] == [True, False, True, False]
    boxed = replay(files, FakeService(), leg_shrink_m=0.025)
    assert [(row["success"], row["candidate"], row["attempts"]) for row in boxed] == [(True, 2, 2)] * 2
    totals = summary(rows)
    assert (totals["calls"], totals["planned"], totals["live_planned"], totals["fine_failures"]) == (4, 0, 2, 4)
