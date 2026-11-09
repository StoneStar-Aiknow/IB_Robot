"""Recording place plan inputs and replaying them: what a place plan saw, and its seconds, kept for offline runs."""

from __future__ import annotations

import json
import pickle
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import place_request
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.joints import arm_joints
from aimdk_grasp.place_replay import recorded_places, replay, shifted, summary
from aimdk_grasp.place_request import X2PlacePlanner
from aimdk_grasp.scene_estimation import SceneEstimate
from aimdk_grasp.trajectory import TimedTrajectory

JOINTS = arm_joints("right")


@dataclass(frozen=True)
class Container:
    scene: SceneEstimate


@dataclass(frozen=True)
class Held:
    arm_side: str = "right"
    joint_names: tuple = JOINTS
    joint_positions: tuple = (0.0,) * 7
    home_positions: tuple = (0.0,) * 7
    return_times: tuple = ()


def scene(received_at):
    geometry = {"cuboid": {"table": {"dims": [1.0, 1.0, 0.1]}}, "mesh": {}}
    return SceneEstimate(geometry, received_at, 1, "camera", True, None, {}, dict(geometry))


def path(seconds):
    return TimedTrajectory(np.array([0.0, seconds]), np.zeros((2, 7)), JOINTS)


class PlanningService:
    """Plans every place in four timed segments; records how it was called."""

    def __init__(self, *, fail=None):
        self._planner, self.calls, self.fail = None, [], fail
        self.place_report = {}

    def locked_joints_match(self, feedback, carrying=False):
        return True

    def prepare(self, feedback, estimate, *, carrying=False):
        self._planner = self
        self.calls.append(("prepare", carrying))

    @contextmanager
    def session(self, feedback, estimate, *, attached=None):
        self.calls.append(("session", len(attached)))
        yield self

    def plan_place(self, feedback, targets, home, now, *, check_deadline, deadline, opening, return_path):
        self.calls.append(("plan_place", feedback.received_at, now, len(targets), len(opening), deadline))
        self.place_report = {"candidate": 1, "rounds": 1, "timings_s": {"screen": 0.25, "total": 0.5}}
        if self.fail is not None:
            raise self.fail
        return path(2.0), path(0.5), path(0.4), path(1.0)


class Camera:
    def __init__(self, feedback):
        self.calls = deque([{"stamp_ns": 7, "wait_s": 0.01, "gpu_wait_s": 0.3, "registration_s": 0.05}])
        self.current = SimpleNamespace(
            observation=SimpleNamespace(transform=SimpleNamespace(stamp_ns=7)), feedback=feedback
        )

    def capture_state(self):
        return self.current

    def clock(self):
        return 0.0


def three_targets(*args):
    return [(np.eye(4), np.eye(4))] * 3


def place_planner(monkeypatch, service, feedback, targets=three_targets):
    spheres = SimpleNamespace(arm=JOINTS, last_link_frames=np.eye(4)[:3][None, None], link_indices={"R_palm": 0})
    arm = SimpleNamespace(spheres=spheres, gpu=nullcontext(), service=service, camera=Camera(feedback))
    robot = SimpleNamespace(place="config", limits=None, gravity=None, planning=SimpleNamespace(revalidation=None))
    monkeypatch.setattr(place_request, "measured_spheres", lambda *args: None)
    monkeypatch.setattr(place_request, "place_targets", targets)
    monkeypatch.setattr(place_request, "placement_cover", lambda *args: np.zeros((2, 4)))
    monkeypatch.setattr(place_request, "opening_cover", lambda *args: np.zeros((5, 4)))
    monkeypatch.setattr(place_request, "require_executable", lambda *args: 0.0)
    monkeypatch.setattr(place_request, "require_fresh", lambda *args: None)
    monkeypatch.setattr(place_request, "require_posture", lambda *args, **kwargs: 0.0)
    return X2PlacePlanner(SimpleNamespace(arms={"right": arm}), {"right": robot})


def run_place(monkeypatch, tmp_path, service, *, record=True, targets=three_targets):
    if record:
        monkeypatch.setenv(place_request.PLACE_INPUTS_DIR_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(place_request.PLACE_INPUTS_DIR_ENV, raising=False)
    saved = []
    original = place_request.save_place_inputs
    monkeypatch.setattr(place_request, "save_place_inputs", lambda *args: saved.append(original(*args)))
    feedback = Feedback(49.98, {name: 0.0 for name in JOINTS}, np.zeros(10))
    admitted = SimpleNamespace(live=SimpleNamespace(feedback=feedback), admitted_at=50.0)
    planner = place_planner(monkeypatch, service, feedback, targets)
    try:
        return planner.plan(
            SimpleNamespace(robot_state=admitted),
            Container(scene(49.9)),
            Held(),
            deadline=float("inf"),
            cancelled=lambda: False,
        )
    finally:
        for thread in saved:
            thread.join(timeout=10)


def place_lines(capsys):
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith('{"place_plan"')]
    assert len(lines) == 1
    return lines[0]["place_plan"]


def test_a_place_plan_logs_where_its_targets_and_its_recheck_spend_their_time(monkeypatch, tmp_path, capsys):
    plan = run_place(monkeypatch, tmp_path, PlanningService(), record=False)
    timings = place_lines(capsys)["timings_s"]
    assert set(timings["targets_parts"]) == {"spheres", "place_targets", "placement_cover", "opening_cover"}
    recheck = timings["recheck_parts"]
    assert set(recheck) == {"capture", "camera", "checks"}
    # The camera's own profile of this recapture: its GPU wait is what varies live.
    assert recheck["camera"] == {"wait_s": 0.01, "gpu_wait_s": 0.3, "registration_s": 0.05}
    assert plan.report["plan_timings_s"]["recheck_parts"] == recheck
    assert not list(tmp_path.iterdir())  # nothing is recorded without the directory


def test_a_recorded_place_keeps_its_inputs_and_live_outcome(monkeypatch, tmp_path, capsys):
    run_place(monkeypatch, tmp_path, PlanningService())
    (saved,) = recorded_places([tmp_path])
    recorded = pickle.loads(saved.read_bytes())
    assert recorded["kind"] == "place" and recorded["arm_side"] == "right"
    assert (
        len(recorded["targets"]) == 3 and recorded["attached"].shape == (2, 4) and recorded["opening"].shape == (5, 4)
    )
    assert recorded["admitted_at"] == 50.0 and recorded["place_config"] == "config"
    assert recorded["outcome"]["outcome"] == "ok" and recorded["outcome"]["planner"]["candidate"] == 1
    assert recorded["outcome"]["timings_s"] == place_lines(capsys)["timings_s"]


def test_a_failed_place_is_recorded_too(monkeypatch, tmp_path):
    from aimdk_grasp.planner import PlacementSearchExhausted

    with pytest.raises(place_request.GraspFailure):
        run_place(monkeypatch, tmp_path, PlanningService(fail=PlacementSearchExhausted("no complete path")))
    (saved,) = recorded_places([tmp_path])
    recorded = pickle.loads(saved.read_bytes())
    assert recorded["outcome"]["outcome"] == "PlacementSearchExhausted"
    assert "recheck" not in recorded["outcome"]["timings_s"]


def test_every_clock_of_a_place_moves_together(monkeypatch, tmp_path):
    run_place(monkeypatch, tmp_path, PlanningService())
    recorded = pickle.loads(recorded_places([tmp_path])[0].read_bytes())
    feedback, container, now = shifted(recorded, 1000.0)
    assert (feedback.received_at, container.scene.received_at, now) == pytest.approx((1049.98, 1049.9, 1050.0))
    assert recorded["container"].scene.received_at == 49.9  # the recording is untouched


def test_a_replay_builds_each_arm_once_and_plans_on_increasing_clocks(monkeypatch, tmp_path):
    run_place(monkeypatch, tmp_path, PlanningService())
    files = recorded_places([tmp_path])
    service = PlanningService()
    rows = replay(files, lambda side: (service, None), repeats=2)
    assert [call[0] for call in service.calls] == ["prepare", "session", "plan_place", "session", "plan_place"]
    assert service.calls[0] == ("prepare", True)  # built carrying, as the live session was
    planned = [call for call in service.calls if call[0] == "plan_place"]
    assert planned[0][2] < planned[1][2] and all(call[3:5] == (3, 5) for call in planned)
    assert [row["rebuilt"] for row in rows] == [True, False]
    assert all(row["success"] and row["candidate"] == row["live_candidate"] == 1 for row in rows)
    assert rows[0]["motion_s"] == [2.0, 0.5, 0.4, 1.0]
    assert summary(rows) | {"median_wall_s": None, "max_wall_s": None} == dict(
        skipped_refusals=0,
        calls=2,
        planned=2,
        live_planned=2,
        same_candidate=2,
        median_wall_s=None,
        max_wall_s=None,
        median_screen_s=0.25,
    )


def test_a_replayed_failure_is_a_row_not_an_abort(monkeypatch, tmp_path):
    run_place(monkeypatch, tmp_path, PlanningService())
    service = PlanningService(fail=RuntimeError("descent: ik"))
    (row,) = replay(recorded_places([tmp_path]), lambda side: (service, None))
    assert not row["success"] and row["error"] == "RuntimeError: descent: ik" and row["rounds"] == 1


def test_a_container_refused_before_planning_is_recorded_with_what_refused_it(monkeypatch, tmp_path):
    def refuse(*args):
        raise place_request.GraspFailure("CONTAINER_TOO_SMALL", "no release position keeps the held object")

    service = PlanningService()
    with pytest.raises(place_request.GraspFailure):
        run_place(monkeypatch, tmp_path, service, targets=refuse)
    assert not service.calls  # nothing was planned
    (saved,) = recorded_places([tmp_path])
    recorded = pickle.loads(saved.read_bytes())
    # place_targets' inputs are all kept (container, held, palm, config); its outputs do not exist.
    assert recorded["outcome"]["outcome"] == "CONTAINER_TOO_SMALL"
    assert recorded["palm"].shape == (4, 4) and recorded["place_config"] == "config" and "targets" not in recorded
    (row,) = replay([saved], lambda side: (PlanningService(), None))
    assert row["skipped"] and row["error"] == "live refusal: CONTAINER_TOO_SMALL"
    assert summary([row])["skipped_refusals"] == 1 and summary([row])["calls"] == 0


def test_a_place_without_a_home_ends_at_its_retreat(monkeypatch, tmp_path):
    class Service(PlanningService):
        def plan_place(self, feedback, targets, home, now, *, check_deadline, deadline, opening, return_path):
            assert home is None
            self.place_report = {"candidate": 0, "rounds": 1, "timings_s": {}}
            return path(2.0), path(0.5), path(0.4)

    feedback = Feedback(49.98, {name: 0.0 for name in JOINTS}, np.zeros(10))
    admitted = SimpleNamespace(live=SimpleNamespace(feedback=feedback), admitted_at=50.0)
    planner = place_planner(monkeypatch, Service(), feedback)
    planner.robots["right"].place = SimpleNamespace(plan_return_home=False)
    monkeypatch.delenv(place_request.PLACE_INPUTS_DIR_ENV, raising=False)
    plan = planner.plan(
        SimpleNamespace(robot_state=admitted),
        Container(scene(49.9)),
        Held(),
        deadline=float("inf"),
        cancelled=lambda: False,
    )
    assert [step.name for step in plan.retreat] == ["retreat"]
