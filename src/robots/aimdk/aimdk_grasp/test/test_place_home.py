"""The empty hand's way home after a drop: a fresh capture, the arm's grasp planner."""

import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import place_request
from aimdk_grasp.joints import arm_joints
from aimdk_grasp.place_request import X2PlacePlanner
from aimdk_grasp.scene_estimation import NoSupportError

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, HeldObject, Move

SIDE = "right"
JOINTS = arm_joints(SIDE)
HERE, HOME = np.full(7, 0.3), np.zeros(7)


class Service:
    def __init__(self, failure=None):
        self.failure, self.calls = failure, []

    @contextmanager
    def session(self, feedback, scene):
        self.calls.append(("session", feedback.name, scene))
        yield self

    def plan_home(self, feedback, home, *, check_deadline, deadline, return_path=None):
        self.calls.append(("plan_home", tuple(home)))
        self.return_path = return_path
        if self.failure:
            raise self.failure
        return SimpleNamespace(positions=np.linspace(feedback.vector(JOINTS), home, 5), times=np.arange(5) * 0.1)


@pytest.fixture
def scenes(monkeypatch):
    built = []

    def prepare(observation, spheres, settings):
        built.append(("prepare", observation.stamp_ns, spheres, settings))
        return "prepared"

    def complete(prepared, mask, *, lazy_validation):
        built.append(("complete", prepared, mask, lazy_validation))
        if built[0][1] == 0:
            raise NoSupportError("no table in view")
        return "fresh scene"

    monkeypatch.setattr(place_request, "prepare_capture_scene", prepare)
    monkeypatch.setattr(place_request, "complete_scene", complete)
    return built


def planner(service, *, stamp_ns=7):
    feedback = SimpleNamespace(name="live", vector=lambda names: HERE.copy())
    observation = SimpleNamespace(
        stamp_ns=stamp_ns, robot_state=SimpleNamespace(live=SimpleNamespace(feedback=feedback))
    )
    captures, paused = [], []

    def capture(*, deadline, cancelled):
        captures.append(deadline)
        return observation

    arm = SimpleNamespace(
        camera=SimpleNamespace(capture=capture), spheres="right spheres", service=service, gpu=threading.RLock()
    )
    head = SimpleNamespace(arms={SIDE: arm}, imu=SimpleNamespace(pause_after=paused.append))
    robot = SimpleNamespace(
        perception=SimpleNamespace(scene="scene settings"),
        limits={name: (-1.0, 1.0) for name in JOINTS},
        gravity=SimpleNamespace(joints=JOINTS, sags=lambda positions: np.zeros(len(positions))),
    )
    return X2PlacePlanner(head, {SIDE: robot}), captures, paused


def held():
    return HeldObject("c1", "an apple", SIDE, JOINTS, tuple(HERE), tuple(HOME), 0.04, (0.0,) * 3, "{}")


def plan_home(place_planner):
    return place_planner.plan_home(held(), deadline=float("inf"), cancelled=lambda: False)


def test_home_is_planned_on_a_fresh_capture_with_the_holding_arms_grasp_planner(scenes):
    service = Service()
    place_planner, captures, paused = planner(service)
    move = plan_home(place_planner)
    assert isinstance(move, Move) and move.segment.name == "home" and move.segment.joint_names == JOINTS
    np.testing.assert_allclose(move.segment.start, HERE)
    np.testing.assert_allclose(move.segment.end, HOME)
    assert len(captures) == 1 and paused == [7]
    # Whole-scene collision world: no target mask, nothing left out of it.
    assert scenes == [("prepare", 7, "right spheres", "scene settings"), ("complete", "prepared", None, False)]
    assert service.calls == [("session", "live", "fresh scene"), ("plan_home", tuple(HOME))]
    assert service.return_path is None  # this custody recorded no way back


def test_the_way_pick_came_is_offered_to_the_home_planner(scenes):
    service = Service()
    place_planner, _, _ = planner(service)
    way_back = np.stack([HERE, (np.asarray(HERE) + HOME) / 2, HOME])
    custody = replace(held(), return_positions=tuple(way_back.ravel()), return_times=(0.0, 0.5, 1.0))
    place_planner.plan_home(custody, deadline=float("inf"), cancelled=lambda: False)
    np.testing.assert_allclose(service.return_path.positions, way_back)
    np.testing.assert_allclose(service.return_path.times, [0.0, 0.5, 1.0])
    assert service.return_path.joint_names == JOINTS


@pytest.mark.parametrize("case", ["planner fails", "no support", "beyond limits"])
def test_no_trusted_way_home_is_a_home_plan_failure(scenes, case):
    service = Service(failure=RuntimeError("start state in collision") if case == "planner fails" else None)
    place_planner, _, _ = planner(service, stamp_ns=0 if case == "no support" else 7)
    if case == "beyond limits":
        place_planner.robots[SIDE].limits = {name: (0.1, 1.0) for name in JOINTS}
    with pytest.raises(GraspFailure) as raised:
        plan_home(place_planner)
    assert raised.value.code == "HOME_PLAN_FAILED"


def test_a_cancelled_goal_plans_nothing(scenes):
    service = Service()
    place_planner, captures, _ = planner(service)
    with pytest.raises(GraspFailure) as raised:
        place_planner.plan_home(held(), deadline=float("inf"), cancelled=lambda: True)
    assert raised.value.code == "CANCELED" and not captures and not service.calls
