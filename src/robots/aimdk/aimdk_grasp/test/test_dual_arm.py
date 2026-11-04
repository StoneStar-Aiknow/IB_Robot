"""Automatic arm selection, fallback budgets and execution routing without a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.dual_arm import (
    ArmScenes,
    ArmWatch,
    AutomaticPlanner,
    ExecutionSelection,
    SelectedPart,
    SelectingPlanner,
    target_arm_order,
)
from aimdk_grasp.joints import arm_joints

from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Move,
    Proposals,
    SegmentKind,
)
from manipulation_execution.grasp_pipeline.pipeline import GraspPipeline, PickMode


@pytest.mark.parametrize(
    "y,expected", [(0.2, ("left",)), (-0.2, ("right",)), (0.01, ("left", "right")), (-0.01, ("right", "left"))]
)
def test_target_geometry_uses_torso_y_instead_of_camera_y(y, expected):
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert target_arm_order([[y, 1.0, 0.5]], transform) == expected
    assert target_arm_order([[y, 1.0, 0.5]], transform, "left") == ("left",)
    assert target_arm_order([[y, 1.0, 0.5]], transform, "right") == ("right",)


@pytest.mark.parametrize("points", [[], [[np.nan, 0, 0]], [[1, 2]]])
def test_unknown_target_geometry_does_not_choose_an_arm(points):
    with pytest.raises(ValueError, match="target points"):
        target_arm_order(points, np.eye(4))


def plan(side):
    segment = ArmSegment("approach", SegmentKind.TARGET, arm_joints(side), np.zeros((2, 7)))
    return GraspPlan((Move(segment), Effect(EffectorAction.CLOSE)), 0, report={"arm_side": side})


@pytest.mark.parametrize("first_fails", [False, True])
def test_midline_returns_the_first_valid_plan_and_reserves_budget_for_the_other_arm(monkeypatch, first_fails):
    monkeypatch.setattr("aimdk_grasp.dual_arm.time.monotonic", lambda: 101.0)
    calls = []

    def attempt(side):
        def run(observation, proposals, *, deadline, cancelled):
            calls.append((side, deadline, proposals.scene))
            if side == "left" and first_fails:
                raise GraspFailure("NO_EXECUTABLE_CANDIDATE", "no IK", retryable=True, reason_code="NO_COMPLETE_PATH")
            return plan(side)

        return SimpleNamespace(plan=run)

    planner = AutomaticPlanner({side: attempt(side) for side in ("left", "right")})
    observation = SimpleNamespace(robot_state=SimpleNamespace(started=100.0))
    scenes = ArmScenes({"left": "left scene", "right": "right scene"})
    proposals = Proposals(np.ones((1, 1), bool), np.zeros((1, 3)), np.eye(4)[None], np.ones(1), scene=scenes)
    result = planner.plan(observation, proposals, deadline=180.0, cancelled=lambda: False)
    assert result.report["arm_side"] == scenes.selected_side == ("right" if first_fails else "left")
    assert calls[0] == ("left", 105.5, "left scene")
    assert calls[1:] == ([("right", 110.0, "right scene")] if first_fails else [])
    if first_fails:
        assert result.report["arm_selection"]["failures"]["left"]["reason_code"] == "NO_COMPLETE_PATH"


@pytest.mark.parametrize(
    "failures, expected_reason, retryable",
    [
        ({"left": ("OUT_OF_REACH", False)}, "OUT_OF_REACH", False),
        ({"right": ("OUT_OF_REACH", False)}, "OUT_OF_REACH", False),
        ({"right": ("NO_COMPLETE_PATH", True)}, "NO_COMPLETE_PATH", True),
        ({"left": ("PLAN_TIMEOUT", True)}, "PLAN_TIMEOUT", True),
        ({"right": ("PLANNER_FAULT", False)}, "PLANNER_FAULT", False),
        ({"left": ("OUT_OF_REACH", False), "right": ("OUT_OF_REACH", False)}, "OUT_OF_REACH", False),
        ({"left": ("OUT_OF_REACH", False), "right": ("NO_COMPLETE_PATH", True)}, "NO_EXECUTABLE_CANDIDATE", True),
    ],
)
def test_selected_arm_failures_preserve_specific_or_mixed_reasons(monkeypatch, failures, expected_reason, retryable):
    monkeypatch.setattr("aimdk_grasp.dual_arm.time.monotonic", lambda: 101.0)

    def attempt(observation, proposals, **budget):
        reason, can_retry = failures[proposals.scene]
        raise GraspFailure("NO_EXECUTABLE_CANDIDATE", "no plan", retryable=can_retry, reason_code=reason)

    planner = AutomaticPlanner({side: SimpleNamespace(plan=attempt) for side in failures})
    observation = SimpleNamespace(robot_state=SimpleNamespace(started=100.0))
    scenes = ArmScenes({side: side for side in failures})
    proposals = Proposals(np.ones((1, 1), bool), np.zeros((1, 3)), np.eye(4)[None], np.ones(1), scene=scenes)
    with pytest.raises(GraspFailure) as error:
        planner.plan(observation, proposals, deadline=180.0, cancelled=lambda: False)
    assert error.value.code == "NO_EXECUTABLE_CANDIDATE"
    assert error.value.reason_code == expected_reason
    assert error.value.retryable is retryable
    assert scenes.selected_side is None


def test_cancel_never_falls_back_to_the_other_arm():
    def canceled(*args, **budget):
        raise GraspFailure("CANCELED", "cancelled while planning")

    planner = AutomaticPlanner({"left": SimpleNamespace(plan=canceled)})
    observation = SimpleNamespace(robot_state=SimpleNamespace(started=0.0))
    proposals = SimpleNamespace(scene=ArmScenes({"left": object(), "right": object()}))
    with pytest.raises(GraspFailure) as error:
        planner.plan(observation, proposals, deadline=0.0, cancelled=lambda: True)
    assert error.value.code == "CANCELED"


def test_spent_observation_budget_requests_a_fresh_attempt(monkeypatch):
    monkeypatch.setattr("aimdk_grasp.dual_arm.time.monotonic", lambda: 111.0)
    planner = AutomaticPlanner({})
    observation = SimpleNamespace(robot_state=SimpleNamespace(started=100.0))
    proposals = SimpleNamespace(scene=ArmScenes({"left": object()}))
    with pytest.raises(GraspFailure) as error:
        planner.plan(observation, proposals, deadline=180.0, cancelled=lambda: False)
    assert error.value.code == "NO_EXECUTABLE_CANDIDATE"
    assert error.value.retryable
    assert error.value.reason_code == "PLAN_TIMEOUT"


@pytest.mark.parametrize("capture_fails", [False, True])
def test_requests_that_never_select_an_arm_keep_their_outcome_during_cleanup(capture_fails):
    def capture(**budget):
        if capture_fails:
            raise GraspFailure("OBSERVATION_UNAVAILABLE", "no camera frame")
        return SimpleNamespace(report={})

    selection = ExecutionSelection()
    part = SelectedPart({}, selection)
    backends = GraspBackends(SimpleNamespace(capture=capture), None, None, part, part)
    result = GraspPipeline(backends, clock=lambda: 0).run(
        "apple", mode=PickMode.OBSERVE_ONLY, deadline=10.0, cancelled=lambda: False
    )
    assert result.success is (not capture_fails)
    assert result.error_code == ("OBSERVATION_UNAVAILABLE" if capture_fails else "")
    assert "finish_errors" not in result.report


def test_remote_watch_follows_the_planned_arm_and_disarms_the_previous_arm():
    calls = []

    def watch(side):
        return SimpleNamespace(
            arm=lambda *args: calls.append((side, "arm")),
            disarm=lambda: calls.append((side, "disarm")),
            status=lambda: side,
        )

    watcher = ArmWatch({side: watch(side) for side in ("left", "right")})
    scenes = ArmScenes({"left": object(), "right": object()}, selected_side="right")
    proposals = SimpleNamespace(scene=scenes)
    watcher.arm(None, proposals, None)  # The remote ARM request carries no plan.
    assert watcher.status() == "right"
    watcher.arm(None, proposals, plan("left"))
    assert watcher.status() == "left"
    assert calls == [("left", "disarm"), ("right", "arm"), ("right", "disarm"), ("left", "arm")]


@pytest.mark.parametrize("running", [True, False])
def test_switching_arms_keeps_the_shared_imu_as_the_plan_left_it(running):
    class Imu:
        paused = not running

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

    imu = Imu()

    def watch(side):
        # As X2TargetWatch: both arms hold the one ChestImu and disarm pauses it.
        return SimpleNamespace(imu=imu, arm=lambda *args: None, disarm=imu.pause)

    watcher = ArmWatch({side: watch(side) for side in ("left", "right")})
    watcher.arm(None, None, plan("right"))
    assert imu.paused is not running
    watcher.arm(None, None, plan("right"))  # same arm: no disarm, no change
    assert imu.paused is not running


def test_executor_and_hand_follow_each_accepted_plan_and_refuse_mismatched_joints():
    selection = ExecutionSelection()
    finished = []
    chosen = ["right"]
    planner = SelectingPlanner(SimpleNamespace(plan=lambda: plan(chosen[0])), selection)
    executor = SelectedPart(
        {
            side: SimpleNamespace(hold=lambda side=side: side, finish=lambda side=side: finished.append((side, "arm")))
            for side in ("left", "right")
        },
        selection,
    )
    effector = SelectedPart(
        {
            side: SimpleNamespace(
                close=lambda side=side: side, finish=lambda side=side: finished.append((side, "hand"))
            )
            for side in ("left", "right")
        },
        selection,
    )
    with pytest.raises(GraspFailure, match="No arm"):
        executor.hold()
    for side in ("right", "left"):
        chosen[0] = side
        planner.plan()
        assert executor.hold() == effector.close() == side
    assert finished == [("right", "arm"), ("right", "hand")]
    wrong = GraspPlan(plan("right").steps, 0, report={"arm_side": "left"})
    with pytest.raises(GraspFailure, match="trajectory joints disagree"):
        selection.accept(wrong)
    assert selection.side == "left"
