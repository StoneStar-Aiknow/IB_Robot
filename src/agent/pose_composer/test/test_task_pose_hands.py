"""Composition and shared task budget over a Mock hand backend.

aimdk_grasp tests exercise the real restorer separately, including its model
coverage refusal; this fixture represents a backend with both hands available.
"""

import pytest
from aimdk_grasp.hand_restore import RestoreError
from test_task_poses import MockArm, snapshot

from pose_composer.composer import ComposeError
from pose_composer.task_poses import TaskPoseRunner
from robot_runtime.task_poses import normalize_task_poses


def hand_snapshot():
    config = snapshot()
    config.pop("digest")
    config["poses"]["grasp_ready"].update(hands={"right": "open", "left": "open"}, timeout_sec=15.0)
    return normalize_task_poses(config)


class MockHands:
    def __init__(self, arm, events, *, advance=None, fault=""):
        self.arm, self.events, self.fault = arm, events, fault
        self.advance = advance or arm.sleep
        self.opened = []

    def stage(self, ctx, name, seconds):
        assert ctx.task_id == "mock_task"
        self.events.append(name)
        ctx.progress(name, 0.0)
        if ctx.cancelled():
            raise RestoreError("POSE_CANCELLED", "mock canceled")
        if self.fault == name:
            raise RestoreError("HAND_NOT_SETTLED", f"mock {name} failure")
        if self.fault == "cancel" and name == "opening_right":
            self.arm.cancelled = True
            raise RestoreError("POSE_CANCELLED", "mock canceled after left")
        self.advance(seconds)
        if self.arm.now >= ctx.deadline:
            raise RestoreError("TASK_POSE_TIMEOUT", "mock hand deadline")

    def preflight(self, ctx, sides, *, arm_target, arm_must_move):
        assert sides == ("left", "right")
        self.stage(ctx, "preflight", 0.0)
        return 2.5

    def begin(self, ctx):
        self.stage(ctx, "takeover", 0.2)

    def open_hand(self, ctx, side):
        assert self.arm.q == pytest.approx(snapshot()["poses"]["grasp_ready"]["joint_positions"])
        self.stage(ctx, f"opening_{side}", 0.2)
        self.opened.append(side)
        return False

    def confirm(self, ctx, sides, arm_check):
        assert self.opened == list(sides)
        self.stage(ctx, "confirming", 2.0)
        arm_check()


def execute(arm, hands, config=None, *, simulation=True, **kwargs):
    config = config or hand_snapshot()
    runner = TaskPoseRunner(config, simulation=simulation, clock=lambda: arm.now, pause=arm.sleep)
    result = runner.execute(
        "grasp_ready",
        config["digest"],
        read_state=arm.read,
        move=arm.move,
        cancelled=lambda: arm.cancelled,
        hands=lambda: hands,
        gateway_task_id="mock_task",
        **kwargs,
    )
    return result, runner.report


@pytest.mark.parametrize("away", [False, True])
def test_combined_preparation_and_phase_timings(away):
    arm, events = MockArm(away=away), []
    hands = MockHands(arm, events)
    skipped, report = execute(arm, hands)
    assert skipped is (not away) and len(arm.moves) == int(away)
    assert events[:2] == ["preflight", "takeover"]
    assert events[-3:] == ["opening_left", "opening_right", "confirming"]
    assert report["completed_hands"] == ["left", "right"]
    assert report["phase_sec"]["confirming"] == pytest.approx(2.0)
    assert report["phase_sec"]["arm_settling"] == 0.0
    assert not report["arrival_check"]["stable_window_verified"]
    assert report["total_sec"] == pytest.approx(2.6)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("opening_right", "HAND_NOT_SETTLED"),
        ("cancel", "POSE_CANCELLED"),
    ],
)
def test_partial_completion_survives_failure_without_rollback(fault, code):
    arm, config, events = MockArm(), hand_snapshot(), []
    hands = MockHands(arm, events, fault=fault)
    runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    with pytest.raises(ComposeError) as error:
        runner.execute(
            "grasp_ready",
            config["digest"],
            read_state=arm.read,
            move=arm.move,
            cancelled=lambda: arm.cancelled,
            hands=lambda: hands,
            gateway_task_id="mock_task",
        )
    assert error.value.code == code and hands.opened == ["left"]
    assert runner.report["completed_hands"] == ["left"]
    assert "confirming" not in events and "total_sec" in runner.report


def test_arm_only_pose_never_constructs_hand_backend():
    arm = MockArm()

    def forbidden():
        pytest.fail("hand backend was requested for an arm-only pose")

    config = snapshot()
    runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    assert runner.execute(
        "grasp_ready", config["digest"], read_state=arm.read, move=arm.move, cancelled=lambda: False, hands=forbidden
    )


def test_unloaded_approval_is_required_separately_from_arm_approval():
    arm, config, events = MockArm(), hand_snapshot(), []
    config.pop("digest")
    config["poses"]["grasp_ready"]["hardware_approved"] = True
    config = normalize_task_poses(config)
    with pytest.raises(ComposeError) as error:
        execute(arm, MockHands(arm, events), config, simulation=False)
    assert error.value.code == "TASK_POSE_HANDS_NOT_APPROVED" and not events and not arm.moves


def test_insufficient_parent_budget_is_refused_before_takeover():
    arm, events = MockArm(away=True), []
    with pytest.raises(ComposeError) as error:
        execute(arm, MockHands(arm, events), timeout_sec=1.0)
    assert error.value.code == "TASK_POSE_TIMEOUT"
    assert events == ["preflight"] and not arm.moves


def test_missing_hand_backend_cannot_move_arms():
    arm = MockArm(away=True)
    with pytest.raises(ComposeError) as error:
        execute(arm, None)
    assert error.value.code == "HAND_CAPABILITY_MISSING" and not arm.moves


@pytest.mark.parametrize("cancel", [False, True])
def test_trajectory_discovery_cannot_dispatch_after_budget_or_cancel(monkeypatch, cancel):
    from types import SimpleNamespace

    from pose_composer import node as node_module

    clock = [10.0]
    goal = SimpleNamespace(is_cancel_requested=False)

    def wait_for_server(timeout_sec):
        assert timeout_sec <= 0.5
        clock[0] += 0.5
        goal.is_cancel_requested = cancel
        return True

    node = object.__new__(node_module.PoseComposerNode)
    node._traj = SimpleNamespace(
        wait_for_server=wait_for_server, send_goal_async=lambda _: pytest.fail("late trajectory dispatched")
    )
    monkeypatch.setattr(node_module.time, "monotonic", lambda: clock[0])
    ok, code, _message = node._run_trajectory(goal, None, deadline=10.5)
    assert not ok and code == ("POSE_CANCELLED" if cancel else "TASK_POSE_TIMEOUT")
