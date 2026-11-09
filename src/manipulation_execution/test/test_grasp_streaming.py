"""The streaming executor against a simulated position-controlled arm that sags under load."""

from __future__ import annotations

import numpy as np
import pytest

from manipulation_execution.pick.pipeline.contracts import ArmSegment, MotionState, SegmentKind
from manipulation_execution.pick.pipeline.sequence import GraspSequence
from manipulation_execution.pick.pipeline.streaming import (
    CommandSample,
    JointSample,
    StreamConfig,
    StreamExecutor,
)

DRIVEN = ("l0", "l1", "l2")
CHANNEL = DRIVEN + ("r0", "r1")
SAG = np.array([0.05, 0.03, 0.02, 0.04, 0.01])  # the arm settles this far below its target


class SimArm:
    """Controller target = last command; the measured arm lags toward target - SAG."""

    channel_joints = CHANNEL

    def __init__(self, start=(0.0, 0.0, 0.0, 0.5, 0.5), *, report=True, gains=(40.0, 2.0)):
        self.t = 100.0
        self.target = np.array(start, dtype=float) + SAG  # the controller's own target at takeover
        self.measured = np.array(start, dtype=float)
        self.head = 0.0
        self.report_enabled, self.gains = report, gains
        self.published, self.released, self.refused = [], 0, None
        self.stuck = False  # the arm stops following
        self.clamp = None  # (joint index, value): the controller replaces that target
        self.feedback_delay = 0.0
        self.report_delay = 0.0
        self.report_lag = 0.0  # a fresh report carrying the target from this long ago
        self._targets: list[tuple[float, np.ndarray]] = []
        self._initial_target = self.target.copy()

    def monotonic(self):
        return self.t

    def wait(self, seconds):
        steps = max(1, int(round(seconds / 0.005)))
        for _ in range(steps):
            self.t += seconds / steps
            if not self.stuck:
                self.measured += (self.target - SAG - self.measured) * min(1.0, (seconds / steps) / 0.05)

    def feedback(self):
        positions = dict(zip(CHANNEL, self.measured, strict=True))
        positions["head"] = self.head
        return JointSample(self.t - self.feedback_delay, positions)

    def command(self):
        if not self.report_enabled:
            return None
        stiffness, damping = self.gains
        target = self.target
        if self.report_lag and self._targets:
            older = [value for at, value in self._targets if at <= self.t - self.report_lag]
            target = older[-1] if older else self._initial_target
        return CommandSample(
            self.t - self.report_delay,
            dict(zip(CHANNEL, target, strict=True)),
            dict.fromkeys(CHANNEL, stiffness),
            dict.fromkeys(CHANNEL, damping),
        )

    def publish(self, positions):
        self.published.append(np.array(positions))
        self.target = np.array(positions, dtype=float)
        if self.clamp is not None:
            self.target[self.clamp[0]] = self.clamp[1]
        self._targets.append((self.t, self.target.copy()))

    def release(self):
        self.released += 1

    def refusal(self):
        return self.refused


def line(start, end, seconds=1.0, frames=26):
    s = np.linspace(0.0, 1.0, frames)[:, None]
    return ArmSegment(
        "approach",
        SegmentKind.TRAJECTORY,
        DRIVEN,
        np.asarray(start) + s * (np.asarray(end) - start),
        np.linspace(0.0, seconds, frames),
    )


CONFIG = StreamConfig(
    tracking_error_rad=0.3,
    arrival_error_rad=0.2,
    expected_stiffness=40.0,
    expected_damping=2.0,
    locked_joints=("head", "r0", "r1"),
)


def run(executor, segment, halted=lambda: False, cancelled=lambda: False):
    return executor.execute(segment, carrying=False, halted=halted, cancelled=cancelled)


def test_takeover_starts_from_the_controllers_target_and_carries_its_offset():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    takeover = arm.target.copy()
    outcome = run(executor, line(np.zeros(3), [0.3, -0.2, 0.1]))
    assert outcome.state is MotionState.SUCCEEDED, outcome.reason
    # The first command is the controller's own target: the arm does not drop.
    np.testing.assert_allclose(arm.published[0], takeover)
    # Every command leads the driven joints by the controller's offset; the others are held.
    np.testing.assert_allclose(arm.published[-1][:3], [0.3, -0.2, 0.1] + SAG[:3], atol=1e-9)
    np.testing.assert_allclose(arm.published[-1][3:], takeover[3:])
    np.testing.assert_allclose(outcome.stopped_at, [0.3, -0.2, 0.1], atol=0.01)


def test_without_a_report_the_measured_pose_is_the_baseline():
    arm = SimArm(report=False)
    run(
        StreamExecutor(arm, StreamConfig(tracking_error_rad=0.3, arrival_error_rad=0.2)),
        line(np.zeros(3), [0.1, 0.0, 0.0]),
    )
    np.testing.assert_allclose(arm.published[0], [0.0, 0.0, 0.0, 0.5, 0.5])


def test_halt_stops_between_commands_with_the_last_command_standing():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    count = [0]

    def halted():
        count[0] += 1
        return count[0] > 10

    outcome = run(executor, line(np.zeros(3), [0.3, 0.0, 0.0]), halted=halted)
    assert outcome.state is MotionState.HALTED and len(arm.published) == 10 and not arm.released
    # The executor still owns the stream: a retreat from where it stopped runs.
    back = ArmSegment(
        "retreat", SegmentKind.TRAJECTORY, DRIVEN, np.stack([outcome.stopped_at, np.zeros(3)]), np.array([0.0, 0.5])
    )
    assert run(executor, back).state is MotionState.SUCCEEDED


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (lambda arm: setattr(arm, "clamp", (2, 0.0)), "not driving the targets"),
        (lambda arm: setattr(arm, "gains", (20.0, 2.0)), "stiffness is not 40"),
        (lambda arm: setattr(arm, "report_delay", 0.5), "report is stale"),
        (lambda arm: setattr(arm, "feedback_delay", 0.5), "feedback unhealthy or stale"),
        (lambda arm: setattr(arm, "stuck", True), "tracking anomaly"),
        (lambda arm: setattr(arm, "refused", "stop latched"), "does not admit output: stop latched"),
    ],
)
def test_a_platform_that_does_not_follow_stops_the_segment(setup, reason):
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    setup(arm)
    outcome = run(executor, line(np.zeros(3), [0.6, 0.0, 0.0], seconds=1.5))
    assert outcome.state is MotionState.FAILED and reason in outcome.reason


def test_a_report_that_does_not_follow_says_which_joint_and_how_stale_it_is():
    arm = SimArm()
    arm.clamp = (2, 0.0)
    outcome = run(StreamExecutor(arm, CONFIG), line(np.zeros(3), [0.0, 0.0, 0.3], seconds=1.5))
    assert "l2 driven to 0.0000" in outcome.reason and "nothing sent in the last 2 s matches it" in outcome.reason
    arm = SimArm()
    arm.report_lag = 0.3  # fresh to the executor, but it carries a target from 0.3 s back
    outcome = run(StreamExecutor(arm, CONFIG), line(np.zeros(3), [0.6, 0.0, 0.0], seconds=1.5))
    assert outcome.state is MotionState.FAILED, outcome.reason
    sent = float(outcome.reason.split("it is what was sent ")[1].split(" s ago")[0])
    assert CONFIG.command_apply_s < sent <= 0.3  # the newest target it equals, older than the match window


def test_a_stop_latches_for_its_goal_only():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    executor.stop("tracking anomaly")
    outcome = run(executor, line(np.zeros(3), [0.3, 0.0, 0.0]))
    assert outcome.state is MotionState.FAILED and "executor stopped: tracking anomaly" in outcome.reason
    executor.finish()  # the goal ends; the next one takes over afresh
    assert run(executor, line(np.zeros(3), [0.3, 0.0, 0.0])).state is MotionState.SUCCEEDED


def test_a_hold_and_an_effector_action_are_supervised_like_a_segment():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    assert run(executor, line(np.zeros(3), [0.3, 0.0, 0.0])).state is MotionState.SUCCEEDED
    assert executor.hold(0.2, carrying=True, cancelled=lambda: False).state is MotionState.SUCCEEDED
    check = executor.supervise()
    assert check() == ""
    arm.stuck = True
    arm.measured[0] -= 0.4  # the object drags the arm down
    assert "tracking anomaly" in check()
    outcome = executor.hold(0.2, carrying=True, cancelled=lambda: False)
    assert outcome.state is MotionState.FAILED and "tracking anomaly while holding" in outcome.reason
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    assert run(executor, line(np.zeros(3), [0.3, 0.0, 0.0])).state is MotionState.SUCCEEDED
    check = executor.supervise()
    arm.head = 0.1
    assert "locked" in check()
    arm.head, arm.clamp = 0.0, (1, 0.5)
    arm.publish(arm.published[-1])  # the controller swaps its target
    assert "not driving the targets" in check()
    arm.feedback_delay = 0.5  # supervision that cannot start says why, every time it is asked
    late = StreamExecutor(arm, CONFIG).supervise()
    assert "stale" in late() and "stale" in late()


def test_a_moved_locked_joint_or_a_distant_start_is_refused():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)

    def nudge():
        arm.head = 0.1
        return False

    outcome = run(executor, line(np.zeros(3), [0.3, 0.0, 0.0]), cancelled=nudge)
    assert outcome.state is MotionState.FAILED and "locked" in outcome.reason
    arm = SimArm()
    outcome = run(StreamExecutor(arm, CONFIG), line([0.25, 0.0, 0.0], [0.3, 0.0, 0.0]))
    assert outcome.state is MotionState.FAILED and "replan" in outcome.reason and not arm.published


def test_cancel_returns_cancelled_and_stop_releases_for_good():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    assert run(executor, line(np.zeros(3), [0.3, 0.0, 0.0]), cancelled=lambda: True).state is MotionState.CANCELLED
    executor.stop("operator")
    assert arm.released == 1
    assert "stopped: operator" in run(executor, line(np.zeros(3), [0.1, 0.0, 0.0])).reason


def test_a_target_segment_is_a_timed_joint_line_from_where_the_arm_is():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    target = ArmSegment("home", SegmentKind.TARGET, DRIVEN, np.stack([np.ones(3), [0.08, 0.0, 0.0]]))
    started = arm.t
    assert run(executor, target).state is MotionState.SUCCEEDED
    # 0.08 rad at 0.05 rad/s: 1.6 s, plus settling.
    assert 1.6 <= arm.t - started <= 1.6 + 0.6


def test_a_target_segment_never_exceeds_the_global_velocity_limit():
    arm = SimArm()
    config = StreamConfig(
        tracking_error_rad=0.3,
        arrival_error_rad=0.2,
        max_velocity_rad_s=0.01,
        target_speed_rad_s=0.05,
    )
    executor = StreamExecutor(arm, config)
    target = ArmSegment("home", SegmentKind.TARGET, DRIVEN, np.stack([np.ones(3), [0.1, 0.0, 0.0]]))
    started = arm.t
    assert run(executor, target).state is MotionState.SUCCEEDED
    assert arm.t - started >= 10.0


def test_a_sparse_trajectory_retreat_starts_at_the_actual_halt():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    polls = [0]

    def halted():
        polls[0] += 1
        return polls[0] > 50

    segment = line(np.zeros(3), [1.0, 0.0, 0.0], frames=2)
    outcome = run(executor, segment, halted=halted)
    assert outcome.state is MotionState.HALTED and 0.0 < outcome.report["path_progress"] < 1.0
    assert executor.hold(0.3, carrying=False, cancelled=lambda: False).state is MotionState.SUCCEEDED
    stopped_at = executor.measured(DRIVEN)
    back = segment.walked_back(stopped_at, path_progress=outcome.report["path_progress"])
    np.testing.assert_allclose(back.start, stopped_at)
    np.testing.assert_allclose(back.end, np.zeros(3))
    assert run(executor, back).state is MotionState.SUCCEEDED


def test_the_raise_stays_inside_the_limit_margin_and_no_command_crosses_the_limit():
    class Shaper:
        def raise_for(self, names, positions):
            return np.full_like(positions, 0.1) * (positions[:, :1] > 0.05)

    arm = SimArm()
    config = StreamConfig(tracking_error_rad=0.3, arrival_error_rad=0.2, limits={"l0": (-1.0, 0.2)})
    # 0.1 + offset 0.05 sits inside the margin; the 0.1 raise would push it to 0.25.
    run(StreamExecutor(arm, config, shaper=Shaper()), line(np.zeros(3), [0.1, 0.0, 0.0], seconds=1.0))
    assert max(command[0] for command in arm.published) == pytest.approx(0.2 - 0.005)
    # A path whose own command reaches past the margin is held at the limit, never beyond.
    arm = SimArm()
    run(StreamExecutor(arm, config), line(np.zeros(3), [0.3, 0.0, 0.0], seconds=1.0))
    assert max(command[0] for command in arm.published) == pytest.approx(0.2)


def test_the_shaper_raise_is_relative_to_the_segment_start():
    class Shaper:
        def raise_for(self, names, positions):
            return 0.1 * positions  # more raise further out

    arm = SimArm()
    takeover = arm.target.copy()
    run(StreamExecutor(arm, CONFIG, shaper=Shaper()), line(np.zeros(3), [0.3, 0.0, 0.0]))
    np.testing.assert_allclose(arm.published[0], takeover)
    np.testing.assert_allclose(arm.published[-1][0], 0.3 + SAG[0] + 0.03, atol=1e-9)


@pytest.mark.parametrize(("board_off", "moves"), [(0.003, False), (0.05, True), (0.2, False)])
def test_return_home_moves_the_controllers_command_to_the_fixed_home(board_off, moves):
    arm = SimArm(start=(0.4, 0.0, -1.2, 0.5, 0.5))
    arm.target[:3] = [0.4 + board_off, 0.0, -1.2]  # where the last landing left the command
    arm.measured[:3] = arm.target[:3] - SAG[:3]
    home_command = {"l0": 0.4, "l1": 0.0, "l2": -1.2}
    executor = StreamExecutor(
        arm, StreamConfig(tracking_error_rad=0.3, arrival_error_rad=0.2, home_command=home_command)
    )
    home = ArmSegment("approach", SegmentKind.TRAJECTORY, DRIVEN, np.zeros((2, 3)), np.array([0.0, 1.0]))
    outcome = executor.return_home(home, cancelled=lambda: False)
    assert outcome.state is MotionState.SUCCEEDED
    assert bool(arm.published) == moves
    if moves:
        np.testing.assert_allclose(arm.target[:3], [0.4, 0.0, -1.2], atol=1e-9)
    elif board_off > 0.1:
        assert executor.report["home_command_skipped"][0] == pytest.approx(0.6)


def test_return_home_without_a_home_command_leaves_it_to_the_sequence():
    executor = StreamExecutor(SimArm(), CONFIG)
    assert executor.return_home(line(np.zeros(3), np.zeros(3)), cancelled=lambda: False) is None


def test_hold_keeps_the_last_command_and_finish_hands_the_robot_back():
    arm = SimArm()
    executor = StreamExecutor(arm, CONFIG)
    run(executor, line(np.zeros(3), [0.2, 0.0, 0.0]))
    sent = len(arm.published)
    assert executor.hold(0.5, carrying=True, cancelled=lambda: False).state is MotionState.SUCCEEDED
    assert len(arm.published) == sent  # the I/O keeps the last command alive
    assert executor.hold(5.0, carrying=True, cancelled=lambda: True).state is MotionState.CANCELLED
    executor.finish()
    assert arm.released == 1
    # A fresh goal takes over from the controller's target again.
    arm.published.clear()
    run(executor, line(arm.measured[:3].copy(), [0.1, 0.0, 0.0]))
    np.testing.assert_allclose(arm.published[0][3:], arm.published[-1][3:])


def test_a_sequence_ends_a_retreat_where_the_executor_says():
    from types import SimpleNamespace

    calls = []

    class Executor:
        def return_home(self, home, *, cancelled):
            calls.append(home.name)
            from manipulation_execution.pick.pipeline.contracts import MotionOutcome

            return MotionOutcome(MotionState.SUCCEEDED, "at the home command")

        def measured(self, names):
            raise AssertionError("the default home check must not run")

    backends = SimpleNamespace(executor=Executor())
    sequence = GraspSequence(backends)
    from manipulation_execution.pick.pipeline.contracts import Effect, EffectorAction, GraspPlan, Move
    from manipulation_execution.pick.pipeline.sequence import SequenceResult

    plan = GraspPlan((Move(line(np.zeros(3), np.ones(3))), Effect(EffectorAction.CLOSE)), 0)
    result = SequenceResult()
    sequence._return_home(plan, lambda: False, result)
    assert calls == ["approach"] and result.completed == ["home"]
