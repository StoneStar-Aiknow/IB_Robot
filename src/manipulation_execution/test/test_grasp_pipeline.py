"""Contract tests: the grasp pipeline drives both robot shapes through fake backends.

A streaming arm (X2-like: timed trajectories, halts between frames, a target
watch) and a motion-server arm (SO-101-like: joint targets, a refinement after
the approach, release in place) must both fit the same contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from manipulation_execution.grasp_pipeline.backends import BackendLoadError, backend_spec, load_backends
from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    EffectorOutcome,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Hold,
    MotionOutcome,
    MotionState,
    Move,
    Observation,
    Proposals,
    SegmentKind,
)
from manipulation_execution.grasp_pipeline.pipeline import GraspPipeline, PickMode, PipelineConfig

JOINTS = tuple(f"j{i}" for i in range(3))
HOME = np.zeros(3)


def trajectory(name, start, end, frames=5):
    positions = np.linspace(start, end, frames)
    return ArmSegment(name, SegmentKind.TRAJECTORY, JOINTS, positions, np.arange(frames) * 0.02)


def target(name, start, end):
    return ArmSegment(name, SegmentKind.TARGET, JOINTS, np.stack([start, end]))


def observation(stamp=1):
    return Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        depth_m=np.ones((4, 4), np.float32),
        intrinsics=np.eye(3),
        frame_id="camera",
        stamp_ns=stamp,
        base_frame="base",
        base_from_camera=np.eye(4),
    )


class Source:
    def __init__(self):
        self.captures = 0

    def capture(self, *, deadline, cancelled):
        self.captures += 1
        return observation(self.captures)


class Proposer:
    def propose(self, observation, query, *, deadline, cancelled):
        return Proposals(np.ones((4, 4), bool), np.zeros((3, 3)), np.stack([np.eye(4)] * 2), [0.9, 0.5])


@dataclass
class Executor:
    """Moves instantly frame by frame; ``streaming`` refuses nothing, otherwise trajectories are refused."""

    streaming: bool = True
    fail_on: str | None = None
    cancel_on: str | None = None
    #: A segment whose execution raises (a broken motion server).
    raise_on: str | None = None
    #: What supervise() reports while an end effector acts.
    supervision: str = ""
    #: Retraced segments land this far off their end (sag), to exercise the home correction.
    retrace_error: float = 0.0
    position: np.ndarray = field(default_factory=lambda: HOME.copy())
    log: list = field(default_factory=list)
    stopped: list = field(default_factory=list)

    def execute(self, segment, *, carrying, halted, cancelled):
        self.log.append((segment.name, segment.kind, carrying))
        if segment.name == self.raise_on:
            raise RuntimeError("motion server went away")
        if not self.streaming and segment.kind is SegmentKind.TRAJECTORY:
            return MotionOutcome(MotionState.FAILED, "motion server takes targets only")
        assert np.allclose(self.position, segment.start, atol=0.05), (segment.name, self.position, segment.start)
        frames = segment.positions if segment.kind is SegmentKind.TRAJECTORY else segment.positions[1:]
        for frame in frames:
            if segment.name == self.cancel_on or cancelled():
                return MotionOutcome(MotionState.CANCELLED, "cancelled", self.position.copy())
            if segment.name == self.fail_on:
                return MotionOutcome(MotionState.FAILED, "tracking error 0.31 rad", self.position.copy())
            if halted():
                return MotionOutcome(MotionState.HALTED, "halted", self.position.copy())
            self.position = np.array(frame, dtype=float)
        if segment.name.startswith("retreat_"):
            self.position = self.position + self.retrace_error
        return MotionOutcome(MotionState.SUCCEEDED, stopped_at=self.position.copy())

    def hold(self, seconds, *, carrying, cancelled):
        self.log.append(("hold", None, carrying))
        return MotionOutcome(MotionState.SUCCEEDED)

    def supervise(self):
        return lambda: self.supervision

    def measured(self, joint_names):
        assert tuple(joint_names) == JOINTS
        return self.position.copy()

    def stop(self, reason):
        self.stopped.append(reason)


@dataclass
class Effector:
    held: list = field(default_factory=lambda: [True])
    halt_close: bool = False
    log: list = field(default_factory=list)

    #: Called with each action before it runs (a test's hook into the goal's flags).
    on_act: object = None

    def act(self, action, *, halted, cancelled):
        self.log.append(action)
        if self.on_act is not None:
            self.on_act(action)
        if cancelled():
            return EffectorOutcome(False, f"cancelled during {action.value}")
        if action is EffectorAction.CLOSE:
            if self.halt_close and halted():
                return EffectorOutcome(True, "halted", halted=True)
            held = self.held.pop(0) if len(self.held) > 1 else self.held[0]
            return EffectorOutcome(True, held=held, confidence=0.8 if held else 0.0)
        return EffectorOutcome(True)


class Watch:
    """Says the target moved on the ``fire_at``-th poll; records the arm position at each poll."""

    def __init__(self, executor, fire_at=None, fire_plans=(0,)):
        self.executor, self.fire_at, self.fire_plans = executor, fire_at, fire_plans
        self.plan_number, self.polls, self.armed, self.phases = -1, 0, False, []

    def arm(self, observation, proposals, plan):
        assert proposals.grasps.shape[1:] == (4, 4)
        self.plan_number, self.polls, self.armed = self.plan_number + 1, 0, True

    def phase(self, name, event):
        self.phases.append((name, event))

    def moved(self):
        assert self.armed
        self.polls += 1
        if self.fire_at is not None and self.plan_number in self.fire_plans and self.polls >= self.fire_at:
            return "target shifted 3 cm"
        return None

    def disarm(self):
        self.armed = False
        return {"polls": self.polls}


class StreamingPlanner:
    """X2-like: timed trajectories approach -> grasp -> close -> lift -> hold; put back by retracing."""

    def __init__(self, failures=()):
        self.failures, self.calls = list(failures), 0

    def plan(self, observation, proposals, *, deadline, cancelled):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        approach_end, contact, lifted = np.array([0.5, 0.2, 0.0]), np.array([0.6, 0.3, 0.0]), np.array([0.6, 0.3, 0.2])
        return GraspPlan(
            (
                Effect(EffectorAction.PRESHAPE),
                Move(trajectory("approach", HOME, approach_end, frames=11)),
                Move(trajectory("grasp", approach_end, contact)),
                Effect(EffectorAction.CLOSE),
                Move(trajectory("lift", contact, lifted)),
                Hold(1.0, "hold_lifted"),
            ),
            candidate_index=1,
            report={"planner": "streaming"},
        )

    def refine(self, plan, completed, *, deadline, cancelled):
        raise AssertionError("the streaming plan never asks for refinement")


class ServerPlanner:
    """SO-101-like: joint targets; the descent is replanned from where the approach ended."""

    def __init__(self, executor, refine_error=None):
        self.executor, self.refine_error, self.refined = executor, refine_error, []

    def plan(self, observation, proposals, *, deadline, cancelled):
        approach_end = np.array([0.4, 0.1, 0.1])
        return GraspPlan(
            (
                Effect(EffectorAction.PRESHAPE),
                Move(target("approach", HOME, approach_end), refine_after=True),
                Move(target("descend", approach_end, np.array([0.4, 0.1, 0.0]))),
                Effect(EffectorAction.CLOSE),
                Move(target("transport", np.array([0.4, 0.1, 0.0]), np.array([0.0, 0.3, 0.3]))),
            ),
            candidate_index=0,
            release=(Effect(EffectorAction.RELEASE),),
        )

    def refine(self, plan, completed, *, deadline, cancelled):
        self.refined.append(plan.steps[completed].name)
        if self.refine_error is not None:
            raise self.refine_error
        here = self.executor.measured(JOINTS)
        contact = here + [0.01, 0.0, -0.1]  # realigned from the measured pose
        transport = plan.steps[4].segment
        steps = plan.steps[: completed + 1] + (
            Move(target("descend", here, contact)),
            Effect(EffectorAction.CLOSE),
            Move(target("transport", contact, transport.end)),
        )
        return GraspPlan(steps, plan.candidate_index, release=plan.release)


def streaming(fire_at=None, fire_plans=(0,), **executor_options):
    executor = Executor(streaming=True, **executor_options)
    effector, planner = Effector(), StreamingPlanner()
    watch = Watch(executor, fire_at, fire_plans)
    backends = GraspBackends(Source(), Proposer(), planner, executor, effector, watch)
    return backends, executor, effector, planner, watch


def server(**planner_options):
    executor = Executor(streaming=False)
    effector = Effector()
    planner = ServerPlanner(executor, **planner_options)
    return GraspBackends(Source(), Proposer(), planner, executor, effector), executor, effector, planner


def run(backends, mode=PickMode.EXECUTE, config=PipelineConfig(), cancelled=lambda: False, **options):
    return GraspPipeline(backends, config, clock=lambda: 0.0).run(
        "red bottle", mode=mode, deadline=100.0, cancelled=cancelled, **options
    )


def names(executor):
    return [name for name, _, _ in executor.log]


def test_a_streaming_arm_picks_and_holds_the_object_lifted():
    backends, executor, effector, _, watch = streaming()
    outcome = run(backends)
    assert outcome.success and outcome.held is True and outcome.candidate_index == 1
    assert names(executor) == ["approach", "grasp", "lift", "hold"]
    assert [carrying for _, _, carrying in executor.log] == [False, False, True, True]
    assert effector.log == [EffectorAction.PRESHAPE, EffectorAction.CLOSE]
    assert outcome.completed_phases == [
        "observe",
        "perceive",
        "plan",
        "preshape",
        "approach",
        "grasp",
        "close",
        "lift",
        "hold_lifted",
    ]
    # The watch is polled only while nothing is held.
    assert watch.polls == 11 + 5 and outcome.report["watch"] == [{"polls": 16}]
    # It hears the sequence's steps, not the pipeline's own phases.
    assert watch.phases[:2] == [("preshape", "start"), ("preshape", "end")]
    assert ("approach", "start") in watch.phases and not any(name == "plan" for name, _ in watch.phases)


def test_a_motion_server_arm_refines_after_the_approach_and_releases_in_place():
    backends, executor, effector, planner = server()
    outcome = run(backends, release_after_success=True)
    assert outcome.success and outcome.released_after_success and planner.refined == ["approach"]
    assert names(executor) == ["approach", "descend", "transport"]
    assert all(kind is SegmentKind.TARGET for _, kind, _ in executor.log)
    np.testing.assert_allclose(executor.position, [0.0, 0.3, 0.3])
    assert effector.log[-1] is EffectorAction.RELEASE and outcome.completed_phases[-1] == "release"


def test_a_moved_target_halts_the_approach_retraces_it_and_plans_again():
    backends, executor, _, planner, watch = streaming(fire_at=4)
    outcome = run(backends)
    assert outcome.success and planner.calls == 2 and backends.observation.captures == 2
    # Halted after three frames: settle, back over them, then the second plan from home.
    assert names(executor) == ["approach", "hold", "retreat_approach_partial", "approach", "grasp", "lift", "hold"]
    assert outcome.report["retreats"] == [{"step": "approach", "reason": "target shifted 3 cm"}]
    assert outcome.attempts == 2


def test_without_replans_left_a_moved_target_ends_the_goal_with_the_arm_home():
    backends, executor, effector, _, _ = streaming(fire_at=14)  # during the grasp segment
    outcome = run(backends, config=PipelineConfig(max_moved_replans=0))
    assert not outcome.success and outcome.error_code == "TARGET_MOVED" and outcome.at_rest
    assert names(executor) == ["approach", "grasp", "hold", "retreat_grasp_partial", "retreat_approach"]
    np.testing.assert_allclose(executor.position, HOME)
    assert EffectorAction.CLOSE not in effector.log


def test_a_moved_target_during_the_close_releases_and_retraces():
    backends, executor, effector, _, watch = streaming(fire_at=17)
    effector.halt_close = True
    outcome = run(backends, config=PipelineConfig(max_moved_replans=0))
    assert outcome.error_code == "TARGET_MOVED" and outcome.report["retreats"][0]["step"] == "close"
    assert effector.log == [EffectorAction.PRESHAPE, EffectorAction.CLOSE, EffectorAction.RELEASE]
    assert names(executor) == ["approach", "grasp", "retreat_grasp", "retreat_approach"]


class FailingWatch(Watch):
    """Can no longer tell from the ``fail_at``-th poll on; the target itself never reads moved."""

    def __init__(self, executor, fail_at):
        super().__init__(executor)
        self.fail_at, self.failed_polls = fail_at, 0

    def failed(self):
        assert self.armed
        self.failed_polls += 1
        return "no watch status for 1.2 s" if self.polls >= self.fail_at else None


class CloseBarrierWatch(Watch):
    """Records a close barrier fault; stop surfaces it, log only keeps observing."""

    def __init__(self, executor, mode):
        super().__init__(executor)
        self.mode, self.fault = mode, None

    def phase(self, name, event):
        super().phase(name, event)
        if (name, event) == ("close", "start"):
            self.fault = "the grasp head did not close its dispatch gate"

    def failed(self):
        assert self.armed
        return self.fault if self.mode == "stop" else None


def failing(fail_at):
    backends, executor, effector, planner, _ = streaming()
    watch = FailingWatch(executor, fail_at)
    return replace(backends, watch=watch), executor, effector, planner, watch


def test_a_watch_that_fails_mid_approach_stops_the_arm_in_place_and_nothing_is_retraced_or_replanned():
    backends, executor, effector, planner, watch = failing(fail_at=4)
    outcome = run(backends)
    assert not outcome.success and outcome.error_code == "WATCH_FAILED" and not outcome.at_rest
    assert "during approach: no watch status for 1.2 s" in outcome.message
    # Halted where the watch failed, then held: no settle-and-retrace, no second plan, no release.
    assert names(executor) == ["approach"] and executor.stopped == [outcome.message]
    np.testing.assert_allclose(executor.position, trajectory("approach", HOME, [0.5, 0.2, 0.0], 11).positions[2])
    assert planner.calls == 1 and effector.log == [EffectorAction.PRESHAPE]
    assert "retreats" not in outcome.report and outcome.report["watch"] == [{"polls": 4}]


def test_a_watch_that_fails_during_the_close_keeps_the_hand_as_it_is():
    backends, executor, effector, _, watch = failing(fail_at=17)
    effector.halt_close = True
    outcome = run(backends)
    assert outcome.error_code == "WATCH_FAILED" and "during close" in outcome.message
    assert effector.log == [EffectorAction.PRESHAPE, EffectorAction.CLOSE]  # no release
    assert names(executor) == ["approach", "grasp"] and len(executor.stopped) == 1


@pytest.mark.parametrize("mode", ["stop", "log"])
def test_a_close_barrier_failure_stops_in_place_only_in_stop_mode(mode):
    backends, executor, effector, _, _ = streaming()
    watch = CloseBarrierWatch(executor, mode)
    backends = replace(backends, watch=watch)
    effector.halt_close = True

    outcome = run(backends)

    assert effector.log == [EffectorAction.PRESHAPE, EffectorAction.CLOSE]
    if mode == "stop":
        assert not outcome.success and outcome.error_code == "WATCH_FAILED"
        assert "during close: the grasp head did not close its dispatch gate" in outcome.message
        assert names(executor) == ["approach", "grasp"] and len(executor.stopped) == 1
        assert "retreats" not in outcome.report and EffectorAction.RELEASE not in effector.log
    else:
        assert outcome.success and names(executor) == ["approach", "grasp", "lift", "hold"]


def test_a_watch_failure_is_not_asked_once_the_object_is_held():
    """Lifting and carrying move the target with the hand: the approach's watch no longer applies."""
    backends, executor, _, _, watch = failing(fail_at=10**6)
    outcome = run(backends)
    assert outcome.success and watch.failed_polls == watch.polls == 16


def test_a_watch_that_cannot_start_ends_the_goal_before_anything_moves():
    backends, executor, effector, _, watch = streaming()

    def refuse(observation, proposals, plan):
        raise GraspFailure("WATCH_FAILED", "the grasp head sent no watch status")

    watch.arm = refuse
    outcome = run(backends)
    assert (outcome.success, outcome.error_code) == (False, "WATCH_FAILED") and outcome.at_rest
    assert names(executor) == [] and effector.log == [] and not executor.stopped


def test_nothing_held_skips_the_lift_and_takes_everything_back():
    backends, executor, effector, _, _ = streaming()
    effector.held = [False]
    outcome = run(backends)
    assert not outcome.success and outcome.error_code == "GRASP_VERIFICATION_FAILED" and outcome.held is False
    assert names(executor) == ["approach", "grasp", "retreat_grasp", "retreat_approach"]
    assert effector.log[-1] is EffectorAction.RELEASE and outcome.at_rest


def test_a_retrace_that_lands_off_home_gets_one_correction():
    backends, executor, effector, _, _ = streaming(retrace_error=0.03)
    effector.held = [False]
    outcome = run(backends)
    assert names(executor)[-1] == "home_correction" and executor.log[-1][1] is SegmentKind.TARGET
    np.testing.assert_allclose(executor.position, HOME)
    assert outcome.at_rest


def test_release_after_success_without_a_release_plan_puts_the_object_back():
    backends, executor, effector, _, _ = streaming()
    outcome = run(backends, release_after_success=True)
    assert outcome.success and outcome.released_after_success
    assert names(executor) == ["approach", "grasp", "lift", "hold", "retreat_lift", "retreat_grasp", "retreat_approach"]
    assert [carrying for _, _, carrying in executor.log][4:] == [True, False, False]


@pytest.mark.parametrize(
    ("fail_on", "cancel_on", "code"), [("grasp", None, "MOTION_FAILED"), (None, "lift", "CANCELED")]
)
def test_a_motion_failure_or_cancel_stops_in_place_without_undoing(fail_on, cancel_on, code):
    backends, executor, effector, _, _ = streaming(fail_on=fail_on, cancel_on=cancel_on)
    outcome = run(backends)
    assert not outcome.success and outcome.error_code == code and not outcome.at_rest
    assert len(executor.stopped) == 1 and not any(name.startswith("retreat") for name in names(executor))


def test_an_arm_that_fails_its_supervision_while_the_hand_acts_stops_the_goal():
    backends, executor, effector, _, _ = streaming(supervision="tracking anomaly while holding")
    outcome = run(backends)
    assert outcome.error_code == "MOTION_FAILED" and "arm during preshape" in outcome.message
    assert not outcome.at_rest and len(executor.stopped) == 1 and names(executor) == []


def test_a_cancel_while_the_hand_acts_is_a_cancel_not_a_broken_hand():
    backends, executor, effector, _, _ = streaming()
    flag = []
    effector.on_act = lambda action: action is EffectorAction.CLOSE and flag.append(True)
    outcome = run(backends, cancelled=lambda: bool(flag))
    assert outcome.error_code == "CANCELED" and executor.stopped


def test_the_deadline_stops_observing_and_planning_but_not_a_running_sequence():
    backends, executor, effector, _, _ = streaming()
    now = [0.0]
    effector.on_act = lambda action: now.__setitem__(0, 200.0)  # the deadline passes as the hand preshapes
    outcome = GraspPipeline(backends, clock=lambda: now[0]).run(
        "cup", mode=PickMode.EXECUTE, deadline=100.0, cancelled=lambda: False
    )
    assert outcome.success and names(executor) == ["approach", "grasp", "lift", "hold"]


def test_a_retreat_that_raises_after_a_failed_refinement_stops_the_arm():
    backends, executor, effector, planner = server(refine_error=GraspFailure("IK_FAILED", "no realigned contact"))
    executor.raise_on = "retreat_approach"
    outcome = run(backends)
    assert outcome.error_code == "INTERNAL_ERROR" and "motion server went away" in outcome.message
    assert not outcome.at_rest and len(executor.stopped) == 1


def test_a_failed_refinement_takes_the_approach_back():
    backends, executor, effector, planner = server(refine_error=GraspFailure("IK_FAILED", "no realigned contact"))
    outcome = run(backends)
    assert outcome.error_code == "IK_FAILED" and outcome.at_rest
    assert names(executor) == ["approach", "retreat_approach"]
    np.testing.assert_allclose(executor.position, HOME)


def test_a_motion_server_refuses_timed_trajectories():
    executor = Executor(streaming=False)
    backends = GraspBackends(Source(), Proposer(), StreamingPlanner(), executor, Effector())
    outcome = run(backends)
    assert outcome.error_code == "MOTION_FAILED" and executor.stopped


def test_planning_observes_again_only_for_retryable_failures():
    backends, executor, *_ = streaming()
    backends.planner.failures = [GraspFailure("NO_EXECUTABLE_CANDIDATE", "all screened out", retryable=True)]
    outcome = run(backends)
    assert outcome.success and outcome.attempts == 2 and backends.observation.captures == 2
    backends, executor, *_ = streaming()
    backends.planner.failures = [GraspFailure("NO_EXECUTABLE_CANDIDATE", "x", retryable=True)] * 2
    outcome = run(backends)
    assert outcome.error_code == "NO_EXECUTABLE_CANDIDATE" and not executor.log
    backends, executor, *_ = streaming()
    backends.planner.failures = [GraspFailure("PLANNER_FAULT", "CUDA error")]
    assert run(backends).error_code == "PLANNER_FAULT" and backends.observation.captures == 1


def test_a_passed_deadline_reports_a_timeout():
    backends, *_ = streaming()
    backends.planner.failures = [GraspFailure("NO_EXECUTABLE_CANDIDATE", "x", retryable=True)]
    outcome = GraspPipeline(backends, clock=lambda: 200.0).run(
        "cup", mode=PickMode.EXECUTE, deadline=100.0, cancelled=lambda: False
    )
    assert outcome.error_code == "SKILL_TIMEOUT"


@pytest.mark.parametrize(
    ("mode", "flip_after", "code"),
    [
        (PickMode.EXECUTE, "plan", "SKILL_TIMEOUT"),
        (PickMode.EXECUTE, "observe", "SKILL_TIMEOUT"),
        (PickMode.PLAN_ONLY, "plan", "SKILL_TIMEOUT"),
        (PickMode.OBSERVE_ONLY, "observe", "SKILL_TIMEOUT"),
        (PickMode.EXECUTE, "plan", "CANCELED"),
        (PickMode.PLAN_ONLY, "plan", "CANCELED"),
        (PickMode.OBSERVE_ONLY, "observe", "CANCELED"),
    ],
)
def test_a_budget_that_ends_inside_a_step_ends_the_goal_at_its_boundary(mode, flip_after, code):
    """A step may return normally after the budget ended; the boundary, not the return, decides."""
    backends, executor, effector, _, _ = streaming()
    now, cancelled = [0.0], [False]

    def end_the_budget():
        if code == "SKILL_TIMEOUT":
            now[0] = 200.0
        else:
            cancelled[0] = True

    class LateSource(Source):
        def capture(self, *, deadline, cancelled):
            made = super().capture(deadline=deadline, cancelled=cancelled)
            end_the_budget()
            return made

    class LatePlanner(StreamingPlanner):
        def plan(self, observation, proposals, *, deadline, cancelled):
            made = super().plan(observation, proposals, deadline=deadline, cancelled=cancelled)
            end_the_budget()
            return made

    if flip_after == "observe":
        late = LateSource()
        backends = replace(backends, observation=late)
    else:
        late = LatePlanner()
        backends = replace(backends, planner=late)
    outcome = GraspPipeline(backends, clock=lambda: now[0]).run(
        "cup", mode=mode, deadline=100.0, cancelled=lambda: cancelled[0]
    )
    assert not outcome.success and outcome.error_code == code
    assert not executor.log and not effector.log and not executor.stopped
    assert (late.calls if flip_after == "plan" else late.captures) == 1


@pytest.mark.parametrize(
    ("mode", "phases"), [(PickMode.PLAN_ONLY, ["observe", "perceive", "plan"]), (PickMode.OBSERVE_ONLY, ["observe"])]
)
def test_plan_and_observe_only_never_move(mode, phases):
    backends, executor, effector, *_ = streaming()
    outcome = run(backends, mode=mode, release_after_success=True)
    assert outcome.success and not outcome.released_after_success
    assert outcome.completed_phases == phases and not executor.log and not effector.log


def test_a_plan_reports_how_long_each_step_took_and_what_perception_measured():
    class Measuring(Proposer):
        def propose(self, observation, query, *, deadline, cancelled):
            proposals = super().propose(observation, query, deadline=deadline, cancelled=cancelled)
            return replace(proposals, report={"perception_timings_s": {"detect": 0.4}})

    backends, *_ = streaming()
    ticks = iter(range(1000))
    outcome = GraspPipeline(replace(backends, proposer=Measuring()), clock=lambda: float(next(ticks))).run(
        "red bottle", mode=PickMode.PLAN_ONLY, deadline=1e9, cancelled=lambda: False
    )
    assert outcome.success
    assert outcome.report["steps_s"] == {"observe": 1.0, "perceive": 1.0, "plan": 1.0}
    assert outcome.report["perceive"] == {"perception_timings_s": {"detect": 0.4}}


def test_the_capture_reports_how_it_went_in_the_goal_report():
    backends, *_ = streaming()
    capture = backends.observation.capture

    def reporting(**budget):
        observation = capture(**budget)
        return observation.__class__(**{**vars(observation), "report": {"wait_s": 0.2, "recaptured": False}})

    backends.observation.capture = reporting
    outcome = run(backends)
    assert outcome.report["observe"] == {"wait_s": 0.2, "recaptured": False}


def test_an_empty_query_is_refused_before_observing():
    backends, *_ = streaming()
    outcome = GraspPipeline(backends).run(" ", mode=PickMode.EXECUTE, deadline=1e12, cancelled=lambda: False)
    assert outcome.error_code == "INVALID_TARGET" and backends.observation.captures == 0


def test_plans_and_segments_are_checked_on_construction():
    move = Move(trajectory("approach", HOME, np.ones(3)))
    with pytest.raises(ValueError, match="closes exactly once"):
        GraspPlan((move,), 0)
    with pytest.raises(ValueError, match="never closes"):
        GraspPlan((move, Effect(EffectorAction.CLOSE)), 0, release=(Effect(EffectorAction.CLOSE),))
    with pytest.raises(ValueError, match="strictly increase"):
        ArmSegment("x", SegmentKind.TRAJECTORY, JOINTS, np.zeros((2, 3)), [0.0, 0.0])
    with pytest.raises(ValueError, match="start and a target"):
        ArmSegment("x", SegmentKind.TARGET, JOINTS, np.zeros((3, 3)))
    segment = trajectory("approach", HOME, np.ones(3), frames=5)
    assert segment.walked_back(HOME) is None
    back = segment.walked_back(np.full(3, 0.5))
    np.testing.assert_allclose(back.positions[[0, -1]], [np.full(3, 0.5), HOME])
    np.testing.assert_allclose(back.times, [0.0, 0.02, 0.04])
    sparse = ArmSegment("sparse", SegmentKind.TRAJECTORY, JOINTS, np.stack([HOME, np.ones(3)]), [0.0, 1.0])
    back = sparse.walked_back(np.full(3, 0.6), path_progress=0.6)
    np.testing.assert_allclose(back.positions, [np.full(3, 0.6), HOME])
    np.testing.assert_allclose(back.times, [0.0, 0.6])
    assert not segment.positions.flags.writeable


def test_backends_load_by_name_and_reject_anything_else():
    assert backend_spec({}) is None and backend_spec({"backend": "a.b:c"}) == "a.b:c"
    for spec in ("no_colon", "builtins:dict", "manipulation_execution.nope:x"):
        with pytest.raises(BackendLoadError):
            load_backends(spec)
    backends = load_backends(f"{__name__}:make_backends", tag="x2")
    assert isinstance(backends, GraspBackends) and backends.planner.tag == "x2"


def make_backends(tag):
    planner = StreamingPlanner()
    planner.tag = tag
    return GraspBackends(Source(), Proposer(), planner, Executor(), Effector())
