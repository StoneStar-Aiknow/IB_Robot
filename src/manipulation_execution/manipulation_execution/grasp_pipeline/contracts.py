"""What a robot provides to the robot-agnostic grasp pipeline.

The pipeline serves ``PickObject`` for any robot: it observes, asks the grasp
models for candidates, has the robot's planner choose one and plan it, and runs
the plan step by step, retracing what was done when the target moves or nothing
was gripped. Everything that depends on the robot body sits behind the
protocols below, so the pipeline never sees vendor types, a motion-planning
library or the shape of an end effector.

Two robot shapes must fit, and the contract tests drive both:

* a streaming arm (AgiBot X2): the planner returns timed joint trajectories,
  the executor streams them at a fixed rate and can halt between frames;
* a motion-server arm (SO-101): the planner returns joint targets, the robot's
  motion server plans the path to each, and the plan is refined after the
  approach from the measured pose.

Models are reached only through the typed perception services, whatever
accelerator serves them (Ascend 310P or CUDA): a backend's ``propose`` calls
services, it never loads a model.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

#: A zero-argument predicate polled between frames or calls, e.g. goal cancellation.
Flag = Callable[[], bool]


def _frozen(array, dtype, shape_check, message):
    value = np.array(array, dtype=dtype, copy=True)
    if not shape_check(value):
        raise ValueError(message)
    value.setflags(write=False)
    return value


class GraspFailure(RuntimeError):
    """A failure that ends the goal with a ``PickObject`` error code.

    ``retryable`` failures (e.g. no executable candidate in this observation)
    let the pipeline observe again while attempts remain.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False, reason_code: str = ""):
        super().__init__(message)
        self.code = code
        self.reason_code = reason_code or code
        self.retryable = retryable


@dataclass(frozen=True)
class Observation:
    """One registered RGB-D frame and where the camera was at its image time.

    ``depth_m`` is registered to ``rgb`` (same pixel grid), in metres; zero or
    non-finite means no measurement. ``base_from_camera`` maps camera coordinates
    into ``base_frame`` at ``stamp_ns``. ``robot_state`` belongs to the backend
    that captured it (e.g. joint feedback at the image time) and is passed back
    to the same robot's planner unread. ``report`` says how the capture went
    (timings, retries); the pipeline puts it in the goal's report.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    frame_id: str
    stamp_ns: int
    base_frame: str
    base_from_camera: np.ndarray
    robot_state: object = None
    report: Mapping = field(default_factory=dict)

    def __post_init__(self):
        rgb = _frozen(self.rgb, np.uint8, lambda a: a.ndim == 3 and a.shape[2] == 3, "rgb must be HxWx3")
        depth = _frozen(self.depth_m, np.float32, lambda a: a.shape == rgb.shape[:2], "depth must match rgb")
        intrinsics = _frozen(self.intrinsics, np.float64, lambda a: a.shape == (3, 3), "intrinsics must be 3x3")
        pose = _frozen(
            self.base_from_camera,
            np.float64,
            lambda a: a.shape == (4, 4) and np.isfinite(a).all(),
            "base_from_camera must be a finite 4x4",
        )
        if not self.frame_id or not self.base_frame or self.stamp_ns <= 0:
            raise ValueError("an observation needs its frames and a positive stamp")
        for name, value in (("rgb", rgb), ("depth_m", depth), ("intrinsics", intrinsics), ("base_from_camera", pose)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Proposals:
    """The target and grasp candidates for one observation, in its camera frame.

    ``grasps`` are poses of the grasp model's gripper frame; only the robot's
    planner knows how that frame sits on its end effector. Candidates are in the
    model's order (confidence-sorted); ``report`` carries what the services said.
    ``scene`` belongs to the backend that proposed (e.g. a collision scene built
    alongside the candidates) and is passed to the same robot's planner unread.
    """

    target_mask: np.ndarray
    target_points: np.ndarray
    grasps: np.ndarray
    confidence: np.ndarray
    report: Mapping = field(default_factory=dict)
    scene: object = None

    def __post_init__(self):
        mask = _frozen(self.target_mask, bool, lambda a: a.ndim == 2, "target_mask must be HxW")
        points = _frozen(
            self.target_points, np.float64, lambda a: a.ndim == 2 and a.shape[1] == 3, "target_points must be Nx3"
        )
        grasps = _frozen(
            self.grasps,
            np.float64,
            lambda a: a.ndim == 3 and a.shape[1:] == (4, 4) and np.isfinite(a).all(),
            "grasps must be finite Kx4x4",
        )
        confidence = _frozen(
            self.confidence,
            np.float64,
            lambda a: a.shape == (len(grasps),) and np.isfinite(a).all(),
            "one finite confidence per grasp",
        )
        for name, value in (("target_mask", mask), ("target_points", points), ("grasps", grasps)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "confidence", confidence)


class SegmentKind(enum.Enum):
    #: Waypoints with times from zero; the executor follows them as timed.
    TRAJECTORY = "trajectory"
    #: A start and a target; the robot's motion server plans the path between.
    TARGET = "target"


@dataclass(frozen=True)
class ArmSegment:
    """One arm motion of a plan, in ``joint_names`` order.

    A ``TRAJECTORY`` has N >= 2 waypoints and N strictly increasing times from
    zero. A ``TARGET`` has exactly two rows, the planned start and the target,
    and no times; the start is what a retreat returns to.
    """

    name: str
    kind: SegmentKind
    joint_names: tuple[str, ...]
    positions: np.ndarray
    times: np.ndarray | None = None

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not self.name or not names or len(set(names)) != len(names):
            raise ValueError("a segment needs a name and distinct joint names")
        positions = _frozen(
            self.positions,
            np.float64,
            lambda a: a.ndim == 2 and a.shape[1] == len(names) and len(a) >= 2 and np.isfinite(a).all(),
            "positions must be finite Nx(joints), N >= 2",
        )
        times = None
        if self.kind is SegmentKind.TRAJECTORY:
            times = _frozen(
                self.times,
                np.float64,
                lambda a: (
                    a.shape == (len(positions),) and a[0] == 0 and np.all(np.diff(a) > 0) and np.isfinite(a).all()
                ),
                "trajectory times must start at zero and strictly increase",
            )
        elif self.times is not None or len(positions) != 2:
            raise ValueError("a target segment is a start and a target, without times")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "times", times)

    @property
    def start(self) -> np.ndarray:
        return self.positions[0]

    @property
    def end(self) -> np.ndarray:
        return self.positions[-1]

    def renamed(self, name: str) -> ArmSegment:
        return ArmSegment(name, self.kind, self.joint_names, self.positions, self.times)

    def reversed(self, name: str | None = None) -> ArmSegment:
        """The same path backwards; a trajectory keeps its timing."""
        times = None if self.times is None else self.times[-1] - self.times[::-1]
        return ArmSegment(name or f"retreat_{self.name}", self.kind, self.joint_names, self.positions[::-1], times)

    def walked_back(
        self, stopped_at, name: str | None = None, *, path_progress: float | None = None
    ) -> ArmSegment | None:
        """Back over the part walked before the arm stopped at ``stopped_at``.

        ``path_progress`` is a trajectory waypoint coordinate reported by an
        executor (0 at the first waypoint, 1 at the second, and so on). Without
        it, the closest point on the piecewise-linear path is used. A target
        goes back to its start. None when nothing was walked.
        """
        stopped_at = np.asarray(stopped_at, dtype=np.float64)
        name = name or f"retreat_{self.name}_partial"
        if self.kind is SegmentKind.TARGET:
            if np.allclose(stopped_at, self.start):
                return None
            return ArmSegment(name, SegmentKind.TARGET, self.joint_names, np.stack([stopped_at, self.start]))
        if path_progress is None:
            vectors = np.diff(self.positions, axis=0)
            squared = np.sum(vectors * vectors, axis=1)
            offsets = stopped_at - self.positions[:-1]
            fractions = np.divide(
                np.sum(offsets * vectors, axis=1),
                squared,
                out=np.zeros_like(squared),
                where=squared > 0,
            )
            fractions = np.clip(fractions, 0.0, 1.0)
            projected = self.positions[:-1] + fractions[:, None] * vectors
            interval = int(np.argmin(np.max(np.abs(projected - stopped_at), axis=1)))
            path_progress = interval + float(fractions[interval])
        if not np.isfinite(path_progress) or not 0 <= path_progress <= len(self.positions) - 1:
            raise ValueError("path_progress must lie on the trajectory")
        if path_progress <= np.finfo(float).eps:
            return None
        index = min(int(np.floor(path_progress)), len(self.positions) - 1)
        fraction = float(path_progress - index)
        if fraction > np.finfo(float).eps:
            elapsed = self.times[index] + fraction * (self.times[index + 1] - self.times[index])
            positions = np.vstack([self.positions[: index + 1], stopped_at])
            times = np.concatenate([self.times[: index + 1], [elapsed]])
        else:
            positions = np.vstack([self.positions[:index], stopped_at])
            times = np.concatenate([self.times[:index], [self.times[index]]])
        return ArmSegment(self.name, self.kind, self.joint_names, positions, times).reversed(name)


class EffectorAction(enum.Enum):
    #: Shape the end effector for the approach (open the gripper or hand).
    PRESHAPE = "preshape"
    #: Close until the object is held, then say whether it is.
    CLOSE = "close"
    #: Let go and open.
    RELEASE = "release"


@dataclass(frozen=True)
class Move:
    """Run an arm segment. ``refine_after`` asks the planner to revise the rest
    of the plan once it has run (e.g. realign from the measured pose)."""

    segment: ArmSegment
    refine_after: bool = False

    @property
    def name(self) -> str:
        return self.segment.name


@dataclass(frozen=True)
class Effect:
    action: EffectorAction

    @property
    def name(self) -> str:
        return self.action.value


@dataclass(frozen=True)
class Hold:
    """Keep the arm and end effector where they are for ``seconds``."""

    seconds: float
    name: str = "hold"

    def __post_init__(self):
        if not np.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("a hold needs a positive finite duration")


Step = Move | Effect | Hold


@dataclass(frozen=True)
class GraspPlan:
    """The steps that take the arm from where it is to the object held.

    ``release`` is what ``release_after_success`` runs after the forward steps;
    None puts the object back where it came from by retracing them. The first
    move's start is the arm's home: a retreat ends there.
    """

    steps: tuple[Step, ...]
    candidate_index: int
    release: tuple[Step, ...] | None = None
    report: Mapping = field(default_factory=dict)

    def __post_init__(self):
        steps = tuple(self.steps)
        if not steps or not any(isinstance(step, Move) for step in steps):
            raise ValueError("a grasp plan moves the arm at least once")
        if sum(isinstance(s, Effect) and s.action is EffectorAction.CLOSE for s in steps) != 1:
            raise ValueError("a grasp plan closes exactly once")
        object.__setattr__(self, "steps", steps)
        if self.release is not None and any(
            isinstance(s, Effect) and s.action is EffectorAction.CLOSE for s in self.release
        ):
            raise ValueError("a release never closes the end effector")
        if self.release is not None:
            object.__setattr__(self, "release", tuple(self.release))


class MotionState(enum.Enum):
    SUCCEEDED = "succeeded"
    #: Stopped on request between frames; the arm holds its last command and the
    #: executor can still move it (a retreat follows).
    HALTED = "halted"
    #: Stopped by the robot or the executor (tracking, timing, health); the arm
    #: is held in place and nothing more is sent.
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class MotionOutcome:
    state: MotionState
    reason: str = ""
    #: Where the arm was when the segment ended, in the segment's joint order.
    stopped_at: np.ndarray | None = None
    report: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class EffectorOutcome:
    """``held`` answers CLOSE: True gripped, False nothing in hand, None unknown."""

    succeeded: bool
    reason: str = ""
    held: bool | None = None
    halted: bool = False
    confidence: float = 0.0
    report: Mapping = field(default_factory=dict)


class ObservationSource(Protocol):
    def capture(self, *, deadline: float, cancelled: Flag) -> Observation: ...


class GraspProposer(Protocol):
    def propose(self, observation: Observation, query: str, *, deadline: float, cancelled: Flag) -> Proposals: ...


class GraspPlanner(Protocol):
    def plan(self, observation: Observation, proposals: Proposals, *, deadline: float, cancelled: Flag) -> GraspPlan:
        """Choose a candidate and plan it from the arm's current state.

        Raises ``GraspFailure`` (retryable when another observation may help).
        The planner owns the candidate search: the order it tries candidates in,
        goalsets, IK branches and wrist checks are its own.
        """
        ...

    def refine(self, plan: GraspPlan, completed: int, *, deadline: float, cancelled: Flag) -> GraspPlan:
        """Revise the steps after ``plan.steps[completed]`` from what is measured now.

        Called only after a ``Move`` with ``refine_after``; the returned plan keeps
        ``plan.steps[: completed + 1]`` as the same step objects. Raise
        ``GraspFailure`` when the plan cannot continue: the steps run so far
        are then taken back.
        """
        ...


class MotionExecutor(Protocol):
    """Moves one robot's arm through plan segments.

    Optional: ``return_home(home_segment, *, cancelled) -> MotionOutcome | None``
    decides where a retreat leaves the arm (None: the sequence's default, the
    plan's start); ``finish()`` ends the executor's hold on the robot after a goal.
    """

    def execute(self, segment: ArmSegment, *, carrying: bool, halted: Flag, cancelled: Flag) -> MotionOutcome:
        """Run one segment. Poll ``halted`` between commands and return HALTED,
        holding the last command, as soon as it is true; ``carrying`` says the
        end effector holds the object. A ``TARGET`` segment is reached by the
        robot's own means (a motion server, or a timed joint line). Report every
        failure as an outcome; cancellation must really stop the arm."""
        ...

    def hold(self, seconds: float, *, carrying: bool, cancelled: Flag) -> MotionOutcome: ...

    def measured(self, joint_names: Sequence[str]) -> np.ndarray: ...

    def stop(self, reason: str) -> None:
        """Stop in place and send nothing more; used after a failure."""
        ...


class EndEffector(Protocol):
    """Optional: ``finish()`` ends its hold on the robot after a goal.

    Optional: ``holding()`` tells, while an object is carried, whether the grip
    still holds it: '' yes, a reason once it was lost, None unknown. The
    sequence asks during every hold that carries the object.
    """

    def act(self, action: EffectorAction, *, halted: Flag, cancelled: Flag) -> EffectorOutcome: ...


class TargetWatch(Protocol):
    """Decides, while the arm moves toward the target, whether the target moved.

    A watch may also define ``phase(name, event)``: the sequence's steps as they
    start and end (event "start" or "end"), e.g. to look only while the arm
    approaches.

    A watch may also define ``failed() -> str | None``: a reason once it can no
    longer tell whether the target moved (it stopped observing, or could not
    decide for too long). That is not a move: nothing says where the target is,
    so the sequence stops the arm in place as after any execution failure
    (``WATCH_FAILED``), with no retreat, no new plan and no release. It is asked
    only when ``moved`` is, while nothing is held.

    ``arm`` raises ``GraspFailure`` when the watch cannot start and execution must
    not begin without it; the goal then fails before anything moves.
    """

    def arm(self, observation: Observation, proposals: Proposals, plan: GraspPlan) -> None: ...

    def moved(self) -> str | None:
        """A reason once the target has clearly moved, else None."""
        ...

    def disarm(self) -> Mapping: ...


@dataclass(frozen=True)
class GraspBackends:
    """One robot's implementations. ``watch`` is optional.

    ``prepare``, when given, warms the robot's backends up (loads planners,
    runs the models once) before the first goal: it is called again until it
    returns without raising, and goals are refused until then.
    """

    observation: ObservationSource
    proposer: GraspProposer
    planner: GraspPlanner
    executor: MotionExecutor
    effector: EndEffector
    watch: TargetWatch | None = None
    close: Callable[[], None] = lambda: None
    prepare: Callable[[], None] | None = None
