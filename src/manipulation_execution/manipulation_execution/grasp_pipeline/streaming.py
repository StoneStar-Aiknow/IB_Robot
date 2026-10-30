"""Stream an arm through a plan, one position command per control period.

For a robot whose runtime accepts joint position streams (joint.position_stream)
and holds the last command when the stream stops. Each segment is followed at
``rate_hz``; between commands the executor checks feedback age, the control
period, the arrival deadline, tracking against the last command, and that the
joints locked into the planner's model have not moved. ``halted`` stops a segment
between commands with the last command standing: the arm holds there and the
executor can still move it (a retreat).

A position-controlled arm settles away from its command under load. Where the
robot reports what its controller is commanding (``StreamIO.command``), the
executor starts from that target, carries the controller's offset from the
measured pose through the segment, and checks every period that the controller
is driving one of the targets it sent: a platform that clamps or replaces a
target stops the segment. Without a report it commands the measured pose plus
the shaper's correction.

The robot supplies the I/O (a ROS binding of its interfaces, or a test double)
and, optionally, a ``CommandShaper`` that adds a load correction along the path
(e.g. gravity sag).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .contracts import ArmSegment, Flag, MotionOutcome, MotionState, SegmentKind

#: How far back a command report that matches no recent target is looked up, to say how stale it is.
MISMATCH_LOOKBACK_S = 2.0


def received_at_or_stale(received: float | None) -> float:
    """A receive time the middleware did not record becomes infinitely old.

    :func:`robot_runtime.on_demand.received_monotonic` returns ``None`` when the
    middleware wrote no receive time (CycloneDDS does not). Such a sample may
    have waited in the reader queue for seconds, so it cannot be dated and must
    not pass a freshness gate; every gate here compares ``received_at`` against
    a limit, and ``-inf`` is already how a report that never arrived is dated.
    """
    return float("-inf") if received is None else received


@dataclass(frozen=True)
class JointSample:
    """Measured joint positions. ``received_at`` is on the executor's clock."""

    received_at: float
    positions: Mapping[str, float]
    healthy: bool = True


@dataclass(frozen=True)
class CommandSample:
    """What the robot's joint controller is commanding, as it reports it."""

    received_at: float
    positions: Mapping[str, float]
    stiffness: Mapping[str, float] = field(default_factory=dict)
    damping: Mapping[str, float] = field(default_factory=dict)


class StreamIO(Protocol):
    #: Every joint the stream carries, in the order a command lists them.
    channel_joints: tuple[str, ...]

    def feedback(self) -> JointSample: ...

    def command(self) -> CommandSample | None:
        """The controller's current target, or None when the robot reports none."""
        ...

    def publish(self, positions: Sequence[float]) -> None:
        """Send one full command and keep it alive (re-sent) until the next or release()."""
        ...

    def release(self) -> None:
        """Stop sending; the robot holds the last command."""
        ...

    def refusal(self) -> str | None:
        """Why the runtime does not admit output now (mode, stop latch, fault), or None."""
        ...

    def monotonic(self) -> float: ...

    def wait(self, seconds: float) -> None: ...


class CommandShaper(Protocol):
    def raise_for(self, joint_names: Sequence[str], positions: np.ndarray) -> np.ndarray:
        """Per waypoint, how far above the planned position to command (NxJ, rad)."""
        ...


@dataclass(frozen=True)
class StreamConfig:
    rate_hz: float = 50.0
    feedback_max_age_s: float = 0.1
    # Measured arm against the last command, every period. Under load the command
    # leads the arm by the whole settling offset, so this covers it plus lag.
    tracking_error_rad: float = 0.15
    # Two different questions, deliberately two tolerances: the start of a plan and
    # the joints locked into the planner's model must stay tight (model validity);
    # arrival only asks whether the arm finished, which has to absorb steady-state
    # droop in an extended posture.
    arrived_error_rad: float = 0.025
    arrival_error_rad: float = 0.1
    arrived_hold_s: float = 0.2
    timeout_margin_s: float = 3.0
    max_velocity_rad_s: float = 0.6
    # Consecutive commands stay closer than this; a stalled loop is caught within it.
    max_control_gap_s: float = 0.10
    #: Joints that must stay where they were at a segment's start (locked into the planner's model).
    locked_joints: tuple[str, ...] = ()
    #: Command limits per joint; commands stay ``limit_margin_rad`` inside.
    limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    limit_margin_rad: float = 0.005
    #: With a command report: the gains the controller must drive the joints with.
    expected_stiffness: float | None = None
    expected_damping: float | None = None
    #: With a command report: how close it must be to a target sent within apply_s.
    command_match_rad: float = 0.001
    command_apply_s: float = 0.2
    #: A TARGET segment is reached by a straight joint line at this speed, at least min_s.
    target_speed_rad_s: float = 0.05
    target_min_s: float = 1.0
    #: Where a sequence leaves the controller's command (return_home), and the band
    #: inside which it is left alone (tolerance) or not touched at all (max: not a correction).
    home_command: Mapping[str, float] | None = None
    home_tolerance_rad: float = 0.005
    home_max_rad: float = 0.1

    def __post_init__(self):
        for name in (
            "rate_hz",
            "feedback_max_age_s",
            "tracking_error_rad",
            "arrived_error_rad",
            "arrival_error_rad",
            "max_velocity_rad_s",
            "max_control_gap_s",
            "command_apply_s",
            "target_speed_rad_s",
            "target_min_s",
            "home_tolerance_rad",
            "home_max_rad",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for joint, (lower, upper) in self.limits.items():
            if not lower + 2 * self.limit_margin_rad < upper:
                raise ValueError(f"limits.{joint} leaves no room inside the margin")


class StreamFailure(RuntimeError):
    pass


class StreamExecutor:
    """A ``MotionExecutor`` over one joint position stream."""

    def __init__(self, io: StreamIO, config: StreamConfig = StreamConfig(), shaper: CommandShaper | None = None):
        self.io = io
        self.config = config
        self.shaper = shaper
        self._channel = tuple(io.channel_joints)
        self._last: np.ndarray | None = None  # last full channel command sent
        # Commands sent within the last MISMATCH_LOOKBACK_S; a report matches the command_apply_s tail of it.
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._initial: np.ndarray | None = None  # the controller's target at takeover
        self._first_at: float | None = None  # when this executor first sent
        self._stopped = ""
        self.report: dict = {"segments": []}

    # -- MotionExecutor -------------------------------------------------------------

    def execute(self, segment: ArmSegment, *, carrying: bool, halted: Flag, cancelled: Flag) -> MotionOutcome:
        names = segment.joint_names
        try:
            self._admit()
            if not set(names) <= set(self._channel):
                raise StreamFailure(f"segment joints {sorted(set(names) - set(self._channel))} are not on the stream")
            sample = self._fresh(self.io.feedback())
            here = self._vector(sample, names)
            timed = self._timed(segment, here)
            raised = self._raise(timed)
            offset, held = self._baseline(sample, names)
        except StreamFailure as exc:
            return MotionOutcome(MotionState.FAILED, str(exc))
        return self._follow(segment.name, names, timed, raised, offset, held, halted, cancelled)

    def hold(self, seconds: float, *, carrying: bool, cancelled: Flag) -> MotionOutcome:
        """Keep the last command for ``seconds``, supervised as a segment is (tracking, locked joints, board)."""
        ended = self.io.monotonic() + seconds
        try:
            self._admit()
            sample = self._fresh(self.io.feedback())
            if self._last is None:
                offset, held = self._baseline(sample, ())
                self._send(held)
            check = self._supervision(sample)
            while self.io.monotonic() < ended:
                if cancelled():
                    return MotionOutcome(MotionState.CANCELLED, "cancelled")
                check()
                self.io.wait(min(1.0 / self.config.rate_hz, max(0.0, ended - self.io.monotonic())))
        except StreamFailure as exc:
            return MotionOutcome(MotionState.FAILED, str(exc))
        return MotionOutcome(MotionState.SUCCEEDED)

    def supervise(self) -> Callable[[], str]:
        """A check for while something else acts (an end effector): '' while the arm holds, else why not.

        The arm is not commanded meanwhile; the last command stands. Each call
        checks what a hold checks, so a hand that drags the arm, a locked joint
        that moves or a controller that drops the target is seen while it acts.
        """
        try:
            self._admit()
            check = self._supervision(self._fresh(self.io.feedback()))
        except StreamFailure as exc:
            reason = str(exc)  # exc itself is gone once this block ends
            return lambda: reason

        def checked() -> str:
            try:
                check()
            except StreamFailure as exc:
                return str(exc)
            return ""

        return checked

    def _supervision(self, start: JointSample) -> Callable[[], None]:
        """Raises StreamFailure when the arm leaves the last command or a locked joint moves from ``start``."""
        config = self.config
        locked = [joint for joint in config.locked_joints if joint in start.positions]
        locked_start = self._vector(start, locked) if locked else np.zeros(0)
        driven = [joint for joint in self._channel if joint not in config.locked_joints]
        index = [self._channel.index(joint) for joint in driven]

        def check() -> None:
            self._admit()
            sample = self._fresh(self.io.feedback())
            if (
                self._last is not None
                and driven
                and np.max(np.abs(self._vector(sample, driven) - self._last[index])) > config.tracking_error_rad
            ):
                raise StreamFailure("tracking anomaly while holding")
            if locked and np.max(np.abs(self._vector(sample, locked) - locked_start)) > config.arrived_error_rad:
                raise StreamFailure("a joint locked into the planner's model moved")
            self._check_report()

        return check

    def measured(self, joint_names: Sequence[str]) -> np.ndarray:
        return self._vector(self._fresh(self.io.feedback()), tuple(joint_names))

    def stop(self, reason: str) -> None:
        self._stopped = reason or "stopped"
        self.io.release()

    def finish(self) -> None:
        """End ownership: stop sending, the robot holds the last command.

        A stop latches only for the rest of its goal: the next goal starts with
        a fresh takeover, as each goal had its own controller session before.
        """
        self.io.release()
        self._last = self._initial = self._first_at = None
        self._history.clear()
        self._stopped = ""

    def return_home(self, home: ArmSegment, *, cancelled: Flag) -> MotionOutcome | None:
        """Move the controller's command to ``home_command`` (None: the caller's default).

        Aiming at the measured pose a previous sequence ended at ratchets: under load
        each landing settles a little differently and the next plan starts there.
        A fixed command settles the same way every time. Inside home_tolerance_rad
        nothing moves; beyond home_max_rad it is not a correction and is left alone.
        """
        target = self.config.home_command
        report = None if target is None else self.io.command()
        if report is None:
            return None
        if self.io.monotonic() - report.received_at > self.config.feedback_max_age_s:
            return MotionOutcome(MotionState.FAILED, "the controller's command report is stale")
        names = home.joint_names
        try:
            board = np.array([report.positions[name] for name in names], dtype=float)
            delta = np.array([target[name] for name in names], dtype=float) - board
        except KeyError as exc:
            return MotionOutcome(MotionState.FAILED, f"home command lacks {exc}")
        off = float(np.max(np.abs(delta)))
        self.report["home_command_off_rad"] = off
        if off <= self.config.home_tolerance_rad:
            return MotionOutcome(MotionState.SUCCEEDED, "at the home command")
        if off > self.config.home_max_rad:
            self.report["home_command_skipped"] = board.tolist()
            return MotionOutcome(MotionState.SUCCEEDED, "far from the home command: left alone")
        try:
            here = self.measured(names)
        except StreamFailure as exc:
            return MotionOutcome(MotionState.FAILED, str(exc))
        # The executor carries the controller's offset, so this line ends with the command at home.
        line = ArmSegment("home_command", SegmentKind.TARGET, names, np.stack([here, here + delta]))
        return self.execute(line, carrying=False, halted=lambda: False, cancelled=cancelled)

    # -- one segment ----------------------------------------------------------------

    def _follow(self, name, names, timed, raised, offset, held, halted, cancelled) -> MotionOutcome:
        config, period = self.config, 1.0 / self.config.rate_hz
        index = [self._channel.index(joint) for joint in names]
        locked = [joint for joint in config.locked_joints if joint not in names]
        start_sample = self._fresh(self.io.feedback())
        locked_start = self._vector(start_sample, locked) if locked else np.zeros(0)
        row = {"name": name, "seconds": float(timed.times[-1])}
        self.report["segments"].append(row)
        started = last_tick = self.io.monotonic()
        arrived_since = None
        last_command = None
        try:
            while True:
                tick = self.io.monotonic()
                if cancelled():
                    return self._end(row, MotionState.CANCELLED, "cancelled", names)
                if halted() and last_command is not None:
                    # Nothing more is sent: the last command stands and stays alive.
                    return self._end(row, MotionState.HALTED, "halted", names)
                self._admit()
                sample = self._fresh(self.io.feedback())
                if last_command is not None and not 0 <= tick - last_tick <= config.max_control_gap_s:
                    raise StreamFailure("control loop timing overrun")
                elapsed = tick - started if last_command is not None else 0.0
                if elapsed > timed.times[-1] + config.timeout_margin_s:
                    raise StreamFailure("arrival timeout")
                actual = self._vector(sample, names)
                if last_command is not None and np.max(np.abs(actual - last_command)) > config.tracking_error_rad:
                    raise StreamFailure("tracking anomaly")
                if locked and np.max(np.abs(self._vector(sample, locked) - locked_start)) > config.arrived_error_rad:
                    raise StreamFailure("a joint locked into the planner's model moved")
                self._check_report()
                if last_command is None:
                    started = tick  # the timed path begins with the first command
                command = self._command(timed, raised, offset, names, 0.0 if last_command is None else elapsed)
                full = held.copy()
                full[index] = command
                self._send(full)
                last_command, last_tick = command, tick
                arrived = (
                    elapsed >= timed.times[-1]
                    and np.max(np.abs(actual - timed.positions[-1])) <= config.arrival_error_rad
                )
                arrived_since = (arrived_since if arrived_since is not None else tick) if arrived else None
                if arrived_since is not None and tick - arrived_since >= config.arrived_hold_s:
                    return self._end(row, MotionState.SUCCEEDED, "", names)
                self.io.wait(max(0.0, tick + period - self.io.monotonic()))
        except StreamFailure as exc:
            return self._end(row, MotionState.FAILED, str(exc), names)

    def _end(self, row, state, reason, names) -> MotionOutcome:
        row.update(state=state.value, reason=reason)
        try:
            stopped_at = self._vector(self.io.feedback(), names)
        except (KeyError, StreamFailure):
            stopped_at = None
        return MotionOutcome(state, reason, stopped_at)

    # -- pieces ---------------------------------------------------------------------

    def _admit(self) -> None:
        if self._stopped:
            raise StreamFailure(f"executor stopped: {self._stopped}")
        refusal = self.io.refusal()
        if refusal:
            raise StreamFailure(f"runtime does not admit output: {refusal}")

    def _fresh(self, sample: JointSample) -> JointSample:
        age = self.io.monotonic() - sample.received_at
        if not sample.healthy or not 0 <= age <= self.config.feedback_max_age_s:
            raise StreamFailure("feedback unhealthy or stale")
        return sample

    @staticmethod
    def _vector(sample, names) -> np.ndarray:
        try:
            values = np.array([sample.positions[name] for name in names], dtype=float)
        except KeyError as exc:
            raise StreamFailure(f"no feedback for {exc}") from None
        if not np.isfinite(values).all():
            raise StreamFailure("nonfinite feedback")
        return values

    def _timed(self, segment: ArmSegment, here: np.ndarray) -> ArmSegment:
        """The segment as a timed path from the measured pose, within the velocity limit."""
        config = self.config
        if segment.kind is SegmentKind.TARGET:
            seconds = max(config.target_min_s, float(np.max(np.abs(segment.end - here))) / config.target_speed_rad_s)
            return ArmSegment(
                segment.name,
                SegmentKind.TRAJECTORY,
                segment.joint_names,
                np.stack([here, segment.end]),
                np.array([0.0, seconds]),
            )
        offset = here - segment.start
        if np.max(np.abs(offset)) > config.arrival_error_rad:
            raise StreamFailure(f"{segment.name} starts {np.max(np.abs(offset)):.3f} rad from the arm; replan")
        # Start at the measured arm and fade the offset out, so a segment's error is
        # not carried as a shift into the next one.
        fade = 1.0 - segment.times / segment.times[-1]
        positions = segment.positions + np.outer(fade, offset)
        # Stretch only the intervals faster than the limit: the start, which
        # accelerates from rest, keeps its timing.
        steps = np.diff(segment.times)
        needed = np.max(np.abs(np.diff(positions, axis=0)), axis=1) / config.max_velocity_rad_s * 1.001
        times = np.concatenate(([0.0], np.cumsum(np.maximum(steps, needed))))
        return ArmSegment(segment.name, SegmentKind.TRAJECTORY, segment.joint_names, positions, times)

    def _raise(self, timed: ArmSegment) -> np.ndarray:
        if self.shaper is None:
            return np.zeros_like(timed.positions)
        raised = np.asarray(self.shaper.raise_for(timed.joint_names, timed.positions), dtype=float)
        if raised.shape != timed.positions.shape or not np.isfinite(raised).all():
            raise StreamFailure("the command shaper returned no finite raise per waypoint")
        # The controller's offset at the start already holds the raise there.
        return raised - raised[0]

    def _baseline(self, sample: JointSample, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(offset of the driven joints' command from the measured pose, full command holding the others)."""
        report = self.io.command()
        measured_channel = self._vector(sample, self._channel)
        index = [self._channel.index(joint) for joint in names]
        if report is not None:
            if self.io.monotonic() - report.received_at > self.config.feedback_max_age_s:
                raise StreamFailure("the controller's command report is stale")
            board = self._report_vector(report)
            self._check_gains(report)
            if self._last is None:
                self._initial = board.copy()
            elif not self._matches(board):
                raise StreamFailure(self._mismatch(board))
        # Once this executor has sent, its own last command is the baseline (the
        # report was just checked to match it); before that, the controller's target.
        if self._last is not None:
            base = self._last.copy()
        elif report is not None:
            base = board.copy()
        else:
            base = measured_channel.copy()
        return base[index] - measured_channel[index], base

    def _command(self, timed, raised, offset, names, elapsed) -> np.ndarray:
        raw = np.array([np.interp(elapsed, timed.times, timed.positions[:, j]) for j in range(len(names))])
        lift = np.array([np.interp(elapsed, timed.times, raised[:, j]) for j in range(len(names))])
        command = raw + lift + offset
        limits = self.config.limits
        if not limits:
            return command
        margin = self.config.limit_margin_rad
        lower = np.array([limits.get(name, (-np.inf, np.inf))[0] for name in names])
        upper = np.array([limits.get(name, (-np.inf, np.inf))[1] for name in names])
        clipped = np.clip(command, lower + margin, upper - margin)
        # Where the unraised command already sits inside the margin, add no raise, and
        # never cross the limit itself: the controller would clamp it.
        edge = (raw + offset < lower + margin) | (raw + offset > upper - margin)
        return np.where(edge, np.clip(raw + offset, lower, upper), clipped)

    def _send(self, full: np.ndarray) -> None:
        now = self.io.monotonic()
        self.io.publish(full.tolist())
        if self._first_at is None:
            self._first_at = now
        self._last = full.copy()
        self._history.append((now, full.copy()))
        kept = max(self.config.command_apply_s, MISMATCH_LOOKBACK_S)
        while self._history and now - self._history[0][0] > kept:
            self._history.popleft()

    def _check_report(self) -> None:
        report = self.io.command()
        if report is None or self._last is None:
            return
        if self.io.monotonic() - report.received_at > self.config.feedback_max_age_s:
            raise StreamFailure("the controller's command report is stale")
        self._check_gains(report)
        board = self._report_vector(report)
        if not self._matches(board):
            raise StreamFailure(self._mismatch(board))

    def _report_vector(self, report: CommandSample) -> np.ndarray:
        try:
            board = np.array([report.positions[name] for name in self._channel], dtype=float)
        except KeyError as exc:
            raise StreamFailure(f"the controller's command report lacks {exc}") from None
        if not np.isfinite(board).all():
            raise StreamFailure("nonfinite controller command report")
        return board

    def _check_gains(self, report: CommandSample) -> None:
        for gains, expected, what in (
            (report.stiffness, self.config.expected_stiffness, "stiffness"),
            (report.damping, self.config.expected_damping, "damping"),
        ):
            if expected is None:
                continue
            values = np.array([gains.get(name, np.nan) for name in self._channel], dtype=float)
            if not np.allclose(values, expected, atol=1e-6, rtol=0.0):
                raise StreamFailure(f"the controller's {what} is not {expected:g}; take the gains over first")

    def _mismatch(self, board: np.ndarray) -> str:
        """Why a report is none of the recent targets: its worst joint, and how long ago it was sent, if at all."""
        reference = board if self._last is None else self._last
        worst = int(np.argmax(np.abs(board - reference)))
        detail = f"{self._channel[worst]} driven to {board[worst]:.4f}, last sent {reference[worst]:.4f}"
        now = self.io.monotonic()
        sent = next(
            (
                now - at
                for at, command in reversed(self._history)
                if np.allclose(board, command, atol=self.config.command_match_rad, rtol=0.0)
            ),
            None,
        )
        if sent is None:
            detail += f"; nothing sent in the last {MISMATCH_LOOKBACK_S:g} s matches it"
        else:
            detail += f"; it is what was sent {sent:.3f} s ago"
        return f"the controller is not driving the targets this executor sent ({detail})"

    def _matches(self, board: np.ndarray) -> bool:
        now = self.io.monotonic()
        candidates = [command for at, command in self._history if now - at <= self.config.command_apply_s]
        if self._last is not None:
            candidates.append(self._last)  # between submissions the controller holds the last target
        if (
            self._initial is not None
            and self._first_at is not None
            and now - self._first_at <= self.config.command_apply_s
        ):
            candidates.append(self._initial)
        return any(np.allclose(board, command, atol=self.config.command_match_rad, rtol=0.0) for command in candidates)
