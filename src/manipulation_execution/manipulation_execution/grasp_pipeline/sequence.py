"""Run a grasp plan step by step, and take it back when it must be abandoned.

Forward steps run in order. While the arm moves toward the target and nothing is
held yet, the target watch may halt it between commands; the arm then retraces
what it walked of that segment, and every completed step is undone in reverse:
moves are retraced, a close is released. Nothing gripped after the close is
undone the same way. A retreat ends at the plan's home (the first move's start),
corrected with one more move if the retrace landed off it.

A failure the executor or end effector reports (tracking, timing, health,
cancellation) stops the arm in place and sends nothing more: what to do next is
the operator's call, not a guess made from stale state. So does a target watch
that can no longer tell whether the target moved (``TargetWatch.failed``):
retracing or releasing without knowing where the target is would be that guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    EffectorOutcome,
    Flag,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Hold,
    MotionOutcome,
    MotionState,
    Move,
    SegmentKind,
    Step,
)

#: on_phase(name, event, detail); event is "start" or "end".
PhaseCallback = Callable[[str, str, dict], None]


#: How long a halted arm settles on its last command before its pose is measured for the retreat.
HALT_SETTLE_S = 0.3


def _never() -> bool:
    return False


@dataclass
class SequenceResult:
    #: The forward steps ran and the object is held (and released, when asked).
    succeeded: bool = False
    error_code: str = ""
    message: str = ""
    #: CLOSE's answer: True gripped, False nothing in hand, None unknown.
    held: bool | None = None
    held_confidence: float = 0.0
    #: Set when the target watch halted the plan: {"step", "reason"}.
    retreated: dict | None = None
    #: The arm ended under the executor's control at rest (False: stopped in place).
    at_rest: bool = True
    #: The requested post-pick release and return completed.
    released_after_success: bool = False
    completed: list[str] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)


class _Stopped(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GraspSequence:
    """Drives one robot's executor and end effector through a plan."""

    def __init__(
        self,
        backends: GraspBackends,
        *,
        home_tolerance_rad: float = 0.02,
        on_phase: PhaseCallback = lambda name, event, detail: None,
    ):
        if not np.isfinite(home_tolerance_rad) or home_tolerance_rad <= 0:
            raise ValueError("home_tolerance_rad must be positive")
        self._backends = backends
        self._home_tolerance = float(home_tolerance_rad)
        self._on_phase = on_phase

    def run(
        self, plan: GraspPlan, *, deadline: float, cancelled: Flag, release_after_success: bool = False
    ) -> SequenceResult:
        result = SequenceResult()
        done: list[Step] = []
        state = {"carrying": False, "moved": None, "failed": None}
        watch = self._backends.watch
        failed = getattr(watch, "failed", None)

        def halted() -> bool:
            # Only while moving toward the target with nothing held, and only once.
            if state["carrying"] or watch is None:
                return False
            if state["moved"] is None and state["failed"] is None:
                state["moved"] = watch.moved()
                if state["moved"] is None and failed is not None:
                    state["failed"] = failed()
            return state["moved"] is not None or state["failed"] is not None

        def blind(step: str) -> None:
            # Not a move: nothing says where the target is, so the arm stays where it halted.
            if state["failed"] is not None:
                raise _Stopped("WATCH_FAILED", f"target watch failed during {step}: {state['failed']}")

        try:
            index, abandoned = 0, False
            while index < len(plan.steps):
                step = plan.steps[index]
                if isinstance(step, Move):
                    outcome = self._move(step.segment, state["carrying"], halted, cancelled, result)
                    if outcome.state is MotionState.HALTED:
                        blind(step.name)
                        result.retreated = {"step": step.name, "reason": str(state["moved"])}
                        back = step.segment.walked_back(
                            self._stopped_at(step.segment, cancelled),
                            path_progress=outcome.report.get("path_progress"),
                        )
                        if back is not None:
                            self._expect(self._move(back, False, _never, cancelled, result))
                        abandoned = True
                        break
                    self._expect(outcome)
                    done.append(step)
                    if step.refine_after:
                        plan = self._refine(plan, index, deadline, cancelled)
                elif isinstance(step, Effect):
                    watched = halted if step.action is EffectorAction.CLOSE else _never
                    outcome = self._act(step.action, watched, cancelled, result)
                    if step.action is EffectorAction.CLOSE:
                        done.append(step)
                        if outcome.halted:
                            blind(step.name)
                            result.retreated = {"step": step.name, "reason": str(state["moved"])}
                            abandoned = True
                            break
                        self._expect_effect(outcome)
                        result.held, result.held_confidence = outcome.held, outcome.confidence
                        if outcome.held is False:
                            abandoned = True
                            break
                        state["carrying"] = True
                    else:
                        self._expect_effect(outcome)
                        done.append(step)
                elif isinstance(step, Hold):
                    self._expect(self._hold(step, state["carrying"], cancelled, result))
                    done.append(step)
                index += 1
            if not abandoned:
                if release_after_success:
                    self._release(plan, done, state, cancelled, result)
                    result.released_after_success = not state["carrying"]
                result.succeeded = True
                return result
            self._undo(plan, done, state, cancelled, result)
            if result.retreated is not None:
                result.error_code, result.message = "TARGET_MOVED", f"target moved: {result.retreated['reason']}"
            else:
                result.error_code, result.message = "GRASP_VERIFICATION_FAILED", "nothing held after closing"
            return result
        except GraspFailure as failure:
            # A refinement that cannot continue: undo from where the arm rests.
            result.error_code, result.message = failure.code, str(failure)
            try:
                self._undo(plan, done, state, cancelled, result)
            except _Stopped as stopped:
                self._stop(result, stopped.code, f"{failure}; then {stopped}")
            except Exception as exc:  # noqa: BLE001 - stop in place and say why
                self._stop(result, "INTERNAL_ERROR", f"{failure}; then {type(exc).__name__}: {exc}")
            return result
        except _Stopped as stopped:
            self._stop(result, stopped.code, str(stopped))
            return result
        except Exception as exc:  # noqa: BLE001 - stop in place and say why
            self._stop(result, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
            return result

    # -- steps -----------------------------------------------------------------

    def _phase(self, result: SequenceResult, name: str, outcome: MotionOutcome | EffectorOutcome, **detail) -> None:
        if isinstance(outcome, MotionOutcome):
            ok, state = outcome.state is MotionState.SUCCEEDED, outcome.state.value
        else:
            ok = outcome.succeeded and not outcome.halted
            state = "halted" if outcome.halted else "succeeded" if outcome.succeeded else "failed"
        row = {"name": name, "state": state, "reason": outcome.reason, **detail}
        row.update({k: v for k, v in dict(outcome.report).items() if k not in row})
        result.phases.append(row)
        if ok:
            result.completed.append(name)
        self._on_phase(name, "end", row)

    def _move(self, segment: ArmSegment, carrying: bool, halted: Flag, cancelled: Flag, result) -> MotionOutcome:
        self._on_phase(segment.name, "start", {"kind": segment.kind.value, "carrying": carrying})
        outcome = self._backends.executor.execute(segment, carrying=carrying, halted=halted, cancelled=cancelled)
        self._phase(result, segment.name, outcome, carrying=carrying)
        return outcome

    def _hold(self, step: Hold, carrying: bool, cancelled: Flag, result) -> MotionOutcome:
        self._on_phase(step.name, "start", {"seconds": step.seconds})
        outcome = self._backends.executor.hold(step.seconds, carrying=carrying, cancelled=cancelled)
        self._phase(result, step.name, outcome)
        return outcome

    def _act(self, action: EffectorAction, halted: Flag, cancelled: Flag, result) -> EffectorOutcome:
        """Run one end effector action with the arm supervised meanwhile, where the executor offers it.

        The arm is not commanded while the hand acts, so a hand that drags it, a
        locked joint that moves or a controller that drops its target would
        otherwise go unseen until the next move (or never, after the last one).
        A failed supervision check stops the action as a cancel would and the
        sequence stops as for a failed move; a cancelled action is CANCELED,
        not a broken end effector.
        """
        self._on_phase(action.value, "start", {})
        supervise = getattr(self._backends.executor, "supervise", None)
        check = supervise() if supervise is not None else (lambda: "")
        fault: list[str] = []

        def stopped() -> bool:
            if cancelled():
                return True
            reason = check()
            if reason:
                fault.append(reason)
            return bool(reason)

        outcome = self._backends.effector.act(action, halted=halted, cancelled=stopped)
        self._phase(result, action.value, outcome, held=outcome.held, halted=outcome.halted)
        if fault:
            raise _Stopped("MOTION_FAILED", f"arm during {action.value}: {fault[0]}")
        if not outcome.succeeded and cancelled():
            raise _Stopped("CANCELED", outcome.reason or f"cancelled during {action.value}")
        return outcome

    def _refine(self, plan: GraspPlan, index: int, deadline: float, cancelled: Flag) -> GraspPlan:
        refined = self._backends.planner.refine(plan, index, deadline=deadline, cancelled=cancelled)
        kept = refined.steps[: index + 1]
        if len(kept) != index + 1 or any(a is not b for a, b in zip(kept, plan.steps, strict=False)):
            raise _Stopped("INTERNAL_ERROR", "refine changed steps that already ran")
        return refined

    @staticmethod
    def _expect(outcome: MotionOutcome) -> None:
        if outcome.state is MotionState.SUCCEEDED:
            return
        code = "CANCELED" if outcome.state is MotionState.CANCELLED else "MOTION_FAILED"
        raise _Stopped(code, outcome.reason or outcome.state.value)

    @staticmethod
    def _expect_effect(outcome: EffectorOutcome) -> None:
        if not outcome.succeeded or outcome.halted:
            raise _Stopped("EFFECTOR_FAILED", outcome.reason or "end effector did not finish")

    def _stopped_at(self, segment: ArmSegment, cancelled: Flag) -> np.ndarray:
        """Where a halted arm came to rest: let it settle on its last command, then measure it fresh.

        The pose at the halt is still moving toward that command (and may be a
        stale sample); a retreat walked back from it starts behind the arm.
        """
        executor = self._backends.executor
        self._expect(executor.hold(HALT_SETTLE_S, carrying=False, cancelled=cancelled))
        return executor.measured(segment.joint_names)

    # -- taking the plan back ----------------------------------------------------

    def _undo(self, plan: GraspPlan, done: list[Step], state: dict, cancelled: Flag, result) -> None:
        for step in reversed(done):
            if isinstance(step, Move):
                self._expect(self._move(step.segment.reversed(), state["carrying"], _never, cancelled, result))
            elif isinstance(step, Effect) and step.action is EffectorAction.CLOSE:
                self._expect_effect(self._act(EffectorAction.RELEASE, _never, cancelled, result))
                state["carrying"] = False
        done.clear()
        self._return_home(plan, cancelled, result)

    def _release(self, plan: GraspPlan, done: list[Step], state: dict, cancelled: Flag, result) -> None:
        if plan.release is None:
            self._undo(plan, done, state, cancelled, result)
            return
        for step in plan.release:
            if isinstance(step, Move):
                self._expect(self._move(step.segment, state["carrying"], _never, cancelled, result))
            elif isinstance(step, Effect):
                self._expect_effect(self._act(step.action, _never, cancelled, result))
                if step.action is EffectorAction.RELEASE:
                    state["carrying"] = False
            else:
                self._expect(self._hold(step, state["carrying"], cancelled, result))

    def _return_home(self, plan: GraspPlan, cancelled: Flag, result) -> None:
        home = next(step.segment for step in plan.steps if isinstance(step, Move))
        executor = self._backends.executor
        # An executor may know better where to leave the arm (e.g. a fixed controller
        # command, so that landings do not ratchet); None leaves it to the default.
        custom = getattr(executor, "return_home", None)
        if custom is not None:
            self._on_phase("home", "start", {})
            outcome = custom(home, cancelled=cancelled)
            if outcome is not None:
                self._phase(result, "home", outcome)
                self._expect(outcome)
                return
        here = executor.measured(home.joint_names)
        if float(np.max(np.abs(here - home.start))) <= self._home_tolerance:
            return
        correction = ArmSegment("home_correction", SegmentKind.TARGET, home.joint_names, np.stack([here, home.start]))
        self._expect(self._move(correction, False, _never, cancelled, result))

    def _stop(self, result: SequenceResult, code: str, message: str) -> None:
        result.succeeded, result.at_rest = False, False
        result.error_code, result.message = code, message
        try:
            self._backends.executor.stop(message)
        except Exception as exc:  # noqa: BLE001 - the first failure is the one to report
            result.message = f"{message}; stop failed: {type(exc).__name__}: {exc}"
