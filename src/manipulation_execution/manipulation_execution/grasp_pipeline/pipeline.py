"""One ``PickObject`` goal for any robot: observe, propose, plan, run.

The pipeline owns the order of work, the goal's budget and the retries; a
robot's backends own everything that depends on its body. A planning failure
that another view may fix observes again while attempts remain. A plan the
target watch abandoned is retraced to home and planned again from a fresh
observation, at most ``max_moved_replans`` times; those replans do not use up
planning attempts.

The ROS node that serves the action maps goals onto ``run`` and outcomes onto
the result; nothing here imports ROS.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .contracts import Flag, GraspBackends, GraspFailure, GraspPlan, Observation, Proposals
from .sequence import GraspSequence, PhaseCallback


class PickMode(enum.Enum):
    EXECUTE = "execute"
    PLAN_ONLY = "plan_only"
    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True)
class PipelineConfig:
    #: Observations a goal may plan from when planning finds nothing executable.
    max_attempts: int = 2
    #: Plans abandoned because the target moved that may be planned again.
    max_moved_replans: int = 1
    #: How far from home a retreat may end before one corrective move.
    home_tolerance_rad: float = 0.02

    def __post_init__(self):
        for name in ("max_attempts", "max_moved_replans"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (1 if name == "max_attempts" else 0):
                raise ValueError(f"{name} must be a non-negative integer (max_attempts at least 1)")


@dataclass
class PickOutcome:
    success: bool = False
    error_code: str = ""
    message: str = ""
    attempts: int = 0
    completed_phases: list[str] = field(default_factory=list)
    candidate_index: int = -1
    #: CLOSE's answer on the last executed plan: True, False or None (unknown / not run).
    held: bool | None = None
    held_confidence: float = 0.0
    #: The arm ended at rest under the executor's control (False: stopped in place).
    at_rest: bool = True
    report: dict = field(default_factory=dict)


class GraspPipeline:
    def __init__(
        self,
        backends: GraspBackends,
        config: PipelineConfig = PipelineConfig(),
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._backends = backends
        self._config = config
        self._clock = clock

    def run(
        self,
        query: str,
        *,
        mode: PickMode,
        deadline: float,
        cancelled: Flag,
        on_phase: PhaseCallback = lambda name, event, detail: None,
        release_after_success: bool = False,
    ) -> PickOutcome:
        """``deadline`` is on this pipeline's clock; ``cancelled`` is the goal's own flag."""
        outcome = PickOutcome()
        if not str(query).strip():
            return self._fail(outcome, "INVALID_TARGET", "empty target query")

        def stop() -> bool:
            return cancelled() or self._clock() >= deadline

        def phase(name: str, event: str, detail: dict) -> None:
            # Steps report their state; a phase counts as completed only when it succeeded.
            if event == "end" and detail.get("state", "succeeded") == "succeeded":
                outcome.completed_phases.append(name)
            on_phase(name, event, detail)

        backends = self._backends
        planning_failures, replans = 0, 0
        try:
            while True:
                outcome.attempts += 1
                budget = {"deadline": deadline, "cancelled": stop}
                steps: dict[str, float] = {}
                observation = self._step("observe", phase, steps, backends.observation.capture, **budget)
                self._check_budget(cancelled, deadline)
                if mode is PickMode.OBSERVE_ONLY:
                    outcome.success = True
                    return outcome
                try:
                    proposals = self._step(
                        "perceive", phase, steps, backends.proposer.propose, observation, query, **budget
                    )
                    self._check_budget(cancelled, deadline)
                    plan = self._step("plan", phase, steps, backends.planner.plan, observation, proposals, **budget)
                    self._check_budget(cancelled, deadline)
                except GraspFailure as failure:
                    planning_failures += 1
                    self._check_budget(cancelled, deadline)
                    if failure.retryable and planning_failures < self._config.max_attempts:
                        outcome.report.setdefault("planning_failures", []).append(
                            {"code": failure.code, "message": str(failure)}
                        )
                        continue
                    raise
                outcome.candidate_index = plan.candidate_index
                outcome.report["steps_s"] = dict(steps)
                outcome.report["perceive"] = dict(proposals.report)
                outcome.report["plan"] = dict(plan.report)
                if mode is PickMode.PLAN_ONLY:
                    outcome.success = True
                    return outcome
                # Once the arm moves, only the goal's own cancel stops it in place:
                # the deadline gates observing, planning and starting to execute,
                # but stopping on it mid-sequence would leave an object lifted in
                # the air. A caller that must stop at its deadline cancels the goal.
                result = self._execute(
                    observation, proposals, plan, deadline, cancelled, phase, release_after_success, outcome
                )
                if (
                    result.retreated is not None
                    and result.at_rest
                    and replans < self._config.max_moved_replans
                    and not stop()
                ):
                    replans += 1
                    outcome.report.setdefault("retreats", []).append(result.retreated)
                    continue
                if result.retreated is not None:
                    outcome.report.setdefault("retreats", []).append(result.retreated)
                outcome.success = result.succeeded
                if not result.succeeded:
                    code = result.error_code
                    if code in ("CANCELED", "MOTION_FAILED") and not cancelled() and self._clock() >= deadline:
                        code = "SKILL_TIMEOUT"
                    elif code == "MOTION_FAILED" and cancelled():
                        code = "CANCELED"
                    outcome.error_code, outcome.message = code, result.message
                return outcome
        except GraspFailure as failure:
            code = failure.code
            if cancelled():
                code = "CANCELED"
            elif self._clock() >= deadline:
                code = "SKILL_TIMEOUT"
            return self._fail(outcome, code, str(failure))
        except Exception as exc:  # noqa: BLE001 - every failure ends the goal with a code
            return self._fail(outcome, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")

    def _execute(
        self, observation: Observation, proposals: Proposals, plan: GraspPlan, deadline, stop, phase, release, outcome
    ):
        watch = self._backends.watch
        watched = getattr(watch, "phase", None)

        def on_phase(name: str, event: str, detail: dict) -> None:
            phase(name, event, detail)
            if watched is not None:
                watched(name, event)

        sequence = GraspSequence(self._backends, home_tolerance_rad=self._config.home_tolerance_rad, on_phase=on_phase)
        if watch is not None:
            watch.arm(observation, proposals, plan)
        try:
            result = sequence.run(plan, deadline=deadline, cancelled=stop, release_after_success=release)
        finally:
            if watch is not None:
                outcome.report.setdefault("watch", []).append(dict(watch.disarm()))
        outcome.held, outcome.held_confidence, outcome.at_rest = result.held, result.held_confidence, result.at_rest
        outcome.report.setdefault("executions", []).append({"phases": result.phases, "retreated": result.retreated})
        return result

    def _step(self, name: str, phase: PhaseCallback, timings: dict, call, *args, **kwargs):
        phase(name, "start", {})
        started = self._clock()
        value = call(*args, **kwargs)
        timings[name] = self._clock() - started
        phase(name, "end", {"seconds": timings[name]})
        return value

    def _check_budget(self, cancelled: Flag, deadline: float) -> None:
        if cancelled():
            raise GraspFailure("CANCELED", "goal cancelled")
        if self._clock() >= deadline:
            raise GraspFailure("SKILL_TIMEOUT", "goal deadline passed")

    @staticmethod
    def _fail(outcome: PickOutcome, code: str, message: str) -> PickOutcome:
        outcome.success, outcome.error_code, outcome.message = False, code, message
        return outcome
