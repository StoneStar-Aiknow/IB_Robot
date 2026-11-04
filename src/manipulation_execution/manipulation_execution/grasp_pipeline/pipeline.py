"""One ``PickObject`` goal for any robot: observe, propose, plan, run.

The pipeline owns the order of work, the goal's budget and the retries; a
robot's backends own everything that depends on its body. An observation or
planning failure that another view may fix (``GraspFailure.retryable``)
observes again while ``max_attempts`` remain; the arm has not moved. A plan the
target watch abandoned is retraced to home and planned again from a fresh
observation, at most ``max_moved_replans`` times. A grasp that held nothing or
lost its grip, with the arm back at home, is tried again from a fresh
observation at most ``max_regrasps`` times. Replans and regrasps do not use up
attempts. Every retry is recorded in ``PickOutcome.retries`` with its reason.

The ROS node that serves the action maps goals onto ``run`` and outcomes onto
the result; nothing here imports ROS.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from embodied_common.execution_detail import ExecutionDetail

from .contracts import Flag, GraspBackends, GraspFailure, GraspPlan, Observation, Proposals
from .outcome_message import failure_action
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
    #: Grasps that held nothing or lost the object, with the arm back at home, that may be tried again.
    max_regrasps: int = 0
    #: Goal time an execution needs: with less left, a plan is not started (the goal's
    #: own timeout would cancel it mid-motion and leave the arm stopped in place).
    execute_reserve_s: float = 0.0
    #: How far from home a retreat may end before one corrective move.
    home_tolerance_rad: float = 0.02
    #: Put the object back after every successful pick, whether or not the goal asks to.
    release_after_success: bool = False

    def __post_init__(self):
        if not isinstance(self.release_after_success, bool):
            raise ValueError("release_after_success must be a boolean")
        for name in ("max_attempts", "max_moved_replans", "max_regrasps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (1 if name == "max_attempts" else 0):
                raise ValueError(f"{name} must be a non-negative integer (max_attempts at least 1)")
        reserve = self.execute_reserve_s
        if isinstance(reserve, bool) or not isinstance(reserve, int | float) or not 0 <= reserve < float("inf"):
            raise ValueError("execute_reserve_s must be a finite non-negative number of seconds")


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
    #: The requested post-pick release and return completed.
    released_after_success: bool = False
    #: One entry per retry, in order: {"attempt", "retry", "code", "message"}; ``retry`` is
    #: "reobserve" (nothing moved), "replan" (target moved, arm retreated) or "regrasp".
    retries: list[dict] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    mode: str = "execute"
    execution_detail: ExecutionDetail = field(
        default_factory=lambda: ExecutionDetail(schema_version=1, recovery_state="NOT_NEEDED", suggested_action="NONE")
    )


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

    @property
    def config(self) -> PipelineConfig:
        return self._config

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
        outcome = PickOutcome(mode=mode.value)
        outcome.execution_detail = replace(outcome.execution_detail, phase="validate", phase_state="RUNNING")
        if not str(query).strip():
            return self._fail(outcome, "INVALID_TARGET", "empty target query")

        def stop() -> bool:
            return cancelled() or self._clock() >= deadline

        def phase(name: str, event: str, detail: dict) -> None:
            previous = outcome.execution_detail
            state = detail.get("state", "succeeded")
            failed = event == "end" and state != "succeeded"
            public_name = "verify" if name == "hold_lifted" else name
            if previous.recovery_state == "IN_PROGRESS" and name != "recover":
                public_name = "recover"
            recovery = detail.get("recovery_state", previous.recovery_state)
            reason = detail.get("reason_code", "")
            outcome.execution_detail = replace(
                previous,
                phase=public_name,
                phase_state="RUNNING"
                if event == "start"
                else "CANCELED"
                if state == "cancelled"
                else "FAILED"
                if failed
                else "SUCCEEDED",
                failed_phase=previous.failed_phase or (public_name if failed else ""),
                reason_code=previous.reason_code or reason,
                recovery_state=recovery,
                suggested_action=failure_action(detail.get("error_code", reason), reason, recovery)
                if failed
                else "NONE",
            )
            # Steps report their state; a phase counts as completed only when it succeeded.
            if event == "end" and detail.get("state", "succeeded") == "succeeded":
                outcome.completed_phases.append(name)
            on_phase(public_name, event, {**detail, "execution_detail": outcome.execution_detail.to_dict()})

        backends = self._backends
        try:
            return self._run(query, mode, deadline, cancelled, stop, phase, release_after_success, outcome)
        finally:
            # The goal is over: the robot keeps holding what it was last sent.
            for part in (backends.executor, backends.effector):
                finish = getattr(part, "finish", None)
                if finish is not None:
                    try:
                        finish()
                    except Exception as exc:  # noqa: BLE001 - the goal's own outcome stands
                        outcome.report.setdefault("finish_errors", []).append(f"{type(exc).__name__}: {exc}")

    def _run(self, query, mode, deadline, cancelled, stop, phase, release_after_success, outcome) -> PickOutcome:
        backends = self._backends
        planning_failures, replans, regrasps = 0, 0, 0
        try:
            while True:
                self._check_budget(cancelled, deadline)
                outcome.execution_detail = ExecutionDetail(
                    schema_version=1, retry_count=outcome.attempts, recovery_state="NOT_NEEDED", suggested_action="NONE"
                )
                outcome.attempts += 1
                if outcome.retries:
                    outcome.retries[-1]["state"] = "started"
                budget = {"deadline": deadline, "cancelled": stop}
                steps: dict[str, float] = {}
                try:
                    observation = self._step("observe", phase, steps, backends.observation.capture, **budget)
                    outcome.report["observe"] = dict(observation.report)
                    self._check_budget(cancelled, deadline)
                    if mode is PickMode.OBSERVE_ONLY:
                        outcome.success = True
                        return outcome
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
                        self._retry(outcome, "reobserve", failure.code, str(failure))
                        continue
                    raise
                outcome.candidate_index = plan.candidate_index
                outcome.report["steps_s"] = dict(steps)
                outcome.report["perceive"] = dict(proposals.report)
                outcome.report["plan"] = dict(plan.report)
                if mode is PickMode.PLAN_ONLY:
                    outcome.success = True
                    return outcome
                left = deadline - self._clock()
                if left < self._config.execute_reserve_s:
                    raise GraspFailure(
                        "SKILL_TIMEOUT",
                        f"{left:.1f} s left of the goal, an execution needs {self._config.execute_reserve_s:g} s;"
                        " the arm was not moved",
                        reason_code="EXECUTION_BUDGET_INSUFFICIENT",
                    )
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
                    self._retry(outcome, "replan", result.error_code, result.message)
                    continue
                if (
                    result.error_code == "GRASP_VERIFICATION_FAILED"
                    and result.reason_code in {"EMPTY_GRASP", "GRIP_LOST"}
                    and result.at_rest
                    and regrasps < self._config.max_regrasps
                    and not stop()
                ):
                    # The sequence took the plan back: the arm is home and the hand open.
                    regrasps += 1
                    self._retry(outcome, "regrasp", result.error_code, result.message)
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
                    self._fail(outcome, code, result.message, result.reason_code)
                else:
                    outcome.execution_detail = replace(
                        outcome.execution_detail, failed_phase="", reason_code="", suggested_action="NONE"
                    )
                return outcome
        except GraspFailure as failure:
            code = failure.code
            if cancelled():
                code = "CANCELED"
            elif self._clock() >= deadline:
                code = "SKILL_TIMEOUT"
            return self._fail(outcome, code, str(failure), failure.reason_code)
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
        outcome.at_rest = False
        outcome.execution_detail = replace(outcome.execution_detail, recovery_state="UNKNOWN")
        try:
            result = sequence.run(plan, deadline=deadline, cancelled=stop, release_after_success=release)
        finally:
            if watch is not None:
                try:
                    report = dict(watch.disarm())
                except Exception as exc:
                    report = {"error": f"{type(exc).__name__}: {exc}"}
                outcome.report.setdefault("watch", []).append(report)
        outcome.held, outcome.held_confidence, outcome.at_rest = result.held, result.held_confidence, result.at_rest
        outcome.released_after_success = result.released_after_success
        outcome.execution_detail = replace(outcome.execution_detail, recovery_state=result.recovery_state)
        outcome.report.setdefault("executions", []).append({"phases": result.phases, "retreated": result.retreated})
        return result

    @staticmethod
    def _retry(outcome: PickOutcome, retry: str, code: str, message: str) -> None:
        outcome.retries.append(
            {"attempt": outcome.attempts, "retry": retry, "code": code, "message": message, "state": "planned"}
        )

    def _step(self, name: str, phase: PhaseCallback, timings: dict, call, *args, **kwargs):
        phase(name, "start", {})
        started = self._clock()
        try:
            value = call(*args, **kwargs)
        except Exception as exc:
            phase(
                name,
                "end",
                {
                    "state": "cancelled" if getattr(exc, "code", "") == "CANCELED" else "failed",
                    "error_code": getattr(exc, "code", "INTERNAL_ERROR"),
                    "reason_code": getattr(exc, "reason_code", "INTERNAL_ERROR"),
                },
            )
            raise
        timings[name] = self._clock() - started
        phase(name, "end", {"seconds": timings[name]})
        return value

    def _check_budget(self, cancelled: Flag, deadline: float) -> None:
        if cancelled():
            raise GraspFailure("CANCELED", "goal cancelled")
        if self._clock() >= deadline:
            raise GraspFailure("SKILL_TIMEOUT", "goal deadline passed")

    @staticmethod
    def _fail(outcome: PickOutcome, code: str, message: str, reason_code: str = "") -> PickOutcome:
        outcome.success, outcome.error_code, outcome.message = False, code, message
        detail = outcome.execution_detail
        reason = detail.reason_code or reason_code or code
        outcome.execution_detail = replace(
            detail,
            # The task can fail after recovery succeeded. Do not rewrite the
            # completed recovery stage as a failed motion.
            phase_state=detail.phase_state
            if detail.phase == "recover" and detail.recovery_state == "COMPLETED"
            else "CANCELED"
            if code == "CANCELED"
            else "FAILED",
            failed_phase=detail.failed_phase or detail.phase,
            reason_code=reason,
            suggested_action=failure_action(code, reason, detail.recovery_state),
        )
        return outcome
