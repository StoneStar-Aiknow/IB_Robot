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
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace

from embodied_common.execution_detail import ExecutionDetail

from .contracts import (
    EffectorAction,
    EffectorOutcome,
    Flag,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Observation,
    Proposals,
)
from .outcome_message import failure_action
from .prefetch import PerceptionPrefetch
from .sequence import GraspSequence, PhaseCallback


def _same(value):
    """The already prefetched result of an observe or perceive step."""
    return value


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
    #: Run the plan's PRESHAPE while planning instead of after it. The preshape does not
    #: depend on the candidate; on the X2 it opens the hand (~0.6 s) while the Orin plans
    #: (1.8-2.9 s), supervised as in the sequence. Its outcome is recorded as the plan's
    #: first PRESHAPE step; a failed or unfinished one is run there again.
    prefetch_preshape: bool = False
    #: Accept a capture and its proposals started before the goal (``prefetch_perception``
    #: below, while the task-pose preparation runs): the first attempt uses them for the same
    #: target when they are at most ``prefetch_max_age_s`` old and the robot's posture
    #: (``GraspBackends.posture``) has not changed since. Retries always observe again. The
    #: age is short because the planner's budget (10 s on the X2) counts from the capture.
    prefetch_perception: bool = False
    prefetch_max_age_s: float = 4.0
    #: After a planning failure that only says this candidate set held no complete path
    #: (``NO_EXECUTABLE_CANDIDATE`` with a reason in ``RESAMPLE_REASONS``; the arm has not
    #: moved), ask the proposer for new candidates on the same observation and target
    #: (``GraspProposer.resample``) instead of capturing and perceiving again, while that
    #: observation is at most ``resample_max_age_s`` old. Live 2026-11-11 (X2, 310P): each
    #: re-observe cost 0.53-0.56 s of capture and 1.26-1.29 s of perception, while GraspGen-X
    #: on the same cloud answers in ~0.7 s with a new candidate set (3 calls, no grasp within
    #: 5 mm and 5 deg of another call's). The planner still checks a fresh frame against the
    #: planned one before anything moves. Proposers without ``resample`` observe again.
    resample_retries: bool = False
    resample_max_age_s: float = 4.0

    def __post_init__(self):
        for name in ("release_after_success", "prefetch_preshape", "prefetch_perception", "resample_retries"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("max_attempts", "max_moved_replans", "max_regrasps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (1 if name == "max_attempts" else 0):
                raise ValueError(f"{name} must be a non-negative integer (max_attempts at least 1)")
        reserve = self.execute_reserve_s
        if isinstance(reserve, bool) or not isinstance(reserve, int | float) or not 0 <= reserve < float("inf"):
            raise ValueError("execute_reserve_s must be a finite non-negative number of seconds")
        age = self.prefetch_max_age_s
        if isinstance(age, bool) or not isinstance(age, int | float) or not 0 < age < float("inf"):
            raise ValueError("prefetch_max_age_s must be a finite positive number of seconds")
        age = self.resample_max_age_s
        if isinstance(age, bool) or not isinstance(age, int | float) or not 0 < age < float("inf"):
            raise ValueError("resample_max_age_s must be a finite positive number of seconds")


#: Planning failure reasons that only condemn the candidate set, not the observation:
#: the search over these candidates found no complete three-segment path.
RESAMPLE_REASONS = frozenset({"NO_COMPLETE_PATH"})


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
    #: "reobserve" (nothing moved), "resample" (nothing moved; new candidates on the same
    #: observation), "replan" (target moved, arm retreated) or "regrasp".
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
        self._prefetch = (
            PerceptionPrefetch(
                capture=backends.observation.capture,
                propose=backends.proposer.propose,
                posture=backends.posture,
                detect_container=backends.container_detector,
                max_age_s=config.prefetch_max_age_s,
                clock=clock,
            )
            if config.prefetch_perception and backends.posture is not None
            else None
        )

    @property
    def prefetch(self) -> PerceptionPrefetch | None:
        """Started by the pick node's prefetch request; None when not configured or not offered."""
        return self._prefetch

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
        on_held: Callable | None = None,
        on_closed: Callable | None = None,
    ) -> PickOutcome:
        """``deadline`` is on this pipeline's clock; ``cancelled`` is the goal's own flag.

        ``on_closed(observation, proposals, plan)`` runs when the hand has closed on
        a plan's target, before the lift: a retained pick starts describing the held
        object there (it must not block; an exception is only reported).
        """
        outcome = PickOutcome(mode=mode.value)
        self._on_closed = on_closed
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
            result = self._run(query, mode, deadline, cancelled, stop, phase, release_after_success, outcome)
            if result.success and result.held is True and not result.released_after_success and on_held is not None:
                began = time.monotonic()
                try:
                    on_held(*self._held_context)
                except Exception as exc:
                    result.at_rest = False
                    return self._fail(result, "CUSTODY_FAILED", f"could not retain held object: {exc}")
                finally:
                    result.report.setdefault("handover_s", {})["retain"] = round(time.monotonic() - began, 4)
            return result
        finally:
            # The goal is over: the robot keeps holding what it was last sent.
            for part in (backends.executor, backends.effector):
                if (
                    part is backends.effector
                    and outcome.success
                    and outcome.held is True
                    and not outcome.released_after_success
                    and on_held is not None
                ):
                    continue
                finish = getattr(part, "finish", None)
                if finish is not None:
                    try:
                        finish()
                    except Exception as exc:  # noqa: BLE001 - the goal's own outcome stands
                        outcome.report.setdefault("finish_errors", []).append(f"{type(exc).__name__}: {exc}")

    def _run(self, query, mode, deadline, cancelled, stop, phase, release_after_success, outcome) -> PickOutcome:
        backends = self._backends
        planning_failures, replans, regrasps = 0, 0, 0
        # (observation, proposals, monotonic capture time) a "resample" retry reuses.
        resample = None
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
                observation = proposals = None
                observed_at = float("-inf")  # nothing captured yet: never resampled
                try:
                    taken = None
                    if self._prefetch is not None and outcome.attempts == 1 and mode is not PickMode.OBSERVE_ONLY:
                        taken, outcome.report["perception_prefetch"] = self._prefetch.take(
                            query, deadline=deadline, cancelled=stop
                        )
                    if resample is not None:
                        # New candidates on the observation the last attempt planned from.
                        observation, previous, observed_at = resample
                        resample = None
                        proposals = self._step(
                            "perceive", phase, steps, backends.proposer.resample, observation, previous, **budget
                        )
                    elif taken is not None:
                        # Started before the goal: recorded as this attempt's observe and perceive.
                        observation = self._step("observe", phase, steps, _same, taken.observation)
                        outcome.report["observe"] = dict(observation.report)
                        observed_at = self._clock() - float(outcome.report["perception_prefetch"].get("age_s", 0.0))
                        proposals = self._step("perceive", phase, steps, _same, taken.proposals)
                    else:
                        observed_at = self._clock()
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
                    preshape = self._prefetch_preshape(stop) if mode is PickMode.EXECUTE else None
                    try:
                        plan = self._step("plan", phase, steps, backends.planner.plan, observation, proposals, **budget)
                    finally:
                        preshaped = self._preshape_result(preshape, outcome)
                    self._check_budget(cancelled, deadline)
                except GraspFailure as failure:
                    planning_failures += 1
                    self._check_budget(cancelled, deadline)
                    if failure.retryable and planning_failures < self._config.max_attempts:
                        outcome.report.setdefault("planning_failures", []).append(
                            {"code": failure.code, "message": str(failure)}
                        )
                        if self._resamples(failure, proposals, observed_at):
                            resample = (observation, proposals, observed_at)
                            self._retry(outcome, "resample", failure.code, str(failure))
                        else:
                            self._retry(outcome, "reobserve", failure.code, str(failure))
                        continue
                    raise
                outcome.candidate_index = plan.candidate_index
                outcome.report["steps_s"] = dict(steps)
                outcome.report["perceive"] = dict(proposals.report)
                outcome.report["plan"] = dict(plan.report)
                if not release_after_success and plan.carry_steps is not None:
                    plan = replace(plan, steps=plan.carry_steps, carry_steps=None)
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
                    observation,
                    proposals,
                    plan,
                    deadline,
                    cancelled,
                    phase,
                    release_after_success,
                    outcome,
                    preshaped=preshaped,
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
                    self._held_context = (observation, proposals, plan)
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

    def _prefetch_preshape(self, stop: Flag):
        """Start the PRESHAPE on a thread while planning; None when not configured or not offered.

        Supervised as GraspSequence._act supervises it: the arm is not commanded,
        and a hand that drags it or a controller that drops its target stops it.
        """
        effector = self._backends.effector
        if not self._config.prefetch_preshape or getattr(effector, "act", None) is None:
            return None
        supervise = getattr(self._backends.executor, "supervise", None)
        box: dict = {"began": time.monotonic()}

        def run() -> None:
            try:
                check = supervise() if supervise is not None else (lambda: "")

                def stopped() -> bool:
                    if stop():
                        return True
                    reason = check()
                    if reason:
                        box["fault"] = reason
                    return bool(reason)

                box["outcome"] = effector.act(EffectorAction.PRESHAPE, halted=lambda: False, cancelled=stopped)
            except Exception as exc:  # noqa: BLE001 - the sequence runs the preshape itself
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                box["ended"] = time.monotonic()

        thread = threading.Thread(target=run, name="prefetch-preshape", daemon=True)
        box["thread"] = thread
        thread.start()
        return box

    @staticmethod
    def _preshape_result(box, outcome) -> EffectorOutcome | None:
        """Wait for the prefetched PRESHAPE; its successful, unsupervision-faulted outcome or None."""
        if box is None:
            return None
        box["thread"].join()
        row = {"s": round(box["ended"] - box["began"], 3)}
        result = box.get("outcome")
        if "error" in box:
            row["error"] = box["error"]
        elif box.get("fault"):
            row["fault"] = box["fault"]
        elif result is not None:
            row.update(succeeded=bool(result.succeeded), reason=result.reason)
        outcome.report.setdefault("preshape_prefetch", []).append(row)
        if result is None or box.get("fault") or "error" in box or not result.succeeded or result.halted:
            return None
        return result

    def _execute(
        self,
        observation: Observation,
        proposals: Proposals,
        plan: GraspPlan,
        deadline,
        stop,
        phase,
        release,
        outcome,
        *,
        preshaped: EffectorOutcome | None = None,
    ):
        watch = self._backends.watch
        watched = getattr(watch, "phase", None)

        def on_phase(name: str, event: str, detail: dict) -> None:
            phase(name, event, detail)
            if watched is not None:
                watched(name, event)
            closed = getattr(self, "_on_closed", None)
            if (
                closed is not None
                and name == "close"
                and event == "end"
                and detail.get("state", "succeeded") == "succeeded"
            ):
                try:
                    closed(observation, proposals, plan)
                except Exception as exc:  # noqa: BLE001 - the pick goes on; the handover computes inline
                    outcome.report.setdefault("on_closed_errors", []).append(f"{type(exc).__name__}: {exc}")

        sequence = GraspSequence(self._backends, home_tolerance_rad=self._config.home_tolerance_rad, on_phase=on_phase)
        if watch is not None:
            watch.arm(observation, proposals, plan)
        outcome.at_rest = False
        outcome.execution_detail = replace(outcome.execution_detail, recovery_state="UNKNOWN")
        try:
            result = sequence.run(
                plan, deadline=deadline, cancelled=stop, release_after_success=release, preshaped=preshaped
            )
        finally:
            # From the arm's last step to the terminal record (the handover to a place).
            sequence_done = time.monotonic()
            handover = outcome.report.setdefault("handover_s", {})
            handover["_sequence_done"] = sequence_done
            if watch is not None:
                try:
                    report = dict(watch.disarm())
                except Exception as exc:
                    report = {"error": f"{type(exc).__name__}: {exc}"}
                outcome.report.setdefault("watch", []).append(report)
            handover["watch_disarm"] = round(time.monotonic() - sequence_done, 4)
        outcome.held, outcome.held_confidence, outcome.at_rest = result.held, result.held_confidence, result.at_rest
        outcome.released_after_success = result.released_after_success
        outcome.execution_detail = replace(outcome.execution_detail, recovery_state=result.recovery_state)
        execution = {
            "attempt": outcome.attempts,
            "phases": result.phases,
            "retreated": result.retreated,
            "error_code": result.error_code,
            "reason_code": result.reason_code,
            "message": result.message,
            "held": result.held,
            "lost": result.lost,
            "at_rest": result.at_rest,
            "recovery_state": result.recovery_state,
            "execution_report": deepcopy(result.execution_report),
            "exception": result.exception,
            "first_failure": dict(result.first_failure),
        }
        outcome.report.setdefault("executions", []).append(execution)
        if not result.succeeded:
            failure = {
                "attempt": outcome.attempts,
                **(
                    result.first_failure
                    or {
                        "phase": outcome.execution_detail.failed_phase or outcome.execution_detail.phase,
                        "error_code": result.error_code,
                        "reason_code": result.reason_code,
                        "message": result.message,
                    }
                ),
            }
            outcome.report.setdefault("execution_failures", []).append(failure)
            outcome.report.setdefault("first_execution_failure", dict(failure))
        return result

    def _resamples(self, failure: GraspFailure, proposals, observed_at: float) -> bool:
        """Whether a retry may keep this observation and ask only for new candidates."""
        return (
            self._config.resample_retries
            and proposals is not None
            and failure.code == "NO_EXECUTABLE_CANDIDATE"
            and failure.reason_code in RESAMPLE_REASONS
            and callable(getattr(self._backends.proposer, "resample", None))
            and self._clock() - observed_at <= self._config.resample_max_age_s
        )

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
