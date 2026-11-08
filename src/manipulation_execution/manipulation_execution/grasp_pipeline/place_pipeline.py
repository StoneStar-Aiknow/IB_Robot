"""Robot-independent placement orchestration and phase timing."""

import time
from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np

from embodied_common.execution_detail import ExecutionDetail, place_execution_detail

from .contracts import GraspFailure
from .place_sequence import PlaceSequence


@dataclass
class PlaceOutcome:
    success: bool = False
    release_status: str = "not_released"
    custody_outcome: str = "held"
    verification: str = "not_run"
    reason_code: str = ""
    message: str = ""
    completed_phases: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    execution_detail: ExecutionDetail = field(
        default_factory=lambda: ExecutionDetail(
            schema_version=2,
            recovery_state="NOT_NEEDED",
            suggested_action="NONE",
            release_status="NOT_RELEASED",
            verification_status="NOT_RUN",
        )
    )


class PlacePipeline:
    def __init__(self, backends, *, max_attempts=2, execute_reserve_s=15.0, clock=time.monotonic):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if (
            isinstance(execute_reserve_s, bool)
            or not isinstance(execute_reserve_s, int | float)
            or not isfinite(execute_reserve_s)
            or execute_reserve_s < 0
        ):
            raise ValueError("execute_reserve_s must be finite and non-negative")
        self.backends, self.max_attempts, self.clock = backends, max_attempts, clock
        self.execute_reserve_s = float(execute_reserve_s)

    def run(self, held, query, *, deadline, cancelled, on_phase=lambda *_: None):
        result, starts = PlaceOutcome(), {}
        planning_deadline = deadline - self.execute_reserve_s

        def phase(name, event, detail):
            if event == "start":
                starts[name] = self.clock()
            elif name in starts:
                result.report.setdefault("stages_s", {}).setdefault(name, []).append(self.clock() - starts.pop(name))
            failed = event == "end" and detail.get("state", "succeeded") != "succeeded"
            if "release_status" in detail:
                result.release_status = detail["release_status"].lower()
            previous = place_execution_detail(
                result.execution_detail,
                release_status=result.release_status.upper(),
                verification_status=result.verification.upper(),
                place_succeeded=False,
            )
            result.execution_detail = replace(
                previous,
                phase=name,
                phase_state="RUNNING" if event == "start" else "FAILED" if failed else "SUCCEEDED",
                failed_phase=previous.failed_phase or (name if failed else ""),
                reason_code=detail.get("reason_code", previous.reason_code),
                recovery_state=detail.get("recovery_state", previous.recovery_state),
            )
            if event == "end" and not failed:
                result.completed_phases.append(name)
            on_phase(name, event, {**detail, "execution_detail": result.execution_detail.to_dict()})

        def check():
            if cancelled():
                raise GraspFailure("CANCELED", "placement canceled")
            if self.clock() >= deadline:
                raise GraspFailure("SKILL_TIMEOUT", "placement budget expired")
            grip = self.backends.effector.holding()
            if grip != "":
                result.custody_outcome = "uncertain"
                result.report["grip_failure"] = {"phase": result.execution_detail.phase, "reason": grip}
                raise GraspFailure(
                    "GRASP_UNCERTAIN" if grip is None else "GRIP_LOST_IN_TRANSFER",
                    f"grip is not confirmed while preparing place: {grip if grip is not None else 'feedback unavailable'}",
                )

        def require_handover():
            check()
            measured = np.asarray(self.backends.executor.measured(held.joint_names), dtype=float)
            expected = np.asarray(held.joint_positions, dtype=float)
            if measured.shape != expected.shape or not np.allclose(measured, expected, atol=0.02, rtol=0):
                result.custody_outcome = "uncertain"
                raise GraspFailure("HELD_OBJECT_MISMATCH", "holding arm differs from the retained handover pose")
            check()

        def remember_failure(code, message, *, phase_name=None):
            record = {
                "attempt": result.execution_detail.retry_count + 1,
                "phase": phase_name or result.execution_detail.failed_phase or result.execution_detail.phase,
                "code": code,
                "message": message,
            }
            failures = result.report.setdefault("failures", [])
            if not failures or any(failures[-1].get(key) != value for key, value in record.items()):
                failures.append(record)
            else:
                record = failures[-1]
            result.report.setdefault("first_failure", dict(record))
            return record

        def call(name, operation, *args, operation_deadline=deadline):
            phase(name, "start", {})
            try:
                if result.custody_outcome == "held":
                    check()
                if self.clock() >= operation_deadline:
                    raise GraspFailure(
                        "SKILL_TIMEOUT",
                        "too little time remains to prepare placement",
                        reason_code="EXECUTION_BUDGET_INSUFFICIENT",
                    )
                value = operation(*args, deadline=operation_deadline, cancelled=cancelled)
            except Exception as exc:
                phase(name, "end", {"state": "failed", "reason_code": getattr(exc, "code", "INTERNAL_ERROR")})
                raise
            phase(name, "end", {})
            return value

        backends = self.backends
        try:
            backends.restore(held)
            for attempt in range(self.max_attempts):
                result.execution_detail = replace(
                    result.execution_detail, retry_count=attempt, failed_phase="", reason_code=""
                )
                require_handover()
                if self.clock() >= planning_deadline:
                    raise GraspFailure(
                        "SKILL_TIMEOUT",
                        "too little time remains to plan and execute placement",
                        reason_code="EXECUTION_BUDGET_INSUFFICIENT",
                    )
                attempt_started = self.clock()
                try:
                    observation = call(
                        "observe_container", backends.observation.capture, operation_deadline=planning_deadline
                    )
                    result.report["observe"] = dict(observation.report)
                    require_handover()
                    container = call(
                        "perceive_container",
                        backends.proposer.propose,
                        observation,
                        query,
                        held,
                        operation_deadline=planning_deadline,
                    )
                    require_handover()
                    plan = call(
                        "plan_place",
                        backends.planner.plan,
                        observation,
                        container,
                        held,
                        operation_deadline=planning_deadline,
                    )
                    require_handover()
                    if self.clock() >= planning_deadline:
                        raise GraspFailure(
                            "SKILL_TIMEOUT",
                            "placement plan consumed the execution reserve",
                            reason_code="EXECUTION_BUDGET_INSUFFICIENT",
                        )
                    break
                except GraspFailure as failure:
                    record = remember_failure(failure.reason_code, str(failure))
                    record.update(retryable=failure.retryable, elapsed_s=self.clock() - attempt_started)
                    if not failure.retryable or attempt + 1 == self.max_attempts:
                        record["retry_decision"] = "not_retryable" if not failure.retryable else "attempt_limit"
                        raise
                    # A new capture must still be safe, and have enough time to repeat
                    # the last preparation while keeping the execution reserve intact.
                    record["retry_decision"] = "checking_handover"
                    require_handover()
                    remaining = planning_deadline - self.clock()
                    record["remaining_planning_s"] = remaining
                    if remaining <= record["elapsed_s"]:
                        record["retry_decision"] = "insufficient_budget"
                        raise
                    record["retry_decision"] = "retry"
                    result.report.setdefault("retries", []).append(dict(record))
            result.report["plan"] = dict(plan.report)
            plan_home = getattr(backends.planner, "plan_home", None)
            execution = PlaceSequence(backends, on_phase=phase).run(
                plan,
                cancelled=lambda: cancelled() or self.clock() >= deadline,
                go_home=None if plan_home is None else lambda: plan_home(held, deadline=deadline, cancelled=cancelled),
            )
            result.release_status, result.custody_outcome = execution.release_status, execution.custody_outcome
            if not execution.success:
                result.execution_detail = replace(result.execution_detail, recovery_state=execution.recovery_state)
                first = execution.first_failure
                result.report["first_execution_failure"] = dict(
                    remember_failure(
                        first.get("code", execution.reason_code),
                        first.get("message", execution.message),
                        phase_name=first.get("phase"),
                    )
                )
                if execution.reason_code == "CANCELED" and not cancelled() and self.clock() >= deadline:
                    raise GraspFailure("SKILL_TIMEOUT", "placement budget expired during motion")
                raise GraspFailure(execution.reason_code, execution.message)
            # All arm motion is over before visual verification starts.
            if cancelled() or self.clock() >= deadline:
                raise GraspFailure(
                    "CANCELED" if cancelled() else "SKILL_TIMEOUT", "placement released; verification did not finish"
                )
            after = call("observe_placed", backends.observation.capture)

            def verify(*, deadline, cancelled):
                verdict = backends.verifier.verify(after, plan, held, deadline=deadline, cancelled=cancelled)
                result.report["verification"] = dict(verdict.report)
                result.verification = (
                    "success" if verdict.success else "uncertain" if verdict.reason_code == "PLACE_UNSEEN" else "failed"
                )
                if not verdict.success:
                    raise GraspFailure(verdict.reason_code or "PLACE_UNSEEN", "placement could not be verified")

            call("verify_place", verify)
        except GraspFailure as failure:
            remember_failure(failure.reason_code, str(failure))
            result.reason_code, result.message = failure.reason_code, str(failure)
        except Exception as exc:
            result.reason_code, result.message = "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"
            remember_failure(result.reason_code, result.message)
            result.custody_outcome = "uncertain"
        finally:
            if result.custody_outcome == "held":
                # Pick may retrace only from the original handover pose.
                # A failed restore/check cannot certify that pose or its grip.
                try:
                    if backends.effector.holding() != "" or not np.allclose(
                        backends.executor.measured(held.joint_names), held.joint_positions, atol=0.02, rtol=0
                    ):
                        result.custody_outcome = "uncertain"
                except Exception:
                    result.custody_outcome = "uncertain"
            for part in (backends.executor, backends.effector):
                finish = getattr(part, "finish", None)
                if finish is not None:
                    try:
                        finish()
                    except Exception as exc:
                        remember_failure("CLEANUP_FAILED", str(exc), phase_name="cleanup")
                        result.success, result.reason_code = False, "CLEANUP_FAILED"
                        result.execution_detail = replace(
                            result.execution_detail,
                            phase="cleanup",
                            failed_phase=result.execution_detail.failed_phase or "cleanup",
                        )
                        result.message = str(exc)
                        result.custody_outcome = "uncertain"
        result.success = not result.reason_code and result.verification == "success"
        result.execution_detail = place_execution_detail(
            result.execution_detail,
            release_status=result.release_status.upper(),
            verification_status=result.verification.upper(),
            place_succeeded=result.success,
        )
        if not result.success:
            previous = result.execution_detail
            result.execution_detail = replace(
                previous,
                phase_state="CANCELED" if result.reason_code == "CANCELED" else "FAILED",
                failed_phase=previous.failed_phase or previous.phase,
                reason_code=result.reason_code,
                suggested_action="MANUAL_CHECK"
                if result.custody_outcome == "uncertain" or previous.recovery_state == "FAILED"
                else "WAIT_USER",
            )
        return result
