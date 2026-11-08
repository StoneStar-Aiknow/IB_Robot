"""Execute a placement and retreat only while the controller remains healthy."""

from dataclasses import dataclass, field

from .contracts import Effect, EffectorAction, Hold, MotionState, Move
from .sequence import GraspSequence, SequenceResult, _Stopped


@dataclass
class PlaceSequenceResult:
    success: bool = False
    release_status: str = "not_released"
    custody_outcome: str = "held"
    reason_code: str = ""
    message: str = ""
    recovery_state: str = "NOT_NEEDED"
    first_failure: dict = field(default_factory=dict)


class PlaceSequence:
    def __init__(self, backends, *, on_phase=lambda *_: None):
        self.backends, self.phase = backends, on_phase

        def sequence_phase(name, event, detail):
            if name == EffectorAction.RELEASE.value and event == "start":
                detail = {**detail, "release_status": "UNKNOWN"}
            on_phase(name, event, detail)

        self.sequence = GraspSequence(backends, on_phase=sequence_phase)

    def _remember_failure(self, result, code, message):
        if not result.first_failure:
            result.first_failure = {"phase": self.sequence._current_phase, "code": code, "message": message}

    def _stop(self, result, reason):
        result.recovery_state = "FAILED"
        try:
            self.backends.executor.stop(reason)
        except Exception as exc:
            result.recovery_state = "UNKNOWN"
            result.message = f"{result.message}; stop failed: {type(exc).__name__}: {exc}"

    def _return_home(self, go_home, cancelled, record, result):
        """Plan and drive the empty hand home after a drop; on failure stop where the arm is."""
        try:
            step = go_home()
            self.sequence._expect(self.sequence._move(step.segment, False, lambda: False, cancelled, record))
            return True
        except Exception as exc:
            # The planning or motion failure is the one to report.
            result.message = f"{result.message}; then could not return home: {exc}"
            self._stop(result, str(exc))
            if result.recovery_state == "UNKNOWN":
                result.custody_outcome = "uncertain"
            self.phase("recover", "end", {"state": "failed", "recovery_state": result.recovery_state})
            return False

    def _release_after_drop(self, cancelled, record, result):
        result.release_status, result.custody_outcome = "unknown", "uncertain"
        self.sequence._expect_effect(self.sequence._act(EffectorAction.RELEASE, lambda: False, cancelled, record))
        released = getattr(self.backends.effector, "released", None)
        if released is None or not released():
            raise _Stopped("RELEASE_UNCERTAIN", "open, empty hand could not be confirmed after a drop")
        result.release_status, result.custody_outcome = "released", "lost"
        self.phase("release_confirmed", "end", {"release_status": "RELEASED"})

    def recover_before_transfer(self, *, go_home, cancelled):
        """Close a known preparation-stage drop with fresh release and home feedback.

        No placement segment has started. Keep the placement failure, but do
        not leave an empty, recovered hand in a permanent uncertain custody.
        """
        result = PlaceSequenceResult(custody_outcome="uncertain")
        record = SequenceResult()
        self.phase("recover", "start", {"recovery_state": "IN_PROGRESS"})
        try:
            if cancelled():
                raise _Stopped("CANCELED", "drop recovery canceled or out of time")
            if go_home is None:
                raise _Stopped("RECOVERY_UNAVAILABLE", "no fresh home planner is available")
            self._release_after_drop(cancelled, record, result)
            if not self._return_home(go_home, cancelled, record, result):
                return result
            result.recovery_state = "COMPLETED"
            self.phase("recover", "end", {"recovery_state": "COMPLETED"})
        except Exception as exc:
            result.reason_code = getattr(exc, "code", "MOTION_FAILED")
            result.message = str(exc)
            result.custody_outcome = "uncertain"
            self._stop(result, result.message)
            self.phase("recover", "end", {"state": "failed", "recovery_state": result.recovery_state})
        return result

    def run(self, plan, *, cancelled, go_home=None):
        """``go_home`` plans the empty hand's way home (a ``Move``) after a drop in transfer."""
        result, record, walked = PlaceSequenceResult(), SequenceResult(), []
        carrying, lost = True, []

        def check_grip():
            reason = self.backends.effector.holding()
            if reason != "":
                lost.append(reason)
            return bool(lost)

        try:
            if self.backends.effector.holding() != "":
                result.reason_code, result.custody_outcome = "GRASP_UNCERTAIN", "uncertain"
                result.message = "grip is not confirmed before transfer"
                self._remember_failure(result, result.reason_code, result.message)
                return result
            for step in (*plan.transfer, *plan.release, *plan.retreat):
                if isinstance(step, Move):
                    outcome = self.sequence._move(
                        step.segment, carrying, check_grip if carrying else lambda: False, cancelled, record
                    )
                    if outcome.state is MotionState.HALTED and lost:
                        if lost[0] is None:
                            raise _Stopped("GRASP_UNCERTAIN", "grip feedback unavailable during transfer")
                        result.reason_code, result.message = "GRIP_LOST_IN_TRANSFER", lost[0]
                        self._remember_failure(result, result.reason_code, result.message)
                        self.phase("recover", "start", {"recovery_state": "IN_PROGRESS"})
                        back = step.segment.walked_back(
                            self.sequence._stopped_at(step.segment, cancelled),
                            path_progress=outcome.report.get("path_progress"),
                        )
                        if back is not None:
                            self.sequence._expect(self.sequence._move(back, False, lambda: False, cancelled, record))
                        for previous in reversed(walked):
                            self.sequence._expect(
                                self.sequence._move(previous.reversed(), False, lambda: False, cancelled, record)
                            )
                        self._release_after_drop(cancelled, record, result)
                        # The planned post-release retreat starts at the release pose,
                        # so it cannot be reused from the handover pose: plan home afresh.
                        if go_home is not None and not self._return_home(go_home, cancelled, record, result):
                            return result
                        result.recovery_state = "COMPLETED"
                        self.phase("recover", "end", {"recovery_state": "COMPLETED"})
                        return result
                    self.sequence._expect(outcome)
                    if carrying:
                        walked.append(step.segment)
                elif isinstance(step, Effect):
                    result.release_status, result.custody_outcome = "unknown", "uncertain"
                    self.sequence._expect_effect(self.sequence._act(step.action, lambda: False, cancelled, record))
                    released = getattr(self.backends.effector, "released", None)
                    if released is None or not released():
                        raise _Stopped("RELEASE_UNCERTAIN", "open hand and release could not be confirmed")
                    carrying = False
                    result.release_status, result.custody_outcome = "released", "released"
                    self.phase("release_confirmed", "end", {"release_status": "RELEASED"})
                elif isinstance(step, Hold):
                    self.sequence._expect(self.sequence._hold(step, carrying, cancelled, record))
            result.success = True
            return result
        except _Stopped as exc:
            self._remember_failure(result, exc.code, str(exc))
            result.reason_code, result.message = exc.code, str(exc)
        except Exception as exc:
            self._remember_failure(result, "MOTION_FAILED", str(exc))
            result.reason_code, result.message = "MOTION_FAILED", str(exc)
        self._stop(result, result.message)
        result.custody_outcome = "uncertain"
        return result
