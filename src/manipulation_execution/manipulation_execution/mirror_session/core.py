"""The mirror session: WAIT / STANDBY / MIRRORING skeleton around one output gate.

The transitions are fixed here; every decision inside them is delegated to a
hook (``hooks.SessionHooks``), whose defaults are conservative no-ops.

======================  ====================  =====================  ==================  ==================
state / phase           ``start``             ``stop``               internal cancel     ``operator_touch``
======================  ====================  =====================  ==================  ==================
WAIT                    take over ->          stay WAIT; release     n/a                 nothing to do
                        STANDBY/approach      (Gateway has switched
                                              to policy_stream)
STANDBY/approach        no-op, success        stop wrap-up -> WAIT;  n/a                 nothing to do
                                              release
STANDBY/cooldown,       no-op, success        close gate -> WAIT;    n/a                 -> MIRRORING
ready                                         release
MIRRORING/running       no-op, success        wrap-up -> WAIT;       wrap-up ->          cancel
                                              release                STANDBY/cooldown
MIRRORING/wrap_up       cancel: success,      cancel: becomes the    n/a                 nothing to do
                        cooldown follows;     stop wrap-up
                        stop: refused
======================  ====================  =====================  ==================  ==================

``start`` only reaches STANDBY. The approach ends in ``ready``; from there
MIRRORING is entered when ``start_criteria`` has held for
``SessionConfig.start_confirm_s``, or at once on ``operator_touch``. A round is
cancelled when ``cancel_criteria`` has held for ``cancel_confirm_s``, or at
once on ``operator_touch``. A false tick restarts either confirm time.

``stop`` wins over a cancel: a stop during the cancel wrap-up turns it into
the stop wrap-up. The cover detector is a ``stop`` raised from inside. Every
change of state or phase is recorded with its reason (``drain_transitions``).

Runtime ownership (``runtime`` module):

* The gate is open exactly while the session holds ``policy_stream``, and the
  gate is the only caller of ``publish``. Every send re-checks it under the
  same lock revocation takes, so nothing is sent after a revocation.
* Revocation (another mode, stop latch, fault, status older than 2 s) closes
  the gate and enters WAIT at once: no wrap-up and no idle request. Recovery
  is an explicit ``start``, never automatic.
* Every exit back to idle calls ``runtime.release_to_idle`` and only after the
  gate is closed: ``stop``, the cover path, startup and shutdown here, and the
  launch exit script in the node module.
* ``stop`` reports success only when the runtime was observed back in idle.
  Wrap-up plus release fit in ``stop_timeout_s``; the wrap-up is cut at
  ``stop_timeout_s - RELEASE_RESERVE_S``. The Gateway checks at startup that
  its ``mirror_stop`` timeout exceeds ``stop_timeout_s`` by
  ``mirror_session_contract.STOP_TIMEOUT_MARGIN_SEC``, so the answer arrives
  before the Gateway gives up.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from embodied_common.mirror_session_contract import COMMANDS
from embodied_common.mirror_session_contract import DEFAULT_STOP_TIMEOUT_SEC as DEFAULT_STOP_TIMEOUT_S
from embodied_common.mirror_session_contract import RELEASE_RESERVE_SEC as RELEASE_RESERVE_S
from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT
from manipulation_execution.mirror_session.hooks import (
    CoverDetector,
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpContext,
    WrapUpPurpose,
    WrapUpStep,
)
from manipulation_execution.mirror_session.runtime import (
    IDLE,
    STATUS_MAX_AGE_S,
    ReleaseOutcome,
    RuntimeObservation,
    RuntimePort,
    WaitReason,
    assess,
    release_to_idle,
    takeover_refusal,
)

TAKEOVER_RPC_TIMEOUT_S = 2.0
SHORT_ROUND_S = 10.0  # rounds shorter than this count towards short_rounds
TRANSITION_BUFFER = 64  # transitions kept until drained; the oldest are dropped beyond it


class SessionState(str, Enum):
    WAIT = "WAIT"
    STANDBY = "STANDBY"
    MIRRORING = "MIRRORING"


class Phase(str, Enum):
    NONE = "none"  # WAIT
    APPROACH = "approach"  # STANDBY, after a takeover, on the way to the reset pose
    COOLDOWN = "cooldown"  # STANDBY, after a cancel: start criteria not evaluated
    READY = "ready"  # STANDBY, start criteria evaluated
    RUNNING = "running"  # MIRRORING, window pipeline
    WRAP_UP = "wrap_up"  # MIRRORING, or STANDBY after a stop during the approach: back to the reset pose


# Transition reason for each way into WAIT; the rest are named after the WaitReason.
_WAIT_TRANSITION_REASONS = {WaitReason.STOPPED: "mirror_stop", WaitReason.REVOKED: "runtime_revoked"}


@dataclass(frozen=True)
class SessionConfig:
    policy_mode: str
    identity: Mapping
    stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S
    status_max_age_s: float = STATUS_MAX_AGE_S
    start_confirm_s: float = 0.0  # start_criteria must hold this long in STANDBY/ready
    cancel_confirm_s: float = 0.0  # cancel_criteria must hold this long in MIRRORING/running

    def __post_init__(self):
        if not isinstance(self.policy_mode, str) or not self.policy_mode or self.policy_mode == IDLE:
            raise ValueError("policy_mode must name the explicit non-idle runtime mode of model_inference")
        if not math.isfinite(self.stop_timeout_s) or self.stop_timeout_s <= RELEASE_RESERVE_S:
            raise ValueError(f"stop_timeout_s must be finite and greater than {RELEASE_RESERVE_S}")
        if not math.isfinite(self.status_max_age_s) or self.status_max_age_s <= 0.0:
            raise ValueError("status_max_age_s must be finite and positive")
        for name in ("start_confirm_s", "cancel_confirm_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and not negative")

    @property
    def wrap_up_budget_s(self) -> float:
        return self.stop_timeout_s - RELEASE_RESERVE_S


def well_formed(command: UpperBodyCommand) -> bool:
    try:
        arm, hand = tuple(command.arm), tuple(command.hand)
    except (AttributeError, TypeError):
        return False
    return (
        len(arm) == ARM_JOINT_COUNT
        and len(hand) == HAND_JOINT_COUNT
        and all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in arm + hand)
    )


def _source_stamp(command) -> float | None:
    stamp = getattr(command, "source_stamp_s", None)
    if isinstance(stamp, bool) or not isinstance(stamp, int | float) or not math.isfinite(stamp):
        return None
    return float(stamp)


class OutputGate:
    """The only path to ``joint.arm_stream`` and ``joint.hand_stream``.

    Closed unless the session holds ``policy_stream``. Checks format only
    (counts, finite values); ranges are application limits in ``post_process``.
    """

    def __init__(self, publish: Callable[[UpperBodyCommand], None]):
        self._publish = publish
        self.is_open = False
        self.last: UpperBodyCommand | None = None
        self.sent = 0
        self.malformed = 0

    def open(self) -> None:
        self.is_open = True
        self.last = None

    def close(self) -> None:
        self.is_open = False

    def emit(self, command: UpperBodyCommand | None) -> bool:
        if not self.is_open or command is None:
            return False
        if not well_formed(command):
            self.malformed += 1
            return False
        command = UpperBodyCommand(
            tuple(map(float, command.arm)), tuple(map(float, command.hand)), _source_stamp(command)
        )
        self._publish(command)
        self.last = command
        self.sent += 1
        return True


class _HookError(Exception):
    pass


class MirrorSession:
    """Thread-safe session core. ``tick`` at 50 Hz; ``request`` and the release block."""

    def __init__(
        self,
        config: SessionConfig,
        port: RuntimePort,
        publish: Callable[[UpperBodyCommand], None],
        hooks: SessionHooks | None = None,
        cover: CoverDetector | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._port = port
        self._hooks = hooks if hooks is not None else SessionHooks()
        self._cover = cover if cover is not None else CoverDetector()
        self._clock = clock
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._release_lock = threading.Lock()
        self._gate = OutputGate(publish)
        self.state = SessionState.WAIT
        self.phase = Phase.NONE
        self.wait_reason: WaitReason | None = WaitReason.BOOT
        self.detail = "startup check pending"
        self.last_release: ReleaseOutcome | None = None
        self._started = False
        self._holding = False
        self._releasing = False
        self._generation = 0
        self._wait_epoch = 0
        self._stop_waiters = 0
        self._last_status: RuntimeObservation | None = None
        self._last_valid_at: float | None = None
        self._measured_arm: tuple[float, ...] | None = None
        self._measured_arm_at: float | None = None
        self._arm_at_takeover: tuple[float, ...] | None = None
        self._phase_started = 0.0
        self._round_started = 0.0
        self._wrap_purpose: WrapUpPurpose | None = None
        self._stop_reason: tuple[WaitReason, str] | None = None
        self._round_open = False
        self._round_error = ""
        self._start_true_since: float | None = None
        self._cancel_true_since: float | None = None
        self._cancel_reason = ""
        self._short_rounds = 0
        self._transitions: deque[dict] = deque(maxlen=TRANSITION_BUFFER)
        self._transitions_dropped = 0
        self._last_transition: dict | None = None

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    @property
    def holding(self) -> bool:
        with self._lock:
            return self._holding

    def observe_status(self, observation: RuntimeObservation) -> None:
        with self._lock:
            self._observe_locked(observation)

    def observe_arm_position(self, arm: Sequence[float]) -> None:
        values = tuple(float(v) for v in arm)
        if len(values) == ARM_JOINT_COUNT and all(math.isfinite(v) for v in values):
            with self._lock:
                self._measured_arm = values
                self._measured_arm_at = self._clock()

    def _observe_locked(self, observation: RuntimeObservation) -> None:
        self._last_status = observation
        problem = assess(observation, self._config.identity, self._config.status_max_age_s)
        if problem is None:
            self._last_valid_at = self._clock()
        if not self._holding:
            return
        if problem is not None:
            self._enter_wait_locked(*problem)
        elif observation.active_mode != self._config.policy_mode:
            self._enter_wait_locked(WaitReason.REVOKED, f"runtime switched to {observation.active_mode!r}")

    # ------------------------------------------------------------------
    # ~/set_session and the operator touch
    # ------------------------------------------------------------------

    def request(self, command: str) -> tuple[bool, str]:
        if command == "start":
            return self._start()
        if command == "stop":
            return self._stop()
        return False, f"unknown session command {command!r}; accepted: {', '.join(COMMANDS)}"

    def _start(self) -> tuple[bool, str]:
        with self._lock:
            if not self._started:
                return False, "startup check has not finished"
            if self.phase == Phase.WRAP_UP:
                if self._wrap_purpose == WrapUpPurpose.STOP:
                    return False, "stop in progress"
                return True, "round ending; cooldown follows the wrap-up"
            if self.state != SessionState.WAIT:
                return True, "already on"
            generation = self._generation

        # WAIT: the Gateway switched the runtime before this call. Read, then take over.
        status = self._port.get_status(TAKEOVER_RPC_TIMEOUT_S)
        with self._lock:
            if status is not None:
                self._observe_locked(status)
            if self.state != SessionState.WAIT:
                return True, "already holding"
            if generation != self._generation:
                return False, "stop arrived during takeover"
            refusal = takeover_refusal(status, self._config.identity, self._config.policy_mode)
            if refusal is not None:
                self.detail = f"start refused: {refusal}"
                return False, refusal
            self._holding = True
            self._gate.open()
            self._arm_at_takeover = self._measured_arm
            self._phase_started = self._clock()
            self._wrap_purpose = WrapUpPurpose.APPROACH
            self.wait_reason, self.detail = None, "took over policy_stream"
            self._transition_locked(SessionState.STANDBY, Phase.APPROACH, "mirror_start")
            self._changed.notify_all()
            return True, "took over policy_stream"

    def _stop(self) -> tuple[bool, str]:
        with self._lock:
            self._generation += 1
            stopping = self.state == SessionState.MIRRORING or self.phase in (Phase.APPROACH, Phase.WRAP_UP)
            if self.state == SessionState.MIRRORING and self.phase != Phase.WRAP_UP:
                self._begin_wrap_locked(WrapUpPurpose.STOP, "mirror_stop", (WaitReason.STOPPED, "mirror_stop"))
            elif self.phase == Phase.APPROACH:
                # The arm may be anywhere on its way to the reset pose: finish there, on a budget of its own.
                self._wrap_purpose = WrapUpPurpose.STOP
                self._stop_reason = (WaitReason.STOPPED, "mirror_stop")
                self._phase_started = self._clock()
                self._transition_locked(SessionState.STANDBY, Phase.WRAP_UP, "mirror_stop")
            elif self.phase == Phase.WRAP_UP and self._wrap_purpose != WrapUpPurpose.STOP:
                # Stop wins over cancel: the cancel wrap-up continues as the stop one.
                self._wrap_purpose = WrapUpPurpose.STOP
                self._stop_reason = (WaitReason.STOPPED, "mirror_stop")
            if stopping:
                epoch = self._wait_epoch
                self._stop_waiters += 1
                try:
                    finished = self._changed.wait_for(
                        lambda: self._wait_epoch != epoch, timeout=self._config.wrap_up_budget_s
                    )
                finally:
                    self._stop_waiters -= 1
                if not finished:
                    self._enter_wait_locked(WaitReason.STOPPED, "wrap-up overran its budget", "wrap_up_overrun")
            elif self.state == SessionState.STANDBY:
                # cooldown / ready: the arm is at the reset pose already.
                self._enter_wait_locked(WaitReason.STOPPED, "mirror_stop")
        # WAIT included: the Gateway switched to policy_stream before this call.
        outcome = self._release()
        return outcome.idle, outcome.message

    def operator_touch(self) -> tuple[bool, str]:
        """Head touch: start from STANDBY/cooldown or ready, cancel a running round; no confirm time."""
        with self._lock:
            if not self._started:
                return False, "startup check has not finished"
            if self.state == SessionState.STANDBY and self.phase in (Phase.COOLDOWN, Phase.READY):
                try:
                    self._enter_mirroring_locked(self._clock(), "operator_touch")
                except _HookError as error:
                    release = self._hook_failed_locked(error)
                    message = str(error)
                else:
                    return True, "standby -> mirroring"
            elif (self.state, self.phase) == (SessionState.MIRRORING, Phase.RUNNING):
                self._cancel_reason = "operator_touch"
                self._begin_wrap_locked(WrapUpPurpose.CANCEL, "operator_touch")
                return True, "round cancelled"
            else:
                return False, f"{self.state.value}/{self.phase.value}: nothing to do"
        if release:
            self._release()
        return False, message

    # ------------------------------------------------------------------
    # exits back to idle
    # ------------------------------------------------------------------

    def startup(self) -> ReleaseOutcome:
        """Restart fallback: a policy_stream left behind (no latch) is ours; release it, stay WAIT."""
        outcome = self._release()
        with self._lock:
            self._started = True
            self.detail = f"startup check: {outcome.message}"
        return outcome

    def shutdown(self) -> ReleaseOutcome:
        """Normal exit: close the gate, then the same release."""
        with self._lock:
            self._started = False
            if self.state != SessionState.WAIT:
                self._enter_wait_locked(WaitReason.SHUTDOWN, "node exiting")
        return self._release()

    def _release(self) -> ReleaseOutcome:
        with self._release_lock:
            with self._lock:
                # Every caller closes the gate first, then drops the session lock.
                # A start that takes over in that gap holds the mode now, and
                # idle is not requested from under a holder.
                if self._holding:
                    outcome = ReleaseOutcome(False, False, "a later start holds the mode; idle not requested")
                    self.last_release = outcome
                    return outcome
                self._releasing = True
            outcome = None
            try:
                outcome = release_to_idle(self._port, self._config.policy_mode, RELEASE_RESERVE_S, self._clock)
            finally:
                with self._lock:
                    self._releasing = False
                    if outcome is not None:
                        self.last_release = outcome
            return outcome

    # ------------------------------------------------------------------
    # 50 Hz
    # ------------------------------------------------------------------

    def tick(self, inputs: SessionInputs | None = None) -> None:
        inputs = inputs if inputs is not None else SessionInputs()
        with self._lock:
            try:
                release = self._tick_locked(self._clock(), inputs)
            except _HookError as error:
                release = self._hook_failed_locked(error)
        if release:
            self._release()

    def _tick_locked(self, now: float, inputs: SessionInputs) -> bool:
        """One step; True when the caller must release (an internal stop nobody waits on)."""
        if self._holding:
            age = None if self._last_valid_at is None else now - self._last_valid_at
            if age is None or age > self._config.status_max_age_s:
                self._enter_wait_locked(WaitReason.RUNTIME_LOST, "no fresh runtime status")
                return False
        if self.state == SessionState.STANDBY:
            return self._tick_standby_locked(now, inputs)
        if self.state == SessionState.MIRRORING:
            return self._tick_mirroring_locked(now, inputs)
        return False

    def _tick_standby_locked(self, now: float, inputs: SessionInputs) -> bool:
        target = self._reset_target()
        if self.phase == Phase.WRAP_UP:
            ending = self._wrap_tick_locked(now, target)
            return ending is not None and self._end_stop_wrap_locked(ending)
        if self._covered(inputs):
            self._enter_wait_locked(WaitReason.COVERED, "camera covered")
            return self._stop_waiters == 0
        elapsed = now - self._phase_started
        if self.phase == Phase.APPROACH:
            step = self._wrap_step(WrapUpPurpose.APPROACH, target, elapsed)
            self._emit(step.command, target)
            if step.done or elapsed >= self._config.wrap_up_budget_s:
                self._wrap_purpose, self._phase_started = None, now
                self._transition_locked(SessionState.STANDBY, Phase.READY, "approach_done")
            return False
        if self.phase == Phase.COOLDOWN and self._hook("cooldown_over", self._hooks.cooldown_over, elapsed):
            self._phase_started = now
            self._transition_locked(SessionState.STANDBY, Phase.READY, "cooldown_done")
        self._emit(target, target)
        if self.phase == Phase.READY:
            if self._hook("start_criteria", self._hooks.start_criteria, self._hook_inputs(inputs, now)):
                if self._start_true_since is None:
                    self._start_true_since = now
                if now - self._start_true_since >= self._config.start_confirm_s:
                    self._enter_mirroring_locked(now, "criteria")
            else:
                self._start_true_since = None
        return False

    def _tick_mirroring_locked(self, now: float, inputs: SessionInputs) -> bool:
        target = self._reset_target()
        if self.phase == Phase.RUNNING:
            if self._covered(inputs):
                self._begin_wrap_locked(WrapUpPurpose.STOP, "covered", (WaitReason.COVERED, "camera covered"))
            else:
                inputs = self._hook_inputs(inputs, now)
                self._check_cancel_locked(now, inputs)
            if self.phase == Phase.RUNNING:
                command = self._hook("retarget", self._hooks.retarget, inputs)
                if command is not None:
                    command = self._hook("post_process", self._hooks.post_process, command)
                self._emit(command, target)
                return False
        elif self._wrap_purpose == WrapUpPurpose.CANCEL and self._covered(inputs):
            self._wrap_purpose = WrapUpPurpose.STOP
            self._stop_reason = (WaitReason.COVERED, "camera covered")

        ending = self._wrap_tick_locked(now, target)
        if ending is None:
            return False
        if self._wrap_purpose == WrapUpPurpose.STOP:
            return self._end_stop_wrap_locked(ending)
        self._wrap_purpose, self._phase_started = None, now
        self._transition_locked(SessionState.STANDBY, Phase.COOLDOWN, ending)
        return False

    def _check_cancel_locked(self, now: float, inputs: SessionInputs) -> None:
        if not self._hook("cancel_criteria", self._hooks.cancel_criteria, inputs, now - self._round_started):
            self._cancel_true_since = None
            return
        if self._cancel_true_since is None:
            self._cancel_true_since = now
        if now - self._cancel_true_since < self._config.cancel_confirm_s:
            return
        reason = self._hook("cancel_reason", getattr, self._hooks, "cancel_reason", "")
        if not isinstance(reason, str):
            raise _HookError(f"cancel_reason is {type(reason).__name__}, not str")
        self._cancel_reason = reason or "criteria"
        self._begin_wrap_locked(WrapUpPurpose.CANCEL, self._cancel_reason)

    # ------------------------------------------------------------------
    # helpers (lock held)
    # ------------------------------------------------------------------

    def _hook_inputs(self, inputs: SessionInputs, now: float) -> SessionInputs:
        """The fields only the core knows; ``publish_enabled`` stays as the node set it."""
        return dataclasses.replace(
            inputs,
            measured_arm=self._measured_arm,
            measured_arm_age_s=math.inf if self._measured_arm_at is None else now - self._measured_arm_at,
            last_sent=self._gate.last,
            phase_elapsed_s=now - self._phase_started,
        )

    def _emit(self, command: UpperBodyCommand | None, fallback: UpperBodyCommand | None) -> None:
        """Send ``command``; None holds the last command, or the reset pose before any."""
        if command is None:
            command = self._gate.last if self._gate.last is not None else fallback
        self._gate.emit(command)

    def _reset_target(self) -> UpperBodyCommand | None:
        if self._arm_at_takeover is None:
            self._arm_at_takeover = self._measured_arm
        return self._hook("reset_pose", self._hooks.reset_pose, self._arm_at_takeover)

    def _wrap_step(self, purpose: WrapUpPurpose, target, elapsed: float) -> WrapUpStep:
        context = WrapUpContext(
            purpose, self._gate.last, self._measured_arm, target, elapsed, self._config.wrap_up_budget_s
        )
        step = self._hook("wrap_up", self._hooks.wrap_up, context)
        if not isinstance(step, WrapUpStep):
            raise _HookError(f"wrap_up returned {type(step).__name__}, not WrapUpStep")
        return step

    def _wrap_tick_locked(self, now: float, target) -> str | None:
        """One wrap-up step; ``wrap_up_done`` / ``wrap_up_overrun`` once it ends, else None."""
        elapsed = now - self._phase_started
        step = self._wrap_step(self._wrap_purpose, target, elapsed)
        self._emit(step.command, target)
        if step.done:
            return "wrap_up_done"
        if elapsed >= self._config.wrap_up_budget_s:
            return "wrap_up_overrun"
        return None

    def _end_stop_wrap_locked(self, ending: str) -> bool:
        """The stop wrap-up ended: WAIT. True when nobody waits on it, so the caller releases."""
        release = self._stop_waiters == 0
        reason, detail = self._stop_reason
        self._enter_wait_locked(reason, detail, "wrap_up_overrun" if ending == "wrap_up_overrun" else None)
        return release

    def _covered(self, inputs: SessionInputs) -> bool:
        frame = inputs.camera_frame
        return frame is not None and bool(self._hook("CoverDetector.update", self._cover.update, frame))

    def _hook(self, name: str, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            raise _HookError(f"{name} raised {exc!r}") from exc

    def _hook_failed_locked(self, error: _HookError) -> bool:
        """A hook raised: stop like ``mirror_stop`` without the wrap-up."""
        release = self._holding and self._stop_waiters == 0
        self._enter_wait_locked(WaitReason.HOOK_ERROR, str(error))
        return release

    def _begin_wrap_locked(self, purpose: WrapUpPurpose, reason: str, stop_reason=None) -> None:
        self._phase_started = self._clock()
        self._wrap_purpose = purpose
        if purpose == WrapUpPurpose.STOP:
            self._stop_reason = stop_reason
        self._close_round_locked()
        self._transition_locked(self.state, Phase.WRAP_UP, reason)

    def _enter_mirroring_locked(self, now: float, reason: str) -> None:
        self._round_started = self._phase_started = now
        self._wrap_purpose = None
        self._round_open, self._round_error = True, ""
        self._transition_locked(SessionState.MIRRORING, Phase.RUNNING, reason)
        self._hook("begin_round", self._hooks.begin_round)

    def _close_round_locked(self) -> None:
        """``end_round`` once per round. A raise goes to the status and on into WAIT's detail; never raised."""
        if not self._round_open:
            return
        self._round_open = False
        if self._clock() - self._round_started < SHORT_ROUND_S:
            self._short_rounds += 1
        else:
            self._short_rounds = 0
        try:
            self._hooks.end_round()
        except Exception as exc:
            self._round_error = self.detail = f"end_round raised {exc!r}"

    def _enter_wait_locked(self, reason: WaitReason, detail: str, transition: str | None = None) -> None:
        self._gate.close()
        self._holding = False
        self._close_round_locked()
        error, self._round_error = self._round_error, ""
        self.wait_reason, self.detail = reason, f"{detail}; {error}" if error else detail
        self._wrap_purpose, self._stop_reason = None, None
        if transition is None:
            transition = _WAIT_TRANSITION_REASONS.get(reason, reason.value)
        self._transition_locked(SessionState.WAIT, Phase.NONE, transition)
        self._wait_epoch += 1
        self._changed.notify_all()

    def _transition_locked(self, state: SessionState, phase: Phase, reason: str) -> None:
        """Every change of state or phase goes through here; the other fields are set before."""
        before = f"{self.state.value}/{self.phase.value}"
        self.state, self.phase = state, phase
        self._start_true_since = self._cancel_true_since = None
        after = f"{state.value}/{phase.value}"
        if after == before:
            return
        record = {
            "from": before,
            "to": after,
            "reason": reason,
            "t": self._clock(),
            "state": state.value,
            "phase": phase.value,
            "wrap_purpose": "" if self._wrap_purpose is None else self._wrap_purpose.value,
            "wait_reason": None if self.wait_reason is None else self.wait_reason.value,
            "holding": self._holding,
            "cancel_reason": self._cancel_reason,
            "short_rounds": self._short_rounds,
        }
        if len(self._transitions) == self._transitions.maxlen:
            self._transitions_dropped += 1
        self._transitions.append(record)
        self._last_transition = record

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def drain_transitions(self) -> list[dict]:
        """The transitions since the last call, oldest first."""
        with self._lock:
            records = [dict(record) for record in self._transitions]
            self._transitions.clear()
            return records

    def snapshot(self) -> dict:
        with self._lock:
            status = self._last_status
            return {
                "state": self.state.value,
                "phase": self.phase.value,
                "wrap_purpose": "" if self._wrap_purpose is None else self._wrap_purpose.value,
                "wait_reason": None if self.wait_reason is None else self.wait_reason.value,
                "detail": self.detail,
                "holding": self._holding,
                "releasing": self._releasing,
                "runtime_mode": None if status is None else status.active_mode,
                "commands_sent": self._gate.sent,
                "commands_malformed": self._gate.malformed,
                "last_release": None if self.last_release is None else self.last_release.message,
                "cancel_reason": self._cancel_reason,
                "short_rounds": self._short_rounds,
                "last_transition": None if self._last_transition is None else dict(self._last_transition),
                "transitions_dropped": self._transitions_dropped,
            }
