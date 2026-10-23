"""The runtime boundary: what a status means for ownership, and the one release.

Rules, after ``action_dispatch.policy_admission`` with two differences the
mirror needs:

* **Take over, never acquire.** The Gateway switches the runtime to
  ``policy_stream`` before it calls ``mirror_start``; the session takes over
  that mode. There is no path that requests ``policy_stream`` from idle. This
  is sound only because the mirror is the sole ``policy_stream`` producer in
  its configuration, which is also what lets the exit paths treat a lingering
  ``policy_stream`` as ours.
* **One release, never over a stop latch.** ``SetRuntimeMode(idle)`` is
  sent only when ``active_mode == policy_stream`` is observed and no stop is
  latched. Requesting idle clears the stop latch, so it must never be sent
  over one. Every exit path calls :func:`release_to_idle`; none sends idle any
  other way.

Revocation (another mode, stop latch, fault, stale status) is never answered
with a request: the caller closes its output and waits for an explicit start.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

IDLE = "idle"
STATUS_MAX_AGE_S = 2.0


class WaitReason(str, Enum):
    """Why the session is in WAIT."""

    BOOT = "boot"
    STOPPED = "stopped"
    COVERED = "covered"
    REVOKED = "revoked"
    STOP_LATCHED = "stop_latched"
    FAULT = "fault"
    RUNTIME_LOST = "runtime_lost"
    HOOK_ERROR = "hook_error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class RuntimeObservation:
    """The ``RuntimeStatus`` fields ownership depends on, ROS-free.

    ``robot`` is the ``robot`` object of ``interface_description_json`` (None
    when unparsable); ``stamp_age_s`` is receipt time minus the status stamp,
    None when the stamp is malformed.
    """

    lifecycle: str
    active_mode: str
    stop_latched: bool
    faults: tuple[str, ...]
    runtime_name: str
    runtime_version: str
    robot: Mapping | None
    stamp_age_s: float | None


class RuntimePort(Protocol):
    """Blocking runtime calls. ``None`` / ``(False, reason)`` on timeout or no service."""

    def get_status(self, timeout_s: float) -> RuntimeObservation | None: ...

    def set_mode(self, mode: str, timeout_s: float) -> tuple[bool, str]: ...

    def stop(self, policy: str, timeout_s: float) -> tuple[bool, str]: ...


def assess(
    observation: RuntimeObservation | None, identity: Mapping | None, max_age_s: float = STATUS_MAX_AGE_S
) -> tuple[WaitReason, str] | None:
    """None for a fresh, healthy status from the bound runtime; otherwise why it is not.

    The active mode is judged separately: a healthy status in another mode is
    a revocation only for a holder.
    """
    if observation is None:
        return WaitReason.RUNTIME_LOST, "runtime status unavailable"
    age = observation.stamp_age_s
    if age is None or not 0.0 <= age <= max_age_s:
        return WaitReason.RUNTIME_LOST, "runtime status stamp is not fresh"
    if identity is not None and (
        observation.robot != dict(identity)
        or observation.runtime_name != identity.get("runtime_name")
        or observation.runtime_version != identity.get("runtime_version")
    ):
        return WaitReason.REVOKED, "runtime identity does not match the bound description"
    if observation.stop_latched:
        return WaitReason.STOP_LATCHED, "runtime stop is latched"
    if observation.lifecycle != "ACTIVE" or observation.faults:
        detail = "; ".join(observation.faults) or "no detail"
        return WaitReason.FAULT, f"runtime {observation.lifecycle}: {detail}"
    return None


def takeover_refusal(
    observation: RuntimeObservation | None, identity: Mapping | None, policy_mode: str
) -> str | None:
    """None when the Gateway-switched ``policy_mode`` may be taken over; else why not."""
    problem = assess(observation, identity)
    if problem is not None:
        return problem[1]
    if observation.active_mode == IDLE:
        return "runtime is idle: the session only takes over a mode the Gateway switched to"
    if observation.active_mode != policy_mode:
        return f"runtime mode {observation.active_mode!r} is owned by another producer"
    return None


@dataclass(frozen=True)
class ReleaseOutcome:
    idle: bool  # idle was observed afterwards: the only condition mirror_stop reports success on
    sent_idle: bool
    message: str


# Before the idle request the output has to run out. The gate is closed when a
# release starts, but the runtime repeats the last arm command for its
# ``stream_staleness_s`` (0.5 s in the aimdk profiles) and the arm needs a moment
# to settle on it. R1 (10-19) asked 60 ms after the gate closed, with that
# stream still live; the platform answered MOTION_BUSY and stayed in
# UPPERBODY_REMOTE_SPLIT until a reboot.
RELEASE_DRAIN_S = 0.8
# A refused idle request is asked again after this long, within the budget.
RELEASE_RETRY_S = 0.3
# Also bounds the retries when the clock does not move (tests).
RELEASE_MAX_ATTEMPTS = 4


def _busy_refusal(message: str) -> bool:
    """True when the platform refused the mode switch because it is still playing."""
    lowered = message.lower()
    return "motion_busy" in lowered or "busy" in lowered or "\u6b63\u5728\u64ad\u653e" in message


def _clip_outcome_message(message: str, limit: int = 240) -> str:
    """Keep a release failure inside the Agent contract's 300-character field.

    The Agent rejects a terminal message longer than 300 characters, and a
    refused release message repeats the platform answer per attempt; the whole
    request then ends UNKNOWN and the robot scope stays quarantined.
    """
    return message if len(message) <= limit else message[: limit - 3] + "..."


def release_to_idle(
    port: RuntimePort,
    policy_mode: str,
    budget_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    held: Callable[[], bool] = lambda: False,
) -> ReleaseOutcome:
    """Return ``policy_mode`` to idle, never over a stop latch. Shared by every exit path.

    Sends ``SetRuntimeMode(idle)`` only when the status read here shows
    ``active_mode == policy_mode`` and no stop latch. In every other case it
    sends nothing and reports what it saw. The request waits
    ``RELEASE_DRAIN_S`` for the output to run out, and a refusal is retried
    while the budget lasts; the status is read again before every request, and
    ``held()`` true (a later start took the mode over meanwhile) ends it
    without one. The message lists every request with its time from the call.
    """
    started = clock()
    deadline = started + budget_s
    attempts: list[str] = []

    def remaining() -> float:
        return max(0.0, deadline - clock())

    def history() -> str:
        return f" [{'; '.join(attempts)}]" if attempts else ""

    def unreleasable(status: RuntimeObservation | None) -> ReleaseOutcome | None:
        sent = bool(attempts)
        if status is None:
            return ReleaseOutcome(False, sent, "runtime status unavailable; idle not requested" + history())
        if status.active_mode == IDLE:
            return ReleaseOutcome(True, sent, "runtime already idle" + history())
        if status.stop_latched:
            return ReleaseOutcome(
                False, sent, _clip_outcome_message("runtime stop is latched; idle is never requested over a latch" + history())
            )
        if status.active_mode != policy_mode:
            message = f"runtime mode {status.active_mode!r} is not {policy_mode!r}; not ours to release"
            return ReleaseOutcome(False, sent, _clip_outcome_message(message + history()))
        if held():
            return ReleaseOutcome(False, sent, _clip_outcome_message("a later start holds the mode; idle not requested" + history()))
        return None

    outcome = unreleasable(port.get_status(remaining()))
    if outcome is not None:
        return outcome
    sleep(min(RELEASE_DRAIN_S, remaining()))
    stop_requested = False
    while True:
        outcome = unreleasable(port.get_status(remaining()))
        if outcome is not None:
            return outcome
        success, message = port.set_mode(IDLE, remaining())
        if not success and not stop_requested and _busy_refusal(message) and hasattr(port, "stop"):
            # The platform refuses STAND_DEFAULT while an action it dispatched is
            # still playing. HOLD cancels that action; the idle request that
            # follows clears the stop latch, so the release can go through.
            stop_requested = True
            stop_ok, stop_message = port.stop("HOLD", remaining())
            attempts.append(f"{clock() - started:.2f} s stop(HOLD) {'accepted' if stop_ok else 'refused'}: {stop_message}")
            success, message = port.set_mode(IDLE, remaining())
        attempts.append(f"{clock() - started:.2f} s {'accepted' if success else 'refused'}: {message}")
        if success:
            break
        if len(attempts) >= RELEASE_MAX_ATTEMPTS or remaining() < RELEASE_RETRY_S:
            return ReleaseOutcome(
                False, True, _clip_outcome_message(f"idle request refused {len(attempts)} times: {message}" + history())
            )
        sleep(RELEASE_RETRY_S)
    after = port.get_status(remaining())
    if after is None or after.active_mode != IDLE:
        mode = None if after is None else after.active_mode
        return ReleaseOutcome(False, True, _clip_outcome_message(f"idle requested but runtime reports {mode!r}" + history()))
    return ReleaseOutcome(True, True, "runtime returned to idle" + history())
