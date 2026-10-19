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


def release_to_idle(
    port: RuntimePort, policy_mode: str, budget_s: float, clock: Callable[[], float] = time.monotonic
) -> ReleaseOutcome:
    """Return ``policy_mode`` to idle, never over a stop latch. Shared by every exit path.

    Sends ``SetRuntimeMode(idle)`` only when the status read here shows
    ``active_mode == policy_mode`` and no stop latch. In every other case it
    sends nothing and reports what it saw.
    """
    deadline = clock() + budget_s

    def remaining() -> float:
        return max(0.0, deadline - clock())

    status = port.get_status(remaining())
    if status is None:
        return ReleaseOutcome(False, False, "runtime status unavailable; idle not requested")
    if status.active_mode == IDLE:
        return ReleaseOutcome(True, False, "runtime already idle")
    if status.stop_latched:
        return ReleaseOutcome(False, False, "runtime stop is latched; idle is never requested over a latch")
    if status.active_mode != policy_mode:
        return ReleaseOutcome(
            False, False, f"runtime mode {status.active_mode!r} is not {policy_mode!r}; not ours to release"
        )
    success, message = port.set_mode(IDLE, remaining())
    if not success:
        return ReleaseOutcome(False, True, f"idle request refused: {message}")
    after = port.get_status(remaining())
    if after is None or after.active_mode != IDLE:
        mode = None if after is None else after.active_mode
        return ReleaseOutcome(False, True, f"idle requested but runtime reports {mode!r}")
    return ReleaseOutcome(True, True, "runtime returned to idle")
