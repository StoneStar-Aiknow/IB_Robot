"""ROS-free lifecycle for one opt-in named sub-action of a mirror round.

The caller retains the public session state and closes its single arm/hand
output gate whenever ``block_public`` is true. All port methods must enqueue
work or return a plain snapshot; none may wait for ROS while the core is locked.
This module never authorizes an asset or a path. Those checks remain mandatory
at the adapter/runtime boundary, including when ``enabled`` is explicitly set.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT
from manipulation_execution.mirror_session.hooks import UpperBodyCommand


class PresetPhase(str, Enum):
    IDLE = "idle"
    DRAINING = "draining"
    ENTERING = "entering"
    HOLDING = "holding"
    EXITING = "exiting"
    CANCELLING = "cancelling"
    STOPPED = "stopped"
    FAULT = "fault"


@dataclass(frozen=True)
class PresetEpoch:
    generation: int
    action: int


@dataclass(frozen=True)
class PresetConfig:
    enabled: bool = False
    motion_name: str = "overhead_heart"
    drain_s: float = 0.6
    lease_period_s: float = 0.1
    lease_timeout_s: float = 1.0
    snapshot_max_age_s: float = 1.0
    final_target_max_age_s: float = 0.5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.motion_name, str) or not self.motion_name.strip():
            raise ValueError("motion_name must be a nonempty configured name")
        for name in ("drain_s", "lease_period_s", "lease_timeout_s", "snapshot_max_age_s", "final_target_max_age_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.drain_s < 0.6:
            raise ValueError("drain_s must cover at least 0.6 s of quiet public arm and hand streams")
        if self.lease_period_s >= self.snapshot_max_age_s:
            raise ValueError("lease_period_s must be smaller than snapshot_max_age_s")
        if self.lease_period_s >= self.lease_timeout_s or self.lease_timeout_s > 1.0:
            raise ValueError("lease_period_s must be below lease_timeout_s, which cannot exceed 1 s")


@dataclass(frozen=True)
class PresetSnapshot:
    """Adapter evidence for one action, timestamped with the injected clock.

    ``observed_at`` is the latest goal/feedback/result evidence, not the time
    ``snapshot()`` was called. ``final_target_verified`` asserts server success
    and a target identity matching this UUID's submitted exit, not cached feedback
    or the pre-preset public command. A terminal alone is not release proof.
    """

    epoch: PresetEpoch | None = None
    stage: str = "idle"
    pending: bool = False
    accepted: bool = False
    terminal: bool = False
    success: bool = False
    release_confirmed: bool = False
    goal_id: str = ""
    final_target: UpperBodyCommand | None = None
    final_target_at: float | None = None
    final_target_verified: bool = False
    fault: str = ""
    observed_at: float | None = None


class PresetPort(Protocol):
    """A single-flight asynchronous adapter, including pending-goal cancellation.

    ``begin`` sends the first UUID-bound lease before dispatch. ``cancel``
    remembers its intent if acceptance is pending and cancels late acceptance.
    Old callbacks must not replace the current epoch's snapshot; the adapter
    cancels an old accepted UUID directly instead. ``refresh`` renews only the
    current UUID and optionally asks it to run its checked same-owner exit.
    """

    def begin(self, name: str, epoch: PresetEpoch) -> None: ...

    def refresh(self, exit_target: UpperBodyCommand | None = None) -> None: ...

    def cancel(self, reason: str) -> None: ...

    def snapshot(self) -> PresetSnapshot: ...


@dataclass(frozen=True)
class PresetStep:
    block_public: bool = False
    resume_seed: UpperBodyCommand | None = None
    fault: str = ""
    phase: PresetPhase = PresetPhase.IDLE
    release_confirmed: bool = True


def _finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _command(command: UpperBodyCommand | None) -> UpperBodyCommand | None:
    if not isinstance(command, UpperBodyCommand):
        return None
    try:
        arm, hand = tuple(command.arm), tuple(command.hand)
    except TypeError:
        return None
    if len(arm) != ARM_JOINT_COUNT or len(hand) != HAND_JOINT_COUNT or not all(_finite(v) for v in arm + hand):
        return None
    return UpperBodyCommand(tuple(map(float, arm)), tuple(map(float, hand)))


class PresetCoordinator:
    """One request, one named goal, periodic leases, and an explicit release barrier.

    Call ``request`` and then ``step`` in the tick that closes both public
    channels. ``public_quiet`` attests that both channels remain closed; any
    false observation restarts the whole drain interval. A stop/fault cannot
    be undone in the same round. A later generation may request again only
    after the previous owner has positively released.
    """

    def __init__(
        self,
        config: PresetConfig = PresetConfig(),
        port: PresetPort | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._port = port
        self._clock = clock
        self.phase = PresetPhase.IDLE
        self.epoch: PresetEpoch | None = None
        self._action = 0
        self._quiet_since: float | None = None
        self._began_at: float | None = None
        self._lease_at: float | None = None
        self._last_now: float | None = None
        self._begun = False
        self._released = True
        self._goal_id = ""
        self._cancelled_goal_id: str | None = None
        self._stop_reason = ""
        self._fault = ""
        self._exit_target: UpperBodyCommand | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self._port is not None

    def request(self, generation: int) -> bool:
        """Start draining for the fixed motion; repeated matches cannot re-dispatch."""
        if not self.enabled or isinstance(generation, bool) or not isinstance(generation, int):
            return False
        if self.phase != PresetPhase.IDLE and (
            not self._released or self.epoch is None or generation == self.epoch.generation
        ):
            return False
        self._action += 1
        self.epoch = PresetEpoch(generation, self._action)
        self.phase = PresetPhase.DRAINING
        self._quiet_since = self._began_at = self._lease_at = self._last_now = None
        self._begun = False
        self._released = True
        self._goal_id = ""
        self._cancelled_goal_id = None
        self._stop_reason = self._fault = ""
        self._exit_target = None
        return True

    def stop(self, reason: str = "stop") -> PresetStep:
        """Stop renewing immediately, including while goal acceptance is pending."""
        if not self.enabled or self.phase == PresetPhase.IDLE:
            return self._result()
        self._cancel(reason or "stop")
        return self._result()

    def step(
        self,
        generation: int,
        human_active: bool,
        exit_target: UpperBodyCommand | None = None,
        exit_target_fresh: bool = False,
        stop_reason: str = "",
        public_quiet: bool = True,
    ) -> PresetStep:
        """Advance one nonblocking tick.

        ``exit_target_fresh`` means only a current, complete 34-value mirror
        target. It is not geometric approval. The named runtime must validate
        the actual exit path before commanding it under the existing owner.
        """
        if not self.enabled or self.phase == PresetPhase.IDLE:
            return self._result()
        now = self._clock()
        if not _finite(now) or (self._last_now is not None and now < self._last_now):
            self._cancel("clock_invalid", fault="preset_clock_invalid")
            return self._result()
        self._last_now = now
        if stop_reason:
            self._cancel(stop_reason)
        elif self.epoch is None or generation != self.epoch.generation:
            self._cancel("generation_changed", fault="preset_generation_changed")
        elif self._begun and not public_quiet:
            self._cancel("public_stream_active", fault="preset_public_stream_active")
        elif self._lease_at is not None and now - self._lease_at > self.config.lease_timeout_s:
            self._cancel("lease_lapsed", fault="preset_lease_lapsed")
        if self._begun:
            snapshot = self._snapshot(now)
            if snapshot is not None:
                result = self._consume(snapshot, now)
                if result is not None:
                    return result
        if self._stop_reason:
            return self._result()
        if self.phase == PresetPhase.DRAINING:
            if not human_active:
                self.phase = PresetPhase.IDLE  # no named goal was dispatched
                return self._result()
            if not public_quiet:
                self._quiet_since = None
            elif self._quiet_since is None:
                self._quiet_since = now
            elif now >= self._quiet_since + self.config.drain_s:
                self._begun = True
                self._released = False
                self._began_at = now
                self.phase = PresetPhase.ENTERING
                if self._call("begin", self.config.motion_name, self.epoch):
                    self._refresh(now)
            return self._result()
        if not human_active and self.phase != PresetPhase.EXITING:
            target = _command(exit_target)
            if exit_target_fresh is not True or target is None:
                self._cancel("exit_target_unavailable", fault="preset_exit_target_unavailable")
                return self._result()
            self._exit_target = target
            self.phase = PresetPhase.EXITING
            self._refresh(now, force=True)
        elif self._begun:
            self._refresh(now)
        return self._result()

    def _call(self, method: str, *args) -> bool:
        try:
            getattr(self._port, method)(*args)
            return True
        except Exception as exc:
            self._fault = f"preset_port_{method}:{type(exc).__name__}"
            self._stop_reason = self._stop_reason or self._fault
            self.phase = PresetPhase.FAULT
            if method != "cancel" and self._begun:
                self._call("cancel", self._stop_reason)
            return False

    def _cancel(self, reason: str, *, fault: str = "") -> None:
        if fault and not self._fault:
            self._fault = fault
        first = not self._stop_reason
        self._stop_reason = self._stop_reason or reason
        self.phase = PresetPhase.FAULT if self._fault else PresetPhase.CANCELLING
        if not self._begun:
            self.phase = PresetPhase.FAULT if self._fault else PresetPhase.STOPPED
        elif first:
            self._cancelled_goal_id = self._goal_id
            self._call("cancel", self._stop_reason)

    def _refresh(self, now: float, *, force: bool = False) -> None:
        if self._stop_reason:
            return
        if (force or self._lease_at is None or now - self._lease_at >= self.config.lease_period_s) and self._call(
            "refresh", self._exit_target
        ):
            self._lease_at = now

    def _snapshot(self, now: float) -> PresetSnapshot | None:
        try:
            snapshot = self._port.snapshot()
        except Exception as exc:
            self._cancel("snapshot_failed", fault=f"preset_port_snapshot:{type(exc).__name__}")
            return None
        if not isinstance(snapshot, PresetSnapshot) or snapshot.epoch != self.epoch:
            if self._began_at is not None and now - self._began_at > self.config.snapshot_max_age_s:
                self._cancel("snapshot_epoch_unavailable", fault="preset_snapshot_epoch_unavailable")
            return None
        if not _finite(snapshot.observed_at) or not 0 <= now - snapshot.observed_at <= self.config.snapshot_max_age_s:
            self._cancel("status_stale", fault="preset_status_stale")
            return None
        if snapshot.accepted:
            if not snapshot.goal_id or (self._goal_id and snapshot.goal_id != self._goal_id):
                self._cancel("goal_identity_changed", fault="preset_goal_identity_changed")
                return None
            self._goal_id = snapshot.goal_id
        elif self._goal_id and snapshot.goal_id != self._goal_id:
            self._cancel("goal_identity_changed", fault="preset_goal_identity_changed")
            return None
        return snapshot

    def _consume(self, snapshot: PresetSnapshot, now: float) -> PresetStep | None:
        if snapshot.fault:
            self._cancel(snapshot.fault, fault=snapshot.fault)
        if self._stop_reason:
            if snapshot.accepted and self._cancelled_goal_id != snapshot.goal_id and not snapshot.terminal:
                self._cancelled_goal_id = snapshot.goal_id
                self._call("cancel", self._stop_reason)
            if snapshot.terminal and snapshot.release_confirmed and not snapshot.pending:
                self._released = True
                self.phase = PresetPhase.FAULT if self._fault else PresetPhase.STOPPED
            return self._result()
        if snapshot.terminal:
            self._released = snapshot.release_confirmed and not snapshot.pending
            if self.phase != PresetPhase.EXITING or not snapshot.success:
                self._cancel("unexpected_terminal", fault="preset_unexpected_terminal")
                return self._result()
            seed = _command(snapshot.final_target)
            valid = (
                self._released
                and snapshot.accepted
                and bool(self._goal_id)
                and snapshot.final_target_verified is True
                and seed is not None
                and seed == self._exit_target
                and _finite(snapshot.final_target_at)
                and 0 <= now - snapshot.final_target_at <= self.config.final_target_max_age_s
            )
            if not valid:
                self._cancel("resume_evidence_invalid", fault="preset_resume_evidence_invalid")
                return self._result()
            self.phase = PresetPhase.IDLE
            return self._result(seed)
        if snapshot.stage == "holding" and snapshot.accepted and self.phase == PresetPhase.ENTERING:
            self.phase = PresetPhase.HOLDING
        return None

    def _result(self, seed: UpperBodyCommand | None = None) -> PresetStep:
        blocked = self.phase != PresetPhase.IDLE and (not self._released or self.phase == PresetPhase.DRAINING)
        return PresetStep(blocked or bool(self._fault), seed, self._fault, self.phase, self._released)
