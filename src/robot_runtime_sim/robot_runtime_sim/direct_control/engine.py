"""Simulation core: motion requests in, one target per platform tick out (no ROS, no simulator).

The platform's clock is the only clock that moves the robot. Every tick the core answers
with the joint and gripper targets of every motion group, or with *idle* (no targets), in
which case the platform must not step physics: thinking time costs no simulated time.

Lifecycle (one platform session = one generation)::

    CREATED -> RESETTING -> READY -> RUNNING -> STOPPING -> FINALIZED
                  any -> FAILED | UNKNOWN;  reset() starts the next generation from any state

* ``reset()`` cancels and drains every command of the previous generation and bumps the
  generation. Observations, IK replies and calls that carry an older session are refused.
* A command whose execution was already handed to the platform when its wall deadline
  passed, or whose platform connection was lost, ends ``UNKNOWN``: the core cannot tell how
  far it got, so it writes no tick at all (not even a hold) and accepts no motion until the
  next reset; only an explicit ``finish`` lets the platform's clock run out with hold ticks.
* Every request is bound to the generation current when it was received; one bound to an
  older generation is refused. The engine has no admission rule of its own: a caller may
  install a motion gate (``set_motion_gate``) that holds motion of a generation back, e.g. an
  evaluation layer that opens a session only once nothing of the previous one can still arrive.
* Simulated time only moves with the platform: ``sim_time_s`` is the time of the latest
  accepted observation, summed over every session of this process (it never goes back at a
  reset), and ``on_clock`` hears each new value. Idle, the time stands still.
* ``stop`` cancels every command and stops ticking until ``resume``; ``cancel`` withdraws one
  command. A tick already handed to the platform is never undone.
* A command id seen before in this generation returns the first command's result instead
  of executing again; the same id with different content is rejected.
* One writer only: ``claim_writer`` binds the tick loop to one driver (platform pull or host
  step); a second driver is refused. Two clients of the same driver kind are told apart by
  generation: the second one's reset makes every later call of the first one stale.

Thread-safe: ROS service threads call ``submit``; the platform driver calls the tick side.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robot_runtime_sim.direct_control.kinematics import (
    interpolate,
    nearest_equivalent,
    orientation_error,
    position_error,
    shift_pose,
)
from robot_runtime_sim.direct_control.ports import (
    COMMAND_GRIPPER,
    COMMAND_IK,
    COMMAND_JOINT,
    COMMAND_POSE,
    AdapterDescriptor,
    IkReply,
    IkRequest,
    ObservationFrame,
    PlatformResult,
    TickDecision,
)

OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_REJECTED = "REJECTED"
OUTCOME_PLANNING_FAILED = "PLANNING_FAILED"
OUTCOME_EXECUTION_FAILED = "EXECUTION_FAILED"
OUTCOME_CANCELLED = "CANCELLED"
OUTCOME_UNKNOWN = "UNKNOWN"
OUTCOME_NOT_REGISTERED = "NOT_REGISTERED"  # cancel result only: the command had not arrived yet

STATE_CREATED = "CREATED"
STATE_RESETTING = "RESETTING"
STATE_READY = "READY"
STATE_RUNNING = "RUNNING"
STATE_STOPPING = "STOPPING"
STATE_FINALIZED = "FINALIZED"
STATE_FAILED = "FAILED"
STATE_UNKNOWN = "UNKNOWN"

_ACCEPTS_MOTION = frozenset({STATE_READY, STATE_RUNNING})
_TICKS = frozenset({STATE_READY, STATE_RUNNING, STATE_STOPPING, STATE_UNKNOWN})


class SessionError(RuntimeError):
    """A tick-side message does not belong to the active session or generation."""


class WriterConflict(RuntimeError):
    """A second driver tried to write physical targets for the same core."""


@dataclass
class MotionConfig:
    max_joint_speed: float = 1.5  # rad/s at velocity_scaling 1.0
    min_motion_ticks: int = 8
    max_motion_ticks: int = 100
    settle_ticks: int = 3
    gripper_ticks: int = 12
    idle_wait_s: float = 0.2  # how long one idle tick request waits for a command before answering idle
    idle_timeout_s: float = 600.0  # no command for this long -> hold ticks until the platform ends the session
    gripper_default: float = 1.0
    position_tolerance_m: float = 0.02  # a pose move ending further away than this is EXECUTION_FAILED
    orientation_tolerance_rad: float = 0.35  # a pose move ending turned further than this is EXECUTION_FAILED
    gripper_tolerance: float = 0.02  # a gripper already this close to its target finishes at once

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> MotionConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(cls().__dict__)
        if unknown:
            raise ValueError(f"unknown motion config keys {sorted(unknown)}")
        return cls(**{k: type(v)(cfg[k]) for k, v in cls().__dict__.items() if k in cfg})


@dataclass
class Command:
    kind: str  # ports.COMMAND_*
    group: str
    pose: list[float] | None = None  # TCP pose [x, y, z, qw, qx, qy, qz]
    joints: list[float] | None = None
    opening: float | None = None
    velocity_scaling: float = 1.0
    command_id: str = ""
    seq: int = 0
    # Session generation the request belongs to; -1 = the generation current at submit. A
    # request bound to an older generation is refused, never moved into the new session.
    generation: int = -1
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)
    # set while executing
    track: deque = field(default_factory=deque)
    settle_left: int = 0
    started_tick: int | None = None
    handed_over: bool = False  # IK request or a tick reached the platform
    joint_target: list[float] | None = None
    result_extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return json.dumps(
            [self.kind, self.group, self.pose, self.joints, self.opening, self.velocity_scaling],
            sort_keys=True,
        )

    def finish(self, outcome: str, message: str, **extra: Any) -> None:
        if self.done.is_set():
            return
        self.result = {"outcome": outcome, "success": outcome == OUTCOME_SUCCEEDED, "message": message, **extra}
        self.done.set()


def motion_ticks(
    start: Sequence[float], end: Sequence[float], cfg: MotionConfig, velocity_scaling: float, dt_s: float
) -> int:
    scale = velocity_scaling if 0.0 < velocity_scaling <= 1.0 else 1.0
    dq = max((abs(a - b) for a, b in zip(start, end, strict=True)), default=0.0)
    ticks = math.ceil(dq / (cfg.max_joint_speed * scale * dt_s)) if dq > 0 else 0
    return max(cfg.min_motion_ticks, min(cfg.max_motion_ticks, ticks))


class SimulationCore:
    """Tick engine for one simulated robot; every public method is thread-safe."""

    def __init__(
        self,
        descriptor: AdapterDescriptor,
        cfg: MotionConfig | None = None,
        *,
        on_event: Callable[[str, dict], None] | None = None,
        on_platform_result: Callable[[dict], None] | None = None,
        on_clock: Callable[[float], None] | None = None,
        motion_gate: Callable[[int], str] | None = None,
    ):
        self.descriptor = descriptor
        self.cfg = cfg or MotionConfig()
        self._groups = {g.name: g for g in descriptor.groups}
        self._on_event = on_event or (lambda _kind, _info: None)
        # The platform's own result of a session (its verdict, if it has one). The engine only
        # hands it on; what it means and who may see it is the business of whoever listens.
        self._on_platform_result = on_platform_result or (lambda _result: None)
        self._on_clock = on_clock or (lambda _t: None)
        self._cv = threading.Condition()
        self._seq = itertools.count(1)
        self._generation = -1
        self._session_id: str | None = None
        self._state = STATE_CREATED
        self._writer: str | None = None
        self._session_info: dict[str, Any] = {}
        self._obs: ObservationFrame | None = None
        self._obs_tick = -1
        self._obs_sim_s = 0.0  # simulated time of the current observation (its source stamp)
        self._commanded: dict[str, dict[str, Any]] | None = None
        self._queue: deque[Command] = deque()
        self._active: Command | None = None
        self._ik_out: dict[int, Command] = {}
        self._by_id: dict[str, Command] = {}
        # Ids cancelled before the runtime saw them (the cancel overtook the request): a later
        # submit with such an id ends CANCELLED without a tick. Cleared with the generation.
        self._tombstones: dict[str, str] = {}
        self._finishing = False
        # Explicit stop: no new physical tick (not even a hold) until resume() or reset().
        self._halted = False
        # Optional gate: gate(generation) -> refusal text, or "" to let motion through.
        self._motion_gate = motion_gate
        # Simulated seconds of the sessions before the current one (sim_time_s never goes back).
        self._time_base_s = 0.0
        self._last_activity = time.monotonic()
        self._ticks_executed = 0
        self._ended_generation: int | None = None  # generation whose end was already published
        self._reset_wall = 0.0
        self.stale_messages = 0

    # ------------------------------------------------------------------ introspection

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def generation(self) -> int:
        return self._generation

    def clock(self) -> dict[str, Any]:
        with self._cv:
            tick = max(self._obs_tick, 0)
            return {
                "session_id": self._session_id,
                "generation": self._generation,
                "state": self._state,
                "tick": tick,
                "sim_time_s": round(tick * self.descriptor.dt_s, 6),
                "dt_s": self.descriptor.dt_s,
                "sim_time_total_s": round(self._sim_time_locked(), 6),
                # Wall time of this generation's reset: anything stamped earlier belongs to an older session.
                "generation_started_wall": self._reset_wall,
                "halted": self._halted,
            }

    def sim_time_s(self) -> float:
        """Simulated seconds of the latest accepted observation, over all sessions (monotonic)."""
        with self._cv:
            return self._sim_time_locked()

    def _sim_time_locked(self) -> float:
        return self._time_base_s + max(self._obs_tick, 0) * self.descriptor.dt_s

    def snapshot(self) -> tuple[ObservationFrame | None, int]:
        with self._cv:
            return self._obs, self._obs_tick

    def observation_source(self) -> tuple[str | None, int, int, float]:
        """(session_id, generation, tick, simulated seconds) of the current observation."""
        with self._cv:
            return self._session_id, self._generation, self._obs_tick, self._obs_sim_s

    def session_info(self) -> dict[str, Any]:
        with self._cv:
            return dict(self._session_info)

    def tcp_pose(self, frame: ObservationFrame, group: str) -> list[float] | None:
        ee = frame.ee_poses.get(group)
        if ee is None:
            return None
        return shift_pose([float(v) for v in list(ee)[:7]], self._groups[group].tcp_offset)

    # ------------------------------------------------------------------ driver side (one writer)

    def claim_writer(self, owner: str) -> None:
        with self._cv:
            if self._writer is not None and self._writer != owner:
                raise WriterConflict(f"tick loop already driven by {self._writer!r}; {owner!r} refused")
            self._writer = owner

    def reset(self) -> dict[str, Any]:
        """Start the next generation: drain the old one, refuse its stragglers."""
        with self._cv:
            self._cancel_all(OUTCOME_CANCELLED, "session reset")
            self._time_base_s = self._sim_time_locked()
            self._generation += 1
            self._session_id = f"s{self._generation:04d}"
            self._session_info = {}
            self._obs, self._obs_tick, self._commanded = None, -1, None
            self._by_id.clear()
            self._tombstones.clear()
            self._finishing = False
            self._halted = False
            self._reset_wall = time.time()
            self._last_activity = time.monotonic()
            self._ticks_executed = 0
            self._set_state(STATE_RESETTING)
            self._cv.notify_all()
            return {"session_id": self._session_id, "generation": self._generation}

    def begin_session(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """The public description the platform gives for this session (as it publishes it); no hidden results."""
        with self._cv:
            if self._state != STATE_RESETTING:
                raise SessionError(f"begin_session in state {self._state}")
            self._session_info = {
                **dict(info),
                "session_id": self._session_id,
                "generation": self._generation,
                "dt_s": self.descriptor.dt_s,
            }
            self._last_activity = time.monotonic()
            event = dict(self._session_info)
        self._on_event("session_start", event)
        return {"session_id": event["session_id"], "generation": event["generation"]}

    def set_motion_gate(self, gate: Callable[[int], str] | None) -> None:
        """Install (or clear) the motion gate: ``gate(generation)`` returns a refusal or ""."""
        with self._cv:
            self._motion_gate = gate
            self._cv.notify_all()

    def check_session(self, session_id: Any, generation: Any = None) -> None:
        with self._cv:
            self._check(session_id, generation)

    def on_observation(self, frame: ObservationFrame) -> None:
        with self._cv:
            self._check(frame.session_id, frame.generation)
            if frame.step_id < self._obs_tick:
                raise SessionError(f"observation tick {frame.step_id} < previous {self._obs_tick}")
            for group in self._groups:
                if group not in frame.joints:
                    raise SessionError(f"observation has no joints for group {group!r}")
            self._obs, self._obs_tick = frame, int(frame.step_id)
            self._obs_sim_s = self._sim_time_locked()
            if self._commanded is None:
                self._commanded = self._hold_from_obs(frame)
            if self._state == STATE_RESETTING:
                self._set_state(STATE_READY)
            self._cv.notify_all()
            now_s = self._sim_time_locked()
        self._on_clock(now_s)

    def next_tick(
        self, writer: str, session_id: Any, generation: Any = None, replies: Sequence[IkReply] = ()
    ) -> TickDecision:
        """Targets for the next tick, or idle. Waits up to ``idle_wait_s`` for a command."""
        deadline = time.monotonic() + self.cfg.idle_wait_s
        with self._cv:
            if self._writer != writer:
                raise WriterConflict(f"tick requested by {writer!r}, the writer is {self._writer!r}")
            self._check(session_id, generation)
            self._apply_replies(replies)
            while True:
                decision = self._next_tick()
                if decision is not None:
                    return decision
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return TickDecision(targets=None)
                self._cv.wait(remaining)
                # The lock was released while waiting: a reset may have started another session.
                self._check(session_id, generation)

    def session_end(self, public: Mapping[str, Any], native: PlatformResult) -> None:
        """The platform ended the session: publish the public end, hand its result to the listener."""
        with self._cv:
            self._check(public.get("session_id"), public.get("generation"))
            self._cancel_all(OUTCOME_EXECUTION_FAILED, "session ended before the motion finished")
            self._set_state(STATE_FINALIZED)
            self._cv.notify_all()
            self._publish_end(native, interrupted=False)

    def interrupt(self, public: Mapping[str, Any] | None, reason: str, extra: Mapping[str, Any] | None = None) -> int:
        """The platform stopped the session without a result (its runtime call failed).

        Commands already handed over end UNKNOWN, the rest CANCELLED; no tick is written any
        more; the result listener hears an interrupted end. Returns the UNKNOWN count.
        """
        with self._cv:
            if (
                public is not None
                and public.get("session_id") == self._session_id
                and self._ended_generation == self._generation
            ):
                return 0  # already ended (e.g. aborted by the caller): nothing more to stop
            if public is not None:
                self._check(public.get("session_id"), public.get("generation"))
            elif self._session_id is None or self._state in (STATE_CREATED, STATE_FINALIZED, STATE_FAILED):
                return 0
            return self._interrupt_locked(reason, extra or {})

    def platform_lost(self, reason: str) -> int:
        """The platform connection dropped: anything handed over has an unknown outcome."""
        with self._cv:
            if self._state in (STATE_FINALIZED, STATE_CREATED, STATE_FAILED) or self._session_id is None:
                return 0
            return self._interrupt_locked(f"platform connection lost ({reason})", {})

    def _interrupt_locked(self, reason: str, extra: Mapping[str, Any]) -> int:
        in_flight = [c for c in self._all_commands() if c.handed_over and not c.done.is_set()]
        for cmd in in_flight:
            cmd.finish(OUTCOME_UNKNOWN, f"{reason}; outcome unknown")
        self._cancel_all(OUTCOME_CANCELLED, reason)
        self._halted = True
        self._finishing = False
        self._set_state(STATE_UNKNOWN if in_flight else STATE_FAILED)
        self._cv.notify_all()
        self._publish_end(
            PlatformResult(native_success=None, terminated=False, native_reason="interrupted", extra=extra),
            interrupted=True,
            reason=reason,
        )
        return len(in_flight)

    def _publish_end(self, native: PlatformResult, *, interrupted: bool, reason: str = "") -> None:
        """Publish the end of the current generation once (lock held; callbacks must not re-enter)."""
        if self._ended_generation == self._generation:
            return
        self._ended_generation = self._generation
        ticks = max(self._obs_tick, 0)
        end = {
            "session_id": self._session_id,
            "generation": self._generation,
            "ticks": ticks,
            "sim_time_s": round(ticks * self.descriptor.dt_s, 6),
            "interrupted": interrupted,
        }
        if native.take_action_cnt is not None:
            # The platform's own count; equals ``ticks`` when the final observation arrived.
            end["platform_ticks"] = int(native.take_action_cnt)
        if reason:
            end["reason"] = reason
        self._on_event("session_end", end)
        self._on_platform_result(
            {
                "session_id": self._session_id,
                "generation": self._generation,
                "native_success": native.native_success,
                "native_score": native.native_score,
                "terminated": native.terminated,
                "truncated": native.truncated,
                "failure_reason": native.native_reason,
                "take_action_cnt": native.take_action_cnt,
                "step_lim": native.step_lim,
                "protocol_version": native.protocol_version,
                "interrupted": interrupted,
                **dict(native.extra),
            }
        )

    # ------------------------------------------------------------------ ROS side

    def submit(self, cmd: Command, timeout_s: float) -> dict[str, Any]:
        """Queue a command and block until the tick loop finishes it or ``timeout_s`` (wall) passes."""
        refusal = self._static_refusal(cmd)
        if refusal:
            cmd.finish(OUTCOME_REJECTED, refusal)
            return cmd.result
        with self._cv:
            if cmd.command_id and cmd.command_id in self._tombstones:
                cmd.finish(
                    OUTCOME_CANCELLED,
                    f"command {cmd.command_id!r} was cancelled before it arrived "
                    f"({self._tombstones[cmd.command_id]}); not executed",
                    ticks_executed=0,
                    ticks=0,
                )
                return dict(cmd.result)
            if cmd.command_id and cmd.command_id in self._by_id:
                first = self._by_id[cmd.command_id]
                if first.fingerprint() != cmd.fingerprint():
                    cmd.finish(OUTCOME_REJECTED, f"command_id {cmd.command_id!r} reused with different content")
                    return cmd.result
                cmd = first  # same request again: wait for (or return) the first execution
            else:
                refusal = self._state_refusal()
                if not refusal and cmd.generation >= 0 and cmd.generation != self._generation:
                    self.stale_messages += 1
                    refusal = (
                        f"request of session generation {cmd.generation} arrived in generation {self._generation}; "
                        "not executed"
                    )
                if refusal:
                    cmd.finish(OUTCOME_REJECTED, refusal)
                    return cmd.result
                cmd.seq = next(self._seq)
                cmd.generation = self._generation
                if not cmd.command_id:
                    cmd.command_id = f"{self._session_id}-c{cmd.seq}"
                self._by_id[cmd.command_id] = cmd
                self._queue.append(cmd)
                self._last_activity = time.monotonic()
                self._cv.notify_all()
        if not cmd.done.wait(timeout_s):
            with self._cv:
                if cmd in self._queue:
                    self._queue.remove(cmd)
                    cmd.finish(OUTCOME_EXECUTION_FAILED, f"timed out after {timeout_s:.1f} s wall time before it ran")
                elif not cmd.done.is_set():
                    # Already handed to the platform: how far it got is unknown. Stop writing.
                    cmd.finish(
                        OUTCOME_UNKNOWN,
                        f"no result within {timeout_s:.1f} s wall time after execution started; outcome unknown",
                    )
                    self._cancel_all(OUTCOME_CANCELLED, "an earlier command has an unknown outcome")
                    # Stop writing: no tick at all until the caller decides (finish / reset).
                    self._halted = True
                    self._finishing = False
                    self._set_state(STATE_UNKNOWN)
                    self._cv.notify_all()
        return dict(cmd.result)

    def stop(self, reason: str = "stop requested") -> int:
        """Cancel every command and stop ticking: no new target (not even a hold) until resume().

        A tick the platform already took cannot be withdrawn; it is the last one written.
        """
        with self._cv:
            n = self._cancel_all(OUTCOME_CANCELLED, reason)
            if self._obs is not None:
                self._commanded = self._hold_from_obs(self._obs)
            self._halted = True
            self._finishing = False
            if self._state == STATE_STOPPING:
                self._set_state(STATE_RUNNING)
            self._cv.notify_all()
            return n

    def resume(self) -> None:
        """Lift an explicit stop; ticks follow commands again (an UNKNOWN session stays blocked)."""
        with self._cv:
            if self._state != STATE_UNKNOWN:
                self._halted = False
            self._cv.notify_all()

    @property
    def halted(self) -> bool:
        return self._halted

    def cancel_by_id(self, command_id: str, reason: str = "cancelled by the caller") -> tuple[bool, dict[str, Any]]:
        """Cancel the command with this id in the current generation; (cancelled, its result).

        An id the runtime has not seen yet is recorded as cancelled, so the request is refused
        when it arrives: (True, {"outcome": "NOT_REGISTERED", ...}). An id that already ended
        gives (False, its final result).
        """
        with self._cv:
            if not command_id:
                return False, {"outcome": "", "message": "empty command_id"}
            cmd = self._by_id.get(command_id)
            if cmd is None:
                self._tombstones[command_id] = reason
                return True, {
                    "outcome": OUTCOME_NOT_REGISTERED,
                    "message": f"command {command_id!r} not registered yet; it will be refused if it arrives",
                    "ticks_executed": 0,
                }
            cancelled = self.cancel(cmd, reason)
            return cancelled, dict(cmd.result)

    def cancel(self, cmd: Command, reason: str = "cancelled by the caller") -> bool:
        """Cancel one command; its group holds where it is. False if it already finished."""
        with self._cv:
            if cmd.done.is_set():
                return False
            if cmd in self._queue:
                self._queue.remove(cmd)
            self._ik_out.pop(cmd.seq, None)
            if cmd is self._active:
                self._active = None
                if self._obs is not None and self._commanded is not None:
                    self._commanded[cmd.group] = self._hold_from_obs(self._obs)[cmd.group]
            extra = {}
            if cmd.started_tick is not None:
                extra["ticks_executed"] = extra["ticks"] = self._obs_tick - cmd.started_tick
            cmd.finish(OUTCOME_CANCELLED, reason, **extra)
            self._cv.notify_all()
            return True

    def finish(self) -> bool:
        """The caller is done with this session: hold until the platform ends it.

        Also the explicit way out of UNKNOWN: the caller chooses to run out the platform's
        clock with hold ticks instead of leaving the session frozen.
        """
        with self._cv:
            if self._state not in _ACCEPTS_MOTION and self._state != STATE_UNKNOWN:
                return False
            self._finishing = True
            self._halted = False
            if self._state != STATE_UNKNOWN:
                self._set_state(STATE_STOPPING)
            self._cv.notify_all()
            return True

    # ------------------------------------------------------------------ internals (lock held)

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._on_event("state", {"session_id": self._session_id, "generation": self._generation, "state": state})

    def _check(self, session_id: Any, generation: Any) -> None:
        if self._session_id is None or session_id != self._session_id:
            self.stale_messages += 1
            raise SessionError(f"session {session_id!r} is not the active session {self._session_id!r}")
        if generation is not None and int(generation) != self._generation:
            self.stale_messages += 1
            raise SessionError(f"generation {generation!r} is not the active generation {self._generation}")
        if self._state in (STATE_FINALIZED, STATE_FAILED):
            raise SessionError(f"session {session_id!r} is {self._state}")

    def _static_refusal(self, cmd: Command) -> str:
        if cmd.group not in self._groups:
            return f"unknown motion group {cmd.group!r}; this robot has {sorted(self._groups)}"
        if not self.descriptor.supports(cmd.kind):
            return f"this simulation platform does not support {cmd.kind!r} commands"
        group = self._groups[cmd.group]
        if cmd.kind == COMMAND_JOINT and (cmd.joints is None or len(cmd.joints) != group.dof):
            return f"group {cmd.group!r} takes {group.dof} joint values"
        if cmd.kind == COMMAND_GRIPPER and group.gripper is None:
            return f"group {cmd.group!r} has no gripper"
        return ""

    def _state_refusal(self) -> str:
        if self._state in _ACCEPTS_MOTION and self._obs is not None and not self._finishing and not self._halted:
            gate = self._motion_gate
            return str(gate(self._generation) or "") if gate is not None else ""
        if self._state == STATE_UNKNOWN:
            return "an earlier command has an unknown outcome; no motion until the session is reset"
        if self._state == STATE_STOPPING or self._finishing:
            return "session is finishing; no new motion accepted"
        if self._state == STATE_RESETTING:
            return "session is resetting; wait for the first observation"
        if self._halted:
            return "runtime is stopped; no motion until it is resumed"
        return "no active simulation session (waiting for the platform)"

    def _hold_from_obs(self, frame: ObservationFrame) -> dict[str, dict[str, Any]]:
        held: dict[str, dict[str, Any]] = {}
        for name, group in self._groups.items():
            joints = [float(v) for v in list(frame.joints[name])[: group.dof]]
            entry: dict[str, Any] = {"joints": joints}
            if group.gripper is not None:
                entry["gripper"] = float(frame.grippers.get(name, self.cfg.gripper_default))
            held[name] = entry
        return held

    def _all_commands(self) -> list[Command]:
        cmds = [*self._queue, *self._ik_out.values()]
        if self._active is not None and self._active not in cmds:
            cmds.append(self._active)
        return cmds

    def _cancel_all(self, outcome: str, message: str) -> int:
        cancelled = [c for c in self._all_commands() if not c.done.is_set()]
        for cmd in cancelled:
            extra = {}
            if cmd.started_tick is not None:
                # Ticks this command already spent; they stay spent (no physics is undone).
                extra["ticks_executed"] = extra["ticks"] = self._obs_tick - cmd.started_tick
                extra.update(cmd.result_extra)
            cmd.finish(outcome, message, **extra)
        self._queue.clear()
        self._ik_out.clear()
        self._active = None
        return len(cancelled)

    def _apply_replies(self, replies: Sequence[IkReply]) -> None:
        for reply in replies:
            cmd = self._ik_out.pop(int(reply.request_id), None)
            if cmd is None or cmd.done.is_set():
                self.stale_messages += 1
                continue
            if not reply.ok or not reply.joints:
                cmd.finish(OUTCOME_PLANNING_FAILED, f"IK found no solution: {reply.message}".strip())
                if cmd is self._active:
                    self._active = None
                continue
            joints = [float(v) for v in reply.joints][: self._groups[cmd.group].dof]
            if cmd.kind == COMMAND_IK:
                cmd.finish(OUTCOME_SUCCEEDED, "IK solved", joints=joints)
                continue
            self._start_track(cmd, joints)

    def _ik_request(self, cmd: Command) -> IkRequest:
        flange = shift_pose(cmd.pose, self._groups[cmd.group].tcp_offset, -1.0)
        return IkRequest(request_id=cmd.seq, group=cmd.group, pose=tuple(flange))

    def _start_track(self, cmd: Command, joints: list[float]) -> None:
        group = self._groups[cmd.group]
        start = self._commanded[cmd.group]["joints"]
        limits = group.joint_wrap_limits or (math.pi,) * group.dof
        joints = nearest_equivalent(start, joints, limits)
        n = motion_ticks(start, joints, self.cfg, cmd.velocity_scaling, self.descriptor.dt_s)
        cmd.result_extra = {"max_joint_delta_rad": max(abs(a - b) for a, b in zip(start, joints, strict=True))}
        cmd.joint_target = joints
        cmd.track = deque(interpolate(start, joints, n))
        cmd.settle_left = self.cfg.settle_ticks
        self._active = cmd

    def _next_tick(self) -> TickDecision | None:
        if self._commanded is None or self._state not in _TICKS or self._halted:
            return None
        if self._active is None and self._queue:
            cmd = self._queue.popleft()
            if cmd.kind in (COMMAND_POSE, COMMAND_IK):
                self._ik_out[cmd.seq] = cmd
                cmd.handed_over = True
                if cmd.kind == COMMAND_POSE:
                    self._active = cmd
                return TickDecision(
                    targets=None,
                    ik_requests=(self._ik_request(cmd),),
                    command_id=cmd.command_id,
                    command_seq=cmd.seq,
                )
            if cmd.kind == COMMAND_JOINT:
                self._start_track(cmd, cmd.joints)
            elif cmd.kind == COMMAND_GRIPPER:
                start = self._commanded[cmd.group]["gripper"]
                if abs(float(start) - float(cmd.opening)) <= self.cfg.gripper_tolerance:
                    # Already there (e.g. opening an open gripper): no ticks spent.
                    cmd.started_tick = self._obs_tick
                    self._complete(cmd)
                    return self._next_tick()
                cmd.track = deque([v[0] for v in interpolate([start], [float(cmd.opening)], self.cfg.gripper_ticks)])
                cmd.settle_left = self.cfg.settle_ticks
                self._active = cmd
        cmd = self._active
        if cmd is not None and cmd.seq in self._ik_out:
            return None  # IK request out; wait for its reply
        if cmd is not None:
            self._last_activity = time.monotonic()
            if cmd.started_tick is None:
                cmd.started_tick = self._obs_tick
            if cmd.track:
                key = "gripper" if cmd.kind == COMMAND_GRIPPER else "joints"
                self._commanded[cmd.group][key] = cmd.track.popleft()
                return self._emit(cmd)
            if cmd.settle_left > 0:
                cmd.settle_left -= 1
                return self._emit(cmd)
            self._complete(cmd)
            self._active = None
            return self._next_tick()
        if self._finishing:
            return self._emit(None)
        if time.monotonic() - self._last_activity > self.cfg.idle_timeout_s:
            self._finishing = True
            self._set_state(STATE_STOPPING)
            self._on_event(
                "idle_timeout",
                {"session_id": self._session_id, "generation": self._generation, "idle_s": self.cfg.idle_timeout_s},
            )
            return self._emit(None)
        return None

    def _emit(self, cmd: Command | None) -> TickDecision:
        self._ticks_executed += 1
        if cmd is not None:
            cmd.handed_over = True
            if self._state == STATE_READY:
                self._set_state(STATE_RUNNING)
        targets = {
            name: {k: (list(v) if isinstance(v, list) else v) for k, v in entry.items()}
            for name, entry in self._commanded.items()
        }
        return TickDecision(
            targets=targets,
            command_id=cmd.command_id if cmd else None,
            command_seq=cmd.seq if cmd else None,
            kind=cmd.kind if cmd else "hold",
        )

    def _complete(self, cmd: Command) -> None:
        frame = self._obs
        started = self._obs_tick if cmd.started_tick is None else cmd.started_tick
        ticks = self._obs_tick - started  # the observation after the last executed tick
        dt = self.descriptor.dt_s
        extra: dict[str, Any] = {"ticks": ticks, "sim_time_s": round(ticks * dt, 3)}
        if cmd.kind == COMMAND_POSE and cmd.pose is not None:
            ee = self.tcp_pose(frame, cmd.group) if frame is not None else None
            if ee is None:
                cmd.finish(
                    OUTCOME_EXECUTION_FAILED,
                    "no end-effector pose in the observation; cannot confirm the target was reached",
                    **extra,
                )
                return
            extra["position_error_m"] = position_error(cmd.pose, ee)
            extra["orientation_error_rad"] = orientation_error(cmd.pose, ee)
            extra["ee_pose"] = ee
        if cmd.kind == COMMAND_GRIPPER and frame is not None and cmd.group in frame.grippers:
            extra["gripper_measured"] = float(frame.grippers[cmd.group])
        if cmd.joint_target is not None:
            extra["joints"] = cmd.joint_target
        extra.update(cmd.result_extra)
        error = extra.get("position_error_m")
        if error is not None and error > self.cfg.position_tolerance_m:
            cmd.finish(
                OUTCOME_EXECUTION_FAILED,
                f"stopped {error:.3f} m from the target (tolerance {self.cfg.position_tolerance_m:.3f} m), "
                "e.g. blocked by contact",
                **extra,
            )
            return
        turned = extra.get("orientation_error_rad")
        if turned is not None and turned > self.cfg.orientation_tolerance_rad:
            cmd.finish(
                OUTCOME_EXECUTION_FAILED,
                f"stopped {turned:.3f} rad from the target orientation "
                f"(tolerance {self.cfg.orientation_tolerance_rad:.3f} rad), e.g. a blocked rotation",
                **extra,
            )
            return
        done = "gripper set" if cmd.kind == COMMAND_GRIPPER else "reached"
        cmd.finish(OUTCOME_SUCCEEDED, done, **extra)
