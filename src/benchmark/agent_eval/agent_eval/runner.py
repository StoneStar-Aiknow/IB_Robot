"""Episode runner: one runtime session = one evaluation episode, one Agent session per decision.

This is the scheduling half of the MR !551 episode driver, made generic. The runner never
moves the robot and never computes success:

* the runtime's public session lifecycle (``/simulation/session``, ``/simulation/clock``) and
  robot state (``/robot_status/<group>/ee_pose``, ``/joint_states``, the planner's cameras)
  feed :class:`EventBook`;
* the request text comes from the episode context (:mod:`agent_eval.episode_context`), which
  only ever sees a whitelisted observation and carries nothing across decisions;
* the verdict channel of the evaluation extension (``/evaluation/native_result``) is read into
  the episode record only, never into the context. An Agent DONE is an end request, not a
  success.

Which observations belong to a session: the runtime stamps state and images with simulated
time, which does not go back at a reset but may stand still across it (a new session's first
observation can carry the previous one's last stamp). So a source counts for a session when
it arrived after the session's first observed ``/simulation/clock`` message (state no longer
RESETTING: the runtime publishes nothing between a reset and that observation, and publishes
that clock before the observation's state and images) and its stamp is not older than the
session's time base (``sim_time_total_s - sim_time_s`` of that clock).

The ROS wiring is in :mod:`agent_eval.runner_node`; everything here runs without ROS.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .body import BodyDescription
from .episode_context import TERMINAL_STATES, EpisodeContext, EpisodeInfo, EpisodeObservation

RECORD_SCHEMA = "agent_eval.episode/1"
EpisodeKey = tuple[str, Any]  # (runtime session_id, generation)


def episode_key(event: Mapping[str, Any]) -> EpisodeKey:
    generation = event.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        generation = None
    return (str(event.get("session_id")), generation)


def _observed_state(clock: Mapping[str, Any]) -> bool:
    """Whether the clock reports a session that has its first observation (not still resetting)."""
    return str(clock.get("state", "")) not in ("CREATED", "RESETTING")


def _time_base(clock: Mapping[str, Any]) -> float | None:
    """Simulated time at the session's start: the stamp of its first observation."""
    total, within = clock.get("sim_time_total_s"), clock.get("sim_time_s")
    if isinstance(total, int | float) and isinstance(within, int | float) and not isinstance(total, bool):
        return float(total) - float(within)
    return None


class PlatformExited(RuntimeError):
    """A process the episode depends on (platform client, robot stack) exited; stop waiting."""


class EventBook:
    """Thread-safe store of what the runner has heard, with the episode admission rules.

    * ``session_start`` events from before the runner started (the topic is latched) are
      ignored, as are events whose generation is older than the newest one taken and
      repeats of a session already taken.
    * ``session_end`` and ``native_result`` are kept per (session_id, generation) and only
      for sessions taken as episodes, so a late result of an earlier session or generation is
      never attributed to the current one.
    """

    LATCH_GRACE_S = 5.0
    # After a process exits, how long a pending episode end / native result may still arrive
    # (the platform client reports the verdict just before it exits).
    ABORT_GRACE_S = 10.0

    def __init__(
        self,
        body: BodyDescription,
        wall_clock: Callable[[], float] = time.time,
        camera_topics: Sequence[str] = (),
    ):
        self._body = body
        self._wall = wall_clock
        self.started_wall = wall_clock()
        self._lock = threading.Lock()
        self.wake = threading.Condition(self._lock)
        self._queue: list[dict[str, Any]] = []
        self._admitted: dict[EpisodeKey, float] = {}
        self._max_generation: int | None = None
        self._ends: dict[EpisodeKey, dict[str, Any]] = {}
        self._natives: dict[EpisodeKey, dict[str, Any]] = {}
        self.dropped: list[dict[str, Any]] = []
        self._clock: dict[str, Any] | None = None
        # Arrival order of every input, and per session the order and time base of its first
        # observed clock: what arrived later, stamped no earlier, is that session's (module doc).
        self._seq = 0
        self._observed: dict[EpisodeKey, tuple[int, float | None]] = {}
        # side -> (xyz, quat, source stamp or None, arrival)
        self._ee: dict[str, tuple[tuple[float, ...], tuple[float, ...] | None, float | None, int]] = {}
        self._joints: dict[str, float] = {}
        self._joints_source: tuple[float | None, int] = (None, 0)
        # Camera topic -> (source stamp, arrival) of its latest frame. The Agent attaches these
        # frames, so an episode is not fresh until every one of them carries a frame of it.
        self._camera_topics = tuple(camera_topics)
        self._cameras: dict[str, tuple[float | None, int]] = {}
        self._abort_check: Callable[[], str | None] = lambda: None
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._responses: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ inputs

    def _drop(self, reason: str, event: Mapping[str, Any]) -> None:
        self.dropped.append({"reason": reason, "event": dict(event)})

    def _arrival(self) -> int:
        self._seq += 1
        return self._seq

    def on_session(self, event: Mapping[str, Any]) -> None:
        with self.wake:
            kind = event.get("event")
            key = episode_key(event)
            if kind == "session_start":
                generation = key[1]
                if float(event.get("wall_time") or 0.0) < self.started_wall - self.LATCH_GRACE_S:
                    self._drop("started before the runner (latched)", event)
                elif generation is not None and self._max_generation is not None and generation < self._max_generation:
                    self._drop("stale generation", event)
                elif key in self._admitted:
                    self._drop("repeated session_start", event)
                else:
                    self._admitted[key] = self._wall()
                    self._queue.append(dict(event))
                    if generation is not None:
                        self._max_generation = max(generation, self._max_generation or generation)
            elif kind == "session_end":
                if key in self._admitted:
                    self._ends.setdefault(key, dict(event))
                else:
                    self._drop("session_end of a session not run here", event)
            self.wake.notify_all()

    def on_native_result(self, result: Mapping[str, Any]) -> None:
        with self.wake:
            key = episode_key(result)
            if key in self._admitted:
                self._natives.setdefault(key, dict(result))
            else:
                self._drop("native_result of a session not run here", result)
            self.wake.notify_all()

    def on_clock(self, clock: Mapping[str, Any]) -> None:
        with self.wake:
            seq = self._arrival()
            self._clock = dict(clock)
            key = episode_key(clock)
            if key not in self._observed and clock.get("session_id") is not None and _observed_state(clock):
                self._observed[key] = (seq, _time_base(clock))
            self.wake.notify_all()

    def on_ee(self, side: str, xyz: Sequence[float], quat: Sequence[float] | None, stamp: float | None = None) -> None:
        """``stamp``: the source (simulated) time of the observation, from the message header."""
        with self.wake:
            self._ee[side] = (tuple(xyz), None if quat is None else tuple(quat), stamp, self._arrival())
            self.wake.notify_all()

    def on_joints(self, names: Sequence[str], positions: Sequence[float], stamp: float | None = None) -> None:
        with self.wake:
            self._joints.update(dict(zip(names, positions, strict=False)))
            self._joints_source = (stamp, self._arrival())
            self.wake.notify_all()

    def on_camera(self, topic: str, stamp: float | None) -> None:
        with self.wake:
            self._cameras[topic] = (stamp, self._arrival())
            self.wake.notify_all()

    def on_agent_event(self, event: Mapping[str, Any]) -> None:
        rid = str((event.get("request_key") or {}).get("request_id", ""))
        with self.wake:
            self._events.setdefault(rid, []).append(dict(event))
            self.wake.notify_all()

    def on_agent_response(self, response: Mapping[str, Any]) -> None:
        with self.wake:
            self._responses[str(response.get("request_id", ""))] = dict(response)
            self.wake.notify_all()

    # ------------------------------------------------------------------ queries

    def set_abort_check(self, check: Callable[[], str | None] | None) -> None:
        """``check()`` returns why waiting is pointless (e.g. the platform client exited), or None."""
        self._abort_check = check or (lambda: None)

    def _raise_if_aborted(self) -> None:
        reason = self._abort_check()
        if reason:
            raise PlatformExited(reason)

    def next_episode(self, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        with self.wake:
            while not self._queue and time.monotonic() < deadline:
                self._raise_if_aborted()
                self.wake.wait(min(1.0, max(0.0, deadline - time.monotonic())))
            return self._queue.pop(0) if self._queue else None

    def episode_ended(self, key: EpisodeKey) -> bool:
        with self._lock:
            return key in self._ends

    def _tick_locked(self, key: EpisodeKey) -> int | None:
        clock = self._clock
        if clock is None or key not in self._observed or episode_key(clock) != key:
            return None
        tick = clock.get("tick")
        return int(tick) if isinstance(tick, int | float) and not isinstance(tick, bool) else None

    def _stale_locked(self, key: EpisodeKey) -> list[str]:
        """What is not yet from this session (see the module doc)."""
        marker = self._observed.get(key)

        def fresh(source: tuple[float | None, int] | None) -> bool:
            if marker is None or source is None:
                return False
            stamp, arrival = source
            base = marker[1]
            return arrival > marker[0] and (base is None or stamp is None or stamp >= base - 1e-6)

        stale = [
            f"ee_pose:{side}"
            for side in self._body.sides
            if not fresh(None if side not in self._ee else self._ee[side][2:])
        ]
        if not fresh(self._joints_source if self._joints_source[1] else None):
            stale.append("joint_states")
        stale += [f"camera:{topic}" for topic in self._camera_topics if not fresh(self._cameras.get(topic))]
        if self._tick_locked(key) is None:
            stale.append("clock")
        return stale

    def _fresh_locked(self, key: EpisodeKey) -> bool:
        return not self._stale_locked(key)

    def stale_sources(self, key: EpisodeKey) -> list[str]:
        with self._lock:
            return self._stale_locked(key)

    def wait_fresh(self, key: EpisodeKey, timeout_s: float) -> bool:
        """Wait for robot state, the planner's cameras and a clock tick of this session."""
        deadline = time.monotonic() + timeout_s
        with self.wake:
            while not self._fresh_locked(key):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._raise_if_aborted()
                self.wake.wait(min(0.2, remaining))
            return True

    def observation(
        self, key: EpisodeKey, instruction: str, step_lim: Any, fields: Sequence[str] | None = None
    ) -> EpisodeObservation:
        """The whitelisted observation; ``fields`` (a TaskSpec grant) blanks everything else."""
        granted = set(fields) if fields is not None else None
        with self._lock:
            arms = {}
            for arm in self._body.arms:
                ee = self._ee.get(arm.side)
                values = {
                    "tcp_xyz": None if ee is None else ee[0],
                    "tcp_quat": None if ee is None else ee[1],
                    "gripper_opening": self._joints.get(arm.gripper_joint),
                    "wrist_joint_rad": self._joints.get(arm.wrist_joint),
                }
                arms[arm.side] = {k: v for k, v in values.items() if granted is None or k in granted}
            data = {"instruction": instruction, "tick": self._tick_locked(key), "step_lim": step_lim, "arms": arms}
        if granted is not None:
            data = {k: v for k, v in data.items() if k == "arms" or k in granted}
        return EpisodeObservation.from_mapping(data)

    def events(self, request_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(request_id, []))

    def wait_terminal(self, request_id: str, timeout_s: float) -> dict[str, Any] | None:
        """The request's refusal or terminal event (with all its events); ``None`` on timeout."""
        deadline = time.monotonic() + timeout_s
        with self.wake:
            while time.monotonic() < deadline:
                response = self._responses.get(request_id)
                if response is not None and response.get("accepted") is False:
                    return {"state": response.get("state"), "event_type": "refused", "response": response, "events": []}
                events = self._events.get(request_id, [])
                terminal = next((e for e in events if e.get("state") in TERMINAL_STATES), None)
                if terminal is not None:
                    return {**terminal, "events": list(events)}
                self._raise_if_aborted()
                self.wake.wait(min(0.5, max(0.0, deadline - time.monotonic())))
        return None

    def wait_end_and_native(
        self, key: EpisodeKey, timeout_s: float
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        deadline = time.monotonic() + timeout_s
        aborted = False
        with self.wake:
            while (key not in self._ends or key not in self._natives) and time.monotonic() < deadline:
                if not aborted and self._abort_check():
                    aborted = True
                    deadline = min(deadline, time.monotonic() + self.ABORT_GRACE_S)
                self.wake.wait(min(1.0, max(0.0, deadline - time.monotonic())))
            return self._ends.get(key), self._natives.get(key)


class RunnerIO(Protocol):
    """The runner's outputs (ROS publishers / service clients in production)."""

    def publish_request(self, payload: dict[str, Any]) -> None: ...

    def publish_control(self, payload: dict[str, Any]) -> None: ...

    def call_finish(self) -> str: ...

    def admit_episode(self, session_id: str, generation: int | None) -> tuple[bool, str]: ...

    def abort_session(self) -> str: ...

    def agent_ready(self, timeout_s: float) -> bool: ...


@dataclass
class RunnerSettings:
    out_dir: str
    max_decisions: int = 40
    decision_timeout_s: float = 300.0
    observation_fields: Sequence[str] | None = None
    episode_tag: bool = True
    run_id: str | None = None
    fresh_wait_s: float = 60.0
    first_frame_settle_s: float = 0.5  # let the first camera frames reach the Agent
    end_wait_s: float = 600.0
    drain_wait_s: float = 30.0  # wait for Agent requests of earlier episodes to end before admitting
    # Images the Agent attaches; each must carry a frame of this episode before admission.
    camera_topics: Sequence[str] = ()
    extra_record: dict[str, Any] = field(default_factory=dict)


class EpisodeRunner:
    def __init__(
        self,
        book: EventBook,
        io: RunnerIO,
        context: EpisodeContext,
        settings: RunnerSettings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        new_hex: Callable[[], str] = lambda: uuid.uuid4().hex,
        log: Callable[[str], None] = print,
    ):
        self.book = book
        self.io = io
        self.context = context
        self.settings = settings
        self._sleep = sleep
        self._new_hex = new_hex
        self._log = log
        self._open_requests: set[str] = set()

    def _drain(self) -> list[str]:
        """Agent requests of earlier decisions that never ended; empty once all are terminal."""
        still = []
        for request_id in sorted(self._open_requests):
            if self.book.wait_terminal(request_id, self.settings.drain_wait_s) is None:
                still.append(request_id)
            else:
                self._open_requests.discard(request_id)
        return still

    def _ask(self, text: str, session_id: str, request_id: str) -> dict[str, Any]:
        payload = {"schema_version": 1, "request_id": request_id, "session_id": session_id, "text": text}
        self._open_requests.add(request_id)
        self.io.publish_request(payload)
        try:
            result = self.book.wait_terminal(request_id, self.settings.decision_timeout_s)
        except PlatformExited:
            # The episode is gone: do not leave the Agent executing for it.
            self.io.publish_control({"operation": "stop", "request_id": request_id})
            raise
        if result is not None:
            if result.get("event_type") == "refused" or result.get("state") in TERMINAL_STATES:
                self._open_requests.discard(request_id)
            return result
        self.io.publish_control({"operation": "stop", "request_id": request_id})
        return {"state": "TIMEOUT", "event_type": "timeout", "events": self.book.events(request_id)}

    def run_episode(self, start: Mapping[str, Any]) -> dict[str, Any]:
        settings, ctx = self.settings, self.context
        key = episode_key(start)
        runtime_session, generation = key
        run_id = str(start.get("run_id") or settings.run_id or "run")
        session_id = f"{run_id}_{runtime_session}_{self._new_hex()[:6]}"
        instruction = str(start.get("instruction") or start.get("task"))
        step_lim = start.get("step_lim")
        tag = self._new_hex()[:8]
        ctx.reset(
            EpisodeInfo(episode_id=runtime_session, generation=generation, tag=tag if settings.episode_tag else None)
        )
        record: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "episode": dict(start),
            "session_id": runtime_session,
            "generation": generation,
            "agent_session_id": session_id,
            "episode_tag": tag if settings.episode_tag else None,
            "decisions": [],
            "agent_request_ids": [],
            **settings.extra_record,
        }
        fields = settings.observation_fields

        def observe() -> EpisodeObservation:
            return self.book.observation(key, instruction, step_lim, fields)

        finish_reason = "max_decisions"
        t_episode = time.monotonic()
        try:
            record["fresh_state"] = self.book.wait_fresh(key, settings.fresh_wait_s)
            self._sleep(settings.first_frame_settle_s)
            idle_streak = 0
            t_episode = time.monotonic()
            # Open the episode for motion only when its own observations arrived (fail closed)
            # and nothing of an earlier episode can still arrive.
            missing = [] if record["fresh_state"] else self.book.stale_sources(key)
            record["stale_sources"] = missing
            stale = self._drain()
            if missing:
                admitted, message = False, f"observations of this episode missing: {missing}"
            elif stale:
                admitted, message = False, f"earlier Agent requests still open: {stale}"
            else:
                admitted, message = self.io.admit_episode(runtime_session, generation)
            record["admission"] = {"admitted": admitted, "message": message}
            decisions = settings.max_decisions if admitted else 0
            if not admitted:
                finish_reason = "stale_observation" if missing else "not_admitted"
            for n in range(decisions):
                if self.book.episode_ended(key):
                    finish_reason = "episode_ended"
                    break
                observation = observe()
                state = ctx.state_view(observation)
                text = ctx.build_request_text(observation)
                request_id = f"{session_id}_d{n:03d}"
                record["agent_request_ids"].append(request_id)
                t0 = time.monotonic()
                # A fresh Agent session per decision (the request id): the planner's conversation
                # store then holds nothing from earlier decisions.
                result = self._ask(text, request_id, request_id)
                decision: dict[str, Any] = {
                    "n": n,
                    "request_id": request_id,
                    "agent_session_id": request_id,
                    "request_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "state_before": state,
                    "latency_s": round(time.monotonic() - t0, 2),
                    "terminal": {k: result.get(k) for k in ("event_type", "state", "user_message", "detail")},
                    "events": [
                        {k: e.get(k) for k in ("event_type", "state", "user_message", "detail")}
                        for e in result.get("events", [])
                    ],
                }
                record["decisions"].append(decision)
                self._log(
                    f"[{runtime_session}] decision {n}: {result.get('event_type')} {result.get('state')} "
                    f"{str(result.get('user_message', ''))[:120]} ({decision['latency_s']} s)"
                )
                if result.get("state") == "UNKNOWN":
                    finish_reason = "agent_quarantined"
                    break
                if ctx.is_done_request(result):
                    finish_reason = "agent_done"
                    break
                # No motion means no simulated time passes: an Agent that keeps declining to act
                # would freeze the episode, so a few such replies in a row end it.
                idle = result.get("state") == "ANSWERED"
                idle_streak = idle_streak + 1 if idle else 0
                if idle_streak >= ctx.max_idle_replies:
                    finish_reason = "agent_stopped"
                    break
                if result.get("event_type") == "refused":
                    decision["response"] = result.get("response")  # never reached the planner
        except PlatformExited as exc:
            finish_reason = "platform_exited"
            record["error"] = str(exc)
        record["finish_reason"] = finish_reason
        if finish_reason in ("stale_observation", "not_admitted") and not self.book.episode_ended(key):
            # Never drain physics for an episode that was not run: stop it without a verdict.
            record["abort_call"] = self.io.abort_session()
        elif not self.book.episode_ended(key) and finish_reason != "platform_exited":
            record["finish_call"] = self.io.call_finish()
        end, native = self.book.wait_end_and_native(key, settings.end_wait_s)
        record["episode_end"] = end
        record["native_result"] = native  # evaluator side only; never handed to the context
        record["wall_s"] = round(time.monotonic() - t_episode, 1)
        os.makedirs(settings.out_dir, exist_ok=True)
        gen = "x" if generation is None else generation
        path = os.path.join(settings.out_dir, f"{run_id}_{runtime_session}_g{gen}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1, ensure_ascii=False, default=str)
        record["record_path"] = path
        native = native or {}
        self._log(
            f"[{runtime_session}] episode done: native success={native.get('native_success')} "
            f"reason={native.get('failure_reason')} decisions={len(record['decisions'])} finish={finish_reason} -> {path}"
        )
        return record

    def run(self, episodes: int, episode_wait_s: float, ready_timeout_s: float = 120.0) -> list[dict[str, Any]]:
        """Wait for the Agent, then run ``episodes`` episodes as the platform starts them."""
        if not self.io.agent_ready(ready_timeout_s):
            raise RuntimeError("the Agent node is not ready")
        records = []
        for _ in range(episodes):
            start = self.book.next_episode(episode_wait_s)
            if start is None:
                raise TimeoutError("no episode started in time")
            records.append(self.run_episode(start))
        return records
