"""The operator: frames, touches and the session state in, calls out. No ROS.

The node feeds it and makes the calls it asks for through ``OperatorCalls``;
every answer comes back through the ``done`` callback it was given.

==================  ==============================  ==========================
input               when                            call
==================  ==============================  ==========================
double cover        session state WAIT              ``mirror_start`` (Gateway)
long cover          session STANDBY or MIRRORING    ``mirror_stop`` (Gateway)
head touch held     any session state               ``operator_touch``
(``tick``)          on/off stable ``stable_s``      the light
(``tick``)          on for ``max_session_s``        ``mirror_stop`` (Gateway)
==================  ==============================  ==========================

* The session state is the ``state`` of the newest message on the session
  state topic. Its age is this process's receive time, not the message's
  ``stamp`` (the session's clock). Silent for longer than
  ``session_silent_s``: no session, nothing is sent and the light goes off.
  The old message a transient_local subscription hands over counts once.
* Two channels, the Gateway and the session, each carry at most one call; a
  gesture or touch whose channel is busy is dropped and logged. A call with
  no answer within its timeout frees the channel; a late answer is ignored.
* ``mirror_stop`` is the skill the agent sends, so a stop from here and one
  from the agent leave the Gateway in the same state. An answer from the
  Gateway is obeyed: a refusal (``SKILL_BUSY``: another skill holds it) is not
  worked around. With no answer (its service or action is not there, the call
  failed, nothing within ``GATEWAY_TIMEOUT_S``) and the session still STANDBY
  or MIRRORING, the session is stopped directly with ``set_session(false)``;
  ``~/status`` shows it as ``stop_fallback``.
* The light is only sent, never waited for, and is not a channel.
* ``send_enabled`` False: every call is recorded as not sent, nothing is sent.
* ``operator_touch`` answering ``success=False`` ("nothing to do") is a normal
  answer, recorded as ``declined``.
* The session timeout (``max_session_s`` > 0) starts at the first state
  message that is not WAIT and is cleared only by a WAIT message: STANDBY and
  MIRRORING back and forth, a head touch, or the state topic going silent do
  not clear it. Due, the stop is sent like a long cover's, only while the
  Gateway channel is free and the session is not silent. A stop that fails or
  is refused is sent again at most every ``TIMEOUT_RETRY_S`` until WAIT comes;
  an answered stop, or one not sent because ``send_enabled`` is False, is not.
  A later ``mirror_start`` while the session is on does not restart the count:
  the session answers "already on" and no WAIT comes.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from embodied_common.mirror_session_contract import DEFAULT_STOP_TIMEOUT_SEC, STOP_TIMEOUT_MARGIN_SEC
from manipulation_execution.mirror_operator.config import LedColor, OperatorConfig
from manipulation_execution.mirror_operator.gateway import RESULT_WAIT_SEC
from manipulation_execution.mirror_operator.gesture import DOUBLE_COVER, LONG_COVER, GestureRecognizer
from manipulation_execution.mirror_operator.judge import CoverJudge, GrayFrame
from manipulation_execution.mirror_operator.led import ON, LedPolicy, status_note
from manipulation_execution.mirror_operator.touch import TouchDetector
from manipulation_execution.mirror_session.core import SessionState

GATEWAY_TIMEOUT_S = RESULT_WAIT_SEC
STOP_TIMEOUT_S = DEFAULT_STOP_TIMEOUT_SEC + STOP_TIMEOUT_MARGIN_SEC
TOUCH_TIMEOUT_S = 2.0
TIMEOUT_RETRY_S = 2.0
FRAME_HZ_WINDOW_S = 1.0
SESSION_ON = (SessionState.STANDBY.value, SessionState.MIRRORING.value)
_STATES = tuple(state.value for state in SessionState)
TOUCH_INTERFACE = "touch.head"

# success True / False is the answer; None: no answer (the call failed).
Done = Callable[[bool | None, str], None]
# The light's status_code, or None with the reason there is none.
LedDone = Callable[[int | None, str], None]


class OperatorCalls(Protocol):
    def mirror_start(self, done: Done) -> None: ...

    def mirror_stop(self, done: Done) -> None: ...

    def stop_session(self, done: Done) -> None: ...

    def operator_touch(self, done: Done) -> None: ...

    def set_led(self, color: LedColor, priority: int, done: LedDone) -> None: ...


@dataclass(frozen=True)
class TouchEndpoint:
    endpoint: str
    message_type: str
    qos: Mapping


def resolve_touch_endpoint(description: Mapping) -> tuple[TouchEndpoint | None, str]:
    """``touch.head`` from the runtime description, or ``(None, reason)``: the touch is then off."""
    interfaces = description.get("interfaces") if isinstance(description, Mapping) else None
    item = interfaces.get(TOUCH_INTERFACE) if isinstance(interfaces, Mapping) else None
    if not isinstance(item, Mapping):
        return None, f"{TOUCH_INTERFACE} is not in the runtime description"
    if (item.get("kind"), item.get("direction")) != ("topic", "publish"):
        return None, f"{TOUCH_INTERFACE} must be a published topic"
    endpoint, message_type = item.get("endpoint"), item.get("message_type")
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        return None, f"{TOUCH_INTERFACE} endpoint must be an absolute ROS name"
    if not isinstance(message_type, str) or not message_type:
        return None, f"{TOUCH_INTERFACE} has no message_type"
    return TouchEndpoint(endpoint, message_type, dict(item.get("qos") or {})), ""


@dataclass
class _Flight:
    call: str
    token: int
    started_at: float
    timeout_s: float


def _age(record: dict | None, now: float) -> dict | None:
    if record is None:
        return None
    shown = {key: value for key, value in record.items() if not key.endswith("_at")}
    for key, value in record.items():
        if key.endswith("_at"):
            shown[key[: -len("_at")] + "_age_s"] = None if value is None else round(now - value, 3)
    return shown


class Operator:
    def __init__(
        self,
        config: OperatorConfig,
        judge: CoverJudge,
        calls: OperatorCalls,
        *,
        send_enabled: bool,
        touch_unavailable: str = "",
        led_unavailable: str = "",
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str, str], None] | None = None,
    ):
        self._config = config
        self._judge = judge
        self._calls = calls
        self._send_enabled = bool(send_enabled)
        self._touch_unavailable = touch_unavailable
        self._led_unavailable = led_unavailable
        self._clock = clock
        self._log = log or (lambda level, message: None)
        self._lock = threading.RLock()
        self._gestures = GestureRecognizer(config)
        self._touch = TouchDetector(config.touch)
        self._led = LedPolicy(config.led)
        self._tokens = 0
        self._flights: dict[str, _Flight | None] = {"gateway": None, "session": None}
        self._frame_times: deque[float] = deque()
        self._frame: GrayFrame | None = None
        self._covered: bool | None = None
        self._frame_error: dict | None = None
        self._gesture: dict | None = None
        self._touch_last: dict | None = None
        self._state: str | None = None
        self._state_at: float | None = None
        self._state_error = ""
        self._wait_at: float | None = None  # the newest WAIT message
        self._last_gateway: dict | None = None
        self._last_session_call: dict | None = None
        self._stop_fallback: dict | None = None
        self._last_led: dict | None = None
        self._led_token = 0
        self.dropped = {"gateway": 0, "session": 0}
        # the session timeout: start, stops sent this round, when the next may go, the last stop's record
        self._timeout_started_at: float | None = None
        self._timeout_attempts = 0
        self._timeout_next_at = 0.0
        self._timeout_done = False
        self._timeout_record: dict | None = None

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def on_frame(self, frame: GrayFrame) -> str | None:
        """Judge one decoded frame; the gesture it completed, if any."""
        with self._lock:
            now = self._clock()
            self._frame_times.append(now)
            self._frame = frame
            try:
                covered = bool(self._judge.covered(frame))
                self._frame_error = None
            except Exception as exc:  # noqa: BLE001 - a business judge must not stop the node
                covered = False
                self._frame_error = {"reason": f"cover judge failed: {exc}", "at": now}
            self._covered = covered
            gesture = self._gestures.update(covered, now)
            if gesture is not None:
                self._on_gesture(gesture, now)
            return gesture

    def frame_failed(self, reason: str) -> None:
        """A frame that did not decode; the gesture state sees it as a gap."""
        with self._lock:
            self._frame_error = {"reason": reason, "at": self._clock()}

    def on_touch(self, event_type: int) -> bool:
        """One touch message; True when it completed a hold."""
        with self._lock:
            now = self._clock()
            if not self._touch.update(int(event_type), now):
                return False
            state = self._session_state(now)
            if state is None:
                self._touch_last = {"action": "ignored: no session", "at": now}
            else:
                self._touch_last = {"action": self._send("session", "touch", now), "at": now}
            return True

    def on_state(self, message: str | Mapping) -> None:
        """One session state message (the JSON text or its mapping); only ``state`` is read."""
        with self._lock:
            now = self._clock()
            try:
                content = json.loads(message) if isinstance(message, str) else message
                state = content.get("state") if isinstance(content, Mapping) else None
                error = "" if state in _STATES else f"unknown session state {state!r}"
            except ValueError as exc:
                error = f"unreadable state message: {exc}"
            if error:
                self._state_error = error
                self._log("warning", error)
                return
            self._state, self._state_at, self._state_error = state, now, ""
            if state == SessionState.WAIT.value:
                self._wait_at = now
                self._timeout_started_at = None
                self._timeout_attempts, self._timeout_next_at, self._timeout_done = 0, 0.0, False
            elif self._timeout_started_at is None:
                self._timeout_started_at = now

    def tick(self) -> None:
        """Call timeouts, the session timeout and the light; the node calls it about 10 times a second."""
        with self._lock:
            now = self._clock()
            for channel, flight in self._flights.items():
                if flight is not None and now - flight.started_at > flight.timeout_s:
                    self._finish(channel, flight.token, "timeout", f"no answer within {flight.timeout_s:g} s", now)
            self._session_timeout(now)
            color = self._led.update(self._session_state(now) in SESSION_ON, now)
            if color is not None:
                self._set_led(color, now)

    # ------------------------------------------------------------------
    # calls
    # ------------------------------------------------------------------

    def _session_state(self, now: float) -> str | None:
        if self._state_at is None or now - self._state_at > self._config.session_silent_s:
            return None
        return self._state

    def _on_gesture(self, gesture: str, now: float) -> None:
        state = self._session_state(now)
        if gesture == DOUBLE_COVER and state == SessionState.WAIT.value:
            action = self._send("gateway", "mirror_start", now)
        elif gesture == LONG_COVER and state in SESSION_ON:
            action = self._send("gateway", "mirror_stop", now)
        else:
            action = f"ignored: session {state or 'none'}"
        self._gesture = {"name": gesture, "action": action, "at": now}
        self._log("info", f"{gesture}: {action}")

    def _session_timeout(self, now: float) -> None:
        limit = self._config.max_session_s
        started = self._timeout_started_at
        if limit <= 0 or started is None or self._timeout_done or now - started < limit:
            return
        record = self._timeout_record
        if self._timeout_attempts and record is not None:
            if record["result"] == "":
                return  # the stop is in flight
            if record["result"] == "ok":
                self._timeout_done = True
                return
        if now < self._timeout_next_at or self._flights["gateway"] is not None:
            return
        if self._session_state(now) not in SESSION_ON:
            return  # silent: no session to stop until a state message comes again
        self._timeout_attempts += 1
        self._timeout_next_at = now + TIMEOUT_RETRY_S
        action = self._send("gateway", "mirror_stop", now, for_timeout=True)
        if not self._send_enabled:
            self._timeout_done = True
        self._log("info", f"session timeout ({limit:g} s, attempt {self._timeout_attempts}): {action}")

    def _send(self, channel: str, call: str, now: float, *, for_timeout: bool = False) -> str:
        # for_timeout: set before the call is made, since a missing service answers within it
        record = {"call": call, "result": "", "message": "", "sent_at": now, "answered_at": None}
        if not self._send_enabled:
            record["result"] = "not_sent"
            record["message"] = "send_enabled is false"
            self._store(channel, record, for_timeout)
            return "not sent: send_enabled is false"
        flight = self._flights[channel]
        if flight is not None:
            self.dropped[channel] += 1
            self._log("warning", f"{call} dropped: {flight.call} is still in flight on the {channel} channel")
            return f"dropped: {flight.call} in flight"
        self._tokens += 1
        token = self._tokens
        timeout_s = {
            "mirror_start": GATEWAY_TIMEOUT_S,
            "mirror_stop": GATEWAY_TIMEOUT_S,
            "stop": STOP_TIMEOUT_S,
            "touch": TOUCH_TIMEOUT_S,
        }[call]
        self._flights[channel] = _Flight(call, token, now, timeout_s)
        self._store(channel, record, for_timeout)
        method = {
            "mirror_start": self._calls.mirror_start,
            "mirror_stop": self._calls.mirror_stop,
            "stop": self._calls.stop_session,
        }.get(call, self._calls.operator_touch)

        def done(success: bool | None, message: str) -> None:
            if success is None:
                result = "error"
            elif success:
                result = "ok"
            else:
                result = "failed" if channel == "gateway" else "declined"
            with self._lock:
                self._finish(channel, token, result, message, self._clock())

        try:
            method(done)
        except Exception as exc:  # noqa: BLE001 - a failed call frees its channel
            self._finish(channel, token, "error", f"call failed: {exc}", now)
        return f"sent {call}"

    def _store(self, channel: str, record: dict, for_timeout: bool) -> None:
        if channel == "gateway":
            self._last_gateway = record
        else:
            self._last_session_call = record
        if for_timeout:
            self._timeout_record = record

    def _finish(self, channel: str, token: int, result: str, message: str, now: float) -> None:
        flight = self._flights[channel]
        if flight is None or flight.token != token:
            return  # an answer after its timeout, or a second answer
        self._flights[channel] = None
        record = self._last_gateway if channel == "gateway" else self._last_session_call
        record.update(result=result, message=message, answered_at=now)
        if result in ("timeout", "error", "failed"):
            self._log("warning", f"{flight.call}: {result}: {message}")
        else:
            self._log("info", f"{flight.call}: {result}: {message}")
        if flight.call == "mirror_stop" and result in ("timeout", "error"):
            self._stop_directly(record, message, now)

    def _stop_directly(self, record: dict, reason: str, now: float) -> None:
        """The Gateway gave ``record`` no answer: stop the session itself, if that round is still on."""
        state = self._session_state(now)
        if self._wait_at is not None and self._wait_at >= record["sent_at"]:
            action = "not sent: the session went back to WAIT after mirror_stop was sent"
        elif state in SESSION_ON:
            for_timeout = record is self._timeout_record
            action = self._send("session", "stop", now, for_timeout=for_timeout)
            if for_timeout:
                self._timeout_next_at = now + TIMEOUT_RETRY_S
        else:
            action = f"not sent: session {state or 'none'}"
        self._stop_fallback = {"reason": f"no answer from the Gateway: {reason}", "action": action, "at": now}
        self._log("warning", f"no answer from the Gateway to mirror_stop, direct session stop: {action}")

    def _set_led(self, color: str, now: float) -> None:
        self._led_token += 1
        token = self._led_token
        record = {"color": color, "result": "", "status_code": None, "note": "", "sent_at": now, "answered_at": None}
        self._last_led = record
        if self._led_unavailable:
            record.update(result="unavailable", note=self._led_unavailable)
            return
        if not self._send_enabled:
            record.update(result="not_sent", note="send_enabled is false")
            return
        record["result"] = "sent"

        def done(code: int | None, message: str) -> None:
            with self._lock:
                if token != self._led_token:
                    return  # a newer request was sent
                if code is None:
                    record.update(result="error", note=message or status_note(None), answered_at=self._clock())
                else:
                    note = status_note(code)
                    record.update(result="answered", status_code=code, note=note, answered_at=self._clock())

        led = self._config.led
        try:
            self._calls.set_led(led.on if color == ON else led.off, led.priority, done)
        except Exception as exc:  # noqa: BLE001 - the light never stops the node
            record.update(result="error", note=f"call failed: {exc}")

    # ------------------------------------------------------------------
    # ~/status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            while self._frame_times and now - self._frame_times[0] > FRAME_HZ_WINDOW_S:
                self._frame_times.popleft()
            frame = self._frame
            state = self._session_state(now)
            started = self._timeout_started_at
            return {
                "send_enabled": self._send_enabled,
                "frame_hz": len(self._frame_times) / FRAME_HZ_WINDOW_S,
                "mean": None if frame is None else round(frame.mean, 2),
                "max": None if frame is None else round(frame.max, 2),
                "level": self._judge.level,
                "covered": self._covered,
                "frame_error": _age(self._frame_error, now),
                "armed": self._gestures.armed,
                "gesture": _age(self._gesture, now),
                "touch": {
                    "available": not self._touch_unavailable,
                    "reason": self._touch_unavailable,
                    "event_type": self._touch.event_type,
                    "last": _age(self._touch_last, now),
                },
                "session_state": state,
                "session_age_s": None if self._state_at is None else round(now - self._state_at, 3),
                "session_error": self._state_error,
                "in_flight": {channel: flight and flight.call for channel, flight in self._flights.items()},
                "dropped": {
                    **self.dropped,
                    "gesture_debounce": self._gestures.dropped,
                    "touch_debounce": self._touch.dropped,
                },
                "last_gateway": _age(self._last_gateway, now),
                "last_session_call": _age(self._last_session_call, now),
                "stop_fallback": _age(self._stop_fallback, now),
                "session_timeout": {
                    "enabled": self._config.max_session_s > 0,
                    "max_session_s": self._config.max_session_s,
                    "elapsed_s": None if started is None else round(now - started, 3),
                    "fired": self._timeout_attempts > 0,
                    "attempts": self._timeout_attempts,
                    "done": self._timeout_done,
                    "last": _age(self._timeout_record, now),
                },
                "led": {
                    "available": not self._led_unavailable,
                    "reason": self._led_unavailable,
                    "wanted": self._led.wanted,
                    "applied": self._led.applied,
                },
                "last_led": _age(self._last_led, now),
            }
