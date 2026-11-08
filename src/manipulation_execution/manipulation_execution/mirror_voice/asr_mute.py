"""The ASR held quiet while a prerecorded clip plays. No ROS.

Without it the ASR hears the robot's own voice: a goodbye clip says 再见, and
再见 is a hotword. The self introduction (``skill_executor``'s
``runtime_play_audio``) has the same problem and the same answer, and this
module speaks to the ASR exactly as it does: a ``std_msgs/String`` whose
``data`` is ``stop`` or ``start`` on ``/voice_control`` (``asr_control_topic``).
The ASR takes ``start`` only when idle and ``stop`` only while listening or
recognizing, so a repeat is harmless.

``mode`` (``internvl_voice.asr``):

========  ================================================================
mode      what is sent
========  ================================================================
``mute``  ``stop`` before a clip plays, if the ASR may be running; ``start``
          ``resume_delay_s`` after it ends, unless another clip starts in
          between (tried again every tick if it cannot be sent, unless a
          clip is playing); ``start`` at shutdown if it is still stopped by us
``off``   ``stop`` at activation, before every clip and whenever the ASR
          reports it is running again; ``start`` at shutdown
``keep``  nothing
========  ================================================================

"May be running" is the last state the ASR published on ``/voice_status``
(``listening`` or ``recognizing``), or no state known at all — the self
introduction stops it unconditionally, so not knowing errs the same way. A
``start`` we send makes the state unknown again until the ASR reports. An ASR
that was idle before a clip (someone else stopped it) is left alone, and is
not started afterwards either: only what this class stopped is restarted.

The owner calls ``playback_started`` / ``playback_ended`` from the speaker's
playback callbacks, ``tick`` from a timer and ``shutdown`` on exit, while the
publisher still works. ``shutdown`` may be called again: it sends nothing once
the ``start`` went out, and after it the other inputs do nothing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

MODES = ("mute", "off", "keep")
STOP = "stop"
START = "start"
# The ASR states (``voice_asr_node`` ``NodeState`` values) it hears in.
ACTIVE_ASR_STATES = frozenset({"listening", "recognizing"})


class AsrMute:
    def __init__(
        self,
        mode: str,
        resume_delay_s: float,
        publish: Callable[[str], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str, str], None] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, not {mode!r}")
        if (
            isinstance(resume_delay_s, bool)
            or not isinstance(resume_delay_s, int | float)
            or not math.isfinite(resume_delay_s)
            or resume_delay_s < 0
        ):
            raise ValueError("resume_delay_s must be a finite number of seconds, 0 or more")
        self.mode = mode
        self._delay = float(resume_delay_s)
        self._publish = publish
        self._clock = clock
        self._log = log or (lambda level, message: None)
        # The last /voice_status state; None: not known.
        self._asr_state: str | None = None
        self._muted = False  # a stop of ours is not yet followed by a start
        self._playing = False
        self._resume_at: float | None = None
        self._closed = False  # shutdown was called
        self._send_failing = False  # one warning per run of failures, not one per tick
        self.stops = 0
        self.starts = 0
        self.publish_errors = 0

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """The node is up; ``off`` stops the ASR now."""
        if self.mode == "off":
            self._stop()

    def on_asr_status(self, state: str) -> None:
        """One ``/voice_status`` message (``idle``, ``listening``, ...)."""
        self._asr_state = str(state).strip().lower()
        if self.mode == "off" and not self._closed and self._asr_state in ACTIVE_ASR_STATES:
            # Someone else (the voice agent, the self introduction) started it.
            self._stop()

    def playback_started(self) -> None:
        if self.mode == "keep" or self._closed:
            return
        self._playing = True
        self._resume_at = None  # the next clip came inside the window: stay stopped
        if self.mode == "off" or (not self._muted and self._may_be_running()):
            self._stop()

    def playback_ended(self) -> None:
        if self.mode == "keep" or self._closed:
            return
        self._playing = False
        if self.mode == "mute" and self._muted:
            self._resume_at = self._clock() + self._delay
            self.tick()

    def tick(self) -> None:
        """Send a due ``start``, or the one that could not be sent last tick."""
        if self._resume_at is None or self._playing or self._clock() < self._resume_at:
            return
        self._resume_at = None
        self._start()

    def shutdown(self) -> None:
        """Leave the ASR as it was found: start it if it is stopped by us."""
        self._closed = True
        self._resume_at = None
        if self.mode != "keep" and self._muted:
            self._start()

    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "muted": self._muted,
            "asr_state": self._asr_state,
            "stops": self.stops,
            "starts": self.starts,
            "publish_errors": self.publish_errors,
        }

    def _may_be_running(self) -> bool:
        return self._asr_state is None or self._asr_state in ACTIVE_ASR_STATES

    def _stop(self) -> None:
        if self._send(STOP):
            self.stops += 1
            self._muted = True

    def _start(self) -> None:
        if not self._send(START):
            # Still stopped by us. The next tick tries again; a clip starting
            # first cancels that, and its end asks for the start anew.
            if self.mode == "mute" and not self._closed:
                self._resume_at = self._clock()
            return
        self.starts += 1
        self._muted = False
        self._asr_state = None  # running again, as far as we know, until it reports

    def _send(self, command: str) -> bool:
        try:
            self._publish(command)
        except Exception as exc:  # noqa: BLE001 - the voice must go on without the ASR
            self.publish_errors += 1
            if not self._send_failing:
                self._log("warning", f"ASR {command} not sent: {exc}")
            self._send_failing = True
            return False
        self._send_failing = False
        return True
