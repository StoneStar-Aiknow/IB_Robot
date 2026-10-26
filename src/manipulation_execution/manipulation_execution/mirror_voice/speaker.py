"""The speaker: session state in, synthesis and playback calls out. No ROS.

The node feeds it session state messages and calls ``tick`` from a timer; it
makes the calls the speaker asks for through ``SpeakerCalls`` and keeps the
WAV files through ``SpeakerFiles``. Every answer comes back through the
``done`` callback it was given.

A phrase goes through:

==========  ==============================================================
step        what happens
==========  ==============================================================
queued      an event from ``core.TransitionFilter`` queues its phrase
            (``core.PhraseQueue``); nothing expires
cache       the phrase's WAV files, when every one of them still exists
synthesize  otherwise, once the synthesis service is ready; the answer's
            segments are checked and written, then cached by the text
play        segment by segment in index order, ``pause_after_ms`` between
            two segments; then the next phrase
==========  ==============================================================

An utterance — one dynamic sentence from the visual interaction feature,
handed to :meth:`Speaker.on_utterance` — goes through the same queue and the
same steps with two differences. It is synthesized on demand and never cached
(each sentence is spoken at most once, so a cache could only grow), and its
WAV files are removed the moment it finishes or fails: the board's temporary
directory is tmpfs, and unbounded utterance audio is a memory leak rather
than a disk one. When an ``utterance_gate`` is given, an utterance is also
re-checked at the moment it would start synthesizing — the state it was
admitted under may have changed while it waited.

* At most one call is out at a time. A call with no answer within its
  timeout is a failure; its late answer is recognised by its token and
  ignored.
* A phrase that fails (synthesis failed or timed out, bad audio, playback
  failed or timed out, playback service not ready) is dropped and counted,
  never retried.
* Precaching: with no phrase queued and the synthesis service ready, every
  phrase of every event is synthesized once, one at a time. A queued phrase
  always goes first, and its synthesis is its precache. A phrase whose
  synthesis failed is synthesized again only when it is next spoken.
* While phrases wait or precaching is unfinished and the synthesis service
  is missing, ``core.ServiceWatch`` says when to log an ERROR.
* Without the synthesis service a phrase is only taken off the queue when
  every phrase is cached; one whose files have gone since waits for the
  service as the phrase being spoken. An utterance cannot be served from a
  cache at all, so it waits for the service the same way, holding the queue.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from embodied_common.mirror_voice_contract import EVENTS, VoiceSection
from manipulation_execution.mirror_voice.core import PhraseQueue, ServiceWatch, TransitionFilter, pick_phrase

AUDIO_FORMAT = "wav_pcm_s16le"
FAILURES = ("synthesis", "synthesis_timeout", "audio", "playback", "playback_timeout", "playback_unavailable")

# The queue entry kind of an on-demand utterance (``on_utterance``), beside the
# state-phrase events of ``core.TransitionFilter``.
UTTERANCE = "utterance"

# success True / False is the answer; None: no answer (the call failed).
Done = Callable[[bool | None, str], None]
# As ``Done``, with the audio segments of a successful answer.
SynthesisDone = Callable[[bool | None, str, Sequence[Any]], None]


@dataclass(frozen=True)
class Segment:
    """One audio segment of a synthesis answer; the ROS message has the same fields."""

    index: int
    audio_format: str
    audio_data: bytes
    pause_after_ms: int = 0


class SpeakerCalls(Protocol):
    def tts_ready(self) -> bool: ...

    def playback_ready(self) -> bool: ...

    def synthesize(self, text: str, done: SynthesisDone) -> None: ...

    def play(self, path: str, done: Done) -> None: ...


class SpeakerFiles(Protocol):
    def write(self, name: str, data: bytes) -> str:
        """Write one WAV file; its path."""

    def exists(self, path: str) -> bool: ...

    def remove_file(self, path: str) -> None:
        """Remove one file; a file that is already gone is fine."""


def wav_name(text: str, index: int) -> str:
    return f"{hashlib.sha256(text.encode()).hexdigest()}-{index:04d}.wav"


class WavDirectory:
    """``SpeakerFiles`` in a directory of its own, removed by ``remove``.

    Someone else may clear a shared ``/tmp``: a write recreates the directory,
    and the speaker finds the gone files before playing them.
    """

    def __init__(self, prefix: str = "mirror-voice-"):
        self.path = tempfile.mkdtemp(prefix=prefix)

    def write(self, name: str, data: bytes) -> str:
        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, name)
        with open(path, "wb") as file:
            file.write(data)
        return path

    def exists(self, path: str) -> bool:
        return os.path.isfile(path)

    def remove_file(self, path: str) -> None:
        # Already gone is fine: the file is temporary either way.
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


@dataclass(frozen=True)
class _Clip:
    path: str
    pause_s: float


@dataclass
class _Speech:
    event: str
    text: str
    clips: tuple[_Clip, ...] = ()
    index: int = 0
    resume_at: float = 0.0


@dataclass
class _Flight:
    call: str
    token: int
    started_at: float
    timeout_s: float
    text: str


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return float(value)


class Speaker:
    def __init__(
        self,
        section: VoiceSection,
        calls: SpeakerCalls,
        files: SpeakerFiles,
        *,
        tts_timeout_s: float,
        playback_timeout_s: float,
        watch: ServiceWatch | None = None,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str, str], None] | None = None,
        utterance_gate: Callable[[], bool] | None = None,
    ):
        self._phrases = section.phrases
        self._calls = calls
        self._files = files
        self._tts_timeout_s = _positive("tts_timeout_s", tts_timeout_s)
        self._playback_timeout_s = _positive("playback_timeout_s", playback_timeout_s)
        self._watch = watch or ServiceWatch()
        self._clock = clock
        self._log = log or (lambda level, message: None)
        # Asked again when a queued utterance would start synthesizing: the
        # state it was admitted under may be gone by then. None gates nothing.
        self._utterance_gate = utterance_gate
        self._lock = threading.RLock()
        self._filter = TransitionFilter()
        self._queue = PhraseQueue(section.queue_max)
        self._texts = tuple(dict.fromkeys(text for event in EVENTS for text in self._phrases[event]))
        self._precache: deque[str] = deque(self._texts)
        self._cache: dict[str, tuple[_Clip, ...]] = {}
        self._speech: _Speech | None = None
        self._flight: _Flight | None = None
        self._tokens = 0
        self._tts_available: bool | None = None
        self.spoken = 0
        self.failed = dict.fromkeys(FAILURES, 0)
        self.utterances = 0
        self.utterances_dropped = 0
        self.cache_missing = 0
        self.late_answers = 0

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def on_state(self, message: str | Mapping) -> str | None:
        """One session state message (the JSON text or its mapping); the event it queued, if any."""
        with self._lock:
            try:
                content = json.loads(message) if isinstance(message, str) else message
            except ValueError:
                content = None  # counted as malformed by the filter
            event = self._filter.feed(content)
            if event is None:
                return None
            text = pick_phrase(self._phrases, event)
            dropped = self._queue.push(event, text)
            if dropped is not None:
                self._log("warning", f"queue full ({len(self._queue)}): dropped {dropped[0]} {dropped[1]!r}")
            self._log("info", f"{event}: queued {text!r}")
            self._advance(self._clock())
            return event

    def on_utterance(self, text: str) -> None:
        """One dynamic sentence from the visual interaction feature.

        The node has already admitted it (speakable kind, mirror MIRRORING, no
        ASCII letters); the queue, the pop-time gate and the same steps as a
        phrase do the rest. Nothing here expires: an utterance that went stale
        waiting is the gate's to drop, not the queue's.
        """
        text = text.strip()
        if not text:
            return
        with self._lock:
            self.utterances += 1
            dropped = self._queue.push(UTTERANCE, text)
            if dropped is not None:
                self._log("warning", f"queue full ({len(self._queue)}): dropped {dropped[0]} {dropped[1]!r}")
            self._log("info", f"{UTTERANCE}: queued {text!r}")
            self._advance(self._clock())

    def tick(self) -> None:
        """Call timeouts, the synthesis service wait, the next step; the node calls it a few times a second."""
        with self._lock:
            now = self._clock()
            flight = self._flight
            if flight is not None and now - flight.started_at > flight.timeout_s:
                reason = f"{flight.call}_timeout"
                self._fail(reason, flight.text, f"no answer within {flight.timeout_s:g} s")
            available = bool(self._calls.tts_ready())
            if available != self._tts_available:
                if available:
                    self._log("info", f"synthesis service ready; {len(self._precache)} phrases to precache")
                elif self._tts_available:
                    self._log("warning", "synthesis service gone")
                self._tts_available = available
            if self._waiting_for_tts():
                if self._watch.poll(now, available):
                    self._log(
                        "error",
                        f"synthesis service not available: {len(self._queue)} phrases waiting, "
                        f"{len(self._precache)} not synthesized; the mirror goes on without voice",
                    )
            else:
                self._watch.poll(now, True)  # nothing waits for it: the next wait starts afresh
            self._advance(now)

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------

    def _advance(self, now: float) -> None:
        """Take steps until a call is out, a pause runs or nothing is left to do."""
        while self._flight is None:
            speech = self._speech
            if speech is not None:
                if speech.clips:
                    if not self._play_next(speech, now):
                        return
                elif self._tts_ready():
                    self._send_synthesis(speech.text, now)
                else:
                    return
                continue
            head = self._queue.peek()
            if head is not None and head[0] == UTTERANCE:
                # The state an utterance was admitted under may be gone by the
                # time it would start synthesizing; the gate is asked again at
                # the only moment that matters. Closed means dropped, not held:
                # holding it would block the phrases behind it for as long as
                # the gate stays closed.
                if self._utterance_gate is not None and not self._utterance_gate():
                    self._queue.pop()
                    self.utterances_dropped += 1
                    self._log("info", f"{UTTERANCE}: dropped {head[1]!r}: the gate closed while it waited")
                    continue
                if not self._tts_ready():
                    # Never cached, so the service is its only way forward. It
                    # waits at the head rather than letting the phrases behind
                    # it jump the queue: their order is part of what is said.
                    return
                event, text = self._queue.pop()
                self._speech = _Speech(event, text)
                continue
            if len(self._queue) and (self._tts_ready() or self._all_cached()):
                event, text = self._queue.pop()
                self._speech = _Speech(event, text, self._cached_clips(text))
                continue
            if not len(self._queue) and self._precache and self._tts_ready():
                text = self._precache.popleft()
                if text not in self._cache:
                    self._send_synthesis(text, now)
                continue
            return

    def _waiting_for_tts(self) -> bool:
        speech = self._speech
        return bool(len(self._queue) or self._precache or (speech is not None and not speech.clips))

    def _tts_ready(self) -> bool:
        return bool(self._calls.tts_ready())

    def _all_cached(self) -> bool:
        return all(text in self._cache for text in self._texts)

    def _cached_clips(self, text: str) -> tuple[_Clip, ...]:
        clips = self._cache.get(text, ())
        if clips and not all(self._files.exists(clip.path) for clip in clips):
            del self._cache[text]
            self.cache_missing += 1
            self._log("warning", f"cached audio of {text!r} is gone; synthesizing it again")
            return ()
        return clips

    def _play_next(self, speech: _Speech, now: float) -> bool:
        """Play the next segment or finish the phrase; False while a pause runs."""
        if speech.index >= len(speech.clips):
            self.spoken += 1
            self._speech = None
            self._discard(speech)
            self._log("info", f"{speech.event}: said {speech.text!r}")
            return True
        if now < speech.resume_at:
            return False
        if not self._calls.playback_ready():
            self._fail("playback_unavailable", speech.text, "playback service not available")
            return True
        self._send("playback", speech.text, self._playback_timeout_s, now, speech.clips[speech.index].path)
        return True

    def _send_synthesis(self, text: str, now: float) -> None:
        if text in self._precache:
            self._precache.remove(text)  # tried once, whatever the answer
        self._send("synthesis", text, self._tts_timeout_s, now, text)

    def _send(self, call: str, text: str, timeout_s: float, now: float, argument: str) -> None:
        self._tokens += 1
        token = self._tokens
        self._flight = _Flight(call, token, now, timeout_s, text)
        try:
            if call == "synthesis":

                def synthesized(ok: bool | None, message: str, segments: Sequence[Any] = ()) -> None:
                    self._answer(token, ok, message, segments)

                self._calls.synthesize(argument, synthesized)
            else:
                self._calls.play(argument, lambda ok, message: self._answer(token, ok, message, ()))
        except Exception as exc:  # noqa: BLE001 - a failed call is a failed phrase
            self._answer(token, None, f"call failed: {exc}", ())

    def _answer(self, token: int, ok: bool | None, message: str, segments: Sequence[Any]) -> None:
        with self._lock:
            flight = self._flight
            if flight is None or flight.token != token:
                self.late_answers += 1  # an answer after its timeout, or a second answer
                return
            if ok is not True:
                self._fail(flight.call, flight.text, f"{'no answer' if ok is None else 'failed'}: {message}")
            elif flight.call == "synthesis":
                self._flight = None
                self._synthesized(flight.text, segments)
            else:
                self._flight = None
                self._played()
            self._advance(self._clock())

    def _synthesized(self, text: str, segments: Sequence[Any]) -> None:
        try:
            clips = self._write(text, segments)
        except Exception as exc:  # noqa: BLE001 - audio that cannot be checked or written is bad audio
            self._fail("audio", text, str(exc))
            return
        speech = self._speech
        if text in self._texts:
            self._cache[text] = clips
        # An utterance's clips are not cached — each is spoken at most once —
        # and its files are removed when its speech ends or fails.
        if speech is not None and speech.text == text and not speech.clips:
            speech.clips = clips

    def _played(self) -> None:
        speech = self._speech
        speech.index += 1
        if speech.index < len(speech.clips):
            speech.resume_at = self._clock() + speech.clips[speech.index - 1].pause_s

    def _discard(self, speech: _Speech) -> None:
        """Remove an utterance's WAV files; a state phrase's stay for the next time."""
        if speech.event == UTTERANCE and speech.text not in self._texts:
            for clip in speech.clips:
                self._files.remove_file(clip.path)

    def _write(self, text: str, segments: Sequence[Any]) -> tuple[_Clip, ...]:
        """Check the segments as the visual game announcer does, then write them."""
        ordered = sorted(segments, key=lambda segment: int(segment.index))
        if not ordered:
            raise ValueError("synthesis returned no audio segments")
        if [int(segment.index) for segment in ordered] != list(range(len(ordered))):
            raise ValueError("audio segment indexes must be contiguous from zero")
        clips = []
        try:
            for segment in ordered:
                if str(segment.audio_format).strip().lower() != AUDIO_FORMAT:
                    raise ValueError(f"unsupported audio format {segment.audio_format!r}")
                data = bytes(segment.audio_data)
                if not data.startswith(b"RIFF") or b"WAVE" not in data[:16]:
                    raise ValueError(f"audio segment {segment.index} is not a WAV file")
                path = self._files.write(wav_name(text, int(segment.index)), data)
                clips.append(_Clip(path, max(0.0, int(segment.pause_after_ms) / 1000.0)))
        except Exception:
            # A speech that cannot be written is a dropped one, and nothing
            # else will ever remove the files written so far.
            for clip in clips:
                self._files.remove_file(clip.path)
            raise
        return tuple(clips)

    def _fail(self, reason: str, text: str, message: str) -> None:
        """Count a failure and drop what it was for: the phrase being spoken, or a precache."""
        self.failed[reason] += 1
        self._flight = None
        speech = self._speech
        if speech is not None and speech.text == text:
            self._speech = None
            self._discard(speech)
            self._log("warning", f"{speech.event}: dropped {text!r}: {reason}: {message}")
        else:
            self._log("warning", f"precache of {text!r}: {reason}: {message}")

    # ------------------------------------------------------------------
    # ~/status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tts_available": self._tts_available,
                "queue": len(self._queue),
                "speaking": self._speech and self._speech.text,
                "in_flight": self._flight and self._flight.call,
                "spoken": self.spoken,
                "utterances": self.utterances,
                "utterances_dropped": self.utterances_dropped,
                "dropped_queue_full": self._queue.dropped,
                "failed": dict(self.failed),
                "late_answers": self.late_answers,
                "cached": len(self._cache),
                "cache_missing": self.cache_missing,
                "precache_pending": len(self._precache),
                "received": self._filter.received,
                "events": self._filter.events,
                "malformed": self._filter.malformed,
            }
