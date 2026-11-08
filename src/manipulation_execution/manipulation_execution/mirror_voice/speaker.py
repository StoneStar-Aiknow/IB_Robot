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

Prerecorded mode (``clips`` given, ``internvl_voice`` on) never synthesizes
and never looks at the synthesis service; everything said is a checked WAV
file of ``PrerecordedClips``:

* an event plays a random usable clip of its pool, never the one it played
  last time; an utterance plays the clip whose text it is (the node refuses
  one without a clip as ``noclip``);
* a clip whose file is gone when its turn comes is counted
  (``clip_missing``) and replaced by another clip of the same event or the
  standby words; an utterance is dropped. Nothing falls back to synthesis;
* an event queued clears the queued utterances and standby words; the clip
  playing finishes. At the pop, an utterance is dropped if the gate is closed
  (the mirror left MIRRORING) and a standby word if the session left STANDBY;
* the goodbye is said as a stop's wrap-up begins, not when the session
  reaches WAIT (``core.TransitionFilter`` ``early_goodbye``): the arm's way
  back to the reset pose covers the time it takes. It also stops the
  standby words;
* standby words: only after ``ready`` and while the session is in STANDBY.
  The first ``standby_first_after_s`` after the ready clip ended (or after
  the event, if it had no clip), the next ``standby_gap_s`` after the last
  word ended, both drawn afresh each time; never the same word twice in a
  row. Leaving STANDBY clears the timer and the queued words;
* ``on_playback_start`` / ``on_playback_end`` bracket every playback call
  (both modes; the node mutes the ASR with them), ``on_speaking`` hears every
  change of ``speaking``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from embodied_common.mirror_session_contract import STATE_STALE_SEC, STATE_STANDBY
from embodied_common.mirror_voice_contract import EVENTS, VoiceSection
from manipulation_execution.mirror_voice.core import GOODBYE, PhraseQueue, ServiceWatch, TransitionFilter, pick_phrase

AUDIO_FORMAT = "wav_pcm_s16le"
FAILURES = ("synthesis", "synthesis_timeout", "audio", "playback", "playback_timeout", "playback_unavailable")

# The queue entry kind of an on-demand utterance (``on_utterance``), beside the
# state-phrase events of ``core.TransitionFilter``.
UTTERANCE = "utterance"
# The queue entry kind of a standby word (prerecorded mode only).
STANDBY = "standby"

# Every prerecorded clip is 24 kHz mono 16-bit PCM, as ``/voice_tts/play`` streams it.
CLIP_RATE = 24000
CLIP_CHANNELS = 1
CLIP_SAMPLE_BYTES = 2

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


class ClipFiles:
    """``SpeakerFiles`` of prerecorded mode: it reads the deployed clips and writes nothing."""

    path = ""

    def write(self, name: str, data: bytes) -> str:
        raise RuntimeError("prerecorded mode writes no audio")

    def exists(self, path: str) -> bool:
        return os.path.isfile(path)

    def remove_file(self, path: str) -> None:
        pass  # the deployed clips are not ours to remove

    def remove(self) -> None:
        pass


# ----------------------------------------------------------------------
# prerecorded clips
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PrerecordedClips:
    """The clips the speaker may play, every one checked; text -> WAV path."""

    paths: Mapping[str, str]
    # Event -> the texts of its usable clips, in table order.
    events: Mapping[str, tuple[str, ...]]
    standby: tuple[str, ...] = ()
    standby_first_after_s: tuple[float, float] = (1.0, 3.0)
    standby_gap_s: tuple[float, float] = (2.0, 5.0)


@dataclass(frozen=True)
class ClipReport:
    """What ``load_clips`` found: the directory used and what is wrong with the rest."""

    directory: str | None
    found: int
    total: int
    tried: tuple[str, ...]
    # Clip id -> why it cannot be played.
    problems: Mapping[str, str] = field(default_factory=dict)
    # Events left with no usable clip.
    silent_events: tuple[str, ...] = ()


def check_wav(path: str) -> str | None:
    """Why ``path`` cannot be played as a clip, or None when it can."""
    if not os.path.isfile(path):
        return "missing"
    try:
        with wave.open(path, "rb") as stream:
            rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
    except (wave.Error, EOFError, OSError) as exc:
        return f"not a PCM WAV file ({exc})"
    if (rate, channels, width) != (CLIP_RATE, CLIP_CHANNELS, CLIP_SAMPLE_BYTES):
        return f"{rate} Hz, {channels} channel(s), {8 * width} bit; needs {CLIP_RATE} Hz mono 16 bit"
    return None


def load_clips(
    playback: Mapping[str, Any],
    *,
    check: Callable[[str], str | None] = check_wav,
    isfile: Callable[[str], bool] = os.path.isfile,
) -> tuple[PrerecordedClips, ClipReport]:
    """The usable clips of ``parse_playback_json``'s dict, from the first ``clip_dirs`` entry that has any."""
    clips: Mapping[str, str] = playback["clips"]
    suffix = playback["wav_suffix"]
    tried = tuple(playback["clip_dirs"])
    directory = next(
        (folder for folder in tried if any(isfile(os.path.join(folder, f"{clip_id}{suffix}")) for clip_id in clips)),
        None,
    )
    problems: dict[str, str] = {}
    paths: dict[str, str] = {}
    usable: set[str] = set()
    for clip_id, text in clips.items():
        if directory is None:
            problems[clip_id] = "missing"
            continue
        path = os.path.join(directory, f"{clip_id}{suffix}")
        reason = check(path)
        if reason is None:
            paths[text] = path
            usable.add(clip_id)
        else:
            problems[clip_id] = reason
    events = {
        event: tuple(clips[clip_id] for clip_id in playback["events"].get(event, ()) if clip_id in usable)
        for event in EVENTS
    }
    standby = playback["standby"]
    loaded = PrerecordedClips(
        paths=paths,
        events=events,
        standby=tuple(clips[clip_id] for clip_id in standby["ids"] if clip_id in usable),
        standby_first_after_s=tuple(standby["first_after_sec"]),
        standby_gap_s=tuple(standby["gap_sec"]),
    )
    report = ClipReport(
        directory=directory,
        found=len(usable),
        total=len(clips),
        tried=tried,
        problems=problems,
        silent_events=tuple(event for event in EVENTS if not events[event]),
    )
    return loaded, report


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
        clips: PrerecordedClips | None = None,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_end: Callable[[], None] | None = None,
        on_speaking: Callable[[str | None], None] | None = None,
        rng: random.Random | None = None,
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
        self._filter = TransitionFilter(early_goodbye=clips is not None)
        self._queue = PhraseQueue(section.queue_max)
        self._clips = clips
        self._on_playback_start = on_playback_start
        self._on_playback_end = on_playback_end
        self._on_speaking = on_speaking
        if clips is None:
            self._texts = tuple(dict.fromkeys(text for event in EVENTS for text in self._phrases[event]))
            self._precache: deque[str] = deque(self._texts)
        else:
            # Every clip text is a phrase, so _discard never touches a deployed
            # file; nothing is precached because nothing is synthesized.
            self._texts = tuple(clips.paths)
            self._precache = deque()
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
        self.synthesis_calls = 0
        # Prerecorded mode.
        self._rng = rng or random.Random()
        self._speaking: str | None = None
        self._session_state = ""
        self._session_state_at: float | None = None
        self._last_pick: dict[str, str] = {}
        self._gone: set[str] = set()
        self._standby_after_ready = False  # waiting for the ready clip to end
        self._standby_due: float | None = None
        self.clip_missing = 0
        self.events_silent = 0
        self.standby_queued = 0
        self.standby_dropped = 0
        self.cleared = 0

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def on_state(self, message: str | Mapping) -> str | None:
        """One session state message (the JSON text or its mapping); the event it queued, if any."""
        if self._clips is not None:
            return self._clips_on_state(message)
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
        if self._clips is not None:
            self._clips_on_utterance(text)
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
        if self._clips is not None:
            self._clips_tick()
            return
        with self._lock:
            now = self._clock()
            flight = self._flight
            if flight is not None and now - flight.started_at > flight.timeout_s:
                reason = f"{flight.call}_timeout"
                self._playback_ended(flight)
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
        if self._clips is not None:
            self._clips_advance(now)
            return
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
            if self._clips is not None:
                self._speech_over(speech)
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
                self.synthesis_calls += 1

                def synthesized(ok: bool | None, message: str, segments: Sequence[Any] = ()) -> None:
                    self._answer(token, ok, message, segments)

                self._calls.synthesize(argument, synthesized)
            else:
                if self._on_playback_start is not None:
                    self._on_playback_start()
                self._calls.play(argument, lambda ok, message: self._answer(token, ok, message, ()))
        except Exception as exc:  # noqa: BLE001 - a failed call is a failed phrase
            self._answer(token, None, f"call failed: {exc}", ())

    def _answer(self, token: int, ok: bool | None, message: str, segments: Sequence[Any]) -> None:
        with self._lock:
            flight = self._flight
            if flight is None or flight.token != token:
                self.late_answers += 1  # an answer after its timeout, or a second answer
                return
            self._playback_ended(flight)
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
            if self._clips is not None:
                self._speech_over(speech)
        else:
            self._log("warning", f"precache of {text!r}: {reason}: {message}")

    def _playback_ended(self, flight: _Flight) -> None:
        """A playback call is over, whatever its answer (or none)."""
        if flight.call == "playback" and self._on_playback_end is not None:
            self._on_playback_end()

    # ------------------------------------------------------------------
    # prerecorded mode
    # ------------------------------------------------------------------

    def has_clip(self, text: str) -> bool:
        """Whether an utterance of ``text`` can be played (prerecorded mode)."""
        text = text.strip()
        with self._lock:
            return self._clips is not None and text in self._clips.paths and text not in self._gone

    def _clips_on_state(self, message: str | Mapping) -> str | None:
        with self._lock:
            now = self._clock()
            try:
                content = json.loads(message) if isinstance(message, str) else message
            except ValueError:
                content = None  # counted as malformed by the filter
            state = content.get("state") if isinstance(content, Mapping) else None
            if isinstance(state, str):
                if state != STATE_STANDBY and self._session_state == STATE_STANDBY:
                    self._standby_reset()
                self._session_state = state
                self._session_state_at = now
            event = self._filter.feed(content)
            queued = self._queue_event(event, now) if event is not None else False
            self._advance(now)
            return event if queued else None

    def _clips_on_utterance(self, text: str) -> None:
        with self._lock:
            self.utterances += 1
            if text not in self._clips.paths or text in self._gone:
                # The node refuses these at the door (noclip); never synthesized.
                self.utterances_dropped += 1
                self._log("info", f"{UTTERANCE}: dropped {text!r}: no clip")
                return
            dropped = self._queue.push(UTTERANCE, text)
            if dropped is not None:
                self._log("warning", f"queue full ({len(self._queue)}): dropped {dropped[0]} {dropped[1]!r}")
            self._log("info", f"{UTTERANCE}: queued {text!r}")
            self._advance(self._clock())

    def _clips_tick(self) -> None:
        with self._lock:
            now = self._clock()
            flight = self._flight
            if flight is not None and now - flight.started_at > flight.timeout_s:
                self._playback_ended(flight)
                self._fail(f"{flight.call}_timeout", flight.text, f"no answer within {flight.timeout_s:g} s")
            self._standby_step(now)
            self._advance(now)

    def _queue_event(self, event: str, now: float) -> bool:
        """Queue an event's clip ahead of what it outranks; False when it has none."""
        self._clear_queue()
        if event == "ready":
            self._standby_reset()
            self._standby_after_ready = True
        elif event == GOODBYE:
            # A stop during the approach says it while still in STANDBY.
            self._standby_reset()
        text = self._pick(event, self._clips.events.get(event, ()))
        if text is None:
            self.events_silent += 1
            self._log("warning", f"{event}: no usable clip; nothing is said")
            self._speech_over(_Speech(event, ""))
            return False
        dropped = self._queue.push(event, text)
        if dropped is not None:
            self._log("warning", f"queue full ({len(self._queue)}): dropped {dropped[0]} {dropped[1]!r}")
        self._log("info", f"{event}: queued {text!r}")
        return True

    def _clear_queue(self, kinds: tuple[str, ...] = (UTTERANCE, STANDBY)) -> None:
        """Take every queued item of ``kinds`` out, the rest keeping their order."""
        items = []
        while len(self._queue):
            items.append(self._queue.pop())
        for item in items:
            if item[0] in kinds:
                self.cleared += 1
                self._log("info", f"{item[0]}: cleared {item[1]!r}")
            else:
                self._queue.push(*item)

    def _pick(self, key: str, texts: Sequence[str]) -> str | None:
        """A random usable text of ``texts``, not the one picked last for ``key`` when there is a choice."""
        candidates = [text for text in texts if text not in self._gone]
        if len(candidates) > 1:
            candidates = [text for text in candidates if text != self._last_pick.get(key)]
        if not candidates:
            return None
        text = self._rng.choice(candidates)
        self._last_pick[key] = text
        return text

    def _playable(self, kind: str, text: str) -> str | None:
        """``text`` if its file is still there, else another clip of the same event or the standby words."""
        path = self._clips.paths.get(text)
        if path is not None and text not in self._gone:
            if self._files.exists(path):
                return text
            self._gone.add(text)
            self.clip_missing += 1
            self._log("warning", f"{kind}: the clip of {text!r} is gone ({path}); it is not said again")
        if kind in EVENTS:
            replacement = self._pick(kind, self._clips.events.get(kind, ()))
        elif kind == STANDBY:
            replacement = self._pick(STANDBY, self._clips.standby)
        else:
            replacement = None
        if replacement is None:
            if kind in EVENTS:
                self.events_silent += 1
            self._log("warning", f"{kind}: dropped {text!r}: no clip left to play")
            self._speech_over(_Speech(kind, text))
            return None
        return self._playable(kind, replacement)

    def _clips_advance(self, now: float) -> None:
        try:
            while self._flight is None:
                speech = self._speech
                if speech is not None:
                    if not self._play_next(speech, now):
                        return
                    self._report_speaking()
                    continue
                item = self._queue.pop()
                if item is None:
                    return
                kind, text = item
                if kind == UTTERANCE and self._utterance_gate is not None and not self._utterance_gate():
                    self.utterances_dropped += 1
                    self._log("info", f"{UTTERANCE}: dropped {text!r}: the mirror left MIRRORING while it waited")
                    continue
                if kind == STANDBY and not self._in_standby(now):
                    self.standby_dropped += 1
                    self._log("info", f"{STANDBY}: dropped {text!r}: the session left STANDBY while it waited")
                    continue
                text = self._playable(kind, text)
                if text is None:
                    continue
                self._speech = _Speech(kind, text, (_Clip(self._clips.paths[text], 0.0),))
                self._report_speaking()
        finally:
            self._report_speaking()

    def _report_speaking(self) -> None:
        current = self._speech.text if self._speech is not None else None
        if current != self._speaking:
            self._speaking = current
            if self._on_speaking is not None:
                self._on_speaking(current)

    # standby words ------------------------------------------------------

    def _in_standby(self, now: float) -> bool:
        return (
            self._session_state == STATE_STANDBY
            and self._session_state_at is not None
            and now - self._session_state_at <= STATE_STALE_SEC
        )

    def _speech_over(self, speech: _Speech) -> None:
        """A clip ended (said, failed or dropped): when the next standby word is due."""
        if self._session_state != STATE_STANDBY:
            return
        if speech.event == "ready" and self._standby_after_ready:
            self._standby_after_ready = False
            self._standby_arm(self._clips.standby_first_after_s)
        elif speech.event == STANDBY and not self._standby_after_ready:
            self._standby_arm(self._clips.standby_gap_s)

    def _standby_arm(self, interval: tuple[float, float]) -> None:
        if not self._clips.standby:
            return
        self._standby_due = self._clock() + self._rng.uniform(*interval)

    def _standby_step(self, now: float) -> None:
        """Queue a standby word once it is due and nothing else is being said or waits."""
        due = self._standby_due
        if due is None or now < due or not self._in_standby(now):
            return
        if self._speech is not None or self._flight is not None or len(self._queue):
            return
        self._standby_due = None  # until this word ends
        text = self._pick(STANDBY, self._clips.standby)
        if text is None:
            self._log("warning", f"{STANDBY}: no usable word left")
            return
        self.standby_queued += 1
        self._queue.push(STANDBY, text)
        self._log("info", f"{STANDBY}: queued {text!r}")

    def _standby_reset(self) -> None:
        """Leaving STANDBY, or a new ready: no timer, no queued word."""
        self._standby_due = None
        self._standby_after_ready = False
        self._clear_queue((STANDBY,))

    # ------------------------------------------------------------------
    # ~/status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = {
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
            if self._clips is not None:
                due = self._standby_due
                status.update(
                    {
                        "mode": "clips",
                        "synthesis_calls": self.synthesis_calls,
                        "clips_usable": len(self._clips.paths) - len(self._gone),
                        "clip_missing": self.clip_missing,
                        "events_silent": self.events_silent,
                        "cleared": self.cleared,
                        "standby_queued": self.standby_queued,
                        "standby_dropped": self.standby_dropped,
                        "standby_due_in": None if due is None else round(due - self._clock(), 2),
                        "session_state": self._session_state,
                    }
                )
            return status
