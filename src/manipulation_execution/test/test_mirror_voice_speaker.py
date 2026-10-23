"""The speaker's on-demand utterances: synthesized, played, cleaned up.

The state phrases were the speaker's only input until the visual interaction
feature arrived. These tests pin the three differences an utterance makes —
never cached, re-checked at the pop, its WAV files removed once spoken — and
the one thing that must not change: phrases and utterances share one queue,
oldest first.

The fake calls answer synchronously, so one ``on_utterance`` runs the whole
chain and then precaches every phrase behind it; the assertions below are
scoped to the utterance's own texts and file names for exactly that reason.
"""

from __future__ import annotations

from typing import NamedTuple

from embodied_common.mirror_voice_contract import parse_voice_section
from manipulation_execution.mirror_voice.core import STANDBY_APPROACH, WAIT_NONE
from manipulation_execution.mirror_voice.speaker import Speaker, wav_name

AUDIO_FORMAT = "wav_pcm_s16le"
UTTERANCE = "这个玩偶真有趣"
# Passes the speaker's check: RIFF at 0, WAVE inside the first 16 bytes.
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x00\x00\x00\x00"


class Segment(NamedTuple):
    index: int
    audio_format: str
    audio_data: bytes
    pause_after_ms: int = 0


class Calls:
    """``SpeakerCalls`` with synchronous answers and a record of every call.

    ``bad_segments_for`` names texts the service answers with a broken segment
    list, so one bad utterance does not also break the phrase precache that
    runs behind it under these synchronous answers.
    """

    def __init__(
        self,
        *,
        segments=(Segment(0, AUDIO_FORMAT, WAV),),
        bad_segments=(),
        bad_segments_for=(),
        tts_ready=True,
        playback_ready=True,
    ):
        self.segments = tuple(segments)
        self.bad_segments = tuple(bad_segments)
        self.bad_segments_for = set(bad_segments_for)
        self.tts_ready_flag = tts_ready
        self.playback_ready_flag = playback_ready
        self.play_ok = True
        self.synthesized: list[str] = []
        self.played: list[str] = []

    def tts_ready(self) -> bool:
        return self.tts_ready_flag

    def playback_ready(self) -> bool:
        return self.playback_ready_flag

    def synthesize(self, text, done) -> None:
        self.synthesized.append(text)
        segments = self.bad_segments if text in self.bad_segments_for else self.segments
        if segments:
            done(True, "", list(segments))
        else:
            done(False, "no voice", [])

    def play(self, path, done) -> None:
        self.played.append(path)
        done(self.play_ok, "")


class Files:
    """``SpeakerFiles`` in memory, remembering what is still on disk."""

    def __init__(self):
        self.written: dict[str, bytes] = {}

    def write(self, name, data):
        self.written[name] = data
        return f"/wav/{name}"

    def exists(self, path):
        return path.rsplit("/", 1)[-1] in self.written

    def remove_file(self, path):
        self.written.pop(path.rsplit("/", 1)[-1], None)


def speaker(calls, files, *, gate=None, queue_max=8):
    return Speaker(
        parse_voice_section({"enabled": True, "queue_max": queue_max}),
        calls,
        files,
        tts_timeout_s=5.0,
        playback_timeout_s=5.0,
        clock=lambda: 100.0,
        log=lambda level, message: None,
        utterance_gate=gate,
    )


def starting_transition(t=1.0):
    """Two state messages whose transition is spoken as the ``starting`` event."""
    first = {"state": "WAIT", "last_transition": None}
    second = {"state": "STANDBY", "last_transition": {"from": WAIT_NONE, "to": STANDBY_APPROACH, "t": t}}
    return first, second


class TestOnDemandUtterance:
    def test_it_is_synthesized_played_and_cleaned_up(self):
        calls = Calls()
        files = Files()
        sess = speaker(calls, files)

        sess.on_utterance(UTTERANCE)

        assert calls.synthesized[0] == UTTERANCE
        assert len(calls.played) == 1
        status = sess.status()
        assert status["utterances"] == 1
        assert status["utterances_dropped"] == 0
        assert status["spoken"] == 1
        # The board's temporary directory is tmpfs: an utterance's files go the
        # moment it finishes, or they add up to a memory leak.
        assert wav_name(UTTERANCE, 0) not in files.written

    def test_it_is_never_cached(self):
        """A phrase's WAV stays after speaking — it IS the cache; an utterance's
        does not, because each is spoken at most once and a cache entry for it
        could never be hit, only grow."""
        calls = Calls()
        files = Files()
        sess = speaker(calls, files)

        sess.on_utterance(UTTERANCE)

        assert wav_name(UTTERANCE, 0) not in files.written
        # The phrases precached behind it keep theirs.
        assert files.written

    def test_a_blank_one_is_ignored_without_a_count(self):
        sess = speaker(Calls(), Files())

        sess.on_utterance("   ")

        assert sess.status()["utterances"] == 0

    def test_a_closed_gate_drops_it_at_the_pop(self):
        """The state it was admitted under may be gone by the time it would
        start synthesizing; the gate is asked again at that moment."""
        calls = Calls()
        sess = speaker(calls, Files(), gate=lambda: False)

        sess.on_utterance(UTTERANCE)

        assert UTTERANCE not in calls.synthesized
        assert sess.status()["utterances_dropped"] == 1

    def test_an_open_gate_lets_it_through(self):
        calls = Calls()
        sess = speaker(calls, Files(), gate=lambda: True)

        sess.on_utterance(UTTERANCE)

        assert calls.synthesized[0] == UTTERANCE
        assert sess.status()["utterances_dropped"] == 0

    def test_the_gate_is_never_asked_for_a_phrase(self):
        """A closed mirror gates utterances only; the session's own phrases are
        spoken whatever the mirror is doing — they ARE the mirror."""
        calls = Calls()
        sess = speaker(calls, Files(), gate=lambda: False)

        first, second = starting_transition()
        sess.on_state(first)
        event = sess.on_state(second)

        assert event == "starting"
        assert "启动中" in calls.synthesized
        assert sess.status()["utterances_dropped"] == 0


class TestTheSharedQueue:
    def test_phrases_and_utterances_keep_their_order(self):
        """One queue, oldest first: the order is part of what is said.

        The synthesis service starts absent so both inputs can queue before
        either is spoken (waiting for playback would not hold the phrase back:
        precache runs while a speech waits for the playback service); one tick
        then speaks them in arrival order.
        """
        calls = Calls(tts_ready=False)
        sess = speaker(calls, Files())

        first, second = starting_transition()
        sess.on_state(first)
        sess.on_state(second)
        sess.on_utterance(UTTERANCE)

        calls.tts_ready_flag = True
        sess.tick()

        assert calls.synthesized[:2] == ["启动中", UTTERANCE]

    def test_an_utterance_waits_for_the_service_at_the_head(self):
        """Never cached, so a missing synthesis service is its only way
        forward. It waits rather than letting the phrases behind it jump."""
        calls = Calls(tts_ready=False)
        sess = speaker(calls, Files())

        sess.on_utterance(UTTERANCE)
        first, second = starting_transition()
        sess.on_state(first)
        sess.on_state(second)

        assert calls.synthesized == []
        assert sess.status()["queue"] == 2

        calls.tts_ready_flag = True
        sess.tick()

        assert calls.synthesized[:2] == [UTTERANCE, "启动中"]

    def test_a_full_queue_still_drops_the_oldest_non_goodbye(self):
        """The queue's own drop rule does not change: a push into a full queue
        drops the oldest non-goodbye, an utterance included."""
        calls = Calls(tts_ready=False)
        sess = speaker(calls, Files(), queue_max=1)

        sess.on_utterance("第一句")
        sess.on_utterance("第二句")

        assert sess.status()["queue"] == 1
        assert sess.status()["dropped_queue_full"] == 1


class TestCleanupOnFailure:
    def test_bad_audio_removes_the_files_written_so_far(self):
        """A half-written speech is a dropped one; nothing else would ever
        remove the files the good segments already wrote. The broken segments
        are served to the utterance only: the phrase precache that runs behind
        it under synchronous answers must keep succeeding."""
        calls = Calls(
            bad_segments=[Segment(0, AUDIO_FORMAT, WAV), Segment(1, "mp3", b"xx")],
            bad_segments_for={UTTERANCE},
        )
        files = Files()
        sess = speaker(calls, files)

        sess.on_utterance(UTTERANCE)

        assert sess.status()["failed"]["audio"] == 1
        assert wav_name(UTTERANCE, 0) not in files.written

    def test_a_failed_playback_removes_the_files(self):
        calls = Calls()
        calls.play_ok = False
        files = Files()
        sess = speaker(calls, files)

        sess.on_utterance(UTTERANCE)

        assert sess.status()["failed"]["playback"] == 1
        assert wav_name(UTTERANCE, 0) not in files.written


class TestPrecacheUnaffected:
    def test_phrases_are_still_precached_after_an_utterance(self):
        """The fixed phrases keep their cache; only the utterance path changed."""
        calls = Calls()
        sess = speaker(calls, Files())

        sess.on_utterance(UTTERANCE)

        assert "启动中" in calls.synthesized
        assert sess.status()["cached"] >= 1
