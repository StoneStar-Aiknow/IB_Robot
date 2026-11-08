"""The speaker's prerecorded mode (feat/x2-internvl-voice): clips only, never synthesis.

Pinned here, all without ROS:

* nothing reaches the synthesis service — the fake one raises on any use;
* a clip whose WAV is missing (at startup or gone later) is the only one
  silent: the event says another clip of its pool;
* an event clears the queued look lines and standby words, never the clip
  playing; at the pop, a look line needs MIRRORING and a standby word STANDBY;
* the goodbye as a stop's wrap-up begins, and not again at WAIT;
* the standby rhythm, on a fake clock with a fixed seed;
* a broken ``clips_json`` still plays the clips it names, with defaults for
  the rest;
* the clips found on disk: first directory with any, 24 kHz mono 16 bit only.
"""

from __future__ import annotations

import json
import random
import wave

import pytest

from embodied_common.internvl_voice_contract import parse_playback_json
from embodied_common.mirror_session_contract import STATE_STALE_SEC
from embodied_common.mirror_voice_contract import parse_voice_section
from manipulation_execution.mirror_voice.core import (
    GOODBYE,
    MIRRORING_RUNNING,
    STANDBY_APPROACH,
    STANDBY_READY,
    WAIT_NONE,
    TransitionFilter,
)
from manipulation_execution.mirror_voice.speaker import (
    STANDBY,
    PrerecordedClips,
    Speaker,
    check_wav,
    load_clips,
)

EVENT_TEXTS = {
    "starting": ("启动中，稍等一下。",),
    "ready": ("我准备好咯！", "热身完毕，随时可以来玩。"),
    "mirroring": ("开始模仿咯！", "看好咯，我要跟着你动啦。", "来吧，你动我就动。"),
    "goodbye": ("下次再见吧！", "谢谢你陪我玩，下次再见吧。"),
}
STANDBY_TEXTS = ("你好呀！", "来咯！", "嘿呀！", "来玩呀！", "看这里！")
LOOK_TEXTS = ("收到你的招呼啦！", "这节奏感，可以啊！")


class NoSynthesis:
    """``SpeakerCalls`` whose synthesis service must never be touched.

    ``play`` answers at once (``sync``) or waits for ``finish``.
    """

    def __init__(self, *, sync=True):
        self.sync = sync
        self.synthesized: list[str] = []
        self.tts_looked = 0
        self.played: list[str] = []
        self.pending = None

    def tts_ready(self):
        self.tts_looked += 1
        raise AssertionError("prerecorded mode looked at the synthesis service")

    def synthesize(self, text, done):
        self.synthesized.append(text)
        raise AssertionError(f"prerecorded mode synthesized {text!r}")

    def playback_ready(self):
        return True

    def play(self, path, done):
        self.played.append(path)
        if self.sync:
            done(True, "")
        else:
            self.pending = done

    def finish(self, ok=True):
        done, self.pending = self.pending, None
        done(ok, "")


class Disk:
    """``SpeakerFiles`` over a set of paths that exist."""

    def __init__(self, paths):
        self.existing = set(paths)

    def write(self, name, data):
        raise AssertionError("prerecorded mode wrote a file")

    def exists(self, path):
        return path in self.existing

    def remove_file(self, path):
        raise AssertionError("prerecorded mode removed a deployed clip")


def path_of(text):
    return f"/clips/{text}.wav"


def clips(*, events=EVENT_TEXTS, standby=STANDBY_TEXTS, look=LOOK_TEXTS, first=(1.0, 3.0), gap=(2.0, 5.0)):
    texts = [text for pool in events.values() for text in pool] + list(standby) + list(look)
    return PrerecordedClips(
        paths={text: path_of(text) for text in texts},
        events=dict(events),
        standby=tuple(standby),
        standby_first_after_s=first,
        standby_gap_s=gap,
    )


class Harness:
    """A speaker in prerecorded mode on a fake clock, and the session messages it reads."""

    def __init__(self, *, sync=True, prerecorded=None, missing=(), seed=7, gate=None, queue_max=8):
        self.t = 100.0
        self.calls = NoSynthesis(sync=sync)
        self.clips = prerecorded or clips()
        self.disk = Disk(path for path in self.clips.paths.values() if path not in missing)
        self.log: list[tuple[str, str]] = []
        self.events: list[str] = []  # playback start / end, speaking changes
        self.state = "WAIT"
        self.purpose = ""
        self.transition = None
        self.last_beat = self.t
        self.speaker = Speaker(
            parse_voice_section({"enabled": True, "queue_max": queue_max}),
            self.calls,
            self.disk,
            tts_timeout_s=15.0,
            playback_timeout_s=10.0,
            clock=lambda: self.t,
            log=lambda level, message: self.log.append((level, message)),
            utterance_gate=gate,
            clips=self.clips,
            on_playback_start=lambda: self.events.append("start"),
            on_playback_end=lambda: self.events.append("end"),
            on_speaking=lambda text: self.events.append(f"speaking {text}"),
            rng=random.Random(seed),
        )
        self.speaker.on_state(json.dumps({"state": "WAIT", "last_transition": None}))  # the replay

    def go(self, target, source=None, purpose=""):
        """Send the transition ``source -> target`` (``STATE/phase``) once; ``purpose`` its wrap_purpose."""
        source = source or (self.transition["to"] if self.transition else WAIT_NONE)
        self.transition = {"from": source, "to": target, "t": self.t}
        self.state = target.split("/")[0]
        self.purpose = purpose
        self.last_beat = self.t
        return self.speaker.on_state(json.dumps(self.message()))

    def message(self):
        return {"state": self.state, "wrap_purpose": self.purpose, "last_transition": self.transition}

    def heartbeat(self):
        self.last_beat = self.t
        self.speaker.on_state(json.dumps(self.message()))

    def run_until(self, end, step=0.1):
        """Tick every ``step`` s, a heartbeat every second, up to ``end``."""
        while self.t < end - 1e-9:
            self.t = round(self.t + step, 6)
            if self.t - self.last_beat >= 1.0 - 1e-9:
                self.heartbeat()
            self.speaker.tick()

    def played_texts(self):
        lookup = {path: text for text, path in self.clips.paths.items()}
        return [lookup[path] for path in self.calls.played]


def assert_never_synthesized(harness):
    assert harness.calls.synthesized == []
    assert harness.calls.tts_looked == 0
    assert harness.speaker.synthesis_calls == 0
    assert harness.speaker.status()["synthesis_calls"] == 0


class TestNoSynthesis:
    def test_a_whole_round_plays_clips_and_never_touches_the_synthesis_service(self):
        h = Harness()

        h.go(STANDBY_APPROACH)
        h.go(STANDBY_READY)
        h.run_until(110.0)
        h.go(MIRRORING_RUNNING)
        h.speaker.on_utterance(LOOK_TEXTS[0])
        h.go(WAIT_NONE)
        h.run_until(130.0)

        assert_never_synthesized(h)
        played = h.played_texts()
        assert played[0] in EVENT_TEXTS["starting"]
        assert played[1] in EVENT_TEXTS["ready"]
        assert LOOK_TEXTS[0] in played
        assert played[-1] in EVENT_TEXTS["goodbye"]
        assert h.speaker.status()["failed"] == dict.fromkeys(h.speaker.failed, 0)

    def test_an_utterance_without_a_clip_is_dropped_not_synthesized(self):
        h = Harness()

        assert h.speaker.has_clip("一句没有录过的话") is False
        h.speaker.on_utterance("一句没有录过的话")

        assert_never_synthesized(h)
        assert h.calls.played == []
        assert h.speaker.status()["utterances_dropped"] == 1

    def test_a_clip_gone_at_its_turn_is_not_synthesized_either(self):
        h = Harness()
        h.disk.existing.discard(path_of(LOOK_TEXTS[0]))

        h.speaker.on_utterance(LOOK_TEXTS[0])

        assert_never_synthesized(h)
        assert h.calls.played == []
        assert h.speaker.status()["clip_missing"] == 1
        assert h.speaker.has_clip(LOOK_TEXTS[0]) is False

    def test_the_status_has_no_synthesis_state(self):
        h = Harness()
        h.run_until(101.0)

        status = h.speaker.status()
        assert status["mode"] == "clips"
        assert status["tts_available"] is None
        assert status["precache_pending"] == 0


class TestEventClips:
    def test_an_event_never_repeats_its_last_clip(self):
        h = Harness()
        picks = []
        for _ in range(12):
            h.go(MIRRORING_RUNNING, source=STANDBY_READY)
            picks.append(h.played_texts()[-1])
            h.go(STANDBY_READY, source=MIRRORING_RUNNING)
            h.t += 1.0

        assert set(picks) <= set(EVENT_TEXTS["mirroring"])
        assert all(a != b for a, b in zip(picks, picks[1:], strict=False)), picks
        assert len(set(picks)) > 1

    def test_a_missing_wav_silences_only_that_clip(self):
        missing = EVENT_TEXTS["mirroring"][0]
        h = Harness(missing={path_of(missing)})
        picks = []
        for _ in range(8):
            h.go(MIRRORING_RUNNING, source=STANDBY_READY)
            picks.append(h.played_texts()[-1])
            h.go(STANDBY_READY, source=MIRRORING_RUNNING)
            h.t += 1.0

        # Every mirroring round still said something, never the missing clip.
        assert len(picks) == 8
        assert missing not in picks
        assert set(picks) <= set(EVENT_TEXTS["mirroring"][1:])
        assert h.speaker.status()["clip_missing"] == 1  # counted once, then never tried again
        assert_never_synthesized(h)

    def test_an_event_with_one_clip_left_still_speaks(self):
        h = Harness(missing={path_of(EVENT_TEXTS["goodbye"][0])})

        h.go(STANDBY_APPROACH)
        h.go(WAIT_NONE)
        h.go(STANDBY_APPROACH, source=WAIT_NONE)
        h.t += 1
        h.go(WAIT_NONE)

        goodbyes = [text for text in h.played_texts() if text in EVENT_TEXTS["goodbye"]]
        assert goodbyes == [EVENT_TEXTS["goodbye"][1]] * 2

    def test_an_event_with_no_clip_is_silent_and_counted(self):
        events = dict(EVENT_TEXTS, starting=())
        h = Harness(prerecorded=clips(events=events))

        assert h.go(STANDBY_APPROACH) is None

        assert h.calls.played == []
        assert h.speaker.status()["events_silent"] == 1
        assert_never_synthesized(h)


MIRRORING_WRAP_UP = "MIRRORING/wrap_up"
STANDBY_WRAP_UP = "STANDBY/wrap_up"


def transition(source, target, t, purpose=""):
    return {
        "state": target.split("/")[0],
        "wrap_purpose": purpose,
        "last_transition": {"from": source, "to": target, "t": t},
    }


class TestEarlyGoodbye:
    def test_a_stops_wrap_up_says_the_goodbye_and_wait_stays_silent(self):
        h = Harness()
        h.go(STANDBY_APPROACH)
        h.go(STANDBY_READY)
        h.go(MIRRORING_RUNNING)
        h.t += 1

        assert h.go(MIRRORING_WRAP_UP, purpose="stop") == GOODBYE
        assert h.played_texts()[-1] in EVENT_TEXTS["goodbye"]
        h.t += 2
        assert h.go(WAIT_NONE) is None
        h.run_until(h.t + 5)

        goodbyes = [text for text in h.played_texts() if text in EVENT_TEXTS["goodbye"]]
        assert len(goodbyes) == 1

    def test_a_stop_during_the_approach_says_it_at_its_wrap_up(self):
        h = Harness()
        h.go(STANDBY_APPROACH)

        assert h.go(STANDBY_WRAP_UP, purpose="stop") == GOODBYE
        assert h.go(WAIT_NONE) is None
        assert h.speaker.status()["standby_due_in"] is None

    def test_a_cancels_wrap_up_is_silent_and_so_is_its_cooldown(self):
        h = Harness()
        h.go(MIRRORING_RUNNING, source=STANDBY_READY)
        before = len(h.calls.played)

        assert h.go(MIRRORING_WRAP_UP, purpose="cancel") is None
        assert h.go("STANDBY/cooldown") is None
        assert len(h.calls.played) == before

    def test_a_stop_during_a_cancels_wrap_up_says_it_at_wait(self):
        h = Harness()
        h.go(MIRRORING_RUNNING, source=STANDBY_READY)
        h.go(MIRRORING_WRAP_UP, purpose="cancel")
        h.purpose = "stop"  # no transition: the cancel's wrap-up goes on as the stop's
        h.heartbeat()

        assert h.go(WAIT_NONE) == GOODBYE

    def test_a_wait_without_a_wrap_up_still_says_it(self):
        h = Harness()
        h.go(STANDBY_APPROACH)
        h.go(STANDBY_READY)

        assert h.go(WAIT_NONE) == GOODBYE

    def test_the_next_visit_says_its_goodbye_again(self):
        h = Harness()
        for _ in range(2):
            h.go(STANDBY_APPROACH, source=WAIT_NONE)
            h.go(STANDBY_READY)
            h.go(MIRRORING_RUNNING)
            h.t += 1
            assert h.go(MIRRORING_WRAP_UP, purpose="stop") == GOODBYE
            assert h.go(WAIT_NONE) is None
            h.t += 1

    def test_a_queued_look_line_is_cleared_and_the_one_playing_finishes_first(self):
        h = Harness(sync=False, gate=lambda: True)
        h.speaker.on_utterance(LOOK_TEXTS[0])  # starts playing
        h.speaker.on_utterance(LOOK_TEXTS[1])  # waits

        h.go(MIRRORING_WRAP_UP, source=MIRRORING_RUNNING, purpose="stop")
        h.calls.finish()

        played = h.played_texts()
        assert played == [LOOK_TEXTS[0], played[1]]
        assert played[1] in EVENT_TEXTS["goodbye"]

    def test_the_filter_says_it_early_only_when_asked(self):
        # The synthesis mode (switch off) keeps the goodbye at WAIT, the wrap-up silent.
        for early, at_wrap_up, at_wait in ((False, None, GOODBYE), (True, GOODBYE, None)):
            words = TransitionFilter(early_goodbye=early)
            words.feed({"last_transition": None})  # the replay
            assert words.feed(transition(MIRRORING_RUNNING, MIRRORING_WRAP_UP, 1.0, "stop")) == at_wrap_up
            assert words.feed(transition(MIRRORING_WRAP_UP, WAIT_NONE, 2.0)) == at_wait

    def test_a_replayed_wrap_up_is_not_spoken_and_wait_then_says_it(self):
        words = TransitionFilter(early_goodbye=True)

        assert words.feed(transition(MIRRORING_RUNNING, MIRRORING_WRAP_UP, 1.0, "stop")) is None
        assert words.feed(transition(MIRRORING_WRAP_UP, WAIT_NONE, 2.0)) == GOODBYE


class TestQueue:
    def test_an_event_clears_queued_look_lines_but_not_the_clip_playing(self):
        h = Harness(sync=False, gate=lambda: True)
        h.speaker.on_utterance(LOOK_TEXTS[0])  # starts playing
        h.speaker.on_utterance(LOOK_TEXTS[1])  # waits
        assert h.speaker.status()["queue"] == 1

        h.go(WAIT_NONE, source=MIRRORING_RUNNING)
        h.calls.finish()
        h.calls.finish()

        played = h.played_texts()
        assert played[0] == LOOK_TEXTS[0]
        assert played[1] in EVENT_TEXTS["goodbye"]
        assert LOOK_TEXTS[1] not in played
        assert h.speaker.status()["cleared"] == 1

    def test_a_look_line_is_dropped_at_the_pop_once_mirroring_ended(self):
        mirroring = {"now": True}
        h = Harness(sync=False, gate=lambda: mirroring["now"])
        h.speaker.on_utterance(LOOK_TEXTS[0])
        h.speaker.on_utterance(LOOK_TEXTS[1])

        mirroring["now"] = False
        h.calls.finish()

        assert h.played_texts() == [LOOK_TEXTS[0]]
        assert h.speaker.status()["utterances_dropped"] == 1

    def test_a_standby_word_is_dropped_at_the_pop_once_standby_ended(self):
        h = Harness(sync=False)
        h.go(STANDBY_APPROACH)
        h.calls.finish()
        h.go(STANDBY_READY)
        h.calls.finish()  # the ready clip ends: the first word is due in 1-3 s
        h.run_until(h.t + 3.1)
        assert h.speaker.status()["standby_queued"] == 1
        word = h.played_texts()[-1]
        assert word in STANDBY_TEXTS

        # Queue a second word behind a slow one by hand, then leave STANDBY.
        h.speaker._queue.push(STANDBY, STANDBY_TEXTS[0])
        h.state = "MIRRORING"
        h.heartbeat()  # leaving STANDBY clears queued words at once
        h.calls.finish()

        assert h.speaker.status()["queue"] == 0
        assert h.played_texts()[-1] == word

    def test_playback_is_bracketed_by_the_callbacks(self):
        h = Harness(sync=False, gate=lambda: True)
        h.speaker.on_utterance(LOOK_TEXTS[0])
        assert h.events == [f"speaking {LOOK_TEXTS[0]}", "start"]

        h.calls.finish()
        assert h.events[-2:] == ["end", "speaking None"]

    def test_a_failed_playback_still_ends_the_bracket(self):
        h = Harness(sync=False, gate=lambda: True)
        h.speaker.on_utterance(LOOK_TEXTS[0])
        h.calls.finish(ok=False)

        assert h.events.count("start") == h.events.count("end") == 1

    def test_a_timed_out_playback_still_ends_the_bracket(self):
        h = Harness(sync=False, gate=lambda: True)
        h.speaker.on_utterance(LOOK_TEXTS[0])
        h.t += 11.0
        h.speaker.tick()

        assert h.events.count("start") == h.events.count("end") == 1
        assert h.speaker.status()["failed"]["playback_timeout"] == 1


class TestStandbyRhythm:
    def schedule(self, seed, *, leave_at=None):
        """Starts of every clip in one STANDBY: (time, text), the ready clip 2 s long, words 1 s."""
        h = Harness(sync=False, seed=seed)
        h.go(STANDBY_APPROACH)
        h.calls.finish()
        h.go(STANDBY_READY)
        ready_started = h.t
        starts = [(h.t, h.played_texts()[-1])]
        ends = []
        playing_until = h.t + 2.0
        end = h.t + 40.0
        while h.t < end:
            if leave_at is not None and h.t >= leave_at and h.state == "STANDBY":
                h.go(MIRRORING_RUNNING, source=STANDBY_READY)
                h.calls.finish()  # the mirroring clip
                playing_until = None
            if h.calls.pending is not None and playing_until is not None and h.t >= playing_until - 1e-9:
                h.calls.finish()
                ends.append(h.t)
                playing_until = None
            before = len(h.calls.played)
            h.run_until(round(h.t + 0.1, 6))
            if len(h.calls.played) > before:
                starts.append((h.t, h.played_texts()[-1]))
                playing_until = h.t + 1.0
        return h, ready_started, starts, ends

    def test_first_word_one_to_three_seconds_after_ready_then_two_to_five_after_each(self):
        h, ready_started, starts, ends = self.schedule(seed=3)

        words = starts[1:]
        assert len(words) >= 6
        assert all(text in STANDBY_TEXTS for _, text in words)
        # ends[0] is the ready clip's; ends[i] the end of word i.
        first_gap = words[0][0] - ends[0]
        assert 1.0 - 0.11 <= first_gap <= 3.0 + 0.11, first_gap
        for (start, _), previous_end in zip(words[1:], ends[1:], strict=False):
            assert 2.0 - 0.11 <= start - previous_end <= 5.0 + 0.11
        texts = [text for _, text in words]
        assert all(a != b for a, b in zip(texts, texts[1:], strict=False)), texts
        assert_never_synthesized(h)

    def test_a_fixed_seed_gives_the_same_rhythm(self):
        assert self.schedule(seed=11)[2] == self.schedule(seed=11)[2]

    def test_leaving_standby_stops_the_words_and_resets_the_timer(self):
        ready_started = self.schedule(seed=5)[1]
        h2, ready2, starts2, _ = self.schedule(seed=5, leave_at=ready_started + 8.0)

        after = [text for t, text in starts2 if t > ready2 + 8.0 and text in STANDBY_TEXTS]
        assert after == []
        assert h2.speaker.status()["standby_due_in"] is None

    def test_no_word_before_ready(self):
        h = Harness(sync=True)
        h.go(STANDBY_APPROACH)
        h.run_until(h.t + 20.0)

        assert [text for text in h.played_texts() if text in STANDBY_TEXTS] == []

    def test_a_ready_without_a_clip_times_the_first_word_from_the_event(self):
        events = dict(EVENT_TEXTS, ready=())
        h = Harness(sync=True, prerecorded=clips(events=events))
        h.go(STANDBY_APPROACH)
        h.go(STANDBY_READY)
        event_at = h.t
        h.run_until(event_at + 3.2)

        played = h.played_texts()
        assert played[-1] in STANDBY_TEXTS
        assert h.speaker.status()["standby_queued"] == 1

    def test_a_stale_session_state_holds_the_word(self):
        h = Harness(sync=True)
        h.go(STANDBY_APPROACH)
        h.go(STANDBY_READY)
        fresh_until = h.t + STATE_STALE_SEC
        # No heartbeat for 20 s: past STATE_STALE_SEC the STANDBY may be over.
        late = []
        for _ in range(200):
            h.t = round(h.t + 0.1, 6)
            before = len(h.calls.played)
            h.speaker.tick()
            if len(h.calls.played) > before and h.t > fresh_until + 1e-9:
                late.append(h.t)

        assert late == []
        h.heartbeat()  # the session speaks again: the word that was due goes out
        h.speaker.tick()
        assert h.played_texts()[-1] in STANDBY_TEXTS


class TestBrokenConfig:
    def test_bad_fields_take_their_defaults_and_the_clips_still_play(self, tmp_path):
        folder = tmp_path / "clips"
        folder.mkdir()
        for clip_id in ("evt_starting_01", "evt_ready_01", "evt_mirroring_01", "evt_goodbye_01", "standby_any_01"):
            write_wav(folder / f"{clip_id}_v1.wav")
        text = json.dumps(
            {
                "version": "one",
                "clip_dirs": [str(folder)],
                "wav_suffix": "_v1.wav",
                "clips": {
                    "evt_starting_01": "启动中。",
                    "evt_ready_01": "准备好咯！",
                    "evt_mirroring_01": "开始模仿咯！",
                    "evt_goodbye_01": "下次再见吧！",
                    "standby_any_01": "你好呀！",
                    "BAD ID": "坏条目",
                },
                "events": {
                    "starting": ["evt_starting_01"],
                    "ready": ["evt_ready_01", "nope_nope_01"],
                    "mirroring": ["evt_mirroring_01"],
                    "goodbye": ["evt_goodbye_01"],
                },
                "standby": {"ids": ["standby_any_01"], "first_after_sec": "soon", "gap_sec": [5.0, 1.0]},
                "asr": {"mode": "loud", "resume_delay_sec": -1, "topic": "voice_control"},
                "surprise": 1,
            },
            ensure_ascii=False,
        )
        playback, warnings = parse_playback_json(text)
        assert warnings  # every bad field said so

        prerecorded, report = load_clips(playback)

        assert report.directory == str(folder)
        assert (report.found, report.total) == (5, 5)
        assert prerecorded.standby_first_after_s == (1.0, 3.0)
        assert prerecorded.standby_gap_s == (2.0, 5.0)
        assert playback["asr"] == {"mode": "mute", "resume_delay_sec": 0.8, "topic": "/voice_control"}

        calls = NoSynthesis()
        speaker = Speaker(
            parse_voice_section({"enabled": True}),
            calls,
            Disk(prerecorded.paths.values()),
            tts_timeout_s=15.0,
            playback_timeout_s=10.0,
            clock=lambda: 0.0,
            clips=prerecorded,
        )
        speaker.on_state(json.dumps({"state": "WAIT", "last_transition": None}))
        speaker.on_state(
            json.dumps({"state": "STANDBY", "last_transition": {"from": WAIT_NONE, "to": STANDBY_APPROACH, "t": 1}})
        )
        assert calls.played == [str(folder / "evt_starting_01_v1.wav")]
        assert calls.synthesized == []

    def test_unreadable_clips_json_is_a_silent_voice_not_synthesis(self):
        playback, warnings = parse_playback_json("not json")
        prerecorded, report = load_clips(playback, isfile=lambda path: False)

        assert warnings
        assert report.directory is None and report.found == 0
        assert set(report.silent_events) == set(EVENT_TEXTS)
        h = Harness(prerecorded=prerecorded)
        h.go(STANDBY_APPROACH)
        assert h.calls.played == []
        assert_never_synthesized(h)


# ----------------------------------------------------------------------
# the clips on disk


def write_wav(path, *, rate=24000, channels=1, width=2, frames=240):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(b"\x00" * frames * channels * width)


def playback(dirs, clip_ids, events=None, standby=()):
    return {
        "version": 1,
        "clip_dirs": [str(d) for d in dirs],
        "wav_suffix": "_v1.wav",
        "clips": {clip_id: f"第{index}句。" for index, clip_id in enumerate(clip_ids)},
        "events": events or {},
        "standby": {"ids": list(standby), "first_after_sec": (1.0, 3.0), "gap_sec": (2.0, 5.0)},
        "asr": {"mode": "mute", "resume_delay_sec": 0.8, "topic": "/voice_control"},
    }


class TestLoadClips:
    def test_the_first_directory_with_any_wav_is_used(self, tmp_path):
        empty, used, later = (tmp_path / name for name in ("empty", "used", "later"))
        for folder in (empty, used, later):
            folder.mkdir()
        write_wav(used / "evt_ready_01_v1.wav")
        write_wav(later / "evt_ready_01_v1.wav")
        write_wav(later / "evt_ready_02_v1.wav")

        loaded, report = load_clips(
            playback(
                [tmp_path / "absent", empty, used, later],
                ["evt_ready_01", "evt_ready_02"],
                events={"ready": ["evt_ready_01", "evt_ready_02"]},
            )
        )

        assert report.directory == str(used)
        assert (report.found, report.total) == (1, 2)
        assert report.problems == {"evt_ready_02": "missing"}
        assert loaded.events["ready"] == ("第0句。",)
        assert loaded.paths == {"第0句。": str(used / "evt_ready_01_v1.wav")}

    def test_only_24k_mono_16_bit_is_usable(self, tmp_path):
        write_wav(tmp_path / "evt_ready_01_v1.wav")
        write_wav(tmp_path / "evt_ready_02_v1.wav", rate=16000)
        write_wav(tmp_path / "evt_ready_03_v1.wav", channels=2)
        write_wav(tmp_path / "evt_ready_04_v1.wav", width=4)
        (tmp_path / "evt_ready_05_v1.wav").write_bytes(b"not a wav at all")
        ids = [f"evt_ready_0{n}" for n in range(1, 6)]

        loaded, report = load_clips(playback([tmp_path], ids, events={"ready": ids}))

        assert report.found == 1
        assert set(report.problems) == set(ids[1:])
        assert "16000 Hz" in report.problems["evt_ready_02"]
        assert loaded.events["ready"] == ("第0句。",)

    def test_events_left_without_a_clip_are_reported(self, tmp_path):
        write_wav(tmp_path / "evt_ready_01_v1.wav")
        _, report = load_clips(
            playback(
                [tmp_path],
                ["evt_ready_01", "evt_goodbye_01"],
                events={"ready": ["evt_ready_01"], "goodbye": ["evt_goodbye_01"]},
            )
        )

        assert set(report.silent_events) == {"starting", "mirroring", "goodbye"}

    @pytest.mark.parametrize("name", ["missing.wav"])
    def test_check_wav_names_a_missing_file(self, tmp_path, name):
        assert check_wav(str(tmp_path / name)) == "missing"
