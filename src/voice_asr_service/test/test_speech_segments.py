"""The ASR and the voiceprint node keep and drop the same segments of one chunk stream.

The addressee gate pairs one transcript with one voiceprint. Each scenario below
is fed, chunk for chunk, to ``ExternalVadInput`` (the ASR's external input) and
to ``UtteranceAssembler`` (the voiceprint node's), and both must come out with
the same segments. Every chunk of a segment carries its BEGIN sequence as PCM,
so a transcript and a voiceprint both say which segment they came from, and
both report how many samples the segment had.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.speech_segments import (  # noqa: E402
    STATE_BEGIN,
    STATE_END,
    STATE_PROCESSING,
    STATE_UNDEFINED,
    SegmentTracker,
    keeps_segment_alive,
)

#: Idle timeouts as on the X2: the voiceprint node closes first on purpose.
ASR_IDLE, VOICEPRINT_IDLE = 2.0, 1.2


class _Node:
    def __init__(self):
        self._subscription = None

    def create_subscription(self, _type, _topic, callback, _qos):
        self._subscription = callback
        return callback

    def get_logger(self):
        return types.SimpleNamespace(info=lambda _m: None, warning=lambda _m: None, error=lambda _m: None)


class _Asr:
    """Transcribes a segment as "<value its PCM carries>:<samples fed>"."""

    def __init__(self):
        self.samples = []
        self.active = False

    def start_streaming(self):
        self.active, self.samples = True, []
        return True

    def accept_waveform(self, samples):
        self.samples.extend(np.asarray(samples).tolist())

    def get_final_result(self):
        self.active = False
        value = round(self.samples[0] * 32768) if self.samples else 0
        return types.SimpleNamespace(text=f"{value}:{len(self.samples)}")

    def reset(self):
        self.active, self.samples = False, []


@pytest.fixture()
def message_module(monkeypatch):
    messages = types.ModuleType("ibrobot_msgs.msg")
    messages.SpeechAudioChunk = object
    monkeypatch.setitem(sys.modules, "ibrobot_msgs", types.ModuleType("ibrobot_msgs"))
    monkeypatch.setitem(sys.modules, "ibrobot_msgs.msg", messages)


def _pcm(value: int, samples: int = 1600) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


def _asr_segments(script, idle_at: float) -> list[tuple[int, int]]:
    from voice_asr_service.external_vad_input import ExternalVadInput

    clock = types.SimpleNamespace(now=0.0)
    finals: list[str] = []
    asr_input = ExternalVadInput(
        node=_Node(),
        asr=_Asr(),
        chunk_topic="/audio/speech_chunk",
        min_samples=1,
        idle_timeout=ASR_IDLE,
        finalize_on_idle=True,
        on_final=finals.append,
        clock=lambda: clock.now,
    )
    asr_input.enabled = True
    for source, sequence, state, audio, at in script:
        clock.now = at
        asr_input._subscription(
            types.SimpleNamespace(source_id=source, sequence=sequence, state=state, audio_data=audio)
        )
    clock.now = idle_at
    asr_input.check_timeout()
    return [tuple(int(part) for part in text.split(":")) for text in finals]


def _voiceprint_segments(script, idle_at: float) -> list[tuple[int, int]]:
    from voice_asr_service.speaker_id.utterance import UtteranceAssembler

    assembler = UtteranceAssembler(idle_timeout_sec=VOICEPRINT_IDLE)
    utterances = []
    for source, sequence, state, audio, at in script:
        utterance = assembler.push(state, audio, at=at, source_id=source, sequence=sequence)
        if utterance is not None:
            utterances.append(utterance)
    stalled = assembler.flush_stalled(idle_at)
    if stalled is not None:
        utterances.append(stalled)
    return [
        (int(np.frombuffer(item.pcm[:2], dtype="<i2")[0]), round(item.speech_seconds * 16000)) for item in utterances
    ]


def _script(*chunks):
    """(source, sequence, state, value) per chunk, 0.1 s apart; value 0 is an empty chunk."""
    return [
        (source, sequence, state, _pcm(value) if value else b"", 0.1 * index)
        for index, (source, sequence, state, value) in enumerate(chunks)
    ]


SCENARIOS = {
    "begin to end": (
        _script(("a", 1, STATE_BEGIN, 1), ("a", 2, STATE_PROCESSING, 1), ("a", 3, STATE_END, 1)),
        [(1, 4800)],
    ),
    "begin over an open segment": (
        _script(
            ("a", 1, STATE_BEGIN, 1),
            ("a", 2, STATE_PROCESSING, 1),
            ("a", 3, STATE_BEGIN, 3),
            ("a", 4, STATE_END, 3),
        ),
        [(3, 3200)],
    ),
    "sequence gap": (
        _script(("a", 1, STATE_BEGIN, 1), ("a", 3, STATE_PROCESSING, 1), ("a", 4, STATE_END, 1)),
        [],
    ),
    "stale chunk": (
        _script(
            ("a", 1, STATE_BEGIN, 1),
            ("a", 2, STATE_PROCESSING, 1),
            ("a", 2, STATE_PROCESSING, 1),
            ("a", 3, STATE_END, 1),
        ),
        [(1, 4800)],
    ),
    "source restart": (
        _script(
            ("a", 1, STATE_BEGIN, 1),
            ("a", 2, STATE_PROCESSING, 1),
            ("b", 1, STATE_END, 1),
            ("b", 2, STATE_BEGIN, 2),
            ("b", 3, STATE_END, 2),
        ),
        [(2, 3200)],
    ),
    "chunks before any begin": (
        _script(("a", 1, STATE_PROCESSING, 1), ("a", 2, STATE_END, 1)),
        [],
    ),
    "no end, closed by idle through heartbeats": (
        _script(
            ("a", 1, STATE_BEGIN, 1),
            ("a", 2, STATE_PROCESSING, 1),
            ("a", 3, STATE_UNDEFINED, 0),
            ("a", 4, STATE_UNDEFINED, 0),
        ),
        [(1, 3200)],
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_asr_and_the_voiceprint_node_keep_the_same_segments(message_module, name):
    script, expected = SCENARIOS[name]
    idle_at = script[-1][-1] + ASR_IDLE + 1.0

    assert _asr_segments(script, idle_at) == expected
    assert _voiceprint_segments(script, idle_at) == expected


def test_a_stale_sequence_is_ignored_without_touching_the_segment():
    tracker = SegmentTracker()
    tracker.admit("a", 1)
    tracker.admit("a", 2)

    assert tracker.admit("a", 2).stale
    assert tracker.admit("a", 1).stale
    assert tracker.admit("a", 3).discard == ""


def test_a_gap_voids_the_segment_and_tracking_resumes_from_the_new_sequence():
    tracker = SegmentTracker()
    tracker.admit("a", 1)

    assert tracker.admit("a", 4).discard == "sequence gap (1 -> 4)"
    assert tracker.admit("a", 5).discard == ""


def test_a_new_source_voids_the_segment_and_restarts_numbering():
    tracker = SegmentTracker()
    tracker.admit("a", 90)

    assert tracker.admit("b", 1).discard == "source restarted"
    assert not tracker.admit("b", 2).stale


def test_the_first_chunk_of_a_stream_voids_nothing():
    assert SegmentTracker().admit("a", 57) == SegmentTracker().admit("b", 1)
    assert SegmentTracker().admit("a", 57).discard == ""


@pytest.mark.parametrize(
    ("state", "audio", "alive"),
    [
        pytest.param(STATE_BEGIN, b"", True, id="an empty BEGIN still opens the clock"),
        pytest.param(STATE_PROCESSING, b"\x01\x00", True, id="speech"),
        pytest.param(STATE_UNDEFINED, b"", False, id="the runtime heartbeat"),
        pytest.param(STATE_END, b"", False, id="an empty END"),
    ],
)
def test_only_speech_keeps_a_segment_alive(state, audio, alive):
    assert keeps_segment_alive(state, audio) is alive
