"""Deterministic tests for the per-utterance voiceprint producer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.speaker_id.embedder import l2_normalized  # noqa: E402
from voice_asr_service.speaker_id.utterance import (  # noqa: E402
    STATE_BEGIN,
    STATE_END,
    STATE_PROCESSING,
    STATE_UNDEFINED,
    UtteranceAssembler,
)

RATE = 16000


def _pcm(seconds: float) -> bytes:
    return b"\x01\x00" * int(seconds * RATE)


class _Stream:
    """One source's chunks in order, the way the runtime numbers them."""

    def __init__(self, assembler, source_id="runtime-1"):
        self.assembler, self.source_id, self.sequence = assembler, source_id, 0

    def push(self, state, audio, *, at, sequence=None, header=None):
        self.sequence = self.sequence + 1 if sequence is None else sequence
        return self.assembler.push(state, audio, at=at, source_id=self.source_id, sequence=self.sequence, header=header)

    def flush_stalled(self, at):
        return self.assembler.flush_stalled(at)


def test_begin_to_end_is_one_utterance_with_its_real_duration():
    assembler = _Stream(UtteranceAssembler(sample_rate=RATE))

    assert assembler.push(STATE_BEGIN, _pcm(0.5), at=0.0) is None
    assert assembler.push(STATE_PROCESSING, _pcm(0.5), at=0.1) is None
    utterance = assembler.push(STATE_END, _pcm(0.5), at=0.2)

    assert utterance is not None
    assert utterance.speech_seconds == pytest.approx(1.5)
    assert len(utterance.pcm) == len(_pcm(1.5))
    assert not utterance.stalled


def test_only_the_first_max_seconds_are_kept_but_the_duration_is_the_full_one():
    """A voiceprint needs a few seconds, not all of them; the start holds the wake phrase."""
    assembler = _Stream(UtteranceAssembler(sample_rate=RATE, max_seconds=2.0))

    assembler.push(STATE_BEGIN, _pcm(1.0), at=0.0)
    utterance = assembler.push(STATE_END, _pcm(5.0), at=1.0)

    assert utterance is not None
    assert len(utterance.pcm) == len(_pcm(2.0))
    assert utterance.speech_seconds == pytest.approx(6.0)


def test_an_utterance_whose_producer_went_quiet_is_closed_by_the_idle_timeout():
    assembler = _Stream(UtteranceAssembler(sample_rate=RATE, idle_timeout_sec=2.0))
    assembler.push(STATE_BEGIN, _pcm(1.0), at=10.0)

    assert assembler.flush_stalled(11.5) is None
    utterance = assembler.flush_stalled(12.5)

    assert utterance is not None
    assert utterance.stalled
    assert assembler.flush_stalled(20.0) is None


def test_an_empty_utterance_produces_nothing():
    """A BEGIN with no audio is not a voice; embedding silence would be noise."""
    assembler = _Stream(UtteranceAssembler(sample_rate=RATE))

    assert assembler.push(STATE_BEGIN, b"", at=0.0) is None
    assert assembler.push(STATE_END, b"", at=0.1) is None


def test_a_heartbeat_does_not_disturb_an_open_utterance():
    assembler = _Stream(UtteranceAssembler(sample_rate=RATE))
    assembler.push(STATE_BEGIN, _pcm(1.0), at=0.0)

    assembler.push(STATE_UNDEFINED, b"", at=0.5)
    utterance = assembler.push(STATE_END, _pcm(1.0), at=1.0)

    assert utterance is not None
    assert utterance.speech_seconds == pytest.approx(2.0)


def test_l2_normalized_makes_a_cosine_a_dot_product():
    normalized = l2_normalized([3.0, 4.0])

    assert normalized == pytest.approx((0.6, 0.8))
    assert math.sqrt(sum(item * item for item in normalized)) == pytest.approx(1.0)


def test_l2_normalized_refuses_a_zero_vector():
    """There is no direction to compare; zeros would read as unlike everything."""
    assert l2_normalized([0.0, 0.0]) == ()
    assert l2_normalized([]) == ()


# --- segment admission: the same rules as voice_asr_node's external input ---


def test_a_sequence_gap_discards_the_open_utterance():
    """Part of it is missing, and the ASR drops it for the same reason."""
    stream = _Stream(UtteranceAssembler(sample_rate=RATE))
    stream.push(STATE_BEGIN, _pcm(1.0), at=0.0)

    assert stream.push(STATE_PROCESSING, _pcm(1.0), at=0.1, sequence=3) is None
    assert stream.assembler.discarded == "sequence gap (1 -> 3)"
    assert stream.push(STATE_END, _pcm(1.0), at=0.2) is None


def test_a_begin_over_an_open_utterance_says_it_discarded_it():
    stream = _Stream(UtteranceAssembler(sample_rate=RATE))
    stream.push(STATE_BEGIN, _pcm(1.0), at=0.0)

    stream.push(STATE_BEGIN, _pcm(1.0), at=1.0)

    assert stream.assembler.discarded == "BEGIN while a segment was open"


def test_an_utterance_names_its_segment():
    """(source_id, segment_sequence) is what a transcript can later be paired on."""
    stream = _Stream(UtteranceAssembler(sample_rate=RATE), source_id="runtime-9")
    stream.push(STATE_BEGIN, _pcm(1.0), at=0.0, sequence=41, header="begin")
    stream.push(STATE_PROCESSING, _pcm(1.0), at=0.1, header="last speech")
    utterance = stream.push(STATE_END, _pcm(1.0), at=0.2, header="end")

    assert (utterance.source_id, utterance.segment_sequence, utterance.header) == ("runtime-9", 41, "end")


def test_a_stalled_utterance_keeps_the_header_of_its_last_speech():
    stream = _Stream(UtteranceAssembler(sample_rate=RATE, idle_timeout_sec=1.0))
    stream.push(STATE_BEGIN, _pcm(1.0), at=0.0, sequence=5, header="begin")
    stream.push(STATE_PROCESSING, _pcm(1.0), at=0.5, header="last speech")
    stream.push(STATE_UNDEFINED, b"", at=1.0, header="heartbeat")

    utterance = stream.flush_stalled(2.0)

    assert (utterance.segment_sequence, utterance.header) == (5, "last speech")
