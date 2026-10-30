"""Deterministic tests for the atomic external-VAD audio contract."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.external_vad_input import ExternalVadInput  # noqa: E402


class Logger:
    def __init__(self):
        self.records = []

    def info(self, message):
        self.records.append(("info", message))

    def warning(self, message):
        self.records.append(("warning", message))

    def error(self, message):
        self.records.append(("error", message))


class Node:
    def __init__(self):
        self.logger = Logger()
        self.callback = None

    def create_subscription(self, _type, topic, callback, _qos):
        self.topic = topic
        self.callback = callback
        return callback

    def get_logger(self):
        return self.logger


class Result:
    def __init__(self, text=""):
        self.text = text


class Asr:
    def __init__(self):
        self.opened = 0
        self.fed = []
        self.reset_calls = 0
        self.fail = False
        self.fail_start = False
        self.fail_final = False
        self.fail_reset = False
        self.active = False
        self.completed_audio = []

    def start_streaming(self):
        if self.fail_start:
            raise RuntimeError("start failed")
        if self.active:
            return False
        self.active = True
        self.opened += 1
        self.fed.append([])
        return True

    def accept_waveform(self, samples):
        assert self.active
        if self.fail:
            raise RuntimeError("decode failed")
        self.fed[-1].append(np.asarray(samples).copy())
        return None

    def get_final_result(self):
        assert self.active
        if self.fail_final:
            raise RuntimeError("final failed")
        self.completed_audio.append(np.concatenate(self.fed[-1]) if self.fed[-1] else np.array([]))
        self.active = False
        return Result("final text")

    def reset(self):
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        self.active = False


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def chunk(source, sequence, state, values=b"\x00\x00"):
    return types.SimpleNamespace(source_id=source, sequence=sequence, state=state, audio_data=values)


@pytest.fixture()
def message_module(monkeypatch):
    package = types.ModuleType("ibrobot_msgs")
    messages = types.ModuleType("ibrobot_msgs.msg")
    messages.SpeechAudioChunk = object
    monkeypatch.setitem(sys.modules, "ibrobot_msgs", package)
    monkeypatch.setitem(sys.modules, "ibrobot_msgs.msg", messages)


def make(clock=None, **kwargs):
    node, asr = Node(), Asr()
    final, partial, states = [], [], []
    input_ = ExternalVadInput(
        node=node,
        asr=asr,
        chunk_topic="/audio/speech_chunk",
        on_final=final.append,
        on_partial=partial.append,
        on_state=states.append,
        clock=clock or time.monotonic,
        **kwargs,
    )
    return input_, node, asr, final, partial, states


def test_atomic_end_chunk_is_fed_before_finalize(message_module):
    input_, _node, asr, final, _partial, states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x01\x00"))
    input_._subscription(chunk("a", 2, 3, b"\x02\x00"))
    assert asr.fed[0][0][0] == pytest.approx(1 / 32768)
    assert asr.fed[0][1][0] == pytest.approx(2 / 32768)
    assert final == ["final text"]
    assert states == ["recognizing", "listening"]


def test_the_final_names_the_segment_its_begin_opened(message_module):
    """SpeechTranscript carries (source_id, BEGIN sequence), the identity the
    voiceprint and the wake event of the same utterance carry too."""
    segment_finals = []
    input_, _node, _asr, final, _partial, _states = make(
        min_samples=1, on_segment_final=lambda *item: segment_finals.append(item)
    )
    input_.enabled = True
    for message in (chunk("a", 7, 1, b"\x01\x00"), chunk("a", 8, 2, b"\x01\x00"), chunk("a", 9, 3, b"\x01\x00")):
        input_._subscription(message)

    assert final == ["final text"]
    assert segment_finals == [("final text", ("a", 7))]


def test_a_discarded_segment_names_nothing(message_module):
    segment_finals = []
    input_, _node, _asr, _final, _partial, _states = make(
        min_samples=1, on_segment_final=lambda *item: segment_finals.append(item)
    )
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    input_._subscription(chunk("a", 3, 3))

    assert segment_finals == []


def test_manual_gate_drops_audio_until_enabled(message_module):
    input_, _node, asr, final, _partial, _states = make(min_samples=1)
    input_._subscription(chunk("a", 1, 1))
    assert asr.opened == 0 and final == []


def test_sequence_gap_aborts_current_segment(message_module):
    input_, _node, asr, final, _partial, states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    input_._subscription(chunk("a", 3, 2))
    assert asr.reset_calls >= 1
    assert final == []
    assert states == ["recognizing", "listening"]
    assert ("warning", "External speech sequence gap (1 -> 3); discarding segment") in _node.logger.records


def test_duplicate_sequence_is_ignored(message_module):
    input_, _node, asr, _final, _partial, _states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    input_._subscription(chunk("a", 1, 3))
    assert len(asr.fed[0]) == 1


def test_source_restart_aborts_old_stream(message_module):
    input_, _node, asr, final, _partial, _states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    input_._subscription(chunk("b", 1, 1))
    input_._subscription(chunk("b", 2, 3))
    assert asr.reset_calls >= 1
    assert final == ["final text"]


def test_stop_abort_releases_stream_and_drops_segment(message_module):
    input_, _node, asr, final, _partial, _states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    input_.enabled = False
    input_.enabled = True
    input_._subscription(chunk("a", 2, 3))
    assert asr.reset_calls >= 1
    assert final == []


def test_deadline_aborts_without_finalizing(message_module):
    clock = Clock()
    input_, _node, asr, final, _partial, _states = make(clock=clock, min_samples=1, max_duration=1, idle_timeout=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1))
    clock.value = 2
    assert input_.check_timeout()
    assert asr.reset_calls >= 1 and final == []


def test_odd_pcm_is_dropped_without_escape(message_module):
    input_, node, asr, final, _partial, _states = make(min_samples=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x00"))
    assert asr.opened == 0 and final == []
    assert any(level == "error" for level, _ in node.logger.records)


def test_invalid_parameters_rejected(message_module):
    with pytest.raises(ValueError):
        make(max_duration=0)
    with pytest.raises(ValueError):
        make(idle_timeout=0)


@pytest.mark.parametrize("failure", ["fail_start", "fail", "fail_final"])
def test_failure_drops_residual_and_next_segment_recovers(message_module, failure):
    voice, _, asr, final, _, _ = make(min_samples=1)
    voice.enabled = True
    setattr(asr, failure, True)
    for seq, state in [(1, 1), (2, 2), (3, 3)]:
        voice._on_chunk(chunk("a", seq, state, bytes([seq, 0])))
    assert final == [] and not asr.active
    setattr(asr, failure, False)
    voice._on_chunk(chunk("a", 4, 1, b"\x04\x00"))
    voice._on_chunk(chunk("a", 5, 3, b"\x05\x00"))
    assert final == ["final text"]
    np.testing.assert_array_equal(asr.completed_audio[-1], np.array([4, 5]) / 32768)


def test_full_begin_backfill_and_end_tail_are_finalized_together(message_module):
    voice, _, asr, final, _, _ = make(min_samples=1)
    voice.enabled = True
    backfill = np.arange(9600, dtype="<i2")
    voice._on_chunk(chunk("a", 1, 1, backfill.tobytes()))
    voice._on_chunk(chunk("a", 2, 3, b"\x01\x00"))
    np.testing.assert_array_equal(asr.completed_audio[0][:-1], backfill.astype(np.float32) / 32768)
    assert final == ["final text"]


def test_repeated_begin_preserves_new_onset(message_module):
    voice, _, asr, final, _, _ = make(min_samples=1)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 1))
    voice._on_chunk(chunk("a", 2, 1, b"\x02\x00"))
    voice._on_chunk(chunk("a", 3, 3, b"\x03\x00"))
    assert final == ["final text"]
    np.testing.assert_array_equal(asr.completed_audio[0], np.array([2, 3]) / 32768)


def test_orphan_end_and_short_segment_do_not_publish(message_module):
    voice, node, asr, final, _, _ = make(min_samples=3200)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 3))
    assert asr.opened == 0
    voice._on_chunk(chunk("a", 2, 1))
    voice._on_chunk(chunk("a", 3, 3))
    assert final == [] and not asr.active
    # A dropped segment says why, so "heard nothing" and "threw it away" differ in the log.
    assert ("info", "External speech segment produced no command: only 2 samples (< 3200)") in node.logger.records


def test_a_refused_stream_start_is_reported(message_module):
    voice, node, asr, final, _, _ = make(min_samples=1)
    voice.enabled = True
    asr.active = True  # the facade still believes a stream is open

    voice._on_chunk(chunk("a", 1, 1))

    assert ("warning", "ASR refused to start a stream; dropping this segment") in node.logger.records


def test_heartbeat_does_not_extend_idle_deadline(message_module):
    clock = Clock()
    voice, _, asr, final, _, _ = make(clock=clock, min_samples=1)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 1))
    clock.value = 1
    voice._on_chunk(chunk("a", 2, 0, b""))
    clock.value = 2.1
    voice._on_chunk(chunk("a", 3, 0, b""))
    assert not asr.active and final == []


def test_sample_limit_applies_even_to_fast_backfill(message_module):
    voice, _, asr, final, _, _ = make(clock=Clock(), max_duration=1, min_samples=1)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 1, b"\x00\x00" * 10000))
    voice._on_chunk(chunk("a", 2, 3, b"\x00\x00" * 10000))
    assert final == [] and not asr.active


def test_reset_failure_does_not_escape(message_module):
    voice, node, asr, final, _, _ = make(min_samples=1)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 1))
    asr.fail = asr.fail_reset = True
    voice._on_chunk(chunk("a", 2, 2))
    voice._on_chunk(chunk("a", 3, 3))
    assert final == []
    assert any("abort failed" in text for _, text in node.logger.records)


# --- finalize_on_idle: for a source that never sends END (X2 firmware) ---


def test_idle_without_end_finalizes_when_opted_in(message_module):
    """Measured on the X2: BEGIN, PROCESSING..., then the stream just stops.

    Discarding there threw away a complete utterance; silence after speech is
    what an idle deadline means, so the segment is finalized instead.
    """
    clock = Clock()
    input_, _node, asr, final, _partial, states = make(
        clock=clock, min_samples=1, max_duration=10, idle_timeout=1, finalize_on_idle=True
    )
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x01\x00"))
    clock.value = 2

    assert input_.check_timeout()
    assert final == ["final text"]
    assert states == ["recognizing", "listening"]
    assert asr.reset_calls == 0


def test_idle_finalization_stays_off_by_default(message_module):
    """Where a producer does send END, an idle expiry is a stall mid-utterance,
    and publishing a truncated command to a robot is worse than publishing none."""
    clock = Clock()
    input_, _node, asr, final, _partial, _states = make(clock=clock, min_samples=1, max_duration=10, idle_timeout=1)
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x01\x00"))
    clock.value = 2

    assert input_.check_timeout()
    assert final == []
    assert asr.reset_calls >= 1


def test_max_duration_is_still_discarded_even_with_idle_finalization(message_module):
    """A runaway stream is not an ended utterance; only the idle rule changes."""
    clock = Clock()
    input_, _node, asr, final, _partial, _states = make(
        clock=clock, min_samples=1, max_duration=1, idle_timeout=10, finalize_on_idle=True
    )
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x01\x00"))
    clock.value = 2

    assert input_.check_timeout()
    assert final == []
    assert asr.reset_calls >= 1


def test_idle_finalization_still_respects_min_samples(message_module):
    clock = Clock()
    input_, _node, _asr, final, _partial, _states = make(
        clock=clock, min_samples=100, max_duration=10, idle_timeout=1, finalize_on_idle=True
    )
    input_.enabled = True
    input_._subscription(chunk("a", 1, 1, b"\x01\x00"))
    clock.value = 2

    assert input_.check_timeout()
    assert final == []


def test_a_begin_over_an_open_segment_is_discarded_even_when_finalizing_on_idle(message_module):
    """On the X2 that segment is the wake word's residue; transcribed, it would open the
    session with no voiceprint and leave the addressee gate open (measured 2026-10-25)."""
    voice, node, asr, final, _, _ = make(min_samples=1, finalize_on_idle=True)
    voice.enabled = True
    voice._on_chunk(chunk("a", 1, 1, b"\x01\x00"))
    voice._on_chunk(chunk("a", 2, 1, b"\x02\x00"))
    voice._on_chunk(chunk("a", 3, 3, b"\x03\x00"))

    assert final == ["final text"]
    assert ("warning", "External speech BEGIN while a segment was open; discarding it") in node.logger.records
    np.testing.assert_array_equal(asr.completed_audio[0], np.array([2, 3]) / 32768)
