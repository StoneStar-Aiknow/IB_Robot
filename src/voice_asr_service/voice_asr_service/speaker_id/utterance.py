"""Reassemble one utterance's PCM from the boundary-carrying chunk contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voice_asr_service.speech_segments import (
    STATE_BEGIN,
    STATE_END,
    STATE_PROCESSING,
    STATE_UNDEFINED,
    SegmentTracker,
    keeps_segment_alive,
)

__all__ = ["STATE_BEGIN", "STATE_END", "STATE_PROCESSING", "STATE_UNDEFINED", "Utterance", "UtteranceAssembler"]

#: Mono signed 16-bit: two bytes per sample.
_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class Utterance:
    """One finished utterance, with the length it actually had."""

    pcm: bytes
    #: Full duration of the utterance, including audio ``max_seconds`` dropped.
    speech_seconds: float
    #: True when the producer never sent an END and the idle timeout closed it.
    stalled: bool = False
    #: The segment this is: its source and the sequence of the BEGIN that opened it.
    source_id: str = ""
    segment_sequence: int = 0
    #: Header of the segment's last chunk that carried speech (or its END), as received.
    header: Any = None


@dataclass
class UtteranceAssembler:
    """Collect one utterance's PCM, bounded by what a voiceprint actually needs.

    Which chunks belong to a segment is decided by ``speech_segments``, the
    rules ``voice_asr_node``'s external input applies too: an utterance the ASR
    drops must not leave a voiceprint (or a "none") behind for the next
    transcript to be judged by. So a stale chunk is ignored, a source restart or
    a sequence gap discards the open utterance, a ``BEGIN`` over an open
    utterance discards it, and chunks before the first ``BEGIN`` are dropped,
    because without the onset there is no telling which utterance they belong to.

    Only the first ``max_seconds`` are kept. Verification needs a few seconds of
    voice rather than all of it, this bounds memory for someone who does not
    stop talking, and the kept part is the start of the utterance, which is
    where the wake phrase is.
    """

    sample_rate: int = 16000
    max_seconds: float = 6.0
    #: A producer that stops mid-utterance would otherwise buffer forever. Same
    #: role as ``voice_asr_node``'s ``external_idle_timeout``, and shorter than it
    #: so the voiceprint is published before the transcript.
    idle_timeout_sec: float = 1.2
    #: Why the last chunk discarded an open utterance; empty when it did not.
    discarded: str = field(default="", init=False)
    _segments: SegmentTracker = field(default_factory=SegmentTracker, repr=False)
    _pcm: bytearray = field(default_factory=bytearray, repr=False)
    _open: bool = False
    _samples: int = 0
    _last_at: float = 0.0
    _source_id: str = ""
    _segment_sequence: int = 0
    _header: Any = None

    def push(
        self, state: int, audio: bytes, *, at: float, source_id: str, sequence: int, header: Any = None
    ) -> Utterance | None:
        """Feed one chunk, in arrival order; returns the utterance when this chunk ended it."""
        self.discarded = ""
        admission = self._segments.admit(source_id, sequence)
        if admission.stale:
            return None
        if admission.discard and self._open:
            self.discarded = admission.discard
            self.reset()
        if state == STATE_BEGIN:
            if self._open:
                self.discarded = "BEGIN while a segment was open"
            self.reset()
            self._open = True
            self._source_id, self._segment_sequence = source_id, int(sequence)
        if not self._open:
            return None
        if keeps_segment_alive(state, audio):
            # The runtime's availability heartbeat (state UNDEFINED, no PCM)
            # arrives every 0.5 s; counting it would mean an utterance whose
            # producer never sends END is never flushed, and the next BEGIN then
            # discards it with no voiceprint (measured on the X2, 2026-10-25: the
            # gate failed open on exactly such a bystander).
            self._last_at = at
            self._header = header
        self._samples += len(audio) // _BYTES_PER_SAMPLE
        keep = int(self.max_seconds * self.sample_rate) * _BYTES_PER_SAMPLE - len(self._pcm)
        if keep > 0:
            self._pcm += audio[:keep]
        if state != STATE_END:
            return None
        if not audio:
            self._header = header
        return self._close()

    def flush_stalled(self, at: float) -> Utterance | None:
        """Close an utterance whose producer went quiet without sending END."""
        if not self._open or at - self._last_at <= self.idle_timeout_sec:
            return None
        return self._close(stalled=True)

    def reset(self) -> None:
        self._pcm = bytearray()
        self._open = False
        self._samples = 0

    def _close(self, *, stalled: bool = False) -> Utterance | None:
        pcm, seconds = bytes(self._pcm), self._samples / float(self.sample_rate)
        identity = {"source_id": self._source_id, "segment_sequence": self._segment_sequence, "header": self._header}
        self.reset()
        return Utterance(pcm, seconds, stalled, **identity) if pcm else None
