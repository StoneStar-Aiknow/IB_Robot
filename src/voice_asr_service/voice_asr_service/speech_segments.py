"""Which chunks of a SpeechAudioChunk stream belong to a usable segment.

``voice_asr_node``'s external input and ``speaker_id_node`` read the same chunk
stream, and the agent's addressee gate pairs each transcript with the
voiceprint of the same segment. Every transcript only has one while both keep
and drop the same segments; a segment the voiceprint node drops leaves its
transcript to be judged without one (fail-open). Measured on the X2
(2026-10-26): with a second runtime publishing on the same topic, the ASR
discarded most segments on interleaved sequence numbers while the voiceprint
node, which did not check them, embedded every one.

The rules both consumers apply, so they live here once:

- A chunk whose sequence is not newer than the last one from its source is
  stale (re-sent or reordered) and is ignored entirely.
- A new ``source_id`` (the producer restarted) voids the open segment.
- A gap in the sequence voids the open segment: part of it is missing.
- Only ``BEGIN`` opens a segment; chunks before it are dropped, and a ``BEGIN``
  over an open segment discards that segment (on the X2 it is almost always the
  wake word's residue).
- Only a chunk that carries PCM (or the ``BEGIN``) keeps a segment alive; the
  runtime's empty heartbeat does not.

Two differences are deliberate and stay with each consumer. The voiceprint node
closes an END-less segment on a shorter idle timeout than the ASR, so that its
voiceprint is published before the transcript; when the next ``BEGIN`` lands
between the two timeouts, the ASR discards a segment the voiceprint node has
already embedded. And the ASR discards a runaway segment past its maximum
duration, while the voiceprint node embeds its first seconds. Both only leave a
voiceprint with no transcript, which, paired by ``(source_id,
segment_sequence)``, judges nobody and expires.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ibrobot_msgs/msg/SpeechAudioChunk state values, repeated here so the rules
#: stay importable without a built ROS workspace.
STATE_UNDEFINED = 0
STATE_BEGIN = 1
STATE_PROCESSING = 2
STATE_END = 3


@dataclass(frozen=True)
class Admission:
    """What one chunk means for the consumer's open segment."""

    #: Not newer than the last chunk of its source: ignore it entirely.
    stale: bool = False
    #: Why the open segment, if any, must be discarded before this chunk is
    #: handled; empty when it may continue.
    discard: str = ""


class SegmentTracker:
    """Track one chunk stream's source and sequence for a segment consumer."""

    def __init__(self) -> None:
        self._source_id: str | None = None
        self._last_sequence = 0

    def admit(self, source_id: str, sequence: int) -> Admission:
        """Judge one chunk by its source and sequence; call it for every chunk, in arrival order."""
        sequence = int(sequence)
        if source_id != self._source_id:
            restarted = self._source_id is not None
            self._source_id, self._last_sequence = source_id, sequence
            return Admission(discard="source restarted" if restarted else "")
        previous = self._last_sequence
        if sequence <= previous:
            return Admission(stale=True)
        self._last_sequence = sequence
        if sequence != previous + 1:
            return Admission(discard=f"sequence gap ({previous} -> {sequence})")
        return Admission()


def keeps_segment_alive(state: int, audio: bytes) -> bool:
    """Whether this chunk counts as speech for the idle timeout."""
    return state == STATE_BEGIN or bool(audio)
