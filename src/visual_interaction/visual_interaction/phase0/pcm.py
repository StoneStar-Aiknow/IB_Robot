"""PCM publish timing on ``/audio/play``, judged against the sample clock.

The playback node (``voice_tts_audio_player``) publishes ``rate // 20`` frames
per message — 50 ms of audio — and sleeps to an absolute sample-clock deadline
after each one, so chunk ``i`` is due ``frames_before_i / rate`` after the
first. There is no prebuffer: a chunk that arrives late is late for the sink.

Two views, because each hides what the other shows (design §12):

* **Intervals** between arrivals, and the share above one chunk period. A
  single long gap shows here and vanishes in an average rate.
* **Lateness** against the sample clock. Several slightly long intervals in a
  row add up to a starved sink without any one of them looking bad; only the
  cumulative view catches that.

What this observes is arrival at a ROS subscriber. It cannot see the vendor
audio buffer or the speaker, so it bounds publish-side trouble only; acoustic
continuity needs a separately authorised recording.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from visual_interaction.phase0.stats import summarise

OBSERVED_AT = "ros_subscriber"
# Ratios of the chunk period reported for intervals and lateness.
_THRESHOLDS = (1.0, 1.5, 2.0)
# Slack on each threshold, to keep float representation out of the counts: a
# stream arriving exactly on the sample clock accumulates errors of ~1e-14 s,
# which would otherwise be reported as real overruns. Twelve orders of
# magnitude below a chunk period, so no timing effect hides under it.
_THRESHOLD_EPSILON = 1e-9


@dataclass(frozen=True)
class PcmFormat:
    """What the playback node is configured to publish: signed 16-bit PCM."""

    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if min(self.sample_rate, self.channels, self.sample_width) <= 0:
            raise ValueError("every format field must be positive")

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * self.sample_width

    @property
    def chunk_frames(self) -> int:
        """The playback node's chunk size: ``max(1, rate // 20)``."""
        return max(1, self.sample_rate // 20)

    @property
    def chunk_period(self) -> float:
        return self.chunk_frames / self.sample_rate

    def frames_in(self, payload_bytes: int) -> int:
        if payload_bytes % self.bytes_per_frame:
            raise ValueError(f"{payload_bytes} bytes is not a whole number of {self.bytes_per_frame}-byte frames")
        return payload_bytes // self.bytes_per_frame


@dataclass(frozen=True)
class Chunk:
    """One message as it arrived: when (monotonic seconds) and how many bytes."""

    arrived: float
    payload_bytes: int


def split_utterances(chunks: Sequence[Chunk], *, gap: float) -> list[list[Chunk]]:
    """Group chunks into utterances wherever arrivals pause longer than ``gap``.

    ``gap`` has to sit well above any plausible stall and well below the
    pause between two sentences; a stall longer than it splits an utterance
    and is reported as two, which undercounts lateness. Pick it from the
    condition being measured and record it.
    """
    if gap <= 0.0:
        raise ValueError("gap must be positive")
    groups: list[list[Chunk]] = []
    for chunk in sorted(chunks, key=lambda item: item.arrived):
        if groups and chunk.arrived - groups[-1][-1].arrived <= gap:
            groups[-1].append(chunk)
        else:
            groups.append([chunk])
    return groups


def _intervals(ordered: Sequence[Chunk]) -> list[float]:
    return [later.arrived - earlier.arrived for earlier, later in zip(ordered, ordered[1:], strict=False)]


def _lateness(ordered: Sequence[Chunk], pcm: PcmFormat) -> list[float]:
    """How long after its sample-clock due time each chunk arrived."""
    start = ordered[0].arrived
    late = []
    frames_before = 0
    for chunk in ordered:
        late.append(chunk.arrived - start - frames_before / pcm.sample_rate)
        frames_before += pcm.frames_in(chunk.payload_bytes)
    return late


def analyse_utterance(chunks: Sequence[Chunk], pcm: PcmFormat) -> dict[str, object]:
    """Intervals and sample-clock lateness for one utterance, in milliseconds."""
    if not chunks:
        raise ValueError("an utterance has at least one chunk")
    ordered = sorted(chunks, key=lambda item: item.arrived)
    period = pcm.chunk_period
    intervals = _intervals(ordered)
    lateness = _lateness(ordered, pcm)
    audio_sec = sum(pcm.frames_in(chunk.payload_bytes) for chunk in ordered) / pcm.sample_rate
    return {
        "chunks": len(ordered),
        "audio_ms": audio_sec * 1000.0,
        "chunk_period_ms": period * 1000.0,
        "interval_ms": summarise(value * 1000.0 for value in intervals),
        "intervals_over_period": _shares(intervals, period),
        "lateness_ms": summarise(value * 1000.0 for value in lateness),
        "late_by_over_period": _shares(lateness, period),
        "max_lateness_ms": max(lateness) * 1000.0,
    }


def analyse(chunks: Sequence[Chunk], pcm: PcmFormat, *, gap: float) -> dict[str, object]:
    """Every utterance, plus the distributions pooled across them."""
    utterances = [sorted(group, key=lambda item: item.arrived) for group in split_utterances(chunks, gap=gap)]
    intervals = [value for group in utterances for value in _intervals(group)]
    lateness = [value for group in utterances for value in _lateness(group, pcm)]
    return {
        "observed_at": OBSERVED_AT,
        "format": {
            "sample_rate": pcm.sample_rate,
            "channels": pcm.channels,
            "sample_width": pcm.sample_width,
            "chunk_frames": pcm.chunk_frames,
        },
        "utterance_gap_sec": gap,
        "utterances": len(utterances),
        "pooled": {
            "interval_ms": summarise(value * 1000.0 for value in intervals),
            "intervals_over_period": _shares(intervals, pcm.chunk_period),
            "lateness_ms": summarise(value * 1000.0 for value in lateness),
            "late_by_over_period": _shares(lateness, pcm.chunk_period),
        },
        "per_utterance": [analyse_utterance(group, pcm) for group in utterances],
    }


def _shares(values: Sequence[float], period: float) -> dict[str, dict[str, float | int]]:
    """For each threshold, how many values exceed it and what share that is."""
    result: dict[str, dict[str, float | int]] = {}
    for ratio in _THRESHOLDS:
        limit = ratio * period + _THRESHOLD_EPSILON
        over = sum(1 for value in values if value > limit)
        result[f"{ratio:g}x"] = {"count": over, "share": over / len(values) if values else 0.0}
    return result


__all__ = ["OBSERVED_AT", "Chunk", "PcmFormat", "analyse", "analyse_utterance", "split_utterances"]
