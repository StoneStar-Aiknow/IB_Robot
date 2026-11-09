"""Keep the owner's words when one speech segment holds two voices.

The platform's VAD cuts speech at pauses, not at speakers: a bystander's remark and
the owner's request said without a pause in between ("……这都不是跟他说的话你还会做
什么动作", 2026-11-09) arrive as one segment with one transcript and one averaged
voiceprint. The average belongs to neither voice, so the segment is either let
through whole, the bystander's words included, or rejected whole, the request
included.

``speaker_id_node`` also publishes a voiceprint per 2 s window (``SpeakerEmbedding``
``window_*``) and the ASR the start time of every token (``SpeechTranscript``
``token_*``). Here the windows are compared with the owner's anchor; when some are
clearly the owner and some clearly not, the owner's stretch of time is found and the
transcript is cut to the tokens that start inside it.

Windows overlap, so a change of speaker is only known to lie between the middles of
the last window of one voice and the first of the other; the cut is put halfway.
Windows between the two thresholds count for neither side, which keeps a quiet or
noisy stretch of the owner's own speech from reading as a second voice.

ROS-free and standard-library only, like the rest of the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: What a segment turned out to be. ``single``: no window is clearly someone else.
#: ``mixed``: the owner and someone else. ``absent``: no window is clearly the owner.
SPLIT_KINDS = ("single", "mixed", "absent")


@dataclass(frozen=True)
class WindowVoiceprints:
    """Per-window voiceprints of one segment (``SpeakerEmbedding.window_*``)."""

    window_seconds: float
    #: Start of each window, in seconds from the segment's first sample.
    starts: tuple[float, ...]
    vectors: tuple[tuple[float, ...], ...]

    @classmethod
    def from_flat(cls, window_seconds: float, starts, flat, dim: int) -> WindowVoiceprints | None:
        """Unpack the message's flattened vectors; None when they do not line up."""
        starts = tuple(float(value) for value in starts)
        flat = [float(value) for value in flat]
        if not starts or dim <= 0 or len(flat) != len(starts) * dim or not window_seconds > 0:
            return None
        vectors = tuple(tuple(flat[index * dim : (index + 1) * dim]) for index in range(len(starts)))
        return cls(float(window_seconds), starts, vectors)


@dataclass(frozen=True)
class TokenAlignment:
    """Where each token of a transcript starts in time and ends in the text (``SpeechTranscript.token_*``)."""

    starts: tuple[float, ...]
    ends: tuple[int, ...]

    @classmethod
    def build(cls, starts, ends, text: str) -> TokenAlignment | None:
        starts = tuple(float(value) for value in starts)
        ends = tuple(int(value) for value in ends)
        if not starts or len(starts) != len(ends) or ends[-1] != len(text):
            return None
        if any(later < earlier for earlier, later in zip(ends, ends[1:], strict=False)):
            return None
        return cls(starts, ends)


@dataclass(frozen=True)
class SpeakerSplit:
    """The verdict on one segment's windows, and for ``mixed`` the owner's part of its transcript."""

    kind: str
    #: Cosine of every window with the owner's anchor, in window order.
    similarities: tuple[float, ...]
    #: ``mixed``: the owner's words ("" when none of them were transcribed).
    text: str = ""
    #: ``mixed``: the words cut away.
    dropped: str = ""
    #: ``mixed``: the owner's stretch, seconds from the segment's first sample (open-ended at
    #: either end when it starts or ends the segment).
    span: tuple[float, float] = (-math.inf, math.inf)
    #: ``mixed``: mean direction of the owner's windows (unit length).
    vector: tuple[float, ...] = ()


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm > 0 else 0.0


def split_by_speaker(
    anchor: tuple[float, ...],
    windows: WindowVoiceprints,
    text: str,
    alignment: TokenAlignment | None,
    *,
    owner_similarity: float,
    other_similarity: float,
    max_voice_similarity: float = 0.5,
    token_delay: float = 0.35,
) -> SpeakerSplit:
    """Tell the owner's windows from another voice's and, for a mixed segment, keep the owner's tokens.

    The anchor only seeds the two voices: a window at or above
    ``owner_similarity`` is surely the owner, one below ``other_similarity``
    surely not. A 2 s window is a noisy sample of a voice and the anchor was
    recorded in another sentence, so against the anchor the two overlap
    (2026-11-09, 510 stitched segments of real speakers: owner windows p5 0.30,
    others p95 0.36). Windows of one segment share the room, the microphone
    and the distance, and against each other they separate cleanly (same voice
    p5 0.48, two voices p99 0.36). So every window is then assigned to the
    nearer of the two voices' in-segment centroids, and the segment counts as
    mixed only when those centroids are distinct voices (cosine below
    ``max_voice_similarity``) and the owner's centroid still matches the
    anchor: one low window of the owner's own speech has its own voice as the
    nearer centroid and splits nothing.

    The owner's stretch is the longest run of owner windows (the later one on
    a tie: a request usually ends a turn). Its edges are where the margin
    between the two centroids crosses zero, interpolated between the middles
    of the windows on either side: a window centred on the change hears both
    voices equally. A streaming ASR emits a token after hearing it, so token
    start times trail the voice by ``token_delay`` seconds (0.35 s measured on
    the zipformer transducer: without it the cut kept the other speaker's last
    syllable in half the segments). Without an alignment a mixed segment keeps
    no text, which the caller treats as "nothing of the owner to act on".
    """
    if not windows.vectors or len(anchor) != len(windows.vectors[0]):
        return SpeakerSplit("single", ())
    sims = tuple(_cosine(anchor, vector) for vector in windows.vectors)
    owner_seeds = [at for at, value in enumerate(sims) if value >= owner_similarity]
    other_seeds = [at for at, value in enumerate(sims) if value < other_similarity]
    if not owner_seeds:
        return SpeakerSplit("absent" if other_seeds else "single", sims)
    if not other_seeds:
        return SpeakerSplit("single", sims)

    owner_centroid = _mean_direction([windows.vectors[at] for at in owner_seeds])
    other_centroid = _mean_direction([windows.vectors[at] for at in other_seeds])
    for _ in range(3):  # two-means from the seeds; converges in one or two rounds
        margins = [_cosine(v, owner_centroid) - _cosine(v, other_centroid) for v in windows.vectors]
        owners = [windows.vectors[at] for at, margin in enumerate(margins) if margin > 0]
        others = [windows.vectors[at] for at, margin in enumerate(margins) if margin <= 0]
        if not owners or not others:
            return SpeakerSplit("single", sims)
        owner_centroid, other_centroid = _mean_direction(owners), _mean_direction(others)
    if _cosine(owner_centroid, other_centroid) >= max_voice_similarity:
        return SpeakerSplit("single", sims)
    if _cosine(anchor, owner_centroid) < owner_similarity:
        # The voice the anchor seeded drifted to someone else's windows.
        return SpeakerSplit("absent", sims)
    margins = [_cosine(v, owner_centroid) - _cosine(v, other_centroid) for v in windows.vectors]

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(margins):
        if margins[index] <= 0:
            index += 1
            continue
        start = index
        while index < len(margins) and margins[index] > 0:
            index += 1
        runs.append((start, index - 1))
    first, last = max(runs, key=lambda run: (run[1] - run[0], run[1]))

    half = windows.window_seconds / 2.0
    middles = [start + half for start in windows.starts]

    def crossing(left: int, right: int) -> float:
        # margins[left] and margins[right] have opposite signs; linear in between.
        a, b = margins[left], margins[right]
        share = a / (a - b) if a != b else 0.5
        return middles[left] + (middles[right] - middles[left]) * share

    # An owner stretch that opens the segment owns every token before its edge, however early the
    # ASR timed it: subtracting the delay must not push the first word out.
    begin = -math.inf if first == 0 else crossing(first - 1, first)
    end = math.inf if last == len(margins) - 1 else crossing(last, last + 1)
    vector = owner_centroid
    if alignment is None:
        return SpeakerSplit("mixed", sims, "", text, (begin, end), vector)

    kept = [at for at, start in enumerate(alignment.starts) if begin <= start - token_delay < end]
    if not kept:
        return SpeakerSplit("mixed", sims, "", text, (begin, end), vector)
    lo = alignment.ends[kept[0] - 1] if kept[0] > 0 else 0
    hi = alignment.ends[kept[-1]]
    owner_text = text[lo:hi].strip()
    dropped = (text[:lo] + " " + text[hi:]).strip()
    return SpeakerSplit("mixed", sims, owner_text, dropped, (begin, end), vector)


def _mean_direction(vectors) -> tuple[float, ...]:
    total = [0.0] * len(vectors[0])
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        for at, value in enumerate(vector):
            total[at] += value / norm
    norm = math.sqrt(sum(value * value for value in total)) or 1.0
    return tuple(value / norm for value in total)
