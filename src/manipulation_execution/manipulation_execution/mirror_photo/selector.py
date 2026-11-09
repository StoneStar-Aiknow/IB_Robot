"""One MIRRORING round: frames in, candidates kept, one photo out; no ROS.

``PhotoCore`` is everything the photo node decides. The node hands it the
session state, the camera frames (raw JPEG bytes, never decoded here), the
PEAR frames, the hand frames and the external requests, and calls ``tick``.
While MIRRORING it keeps the last ``BUFFER_SEC`` of frames and, for every
trigger past the warm-up, the frame the trigger names (by camera time) as a
candidate, within ``Limits``. When the round ends (the state leaves
MIRRORING, its heartbeat is lost for ``STATE_STALE_SEC``, or the node stops)
``on_round_end`` gets a ``FinishedRound``; ``choose_photo`` turns that into
the one ``Photo`` to send. A round that never got a candidate still sends its
last frame (``quality: fallback``); only a round without a single frame sends
nothing.

``on_provisional`` hears of the round's provisional best (highest trigger
weight, the later on a tie) whenever it changes, so the node can keep a copy
on disk that survives a crash in the middle of a round.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import secrets
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from embodied_common.mirror_photo_contract import TriggerRequest
from embodied_common.mirror_session_contract import STATE_MIRRORING, STATE_STALE_SEC

from .config import PhotoSection
from .scoring import Candidate, Choice, jpeg_size, looks_like_jpeg, select
from .stillness import SpeedMeter, StillTrigger
from .triggers import EVENT, STILL, UNIFORM, GestureTrigger, Trigger, UniformTrigger, event_trigger

SCHEMA = "x2_mirror_photo/1"
# Frames kept while MIRRORING (camera time): a trigger names a frame up to this old.
BUFFER_SEC = 3.0
# A trigger's frame: the nearest one within this; an event's within EVENT_MATCH_SEC, else the newest.
MATCH_SEC = 0.1
EVENT_MATCH_SEC = 0.5
# A trigger up to this far ahead of the newest frame waits until the camera has
# passed its stamp; further ahead (a clock off) only an event is kept, on the newest frame.
PENDING_SEC = 1.0
# PEAR speed and person box for a candidate: the nearest sample within this.
SAMPLE_SEC = 0.25
RATE_WINDOW_SEC = 60.0
REFUSALS = ("warmup", "rate", "nomatch", "late", "badjpeg", "duplicate", "full")


@dataclass(frozen=True)
class FinishedRound:
    round_id: str
    started_at: float  # wall clock
    ended_at: float
    # state | stale | shutdown
    reason: str
    candidates: tuple[Candidate, ...]
    # The round's newest frame (camera time, JPEG); None if no frame came at all.
    last_frame: tuple[float, bytes] | None
    speeds: tuple[tuple[float, float], ...]
    boxes: tuple[tuple[float, tuple[float, ...]], ...]
    refused: dict[str, int]
    frames: int

    def speed_at(self, stamp: float) -> float | None:
        return _nearest(self.speeds, stamp)

    def bbox_at(self, stamp: float) -> tuple[float, ...] | None:
        return _nearest(self.boxes, stamp)


def _nearest(samples: Sequence[tuple[float, object]], stamp: float):
    index = bisect.bisect_left(samples, stamp, key=lambda sample: sample[0])
    best = None
    for near in (index - 1, index):
        if 0 <= near < len(samples):
            gap = abs(samples[near][0] - stamp)
            if gap <= SAMPLE_SEC and (best is None or gap < best[0]):
                best = (gap, samples[near][1])
    return None if best is None else best[1]


@dataclass(frozen=True)
class Photo:
    name: str  # the round id: the file name and the receiver's dedupe key
    jpeg: bytes
    meta: dict


class _Round:
    def __init__(self, round_id: str, started_at: float):
        self.round_id = round_id
        self.started_at = started_at
        self.first_stamp: float | None = None
        self.frames: deque[tuple[float, bytes]] = deque()
        self.frame_count = 0
        self.candidates: list[Candidate] = []
        self.pending: list[Trigger] = []
        self.priority_stamps: deque[float] = deque()
        self.speeds: list[tuple[float, float]] = []
        self.boxes: list[tuple[float, tuple[float, ...]]] = []
        self.refused = dict.fromkeys(REFUSALS, 0)
        self.provisional: Candidate | None = None


@dataclass
class _Stats:
    rounds: int = 0
    last_round: dict = field(default_factory=dict)


class PhotoCore:
    def __init__(
        self,
        section: PhotoSection,
        *,
        on_round_end: Callable[[FinishedRound], None],
        on_provisional: Callable[[str, float, Candidate], None] | None = None,
        wall: Callable[[], float] = time.time,
        round_suffix: Callable[[], str] = lambda: secrets.token_hex(2),
    ):
        self.section = section
        self._on_round_end = on_round_end
        self._on_provisional = on_provisional
        self._wall = wall
        self._suffix = round_suffix
        self._uniform = UniformTrigger(section.uniform)
        self._gesture = GestureTrigger(section.gesture)
        self._still = StillTrigger(section.still)
        self._speed = SpeedMeter()
        self._round: _Round | None = None
        self._state = ""
        self._state_at: float | None = None
        self._stats = _Stats()

    # -- session state -------------------------------------------------------------

    def on_state(self, text: str, now: float) -> None:
        """One ``/mirror_session/state`` message, ``now`` its arrival (monotonic s). Not JSON: ignored."""
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return
        state = payload.get("state") if isinstance(payload, dict) else None
        if not isinstance(state, str):
            return
        self._state = state
        self._state_at = now
        if state == STATE_MIRRORING and self._round is None:
            self._start_round()
        elif state != STATE_MIRRORING and self._round is not None:
            self._end_round("state")

    def tick(self, now: float) -> None:
        if self._round is not None and (self._state_at is None or now - self._state_at > STATE_STALE_SEC):
            self._end_round("stale")
            self._state = ""

    def shutdown(self) -> None:
        if self._round is not None:
            self._end_round("shutdown")

    @property
    def mirroring(self) -> bool:
        return self._round is not None

    def _start_round(self) -> None:
        started = self._wall()
        round_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}-{self._suffix()}"
        self._round = _Round(round_id, started)
        for rule in (self._uniform, self._gesture, self._still, self._speed):
            rule.reset()

    def _end_round(self, reason: str) -> None:
        current, self._round = self._round, None
        finished = FinishedRound(
            round_id=current.round_id,
            started_at=current.started_at,
            ended_at=self._wall(),
            reason=reason,
            candidates=tuple(current.candidates),
            last_frame=current.frames[-1] if current.frames else None,
            speeds=tuple(sorted(current.speeds)),
            boxes=tuple(sorted(current.boxes)),
            refused=dict(current.refused),
            frames=current.frame_count,
        )
        self._stats.rounds += 1
        self._stats.last_round = {
            "round_id": finished.round_id,
            "reason": reason,
            "frames": finished.frames,
            "candidates": _source_counts(finished.candidates),
            "refused": {key: value for key, value in finished.refused.items() if value},
        }
        self._on_round_end(finished)

    # -- inputs ------------------------------------------------------------------------

    def on_image(self, stamp: float, jpeg: bytes) -> None:
        current = self._round
        if current is None:
            return
        frames = current.frames
        if frames and stamp <= frames[-1][0]:
            return  # out of order or repeated: the buffer stays sorted
        frames.append((stamp, jpeg))
        current.frame_count += 1
        while frames and stamp - frames[0][0] > BUFFER_SEC:
            frames.popleft()
        if current.first_stamp is None:
            current.first_stamp = stamp
        self._resolve_pending(current, stamp)
        if self._warm(current, stamp) and self.section.uniform.enabled:
            trigger = self._uniform.update(stamp)
            if trigger is not None:
                self._offer(current, trigger)

    def on_pear(self, stamp: float, raw, bbox: Sequence[float] | None) -> None:
        """One PEAR frame; ``bbox`` None for a frame with no actor (``center_fallback``)."""
        current = self._round
        if current is None:
            return
        if bbox is not None:
            current.boxes.append((stamp, tuple(float(value) for value in bbox)))
        try:
            speed = self._speed.update(stamp, raw)
        except ValueError:
            speed = None
        if speed is not None:
            current.speeds.append((stamp, speed))
        if not self.section.still.enabled:
            return
        if not self._warm(current, stamp):
            self._still.reset()
            return
        fired = self._still.update(stamp, speed)
        if fired is not None:
            self._offer(current, Trigger(STILL, STILL, self.section.still.weight, 1.0, fired))

    def on_hand(self, stamp: float, observations: Iterable[tuple[str, float]]) -> None:
        current = self._round
        if current is None or not self.section.gesture.enabled:
            return
        if not self._warm(current, stamp):
            self._gesture.reset()
            return
        for trigger in self._gesture.update(stamp, observations):
            self._offer(current, trigger)

    def on_event(self, request: TriggerRequest) -> None:
        current = self._round
        if current is None or not self.section.event.enabled:
            return
        newest = current.frames[-1][0] if current.frames else None
        if newest is None or not self._warm(current, newest):
            current.refused["warmup"] += 1
            return
        self._offer(
            current, event_trigger(self.section.event, request.source, request.label, request.score, request.stamp)
        )

    def _warm(self, current: _Round, stamp: float) -> bool:
        return current.first_stamp is not None and stamp - current.first_stamp >= self.section.warmup_sec

    # -- candidates ----------------------------------------------------------------

    def _offer(self, current: _Round, trigger: Trigger) -> None:
        if trigger.priority:
            window = current.priority_stamps
            now = current.frames[-1][0] if current.frames else (trigger.stamp or 0.0)
            while window and now - window[0] > RATE_WINDOW_SEC:
                window.popleft()
            if len(window) >= self.section.limits.max_priority_per_minute:
                current.refused["rate"] += 1
                return
            window.append(now)
        frame = self._frame_for(current, trigger)
        if frame is None:
            return
        self._keep(current, Candidate(trigger, frame[0], frame[1]))

    def _frame_for(self, current: _Round, trigger: Trigger) -> tuple[float, bytes] | None:
        frames = current.frames
        if not frames:
            current.refused["nomatch"] += 1
            return None
        if trigger.stamp is None:
            return frames[-1]
        frame = _nearest_frame(frames, trigger.stamp, _tolerance(trigger))
        if frame is not None and frame[0] >= trigger.stamp:
            return frame
        if trigger.stamp > frames[-1][0]:
            if trigger.stamp - frames[-1][0] <= PENDING_SEC:
                # The frame after the stamp may be nearer: decide once the camera has passed it.
                current.pending.append(trigger)
                return None
            if trigger.source == EVENT:
                return frames[-1]
            current.refused["nomatch"] += 1
            return None
        if frame is not None:
            return frame
        if trigger.source == EVENT:
            return frames[-1]
        current.refused["late" if trigger.stamp < frames[0][0] else "nomatch"] += 1
        return None

    def _resolve_pending(self, current: _Round, stamp: float) -> None:
        if not current.pending:
            return
        waiting, current.pending = current.pending, []
        for trigger in waiting:
            if stamp < trigger.stamp:
                current.pending.append(trigger)
                continue
            frame = _nearest_frame(current.frames, trigger.stamp, _tolerance(trigger))
            if frame is None and trigger.source == EVENT:
                frame = current.frames[-1]
            if frame is None:
                current.refused["nomatch"] += 1
            else:
                self._keep(current, Candidate(trigger, frame[0], frame[1]))

    def _keep(self, current: _Round, candidate: Candidate) -> None:
        if not looks_like_jpeg(candidate.jpeg):
            current.refused["badjpeg"] += 1
            return
        candidates = current.candidates
        for index, kept in enumerate(candidates):
            if kept.stamp == candidate.stamp:
                if candidate.trigger.weight > kept.trigger.weight:
                    candidates[index] = candidate
                    self._provisional(current)
                else:
                    current.refused["duplicate"] += 1
                return
        limits = self.section.limits
        uniform = [index for index, kept in enumerate(candidates) if kept.trigger.source == UNIFORM]
        if (
            not candidate.trigger.priority
            and len(uniform) >= limits.max_candidates_per_round - limits.reserve_for_priority
        ):
            # The rhythm rolls: its newest frames are kept.
            del candidates[uniform[0]]
        elif len(candidates) >= limits.max_candidates_per_round:
            if uniform:
                del candidates[uniform[0]]
            else:
                weakest = min(
                    range(len(candidates)),
                    key=lambda index: (candidates[index].trigger.weight, candidates[index].stamp),
                )
                if not candidate.trigger.priority or candidates[weakest].trigger.weight >= candidate.trigger.weight:
                    current.refused["full"] += 1
                    return
                del candidates[weakest]
        candidates.append(candidate)
        self._provisional(current)

    def _provisional(self, current: _Round) -> None:
        best = max(current.candidates, key=lambda kept: (kept.trigger.weight, kept.stamp))
        if best is not current.provisional:
            current.provisional = best
            if self._on_provisional is not None:
                self._on_provisional(current.round_id, current.started_at, best)

    # -- status ----------------------------------------------------------------------

    def status(self) -> dict:
        current = self._round
        status = {
            "mirror_state": self._state,
            "mirroring": current is not None,
            "rounds": self._stats.rounds,
            "last_round": dict(self._stats.last_round),
        }
        if current is not None:
            status["round"] = {
                "round_id": current.round_id,
                "frames": current.frame_count,
                "candidates": _source_counts(current.candidates),
                "pending": len(current.pending),
            }
        return status


def _tolerance(trigger: Trigger) -> float:
    return EVENT_MATCH_SEC if trigger.source == EVENT else MATCH_SEC


def _nearest_frame(frames: Sequence[tuple[float, bytes]], stamp: float, tolerance: float) -> tuple[float, bytes] | None:
    index = bisect.bisect_left(frames, stamp, key=lambda frame: frame[0])
    best = None
    for near in (index - 1, index):
        if 0 <= near < len(frames):
            gap = abs(frames[near][0] - stamp)
            if gap <= tolerance and (best is None or gap < best[0]):
                best = (gap, frames[near])
    return None if best is None else best[1]


def _source_counts(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.trigger.source] = counts.get(candidate.trigger.source, 0) + 1
    return counts


def _iso(seconds: float) -> str:
    local = time.localtime(seconds)
    return time.strftime("%Y-%m-%dT%H:%M:%S", local) + f".{int(seconds % 1 * 1000):03d}" + time.strftime("%z", local)


def photo_meta(
    round_id: str,
    jpeg: bytes,
    *,
    stamp: float | None,
    trigger: Trigger | None,
    quality: str,
    bbox: Sequence[float] | None,
    extra: dict,
) -> dict:
    """The metadata that travels with a photo (and is stored beside it in the outbox)."""
    size = jpeg_size(jpeg)
    meta = {
        "schema": SCHEMA,
        "round_id": round_id,
        "captured_at": _iso(stamp) if stamp else None,
        "stamp": stamp,
        "source": trigger.source if trigger else None,
        "label": trigger.label if trigger else None,
        "origin": trigger.origin if trigger and trigger.origin else None,
        "score": trigger.score if trigger else None,
        "weight": trigger.weight if trigger else None,
        "quality": quality,
        "bbox": [round(float(value), 1) for value in bbox] if bbox is not None else None,
        "width": size[0] if size else None,
        "height": size[1] if size else None,
        "bytes": len(jpeg),
        "sha256": hashlib.sha256(jpeg).hexdigest(),
    }
    meta.update(extra)
    return meta


def _round_info(finished: FinishedRound) -> dict:
    return {
        "round_started_at": _iso(finished.started_at),
        "round_ended_at": _iso(finished.ended_at),
        "end_reason": finished.reason,
        "candidates": len(finished.candidates),
        "candidate_sources": _source_counts(finished.candidates),
    }


def choose_photo(
    finished: FinishedRound, section: PhotoSection, *, clock: Callable[[], float] = time.monotonic
) -> tuple[Photo | None, Choice | None]:
    """The round's one photo, or None when the round saw no frame at all."""
    round_info = _round_info(finished)
    if not finished.candidates:
        if finished.last_frame is None:
            return None, None
        stamp, jpeg = finished.last_frame
        meta = photo_meta(
            finished.round_id,
            jpeg,
            stamp=stamp,
            trigger=None,
            quality="fallback",
            bbox=finished.bbox_at(stamp),
            extra={**round_info, "scoring": "none"},
        )
        return Photo(finished.round_id, jpeg, meta), None
    choice = select(
        finished.candidates,
        speed_at=finished.speed_at,
        bbox_at=finished.bbox_at,
        stillness_weight=section.stillness_weight,
        sharpness_weight=section.sharpness_weight,
        deadline_sec=section.scoring_deadline_sec,
        clock=clock,
    )
    chosen = choice.candidate
    meta = photo_meta(
        finished.round_id,
        chosen.jpeg,
        stamp=chosen.stamp,
        trigger=chosen.trigger,
        quality="normal",
        bbox=finished.bbox_at(chosen.stamp),
        extra={
            **round_info,
            "scoring": choice.scoring,
            "total": round(choice.total, 3),
            "speed": round(choice.speed, 4) if choice.speed is not None else None,
            "undecodable": choice.undecodable,
            "scoring_ms": round(choice.elapsed_sec * 1000.0, 1),
        },
    )
    return Photo(finished.round_id, chosen.jpeg, meta), choice


def emergency_photo(finished: FinishedRound, why: str) -> Photo | None:
    """The round's photo when ``choose_photo`` failed: the heaviest candidate (the later on a tie), else the last frame.

    No decoding and no scoring, so nothing here can fail the same way; None
    only for a round without a single frame.
    """
    if finished.candidates:
        best = max(finished.candidates, key=lambda kept: (kept.trigger.weight, kept.stamp))
        stamp, jpeg, trigger = best.stamp, best.jpeg, best.trigger
    elif finished.last_frame is not None:
        (stamp, jpeg), trigger = finished.last_frame, None
    else:
        return None
    meta = photo_meta(
        finished.round_id,
        jpeg,
        stamp=stamp,
        trigger=trigger,
        quality="fallback",
        bbox=finished.bbox_at(stamp),
        extra={**_round_info(finished), "scoring": "error", "error": why},
    )
    return Photo(finished.round_id, jpeg, meta)


def provisional_photo(
    round_id: str, started_at: float, candidate: Candidate, bbox: Sequence[float] | None = None
) -> Photo:
    """The round's provisional best as a photo, for the crash copy (``quality: recovered`` once read back)."""
    meta = photo_meta(
        round_id,
        candidate.jpeg,
        stamp=candidate.stamp,
        trigger=candidate.trigger,
        quality="provisional",
        bbox=bbox,
        extra={"round_started_at": _iso(started_at), "scoring": "none"},
    )
    return Photo(round_id, candidate.jpeg, meta)
