"""Standing-zone actor selection, without ROS: geometry, tracks, state machine.

``ZoneCore`` decides, once per YOLOX refresh, which person standing inside a
floor zone the mirror follows. The ROS side (``actor_zone_selector``) turns each
new ``DetectionArray`` into a call to ``step`` and each stale or missing one into
a call to ``unknown``, and reads the current answer from ``present``, ``bbox``,
``confidence``, ``reason`` and ``actor_id`` for every PEAR frame in between.

Only the feet identify a person: the box bottom ``y2`` and a sticky foot ``x``
that stays put while an arm stretches the box sideways. Every person inside the
image band has a track; boxes go to tracks greedily by foot distance in units of
the track's box height, one box per track. The target's reference position is
updated only on a good refresh (matched, inside the zone, not occluded), so a
stranger matched to the target while the target was missed cannot drag the
reference out of the zone.

A refresh is bad for the target when it was not matched, its feet are outside
the zone, or someone visibly stands in front of it (rule O: their body cores
overlap and the other box bottom is lower). ``release_updates`` bad refreshes in
a row release the target; nobody is picked for ``release_hold_s`` of camera time
after that, so mirror_session sees ``present`` false long enough to return to
STANDBY before the next person. When the target cannot be told apart from
someone who covers it entirely, its track takes that box and the mirror follows
it (zll D2); ``handoff_suspected`` records that this may have happened and
changes nothing.

The answer depends only on the boxes, their stamps, the image size and the two
files, so a run's ``actor_candidates`` reports replay to the same answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from manipulation_execution.actor_zone_config import TuningConfig, ZoneConfig

IDLE = "IDLE"
ACQUIRING = "ACQUIRING"
LOCKED = "LOCKED"
GRACE = "GRACE"
HOLD = "HOLD"

NO_ACTOR = -1
# ``reason`` values; "no_person" is actor_selector.REASON_NO_PERSON.
REASON_NO_PERSON = "no_person"
REASON_ZONE_EMPTY = "zone_empty"
REASON_ACQUIRING = "zone_acquiring"
REASON_LOCKED = "zone_locked"
REASON_GRACE = "zone_grace"
REASON_HOLD = "zone_hold"

PERSON_LABEL = "person"
# handoff_suspected (a record only): the target's box is within this distance,
# in units of the other track's box height, of an unmatched track's position.
HANDOFF_NEAR = 0.5

_ZERO_BOX = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Box:
    """One YOLOX detection, as received."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {"label": self.label, "conf": _finite(self.confidence), "bbox": [_finite(v, 1) for v in self.bbox]}


def boxes_from_report(report: dict) -> list[Box]:
    """The raw boxes recorded in a report, for replay."""

    def number(value) -> float:
        return math.nan if value is None else float(value)

    return [Box(str(b["label"]), number(b["conf"]), tuple(number(v) for v in b["bbox"])) for b in report["raw"]]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _finite(value: float, digits: int = 4) -> float | None:
    # Raw boxes are recorded before filtering; JSON has no NaN.
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


class Homography:
    """Image point -> zone coordinates (u across, v in depth), from the zone's 4 corners.

    Corners map to (0,0) near_left, (1,0) near_right, (1,1) far_right, (0,1)
    far_left. Closed-form square-to-quad (Heckbert) inverted with the adjugate.
    """

    def __init__(self, quad: tuple[tuple[float, float], ...]) -> None:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
        sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
        if abs(sx) < 1e-12 and abs(sy) < 1e-12:
            g = h = 0.0
        else:
            dx1, dy1, dx2, dy2 = x1 - x2, y1 - y2, x3 - x2, y3 - y2
            den = dx1 * dy2 - dy1 * dx2
            g = (sx * dy2 - sy * dx2) / den
            h = (dx1 * sy - dy1 * sx) / den
        a, b, c = x1 - x0 + g * x1, x3 - x0 + h * x3, x0
        d, e, f = y1 - y0 + g * y1, y3 - y0 + h * y3, y0
        # forward: [x, y, w] = M @ [u, v, 1]; inverse by adjugate (scale is irrelevant).
        self._inv = (
            (e - f * h, c * h - b, b * f - c * e),
            (f * g - d, a - c * g, c * d - a * f),
            (d * h - e * g, b * g - a * h, a * e - b * d),
        )
        cx = sum(p[0] for p in quad) / 4.0
        cy = sum(p[1] for p in quad) / 4.0
        self._w_sign = math.copysign(1.0, self._w(cx, cy))

    def _w(self, x: float, y: float) -> float:
        r = self._inv[2]
        return r[0] * x + r[1] * y + r[2]

    def to_zone(self, x: float, y: float) -> tuple[float, float] | None:
        """(u, v), or None for a point beyond the horizon of the zone's plane."""
        w = self._w(x, y)
        if w * self._w_sign <= 1e-12:
            return None
        r0, r1 = self._inv[0], self._inv[1]
        return (r0[0] * x + r0[1] * y + r0[2]) / w, (r1[0] * x + r1[1] * y + r1[2]) / w


class _Geometry:
    """The zone scaled to one frame size."""

    def __init__(self, zone: ZoneConfig, width: int, height: int) -> None:
        zw, zh = zone.image_size
        sx, sy = width / zw, height / zh
        self.size = (width, height)
        self.scaled = (width, height) != (zw, zh)
        self.crop_x = (zone.crop_x[0] * sx, zone.crop_x[1] * sx)
        self.quad = tuple((x * sx, y * sy) for x, y in zone.floor_quad)
        self.homography = Homography(self.quad)

    def in_band(self, bbox: tuple[float, float, float, float]) -> bool:
        return bbox[2] >= self.crop_x[0] and bbox[0] <= self.crop_x[1]

    def uv(self, x: float, y: float) -> tuple[float, float] | None:
        return self.homography.to_zone(x, y)


def in_zone(uv: tuple[float, float] | None) -> bool:
    return uv is not None and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0


def sticky_foot_x(previous_x: float, bbox: tuple[float, float, float, float], half_width: float) -> float:
    """Keep the previous foot x inside [x1 + b, x2 - b] of the new box; the midline if it is too narrow."""
    x1, y1, x2, y2 = bbox
    b = half_width * (y2 - y1)
    if x2 - x1 < 2.0 * b:
        return (x1 + x2) / 2.0
    return min(max(previous_x, x1 + b), x2 - b)


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    foot_x: float
    misses: int = 0

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class _Observation:
    """A track's box in this refresh."""

    track: _Track
    box_index: int
    box: Box
    foot_x: float
    uv: tuple[float, float] | None
    inside: bool
    dist: float | None = None
    dx: float | None = None
    dy: float | None = None
    h_ratio: float | None = None
    occluded_by: int | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.box.bbox

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def _usable(box: Box, threshold: float) -> bool:
    if box.label != PERSON_LABEL or not math.isfinite(box.confidence) or box.confidence < threshold:
        return False
    x1, y1, x2, y2 = box.bbox
    return all(math.isfinite(v) for v in box.bbox) and x2 > x1 and y2 > y1


class ZoneCore:
    """One standing-zone selector. Feed it refreshes in camera-time order."""

    def __init__(self, zone: ZoneConfig, tuning: TuningConfig, person_confidence_threshold: float) -> None:
        self._zone = zone
        self._t = tuning
        self._threshold = float(person_confidence_threshold)
        self._geometry: _Geometry | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track = 1
        self._last_stamp: float | None = None
        self._state = IDLE
        self._target: int | None = None
        self._acquire_track: int | None = None
        self._acquire_count = 0
        self._bad = 0
        self._hold_start: float | None = None
        self._actor_id = NO_ACTOR
        self._locks = 0
        self._persons_count = 0
        self._bbox = _ZERO_BOX
        self._confidence = 0.0
        self._reason = REASON_NO_PERSON
        self._scale_warned = False

    # ----- the current answer, read for every frame -----
    @property
    def state(self) -> str:
        return self._state

    @property
    def present(self) -> bool:
        return self._state in (LOCKED, GRACE)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox if self.present else _ZERO_BOX

    @property
    def confidence(self) -> float:
        return self._confidence if self.present else 0.0

    @property
    def actor_id(self) -> int:
        return self._actor_id if self.present else NO_ACTOR

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def persons_count(self) -> int:
        return self._persons_count

    @property
    def last_stamp(self) -> float | None:
        return self._last_stamp

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    # ----- transitions -----
    def unknown(self, frame_stamp: float) -> list[str]:
        """Detections are missing or stale for a frame at ``frame_stamp``: release, never pick."""
        events = []
        if self._state in (LOCKED, GRACE):
            self._release(frame_stamp)
            events.append("release:unknown")
        elif self._state == ACQUIRING:
            self._state = IDLE
            self._acquire_track = None
            self._acquire_count = 0
            events.append("acquire_reset:unknown")
        self._persons_count = 0
        return events

    def step(self, stamp: float, image_size: tuple[int, int], boxes: list[Box]) -> dict | None:
        """One YOLOX refresh. Returns its report, or None when ``stamp`` was already seen."""
        stamp = float(stamp)
        if self._last_stamp is not None and stamp == self._last_stamp:
            return None
        events: list[str] = []
        warnings: list[str] = []
        if self._last_stamp is not None and stamp < self._last_stamp:
            # Camera restarted: nothing from the old clock carries over.
            self._tracks.clear()
            self._target = None
            self._acquire_track = None
            self._acquire_count = 0
            self._bad = 0
            self._state = HOLD
            self._hold_start = stamp
            self._reason = REASON_HOLD
            events.append("stamp_backwards")
        self._last_stamp = stamp

        geometry = self._geometry_for(image_size, warnings)
        qualified = [(i, b) for i, b in enumerate(boxes) if _usable(b, self._threshold)]
        self._persons_count = len(qualified)
        banded = [(i, b) for i, b in qualified if geometry.in_band(b.bbox)]

        observations, pairs, unmatched = self._associate(banded, geometry)
        self._find_occlusions(observations)
        by_track = {o.track.track_id: o for o in observations}
        target_obs = by_track.get(self._target) if self._target is not None else None
        handoff = False
        if target_obs is not None:
            handoff = handoff_suspected(target_obs, unmatched)

        # Reference positions: everyone's on a match, the target's only on a good refresh.
        for obs in observations:
            if obs.track.track_id == self._target:
                continue
            self._update_track(obs)
        for track in unmatched:
            if track.track_id == self._target:
                track.misses += 1
                continue
            track.misses += 1
            if track.misses > self._t.track_keep_updates:
                del self._tracks[track.track_id]

        self._advance(stamp, observations, target_obs, events)

        return {
            "stamp": stamp,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "zone_sha256": self._zone.sha256,
            "tuning_sha256": self._t.sha256,
            "raw": [b.as_dict() for b in boxes],
            "persons_count": self._persons_count,
            "tracks": [self._obs_report(o, o.track.track_id == self._target and handoff) for o in observations],
            "unmatched_tracks": [
                {"track": t.track_id, "misses": t.misses} for t in unmatched if t.track_id in self._tracks
            ],
            "pairs": pairs,
            "state": self._state,
            "bad": self._bad,
            "target_track": self._target,
            "acquire_track": self._acquire_track,
            "acquire_count": self._acquire_count,
            "actor_id": self.actor_id,
            "handoff_suspected": handoff,
            "decision": {
                "present": self.present,
                "bbox": [_round(v, 1) for v in self.bbox],
                "conf": _round(self.confidence),
                "reason": self._reason,
            },
            "events": events,
            "warnings": warnings,
        }

    # ----- internals -----
    def _geometry_for(self, image_size: tuple[int, int], warnings: list[str]) -> _Geometry:
        size = (int(image_size[0]), int(image_size[1]))
        if self._geometry is None or self._geometry.size != size:
            self._geometry = _Geometry(self._zone, *size)
            if self._geometry.scaled and not self._scale_warned:
                self._scale_warned = True
                warnings.append(
                    f"frame {size[0]}x{size[1]} differs from the zone's image_size "
                    f"{self._zone.image_size[0]}x{self._zone.image_size[1]}: zone scaled"
                )
        return self._geometry

    def _associate(self, banded: list[tuple[int, Box]], geometry: _Geometry):
        t = self._t
        candidates = []
        pairs = []
        for track in self._tracks.values():
            for index, box in banded:
                foot_x = sticky_foot_x(track.foot_x, box.bbox, t.foot_half_width)
                dx = (foot_x - track.foot_x) / track.height
                dy = (box.bbox[3] - track.y2) / track.height
                h_ratio = (box.bbox[3] - box.bbox[1]) / track.height
                dist = math.hypot(dx, dy)
                allowed = (
                    (t.gate_dx is None or abs(dx) <= t.gate_dx)
                    and (t.gate_dy is None or abs(dy) <= t.gate_dy)
                    and (t.gate_h_ratio is None or 1.0 / t.gate_h_ratio <= h_ratio <= t.gate_h_ratio)
                )
                pairs.append([track.track_id, index, _round(dist), _round(dx), _round(dy), _round(h_ratio), allowed])
                if allowed:
                    candidates.append((dist, track.track_id, index, foot_x, dx, dy, h_ratio))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        observations = []
        boxes = dict(banded)
        for dist, track_id, index, foot_x, dx, dy, h_ratio in candidates:
            if track_id in used_tracks or index in used_boxes:
                continue
            used_tracks.add(track_id)
            used_boxes.add(index)
            uv = geometry.uv(foot_x, boxes[index].bbox[3])
            observations.append(
                _Observation(
                    self._tracks[track_id], index, boxes[index], foot_x, uv, in_zone(uv), dist, dx, dy, h_ratio
                )
            )
        unmatched = [track for track in self._tracks.values() if track.track_id not in used_tracks]
        for index, box in banded:
            if index in used_boxes:
                continue
            x1, _, x2, y2 = box.bbox
            track = _Track(self._next_track, box.bbox, box.confidence, (x1 + x2) / 2.0)
            self._next_track += 1
            self._tracks[track.track_id] = track
            uv = geometry.uv(track.foot_x, y2)
            observations.append(_Observation(track, index, box, track.foot_x, uv, in_zone(uv)))
        observations.sort(key=lambda o: o.track.track_id)
        return observations, pairs, unmatched

    def _find_occlusions(self, observations: list[_Observation]) -> None:
        if self._t.occlude_overlap is None:
            return
        for obs in observations:
            b = self._t.foot_half_width * obs.height
            lo, hi = obs.foot_x - b, obs.foot_x + b
            for other in observations:
                if other is obs:
                    continue
                ob = self._t.foot_half_width * other.height
                overlap = min(hi, other.foot_x + ob) - max(lo, other.foot_x - ob)
                vertical = min(obs.bbox[3], other.bbox[3]) - max(obs.bbox[1], other.bbox[1])
                in_front = other.bbox[3] - obs.bbox[3] > self._t.occlude_dy * obs.height
                if vertical > 0.0 and in_front and overlap > self._t.occlude_overlap * (hi - lo):
                    obs.occluded_by = other.track.track_id
                    break

    def _update_track(self, obs: _Observation) -> None:
        track = obs.track
        track.bbox = obs.bbox
        track.confidence = obs.box.confidence
        track.foot_x = obs.foot_x
        track.misses = 0

    def _release(self, stamp: float) -> None:
        self._state = HOLD
        self._target = None
        self._bad = 0
        self._hold_start = stamp
        self._reason = REASON_HOLD

    def _advance(self, stamp: float, observations, target_obs, events: list[str]) -> None:
        t = self._t
        if self._state == HOLD:
            if stamp - self._hold_start >= t.release_hold_s:
                self._state = IDLE
                events.append("hold_end")
            else:
                self._reason = REASON_HOLD
                return

        if self._state in (LOCKED, GRACE):
            good = target_obs is not None and target_obs.inside and target_obs.occluded_by is None
            if good:
                self._update_track(target_obs)
                self._bad = 0
                self._state = LOCKED
                self._bbox = target_obs.bbox
                self._confidence = float(target_obs.box.confidence)
                self._reason = REASON_LOCKED
                return
            self._bad += 1
            why = "missed" if target_obs is None else ("outside" if not target_obs.inside else "occluded")
            if self._bad >= t.release_updates:
                self._release(stamp)
                events.append(f"release:{why}")
                return
            self._state = GRACE
            if target_obs is not None:
                self._bbox = target_obs.bbox
                self._confidence = float(target_obs.box.confidence)
            self._reason = REASON_GRACE
            events.append(f"grace:{why}")
            return

        # IDLE or ACQUIRING: pick the most prominent unoccluded person inside the zone.
        best = None
        for obs in observations:
            if not obs.inside or obs.occluded_by is not None:
                continue
            u, v = obs.uv
            score = t.center_weight * (1.0 - abs(2.0 * u - 1.0)) + t.front_weight * (1.0 - v)
            if best is None or score > best[0]:
                best = (score, obs)
        if best is None:
            self._state = IDLE
            self._acquire_track = None
            self._acquire_count = 0
            self._reason = REASON_ZONE_EMPTY if self._persons_count else REASON_NO_PERSON
            return
        obs = best[1]
        if obs.track.track_id == self._acquire_track:
            self._acquire_count += 1
        else:
            self._acquire_track = obs.track.track_id
            self._acquire_count = 1
        if self._acquire_count >= t.acquire_updates:
            self._target = obs.track.track_id
            self._acquire_track = None
            self._acquire_count = 0
            self._bad = 0
            self._locks += 1
            self._actor_id = self._locks
            self._state = LOCKED
            self._bbox = obs.bbox
            self._confidence = float(obs.box.confidence)
            self._reason = REASON_LOCKED
            events.append(f"lock:{self._target}")
            return
        self._state = ACQUIRING
        self._reason = REASON_ACQUIRING

    @staticmethod
    def _obs_report(obs: _Observation, handoff: bool) -> dict:
        return {
            "track": obs.track.track_id,
            "box": obs.box_index,
            "dist": None if obs.dist is None else _round(obs.dist),
            "dx": None if obs.dx is None else _round(obs.dx),
            "dy": None if obs.dy is None else _round(obs.dy),
            "h_ratio": None if obs.h_ratio is None else _round(obs.h_ratio),
            "foot": [_round(obs.foot_x, 1), _round(obs.bbox[3], 1)],
            "uv": None if obs.uv is None else [_round(obs.uv[0]), _round(obs.uv[1])],
            "in_zone": obs.inside,
            "occluded_by": obs.occluded_by,
            "handoff_suspected": handoff,
        }


def handoff_suspected(target_obs: _Observation, unmatched: list[_Track]) -> bool:
    """Record only: the target's box is also close to someone who went unmatched this refresh."""
    x1, y1, x2, y2 = target_obs.bbox
    for track in unmatched:
        if track is target_obs.track:
            continue
        foot_x = sticky_foot_x(track.foot_x, target_obs.bbox, 0.12)
        dist = math.hypot(foot_x - track.foot_x, y2 - track.y2) / track.height
        if dist <= HANDOFF_NEAR:
            return True
    return False


__all__ = [
    "ACQUIRING",
    "GRACE",
    "HOLD",
    "IDLE",
    "LOCKED",
    "NO_ACTOR",
    "REASON_ACQUIRING",
    "REASON_GRACE",
    "REASON_HOLD",
    "REASON_LOCKED",
    "REASON_NO_PERSON",
    "REASON_ZONE_EMPTY",
    "Box",
    "Homography",
    "ZoneCore",
    "boxes_from_report",
    "handoff_suspected",
    "in_zone",
    "sticky_foot_x",
]
