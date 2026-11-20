"""The matcher's per-template state (``/preset_motions/state``), as the blender sees it.

The matcher publishes one message per template per tick (``schema_version``
1: ``template_name``, ``actor_id``, ``active``, ``input_valid``, ...). Only
the newest message per template is kept, with the time this process received
it. A template is active while its newest message says ``active`` and
``input_valid`` and is at most ``stale_s`` old.

``episode`` counts each start of an activation: inactive (or stale, or
never seen) -> active. The blender enters a template at most once per
episode, so a template that was abandoned is not entered again until the
person lowers their hands and makes the gesture again.

Malformed messages are counted and dropped; nothing here raises.

For the logs, a gap of more than ``HOLE_S`` between two messages of a
template is a hole (the matcher sends every template at least every 50 ms).
The message that ends it tells why, from the matcher's ``stamp`` (its clock
when it published) and ``sequence``:

* ``late``: the stamps are closer than half the gap, so the matcher
  published on time and the message arrived late (transport, a queue, or
  this node's callbacks: the data-plane status shares their group);
* ``lost``: a gesture template's sequence skipped (it counts publishes), so
  messages were published that never arrived;
* ``matcher``: the stamps are as far apart as the gap, so the matcher
  published nothing (stalled);
* ``unknown``: no stamp to tell.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

HOLE_S = 0.25  # five matcher timer periods
# gesture_vote's sequence counts each publish; overhead_heart's is the PEAR frame's
PUBLISH_SEQUENCED = frozenset({"heart", "holy", "xsign"})
HOLE_KINDS = ("late", "lost", "matcher", "unknown")


@dataclass(frozen=True)
class Label:
    template: str
    actor_id: int
    episode: int


class LabelBook:
    def __init__(self, templates: Iterable[str], stale_s: float = 0.5):
        self._templates = tuple(templates)
        self._stale_s = float(stale_s)
        self._lock = threading.Lock()
        # template -> (active, actor_id, received_at)
        self._latest: dict[str, tuple[bool, int | None, float]] = {}
        self._episode = dict.fromkeys(self._templates, 0)
        # template -> (received_at, stamp, sequence, latency): the newest message, for the holes
        self._meta: dict[str, tuple[float, float | None, int | None, float | None]] = {}
        self._holes: dict[str, dict[str, float]] = {}  # template -> count per kind, longest gap
        self._new_holes: deque[tuple[str, float, float | None, int | None, float | None, str]] = deque(maxlen=64)
        self.received = 0
        self.malformed = 0

    def observe(self, data: str, now: float) -> None:
        try:
            message = json.loads(data)
            name = message["template_name"]
            if message.get("schema_version") != 1 or name not in self._episode:
                raise ValueError("schema or template")
            active = message["active"] is True and message["input_valid"] is True
            actor = message.get("actor_id")
            actor = actor if isinstance(actor, int) and not isinstance(actor, bool) and actor >= 0 else None
            stamp = message.get("stamp")
            stamp = float(stamp) if isinstance(stamp, int | float) and not isinstance(stamp, bool) else None
            stamp = stamp if stamp is not None and math.isfinite(stamp) else None
            sequence = message.get("sequence")
            sequence = sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else None
            latency = time.time() - stamp if stamp is not None else None  # the matcher's clock is this machine's
        except Exception:
            with self._lock:
                self.malformed += 1
            return
        with self._lock:
            self.received += 1
            before = self._latest.get(name)
            was_active = before is not None and before[0] and 0 <= now - before[2] <= self._stale_s
            if active and not was_active:
                self._episode[name] += 1
            self._latest[name] = (active, actor, now)
            meta = self._meta.get(name)
            if meta is not None and now - meta[0] > HOLE_S:
                self._hole(name, now - meta[0], meta, stamp, sequence, latency)
            self._meta[name] = (now, stamp, sequence, latency)

    def _hole(self, name, gap, before, stamp, sequence, latency) -> None:
        _, stamp_before, sequence_before, _ = before
        stamp_gap = stamp - stamp_before if None not in (stamp, stamp_before) else None
        skipped = sequence - sequence_before if None not in (sequence, sequence_before) else None
        if stamp_gap is not None and stamp_gap < gap / 2.0:
            kind = "late"
        elif name in PUBLISH_SEQUENCED and skipped is not None and skipped > 1:
            kind = "lost"
        elif stamp_gap is not None:
            kind = "matcher"
        else:
            kind = "unknown"
        counts = self._holes.setdefault(name, dict.fromkeys((*HOLE_KINDS, "longest"), 0.0))
        counts[kind] += 1
        counts["longest"] = max(counts["longest"], gap)
        self._new_holes.append((name, gap, stamp_gap, skipped, latency, kind))

    def current(
        self, template: str, now: float, older: Callable[[float, float], bool] | None = None
    ) -> Label | None:
        """The template's activation, if it is active and fresh now.

        ``older(age, stale_s)`` judges the newest message's age instead of ``age > stale_s``.
        """
        with self._lock:
            latest = self._latest.get(template)
            if latest is None:
                return None
            active, actor, at = latest
            episode = self._episode[template]
        age = now - at
        if not active or actor is None or not math.isfinite(now) or age < 0.0:
            return None
        if older(age, self._stale_s) if older is not None else age > self._stale_s:
            return None
        return Label(template, actor, episode)

    def describe(self, template: str, now: float) -> str:
        """For the logs: the template's newest message as ``active|inactive|stale|none``, its age and episode,
        and its ``sequence`` and how late it arrived after its ``stamp`` when it had them."""
        with self._lock:
            latest = self._latest.get(template)
            episode = self._episode.get(template, 0)
            meta = self._meta.get(template)
        if latest is None:
            return f"none ep={episode}"
        active, actor, at = latest
        age = now - at
        word = "stale" if not 0 <= age <= self._stale_s else "active" if active and actor is not None else "inactive"
        seq = f" seq={meta[2]}" if meta is not None and meta[2] is not None else ""
        lag = f" lag={meta[3]:.3f}" if meta is not None and meta[3] is not None else ""
        return f"{word} age={age:.2f} ep={episode}{seq}{lag}"

    def take_holes(self) -> list[str]:
        """For the logs: the holes that ended since the last call, one text each."""
        with self._lock:
            holes, self._new_holes = list(self._new_holes), deque(maxlen=64)
        return [
            f"{name} gap={gap:.2f} stamp_gap={_text(stamp_gap)} skipped={_text(skipped, 'd')} "
            f"lag={_text(latency, '.3f')} kind={kind}"
            for name, gap, stamp_gap, skipped, latency, kind in holes
        ]

    def holes(self) -> str:
        """For the tally: per template, its holes per kind and the longest gap since the start ("-": none)."""
        with self._lock:
            holes = {name: dict(counts) for name, counts in self._holes.items()}
        return (
            ",".join(
                f"{name}:{sum(int(counts[k]) for k in HOLE_KINDS)}/max={counts['longest']:.2f}"
                f"({' '.join(f'{k}={int(counts[k])}' for k in HOLE_KINDS if counts[k])})"
                for name, counts in sorted(holes.items())
            )
            or "-"
        )

    def snapshot(self, now: float) -> dict:
        with self._lock:
            return {
                "received": self.received,
                "malformed": self.malformed,
                "active": sorted(
                    name for name, (active, _, at) in self._latest.items() if active and 0 <= now - at <= self._stale_s
                ),
            }


def _text(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)
