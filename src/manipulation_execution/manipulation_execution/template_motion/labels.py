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
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass


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

    def current(self, template: str, now: float) -> Label | None:
        """The template's activation, if it is active and fresh now."""
        with self._lock:
            latest = self._latest.get(template)
            if latest is None:
                return None
            active, actor, at = latest
            if not active or actor is None or not 0 <= now - at <= self._stale_s or not math.isfinite(now):
                return None
            return Label(template, actor, self._episode[template])

    def snapshot(self, now: float) -> dict:
        with self._lock:
            return {
                "received": self.received,
                "malformed": self.malformed,
                "active": sorted(
                    name for name, (active, _, at) in self._latest.items() if active and 0 <= now - at <= self._stale_s
                ),
            }
