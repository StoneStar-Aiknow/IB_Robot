"""Head touch held long enough, from the vendor touch events. No ROS.

``update(event_type, now)`` once per touch message returns True once per hold:

* only ``pressed_event_types`` count as pressed; every other event type
  (idle, unknown, the pats) is a release;
* after subscribing, a release must be seen first: the topic is
  transient_local and hands over an old message, which must not start a hold;
* a press held for ``hold_s`` fires once; holding on does not fire again;
* a fire within ``debounce_s`` of the previous one is dropped;
* the sensor publishes continuously, so a gap between messages longer than
  ``hold_s`` is treated like subscribing again (a release must be seen).
"""

from __future__ import annotations

from manipulation_execution.mirror_operator.config import TouchConfig


class TouchDetector:
    def __init__(self, config: TouchConfig):
        self._config = config
        self._released = False
        self._pressed_since: float | None = None
        self._held = False
        self._last_at: float | None = None
        self._last_fired_at: float | None = None
        self.event_type: int | None = None
        self.dropped = 0  # holds dropped by the debounce

    def update(self, event_type: int, now: float) -> bool:
        config = self._config
        if self._last_at is not None and not 0 <= now - self._last_at <= config.hold_s:
            self._released = False
        self._last_at = now
        self.event_type = event_type
        if event_type not in config.pressed_event_types:
            self._released = True
            self._pressed_since = None
            self._held = False
            return False
        if not self._released:
            self._pressed_since = None
            return False
        if self._pressed_since is None:
            self._pressed_since = now
        if self._held or now - self._pressed_since < config.hold_s:
            return False
        self._held = True
        if self._last_fired_at is not None and now - self._last_fired_at < config.debounce_s:
            self.dropped += 1
            return False
        self._last_fired_at = now
        return True
