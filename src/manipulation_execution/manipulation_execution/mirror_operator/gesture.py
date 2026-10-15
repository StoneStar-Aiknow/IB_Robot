"""Cover gestures from the per-frame covered/clear sequence. No ROS.

``update(covered, now)`` once per judged frame returns ``"double_cover"``,
``"long_cover"`` or None. Times are the caller's receive times.

* Armed only after the camera has been clear for ``startup_clear_s`` (at
  startup, and after a frame gap) or ``rearm_clear_s`` (after a gesture), so
  a hand already over the camera at startup fires nothing.
* A gap between frames longer than ``frame_gap_reset_s`` clears everything
  and needs the startup clear again: a stalled stream is not a cover.
* Double cover: a cover of ``cover_min_s``..``cover_max_s``, a gap of
  ``gap_min_s``..``gap_max_s``, another such cover, both within
  ``total_max_s``; it fires when the second cover ends.
* Long cover: fires on the frame the cover reaches ``long_cover_s``, once.
* A cover of any other length clears a pending first cover.
* A gesture within ``debounce_s`` of the last one fired is dropped, and
  still needs the rearm clear.
"""

from __future__ import annotations

from manipulation_execution.mirror_operator.config import OperatorConfig

DOUBLE_COVER = "double_cover"
LONG_COVER = "long_cover"


class GestureRecognizer:
    def __init__(self, config: OperatorConfig):
        self._config = config
        self._last_at: float | None = None
        self._last_fired_at: float | None = None
        self.dropped = 0  # gestures dropped by the debounce
        self._disarm(config.startup_clear_s)

    def _disarm(self, clear_s: float) -> None:
        self._armed = False
        self._clear_needed_s = clear_s
        self._clear_since: float | None = None
        self._cover_since: float | None = None
        self._first: tuple[float, float] | None = None  # (start, end) of a qualifying first cover

    @property
    def armed(self) -> bool:
        return self._armed

    def update(self, covered: bool, now: float) -> str | None:
        config = self._config
        if self._last_at is not None and not 0 <= now - self._last_at <= config.frame_gap_reset_s:
            self._disarm(config.startup_clear_s)
        self._last_at = now
        if not self._armed:
            if covered:
                self._clear_since = None
            elif self._clear_since is None:
                self._clear_since = now
            if self._clear_since is not None and now - self._clear_since >= self._clear_needed_s:
                self._armed = True
            return None
        if covered:
            if self._cover_since is None:
                self._cover_since = now
            if now - self._cover_since >= config.long_cover_s:
                return self._fire(LONG_COVER, now)
            return None
        gesture = None
        if self._cover_since is not None:
            gesture = self._cover_ended(self._cover_since, now)
            if not self._armed:  # fired or dropped: this clear frame starts the rearm clear
                self._clear_since = now
        if self._first is not None and now - self._first[1] > config.double_cover.gap_max_s:
            self._first = None
        return gesture

    def _cover_ended(self, start: float, end: float) -> str | None:
        double = self._config.double_cover
        self._cover_since = None
        if not double.cover_min_s <= end - start <= double.cover_max_s:
            self._first = None
            return None
        first = self._first
        if (
            first is not None
            and double.gap_min_s <= start - first[1] <= double.gap_max_s
            and end - first[0] <= double.total_max_s
        ):
            return self._fire(DOUBLE_COVER, end)
        self._first = (start, end)
        return None

    def _fire(self, gesture: str, now: float) -> str | None:
        self._disarm(self._config.rearm_clear_s)
        if self._last_fired_at is not None and now - self._last_fired_at < self._config.debounce_s:
            self.dropped += 1
            return None
        self._last_fired_at = now
        return gesture
