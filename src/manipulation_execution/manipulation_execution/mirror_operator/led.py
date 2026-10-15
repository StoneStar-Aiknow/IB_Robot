"""When the light strip follows the session. No ROS.

The light shows "on" while there is a session (STANDBY and MIRRORING alike,
so STANDBY <-> MIRRORING never changes it) and "off" in WAIT or with no
session. ``update(on, now)`` returns the colour name to set once the wanted
state has held for ``stable_s`` and differs from the last one set, else None.
The request is only sent, not waited for: its ``status_code`` is recorded
whenever it arrives, named by ``status_note``.
"""

from __future__ import annotations

from manipulation_execution.mirror_operator.config import LedConfig
from manipulation_execution.mirror_operator.vendor import LED_PRIORITY_REJECTED, LED_STATUS_OK

ON = "on"
OFF = "off"


def status_note(code: int | None) -> str:
    if code is None:
        return "no answer"
    if code == LED_STATUS_OK:
        return "ok"
    if code == LED_PRIORITY_REJECTED:
        return "vendor holds the light"
    return "failed"


class LedPolicy:
    def __init__(self, config: LedConfig):
        self._stable_s = config.stable_s
        self.wanted: str | None = None
        self.applied: str | None = None
        self._since = 0.0

    def update(self, on: bool, now: float) -> str | None:
        wanted = ON if on else OFF
        if wanted != self.wanted:
            self.wanted = wanted
            self._since = now
        if wanted == self.applied or now - self._since < self._stable_s:
            return None
        self.applied = wanted
        return wanted
