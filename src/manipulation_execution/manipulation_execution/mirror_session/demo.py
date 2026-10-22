"""Demo start/cancel rules (business hooks ``demo``): just enough for the main line to run.

A stand-in like the mock animation, not the mirror behaviour; a business hook
class replaces it. It answers four things itself and passes everything else to
``inner`` (the mock playback, or the ``SessionHooks`` defaults):

* ``start_criteria``: all of the rules below hold;
* ``cancel_criteria``: one of them does not; ``cancel_reason`` is the first
  that fails, in this order:

  ``no_actor``  no ``ActorState`` received yet, or it reports no actor;
  ``stale``     the ``ActorState`` is older than ``actor_max_age_s`` (the
                stream stopped);
  ``no_pose``   no PEAR frame with ``source == "yolox"`` received within
                ``pear_max_age_s``. ``center_fallback`` frames, which the
                stream publishes when there is nobody, do not count;
  ``too_far``   the box of the newest ``yolox`` frame is lower than
                ``min_box_height_px`` (``bbox`` is ``[x_min, y_min, x_max,
                y_max]`` in source pixels); 0 turns the rule off;

* ``cooldown_over``: ``cooldown_s`` has passed.

The confirm times (``timing`` section) are the core's, not applied here.

The PEAR window only holds the frames since the previous tick, so the newest
``yolox`` frame is remembered here, scanned at the start of both criteria. Its
age is the time this process received it, not the camera time: a stream that
stalls with frames still buffered is noticed late. Enough for a demo.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from manipulation_execution.mirror_session.hooks import (
    HookContext,
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpContext,
    WrapUpStep,
)

DEMO_SESSION_DEFAULTS = {
    "actor_max_age_s": 1.0,
    "pear_max_age_s": 1.0,
    "min_box_height_px": 0.0,
    "cooldown_s": 2.0,
}
POSE_SOURCE = "yolox"


def _settings(config: Mapping[str, Any] | None) -> dict[str, float]:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("the demo session config must be a mapping")
    unknown = sorted(set(config) - set(DEMO_SESSION_DEFAULTS))
    if unknown:
        raise ValueError(f"the demo session config has unknown keys {unknown}; known: {sorted(DEMO_SESSION_DEFAULTS)}")
    settings = {}
    for name, default in DEMO_SESSION_DEFAULTS.items():
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"session.{name} must be a finite number, not negative")
        settings[name] = float(value)
    return settings


def _box_height(frame) -> float:
    try:
        return float(frame.bbox[3]) - float(frame.bbox[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return math.nan  # fails any min_box_height_px check


class DemoHooks(SessionHooks):
    def __init__(
        self,
        config: Mapping[str, Any] | None,
        context: HookContext | None,
        inner: SessionHooks,
        clock: Callable[[], float] = time.monotonic,
    ):
        settings = _settings(config)
        self._actor_max_age_s = settings["actor_max_age_s"]
        self._pear_max_age_s = settings["pear_max_age_s"]
        self._min_box_height_px = settings["min_box_height_px"]
        self._cooldown_s = settings["cooldown_s"]
        self._context = context
        self._inner = inner
        self._clock = clock
        self._pose_at: float | None = None
        self._box_height = math.nan
        self.cancel_reason = ""

    # ------------------------------------------------------------------
    # the demo rules
    # ------------------------------------------------------------------

    def _observe(self, inputs: SessionInputs) -> None:
        for frame in inputs.window.frames:
            if getattr(frame, "source", None) == POSE_SOURCE:
                self._pose_at = self._clock()
                self._box_height = _box_height(frame)

    def _failed_rule(self, inputs: SessionInputs) -> str:
        """The first rule that does not hold, or ``""``."""
        actor = inputs.actor
        if actor is None or not actor.present:
            return "no_actor"
        if actor.age_s > self._actor_max_age_s:
            return "stale"
        if self._pose_at is None or self._clock() - self._pose_at > self._pear_max_age_s:
            return "no_pose"
        if self._min_box_height_px > 0 and not self._box_height >= self._min_box_height_px:
            return "too_far"
        return ""

    def start_criteria(self, inputs: SessionInputs) -> bool:
        self._observe(inputs)
        return self._failed_rule(inputs) == ""

    def cancel_criteria(self, inputs: SessionInputs, round_elapsed_s: float) -> bool:
        self._observe(inputs)
        self.cancel_reason = self._failed_rule(inputs)
        return self.cancel_reason != ""

    def cooldown_over(self, elapsed_s: float) -> bool:
        return elapsed_s >= self._cooldown_s

    # ------------------------------------------------------------------
    # everything else is the inner hooks'
    # ------------------------------------------------------------------

    def begin_round(self) -> None:
        self._inner.begin_round()

    def end_round(self) -> None:
        self._inner.end_round()

    def retarget(self, inputs: SessionInputs) -> UpperBodyCommand | None:
        return self._inner.retarget(inputs)

    def post_process(self, command: UpperBodyCommand) -> UpperBodyCommand | None:
        return self._inner.post_process(command)

    def reset_pose(self, arm_at_takeover: Sequence[float] | None) -> UpperBodyCommand | None:
        return self._inner.reset_pose(arm_at_takeover)

    def wrap_up(self, context: WrapUpContext) -> WrapUpStep:
        return self._inner.wrap_up(context)
